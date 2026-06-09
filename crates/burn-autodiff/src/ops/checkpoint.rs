//! Block-level gradient (activation) checkpointing.
//!
//! Runs a "block" closure WITHOUT building an autodiff graph during the forward
//! pass (so the block's interior activations are freed immediately, like
//! PyTorch's `torch.utils.checkpoint`), and RECOMPUTES the block WITH gradients
//! during the backward pass, routing gradients to the block's inputs (which
//! include any trainable adapter parameters the block depends on).
//!
//! # The nested-autodiff insight
//!
//! Inside the [`backward`](Backward::backward) of an `Autodiff<B>` op, the
//! tensors we hold are the INNER backend `B` (e.g. `NdArray`) primitives --
//! already unwrapped. Recomputing the block WITH gradients means constructing a
//! SEPARATE, fresh `Autodiff<B>` graph from new `B` leaves, running the block on
//! it, and calling `.backward()` on that local graph. That is still
//! FIRST-order `Autodiff<B>`, never `Autodiff<Autodiff<B>>`, so it does not hit
//! the "Autodiff should not wrap an autodiff tensor" guard.
//!
//! The fresh graph lives on a brand-new set of [`NodeId`]s, so it maps to a
//! different per-graph mutex than the outer backward currently executing; the
//! global graph-locator mutex is only held transiently (never across the inner
//! `.backward()`), so there is no re-entrant deadlock.

use alloc::{boxed::Box, vec::Vec};
use core::marker::PhantomData;

#[cfg(target_has_atomic = "ptr")]
use alloc::sync::Arc;
#[cfg(not(target_has_atomic = "ptr"))]
use portable_atomic_util::Arc;

use burn_backend::{Backend, ops::FloatTensorOps, tensor::FloatTensor};

use crate::{
    Autodiff,
    checkpoint::{base::Checkpointer, builder::CheckpointerBuilder},
    grads::Gradients,
    graph::{ComputingProperty, NodeId, NodeRef, Parent, Requirement, Step},
    tensor::AutodiffTensor,
};

#[cfg(feature = "distributed")]
use burn_backend::distributed::DistributedParams;

/// A differentiable block that can be evaluated against ANY backend.
///
/// The block must be a *pure function of all of its differentiable inputs*: the
/// block's tensor inputs AND every trainable parameter it depends on must be
/// passed through `inputs`, so their gradients can be captured during the
/// recompute. Anything captured by value inside the implementation (constants,
/// frozen weights) is treated as non-differentiable.
///
/// The same implementation is monomorphized twice by [`checkpoint_block`]:
/// - over the inner backend `B` for the no-grad forward pass, and
/// - over `Autodiff<B>` for the grad-tracked recompute in the backward pass.
///
/// Implement this when the block is genuinely backend-generic (its body is
/// written in terms of `B::float_*` ops). For callers whose block is monomorphic
/// in a single backend (e.g. the high-level dispatch `Tensor`), implement
/// [`CheckpointBlockSplit`] directly instead.
pub trait CheckpointBlock: Send + Sync + 'static {
    /// Evaluate the block. `inputs.len()` and the output arity are fixed for a
    /// given block and must match across both monomorphizations.
    fn forward<B: Backend>(&self, inputs: Vec<FloatTensor<B>>) -> Vec<FloatTensor<B>>;
}

/// A differentiable block expressed as two *monomorphic* evaluations: one on the
/// inner backend `B` (no-grad forward) and one on `Autodiff<B>` (grad-tracked
/// recompute).
///
/// This is the lower-level trait actually used by [`checkpoint_block`]. It exists
/// because a backend that erases its concrete type (the dispatch `Tensor`) cannot
/// be wrapped/unwrapped from a single backend-generic method — the inner-backend
/// and autodiff cases need different re-wrapping. Any [`CheckpointBlock`] is
/// automatically a `CheckpointBlockSplit` via a blanket impl, so backend-generic
/// blocks keep working unchanged.
///
/// The same purity contract as [`CheckpointBlock`] applies: every differentiable
/// leaf must be passed through `inputs`.
pub trait CheckpointBlockSplit<B: Backend>: Send + Sync + 'static {
    /// No-grad forward on the inner backend.
    ///
    /// `inputs` are the differentiable leaves; `consts` are non-differentiable
    /// constants the block also touches (e.g. modulation / rope / mask tensors).
    /// Both are already on the inner backend `B` for this pass.
    fn forward_inner(
        &self,
        inputs: Vec<FloatTensor<B>>,
        consts: Vec<FloatTensor<B>>,
    ) -> Vec<FloatTensor<B>>;
    /// Grad-tracked recompute on `Autodiff<B>`.
    ///
    /// `inputs` are grad-tracked leaves; `consts` are wrapped on `Autodiff<B>`
    /// but are **not** grad-tracked (untracked), so they participate in the
    /// recompute as constants without producing gradients.
    fn forward_autodiff(
        &self,
        inputs: Vec<FloatTensor<Autodiff<B>>>,
        consts: Vec<FloatTensor<Autodiff<B>>>,
    ) -> Vec<FloatTensor<Autodiff<B>>>;
}

impl<B: Backend, T: CheckpointBlock> CheckpointBlockSplit<B> for T {
    fn forward_inner(
        &self,
        mut inputs: Vec<FloatTensor<B>>,
        consts: Vec<FloatTensor<B>>,
    ) -> Vec<FloatTensor<B>> {
        // Backend-generic blocks take a single flat input vec; append consts.
        inputs.extend(consts);
        self.forward::<B>(inputs)
    }
    fn forward_autodiff(
        &self,
        mut inputs: Vec<FloatTensor<Autodiff<B>>>,
        consts: Vec<FloatTensor<Autodiff<B>>>,
    ) -> Vec<FloatTensor<Autodiff<B>>> {
        inputs.extend(consts);
        self.forward::<Autodiff<B>>(inputs)
    }
}

/// Run `block` with block-level activation checkpointing.
///
/// During the forward pass the block runs on the detached inner primitives, so
/// no interior autodiff graph (and therefore no interior activations) is
/// retained. A single autodiff node is registered whose parents are all the
/// `inputs`. During the backward pass the block is recomputed on a fresh
/// first-order `Autodiff<B>` graph seeded with the incoming output gradients,
/// and the resulting input gradients are routed to the corresponding parents.
///
/// All differentiable leaves the block depends on (its tensor inputs and any
/// trainable adapter parameters) must be supplied via `inputs`.
///
/// `consts` are non-differentiable tensors the block also touches (constants
/// such as modulation / rope / attention-mask tensors). They are detached inner
/// primitives: re-wrapped onto the correct backend for each pass (so they never
/// cause a backend mismatch with the activations) but never grad-tracked and
/// never receive gradients. Pass an empty vec if the block has no external
/// constants.
///
/// `block` is taken by concrete type (wrapped in an [`Arc`] for cheap cloning
/// into the backward step); it is statically dispatched and monomorphized once
/// for `B` (forward) and once for `Autodiff<B>` (recompute). The block trait is
/// therefore *not* required to be dyn-compatible.
pub fn checkpoint_block<B: Backend, Block: CheckpointBlockSplit<B>>(
    inputs: Vec<AutodiffTensor<B>>,
    consts: Vec<FloatTensor<B>>,
    block: Arc<Block>,
) -> Vec<AutodiffTensor<B>> {
    // 1. Forward: run on the detached inner primitives -> no interior graph.
    let input_primitives: Vec<FloatTensor<B>> =
        inputs.iter().map(|t| t.primitive.clone()).collect();
    let output_primitives = block.forward_inner(input_primitives.clone(), consts.clone());

    let input_nodes: Vec<NodeRef> = inputs.iter().map(|t| t.node.clone()).collect();
    let requirement = Requirement::from_nodes(&input_nodes);

    // Checkpointing recomputes the whole block; treat as compute-bound so the
    // surrounding autodiff machinery never tries to retro-forward it.
    let computing_property = ComputingProperty::ComputeBound;

    if requirement.is_none() {
        // No grad required anywhere: just wrap the outputs as detached leaves of
        // the input nodes (mirrors the untracked path of regular ops).
        return output_primitives
            .into_iter()
            .map(|primitive| {
                AutodiffTensor::from_parents(
                    primitive,
                    &input_nodes,
                    requirement,
                    computing_property.clone(),
                )
            })
            .collect();
    }

    // 2. Build the output autodiff tensors, all sharing the input nodes as
    //    parents. Only the FIRST output owns the backward step (it routes grads
    //    to every input); the remaining outputs are plain children so their
    //    incoming grads are stored under their own node ids and consumed by the
    //    step. See `CheckpointStep::backward`.
    let parents: Vec<Option<NodeRef>> = input_nodes
        .iter()
        .map(|node| node.clone_if_require_grad())
        .collect();
    let parent_ids: Vec<Parent> = parents
        .iter()
        .flatten()
        .map(|node| Parent { id: node.id })
        .collect();

    let outputs: Vec<AutodiffTensor<B>> = output_primitives
        .into_iter()
        .map(|primitive| {
            AutodiffTensor::from_parents(
                primitive,
                &input_nodes,
                requirement,
                computing_property.clone(),
            )
        })
        .collect();

    let output_nodes: Vec<NodeRef> = outputs.iter().map(|t| t.node.clone()).collect();

    // The step is owned by the first output node.
    let primary = outputs
        .first()
        .expect("checkpoint_block requires at least one output")
        .clone();

    let step = CheckpointStep::<B, Block> {
        block,
        input_primitives,
        consts,
        parents,
        parent_ids,
        output_nodes,
        primary_node: primary.node.clone(),
        phantom: PhantomData,
    };

    // Register the step on the primary output. The other outputs need no step
    // (their grads are consumed by the primary step), but they DO need to be
    // valid graph nodes pointing at the same parents, which `from_parents`
    // already arranged.
    let _ = primary.register_step(step, CheckpointerBuilder::default());

    outputs
}

/// Backward step for a checkpointed block. Owned by the block's first output.
struct CheckpointStep<B: Backend, Block: CheckpointBlockSplit<B>> {
    block: Arc<Block>,
    /// Detached inner primitives of every input, saved from the forward pass.
    input_primitives: Vec<FloatTensor<B>>,
    /// Detached inner primitives of every non-differentiable constant the block
    /// touches. Re-wrapped (untracked) on `Autodiff<B>` during recompute.
    consts: Vec<FloatTensor<B>>,
    /// Parent nodes (one per input; `None` if that input does not require grad).
    parents: Vec<Option<NodeRef>>,
    parent_ids: Vec<Parent>,
    /// Node id of every output, in order, so we can collect their incoming grads.
    output_nodes: Vec<NodeRef>,
    primary_node: NodeRef,
    phantom: PhantomData<B>,
}

impl<B: Backend, Block: CheckpointBlockSplit<B>> core::fmt::Debug for CheckpointStep<B, Block> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CheckpointStep")
            .field("num_inputs", &self.input_primitives.len())
            .field("num_outputs", &self.output_nodes.len())
            .finish()
    }
}

impl<B: Backend, Block: CheckpointBlockSplit<B>> Step for CheckpointStep<B, Block> {
    fn step(self: Box<Self>, grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
        let CheckpointStep {
            block,
            input_primitives,
            consts,
            parents,
            output_nodes,
            ..
        } = *self;

        // Collect the incoming output gradients (inner primitives). Any output
        // that received no grad gets a zero grad of the matching shape so the
        // recompute is well defined.
        let output_grads: Vec<FloatTensor<B>> = output_nodes
            .iter()
            .map(|node| collect_output_grad::<B>(grads, node))
            .collect();

        // --- Recompute the block on a FRESH first-order Autodiff<B> graph. ---
        // The leaves are brand new nodes (fresh NodeIds -> a different per-graph
        // mutex than the outer backward currently holds), seeded as
        // require_grad so the inner backward populates their gradients.
        let leaves: Vec<AutodiffTensor<B>> = input_primitives
            .into_iter()
            .map(|primitive| AutodiffTensor::<B>::new(primitive).require_grad())
            .collect();

        // Constants: wrap on `Autodiff<B>` but DO NOT require grad. They are
        // plain leaves (no step), so they participate as constants and receive
        // no gradient.
        let const_leaves: Vec<AutodiffTensor<B>> = consts
            .into_iter()
            .map(|primitive| AutodiffTensor::<B>::new(primitive))
            .collect();

        let recomputed: Vec<AutodiffTensor<B>> =
            block.forward_autodiff(leaves.clone(), const_leaves);

        debug_assert_eq!(
            recomputed.len(),
            output_grads.len(),
            "checkpoint_block: recompute produced a different number of outputs"
        );

        // Seed the inner backward with a scalar surrogate loss
        //   L = sum_i sum( output_i * grad_i )
        // where each `grad_i` is the incoming (constant) gradient of output i.
        // Then dL/dleaf == J^T grad, exactly the vector-Jacobian product we must
        // propagate to the block's inputs.
        //
        // All arithmetic below runs on `Autodiff<B>`: `out` is grad-tracked
        // (descends from the leaves), while `grad_i` is wrapped as a plain
        // (non-require_grad) constant so it contributes no spurious gradient.
        type AD<B> = Autodiff<B>;
        let seed = recomputed
            .into_iter()
            .zip(output_grads)
            .map(|(out, grad)| {
                let grad_const = AutodiffTensor::<B>::new(grad);
                let weighted = <AD<B> as FloatTensorOps<AD<B>>>::float_mul(out, grad_const);
                <AD<B> as FloatTensorOps<AD<B>>>::float_sum(weighted)
            })
            .reduce(|a, b| <AD<B> as FloatTensorOps<AD<B>>>::float_add(a, b));

        let seed = match seed {
            Some(seed) => seed,
            None => return,
        };

        let inner_grads = seed.backward();

        // Route each leaf's grad onto the corresponding outer parent node.
        for (leaf, parent) in leaves.into_iter().zip(parents) {
            if let Some(parent) = parent {
                if let Some(grad) = leaf.grad(&inner_grads) {
                    grads.register::<B>(parent.id, grad);
                }
            }
        }
    }

    fn node(&self) -> NodeId {
        self.primary_node.id
    }

    fn parents(&self) -> &[Parent] {
        &self.parent_ids
    }

    fn depth(&self) -> usize {
        self.primary_node.order
    }

    #[cfg(feature = "distributed")]
    fn distributed_params(&self) -> Option<DistributedParams> {
        self.primary_node.distributed_params.clone()
    }
}

/// Collect the incoming gradient for `node`, or a zero tensor of the matching
/// shape if no gradient was registered (e.g. an output that did not contribute
/// to the loss).
fn collect_output_grad<B: Backend>(grads: &mut Gradients, node: &NodeRef) -> FloatTensor<B> {
    // The output nodes always require grad in backward, so consume removes them.
    grads.consume::<B>(node)
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::ops::{Backward, Ops, OpsKind};
    use alloc::vec;
    use burn_backend::{
        TensorData, backend::BackendTypes, ops::FloatTensorOps, tensor::FloatTensor, try_read_sync,
    };
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;
    type AD = Autodiff<B>;

    fn device() -> <B as BackendTypes>::Device {
        Default::default()
    }

    fn from_vec(data: Vec<f32>, shape: [usize; 2]) -> FloatTensor<B> {
        B::float_from_data(TensorData::new(data, shape), &device())
    }

    fn to_vec(t: FloatTensor<B>) -> Vec<f32> {
        try_read_sync(B::float_into_data(t))
            .expect("sync read")
            .expect("into_data")
            .to_vec::<f32>()
            .expect("to_vec")
    }

    /// SPIKE: prove that, INSIDE the `backward` of a first-order `Autodiff<B>`
    /// op, we can build a FRESH `Autodiff<B>` graph, run a forward, and call
    /// `.backward()` on it without deadlocking or hitting the
    /// "Autodiff should not wrap an autodiff tensor" guard.
    #[test]
    fn spike_nested_fresh_autodiff_in_backward() {
        #[derive(Debug)]
        struct OuterOp;

        impl Backward<B, 1> for OuterOp {
            // Save the input primitive so we can rebuild a fresh graph from it.
            type State = FloatTensor<B>;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let saved = ops.state;

                // ---- Build a FRESH Autodiff<B> graph inside this backward. ----
                let leaf = AutodiffTensor::<B>::new(saved.clone()).require_grad();
                // y = x * x  (on Autodiff<B>)
                let y = <AD as FloatTensorOps<AD>>::float_mul(leaf.clone(), leaf.clone());
                let loss = <AD as FloatTensorOps<AD>>::float_sum(y);
                let inner_grads = loss.backward(); // <-- nested backward
                let dx = leaf.grad(&inner_grads).expect("inner grad");

                // Route the recomputed grad onto the outer parent.
                crate::ops::unary::<B, _>(ops.parents, ops.node, grads, |_g| dx);
            }
        }

        let x = AutodiffTensor::<B>::new(from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2])).require_grad();

        // Forward: identity (state = detached input). This registers OuterOp.
        let out = match OuterOp
            .prepare::<crate::checkpoint::strategy::NoCheckpointing>([x.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(x.primitive.clone(), x.primitive.clone()),
            OpsKind::UnTracked(prep) => prep.finish(x.primitive.clone()),
        };

        let grads = out.backward();
        let dx = x.grad(&grads).expect("outer grad");
        // d/dx sum(x^2) = 2x
        assert_eq!(to_vec(dx), vec![2.0, 4.0, 6.0, 8.0]);
    }

    // A 2-layer block: out = relu(x @ w1) @ w2 + adapter_bias (broadcast add).
    // Inputs (all differentiable leaves) are: [x, w1, w2, adapter].
    #[derive(Debug)]
    struct LinearReluBlock;

    impl CheckpointBlock for LinearReluBlock {
        fn forward<BB: Backend>(&self, inputs: Vec<FloatTensor<BB>>) -> Vec<FloatTensor<BB>> {
            let mut it = inputs.into_iter();
            let x = it.next().unwrap();
            let w1 = it.next().unwrap();
            let w2 = it.next().unwrap();
            let adapter = it.next().unwrap();

            let h = BB::float_matmul(x, w1);
            let h = BB::relu(h);
            let h = BB::float_matmul(h, w2);
            let out = BB::float_add(h, adapter);
            vec![out]
        }
    }

    /// Reference forward+backward WITHOUT checkpointing, on the same block.
    fn reference_grads() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let x = AutodiffTensor::<B>::new(from_vec(vec![0.5, -1.0, 2.0, 0.25], [2, 2])).require_grad();
        let w1 =
            AutodiffTensor::<B>::new(from_vec(vec![1.0, -2.0, 0.5, 3.0], [2, 2])).require_grad();
        let w2 =
            AutodiffTensor::<B>::new(from_vec(vec![0.3, 1.2, -0.7, 0.9], [2, 2])).require_grad();
        let adapter =
            AutodiffTensor::<B>::new(from_vec(vec![0.1, -0.1, 0.2, 0.05], [2, 2])).require_grad();

        let out = LinearReluBlock
            .forward::<AD>(vec![x.clone(), w1.clone(), w2.clone(), adapter.clone()]);
        let out = out.into_iter().next().unwrap();
        let loss = <AD as FloatTensorOps<AD>>::float_sum(out);
        let grads = loss.backward();

        (
            to_vec(x.grad(&grads).unwrap()),
            to_vec(w1.grad(&grads).unwrap()),
            to_vec(w2.grad(&grads).unwrap()),
            to_vec(adapter.grad(&grads).unwrap()),
        )
    }

    fn assert_close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-5, "grad mismatch: {x} vs {y}");
        }
    }

    #[test]
    fn checkpoint_block_matches_reference_grads() {
        let (ref_x, ref_w1, ref_w2, ref_adapter) = reference_grads();

        // Same leaves, but run the block through checkpoint_block.
        let x = AutodiffTensor::<B>::new(from_vec(vec![0.5, -1.0, 2.0, 0.25], [2, 2])).require_grad();
        let w1 =
            AutodiffTensor::<B>::new(from_vec(vec![1.0, -2.0, 0.5, 3.0], [2, 2])).require_grad();
        let w2 =
            AutodiffTensor::<B>::new(from_vec(vec![0.3, 1.2, -0.7, 0.9], [2, 2])).require_grad();
        let adapter =
            AutodiffTensor::<B>::new(from_vec(vec![0.1, -0.1, 0.2, 0.05], [2, 2])).require_grad();

        let outputs = checkpoint_block::<B, _>(
            vec![x.clone(), w1.clone(), w2.clone(), adapter.clone()],
            vec![],
            Arc::new(LinearReluBlock),
        );
        let out = outputs.into_iter().next().unwrap();
        let loss = <AD as FloatTensorOps<AD>>::float_sum(out);
        let grads = loss.backward();

        assert_close(&to_vec(x.grad(&grads).unwrap()), &ref_x);
        assert_close(&to_vec(w1.grad(&grads).unwrap()), &ref_w1);
        assert_close(&to_vec(w2.grad(&grads).unwrap()), &ref_w2);
        assert_close(&to_vec(adapter.grad(&grads).unwrap()), &ref_adapter);
    }

    // Multi-output block: returns TWO tensors (like the transformer double-block).
    #[derive(Debug)]
    struct TwoOutputBlock;

    impl CheckpointBlock for TwoOutputBlock {
        fn forward<BB: Backend>(&self, inputs: Vec<FloatTensor<BB>>) -> Vec<FloatTensor<BB>> {
            let mut it = inputs.into_iter();
            let a = it.next().unwrap();
            let b = it.next().unwrap();
            // out0 = relu(a @ b); out1 = a + b
            let out0 = BB::relu(BB::float_matmul(a.clone(), b.clone()));
            let out1 = BB::float_add(a, b);
            vec![out0, out1]
        }
    }

    fn reference_two_output() -> (Vec<f32>, Vec<f32>) {
        let a = AutodiffTensor::<B>::new(from_vec(vec![1.0, -2.0, 0.5, 3.0], [2, 2])).require_grad();
        let b = AutodiffTensor::<B>::new(from_vec(vec![0.3, 1.2, -0.7, 0.9], [2, 2])).require_grad();
        let outs = TwoOutputBlock.forward::<AD>(vec![a.clone(), b.clone()]);
        let mut it = outs.into_iter();
        let out0 = it.next().unwrap();
        let out1 = it.next().unwrap();
        // loss uses BOTH outputs with different weights so both grad paths matter.
        let l0 = <AD as FloatTensorOps<AD>>::float_sum(out0);
        let l1 = <AD as FloatTensorOps<AD>>::float_sum(
            <AD as FloatTensorOps<AD>>::float_mul_scalar(out1, 2.0f32.into()),
        );
        let loss = <AD as FloatTensorOps<AD>>::float_add(l0, l1);
        let grads = loss.backward();
        (
            to_vec(a.grad(&grads).unwrap()),
            to_vec(b.grad(&grads).unwrap()),
        )
    }

    #[test]
    fn checkpoint_block_multi_output_matches_reference() {
        let (ref_a, ref_b) = reference_two_output();

        let a = AutodiffTensor::<B>::new(from_vec(vec![1.0, -2.0, 0.5, 3.0], [2, 2])).require_grad();
        let b = AutodiffTensor::<B>::new(from_vec(vec![0.3, 1.2, -0.7, 0.9], [2, 2])).require_grad();

        let outs =
            checkpoint_block::<B, _>(vec![a.clone(), b.clone()], vec![], Arc::new(TwoOutputBlock));
        let mut it = outs.into_iter();
        let out0 = it.next().unwrap();
        let out1 = it.next().unwrap();
        let l0 = <AD as FloatTensorOps<AD>>::float_sum(out0);
        let l1 = <AD as FloatTensorOps<AD>>::float_sum(
            <AD as FloatTensorOps<AD>>::float_mul_scalar(out1, 2.0f32.into()),
        );
        let loss = <AD as FloatTensorOps<AD>>::float_add(l0, l1);
        let grads = loss.backward();

        assert_close(&to_vec(a.grad(&grads).unwrap()), &ref_a);
        assert_close(&to_vec(b.grad(&grads).unwrap()), &ref_b);
    }

    // Block that multiplies its (single) input by a constant tensor and adds an
    // adapter: out = (x * c) @ w + adapter. `c` and `w`/`adapter`... here `c` is
    // the CONSTANT, passed via the `consts` channel; x, w, adapter are inputs.
    #[derive(Debug)]
    struct ConstBlock;

    impl CheckpointBlock for ConstBlock {
        fn forward<BB: Backend>(&self, inputs: Vec<FloatTensor<BB>>) -> Vec<FloatTensor<BB>> {
            // inputs = [x, w, adapter, <consts: c>]
            let mut it = inputs.into_iter();
            let x = it.next().unwrap();
            let w = it.next().unwrap();
            let adapter = it.next().unwrap();
            let c = it.next().unwrap();
            let out = BB::float_add(BB::float_matmul(BB::float_mul(x, c), w), adapter);
            vec![out]
        }
    }

    #[test]
    fn checkpoint_block_with_const_matches_reference_and_const_has_no_grad() {
        let x_data = vec![0.5, -1.0, 2.0, 0.25];
        let w_data = vec![1.0, -2.0, 0.5, 3.0];
        let a_data = vec![0.1, -0.1, 0.2, 0.05];
        let c_data = vec![2.0, 0.5, -1.0, 3.0];

        // Reference: const folded in as a plain (untracked) input.
        let x = AutodiffTensor::<B>::new(from_vec(x_data.clone(), [2, 2])).require_grad();
        let w = AutodiffTensor::<B>::new(from_vec(w_data.clone(), [2, 2])).require_grad();
        let adapter = AutodiffTensor::<B>::new(from_vec(a_data.clone(), [2, 2])).require_grad();
        let c = AutodiffTensor::<B>::new(from_vec(c_data.clone(), [2, 2])); // untracked
        let out = ConstBlock.forward::<AD>(vec![x.clone(), w.clone(), adapter.clone(), c.clone()]);
        let loss = <AD as FloatTensorOps<AD>>::float_sum(out.into_iter().next().unwrap());
        let grads = loss.backward();
        let (rx, rw, ra) = (
            to_vec(x.grad(&grads).unwrap()),
            to_vec(w.grad(&grads).unwrap()),
            to_vec(adapter.grad(&grads).unwrap()),
        );

        // Checkpointed: const passed through the `consts` channel.
        let x = AutodiffTensor::<B>::new(from_vec(x_data, [2, 2])).require_grad();
        let w = AutodiffTensor::<B>::new(from_vec(w_data, [2, 2])).require_grad();
        let adapter = AutodiffTensor::<B>::new(from_vec(a_data, [2, 2])).require_grad();
        let c_prim = from_vec(c_data, [2, 2]);

        let outputs = checkpoint_block::<B, _>(
            vec![x.clone(), w.clone(), adapter.clone()],
            vec![c_prim],
            Arc::new(ConstBlock),
        );
        let loss = <AD as FloatTensorOps<AD>>::float_sum(outputs.into_iter().next().unwrap());
        let grads = loss.backward();

        assert_close(&to_vec(x.grad(&grads).unwrap()), &rx);
        assert_close(&to_vec(w.grad(&grads).unwrap()), &rw);
        assert_close(&to_vec(adapter.grad(&grads).unwrap()), &ra);
    }
}
