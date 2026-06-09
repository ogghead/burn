//! High-level block-level activation checkpointing for the dispatch backend.
//!
//! This is the bridge that lets a block written against the high-level dispatch
//! [`DispatchTensor`] (i.e. `burn::tensor::Tensor`) be wrapped in the
//! burn-autodiff [`checkpoint_block`](burn_autodiff::ops::checkpoint::checkpoint_block)
//! op. During the forward pass the block runs without retaining an autodiff
//! graph (its activations free immediately); during the backward pass it is
//! recomputed on a fresh first-order autodiff graph and gradients are routed to
//! every input (block inputs AND any trainable adapter params passed as inputs).
//!
//! # Inference is untouched
//!
//! If the inputs are not autodiff-tracked (generation / non-training), the block
//! is simply run directly — no checkpointing, no extra allocation, identical to
//! calling the block inline.

use alloc::{boxed::Box, vec::Vec};

#[cfg(target_has_atomic = "ptr")]
use alloc::sync::Arc;
#[cfg(not(target_has_atomic = "ptr"))]
use portable_atomic_util::Arc;

use crate::backends::*;
use crate::{BackendTensor, DispatchTensor, DispatchTensorKind};

use burn_autodiff::AutodiffTensor;
use burn_autodiff::ops::checkpoint::{CheckpointBlockSplit, checkpoint_block};
use burn_backend::tensor::FloatTensor;

/// A block expressed over the high-level dispatch tensor, suitable for
/// [`checkpoint`].
///
/// The closure must be a *pure function of all of its differentiable inputs*:
/// every tensor input AND every trainable parameter the block touches must be
/// passed through the `inputs` vec, so their gradients are captured during the
/// recompute. Frozen/`no_grad` weights captured by the closure are treated as
/// constants.
pub type CheckpointFn =
    Arc<dyn Fn(Vec<DispatchTensor>) -> Vec<DispatchTensor> + Send + Sync + 'static>;

/// Run `block` with block-level activation checkpointing.
///
/// - If `inputs` are autodiff-tracked, the block runs without retaining interior
///   activations during the forward and is recomputed during the backward,
///   producing gradients for every tracked input.
/// - If `inputs` are NOT autodiff-tracked (inference), the block is executed
///   directly with no overhead.
///
/// `inputs` carries the differentiable leaves (block activations + trainable
/// adapter params). `consts` carries non-differentiable tensors the block also
/// touches (e.g. modulation / rope / mask tensors). The block closure receives a
/// single flat vec `[inputs..., consts...]`, re-wrapped onto the correct backend
/// for whichever pass (detached inner forward or grad-tracked recompute) is
/// running — so a constant produced on the outer autodiff device never causes a
/// backend mismatch with the activations. Constants never receive gradients.
///
/// The outputs are the block's outputs in order.
pub fn checkpoint(
    inputs: Vec<DispatchTensor>,
    consts: Vec<DispatchTensor>,
    block: CheckpointFn,
) -> Vec<DispatchTensor> {
    let is_autodiff = inputs
        .first()
        .map(|t| matches!(t.kind, DispatchTensorKind::Autodiff(_)))
        .unwrap_or(false);

    if !is_autodiff {
        // Inference / untracked: just run the block on [inputs..., consts...].
        let mut all = inputs;
        all.extend(consts);
        return block(all);
    }

    let checkpointing = inputs.first().and_then(|t| t.checkpointing);

    // Descend to the concrete backend behind the autodiff wrapper and run the
    // burn-autodiff checkpoint op there.
    dispatch_checkpoint(inputs, consts, block, checkpointing)
}

/// Adapter implementing the burn-autodiff split block trait by re-wrapping the
/// inner-backend / autodiff primitives back into the high-level dispatch tensor
/// so the high-level `block` closure can run during both the forward and the
/// (autodiff) recompute. The closure always receives `[inputs..., consts...]`.
struct DispatchBlock {
    block: CheckpointFn,
    checkpointing: Option<crate::CheckpointingStrategy>,
}

macro_rules! impl_dispatch_block {
    ($Backend:ident) => {
        impl CheckpointBlockSplit<$Backend<f32>> for DispatchBlock {
            fn forward_inner(
                &self,
                inputs: Vec<FloatTensor<$Backend<f32>>>,
                consts: Vec<FloatTensor<$Backend<f32>>>,
            ) -> Vec<FloatTensor<$Backend<f32>>> {
                // Wrap inner-backend primitives (inputs AND consts) as plain
                // (non-autodiff) float dispatch tensors, run the block, unwrap.
                let dispatch_inputs = inputs
                    .into_iter()
                    .chain(consts)
                    .map(|t| DispatchTensor {
                        kind: DispatchTensorKind::$Backend(BackendTensor::Float(t)),
                        checkpointing: None,
                    })
                    .collect();
                let outputs = (self.block)(dispatch_inputs);
                outputs
                    .into_iter()
                    .map(|t| match t.kind {
                        DispatchTensorKind::$Backend(inner) => inner.float(),
                        _ => panic!(
                            "checkpoint block output is on the wrong backend (expected {})",
                            stringify!($Backend)
                        ),
                    })
                    .collect()
            }

            fn forward_autodiff(
                &self,
                inputs: Vec<FloatTensor<Autodiff<$Backend<f32>>>>,
                consts: Vec<FloatTensor<Autodiff<$Backend<f32>>>>,
            ) -> Vec<FloatTensor<Autodiff<$Backend<f32>>>> {
                // Wrap autodiff primitives (inputs AND consts) as autodiff-tagged
                // dispatch tensors so the block builds a grad graph. The consts
                // are untracked `AutodiffTensor`s (no require_grad), so they are
                // constants here but live on the same backend as the activations.
                let dispatch_inputs = inputs
                    .into_iter()
                    .chain(consts)
                    .map(|t: AutodiffTensor<$Backend<f32>>| DispatchTensor {
                        kind: DispatchTensorKind::Autodiff(Box::new(DispatchTensorKind::$Backend(
                            BackendTensor::Autodiff(t),
                        ))),
                        checkpointing: self.checkpointing,
                    })
                    .collect();
                let outputs = (self.block)(dispatch_inputs);
                outputs
                    .into_iter()
                    .map(|t| match t.kind {
                        DispatchTensorKind::Autodiff(inner) => match *inner {
                            DispatchTensorKind::$Backend(inner) => inner.autodiff(),
                            _ => panic!(
                                "checkpoint recompute output is on the wrong backend (expected {})",
                                stringify!($Backend)
                            ),
                        },
                        _ => panic!("checkpoint recompute produced a non-autodiff output"),
                    })
                    .collect()
            }
        }
    };
}

#[cfg(feature = "cpu")]
impl_dispatch_block!(Cpu);
#[cfg(feature = "cuda")]
impl_dispatch_block!(Cuda);
#[cfg(wgpu_metal)]
impl_dispatch_block!(Metal);
#[cfg(feature = "rocm")]
impl_dispatch_block!(Rocm);
#[cfg(wgpu_vulkan)]
impl_dispatch_block!(Vulkan);
#[cfg(wgpu_webgpu)]
impl_dispatch_block!(Wgpu);
#[cfg(feature = "flex")]
impl_dispatch_block!(Flex);
#[cfg(any(feature = "ndarray", default_backend))]
impl_dispatch_block!(NdArray);
#[cfg(feature = "tch")]
impl_dispatch_block!(LibTorch);

/// Unwrap the autodiff-tracked dispatch inputs to the concrete `AutodiffTensor`,
/// detach the constants to inner primitives, call the burn-autodiff checkpoint
/// op, and re-wrap the outputs.
fn dispatch_checkpoint(
    inputs: Vec<DispatchTensor>,
    consts: Vec<DispatchTensor>,
    block: CheckpointFn,
    checkpointing: Option<crate::CheckpointingStrategy>,
) -> Vec<DispatchTensor> {
    // Determine the concrete backend from the first input.
    let first = inputs
        .first()
        .expect("checkpoint requires at least one input");

    macro_rules! run {
        ($Backend:ident) => {{
            let leaves: Vec<AutodiffTensor<$Backend<f32>>> = inputs
                .into_iter()
                .map(|t| match t.kind {
                    DispatchTensorKind::Autodiff(inner) => match *inner {
                        DispatchTensorKind::$Backend(inner) => inner.autodiff(),
                        _ => panic!("checkpoint inputs are on mixed/wrong backends"),
                    },
                    _ => panic!("checkpoint input is not autodiff-tracked"),
                })
                .collect();

            // Constants: detach to the inner backend primitive whether they
            // arrive autodiff-wrapped (produced on the outer autodiff device) or
            // already plain. They carry no grad and no outer-graph node.
            let const_prims: Vec<FloatTensor<$Backend<f32>>> = consts
                .into_iter()
                .map(|t| match t.kind {
                    DispatchTensorKind::Autodiff(inner) => match *inner {
                        DispatchTensorKind::$Backend(inner) => inner.autodiff_inner(),
                        _ => panic!("checkpoint consts are on mixed/wrong backends"),
                    },
                    DispatchTensorKind::$Backend(inner) => inner.float(),
                    _ => panic!("checkpoint const is on the wrong backend"),
                })
                .collect();

            let adapter = Arc::new(DispatchBlock {
                block,
                checkpointing,
            });

            let outputs =
                checkpoint_block::<$Backend<f32>, DispatchBlock>(leaves, const_prims, adapter);

            outputs
                .into_iter()
                .map(|t| DispatchTensor {
                    kind: DispatchTensorKind::Autodiff(Box::new(DispatchTensorKind::$Backend(
                        BackendTensor::Autodiff(t),
                    ))),
                    checkpointing,
                })
                .collect()
        }};
    }

    match &first.kind {
        DispatchTensorKind::Autodiff(inner) => match &**inner {
            #[cfg(feature = "cpu")]
            DispatchTensorKind::Cpu(_) => run!(Cpu),
            #[cfg(feature = "cuda")]
            DispatchTensorKind::Cuda(_) => run!(Cuda),
            #[cfg(wgpu_metal)]
            DispatchTensorKind::Metal(_) => run!(Metal),
            #[cfg(feature = "rocm")]
            DispatchTensorKind::Rocm(_) => run!(Rocm),
            #[cfg(wgpu_vulkan)]
            DispatchTensorKind::Vulkan(_) => run!(Vulkan),
            #[cfg(wgpu_webgpu)]
            DispatchTensorKind::Wgpu(_) => run!(Wgpu),
            #[cfg(feature = "flex")]
            DispatchTensorKind::Flex(_) => run!(Flex),
            #[cfg(any(feature = "ndarray", default_backend))]
            DispatchTensorKind::NdArray(_) => run!(NdArray),
            #[cfg(feature = "tch")]
            DispatchTensorKind::LibTorch(_) => run!(LibTorch),
            DispatchTensorKind::Autodiff(_) => {
                panic!("Autodiff should not wrap an autodiff tensor.")
            }
        },
        _ => panic!("checkpoint requires autodiff-tracked inputs"),
    }
}
