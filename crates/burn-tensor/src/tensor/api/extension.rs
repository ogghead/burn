use burn_backend::TensorMetadata;
pub use burn_dispatch::DispatchTensor;
use burn_std::DType;

use crate::{
    Tensor,
    kind::Basic,
    ops::{BridgeTensor, TensorKindId},
};

impl<const D: usize, K> Tensor<D, K>
where
    K: Basic,
{
    /// Converts the tensor into its bridge-layer representation.
    ///
    /// This is primarily intended for backend extensions, allowing custom operations
    /// to be inserted between the high-level tensor API and the dispatch layer before
    /// deferring to a concrete backend.
    pub fn into_bridge(self) -> BridgeTensor {
        self.primitive
    }

    /// Reconstructs a tensor from its [`BridgeTensor`] bridge representation.
    ///
    /// This is the inverse of [`Tensor::into_bridge`] and is primarily intended
    /// for backend extensions, to wrap the output of a custom operation back into
    /// the high-level tensor API.
    ///
    /// # Panics
    ///
    /// Panics if the [`BridgeTensor`] variant does not match the tensor kind `K`
    /// (e.g. passing [`BridgeTensor::Int`] when `K` is [`Float`](crate::Float).
    pub fn from_bridge(tensor: BridgeTensor) -> Self {
        let dtype = tensor.dtype();
        match (&tensor, K::id()) {
            (BridgeTensor::Bool(_), TensorKindId::Bool) if dtype.is_bool() => Self::new(tensor),
            (BridgeTensor::Int(_), TensorKindId::Int) if dtype.is_int() || dtype.is_uint() => {
                Self::new(tensor)
            }
            (BridgeTensor::Float(_), TensorKindId::Float) if dtype.is_float() => Self::new(tensor),
            (BridgeTensor::QFloat(_), TensorKindId::Float) if matches!(dtype, DType::QFloat(_)) => {
                Self::new(tensor)
            }
            (_, kind) => panic!("Expected kind {kind:?}, got dtype {dtype:?}"),
        }
    }

    /// Converts from a primitive tensor into a tensor.
    ///
    /// # Panics
    /// Panis if the primitive dtype does not match the tensor kind `K`.
    pub fn from_primitive(tensor: DispatchTensor) -> Self {
        match (tensor.dtype(), K::id()) {
            (DType::QFloat(_), TensorKindId::Float) => Self::new(BridgeTensor::QFloat(tensor)),
            (dtype, TensorKindId::Float) if dtype.is_float() => {
                Self::new(BridgeTensor::Float(tensor))
            }
            (dtype, TensorKindId::Int) if dtype.is_int() || dtype.is_uint() => {
                Self::new(BridgeTensor::Int(tensor))
            }
            (dtype, TensorKindId::Bool) if dtype.is_bool() => Self::new(BridgeTensor::Bool(tensor)),
            (dtype, kind) => panic!("Expected kind {kind:?}, got dtype {dtype:?}"),
        }
    }

    /// Converts the tensor into a primitive tensor.
    pub fn into_primitive(self) -> DispatchTensor {
        self.primitive.into()
    }
}

#[cfg(test)]
mod tests {
    use crate::{Bool, Int};

    use super::*;

    // -- into_bridge / from_bridge roundtrip --

    #[test]
    fn float_tensor_bridge_roundtrip() {
        let tensor = Tensor::<2>::zeros([2, 3], &Default::default());
        let shape = tensor.shape();
        let bridge = tensor.into_bridge();
        assert!(matches!(bridge, BridgeTensor::Float(_)));
        let tensor = Tensor::<2>::from_bridge(bridge);
        assert_eq!(tensor.shape(), shape);
    }

    #[test]
    fn int_tensor_bridge_roundtrip() {
        let tensor = Tensor::<2, Int>::zeros([2, 3], &Default::default());
        let shape = tensor.shape();
        let bridge = tensor.into_bridge();
        assert!(matches!(bridge, BridgeTensor::Int(_)));
        let tensor = Tensor::<2, Int>::from_bridge(bridge);
        assert_eq!(tensor.shape(), shape);
    }

    #[test]
    fn bool_tensor_bridge_roundtrip() {
        let tensor = Tensor::<2, Bool>::empty([2, 3], &Default::default());
        let shape = tensor.shape();
        let bridge = tensor.into_bridge();
        assert!(matches!(bridge, BridgeTensor::Bool(_)));
        let tensor = Tensor::<2, Bool>::from_bridge(bridge);
        assert_eq!(tensor.shape(), shape);
    }

    // -- from_bridge panics on kind mismatch --

    #[test]
    #[should_panic(expected = "Expected kind Float")]
    fn from_bridge_int_as_float_panics() {
        let bridge = Tensor::<2, Int>::zeros([2, 3], &Default::default()).into_bridge();
        let _tensor = Tensor::<2>::from_bridge(bridge);
    }

    #[test]
    #[should_panic(expected = "Expected kind Float")]
    fn from_bridge_bool_as_float_panics() {
        let bridge = Tensor::<2, Bool>::empty([2, 3], &Default::default()).into_bridge();
        let _tensor = Tensor::<2>::from_bridge(bridge);
    }

    #[test]
    #[should_panic(expected = "Expected kind Int")]
    fn from_bridge_float_as_int_panics() {
        let bridge = Tensor::<2>::zeros([2, 3], &Default::default()).into_bridge();
        let _tensor = Tensor::<2, Int>::from_bridge(bridge);
    }

    #[test]
    #[should_panic(expected = "Expected kind Bool")]
    fn from_bridge_int_as_bool_panics() {
        let bridge = Tensor::<2, Int>::zeros([2, 3], &Default::default()).into_bridge();
        let _tensor = Tensor::<2, Bool>::from_bridge(bridge);
    }

    #[test]
    #[should_panic(expected = "Expected kind Float")]
    fn from_bridge_qfloat_variant_with_int_dtype_panics() {
        // Construct a BridgeTensor::QFloat wrapping a non-qfloat dispatch tensor
        // kind tag says Float but dtype says otherwise.
        let inner = Tensor::<2, Int>::zeros([2, 3], &Default::default()).into_primitive();
        let bridge = BridgeTensor::QFloat(inner);
        let _tensor = Tensor::<2>::from_bridge(bridge);
    }

    // -- into_primitive / from_primitive roundtrip --

    #[test]
    fn float_primitive_roundtrip() {
        let tensor = Tensor::<2>::zeros([2, 3], &Default::default());
        let shape = tensor.shape();
        let primitive = tensor.into_primitive();
        let tensor = Tensor::<2>::from_primitive(primitive);
        assert_eq!(tensor.shape(), shape);
    }

    #[test]
    fn int_primitive_roundtrip() {
        let tensor = Tensor::<2, Int>::zeros([2, 3], &Default::default());
        let shape = tensor.shape();
        let primitive = tensor.into_primitive();
        let tensor = Tensor::<2, Int>::from_primitive(primitive);
        assert_eq!(tensor.shape(), shape);
    }

    #[test]
    fn bool_primitive_roundtrip() {
        let tensor = Tensor::<2, Bool>::empty([2, 3], &Default::default());
        let shape = tensor.shape();
        let primitive = tensor.into_primitive();
        let tensor = Tensor::<2, Bool>::from_primitive(primitive);
        assert_eq!(tensor.shape(), shape);
    }

    // -- from_primitive panics on dtype/kind mismatch --

    #[test]
    #[should_panic(expected = "Expected kind Float")]
    fn from_primitive_int_dtype_as_float_panics() {
        let primitive = Tensor::<2, Int>::zeros([2, 3], &Default::default()).into_primitive();
        let _tensor = Tensor::<2>::from_primitive(primitive);
    }

    #[test]
    #[should_panic(expected = "Expected kind Int")]
    fn from_primitive_float_dtype_as_int_panics() {
        let primitive = Tensor::<2>::zeros([2, 3], &Default::default()).into_primitive();
        let _tensor = Tensor::<2, Int>::from_primitive(primitive);
    }

    #[test]
    #[should_panic(expected = "Expected kind Bool")]
    fn from_primitive_float_dtype_as_bool_panics() {
        let primitive = Tensor::<2>::zeros([2, 3], &Default::default()).into_primitive();
        let _tensor = Tensor::<2, Bool>::from_primitive(primitive);
    }
}

#[cfg(all(test, feature = "autodiff"))]
mod checkpoint_tests {
    use crate::{Tensor, activation::relu};
    use alloc::{sync::Arc, vec, vec::Vec};
    use burn_dispatch::{DispatchTensor, checkpoint};

    fn device() -> crate::Device {
        crate::Device::default().autodiff()
    }

    // A 2-layer block expressed in high-level dispatch `Tensor`, closing over an
    // extra "adapter" param: out = relu(x @ w1) @ w2 + adapter.
    // The block receives ALL differentiable leaves (x, w1, w2, adapter) as the
    // rank-erased `DispatchTensor` inputs.
    fn block(inputs: Vec<DispatchTensor>) -> Vec<DispatchTensor> {
        let mut it = inputs.into_iter();
        let x = Tensor::<2>::from_primitive(it.next().unwrap());
        let w1 = Tensor::<2>::from_primitive(it.next().unwrap());
        let w2 = Tensor::<2>::from_primitive(it.next().unwrap());
        let adapter = Tensor::<2>::from_primitive(it.next().unwrap());

        let h = relu(x.matmul(w1));
        let out = h.matmul(w2).add(adapter);
        vec![out.into_primitive()]
    }

    fn leaves() -> (Tensor<2>, Tensor<2>, Tensor<2>, Tensor<2>) {
        let d = device();
        let x = Tensor::<2>::from_data([[0.5, -1.0], [2.0, 0.25]], &d).require_grad();
        let w1 = Tensor::<2>::from_data([[1.0, -2.0], [0.5, 3.0]], &d).require_grad();
        let w2 = Tensor::<2>::from_data([[0.3, 1.2], [-0.7, 0.9]], &d).require_grad();
        let adapter = Tensor::<2>::from_data([[0.1, -0.1], [0.2, 0.05]], &d).require_grad();
        (x, w1, w2, adapter)
    }

    fn assert_close(a: Tensor<2>, b: Tensor<2>) {
        let a = a.into_data().to_vec::<f32>().unwrap();
        let b = b.into_data().to_vec::<f32>().unwrap();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-5, "grad mismatch: {x} vs {y}");
        }
    }

    #[test]
    fn high_level_checkpoint_matches_reference_grads() {
        // Reference: run the block directly (no checkpointing).
        let (x, w1, w2, adapter) = leaves();
        let out = block(vec![
            x.clone().into_primitive(),
            w1.clone().into_primitive(),
            w2.clone().into_primitive(),
            adapter.clone().into_primitive(),
        ]);
        let out = Tensor::<2>::from_primitive(out.into_iter().next().unwrap());
        let grads = out.sum().backward();
        let (rx, rw1, rw2, ra) = (
            x.grad(&grads).unwrap(),
            w1.grad(&grads).unwrap(),
            w2.grad(&grads).unwrap(),
            adapter.grad(&grads).unwrap(),
        );

        // Checkpointed: same block via the bridge.
        let (x, w1, w2, adapter) = leaves();
        let outputs = checkpoint(
            vec![
                x.clone().into_primitive(),
                w1.clone().into_primitive(),
                w2.clone().into_primitive(),
                adapter.clone().into_primitive(),
            ],
            vec![],
            Arc::new(block),
        );
        let out = Tensor::<2>::from_primitive(outputs.into_iter().next().unwrap());
        let grads = out.sum().backward();

        assert_close(x.grad(&grads).unwrap(), rx);
        assert_close(w1.grad(&grads).unwrap(), rw1);
        assert_close(w2.grad(&grads).unwrap(), rw2);
        assert_close(adapter.grad(&grads).unwrap(), ra);
    }

    // Two-layer stack of checkpointed blocks (mimics stacking transformer blocks):
    // grads through both checkpoint boundaries must match the non-checkpointed ref.
    #[test]
    fn high_level_checkpoint_stack_matches_reference() {
        let run_stack = |checkpointed: bool| {
            let (x, w1, w2, adapter) = leaves();
            let inputs0 = vec![
                x.clone().into_primitive(),
                w1.clone().into_primitive(),
                w2.clone().into_primitive(),
                adapter.clone().into_primitive(),
            ];
            let out0 = if checkpointed {
                checkpoint(inputs0, vec![], Arc::new(block))
            } else {
                block(inputs0)
            };
            let h = Tensor::<2>::from_primitive(out0.into_iter().next().unwrap());

            // Second block reuses w1/w2/adapter as the same leaves.
            let inputs1 = vec![
                h.into_primitive(),
                w1.clone().into_primitive(),
                w2.clone().into_primitive(),
                adapter.clone().into_primitive(),
            ];
            let out1 = if checkpointed {
                checkpoint(inputs1, vec![], Arc::new(block))
            } else {
                block(inputs1)
            };
            let out = Tensor::<2>::from_primitive(out1.into_iter().next().unwrap());
            let grads = out.sum().backward();
            (
                x.grad(&grads).unwrap(),
                w1.grad(&grads).unwrap(),
                w2.grad(&grads).unwrap(),
                adapter.grad(&grads).unwrap(),
            )
        };

        let (rx, rw1, rw2, ra) = run_stack(false);
        let (cx, cw1, cw2, ca) = run_stack(true);
        assert_close(cx, rx);
        assert_close(cw1, rw1);
        assert_close(cw2, rw2);
        assert_close(ca, ra);
    }

    // Inference bypass: untracked inputs must skip checkpointing and just run.
    #[test]
    fn high_level_checkpoint_bypasses_when_not_tracked() {
        let d = crate::Device::default(); // NOT autodiff
        let x = Tensor::<2>::from_data([[0.5, -1.0], [2.0, 0.25]], &d);
        let w1 = Tensor::<2>::from_data([[1.0, -2.0], [0.5, 3.0]], &d);
        let w2 = Tensor::<2>::from_data([[0.3, 1.2], [-0.7, 0.9]], &d);
        let adapter = Tensor::<2>::from_data([[0.1, -0.1], [0.2, 0.05]], &d);

        let inputs = vec![
            x.clone().into_primitive(),
            w1.clone().into_primitive(),
            w2.clone().into_primitive(),
            adapter.clone().into_primitive(),
        ];
        let direct = Tensor::<2>::from_primitive(
            block(inputs.clone()).into_iter().next().unwrap(),
        );
        let viacp = Tensor::<2>::from_primitive(
            checkpoint(inputs, vec![], Arc::new(block))
                .into_iter()
                .next()
                .unwrap(),
        );
        assert_close(viacp, direct);
    }

    // REGRESSION (GPU OOM-fix follow-up): a checkpointed block whose closure uses
    // an EXTERNAL constant tensor created on the autodiff device. Before consts
    // were routed through the bridge, the detached inner forward produced
    // PLAIN-backend activations while the captured constant stayed on the OUTER
    // autodiff backend -> `float_mul` backend mismatch panic. Now the constant is
    // passed via the `consts` channel and re-wrapped per pass.
    //
    // Block: out = (x * c) @ w + adapter, where `c` is the captured constant.
    #[test]
    fn high_level_checkpoint_with_captured_const_matches_reference() {
        let d = device();
        // `c` is created on the autodiff device (like flux-klein's modulation
        // tensors, which descend from the autodiff-tracked timestep embedding).
        let c = Tensor::<2>::from_data([[2.0, 0.5], [-1.0, 3.0]], &d);

        // The block splits its flat input vec as [x, w, adapter, c].
        let const_block = move |inputs: Vec<DispatchTensor>| -> Vec<DispatchTensor> {
            let mut it = inputs.into_iter();
            let x = Tensor::<2>::from_primitive(it.next().unwrap());
            let w = Tensor::<2>::from_primitive(it.next().unwrap());
            let adapter = Tensor::<2>::from_primitive(it.next().unwrap());
            let c = Tensor::<2>::from_primitive(it.next().unwrap());
            let out = x.mul(c).matmul(w).add(adapter);
            vec![out.into_primitive()]
        };

        let mk = || {
            let d = device();
            let x = Tensor::<2>::from_data([[0.5, -1.0], [2.0, 0.25]], &d).require_grad();
            let w = Tensor::<2>::from_data([[1.0, -2.0], [0.5, 3.0]], &d).require_grad();
            let adapter = Tensor::<2>::from_data([[0.1, -0.1], [0.2, 0.05]], &d).require_grad();
            (x, w, adapter)
        };

        // Reference: run the block directly (const as a trailing plain input).
        let (x, w, adapter) = mk();
        let out = const_block(vec![
            x.clone().into_primitive(),
            w.clone().into_primitive(),
            adapter.clone().into_primitive(),
            c.clone().into_primitive(),
        ]);
        let out = Tensor::<2>::from_primitive(out.into_iter().next().unwrap());
        let grads = out.sum().backward();
        let (rx, rw, ra) = (
            x.grad(&grads).unwrap(),
            w.grad(&grads).unwrap(),
            adapter.grad(&grads).unwrap(),
        );

        // Checkpointed: `c` routed through the `consts` channel.
        let (x, w, adapter) = mk();
        let outputs = checkpoint(
            vec![
                x.clone().into_primitive(),
                w.clone().into_primitive(),
                adapter.clone().into_primitive(),
            ],
            vec![c.clone().into_primitive()],
            Arc::new(const_block),
        );
        let out = Tensor::<2>::from_primitive(outputs.into_iter().next().unwrap());
        let grads = out.sum().backward();

        assert_close(x.grad(&grads).unwrap(), rx);
        assert_close(w.grad(&grads).unwrap(), rw);
        assert_close(adapter.grad(&grads).unwrap(), ra);
    }

    // Demonstrates the ORIGINAL bug: capturing an autodiff-device constant inside
    // the closure (instead of routing it through `consts`) makes the detached
    // inner forward mix a plain-backend activation with an autodiff-backend
    // constant -> backend-mismatch panic. This is exactly the flux-klein pattern
    // that panicked on GPU; the fix is to route such constants via `consts` (see
    // the test above). Kept as a guard that the mismatch is still detected.
    #[test]
    #[should_panic(expected = "not on the same backend")]
    fn high_level_checkpoint_captured_autodiff_const_panics() {
        let d = device();
        let c = Tensor::<2>::from_data([[2.0, 0.5], [-1.0, 3.0]], &d); // autodiff device

        // BUGGY: `c` captured by the closure, NOT passed through `consts`.
        let buggy_block = move |inputs: Vec<DispatchTensor>| -> Vec<DispatchTensor> {
            let mut it = inputs.into_iter();
            let x = Tensor::<2>::from_primitive(it.next().unwrap());
            let w = Tensor::<2>::from_primitive(it.next().unwrap());
            let adapter = Tensor::<2>::from_primitive(it.next().unwrap());
            // `c` lives on the OUTER autodiff backend; `x` is plain-backend during
            // the detached forward -> mismatch.
            let out = x.mul(c.clone()).matmul(w).add(adapter);
            vec![out.into_primitive()]
        };

        let dd = device();
        let x = Tensor::<2>::from_data([[0.5, -1.0], [2.0, 0.25]], &dd).require_grad();
        let w = Tensor::<2>::from_data([[1.0, -2.0], [0.5, 3.0]], &dd).require_grad();
        let adapter = Tensor::<2>::from_data([[0.1, -0.1], [0.2, 0.05]], &dd).require_grad();

        let _ = checkpoint(
            vec![
                x.into_primitive(),
                w.into_primitive(),
                adapter.into_primitive(),
            ],
            vec![],
            Arc::new(buggy_block),
        );
    }
}
