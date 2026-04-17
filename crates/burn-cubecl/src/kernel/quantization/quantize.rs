use crate::CubeRuntime;
use crate::{ops::empty_qtensor_optimized, tensor::CubeTensor};
use burn_backend::{TensorMetadata, quantization::QuantScheme};

/// Convert the tensor to a lower precision data type based on the quantization scheme and parameters.
///
/// When `tensor_scale` is `Some(ts)`, the kernel uses the two-level NVFP4 math:
/// `q = clamp(w / (block_scale * ts))`. The caller is expected to have
/// *renormalized* `scale` by `ts` host-side (i.e. `scale = true_block_scale / ts`),
/// so the effective divisor equals the true block scale bit-for-bit.
///
/// When `tensor_scale` is `None`, single-level math is used (no-op: `ts = 1.0`).
pub fn quantize<R>(
    tensor: CubeTensor<R>,
    scheme: &QuantScheme,
    scale: CubeTensor<R>,
    tensor_scale: Option<f32>,
) -> CubeTensor<R>
where
    R: CubeRuntime,
{
    let output = empty_qtensor_optimized(tensor.shape(), *scheme, &tensor.device);
    let (out_values, out_params) = output.clone().quantized_handles().unwrap();
    let dtype = tensor.dtype;

    cubek::quantization::quantize::launch_ref(
        &output.client,
        tensor.binding(),
        out_values.binding(),
        scale.binding(),
        out_params.binding(),
        tensor_scale,
        scheme,
        dtype.into(),
    )
    .expect("Kernel to never fail");

    output
}
