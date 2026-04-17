use crate::tensor::CubeTensor;
use crate::{CubeRuntime, ops::numeric::empty_device_dtype};
use burn_backend::{DType, TensorData, TensorMetadata};

/// Convert the tensor back to a higher precision data type.
///
/// If the input tensor's qparams carry a two-level `tensor_scale` (NVFP4
/// Phase B), it is materialized as a 1-element F32 device tensor and passed
/// to the kernel as a post-multiply scalar so the dequant math becomes
/// `w ≈ q * block_scale * tensor_scale`. When absent, single-level math
/// is used (pure block-scale multiply).
pub fn dequantize<R>(tensor: CubeTensor<R>, dtype: DType) -> CubeTensor<R>
where
    R: CubeRuntime,
{
    let scheme = match tensor.dtype {
        DType::QFloat(scheme) => scheme,
        _ => return tensor,
    };

    let output = empty_device_dtype(
        tensor.client.clone(),
        tensor.device.clone(),
        tensor.shape(),
        dtype,
    );
    // Materialise the two-level `tensor_scale` as a 1-element F32 device
    // tensor. Allocated on the same device as `tensor` so the kernel sees
    // a normal binding; it adds one tiny buffer per two-level dequant call.
    let tensor_scale_host = tensor
        .qparams
        .as_ref()
        .and_then(|q| q.tensor_scale);
    let tensor_scale_dev = tensor_scale_host.map(|ts| {
        let bytes = ts.to_ne_bytes().to_vec();
        let data = TensorData::from_bytes_vec(bytes, vec![1usize], DType::F32);
        crate::ops::from_data::<R>(data, &tensor.device)
    });

    let (values, params) = tensor.quantized_handles().unwrap();

    cubek::quantization::dequantize::launch_ref(
        &output.client,
        values.binding(),
        output.clone().binding(),
        params.binding(),
        tensor_scale_dev.map(|t| t.binding()),
        &scheme,
        dtype.into(),
    )
    .expect("Kernel to never fail");

    output
}
