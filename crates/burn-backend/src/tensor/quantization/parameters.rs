use crate::Backend;

pub use burn_std::quantization::{QParamTensor, QParams};

/// The quantization parameters primitive.
///
/// # Remarks
///
/// This is a low-level struct used internally by the library to provide the quantization parameters
/// to the backends. It is not designed for direct usage by users, and not recommended to import
/// or use this struct directly.
pub struct QuantizationParametersPrimitive<B: Backend> {
    /// The scaling factor.
    pub scales: B::FloatTensorPrimitive,
    /// Optional per-tensor scale for two-level quantization (e.g. NVFP4 Phase B).
    ///
    /// Host-side scalar because two-level decomposition yields a single
    /// `f32` per tensor; stored on the quantized tensor's metadata and
    /// registered as a kernel launch scalar at matmul time.
    pub tensor_scale: Option<f32>,
}
