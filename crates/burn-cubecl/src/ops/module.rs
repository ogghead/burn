use crate::{
    CubeBackend, CubeRuntime, FloatElement, IntElement,
    element::BoolElement,
    kernel::{self, conv::ConvTranspose2dStrategy},
};
use burn_backend::tensor::{BoolTensor, FloatTensor, IntTensor};
use burn_backend::{
    TensorMetadata,
    ops::{
        AttentionModuleOptions, ConvOptions, ConvTransposeOptions, DeformConv2dBackward,
        DeformConvOptions, InterpolateOptions, MaxPool2dBackward, MaxPool2dWithIndices, ModuleOps,
    },
};

impl<R, F, I, BT> ModuleOps<Self> for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn conv1d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<1>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv_forward::<R, 1>(x, weight, bias, options, Default::default()).unwrap()
    }

    fn conv1d_x_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: ConvOptions<1>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv_data_backward(
            output_grad,
            weight,
            x.shape(),
            options,
            Default::default(),
        )
        .unwrap()
    }

    fn conv1d_weight_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: ConvOptions<1>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv_weight_backward::<R, 1>(
            x,
            output_grad,
            weight.shape(),
            options,
            Default::default(),
        )
        .unwrap()
    }

    fn conv2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<2>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv_forward::<R, 2>(x, weight, bias, options, Default::default()).unwrap()
    }

    fn conv2d_x_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: ConvOptions<2>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv_data_backward(
            output_grad,
            weight,
            x.shape(),
            options,
            Default::default(),
        )
        .unwrap()
    }

    fn conv2d_weight_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: ConvOptions<2>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv_weight_backward::<R, 2>(
            x,
            output_grad,
            weight.shape(),
            options,
            Default::default(),
        )
        .unwrap()
    }

    fn deform_conv2d(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        options: DeformConvOptions<2>,
    ) -> FloatTensor<Self> {
        kernel::conv::deform_conv2d(x, offset, weight, mask, bias, options).unwrap()
    }

    fn deform_conv2d_backward(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        output_grad: FloatTensor<Self>,
        options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<Self> {
        let (x, o, w, m, b) = kernel::conv::deform_conv2d_backward(
            x,
            offset,
            weight,
            mask,
            bias,
            output_grad,
            options,
        )
        .unwrap();
        DeformConv2dBackward::new(x, o, w, m, b)
    }

    fn conv3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<3>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv_forward::<R, 3>(x, weight, bias, options, Default::default()).unwrap()
    }

    fn conv3d_x_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: ConvOptions<3>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv_data_backward(
            output_grad,
            weight,
            x.shape(),
            options,
            Default::default(),
        )
        .unwrap()
    }

    fn conv3d_weight_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: ConvOptions<3>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv_weight_backward::<R, 3>(
            x,
            output_grad,
            weight.shape(),
            options,
            Default::default(),
        )
        .unwrap()
    }

    fn conv_transpose2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv_transpose2d(x, weight, bias, options, ConvTranspose2dStrategy::default())
            .unwrap()
    }

    fn conv_transpose3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv_transpose3d(x, weight, bias, options).expect("Kernel to never fail")
    }

    fn avg_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        kernel::pool::avg_pool2d(
            x,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        )
    }

    fn avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        kernel::pool::avg_pool2d_backward(
            x,
            grad,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        )
    }

    fn max_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        kernel::pool::max_pool2d(x, kernel_size, stride, padding, dilation, ceil_mode)
    }

    fn max_pool2d_with_indices(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> MaxPool2dWithIndices<Self> {
        let (output, indices) = kernel::pool::max_pool2d_with_indices(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            I::dtype(),
        );

        MaxPool2dWithIndices::new(output, indices)
    }

    fn max_pool2d_with_indices_backward(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
        output_grad: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> MaxPool2dBackward<Self> {
        MaxPool2dBackward::new(kernel::pool::max_pool2d_with_indices_backward(
            x,
            output_grad,
            indices,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        ))
    }

    fn adaptive_avg_pool2d(x: FloatTensor<Self>, output_size: [usize; 2]) -> FloatTensor<Self> {
        kernel::pool::adaptive_avg_pool2d(x, output_size)
    }

    fn adaptive_avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        kernel::pool::adaptive_avg_pool2d_backward(x, grad)
    }

    fn interpolate(
        x: FloatTensor<Self>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self> {
        kernel::interpolate::interpolate(x, output_size, options)
    }

    fn interpolate_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self> {
        kernel::interpolate::interpolate_backward(x, grad, output_size, options)
    }

    fn attention(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        mask: Option<BoolTensor<Self>>,
        attn_bias: Option<FloatTensor<Self>>,
        options: AttentionModuleOptions,
    ) -> FloatTensor<Self> {
        // Fall back to naive attention for features the flash kernel doesn't support.
        if attn_bias.is_some() || options.softcap.is_some() || options.scale.is_some() {
            return burn_backend::ops::attention::attention_fallback::<Self>(
                query, key, value, mask, attn_bias, options,
            );
        }

        // LORA_ATTN_FLASHUNIT forces the fixed, non-autotuned, O(seq) FlashUnit
        // forward (bypassing autotune's speed-only kernel selection). Diagnostic for
        // whether an autotuned attention-forward kernel selected at the >1024px
        // sequence bucket is numerically wrong (the >1024 training divergence). The
        // backward is fixed either way, so convergence under FlashUnit isolates the
        // cause to the forward autotune AND exonerates the fixed FA-backward.
        // ATTN_QK_QUANT=fp8 forces the Stage A FP8 (e4m3) quantized-QK^T flash
        // kernel (Sage-v1 style: per-16-row-tile symmetric quant + K mean
        // smoothing, prepass on device per call). Env-forced ONLY — never
        // registered in autotune, so the tuner cache is untouched — and it
        // FAILS LOUD (panic via the expect below) when the device lacks the
        // e4m3 MMA instruction. Inference path; don't set during training.
        //
        // ATTN_SPARGE_TAU=<f32 in (0,1]> forces the Stage B training-free
        // block-sparse kernel (SpargeAttn style: device predictor keeps the
        // top-tau softmax-mass kv blocks per q-stage row; the global loop
        // skips the rest). Composes with ATTN_QK_QUANT=fp8 (sparse on the
        // quant substrate). Host-side fallback to dense when predicted
        // sparsity < ATTN_SPARGE_MIN_SPARSITY (default 0.15); a materialized
        // mask always takes the dense route. Env-forced ONLY — autotune
        // untouched. Inference path; don't set during training.
        let quant = std::env::var("ATTN_QK_QUANT").as_deref() == Ok("fp8");
        let sparge_tau = std::env::var("ATTN_SPARGE_TAU")
            .ok()
            .map(|v| {
                v.parse::<f32>().unwrap_or_else(|_| {
                    panic!("ATTN_SPARGE_TAU={v}: expected a float in (0, 1]")
                })
            })
            .filter(|_| mask.is_none());
        let strategy = if let Some(tau) = sparge_tau {
            let min_sparsity = std::env::var("ATTN_SPARGE_MIN_SPARSITY")
                .ok()
                .map(|v| {
                    v.parse::<f32>().unwrap_or_else(|_| {
                        panic!("ATTN_SPARGE_MIN_SPARSITY={v}: expected a float in [0, 1]")
                    })
                })
                .unwrap_or(0.15);
            kernel::attention::AttentionStrategy::Sparse {
                tau,
                min_sparsity,
                quant,
            }
        } else if quant {
            kernel::attention::AttentionStrategy::QuantFp8
        } else if std::env::var("LORA_ATTN_FLASHUNIT").is_ok() {
            kernel::attention::AttentionStrategy::FlashUnit
        } else {
            Default::default()
        };
        kernel::attention::attention(query, key, value, mask, attn_bias, options, strategy)
            .expect("Kernel to never fail")
    }

    fn attention_backward(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        out: FloatTensor<Self>,
        grad_out: FloatTensor<Self>,
        mask: Option<BoolTensor<Self>>,
        attn_bias: Option<FloatTensor<Self>>,
        options: AttentionModuleOptions,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        // The fused FlashAttention backward kernel supports plain + causal
        // attention (with an arbitrary softmax scale). Defer to the decomposed
        // fallback for an explicit bool mask, additive bias, or softcap.
        //
        // LORA_DECOMPOSED_BWD forces the decomposed fallback (dense-F32 / O(seq)
        // blocked) for the PLAIN case too — a diagnostic to test whether the fused
        // cubek FA-backward is the source of the >1024px training divergence (the
        // fused forward + fallbacks are exonerated; the fixed FA-backward is the
        // remaining suspect). The fallback is the reference decomposition, so if
        // 1280 converges under it, the fused FA-backward is the culprit.
        // `is_causal` also defers: the naive dq kernel has a known f16-causal
        // gap (cubek-bwd dq_f16_causal test), and the training workloads that
        // reach this path (FLUX) are never causal. Revisit when the tiled
        // kernels validate causal.
        if std::env::var("LORA_DECOMPOSED_BWD").is_ok()
            || mask.is_some()
            || attn_bias.is_some()
            || options.softcap.is_some()
            || options.is_causal
        {
            return burn_backend::ops::attention::attention_backward_fallback::<Self>(
                query, key, value, out, grad_out, mask, attn_bias, options,
            );
        }

        kernel::attention::flash_attention_backward(query, key, value, out, grad_out, options)
            .expect("Flash attention backward kernel to never fail")
    }

    fn has_ctc_loss_backward() -> bool {
        true
    }

    fn ctc_loss(
        log_probs: FloatTensor<Self>,
        targets: IntTensor<Self>,
        input_lengths: IntTensor<Self>,
        target_lengths: IntTensor<Self>,
        blank: usize,
    ) -> FloatTensor<Self> {
        kernel::ctc::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank)
    }

    fn ctc_loss_backward(
        log_probs: FloatTensor<Self>,
        targets: IntTensor<Self>,
        input_lengths: IntTensor<Self>,
        target_lengths: IntTensor<Self>,
        grad_loss: FloatTensor<Self>,
        blank: usize,
    ) -> FloatTensor<Self> {
        let (log_alpha_full, log_beta_full, nll) = kernel::ctc::ctc_alpha_beta(
            log_probs.clone(),
            targets.clone(),
            input_lengths.clone(),
            target_lengths,
            blank,
        );
        burn_backend::ops::ctc::ctc_grad_from_alpha_beta_default::<Self>(
            log_probs,
            targets,
            input_lengths,
            grad_loss,
            log_alpha_full,
            log_beta_full,
            nll,
            blank,
        )
    }

    fn rfft(
        signal: FloatTensor<Self>,
        dim: usize,
        n: Option<usize>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        kernel::fft::rfft(signal, dim, n)
    }

    fn irfft(
        spectrum_re: FloatTensor<Self>,
        spectrum_im: FloatTensor<Self>,
        dim: usize,
        n: Option<usize>,
    ) -> FloatTensor<Self> {
        kernel::fft::irfft(spectrum_re, spectrum_im, dim, n)
    }
}
