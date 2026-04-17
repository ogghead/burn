use core::f32;
#[allow(unused_imports)]
use num_traits::Float as _;

use burn_std::{Shape, Slice};

use burn_std::FloatDType;

use crate::{
    Backend, TensorMetadata, get_device_settings,
    ops::AttentionModuleOptions,
    tensor::{BoolTensor, FloatTensor},
};

/// Maximum number of elements in the QK attention matrix before we split by heads.
/// cubecl CUDA kernels use 32-bit unsigned indices; exceeding ~2^31 causes index overflow.
const MAX_ATTN_ELEMENTS: u64 = i32::MAX as u64;

/// Any finite floor prevents `-inf - (-inf) = NaN` in the softmax max-shift.
/// Only activates when every position in a row is masked (`-inf`).
const SOFTMAX_MAX_FLOOR: f64 = -1e4;

/// Prevents `0 / 0 = NaN` when all numerators are zero (fully-masked row).
const SOFTMAX_SUM_EPS: f64 = 1e-6;

/// Computes softmax(QKᵗ * scale) · V using separate kernels.
/// Serves as a fallback when FlashAttention is not used.
///
/// When the attention matrix would exceed `MAX_ATTN_ELEMENTS` (i32::MAX), the computation
/// is split across heads to avoid 32-bit index overflow in CUDA kernels.
pub fn attention_fallback<B: Backend>(
    query: FloatTensor<B>,
    key: FloatTensor<B>,
    value: FloatTensor<B>,
    mask: Option<BoolTensor<B>>,
    attn_bias: Option<FloatTensor<B>>,
    options: AttentionModuleOptions,
) -> FloatTensor<B> {
    let shape = query.shape().dims::<4>();
    let [batch_size, num_heads, seq_q, _head_dim] = shape;
    let key_shape = key.shape().dims::<4>();
    let seq_k = key_shape[2];

    // Check if the QK attention matrix would overflow 32-bit indices
    let attn_elements = batch_size as u64 * num_heads as u64 * seq_q as u64 * seq_k as u64;
    if attn_elements > MAX_ATTN_ELEMENTS {
        return attention_fallback_chunked::<B>(
            query, key, value, mask, attn_bias, options,
            batch_size, num_heads, seq_q, seq_k,
        );
    }

    attention_fallback_inner::<B>(query, key, value, mask, attn_bias, options)
}

/// Core attention computation for tensors that fit within 32-bit index limits.
fn attention_fallback_inner<B: Backend>(
    query: FloatTensor<B>,
    key: FloatTensor<B>,
    value: FloatTensor<B>,
    mask: Option<BoolTensor<B>>,
    attn_bias: Option<FloatTensor<B>>,
    options: AttentionModuleOptions,
) -> FloatTensor<B> {
    if let Some(softcap) = options.softcap {
        assert!(softcap > 0.0, "softcap must be positive, got {softcap}");
    }

    // Attention scores: A = QKᵗ * scale
    let query_shape = query.shape().dims::<4>();
    let scale = options
        .scale
        .unwrap_or_else(|| 1.0 / (*query_shape.last().unwrap() as f64).sqrt());
    let transposed_key = B::float_transpose(key);
    let qk = B::float_matmul(query, transposed_key);
    let attention_scores = B::float_mul_scalar(qk, scale.into());

    // Softcap: softcap * tanh(scores / softcap)
    // Applied to raw logits before any -inf masking, so that tanh does not
    // map -inf to a finite value (which would break masking semantics).
    let attention_scores = if let Some(softcap) = options.softcap {
        let scaled = B::float_div_scalar(attention_scores, softcap.into());
        let tanh = B::float_tanh(scaled);
        B::float_mul_scalar(tanh, softcap.into())
    } else {
        attention_scores
    };

    // Bool masking
    let attention_scores = if let Some(mask) = mask {
        B::float_mask_fill(attention_scores, mask, f32::NEG_INFINITY.into())
    } else {
        attention_scores
    };

    // Causal masking: mask positions where col > row (future positions)
    let attention_scores = if options.is_causal {
        let causal_mask = build_causal_mask::<B>(&attention_scores);
        B::float_mask_fill(attention_scores, causal_mask, f32::NEG_INFINITY.into())
    } else {
        attention_scores
    };

    // Additive bias (ALiBi, relative position biases, etc.)
    let attention_scores = if let Some(bias) = attn_bias {
        B::float_add(attention_scores, bias)
    } else {
        attention_scores
    };

    // NaN-safe softmax in f32: S = softmax(A)
    // Upcast to f32 for numerical stability — bf16 softmax loses precision at
    // large sequence lengths (e.g. 7840 tokens), causing washed-out images.
    // When all positions in a row are masked (-inf), naive softmax has two NaN paths:
    //   (1) max is -inf, so the shift -inf - (-inf) = NaN;
    //   (2) after fixing (1), all exp values are 0, so sum is 0 and 0/0 = NaN.
    // Clamping max and sum (see SOFTMAX_MAX_FLOOR / SOFTMAX_SUM_EPS) avoids both, yielding 0 for
    // fully-masked rows (no position contributes to output).
    let orig_dtype: FloatDType = attention_scores.dtype().into();
    let scores_f32 = B::float_cast(attention_scores, FloatDType::F32);
    let max_per_dim = B::float_max_dim(scores_f32.clone(), 3);
    let max_per_dim = B::float_clamp_min(max_per_dim, SOFTMAX_MAX_FLOOR.into());
    let minus_max = B::float_sub(scores_f32, max_per_dim);
    let numerator = B::float_exp(minus_max);
    let sum_exp = B::float_sum_dim(numerator.clone(), 3);
    let sum_exp = B::float_clamp_min(sum_exp, SOFTMAX_SUM_EPS.into());
    let softmax = B::float_div(numerator, sum_exp);
    let softmax = B::float_cast(softmax, orig_dtype);

    // Context: S · V
    B::float_matmul(softmax, value)
}

/// Splits attention across heads to keep each chunk's element count within 32-bit index limits.
/// Each chunk processes a group of heads independently, then results are concatenated.
#[allow(clippy::too_many_arguments)]
fn attention_fallback_chunked<B: Backend>(
    query: FloatTensor<B>,
    key: FloatTensor<B>,
    value: FloatTensor<B>,
    mask: Option<BoolTensor<B>>,
    attn_bias: Option<FloatTensor<B>>,
    options: AttentionModuleOptions,
    batch_size: usize,
    num_heads: usize,
    seq_q: usize,
    seq_k: usize,
) -> FloatTensor<B> {
    // Determine how many heads fit per chunk
    let per_head = batch_size as u64 * seq_q as u64 * seq_k as u64;
    let max_heads_per_chunk = if per_head == 0 {
        num_heads
    } else {
        (MAX_ATTN_ELEMENTS / per_head).max(1) as usize
    };

    let full = Slice::full();
    let mut outputs = Vec::new();

    let mut h = 0;
    while h < num_heads {
        let chunk_end = (h + max_heads_per_chunk).min(num_heads);
        let head_slice: Slice = (h..chunk_end).into();

        let q_chunk = B::float_slice(query.clone(), &[full, head_slice, full, full]);
        let k_chunk = B::float_slice(key.clone(), &[full, head_slice, full, full]);
        let v_chunk = B::float_slice(value.clone(), &[full, head_slice, full, full]);
        let mask_chunk = mask
            .as_ref()
            .map(|m| B::bool_slice(m.clone(), &[full, head_slice, full, full]));
        let bias_chunk = attn_bias
            .as_ref()
            .map(|b| B::float_slice(b.clone(), &[full, head_slice, full, full]));

        let out = attention_fallback_inner::<B>(
            q_chunk, k_chunk, v_chunk, mask_chunk, bias_chunk, options,
        );
        outputs.push(out);

        h = chunk_end;
    }

    B::float_cat(outputs, 1)
}

/// Builds a causal (upper-triangular) bool mask where `true` means "mask this position".
/// Shape: [batch_size, num_heads, seq_q, seq_k], masking positions where col > row.
fn build_causal_mask<B: Backend>(attention_scores: &FloatTensor<B>) -> BoolTensor<B> {
    let device = B::float_device(attention_scores);
    let scores_shape = attention_scores.shape().dims::<4>();
    let [batch_size, num_heads, seq_q, seq_k] = scores_shape;
    let settings = get_device_settings::<B>(&device);

    // row indices [seq_q, 1] and col indices [1, seq_k]
    // Offset col indices so that the causal boundary aligns at the bottom-right corner,
    // which handles cross-attention (seq_k > seq_q) correctly.
    let offset = seq_k as i64 - seq_q as i64;
    let rows = B::int_reshape(
        B::int_arange(0..seq_q as i64, &device, settings.int_dtype),
        Shape::new([seq_q, 1]),
    );
    let cols = B::int_reshape(
        B::int_arange(0..seq_k as i64, &device, settings.int_dtype),
        Shape::new([1, seq_k]),
    );

    // mask where col > row + offset (upper triangle)
    let rows_shifted = B::int_add_scalar(rows, offset.into());
    let mask_2d = B::int_lower(rows_shifted, cols, settings.bool_dtype);

    // Reshape to [1, 1, seq_q, seq_k] then expand to [batch_size, num_heads, seq_q, seq_k]
    let mask_4d = B::bool_reshape(mask_2d, Shape::new([1, 1, seq_q, seq_k]));
    B::bool_expand(mask_4d, Shape::new([batch_size, num_heads, seq_q, seq_k]))
}
