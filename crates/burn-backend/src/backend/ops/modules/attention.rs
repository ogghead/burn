use core::f32;
#[allow(unused_imports)]
use num_traits::Float as _;

use burn_std::{Shape, Slice};

use crate::{
    Backend, TensorMetadata, get_device_settings,
    ops::AttentionModuleOptions,
    tensor::{BoolTensor, FloatTensor},
};

/// Computes softmax(QKᵗ * scale) · V using separate kernels.
/// Serves as a fallback when FlashAttention is not used.
pub fn attention_fallback<B: Backend>(
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

    // NaN-safe softmax: S = softmax(A)
    // When all positions in a row are masked (-inf), naive softmax has two NaN paths:
    //   (1) max is -inf, so the shift -inf - (-inf) = NaN;
    //   (2) after fixing (1), all exp values are 0, so sum is 0 and 0/0 = NaN.
    // Clamping max to finfo.min (most negative finite value) and sum to finfo.min_positive
    // (smallest positive normal) avoids both, yielding 0 for fully-masked rows.
    let finfo = attention_scores.dtype().finfo().expect("float tensor");
    let max_per_dim = B::float_max_dim(attention_scores.clone(), 3);
    let max_per_dim = B::float_clamp_min(max_per_dim, finfo.min.into());
    let minus_max = B::float_sub(attention_scores, max_per_dim);
    let numerator = B::float_exp(minus_max);
    let sum_exp = B::float_sum_dim(numerator.clone(), 3);
    let sum_exp = B::float_clamp_min(sum_exp, finfo.min_positive.into());
    let softmax = B::float_div(numerator, sum_exp);

    // Context: S · V
    B::float_matmul(softmax, value)
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

/// Decomposed backward for [`attention_fallback`]. Returns
/// `(grad_query, grad_key, grad_value)`.
///
/// Supports bool `mask`, causal masking, and additive `attn_bias` (which enters
/// only through the recomputed probabilities `P`). Softcap is NOT supported (it
/// changes the score Jacobian) and is asserted absent. This is the backend
/// default for `ModuleOps::attention_backward`; backends with a fused
/// FlashAttention backward (e.g. cube) override it.
///
/// Math (scale = 1/sqrt(d) unless overridden):
///   P  = softmax(scale·QKᵀ [+bias] [+masks])
///   D  = rowsum(dO ⊙ O)                         (per query row)
///   dP = dO · Vᵀ
///   dS = P ⊙ (dP − D)
///   dQ = scale · dS · K
///   dK = scale · dSᵀ · Q
///   dV = Pᵀ · dO
#[allow(clippy::too_many_arguments)]
pub fn attention_backward_fallback<B: Backend>(
    query: FloatTensor<B>,
    key: FloatTensor<B>,
    value: FloatTensor<B>,
    out: FloatTensor<B>,
    grad_out: FloatTensor<B>,
    mask: Option<BoolTensor<B>>,
    attn_bias: Option<FloatTensor<B>>,
    options: AttentionModuleOptions,
) -> (FloatTensor<B>, FloatTensor<B>, FloatTensor<B>) {
    assert!(
        options.softcap.is_none(),
        "attention_backward_fallback: softcap is not supported"
    );

    let query_shape = query.shape().dims::<4>();
    let scale = options
        .scale
        .unwrap_or_else(|| 1.0 / (*query_shape.last().unwrap() as f64).sqrt());

    // Plain case (no bool mask / additive bias / causal — the flux training path):
    // choose between two correct backends by attention size.
    //
    // The dense F32 path below is faster (single batched tensor-core matmul per
    // step, ~8s/step at 1536) but materializes the full [seq_q × seq_kv] score /
    // P / dP / dS matrices in f32 — several copies at B·H·seq_q·seq_kv·4 bytes.
    // At joint seq ≈ 9.7k (1536px) that peaks ~80GB and still fits the 96GB card;
    // above that (≥1792px, seq ≈ 13k) it OOMs.
    //
    // The BLOCKED backward tiles over KV blocks and never materializes those
    // matrices (O(seq) memory), so it survives arbitrarily high resolutions, but
    // its serial per-block float_matmul loop is ~60% slower per step. So we only
    // switch to it when the dense path would blow the memory budget — keeping the
    // fast path for every resolution that fits. Both are numerically identical
    // (ema matched at 1280). The dense path also stays for the rare
    // masked/causal/bias cases below.
    //
    // Threshold history: originally 1.2e8 ("dense fits to 1536"), measured on
    // small July-2026 probe configs. The production LoRA-training footprint has
    // since grown (bigger cached datasets, held-out identity eval, face-weight)
    // — a 1280px run (seq ≈ 6.8k, 4.7e7 elems) OOM'd a 96GB card on the DENSE
    // path 2026-07-16 while 1152px (3.2e7) peaked ~89GB. New default 4.0e7
    // keeps ≤1152 dense and sends 1280+ through the blocked path. Env override
    // `ATTN_DENSE_MAX_SCORE_ELEMS` (absolute element count) for experiments.
    let seq_q = query_shape[2] as u64;
    let seq_kv = key.shape().dims::<4>()[2] as u64;
    // between 1152 (3.2e7) and 1280 (4.7e7)
    const DEFAULT_DENSE_MAX_SCORE_ELEMS: u64 = 40_000_000;
    #[cfg(feature = "std")]
    let blocked_score_elems: u64 = std::env::var("ATTN_DENSE_MAX_SCORE_ELEMS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_DENSE_MAX_SCORE_ELEMS);
    #[cfg(not(feature = "std"))]
    let blocked_score_elems: u64 = DEFAULT_DENSE_MAX_SCORE_ELEMS;
    let plain = mask.is_none() && attn_bias.is_none() && !options.is_causal;
    if plain && seq_q.saturating_mul(seq_kv) > blocked_score_elems {
        return attention_backward_blocked::<B>(query, key, value, out, grad_out, scale);
    }

    // Compute the whole backward in F32. The inputs are often f16 (the fused-flash
    // path casts q/k/v to f16 for speed), but `dK = scale·Σ_q dSᵀ·Q` and
    // `dV = Σ_q Pᵀ·dO` sum over the QUERY axis WITHOUT the softmax normalization
    // (unlike dQ, which is a p-weighted average), so their magnitude grows with
    // seq_q and exceeds f16's max (~65504) above ~6000 tokens → inf → NaN via Adam.
    // Upcasting to f32 gives the range/precision to represent these legitimately
    // large grads; the returned grads stay f32 (the autodiff cast-back to the bf16
    // param dtype has ample range). This is why the unfused (bf16) path was finite.
    let f32 = burn_std::FloatDType::F32;
    let query = B::float_cast(query, f32);
    let key = B::float_cast(key, f32);
    let value = B::float_cast(value, f32);
    let out = B::float_cast(out, f32);
    let grad_out = B::float_cast(grad_out, f32);
    let attn_bias = attn_bias.map(|b| B::float_cast(b, f32));

    // Recompute P = softmax(scale·QKᵀ [+bias] [+masks]) — NaN-safe, mirroring the
    // forward fallback.
    let transposed_key = B::float_transpose(key.clone());
    let qk = B::float_matmul(query.clone(), transposed_key);
    let scores = B::float_mul_scalar(qk, scale.into());
    let scores = if let Some(mask) = mask {
        B::float_mask_fill(scores, mask, f32::NEG_INFINITY.into())
    } else {
        scores
    };
    let scores = if options.is_causal {
        let causal_mask = build_causal_mask::<B>(&scores);
        B::float_mask_fill(scores, causal_mask, f32::NEG_INFINITY.into())
    } else {
        scores
    };
    let scores = if let Some(bias) = attn_bias {
        B::float_add(scores, bias)
    } else {
        scores
    };
    let finfo = scores.dtype().finfo().expect("float tensor");
    let max_per_dim = B::float_clamp_min(B::float_max_dim(scores.clone(), 3), finfo.min.into());
    let numerator = B::float_exp(B::float_sub(scores, max_per_dim));
    let sum_exp = B::float_clamp_min(B::float_sum_dim(numerator.clone(), 3), finfo.min_positive.into());
    let p = B::float_div(numerator, sum_exp);

    // D = rowsum(dO ⊙ O) over the last (value) dim → [B, H, seq_q, 1].
    let d_row = B::float_sum_dim(B::float_mul(grad_out.clone(), out), 3);
    // dP = dO · Vᵀ → [B, H, seq_q, seq_kv].
    let transposed_value = B::float_transpose(value);
    let dp = B::float_matmul(grad_out.clone(), transposed_value);
    // dS = P ⊙ (dP − D).
    let ds = B::float_mul(p.clone(), B::float_sub(dp, d_row));

    // dQ = scale · dS · K ; dK = scale · dSᵀ · Q ; dV = Pᵀ · dO.
    let grad_query = B::float_mul_scalar(B::float_matmul(ds.clone(), key), scale.into());
    let ds_t = B::float_transpose(ds);
    let grad_key = B::float_mul_scalar(B::float_matmul(ds_t, query), scale.into());
    let p_t = B::float_transpose(p);
    let grad_value = B::float_matmul(p_t, grad_out);

    (grad_query, grad_key, grad_value)
}

/// O(seq)-MEMORY blocked FlashAttention backward for the PLAIN case (no bool
/// mask, no additive bias, non-causal — the flux training path). Tiles over KV
/// blocks so the full `[seq_q × seq_kv]` score/dP/dS matrices are never
/// materialized (peak `O(seq_q × block)`); every matmul is a batched f32
/// `float_matmul` (tensor cores). Numerically identical to the dense fallback.
///
/// Pass 1 recomputes the per-row `lse` via an online (running max/sum) softmax
/// over KV blocks. Pass 2 computes the grads block-by-block:
///   P_j = exp(scale·Q·K_jᵀ − lse);  dP_j = dO·V_jᵀ;  dS_j = P_j⊙(dP_j − D)
///   dV[j] = P_jᵀ·dO;  dK[j] = scale·dS_jᵀ·Q;  dQ += scale·dS_j·K_j.
/// Returns f32 (dq, dk, dv) — the caller casts back to the param dtype.
fn attention_backward_blocked<B: Backend>(
    query: FloatTensor<B>,
    key: FloatTensor<B>,
    value: FloatTensor<B>,
    out: FloatTensor<B>,
    grad_out: FloatTensor<B>,
    scale: f64,
) -> (FloatTensor<B>, FloatTensor<B>, FloatTensor<B>) {
    let f32 = burn_std::FloatDType::F32;

    // MIXED PRECISION (default ON; `ATTN_BLOCKED_BF16=0` reverts to all-f32):
    // the MATMULS — the O(seq·block·dim) flops that dominate this backward —
    // run on bf16 operands at full tensor-core rate (the kernels accumulate
    // in f32 internally either way), while every ELEMENTWISE numeric —
    // softmax/lse state, dS math, the dQ/dK/dV accumulators — stays f32.
    // bf16 keeps f32's exponent RANGE, so the historic dK/dV
    // sum-over-queries overflow (an f16 range problem, see the dense-path
    // comment) cannot recur; the only cost is bf16 mantissa on matmul
    // inputs/outputs, which is the precision every native-bf16 trainer
    // (torch flash backward included) already accepts. Bonus: the resident
    // operand copies are half the size of the old all-f32 upcast.
    #[cfg(feature = "std")]
    let mm_bf16 = std::env::var("ATTN_BLOCKED_BF16").map(|v| v != "0").unwrap_or(true);
    #[cfg(not(feature = "std"))]
    let mm_bf16 = true;
    let mm_dtype = if mm_bf16 { burn_std::FloatDType::BF16 } else { f32 };

    // Matmul-operand copies in the matmul dtype…
    let query = B::float_cast(query, mm_dtype);
    let key = B::float_cast(key, mm_dtype);
    let value = B::float_cast(value, mm_dtype);
    let grad_out_mm = B::float_cast(grad_out.clone(), mm_dtype);
    // …and a transient f32 pair just for D = rowsum(dO ⊙ O) (dropped after).
    let out = B::float_cast(out, f32);
    let grad_out = B::float_cast(grad_out, f32);
    // Upcast matmul results back to f32 for the elementwise math.
    let up = |t: FloatTensor<B>| -> FloatTensor<B> {
        if mm_bf16 { B::float_cast(t, f32) } else { t }
    };
    let down = |t: FloatTensor<B>| -> FloatTensor<B> {
        if mm_bf16 { B::float_cast(t, burn_std::FloatDType::BF16) } else { t }
    };

    let device = B::float_device(&query);
    let [b, h, seq_q, head_dim] = query.shape().dims::<4>();
    let [_, _, seq_kv, val_dim] = value.shape().dims::<4>();
    let finfo = grad_out.dtype().finfo().expect("float tensor");

    // Full-range slices for the leading/trailing dims (only dim 2 = KV is tiled).
    let hd_slice = || {
        [
            Slice::new(0, Some(b as isize), 1),
            Slice::new(0, Some(h as isize), 1),
            Slice::new(0, Some(0), 1), // placeholder, overwritten below
            Slice::new(0, Some(head_dim as isize), 1),
        ]
    };
    let vd_slice = || {
        [
            Slice::new(0, Some(b as isize), 1),
            Slice::new(0, Some(h as isize), 1),
            Slice::new(0, Some(0), 1),
            Slice::new(0, Some(val_dim as isize), 1),
        ]
    };

    // D = rowsum(dO ⊙ O) → [b, h, seq_q, 1]. Computed in f32, then the f32
    // dO/O copies are dropped — the block loops only touch the mm-dtype pair.
    let d_row = B::float_sum_dim(B::float_mul(grad_out.clone(), out), 3);
    drop(grad_out);

    let block: usize = seq_kv.min(1024);
    let num_blocks = seq_kv.div_ceil(block);

    // --- Pass 1: lse = m + log(l), online softmax over KV blocks ---
    let mut m = B::float_full(
        Shape::new([b, h, seq_q, 1]),
        finfo.min.into(),
        &device,
        f32,
    );
    let mut l = B::float_zeros(Shape::new([b, h, seq_q, 1]), &device, f32);
    for jb in 0..num_blocks {
        let s = (jb * block) as isize;
        let e = ((jb + 1) * block).min(seq_kv) as isize;
        let mut sl = hd_slice();
        sl[2] = Slice::new(s, Some(e), 1);
        let k_j = B::float_slice(key.clone(), &sl); // [b,h,bk,hd] (mm dtype)
        let k_jt = B::float_swap_dims(k_j, 2, 3); // [b,h,hd,bk]
        let s_j =
            B::float_mul_scalar(up(B::float_matmul(query.clone(), k_jt)), scale.into());
        let row_max = B::float_max_dim(s_j.clone(), 3); // [b,h,sq,1]
        // m_new = max(m, row_max) = m + relu(row_max - m).
        let m_new = B::float_add(
            m.clone(),
            B::float_clamp_min(B::float_sub(row_max, m.clone()), 0.0.into()),
        );
        // l = l·exp(m - m_new) + rowsum(exp(s_j - m_new)).
        let corr = B::float_exp(B::float_sub(m.clone(), m_new.clone()));
        let l_rescaled = B::float_mul(l, corr);
        let exp_sj = B::float_exp(B::float_sub(s_j, m_new.clone()));
        l = B::float_add(l_rescaled, B::float_sum_dim(exp_sj, 3));
        m = m_new;
    }
    let l = B::float_clamp_min(l, finfo.min_positive.into());
    let lse = B::float_add(m, B::float_log(l)); // [b,h,sq,1]

    // --- Pass 2: grads, blocked over KV ---
    let mut dq = B::float_zeros(Shape::new([b, h, seq_q, head_dim]), &device, f32);
    let mut dk = B::float_zeros(Shape::new([b, h, seq_kv, head_dim]), &device, f32);
    let mut dv = B::float_zeros(Shape::new([b, h, seq_kv, val_dim]), &device, f32);
    for jb in 0..num_blocks {
        let s = (jb * block) as isize;
        let e = ((jb + 1) * block).min(seq_kv) as isize;
        let mut k_sl = hd_slice();
        k_sl[2] = Slice::new(s, Some(e), 1);
        let mut v_sl = vd_slice();
        v_sl[2] = Slice::new(s, Some(e), 1);

        let k_j = B::float_slice(key.clone(), &k_sl); // [b,h,bk,hd] (mm dtype)
        let v_j = B::float_slice(value.clone(), &v_sl); // [b,h,bk,vd] (mm dtype)

        let k_jt = B::float_swap_dims(k_j.clone(), 2, 3); // [b,h,hd,bk]
        let s_j =
            B::float_mul_scalar(up(B::float_matmul(query.clone(), k_jt)), scale.into());
        let p_j = B::float_exp(B::float_sub(s_j, lse.clone())); // [b,h,sq,bk] f32

        let v_jt = B::float_swap_dims(v_j, 2, 3); // [b,h,vd,bk]
        let dp_j = up(B::float_matmul(grad_out_mm.clone(), v_jt)); // [b,h,sq,bk] f32
        let ds_j = B::float_mul(p_j.clone(), B::float_sub(dp_j, d_row.clone())); // [b,h,sq,bk] f32
        let p_mm = down(p_j); // matmul-dtype copy for the dV matmul
        let ds_mm = down(ds_j); // matmul-dtype copy for the dK/dQ matmuls

        // dV[j] = P_jᵀ · dO → [b,h,bk,vd].
        let p_jt = B::float_swap_dims(p_mm, 2, 3); // [b,h,bk,sq]
        let dv_j = up(B::float_matmul(p_jt, grad_out_mm.clone()));
        // dK[j] = scale · dS_jᵀ · Q → [b,h,bk,hd].
        let ds_jt = B::float_swap_dims(ds_mm.clone(), 2, 3); // [b,h,bk,sq]
        let dk_j =
            B::float_mul_scalar(up(B::float_matmul(ds_jt, query.clone())), scale.into());

        dv = B::float_slice_assign(dv, &v_sl, dv_j);
        dk = B::float_slice_assign(dk, &k_sl, dk_j);

        // dQ += scale · dS_j · K_j → [b,h,sq,hd].
        let dq_inc =
            B::float_mul_scalar(up(B::float_matmul(ds_mm, k_j)), scale.into());
        dq = B::float_add(dq, dq_inc);
    }

    (dq, dk, dv)
}
