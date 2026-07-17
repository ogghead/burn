use crate::{
    CubeBackend, CubeRuntime, kernel::attention::attention_autotune,
    kernel::into_contiguous, ops::numeric::empty_device_dtype, tensor::CubeTensor,
};
use burn_backend::{
    DType, Shape,
    ops::{AttentionModuleOptions, attention::attention_fallback},
};
use cubek::attention::forward::{
    definition::{
        AccumulatorPrecision, AttentionGlobalTypes, AttentionOptions, AttentionSetupError,
    },
    launch,
    routines::blackbox_accelerated::BlackboxAcceleratedStrategy,
};

#[derive(Debug)]
/// Strategy used to select which attention implementation to run.
pub enum AttentionStrategy {
    /// Flash Attention using accelerated inner matmuls.
    FlashBlackboxAccelerated(BlackboxAcceleratedStrategy),

    /// Flash Attention using unit inner matmuls.
    FlashUnit,

    /// Flash Attention with FP8 (e4m3) quantized QK^T (Sage-v1 style: device
    /// prepass quantizes Q and mean-smoothed K per 16-seq-row tile, scales
    /// fold into the softmax factor). Env-forced via `ATTN_QK_QUANT=fp8`
    /// only — never registered in autotune. FAILS LOUD when the device lacks
    /// the e4m3×e4m3→f32 MMA instruction (no silent fallback).
    QuantFp8,

    /// Stage B: training-free block-sparse Flash Attention (SpargeAttn
    /// style). A device predictor prepass (block means → similarity map →
    /// per-row top-p at `tau` + self-similarity gate + always-keep
    /// {block 0, diagonal ± 1}) selects which 16-row KV blocks each q-stage
    /// visits; the kernel's global loop skips the rest (restricted-set
    /// online softmax — no correction term). Host-side fallback: if the
    /// predicted sparsity is below `min_sparsity`, launch the dense route
    /// instead (the predictor cost is bounded and already paid).
    /// `quant = true` runs the FP8 substrate underneath (Stage A quant +
    /// sparsity compose). Env-forced via `ATTN_SPARGE_TAU=<f32>` only —
    /// never registered in autotune. Inference-only.
    Sparse {
        /// Top-p softmax mass to keep per (q-stage, head) row (e.g. 0.99).
        tau: f32,
        /// Dense fallback threshold on predicted sparsity (default 0.15).
        min_sparsity: f32,
        /// Use the FP8 (e4m3) quantized QK^T substrate for the kept blocks.
        quant: bool,
    },

    /// Fallback implementation using multiple separate kernels.
    Fallback,

    /// Automatically benchmark and select the best strategy at runtime.
    #[cfg(feature = "autotune")]
    Autotune,
}

impl Default for AttentionStrategy {
    fn default() -> Self {
        // if autotune is enabled, default to autotune
        #[cfg(feature = "autotune")]
        return AttentionStrategy::Autotune;

        // if autotune is disabled, default to fallback to make sure it runs
        #[cfg(not(feature = "autotune"))]
        AttentionStrategy::Fallback
    }
}

#[allow(clippy::too_many_arguments)]
/// Launch an attention kernel with given strategy
pub fn attention<R: CubeRuntime>(
    query: CubeTensor<R>,
    key: CubeTensor<R>,
    value: CubeTensor<R>,
    mask: Option<CubeTensor<R>>,
    attn_bias: Option<CubeTensor<R>>,
    options: AttentionModuleOptions,
    strategy: AttentionStrategy,
) -> Result<CubeTensor<R>, AttentionSetupError> {
    match strategy {
        AttentionStrategy::FlashBlackboxAccelerated(strategy) => flash_attention(
            query,
            key,
            value,
            mask,
            attn_bias,
            options,
            launch::Strategy::BlackboxAccelerated(launch::BlueprintStrategy::Inferred(strategy)),
        ),
        AttentionStrategy::FlashUnit => flash_attention(
            query,
            key,
            value,
            mask,
            attn_bias,
            options,
            launch::Strategy::Unit(launch::BlueprintStrategy::Inferred(())),
        ),
        AttentionStrategy::QuantFp8 => {
            flash_attention_quant_fp8(query, key, value, mask, attn_bias, options)
        }
        AttentionStrategy::Sparse {
            tau,
            min_sparsity,
            quant,
        } => flash_attention_sparse(
            query,
            key,
            value,
            mask,
            attn_bias,
            options,
            tau,
            min_sparsity,
            quant,
        ),
        AttentionStrategy::Fallback => Ok(attention_fallback::<CubeBackend<R, f32, i32, u8>>(
            query, key, value, mask, attn_bias, options,
        )),
        #[cfg(feature = "autotune")]
        AttentionStrategy::Autotune => Ok(attention_autotune(
            query, key, value, mask, attn_bias, options,
        )),
    }
}

#[allow(clippy::too_many_arguments)]
/// Launch a flash attention kernel
pub fn flash_attention<R: CubeRuntime>(
    query: CubeTensor<R>,
    key: CubeTensor<R>,
    value: CubeTensor<R>,
    mask: Option<CubeTensor<R>>,
    _attn_bias: Option<CubeTensor<R>>,
    options: AttentionModuleOptions,
    strategy: launch::Strategy,
) -> Result<CubeTensor<R>, AttentionSetupError> {
    let query = into_contiguous(query);
    let key = into_contiguous(key);
    let value = into_contiguous(value);

    let client = query.client.clone();
    let out = init_attention_output(&query, &value);

    let dtypes = AttentionGlobalTypes {
        query: query.dtype.into(),
        key: key.dtype.into(),
        value: value.dtype.into(),
        mask: mask.as_ref().map(|m| m.dtype).unwrap_or(DType::U8).into(),
        out: out.dtype.into(),
    };

    launch::launch_ref::<R>(
        strategy,
        &client,
        query.binding(),
        key.binding(),
        value.binding(),
        mask.map(|mask| mask.binding()),
        out.clone().binding(),
        &dtypes,
        AttentionOptions {
            causal: options.is_causal,
            accumulator_precision: AccumulatorPrecision::Strict(cubecl::ir::StorageType::Scalar(
                cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F32),
            )),
        },
    )?;

    Ok(out)
}

/// Launch the Stage A FP8 (e4m3) quantized-QK^T flash attention.
///
/// From full-precision q/k/v (any float dtype the prepass supports: f16 /
/// bf16 / f32), this runs the quantization prepass on device — per-channel K
/// mean (smoothing), then symmetric per-16-seq-row-tile e4m3 quantization of
/// Q and of (K − k̄) — and calls cubek's `launch_quant_fp8_ref` with the
/// quantized operands + f32 scale tensors. The value matmul and output stay
/// full precision inside the kernel (tf32 value tiles, f32 out stage; the
/// output tensor dtype is the query dtype, exactly like [`flash_attention`]).
///
/// FAILS LOUD (`AttentionSetupError`) when the device lacks the
/// e4m3×e4m3→f32 MMA instruction (CUDA arch >= 89) or when `seq_q` is not a
/// multiple of 16 (the quant routine's stage granularity) — never a silent
/// fallback, so a mis-deployed `ATTN_QK_QUANT=fp8` is visible immediately.
pub fn flash_attention_quant_fp8<R: CubeRuntime>(
    query: CubeTensor<R>,
    key: CubeTensor<R>,
    value: CubeTensor<R>,
    mask: Option<CubeTensor<R>>,
    _attn_bias: Option<CubeTensor<R>>,
    options: AttentionModuleOptions,
) -> Result<CubeTensor<R>, AttentionSetupError> {
    use cubecl::ir::{ElemType, FloatKind, StorageType};
    use cubecl::std::tensor::TensorHandle;
    use cubek::attention::forward::launch::BlueprintStrategy;
    use cubek::attention::forward::prepass::{
        QUANT_FP8_TILE_ROWS, k_mean_launch, quant_fp8_launch,
    };
    use cubek::attention::forward::routines::quant_fp8::QuantFp8Strategy;

    let query = into_contiguous(query);
    let key = into_contiguous(key);
    let value = into_contiguous(value);

    let client = query.client.clone();

    let e4m3_st = StorageType::Scalar(ElemType::Float(FloatKind::E4M3));
    let f32_st = StorageType::Scalar(ElemType::Float(FloatKind::F32));

    // Fail loud BEFORE the prepass: the routine re-checks this, but by then
    // the e4m3 intermediates would already have been written — and runtimes
    // without fp8 support store e4m3 as silent zeros.
    let has_fp8_mma = client
        .properties()
        .features
        .matmul
        .mma
        .iter()
        .any(|it| it.a_type == e4m3_st && it.b_type == e4m3_st && it.cd_type == f32_st);
    if !has_fp8_mma {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "ATTN_QK_QUANT=fp8: device has no e4m3*e4m3->f32 MMA instruction \
             (needs CUDA arch >= 89); refusing to fall back silently"
                .to_string(),
        )));
    }

    let [batch, num_heads, seq_q, head_dim] = [
        query.meta.shape[0],
        query.meta.shape[1],
        query.meta.shape[2],
        query.meta.shape[3],
    ];
    let seq_kv = key.meta.shape[2];
    let tile_rows = QUANT_FP8_TILE_ROWS as usize;

    // First plane count whose stage (num_planes * 16 seq_q rows) divides
    // seq_q; preference order per the A3 flux_4608/flux_16384 bench sweep
    // (np=4 fastest at both sizes, np=8 close second).
    let num_planes = [4u8, 8, 2, 1]
        .into_iter()
        .find(|np| seq_q % (*np as usize * tile_rows) == 0)
        .ok_or_else(|| {
            AttentionSetupError::InvalidConfig(Box::new(format!(
                "ATTN_QK_QUANT=fp8: seq_q {seq_q} is not a multiple of the quant \
                 tile granularity ({tile_rows}); refusing to fall back silently"
            )))
        })?;

    // Prepass intermediates live outside burn's tensor world (burn DType has
    // no fp8), as raw cubecl handles: e4m3 Q/K plus f32 k-mean and scales.
    let alloc = |shape: Vec<usize>, st: StorageType| {
        let bytes = shape.iter().product::<usize>() * st.size();
        TensorHandle::<R>::new_contiguous(shape, client.empty(bytes), st)
    };
    let mean = alloc(vec![batch, num_heads, head_dim], f32_st);
    let q_fp8 = alloc(vec![batch, num_heads, seq_q, head_dim], e4m3_st);
    let k_fp8 = alloc(vec![batch, num_heads, seq_kv, head_dim], e4m3_st);
    let scale_q = alloc(vec![batch, num_heads, seq_q.div_ceil(tile_rows)], f32_st);
    let scale_k = alloc(vec![batch, num_heads, seq_kv.div_ceil(tile_rows)], f32_st);

    let q_st: cubecl::ir::StorageType = query.dtype.into();
    let k_st: cubecl::ir::StorageType = key.dtype.into();

    k_mean_launch::<R>(&client, key.clone().binding(), mean.clone().binding(), k_st)?;
    quant_fp8_launch::<R>(
        &client,
        query.clone().binding(),
        None,
        q_fp8.clone().binding(),
        scale_q.clone().binding(),
        q_st,
    )?;
    quant_fp8_launch::<R>(
        &client,
        key.binding(),
        Some(mean.binding()),
        k_fp8.clone().binding(),
        scale_k.clone().binding(),
        k_st,
    )?;

    let out = init_attention_output(&query, &value);

    // Built manually: burn's DType has no fp8 representation, so the e4m3
    // query/key storage types can't round-trip through `dtype.into()`.
    let dtypes = AttentionGlobalTypes {
        query: e4m3_st,
        key: e4m3_st,
        value: value.dtype.into(),
        mask: mask.as_ref().map(|m| m.dtype).unwrap_or(DType::U8).into(),
        out: out.dtype.into(),
    };

    launch::launch_quant_fp8_ref::<R>(
        BlueprintStrategy::Inferred(QuantFp8Strategy {
            num_planes,
            seq_q: 1,
            seq_kv: 1,
        }),
        &client,
        q_fp8.binding(),
        k_fp8.binding(),
        value.binding(),
        mask.map(|mask| mask.binding()),
        out.clone().binding(),
        scale_q.binding(),
        scale_k.binding(),
        &dtypes,
        AttentionOptions {
            causal: options.is_causal,
            accumulator_precision: AccumulatorPrecision::Strict(cubecl::ir::StorageType::Scalar(
                cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F32),
            )),
        },
    )?;

    Ok(out)
}

/// Launch the Stage B block-sparse flash attention (SpargeAttn style,
/// training-free), optionally on the FP8 quantized substrate.
///
/// Per call, on device: sparse predictor prepass (block means over q-stage /
/// kv blocks → self-similarity gate → block-similarity softmax → per-row
/// top-p selection at `tau` + always-keep {block 0, diagonal ± 1}), then a
/// SMALL host readback of the per-row kept counts to decide the fallback:
/// if predicted sparsity < `min_sparsity` the dense route launches instead
/// (Stage A quant when `quant`, dense f16 blackbox otherwise), so the worst
/// case is dense + a bounded predictor cost. Materialized masks fall back
/// to dense BEFORE this shim (see the `ATTN_SPARGE_TAU` gate in
/// `ops/module.rs`).
///
/// FAILS LOUD on invalid `tau` and (via cubek) on any kept-tensor geometry
/// mismatch; `quant` additionally inherits the fail-loud e4m3 MMA check.
/// Inference-only: do not enable during training (the sparse forward would
/// feed the backward).
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_sparse<R: CubeRuntime>(
    query: CubeTensor<R>,
    key: CubeTensor<R>,
    value: CubeTensor<R>,
    mask: Option<CubeTensor<R>>,
    attn_bias: Option<CubeTensor<R>>,
    options: AttentionModuleOptions,
    tau: f32,
    min_sparsity: f32,
    quant: bool,
) -> Result<CubeTensor<R>, AttentionSetupError> {
    use cubecl::CubeElement as _;
    use cubecl::ir::{ElemType, StorageType};
    use cubecl::prelude::CubePrimitive as _;
    use cubecl::std::tensor::TensorHandle;
    use cubek::attention::forward::prepass::{
        QUANT_FP8_TILE_ROWS, SPARSE_DEFAULT_SELF_SIM, sparse_predict_launch,
    };

    if !(0.0..=1.0).contains(&tau) {
        return Err(AttentionSetupError::InvalidConfig(Box::new(format!(
            "ATTN_SPARGE_TAU={tau}: tau must be in [0, 1]"
        ))));
    }
    if mask.is_some() {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "flash_attention_sparse: materialized masks are not supported; \
             the caller must route masked attention to the dense path"
                .to_string(),
        )));
    }

    let query = into_contiguous(query);
    let key = into_contiguous(key);
    let value = into_contiguous(value);

    let client = query.client.clone();

    let [batch, num_heads, seq_q, _head_dim] = [
        query.meta.shape[0],
        query.meta.shape[1],
        query.meta.shape[2],
        query.meta.shape[3],
    ];
    let seq_kv = key.meta.shape[2];
    let tile_rows = QUANT_FP8_TILE_ROWS as usize; // 16: kv block = score-tile rows

    // Same plane-count preference as the quant shim (A3 sweep: np4 best).
    // The q-stage block is np * 16 rows; the predictor must match it.
    let num_planes = [4u8, 8, 2, 1]
        .into_iter()
        .find(|np| seq_q % (*np as usize * tile_rows) == 0)
        .ok_or_else(|| {
            AttentionSetupError::InvalidConfig(Box::new(format!(
                "ATTN_SPARGE_TAU: seq_q {seq_q} is not a multiple of the stage \
                 granularity ({tile_rows}); refusing to fall back silently"
            )))
        })?;
    let q_block_rows = num_planes as usize * tile_rows;
    let n_q_stages = seq_q / q_block_rows;
    let n_kv_blocks = seq_kv.div_ceil(tile_rows);

    // Kept-index tensors live outside burn's tensor world (u32 payload) as
    // raw cubecl handles, like the fp8 prepass intermediates.
    let u32_st = u32::as_type_native_unchecked().storage_type();
    let alloc = |shape: Vec<usize>, st: StorageType| {
        let bytes = shape.iter().product::<usize>() * st.size();
        TensorHandle::<R>::new_contiguous(shape, client.empty(bytes), st)
    };
    let kept_idx = alloc(
        vec![batch, num_heads, n_q_stages, n_kv_blocks],
        u32_st,
    );
    let kept_count = alloc(vec![batch, num_heads, n_q_stages], u32_st);

    let q_st: cubecl::ir::StorageType = query.dtype.into();
    let k_st: cubecl::ir::StorageType = key.dtype.into();

    sparse_predict_launch::<R>(
        &client,
        query.clone().binding(),
        key.clone().binding(),
        kept_idx.clone().binding(),
        kept_count.clone().binding(),
        q_block_rows as u32,
        tile_rows as u32,
        tau,
        SPARSE_DEFAULT_SELF_SIM,
        q_st,
        k_st,
    )?;

    // Host fallback decision: mean kept fraction across all rows.
    let counts_bytes =
        client.read_one_unchecked_tensor(kept_count.clone().into_copy_descriptor());
    let counts = u32::from_bytes(&counts_bytes);
    let kept_total: usize = counts.iter().map(|&c| c as usize).sum();
    let sparsity = 1.0 - kept_total as f32 / (counts.len() * n_kv_blocks) as f32;

    if sparsity < min_sparsity {
        // Not enough to win: dense launch (quant substrate if requested).
        // The predictor cost is already paid and bounded (~1-2% of a dense
        // 16k attention).
        return if quant {
            flash_attention_quant_fp8(query, key, value, None, attn_bias, options)
        } else {
            flash_attention(
                query,
                key,
                value,
                None,
                attn_bias,
                options,
                launch::Strategy::BlackboxAccelerated(launch::BlueprintStrategy::Inferred(
                    BlackboxAcceleratedStrategy {
                        num_planes,
                        seq_q: 1,
                        seq_kv: 1,
                    },
                )),
            )
        };
    }

    let attention_options = AttentionOptions {
        causal: options.is_causal,
        accumulator_precision: AccumulatorPrecision::Strict(cubecl::ir::StorageType::Scalar(
            cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F32),
        )),
    };

    if quant {
        use cubecl::ir::FloatKind;
        use cubek::attention::forward::launch::BlueprintStrategy;
        use cubek::attention::forward::prepass::{k_mean_launch, quant_fp8_launch};
        use cubek::attention::forward::routines::quant_fp8::QuantFp8Strategy;

        let e4m3_st = StorageType::Scalar(ElemType::Float(FloatKind::E4M3));
        let f32_st = StorageType::Scalar(ElemType::Float(FloatKind::F32));

        // Same fail-loud pre-check as the dense quant shim.
        let has_fp8_mma = client
            .properties()
            .features
            .matmul
            .mma
            .iter()
            .any(|it| it.a_type == e4m3_st && it.b_type == e4m3_st && it.cd_type == f32_st);
        if !has_fp8_mma {
            return Err(AttentionSetupError::InvalidConfig(Box::new(
                "ATTN_SPARGE_TAU + ATTN_QK_QUANT=fp8: device has no \
                 e4m3*e4m3->f32 MMA instruction (needs CUDA arch >= 89); \
                 refusing to fall back silently"
                    .to_string(),
            )));
        }

        let head_dim = query.meta.shape[3];
        let mean = alloc(vec![batch, num_heads, head_dim], f32_st);
        let q_fp8 = alloc(vec![batch, num_heads, seq_q, head_dim], e4m3_st);
        let k_fp8 = alloc(vec![batch, num_heads, seq_kv, head_dim], e4m3_st);
        let scale_q = alloc(vec![batch, num_heads, seq_q.div_ceil(tile_rows)], f32_st);
        let scale_k = alloc(vec![batch, num_heads, seq_kv.div_ceil(tile_rows)], f32_st);

        k_mean_launch::<R>(&client, key.clone().binding(), mean.clone().binding(), k_st)?;
        quant_fp8_launch::<R>(
            &client,
            query.clone().binding(),
            None,
            q_fp8.clone().binding(),
            scale_q.clone().binding(),
            q_st,
        )?;
        quant_fp8_launch::<R>(
            &client,
            key.binding(),
            Some(mean.binding()),
            k_fp8.clone().binding(),
            scale_k.clone().binding(),
            k_st,
        )?;

        let out = init_attention_output(&query, &value);
        let dtypes = AttentionGlobalTypes {
            query: e4m3_st,
            key: e4m3_st,
            value: value.dtype.into(),
            mask: DType::U8.into(),
            out: out.dtype.into(),
        };

        launch::launch_quant_fp8_sparse_ref::<R>(
            BlueprintStrategy::Inferred(QuantFp8Strategy {
                num_planes,
                seq_q: 1,
                seq_kv: 1,
            }),
            &client,
            q_fp8.binding(),
            k_fp8.binding(),
            value.binding(),
            out.clone().binding(),
            scale_q.binding(),
            scale_k.binding(),
            kept_idx.binding(),
            kept_count.binding(),
            &dtypes,
            attention_options,
        )?;
        return Ok(out);
    }

    let out = init_attention_output(&query, &value);
    let dtypes = AttentionGlobalTypes {
        query: query.dtype.into(),
        key: key.dtype.into(),
        value: value.dtype.into(),
        mask: DType::U8.into(),
        out: out.dtype.into(),
    };

    launch::launch_sparse_ref::<R>(
        launch::Strategy::BlackboxAccelerated(launch::BlueprintStrategy::Inferred(
            BlackboxAcceleratedStrategy {
                num_planes,
                seq_q: 1,
                seq_kv: 1,
            },
        )),
        &client,
        query.binding(),
        key.binding(),
        value.binding(),
        out.clone().binding(),
        kept_idx.binding(),
        kept_count.binding(),
        &dtypes,
        attention_options,
    )?;

    Ok(out)
}

#[allow(clippy::too_many_arguments)]
/// Launch the fused FlashAttention backward, returning `(grad_query, grad_key,
/// grad_value)`. Self-contained: cubek recomputes the softmax `lse` from Q,K
/// internally, so only the forward inputs + output + upstream gradient are
/// needed. O(seq) memory (the score matrix is never materialized).
pub fn flash_attention_backward<R: CubeRuntime>(
    query: CubeTensor<R>,
    key: CubeTensor<R>,
    value: CubeTensor<R>,
    out: CubeTensor<R>,
    grad_out: CubeTensor<R>,
    options: AttentionModuleOptions,
) -> Result<(CubeTensor<R>, CubeTensor<R>, CubeTensor<R>), AttentionSetupError> {
    use cubek::attention::backward::{BackwardConfig, flash_attention_backward as cubek_backward};

    let query = into_contiguous(query);
    let key = into_contiguous(key);
    let value = into_contiguous(value);
    let out = into_contiguous(out);
    let grad_out = into_contiguous(grad_out);

    let client = query.client.clone();
    let device = query.device.clone();
    let head_dim = query.meta.shape[3];

    let q_shape = Shape::new([
        query.meta.shape[0],
        query.meta.shape[1],
        query.meta.shape[2],
        query.meta.shape[3],
    ]);
    let k_shape = Shape::new([
        key.meta.shape[0],
        key.meta.shape[1],
        key.meta.shape[2],
        key.meta.shape[3],
    ]);
    let v_shape = Shape::new([
        value.meta.shape[0],
        value.meta.shape[1],
        value.meta.shape[2],
        value.meta.shape[3],
    ]);

    // dq/dk/dv grads are written in F32 by the cubek backward (its accumulators
    // are f32; storing to the f16 input dtype overflowed at large seq → NaN).
    // The autodiff cast-back to the param dtype (bf16, huge range) happens
    // downstream, so returning f32 grads for f16 q/k/v is safe and avoids the
    // overflow. Must match the kernels' `&mut Tensor<f32>` outputs.
    let dq = empty_device_dtype::<R>(client.clone(), device.clone(), q_shape, DType::F32);
    let dk = empty_device_dtype::<R>(client.clone(), device.clone(), k_shape, DType::F32);
    let dv = empty_device_dtype::<R>(client.clone(), device.clone(), v_shape, DType::F32);

    let dtypes = AttentionGlobalTypes {
        query: query.dtype.into(),
        key: key.dtype.into(),
        value: value.dtype.into(),
        mask: DType::U8.into(),
        out: out.dtype.into(),
    };

    let mut config = BackwardConfig::from_head_dim(head_dim);
    config.scale = options
        .scale
        .map(|s| s as f32)
        .unwrap_or_else(|| 1.0 / (head_dim as f32).sqrt());
    config.causal = options.is_causal;

    cubek_backward::<R>(
        &client,
        query.binding(),
        key.binding(),
        value.binding(),
        out.binding(),
        grad_out.binding(),
        dq.clone().binding(),
        dk.clone().binding(),
        dv.clone().binding(),
        &dtypes,
        config,
    )?;

    Ok((dq, dk, dv))
}

pub(crate) fn init_attention_output<R: CubeRuntime>(
    query: &CubeTensor<R>,
    value: &CubeTensor<R>,
) -> CubeTensor<R> {
    let num_batches = query.meta.shape[0];
    let num_heads = query.meta.shape[1];
    let seq_q = query.meta.shape[2];
    let val_dim = value.meta.shape[3];
    let out_shape = Shape::new([num_batches, num_heads, seq_q, val_dim]);

    // The attention output `O` is stored in the query dtype (f16 on the training
    // path). The forward/backward f16-O precision mismatch that diverged high-res
    // training is fixed in the BACKWARD (the cubek prepass now recomputes O in
    // f32 from q/k/v + the f32 lse instead of reading this stored, truncated O),
    // so no forward change is needed — O's storage dtype no longer affects grads.
    empty_device_dtype::<R>(
        query.client.clone(),
        query.device.clone(),
        out_shape,
        query.dtype,
    )
}
