use crate::{
    CubeBackend, CubeRuntime, kernel::attention::attention_autotune,
    ops::numeric::empty_device_dtype, tensor::CubeTensor,
};
use burn_backend::{
    DType, Shape,
    ops::{AttentionModuleOptions, attention::attention_fallback},
};
use cubek::attention::launch;
use cubek::attention::{
    definition::{
        AccumulatorPrecision, AttentionGlobalTypes, AttentionOptions, AttentionSetupError,
    },
    routines::blackbox_accelerated::BlackboxAcceleratedStrategy,
    routines::fp8_accelerated::Fp8AcceleratedStrategy,
};

#[derive(Debug)]
/// Strategy used to select which attention implementation to run.
pub enum AttentionStrategy {
    /// Flash Attention using accelerated inner matmuls.
    FlashBlackboxAccelerated(BlackboxAcceleratedStrategy),

    /// Flash Attention using native FP8 MMA (e4m3×e4m3→f32).
    FlashFp8Accelerated(Fp8AcceleratedStrategy),

    /// Flash Attention using unit inner matmuls.
    FlashUnit,

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
    // FP8A probe: log which attention strategy is dispatched. Set FP8A_PROBE=1 to enable.
    // Uses eprintln! to avoid depending on tracing-subscriber configuration in the host
    // crate. Future runs can re-enable by exporting FP8A_PROBE=1. See task #164.
    if std::env::var_os("FP8A_PROBE").is_some() {
        eprintln!(
            "fp8a-probe: attention strategy dispatch — strategy={:?} q_shape={:?} kv_shape={:?} q_dtype={:?}",
            strategy, query.meta.shape, key.meta.shape, query.dtype
        );
    }
    match strategy {
        AttentionStrategy::FlashBlackboxAccelerated(strategy) => flash_attention(
            query,
            key,
            value,
            mask,
            attn_bias,
            options,
            launch::Strategy::BlackboxAccelerated(
                cubek::attention::launch::BlueprintStrategy::Inferred(strategy),
            ),
        ),
        AttentionStrategy::FlashFp8Accelerated(strategy) => flash_attention(
            query,
            key,
            value,
            mask,
            attn_bias,
            options,
            launch::Strategy::Fp8Accelerated(
                cubek::attention::launch::BlueprintStrategy::Inferred(strategy),
            ),
        ),
        AttentionStrategy::FlashUnit => flash_attention(
            query,
            key,
            value,
            mask,
            attn_bias,
            options,
            launch::Strategy::Unit(cubek::attention::launch::BlueprintStrategy::Inferred(())),
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
    let client = query.client.clone();
    let out = init_attention_output(&query, &value);

    let dtypes = AttentionGlobalTypes {
        query: query.dtype.into(),
        key: key.dtype.into(),
        value: value.dtype.into(),
        mask: mask.as_ref().map(|m| m.dtype).unwrap_or(DType::U8).into(),
        out: out.dtype.into(),
    };

    cubek::attention::launch::launch_ref::<R>(
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

pub(crate) fn init_attention_output<R: CubeRuntime>(
    query: &CubeTensor<R>,
    value: &CubeTensor<R>,
) -> CubeTensor<R> {
    let num_batches = query.meta.shape[0];
    let num_heads = query.meta.shape[1];
    let seq_q = query.meta.shape[2];
    let val_dim = value.meta.shape[3];
    let out_shape = Shape::new([num_batches, num_heads, seq_q, val_dim]);

    empty_device_dtype::<R>(
        query.client.clone(),
        query.device.clone(),
        out_shape,
        query.dtype,
    )
}
