#![cfg_attr(docsrs, feature(doc_cfg))]

extern crate alloc;

use burn_cubecl::CubeBackend;
pub use cubecl::cuda::CudaDevice;
use cubecl::cuda::CudaRuntime;

#[cfg(not(feature = "fusion"))]
pub type Cuda<F = f32, I = i32> = CubeBackend<CudaRuntime, F, I, u8>;

#[cfg(feature = "fusion")]
pub type Cuda<F = f32, I = i32> = burn_fusion::Fusion<CubeBackend<CudaRuntime, F, I, u8>>;

/// PATCH (diffusion-app): live-vs-reserved pool introspection for a device.
///
/// Returns `(bytes_in_use, bytes_reserved, number_allocs)` from cubecl's
/// memory manager, or `None` if the server can't be queried. `bytes_in_use`
/// is the live working set; `bytes_reserved - bytes_in_use` is pool slack
/// (freed slices + pinned sliced pages). The burn `Backend` trait only
/// exposes `memory_cleanup`, so this goes through the cubecl client directly
/// — same path `memory_cleanup` takes internally.
pub fn device_memory_usage(device: &CudaDevice) -> Option<(u64, u64, u64)> {
    use cubecl::prelude::Runtime;
    CudaRuntime::client(device)
        .memory_usage()
        .ok()
        .map(|u| (u.bytes_in_use, u.bytes_reserved, u.number_allocs))
}

#[cfg(all(test, not(target_os = "macos")))]
mod tests {
    use super::*;
    use burn_backend::{Backend, BoolStore, DType, QTensorPrimitive};
    use burn_cubecl::tensor::CubeTensor;

    #[test]
    fn should_support_dtypes() {
        type B = Cuda;
        let device = Default::default();

        assert!(B::supports_dtype(&device, DType::F32));
        assert!(B::supports_dtype(&device, DType::Flex32));
        assert!(B::supports_dtype(&device, DType::F16));
        assert!(B::supports_dtype(&device, DType::BF16));
        assert!(B::supports_dtype(&device, DType::I64));
        assert!(B::supports_dtype(&device, DType::I32));
        assert!(B::supports_dtype(&device, DType::I16));
        assert!(B::supports_dtype(&device, DType::I8));
        assert!(B::supports_dtype(&device, DType::U64));
        assert!(B::supports_dtype(&device, DType::U32));
        assert!(B::supports_dtype(&device, DType::U16));
        assert!(B::supports_dtype(&device, DType::U8));
        assert!(B::supports_dtype(&device, DType::Bool(BoolStore::Native)));
        assert!(B::supports_dtype(
            &device,
            DType::QFloat(CubeTensor::<CudaRuntime>::default_scheme())
        ));

        // Currently not registered in supported types
        assert!(!B::supports_dtype(&device, DType::F64));
    }

    /// A3 smoke for the FP8 quantized-QK^T attention shim
    /// (`AttentionStrategy::QuantFp8`, the `ATTN_QK_QUANT=fp8` path):
    /// exercises the burn-side prepass allocation + launches + quant kernel
    /// end-to-end on CUDA and gates parity against the decomposed fallback.
    /// Ignored by default (needs an fp8-capable GPU, arch >= 89); run with:
    ///
    /// ```text
    /// CUDA_VISIBLE_DEVICES=1 cargo test -p burn-cuda --release \
    ///   quant_fp8_attention_smoke -- --ignored
    /// ```
    #[test]
    #[ignore = "needs an fp8-capable CUDA GPU (arch >= 89); run explicitly"]
    fn quant_fp8_attention_smoke() {
        use burn_backend::TensorData;
        use burn_backend::ops::{AttentionModuleOptions, FloatTensorOps};
        use burn_cubecl::kernel::attention::{AttentionStrategy, attention};
        use burn_cubecl::ops::into_data_sync;

        type B = CubeBackend<CudaRuntime, f32, i32, u8>;
        let device: CudaDevice = Default::default();

        let (b, h, s, d) = (1usize, 2usize, 128usize, 64usize);
        let numel = b * h * s * d;
        // Deterministic pseudo-uniform in [-1, 1) (splitmix64-style), same
        // generator as cubek's quant_kernel tests.
        let uniform = |seed: u64| -> Vec<f32> {
            (0..numel)
                .map(|i| {
                    let mut z = seed
                        .wrapping_add(0x9E3779B97F4A7C15u64.wrapping_mul(i as u64 + 1))
                        .wrapping_mul(0xBF58476D1CE4E5B9);
                    z ^= z >> 31;
                    ((z >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
                })
                .collect()
        };
        let tensor = |seed: u64| {
            B::float_from_data(TensorData::new(uniform(seed), vec![b, h, s, d]), &device)
        };

        let options = AttentionModuleOptions::default();
        let quant = attention::<CudaRuntime>(
            tensor(1),
            tensor(2),
            tensor(3),
            None,
            None,
            options,
            AttentionStrategy::QuantFp8,
        )
        .expect("QuantFp8 attention should set up and launch");
        let reference = attention::<CudaRuntime>(
            tensor(1),
            tensor(2),
            tensor(3),
            None,
            None,
            options,
            AttentionStrategy::Fallback,
        )
        .expect("fallback attention");

        let quant = into_data_sync(quant).convert::<f32>().to_vec::<f32>().unwrap();
        let reference = into_data_sync(reference)
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();

        let (mut dot, mut nq, mut nr, mut max_abs) = (0f64, 0f64, 0f64, 0f64);
        for (&q, &r) in quant.iter().zip(reference.iter()) {
            assert!(q.is_finite(), "non-finite value {q} in quant output");
            let (q, r) = (f64::from(q), f64::from(r));
            dot += q * r;
            nq += q * q;
            nr += r * r;
            max_abs = max_abs.max((q - r).abs());
        }
        let cosine = dot / (nq.sqrt() * nr.sqrt());
        // Same gates as cubek's A2 kernel-parity tests (design §5).
        assert!(cosine >= 0.999, "cosine {cosine} < 0.999 vs fallback");
        assert!(max_abs <= 5e-2, "max_abs {max_abs} > 5e-2 vs fallback");
    }

    /// Stage B smoke: the burn-side `AttentionStrategy::Sparse` shim
    /// (predictor prepass → kept-count readback → sparse kernel launch)
    /// end-to-end on CUDA, gated against the decomposed fallback. Uniform
    /// random input has ~no exploitable sparsity, so `min_sparsity: 0.0`
    /// forces the SPARSE kernel branch (kept ≈ all blocks — still exercises
    /// the jump_to loop), and a second call with `min_sparsity: 1.1` forces
    /// the host dense-fallback branch. Run with:
    ///
    /// ```text
    /// CUDA_VISIBLE_DEVICES=1 cargo test -p burn-cuda --release \
    ///   sparse_attention_smoke -- --ignored
    /// ```
    #[test]
    #[ignore = "needs a CUDA GPU; run explicitly"]
    fn sparse_attention_smoke() {
        use burn_backend::TensorData;
        use burn_backend::ops::{AttentionModuleOptions, FloatTensorOps};
        use burn_cubecl::kernel::attention::{AttentionStrategy, attention};
        use burn_cubecl::ops::into_data_sync;

        // f16 element type: the sparse shim launches the dense blackbox
        // route, whose query global/tile types must match (f16). This is
        // the production shape — FLUX casts attention Q/K/V to f16.
        type B = CubeBackend<CudaRuntime, burn_backend::f16, i32, u8>;
        let device: CudaDevice = Default::default();

        let (b, h, s, d) = (1usize, 2usize, 128usize, 64usize);
        let numel = b * h * s * d;
        let uniform = |seed: u64| -> Vec<f32> {
            (0..numel)
                .map(|i| {
                    let mut z = seed
                        .wrapping_add(0x9E3779B97F4A7C15u64.wrapping_mul(i as u64 + 1))
                        .wrapping_mul(0xBF58476D1CE4E5B9);
                    z ^= z >> 31;
                    ((z >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
                })
                .collect()
        };
        let tensor = |seed: u64| {
            B::float_from_data(
                TensorData::new(uniform(seed), vec![b, h, s, d])
                    .convert::<burn_backend::f16>(),
                &device,
            )
        };

        let options = AttentionModuleOptions::default();
        let reference = attention::<CudaRuntime>(
            tensor(1),
            tensor(2),
            tensor(3),
            None,
            None,
            options,
            AttentionStrategy::Fallback,
        )
        .expect("fallback attention");
        let reference = into_data_sync(reference)
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();

        let gate = |label: &str, min_sparsity: f32| {
            let sparse = attention::<CudaRuntime>(
                tensor(1),
                tensor(2),
                tensor(3),
                None,
                None,
                options,
                AttentionStrategy::Sparse {
                    tau: 0.99,
                    min_sparsity,
                    quant: false,
                },
            )
            .unwrap_or_else(|e| panic!("{label}: sparse attention should launch: {e:?}"));
            let sparse = into_data_sync(sparse)
                .convert::<f32>()
                .to_vec::<f32>()
                .unwrap();

            let (mut dot, mut ns, mut nr, mut max_abs) = (0f64, 0f64, 0f64, 0f64);
            for (&a, &r) in sparse.iter().zip(reference.iter()) {
                assert!(a.is_finite(), "{label}: non-finite value {a} in output");
                let (a, r) = (f64::from(a), f64::from(r));
                dot += a * r;
                ns += a * a;
                nr += r * r;
                max_abs = max_abs.max((a - r).abs());
            }
            let cosine = dot / (ns.sqrt() * nr.sqrt());
            assert!(
                cosine >= 0.998,
                "{label}: cosine {cosine} < 0.998 vs fallback"
            );
            assert!(
                max_abs <= 5e-2,
                "{label}: max_abs {max_abs} > 5e-2 vs fallback"
            );
        };

        gate("sparse kernel branch", 0.0);
        gate("dense fallback branch", 1.1);
    }
}
