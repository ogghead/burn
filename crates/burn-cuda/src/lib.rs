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

    /// Regression for the in-app mixed-dtype matmul routing (Task A). Production
    /// base-linear matmuls are bf16 weight × f32 activation; through the FUSION
    /// backend they must run on tensor cores (the cubek adjust_dtypes mixed→TF32
    /// branch), not the ~14-24ms single-warp unit kernel. This drives the exact
    /// path (Fusion<CubeBackend<Cuda>>) at the LoRA down-proj shape and times a
    /// matmul+bias (epilogue triggers the FusedMatmul), for f32×f32, bf16×bf16,
    /// and the mixed f32×bf16 case. Run:
    ///
    /// ```text
    /// CUDA_VISIBLE_DEVICES=0 cargo test -p burn-cuda --release \
    ///   fused_mixed_matmul_routing -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "needs a CUDA GPU; run explicitly on GPU 0"]
    fn fused_mixed_matmul_routing() {
        use burn_backend::TensorData;
        use burn_backend::ops::FloatTensorOps;

        type B = Cuda; // Fusion<CubeBackend<CudaRuntime, f32, i32, u8>>
        let device: CudaDevice = Default::default();

        let (m, k, n) = (6848usize, 3072usize, 32usize);
        let fill = |len: usize, v: f32| vec![v; len];

        // Autotune caches its (possibly pre-fix "unit") decision per key on disk;
        // a stale cache masks the fix. Best-effort clear so the test is
        // deterministic. (The cwd-relative path varies by invocation, so try both.)
        let _ = std::fs::remove_dir_all("target/autotune");
        let _ = std::fs::remove_dir_all("crates/burn-cuda/target/autotune");

        let time_case = |label: &str, lhs_dt: DType, rhs_dt: DType| -> f64 {
            let lhs_data = TensorData::new(fill(m * k, 0.02), vec![m, k]).convert_dtype(lhs_dt);
            let rhs_data = TensorData::new(fill(k * n, 0.02), vec![k, n]).convert_dtype(rhs_dt);
            // Upload ONCE (host→device) so the loop measures the matmul, not PCIe.
            let lhs = B::float_from_data(lhs_data, &device);
            let rhs = B::float_from_data(rhs_data, &device);

            let run_once = || {
                // Clone = cheap handle clone (no re-upload); matmul consumes them.
                let out = B::float_matmul(lhs.clone(), rhs.clone());
                let out = B::float_add_scalar(out, 1.0.into()); // epilogue → FusedMatmul
                // Force execution (blocks). Output is small [m,n] so readback is cheap.
                let _ = cubecl::future::block_on(B::float_into_data(out));
            };

            for _ in 0..5 {
                run_once(); // warmup + autotune
            }
            let iters = 30;
            let start = std::time::Instant::now();
            for _ in 0..iters {
                run_once();
            }
            let ms = start.elapsed().as_secs_f64() * 1e3 / iters as f64;
            println!("FUSED_MM {label} [{m}x{n}x{k}]: {ms:.3} ms/call");
            ms
        };

        // Same-dtype baselines (already accelerated pre-fix).
        time_case("f32xf32", DType::F32, DType::F32);
        time_case("bf16xbf16", DType::BF16, DType::BF16);
        time_case("f16xf16", DType::F16, DType::F16);

        // MIXED float pairs — the regression. Pre-fix these fell to the unit
        // kernel (4–18ms); with the cubek adjust_dtypes mixed→TF32 branch they
        // must run accelerated (~0.4–0.6ms incl. readback/trace overhead).
        // 2ms bound leaves generous margin over accelerated while still catching
        // a unit-kernel regression.
        let mixed = [
            ("f32xbf16", DType::F32, DType::BF16),
            ("bf16xf32", DType::BF16, DType::F32),
            ("f32xf16", DType::F32, DType::F16),
            ("f16xf32", DType::F16, DType::F32),
            ("f16xbf16", DType::F16, DType::BF16),
            ("bf16xf16", DType::BF16, DType::F16),
        ];
        for (label, lhs, rhs) in mixed {
            let ms = time_case(label, lhs, rhs);
            assert!(
                ms < 2.0,
                "mixed {label} matmul ran {ms:.3} ms/call (>= 2ms) — fell to the unit \
                 kernel; the cubek adjust_dtypes mixed→TF32 branch isn't engaging. \
                 If this regressed after a clean build, clear target/autotune (stale \
                 cache pins the pre-fix unit decision)."
            );
        }
    }

    /// Repro for the per-step fused-kernel RECOMPILATION storm. A fixed fusible
    /// graph (matmul → mul_scalar → add) run many times with a DIFFERENT scalar
    /// each iteration (mimics sigma changing every training step). If scalars are
    /// runtime args (correct), the fused kernel compiles once (iter 0) and every
    /// later iter is fast. If a step-varying value leaks into the kernel identity,
    /// every iter recompiles (NVRTC ~50–200ms) and stays slow — the training
    /// step-time storm.
    ///
    /// ```text
    /// CUDA_VISIBLE_DEVICES=0 cargo test -p burn-cuda --release \
    ///   fused_scalar_recompile_storm -- --ignored --nocapture
    /// # to see the compiles: prefix CUBECL_DEBUG_LOG=stderr and grep -c START_KERNEL_COMPILATION
    /// ```
    #[test]
    #[ignore = "needs a CUDA GPU; run explicitly on GPU 0"]
    fn fused_scalar_recompile_storm() {
        use burn_backend::Scalar;
        use burn_backend::TensorData;
        use burn_backend::ops::FloatTensorOps;

        type B = Cuda;
        let device: CudaDevice = Default::default();
        let (m, k, n) = (512usize, 512usize, 512usize);

        let a = B::float_from_data(TensorData::new(vec![0.01f32; m * k], vec![m, k]), &device);
        let w = B::float_from_data(TensorData::new(vec![0.01f32; k * n], vec![k, n]), &device);
        let bias = B::float_from_data(TensorData::new(vec![0.5f32; m * n], vec![m, n]), &device);

        // A different scalar every iteration (like per-step sigma).
        let step = |scalar: f64| {
            let out = B::float_matmul(a.clone(), w.clone());
            let out = B::float_mul_scalar(out, Scalar::Float(scalar));
            let out = B::float_add(out, bias.clone());
            let _ = cubecl::future::block_on(B::float_into_data(out));
        };

        // Warmup: first unique-scalar iters trigger the initial compile(s).
        step(0.001);
        step(0.002);

        // Steady state: each iter uses a NEW scalar. Should be fast (cached kernel)
        // if scalars are runtime args; slow every time if the scalar leaks into the
        // kernel identity → recompile.
        let iters = 12;
        let mut times = Vec::with_capacity(iters);
        for i in 0..iters {
            let scalar = 0.1 + (i as f64) * 0.01337; // distinct each iter
            let start = std::time::Instant::now();
            step(scalar);
            times.push(start.elapsed().as_secs_f64() * 1e3);
        }
        let median = {
            let mut t = times.clone();
            t.sort_by(|a, b| a.partial_cmp(b).unwrap());
            t[t.len() / 2]
        };
        println!(
            "RECOMPILE_STORM per-iter ms (distinct scalar each): {times:?}\n  median={median:.2} ms"
        );
        // A cached fused kernel at this size is well under 5ms/iter; a per-iter
        // NVRTC recompile is tens–hundreds of ms.
        assert!(
            median < 10.0,
            "median {median:.2} ms/iter with a distinct scalar each step — the fused \
             kernel is recompiling per step (scalar leaks into kernel identity). This \
             is the training recompilation storm."
        );
    }

    /// Fragmentation repro: identical fusable compute each iter, but a Drop of a
    /// throwaway tensor interleaved at a VARYING position (mimics autodiff temp
    /// dealloc-order nondeterminism). If a non-compute op landing mid-stream
    /// re-chops the fused blocks → the blocks re-explore/recompile every iter →
    /// the storm. Observe with the explore-trace:
    ///
    /// ```text
    /// CUDA_VISIBLE_DEVICES=0 BURN_FUSION_EXPLORE_TRACE=1 cargo test -p burn-cuda \
    ///   --release fused_drop_fragmentation -- --ignored --nocapture
    /// # healthy: total_explores plateaus after warmup. storm: climbs every iter.
    /// ```
    #[test]
    #[ignore = "needs a CUDA GPU; run explicitly on GPU 0"]
    fn fused_drop_fragmentation() {
        use burn_backend::ops::FloatTensorOps;
        use burn_backend::{Scalar, TensorData};

        type B = Cuda;
        let device: CudaDevice = Default::default();
        let (m, k, n) = (256usize, 256usize, 256usize);
        let a = B::float_from_data(TensorData::new(vec![0.01f32; m * k], vec![m, k]), &device);
        let w = B::float_from_data(TensorData::new(vec![0.01f32; k * n], vec![k, n]), &device);
        let bias = B::float_from_data(TensorData::new(vec![0.5f32; m * n], vec![m, n]), &device);

        let step = |drop_pos: usize| {
            // A throwaway temporary whose Drop is positioned by drop_pos among the
            // main fusable chain (matmul → add → mul → add).
            let mut junk = Some(B::float_add(a.clone(), a.clone()));
            let mut maybe_drop = |p: usize| {
                if p == drop_pos {
                    junk.take(); // drop → enqueues OperationIr::Drop at this position
                }
            };
            maybe_drop(0);
            let out = B::float_matmul(a.clone(), w.clone());
            maybe_drop(1);
            let out = B::float_add(out, bias.clone());
            maybe_drop(2);
            let out = B::float_mul_scalar(out, Scalar::Float(1.5));
            maybe_drop(3);
            let out = B::float_add(out, bias.clone());
            junk.take();
            let _ = cubecl::future::block_on(B::float_into_data(out));
        };

        for p in 0..4 {
            step(p); // warmup all positions
        }
        let iters = 16;
        let mut times = Vec::with_capacity(iters);
        for i in 0..iters {
            let start = std::time::Instant::now();
            step(i % 4);
            times.push(start.elapsed().as_secs_f64() * 1e3);
        }
        let median = {
            let mut t = times.clone();
            t.sort_by(|a, b| a.partial_cmp(b).unwrap());
            t[t.len() / 2]
        };
        println!("DROP_FRAG per-iter ms (varying drop pos): {times:?}\n  median={median:.2} ms");
    }
}
