//! Env-gated matmul shape census (diagnostic).
//!
//! When `BURN_MATMUL_SHAPE_LOG=1`, every matmul launch records its
//! `(m, n, k, lhs_dtype, rhs_dtype, out_dtype, path)` and, every
//! `BURN_MATMUL_SHAPE_LOG_EVERY` launches (default 10000), dumps one line per
//! unique key with its cumulative launch count to stderr. This lets an nsys
//! `cuda_gpu_kern_sum` (whose matmul kernels only carry generic names like
//! `matmul_entry_lhs_f32_size_4`) be mapped back to the actual problem shapes +
//! dtypes without guessing. Off by default and a single cached env read per
//! launch when off — no measurable overhead on the hot path.

#[cfg(feature = "std")]
mod imp {
    use burn_backend::DType;
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    fn enabled() -> bool {
        static E: OnceLock<bool> = OnceLock::new();
        *E.get_or_init(|| {
            matches!(
                std::env::var("BURN_MATMUL_SHAPE_LOG").as_deref(),
                Ok("1") | Ok("true")
            )
        })
    }

    fn dump_every() -> u64 {
        static N: OnceLock<u64> = OnceLock::new();
        *N.get_or_init(|| {
            std::env::var("BURN_MATMUL_SHAPE_LOG_EVERY")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .filter(|n| *n > 0)
                .unwrap_or(10_000)
        })
    }

    type Key = (usize, usize, usize, DType, DType, DType, &'static str);

    fn table() -> &'static Mutex<(HashMap<Key, u64>, u64)> {
        static T: OnceLock<Mutex<(HashMap<Key, u64>, u64)>> = OnceLock::new();
        T.get_or_init(|| Mutex::new((HashMap::new(), 0)))
    }

    /// Record one matmul launch. `path` distinguishes the launch route
    /// (e.g. "plain", "fused"). See module docs.
    pub fn record(
        m: usize,
        n: usize,
        k: usize,
        lhs: DType,
        rhs: DType,
        out: DType,
        path: &'static str,
    ) {
        if !enabled() {
            return;
        }
        let mut guard = match table().lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        let (map, total) = &mut *guard;
        *map.entry((m, n, k, lhs, rhs, out, path)).or_insert(0) += 1;
        *total += 1;
        if *total % dump_every() == 0 {
            let mut rows: Vec<(&Key, &u64)> = map.iter().collect();
            rows.sort_by(|a, b| b.1.cmp(a.1));
            eprintln!(
                "[MATMUL_SHAPE_LOG] {} unique shapes after {} launches:",
                rows.len(),
                *total
            );
            for ((m, n, k, lhs, rhs, out, path), count) in rows {
                eprintln!(
                    "[MATMUL_SHAPE_LOG]   m={m} n={n} k={k} lhs={lhs:?} rhs={rhs:?} out={out:?} path={path} count={count}"
                );
            }
        }
    }
}

#[cfg(feature = "std")]
pub use imp::record;

/// No-op without `std` (no env / stderr).
#[cfg(not(feature = "std"))]
pub fn record(
    _m: usize,
    _n: usize,
    _k: usize,
    _lhs: burn_backend::DType,
    _rhs: burn_backend::DType,
    _out: burn_backend::DType,
    _path: &'static str,
) {
}
