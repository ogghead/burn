mod base;
/// Env-gated matmul shape census (`BURN_MATMUL_SHAPE_LOG=1`), maps nsys matmul
/// kernels back to problem shapes + dtypes.
pub mod shape_log;
mod tune;

/// Contains utilities for matmul operation
pub mod utils;

pub use base::*;
#[cfg(feature = "autotune")]
pub use tune::*;
pub use utils::*;
