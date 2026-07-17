//! Tests for the [StreamOptimizer] and the underlying [BlocksOptimizer].
//!
//! These tests focus on whether the optimizer reorders operations across
//! independent blocks to find better fusion opportunities. Two operations end
//! up in distinct blocks when their tensor ids are disjoint — meaning they
//! have no data dependencies and can safely be reordered.

use std::sync::Arc;

use super::StreamOptimizer;
use crate::stream::execution::tests::{TestOptimization, TestOptimizationBuilder};
use crate::{OperationFuser, search::BlockOptimization, stream::store::ExecutionStrategy};
use burn_backend::{DType, Shape};
use burn_ir::{BinaryOpIr, NumericOperationIr, OperationIr, TensorId, TensorIr, TensorStatus};

fn tensor(id: u64) -> TensorIr {
    TensorIr {
        id: TensorId::new(id),
        shape: Shape::new([32, 32]),
        status: TensorStatus::ReadOnly,
        dtype: DType::F32,
    }
}

fn add(lhs: u64, rhs: u64, out: u64) -> OperationIr {
    OperationIr::NumericFloat(
        DType::F32,
        NumericOperationIr::Add(BinaryOpIr {
            lhs: tensor(lhs),
            rhs: tensor(rhs),
            out: tensor(out),
        }),
    )
}

/// Build a [StreamOptimizer] with one [TestOptimizationBuilder] per pattern.
fn optimizer(patterns: Vec<Vec<OperationIr>>) -> StreamOptimizer<TestOptimization> {
    let builders = patterns
        .into_iter()
        .enumerate()
        .map(|(i, p)| {
            Box::new(TestOptimizationBuilder::new(i, p))
                as Box<dyn OperationFuser<TestOptimization>>
        })
        .collect();
    StreamOptimizer::new(builders)
}

/// Register every op in the stream and return the resulting [BlockOptimization].
fn run(
    ops: &[OperationIr],
    patterns: Vec<Vec<OperationIr>>,
) -> BlockOptimization<TestOptimization> {
    let mut opt = optimizer(patterns);
    for op in ops {
        opt.register(op);
    }
    opt.optimize(ops)
}

// --- Strategy constructors (make expected values short and readable) -------

fn optimization(
    builder_id: usize,
    size: usize,
    ordering: Vec<usize>,
    score: u64,
) -> ExecutionStrategy<TestOptimization> {
    ExecutionStrategy::Optimization {
        opt: TestOptimization::new(builder_id, size),
        ordering: Arc::new(ordering),
        score,
    }
}

fn operations(ordering: Vec<usize>) -> ExecutionStrategy<TestOptimization> {
    ExecutionStrategy::Operations {
        ordering: Arc::new(ordering),
    }
}

fn composed(
    parts: Vec<ExecutionStrategy<TestOptimization>>,
) -> ExecutionStrategy<TestOptimization> {
    ExecutionStrategy::Composed(parts.into_iter().map(Box::new).collect())
}

// ---------------------------------------------------------------------------
// Baseline: a single block with a builder whose pattern matches the stream.
// ---------------------------------------------------------------------------

#[test]
fn single_block_fuses_full_stream() {
    let a1 = add(100, 101, 102);
    let a2 = add(102, 103, 104);
    let ops = vec![a1.clone(), a2.clone()];

    let res = run(&ops, vec![vec![a1, a2]]);

    assert_eq!(res.ordering, vec![0, 1]);
    assert_eq!(res.strategy, optimization(0, 2, vec![0, 1], 2));
}

// ---------------------------------------------------------------------------
// Two independent blocks, registered contiguously (all A then all B).
// Both should fuse — no reordering required.
// ---------------------------------------------------------------------------

#[test]
fn contiguous_independent_blocks_fuse_both() {
    let a1 = add(100, 101, 102);
    let a2 = add(102, 103, 104);
    let b1 = add(200, 201, 202);
    let b2 = add(202, 203, 204);
    let ops = vec![a1.clone(), a2.clone(), b1.clone(), b2.clone()];

    let res = run(
        &ops,
        vec![vec![a1.clone(), a2.clone()], vec![b1.clone(), b2.clone()]],
    );

    assert_eq!(res.ordering, vec![0, 1, 2, 3]);
    assert_eq!(
        res.strategy,
        composed(vec![
            optimization(0, 2, vec![0, 1], 2),
            optimization(1, 2, vec![2, 3], 2),
        ]),
    );
}

// ---------------------------------------------------------------------------
// Two independent blocks *interleaved* in the stream.
//
// Block A has no shared tensors with block B, so the two blocks have no data
// dependency — reordering them is safe. Both blocks fuse, each block's ops
// grouped contiguously.
// ---------------------------------------------------------------------------

#[test]
fn interleaved_independent_blocks_fuse_both() {
    let a1 = add(100, 101, 102);
    let a2 = add(102, 103, 104);
    let b1 = add(200, 201, 202);
    let b2 = add(202, 203, 204);
    // Stream is A1, B1, A2, B2 — each pair belongs to its own independent block.
    let ops = vec![a1.clone(), b1.clone(), a2.clone(), b2.clone()];

    let res = run(
        &ops,
        vec![vec![a1.clone(), a2.clone()], vec![b1.clone(), b2.clone()]],
    );

    assert_eq!(res.ordering, vec![0, 2, 1, 3]);
    assert_eq!(
        res.strategy,
        composed(vec![
            optimization(0, 2, vec![0, 2], 2),
            optimization(1, 2, vec![1, 3], 2),
        ]),
    );
}

// ---------------------------------------------------------------------------
// Interleaved independent blocks with only ONE builder that matches the
// *combined*, reordered stream [A1, A2, B1, B2]. The merging pass combines
// the two blocks so the builder sees its full pattern and fuses everything
// into one optimization.
// ---------------------------------------------------------------------------

#[test]
fn interleaved_blocks_with_merged_pattern_fuses_full() {
    let a1 = add(100, 101, 102);
    let a2 = add(102, 103, 104);
    let b1 = add(200, 201, 202);
    let b2 = add(202, 203, 204);
    let ops = vec![a1.clone(), b1.clone(), a2.clone(), b2.clone()];

    // Single builder whose pattern is the combined, reordered stream.
    let res = run(
        &ops,
        vec![vec![a1.clone(), a2.clone(), b1.clone(), b2.clone()]],
    );

    assert_eq!(res.ordering, vec![0, 2, 1, 3]);
    assert_eq!(res.strategy, optimization(0, 4, vec![0, 2, 1, 3], 4));
}

// ---------------------------------------------------------------------------
// Interleaved independent blocks, only one has a matching builder. Block A
// fuses; block B's ops survive as un-fused Operations in the composed
// strategy.
// ---------------------------------------------------------------------------

#[test]
fn interleaved_blocks_single_builder_preserves_other_ops() {
    let a1 = add(100, 101, 102);
    let a2 = add(102, 103, 104);
    let b1 = add(200, 201, 202);
    let b2 = add(202, 203, 204);
    let ops = vec![a1.clone(), b1.clone(), a2.clone(), b2.clone()];

    // Only a builder for block A — B's ops survive as Operations.
    let res = run(&ops, vec![vec![a1.clone(), a2.clone()]]);

    assert_eq!(res.ordering, vec![0, 2, 1, 3]);
    assert_eq!(
        res.strategy,
        composed(vec![
            optimization(0, 2, vec![0, 2], 2),
            operations(vec![1, 3]),
        ]),
    );
}

// ---------------------------------------------------------------------------
// Hole-fill correctness (the WithHoles branch of StreamOptimizer::optimize).
//
// When the optimizer resolves interior "holes" across multiple iterations of
// its fill loop, the accumulated global ordering must stay a valid permutation
// of the resolved positions (in range, no duplicates). These tests guard that
// invariant both end-to-end (through the block search) and directly at the
// hole-removal remap (`remove_resolved_holes`), which is where the historical
// bug lived: it dropped a positional PREFIX of `holes` instead of the actually
// resolved local indices.
// ---------------------------------------------------------------------------

use super::stream::{is_valid_ordering, remove_resolved_holes};

/// Collect every operation index referenced by a strategy tree, flattening
/// through `Composed`. Used to assert the fully-resolved ordering is a valid,
/// duplicate-free permutation.
fn flatten_ordering(strategy: &ExecutionStrategy<TestOptimization>, out: &mut Vec<usize>) {
    match strategy {
        ExecutionStrategy::Optimization { ordering, .. }
        | ExecutionStrategy::Operations { ordering } => out.extend(ordering.iter().copied()),
        ExecutionStrategy::Composed(parts) => {
            for part in parts {
                flatten_ordering(part, out);
            }
        }
    }
}

/// End-to-end: a stream whose interior holes force the hole-fill loop to run
/// more than one iteration. Whatever ordering comes back — from the top-level
/// `ordering` field and from the flattened strategy tree — must be a valid
/// permutation of the resolved positions (in range, no duplicates).
#[test]
fn multi_iteration_hole_fill_yields_valid_permutation() {
    let a1 = add(100, 101, 102);
    let a2 = add(102, 103, 104);
    let a3 = add(104, 105, 106);
    let a4 = add(106, 107, 108);
    let b1 = add(200, 201, 202);
    let ops = vec![a1.clone(), a2.clone(), a3.clone(), a4.clone(), b1.clone()];

    // builder0 fuses the A-prefix (a1,a2); builder1 fuses b1; builder2 fuses
    // only a3. a3/a4 become interior holes (b1 resolves after them); the
    // hole-fill sub-search resolves a3 then leaves a4 for a second iteration.
    let res = run(
        &ops,
        vec![
            vec![a1.clone(), a2.clone()],
            vec![b1.clone()],
            vec![a3.clone()],
        ],
    );

    let n = ops.len();

    // The BlockOptimization's own ordering must be a valid permutation.
    assert!(
        is_valid_ordering(&res.ordering, n),
        "top-level ordering not a valid permutation: {:?}",
        res.ordering,
    );

    // The flattened strategy ordering must match it exactly as a set (sorted),
    // covering the full resolved range with no duplicates.
    let mut flat = Vec::new();
    flatten_ordering(&res.strategy, &mut flat);
    assert!(
        is_valid_ordering(&flat, n),
        "flattened strategy ordering not a valid permutation: {flat:?}",
    );

    let mut sorted = flat.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..n).collect::<Vec<_>>(), "must cover 0..n exactly");

    let mut top_sorted = res.ordering.clone();
    top_sorted.sort_unstable();
    assert_eq!(
        top_sorted, sorted,
        "top-level and strategy orderings must agree",
    );
}

/// Direct proof that the fix matters: `remove_resolved_holes` must drop the
/// holes the sub-search actually resolved (by local index), NOT a positional
/// prefix. This constructs the exact case the historical `holes.drain(0..len)`
/// bug got wrong — a NON-prefix resolved set — and shows the remaining holes
/// (and thus the local->global remap) stay correct only with the fix.
#[test]
fn remove_resolved_holes_drops_exact_local_indices_not_prefix() {
    // Global stream positions that are holes, in ascending order. These double
    // as the local->global mapping used by `map_ordering`.
    let holes = vec![3usize, 7, 8, 11];

    // The sub-search resolved local indices {1, 3} (a NON-prefix subset:
    // global positions 7 and 11), leaving locals {0, 2} (globals 3, 8) as the
    // tail for the next iteration.
    let resolved_local = vec![3usize, 1];

    let mut fixed = holes.clone();
    remove_resolved_holes(&mut fixed, &resolved_local);
    assert_eq!(
        fixed,
        vec![3, 8],
        "fix must leave exactly the UNresolved holes (globals 3 and 8)",
    );

    // The buggy positional-prefix drain would instead drop holes[0..2] = the
    // FIRST two entries (globals 3 and 7) — dropping global 3 (never resolved)
    // and keeping global 11 (already resolved). That leaves the WRONG tail and
    // corrupts the remap. Assert the fix does NOT match that wrong behavior.
    let mut buggy = holes.clone();
    buggy.drain(0..resolved_local.len());
    assert_eq!(buggy, vec![8, 11], "sanity: buggy prefix-drain leaves 8,11");
    assert_ne!(
        fixed, buggy,
        "fix must diverge from the buggy positional-prefix drain",
    );
}

// ---------------------------------------------------------------------------
// Within a single block, operations are registered in stream order. A builder
// whose pattern would only match a *reordered* version of the stream cannot
// fuse. This documents that reordering does NOT happen inside a block (unlike
// across blocks).
// ---------------------------------------------------------------------------

#[test]
fn within_block_order_is_not_reordered() {
    let a1 = add(100, 101, 102);
    let a2 = add(102, 103, 104);
    let ops = vec![a1.clone(), a2.clone()];

    // Builder wants [a2, a1] — only reachable if we reorder within the block.
    let res = run(&ops, vec![vec![a2.clone(), a1.clone()]]);

    assert_eq!(res.ordering, vec![0, 1]);
    assert_eq!(res.strategy, operations(vec![0, 1]));
}

// ---------------------------------------------------------------------------
// Lazy-optimization semantics.
//
// `still_optimizing()` is a "keep feeding me" signal for the caller: it
// returns true if at least one builder might still match a longer pattern.
// `optimize()` is the commit point — it picks the best *ready* fusion, even
// if other builders are still open (this is what happens on a forced sync).
// ---------------------------------------------------------------------------

/// A builder whose pattern is longer than the stream so far. The builder's
/// actual ops match the pattern prefix, so the builder stays Open and
/// `still_optimizing` keeps signaling "feed me more". A forced `optimize`
/// (simulating a sync) falls back to unfused operations because no builder
/// has reached `ready`.
#[test]
fn pattern_longer_than_stream_stays_optimizing() {
    let a1 = add(100, 101, 102);
    let a2 = add(102, 103, 104);
    let a3 = add(104, 105, 106);

    let mut opt = optimizer(vec![vec![a1.clone(), a2.clone(), a3.clone()]]);
    opt.register(&a1);
    opt.register(&a2);

    assert!(opt.still_optimizing(), "builder still waiting for a3");

    let ops = vec![a1, a2];
    let res = opt.optimize(&ops);
    assert_eq!(res.ordering, vec![0, 1]);
    assert_eq!(res.strategy, operations(vec![0, 1]));
}

/// One builder is ready at two ops, another is still open waiting for a
/// third. `still_optimizing` stays true so the caller has the option to keep
/// feeding and pick up the longer fusion. A forced commit takes the ready
/// one.
#[test]
fn ready_and_open_coexist_keeps_optimizing() {
    let a1 = add(100, 101, 102);
    let a2 = add(102, 103, 104);
    let a3 = add(104, 105, 106);

    let mut opt = optimizer(vec![
        vec![a1.clone(), a2.clone()],             // ready after 2 ops
        vec![a1.clone(), a2.clone(), a3.clone()], // still open after 2 ops
    ]);
    opt.register(&a1);
    opt.register(&a2);

    assert!(opt.still_optimizing(), "builder 2 still waiting for a3");

    let ops = vec![a1, a2];
    let res = opt.optimize(&ops);
    assert_eq!(res.ordering, vec![0, 1]);
    assert_eq!(res.strategy, optimization(0, 2, vec![0, 1], 2));
}

/// Multiple builders are ready at commit time — the highest-scoring fusion
/// wins.
#[test]
fn commit_picks_highest_score_ready_builder() {
    let a1 = add(100, 101, 102);
    let a2 = add(102, 103, 104);
    let a3 = add(104, 105, 106);

    let mut opt = optimizer(vec![
        vec![a1.clone(), a2.clone()],             // score 2
        vec![a1.clone(), a2.clone(), a3.clone()], // score 3
    ]);
    opt.register(&a1);
    opt.register(&a2);
    opt.register(&a3);

    // Every builder is closed (everyone has seen at least as many ops as its
    // pattern) — there's nothing more to gain by feeding.
    assert!(!opt.still_optimizing());

    let ops = vec![a1, a2, a3];
    let res = opt.optimize(&ops);
    assert_eq!(res.ordering, vec![0, 1, 2]);
    assert_eq!(res.strategy, optimization(1, 3, vec![0, 1, 2], 3));
}

/// Every builder is closed with `ready=false` (the stream diverged from
/// every pattern). `still_optimizing` flips to false — caller should
/// commit. Forced commit returns unfused operations.
#[test]
fn all_closed_without_ready_bails_out() {
    let a1 = add(100, 101, 102);
    let a2 = add(102, 103, 104);
    // Builder wants [a1, b1]: stream diverges at the second op.
    let b1 = add(200, 201, 202);

    let mut opt = optimizer(vec![vec![a1.clone(), b1.clone()]]);
    opt.register(&a1);
    opt.register(&a2);

    assert!(!opt.still_optimizing(), "mismatch closes the builder");

    let ops = vec![a1, a2];
    let res = opt.optimize(&ops);
    assert_eq!(res.ordering, vec![0, 1]);
    assert_eq!(res.strategy, operations(vec![0, 1]));
}

/// A single pattern that spans two independent blocks, but the last op of
/// the pattern hasn't arrived yet. The merge succeeds (builder stays Open in
/// the merged block → `merge` returns true), so `still_optimizing` stays
/// true. A forced commit falls back to unfused operations in the merged
/// block's registration order.
#[test]
fn pattern_straddles_incomplete_merged_blocks() {
    let a1 = add(100, 101, 102);
    let a2 = add(102, 103, 104);
    let b1 = add(200, 201, 202);
    let b2 = add(202, 203, 204);

    // Stream is [a1, b1, a2] — builder needs b2 too.
    let mut opt = optimizer(vec![vec![a1.clone(), a2.clone(), b1.clone(), b2.clone()]]);
    opt.register(&a1);
    opt.register(&b1);
    opt.register(&a2);

    assert!(opt.still_optimizing(), "builder waiting for b2");

    let ops = vec![a1, b1, a2];
    let res = opt.optimize(&ops);
    // After merging, the merged block's registration order is
    // [a1 (pos 0), a2 (pos 2), b1 (pos 1)].
    assert_eq!(res.ordering, vec![0, 2, 1]);
    assert_eq!(res.strategy, operations(vec![0, 2, 1]));
}
