use super::blocks::BlocksOptimizer;
use crate::{
    NumOperations, OperationFuser,
    search::{
        Block, BlockOptimization, RegistrationResult,
        merging::{MergeBlocksResult, merge_blocks},
        optimization::blocks::BlocksOptimizerResult,
    },
    stream::{execution::op_kind, store::ExecutionStrategy},
};
use burn_ir::OperationIr;
use burn_std::config::{config, fusion::FusionLogLevel, log_fusion};

/// Optimize a stream of [operations](OperationIr) using a list of [builders](OptimizationBuilder).
pub struct StreamOptimizer<O> {
    builders: Vec<Box<dyn OperationFuser<O>>>,
    blocks: Vec<Block<O>>,
    length: usize,
    stopped: bool,
    max_blocks: Option<usize>,
}

impl<O: NumOperations> StreamOptimizer<O> {
    /// Create a new stream optimizer.
    pub fn new(builders: Vec<Box<dyn OperationFuser<O>>>) -> Self {
        // Too high and it may break the fusion cache always retriggering explorations.
        // BURN_FUSION_MAX_BLOCKS: override the beam-search block cap. Diagnostic for the
        // >1024px training divergence — at the default cap (5) a longer sequence produces
        // more dependency blocks and the optimizer FORCE-MERGES ops across independent
        // graphs (`merge_blocks(op, true)`), building a structurally different fused trace
        // only at the larger resolution. Raising the cap avoids the force-merge; if 1280
        // then converges, the cross-block force-merge is the culprit.
        let max_blocks = Some(
            std::env::var("BURN_FUSION_MAX_BLOCKS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or_else(|| config().fusion().beam_search.max_blocks),
        );
        Self {
            builders,
            blocks: Vec::new(),
            length: 0,
            stopped: false,
            max_blocks,
        }
    }

    /// Register a new [operation](OperationIr) in the optimizer.
    ///
    /// You can use the function [Self::still_optimizing] to know if the operations are actually
    /// being registered.
    pub fn register(&mut self, operation: &OperationIr) {
        if self.stopped {
            let length = self.length;
            log_fusion(FusionLogLevel::Full, || {
                format!(
                    "[stream] {} dropped (optimizer stopped at op {length})",
                    op_kind(operation)
                )
            });
            return;
        }

        if self.blocks.is_empty() {
            self.on_new_block(operation);
            self.length += 1;
            return;
        }

        match self.merge_blocks(operation, false) {
            MergeBlockStep::Full | MergeBlockStep::NoNeed => {}
            step @ (MergeBlockStep::Fail | MergeBlockStep::Partial) => {
                // With the given operation, blocks are no longer independent.
                let reason = match step {
                    MergeBlockStep::Fail => "merge failed",
                    MergeBlockStep::Partial => "merge partial",
                    _ => unreachable!(),
                };
                let num_blocks = self.blocks.len();
                let length = self.length;
                log_fusion(FusionLogLevel::Medium, || {
                    format!(
                        "[stream] stopped ({reason}) at op {length} ({}); {num_blocks} blocks",
                        op_kind(operation)
                    )
                });
                self.stopped = true;
                return;
            }
        }

        if let Some(max_blocks) = self.max_blocks {
            if self.register_max_block(operation, max_blocks) {
                self.length += 1;
            } else {
                let length = self.length;
                log_fusion(FusionLogLevel::Medium, || {
                    format!(
                        "[stream] stopped (max_blocks={max_blocks} reached) at op {length} ({})",
                        op_kind(operation)
                    )
                });
                self.stopped = true;
            }
            return;
        }

        let added_count = self.register_inner(operation, false);
        if added_count == 0 {
            self.on_new_block(operation);
        } else {
            self.log_accepted(operation, added_count);
        }

        self.length += 1;
    }

    /// Optimize the current stream on the given [operations](OperationIr).
    ///
    /// # Notes
    ///
    /// The operations provided are the same as the ones used in the [register](Self::register)
    /// method, this simply remove the need for the current type to also keep track of the list of
    /// operations.
    pub fn optimize(&self, operations: &[OperationIr]) -> BlockOptimization<O> {
        let result = BlocksOptimizer::new(self.blocks.clone()).optimize();

        match result {
            BlocksOptimizerResult::Full(block_optimization) => block_optimization,
            BlocksOptimizerResult::WithHoles {
                mut strategies,
                mut ordering,
                mut holes,
            } => {
                loop {
                    let mut search = self.new_empty_search();

                    let mut operations_holes = Vec::with_capacity(holes.len());

                    for index in holes.iter() {
                        let op = &operations[*index];
                        operations_holes.push(op.clone());
                        search.register(op);
                    }

                    let mut optimization_of_holes = search.optimize(&operations_holes);

                    // Snapshot the LOCAL hole-indices this sub-optimization actually
                    // resolved, BEFORE map_ordering rebases them into global stream
                    // indices. `holes` is the local->global map for this pass, so its
                    // entries must stay fixed while it is used as the mapping.
                    let resolved_local: Vec<usize> = optimization_of_holes.ordering.clone();

                    optimization_of_holes.map_ordering(&holes);

                    strategies.push(Box::new(optimization_of_holes.strategy));
                    ordering.append(&mut optimization_of_holes.ordering);

                    // Remove exactly the holes that were resolved (by local index),
                    // NOT a positional prefix. The sub-search can resolve a non-prefix
                    // subset and leave a trailing tail for the next iteration; the old
                    // `holes.drain(0..ordering.len())` then dropped the WRONG holes,
                    // corrupting the local->global remap across iterations — producing
                    // out-of-range indices (ordering.rs:65 OOB crash) or wrong-but-in-
                    // bounds indices (silent wrong-handle → the >1024px training
                    // divergence).
                    remove_resolved_holes(&mut holes, &resolved_local);

                    if holes.is_empty() {
                        break;
                    }
                }

                // The accumulated global `ordering` must be a valid permutation
                // of the positions it resolved: every index in range and no
                // duplicates. A violation here means the local->global hole remap
                // was corrupted (e.g. the wrong holes were dropped between
                // iterations) — which downstream turns into an index-OOB crash at
                // `stream/execution/ordering.rs` or a silently wrong tensor handle.
                // `debug_assert!` keeps the release hot path clean.
                debug_assert!(
                    is_valid_ordering(&ordering, operations.len()),
                    "hole-filled ordering is not a valid permutation: {ordering:?} (num_ops={})",
                    operations.len(),
                );

                BlockOptimization::new(ExecutionStrategy::Composed(strategies), ordering)
            }
        }
    }

    /// Reset the state of the optimizer.
    pub fn reset(&mut self) {
        self.builders.iter_mut().for_each(|b| b.reset());
        self.length = 0;
        self.blocks.clear();
        self.stopped = false;
    }

    /// Returns if some optimizations are still possible within the stream.
    pub fn still_optimizing(&self) -> bool {
        if self.stopped {
            return false;
        }
        if self.blocks.is_empty() {
            return true;
        }

        let mut num_stopped = 0;

        for block in self.blocks.iter() {
            if !block.still_optimizing() {
                num_stopped += 1
            }
        }

        num_stopped < self.blocks.len()
    }

    fn register_max_block(&mut self, operation: &OperationIr, max_blocks: usize) -> bool {
        if max_blocks == 1 {
            // Register in the single block with a force.
            self.register_inner(operation, true);
            return true;
        }
        let added_count = self.register_inner(operation, false);

        if added_count > 0 {
            self.log_accepted(operation, added_count);
            return true;
        }

        if added_count == 0 && self.blocks.len() < max_blocks {
            self.on_new_block(operation);
            return true;
        }

        self.merge_blocks(operation, true);

        if self.blocks.len() >= max_blocks {
            self.stopped = true;
            return false;
        }

        let added_count = self.register_inner(operation, false);

        if added_count == 0 {
            self.on_new_block(operation);
        } else {
            self.log_accepted(operation, added_count);
        }

        true
    }

    fn log_accepted(&self, operation: &OperationIr, added_count: usize) {
        let length = self.length;
        let num_blocks = self.blocks.len();
        log_fusion(FusionLogLevel::Full, || {
            format!(
                "[stream] op {length} {} → accepted in {added_count}/{num_blocks} block(s)",
                op_kind(operation)
            )
        });
    }

    fn register_inner(&mut self, operation: &OperationIr, force: bool) -> usize {
        let mut added_count = 0;
        for block in self.blocks.iter_mut() {
            match block.register(operation, self.length, force) {
                RegistrationResult::Accepted => {
                    added_count += 1;
                }
                RegistrationResult::NotPartOfTheGraph => {}
            }
        }
        added_count
    }

    fn new_empty_search(&self) -> Self {
        Self::new(
            self.builders
                .iter()
                .map(|b| {
                    let mut b = b.clone_dyn();
                    b.reset();
                    b
                })
                .collect(),
        )
    }

    fn merge_blocks(&mut self, operation: &OperationIr, all: bool) -> MergeBlockStep {
        let nodes = operation.nodes();
        let mut block_merges = Vec::new();

        for (i, block) in self.blocks.iter().enumerate() {
            if all || block.contains_tensors(&nodes) {
                block_merges.push(i);
            }
        }

        if block_merges.len() <= 1 {
            return MergeBlockStep::NoNeed;
        }

        let blocks_to_merge = self
            .blocks
            .iter()
            .enumerate()
            .filter_map(|(i, g)| match block_merges.contains(&i) {
                true => Some(g),
                false => None,
            })
            .collect::<Vec<_>>();

        let merged = merge_blocks(&blocks_to_merge, false);

        let mut clear_blocks = || {
            let mut indices = block_merges.to_vec();
            indices.sort();

            for g in indices.into_iter().rev() {
                self.blocks.remove(g);
            }
        };

        match merged {
            MergeBlocksResult::Full(block) => {
                clear_blocks();
                self.blocks.push(block);
                Block::sort(&mut self.blocks);
                MergeBlockStep::Full
            }
            MergeBlocksResult::Partial {
                mut merged,
                mut failed,
            } => {
                clear_blocks();
                self.blocks.append(&mut merged);
                self.blocks.append(&mut failed);
                Block::sort(&mut self.blocks);
                MergeBlockStep::Partial
            }
            MergeBlocksResult::Fail => MergeBlockStep::Fail,
        }
    }

    fn on_new_block(&mut self, operation: &OperationIr) {
        let mut block = Block::new(&self.builders);
        block.register(operation, self.length, true);
        self.blocks.push(block);

        let length = self.length;
        let num_blocks = self.blocks.len();
        log_fusion(FusionLogLevel::Full, || {
            format!(
                "[stream] op {length} {} → new block (total: {num_blocks})",
                op_kind(operation)
            )
        });
    }
}

/// Drop the holes that a hole-fill sub-optimization resolved, identified by
/// their LOCAL index into `holes` (i.e. positions into `holes` itself, not
/// global stream positions).
///
/// The sub-search can resolve a NON-prefix subset of the holes and leave a
/// trailing tail for the next iteration, so we must remove exactly the resolved
/// local indices — not a positional prefix (`holes.drain(0..n)`), which would
/// drop the wrong entries and corrupt the local->global remap. Removing in
/// descending order keeps the not-yet-removed indices valid as we go.
pub(super) fn remove_resolved_holes(holes: &mut Vec<usize>, resolved_local: &[usize]) {
    let mut resolved: Vec<usize> = resolved_local.to_vec();
    resolved.sort_unstable();
    for local in resolved.into_iter().rev() {
        holes.remove(local);
    }
}

/// Returns whether `ordering` is a valid permutation of the positions it
/// resolved: every index is `< num_ops` and no index appears twice.
///
/// This is the invariant the hole-fill loop in [`StreamOptimizer::optimize`]
/// must uphold. It intentionally does NOT require `ordering` to cover *all* of
/// `0..num_ops` — a trailing tail of unresolved positions is legitimately left
/// for the next processing round — only that whatever it does contain is a
/// duplicate-free, in-range set.
// Always compiled: the `debug_assert!` in `optimize()` name-resolves this in every
// profile (its `if cfg!(debug_assertions)` body is still type-checked in release),
// so gating it on `debug_assertions` broke the release build. `allow(dead_code)`
// because in release the only call site sits in the assert's compiled-out branch.
#[allow(dead_code)]
pub(super) fn is_valid_ordering(ordering: &[usize], num_ops: usize) -> bool {
    let mut seen = vec![false; num_ops];
    for &idx in ordering {
        if idx >= num_ops || seen[idx] {
            return false;
        }
        seen[idx] = true;
    }
    true
}

enum MergeBlockStep {
    Full,
    Partial,
    Fail,
    NoNeed,
}
