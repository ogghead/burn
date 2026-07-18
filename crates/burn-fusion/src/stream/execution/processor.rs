use burn_ir::OperationIr;
use burn_std::config::{fusion::FusionLogLevel, log_fusion};

use super::{ExecutionMode, ExplorationAction, Explorer};
use crate::search::BlockOptimization;
use crate::stream::execution::{Action, Policy};
use crate::stream::store::{ExecutionPlan, ExecutionPlanId, ExecutionPlanStore, ExecutionTrigger};
use crate::{NumOperations, OperationFuser};

/// Diagnostic (env-gated by `BURN_FUSION_EXPLORE_TRACE=1`) for the per-step
/// fused-kernel recompilation storm: a plan is only compiled the first time its
/// RELATIVE op sequence is explored, so a healthy training loop should stop
/// exploring new plans after warmup. This hashes each explored relative
/// op-sequence and reports, per explore, whether that exact relative graph has
/// been explored before. If the same `relhash` keeps re-appearing as `new=false`
/// (re-explored despite an identical plan already in the store) → a plan
/// lookup/matching bug. If every explore is a distinct `new=true` hash for the
/// same logical graph → something step-varying is leaking into the RELATIVE IR
/// (relativization leak). Consecutive-step counts crack which one it is.
#[cfg(feature = "std")]
fn explore_trace(relative: &[OperationIr]) {
    use std::collections::HashSet;
    use std::hash::{Hash, Hasher};
    use std::sync::{Mutex, OnceLock};

    static ENABLED: OnceLock<bool> = OnceLock::new();
    if !*ENABLED.get_or_init(|| {
        matches!(std::env::var("BURN_FUSION_EXPLORE_TRACE").as_deref(), Ok("1") | Ok("true"))
    }) {
        return;
    }

    static SEEN: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();
    static TOTAL: OnceLock<Mutex<u64>> = OnceLock::new();

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    relative.hash(&mut hasher);
    let relhash = hasher.finish();

    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    let total = TOTAL.get_or_init(|| Mutex::new(0));
    let (is_new, distinct, count) = {
        let mut seen = seen.lock().unwrap();
        let is_new = seen.insert(relhash);
        let mut total = total.lock().unwrap();
        *total += 1;
        (is_new, seen.len(), *total)
    };
    eprintln!(
        "[EXPLORE_TRACE] relhash={relhash:016x} ops={} new={is_new} distinct_plans={distinct} total_explores={count}",
        relative.len()
    );

    // `BURN_FUSION_EXPLORE_DUMP=N`: print the full Debug of the relative op
    // sequence for the first N SINGLE-OP (`ops=1`) explorations — the ones with
    // unique-forever relhashes. Spanning >1 step, diffing two same-structure
    // dumps reveals which field leaks a globally-varying value into the relative
    // form (the `new=true`-forever cause).
    static DUMP_N: OnceLock<u64> = OnceLock::new();
    let dump_n = *DUMP_N.get_or_init(|| {
        std::env::var("BURN_FUSION_EXPLORE_DUMP")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0)
    });
    if dump_n > 0 && relative.len() == 1 {
        static DUMPED: OnceLock<Mutex<u64>> = OnceLock::new();
        let n = {
            let m = DUMPED.get_or_init(|| Mutex::new(0));
            let mut m = m.lock().unwrap();
            *m += 1;
            *m
        };
        if n <= dump_n {
            eprintln!("[EXPLORE_DUMP #{n} relhash={relhash:016x}] {relative:?}");
        }
    }
}

#[cfg(not(feature = "std"))]
fn explore_trace(_relative: &[OperationIr]) {}

/// Process a [stream segment](StreamSegment) following a [policy](Policy).
pub(crate) struct Processor<O> {
    policy: Policy<O>,
    explorer: Explorer<O>,
}

/// A part of a stream that can be executed partially using [execution plan](ExecutionPlan).
pub(crate) trait StreamSegment<O> {
    /// The operations in the segment.
    fn operations(&self) -> &[OperationIr];
    /// Execute part of the segment using the given plan id.
    fn execute(&mut self, id: ExecutionPlanId, store: &mut ExecutionPlanStore<O>);
}

impl<O: NumOperations> Processor<O> {
    /// Create a new stream processor.
    pub fn new(optimizations: Vec<Box<dyn OperationFuser<O>>>) -> Self {
        Self {
            policy: Policy::new(),
            explorer: Explorer::new(optimizations),
        }
    }

    /// Process the [stream segment](StreamSegment) with the provided [mode](ExecutionMode).
    pub fn process<Segment>(
        &mut self,
        mut segment: Segment,
        store: &mut ExecutionPlanStore<O>,
        mode: ExecutionMode,
    ) where
        Segment: StreamSegment<O>,
    {
        // We assume that we always register a new operation in lazy mode.
        if let ExecutionMode::Lazy = mode {
            self.on_new_operation(&segment, store);
        }

        loop {
            if segment.operations().is_empty() {
                break;
            }

            let action = self.policy.action(store, segment.operations(), mode);

            match action {
                Action::Explore => {
                    self.explore(&mut segment, store, mode);

                    if self.explorer.is_up_to_date() {
                        break;
                    }
                }
                Action::Defer => {
                    match mode {
                        ExecutionMode::Lazy => break,
                        ExecutionMode::Sync => panic!("Can't defer while sync"),
                    };
                }
                Action::Execute(id) => {
                    let mode_dbg = match mode {
                        ExecutionMode::Lazy => "lazy",
                        ExecutionMode::Sync => "sync",
                    };
                    let num_ops = segment.operations().len();
                    log_fusion(FusionLogLevel::Full, move || {
                        format!(
                            "[plan] cache hit: execute plan #{id} ({mode_dbg}, segment has {num_ops} ops)"
                        )
                    });

                    if let ExecutionMode::Sync = mode {
                        store.add_trigger(id, ExecutionTrigger::OnSync);
                    }

                    segment.execute(id, store);
                    self.reset(store, segment.operations());
                }
            };
        }
    }

    fn on_new_operation<Segment>(&mut self, segment: &Segment, store: &mut ExecutionPlanStore<O>)
    where
        Segment: StreamSegment<O>,
    {
        self.policy.update(
            store,
            segment
                .operations()
                .last()
                .expect("At least one operation in the operation list."),
        );
        self.explorer.on_new_operation();
    }

    fn explore<Item: StreamSegment<O>>(
        &mut self,
        item: &mut Item,
        store: &mut ExecutionPlanStore<O>,
        mode: ExecutionMode,
    ) {
        match self.explorer.explore(item.operations(), mode) {
            ExplorationAction::Completed(optim) => {
                let id = Self::on_exploration_completed(
                    &self.policy,
                    item.operations(),
                    store,
                    optim,
                    mode,
                );
                item.execute(id, store);
                self.reset(store, item.operations());
            }
            ExplorationAction::Continue => {
                if let ExecutionMode::Sync = mode {
                    panic!("Can't continue exploring when sync.")
                }
            }
        }
    }

    fn reset(&mut self, store: &mut ExecutionPlanStore<O>, operations: &[OperationIr]) {
        self.explorer.reset(operations);
        self.policy.reset();

        // Reset the policy state with the remaining operations
        for operation in operations.iter() {
            self.policy.update(store, operation);
        }
    }

    /// We found an optimization (i.e. a new execution plan).
    /// Cache it in the store.
    fn on_exploration_completed(
        policy: &Policy<O>,
        operations: &[OperationIr],
        store: &mut ExecutionPlanStore<O>,
        optimization: BlockOptimization<O>,
        mode: ExecutionMode,
    ) -> ExecutionPlanId {
        let num_optimized = optimization.ordering.len();
        let relative = &operations[0..num_optimized];

        explore_trace(relative);

        {
            let total_ops = operations.len();
            let mode_dbg = match mode {
                ExecutionMode::Lazy => "lazy",
                ExecutionMode::Sync => "sync",
            };
            log_fusion(FusionLogLevel::Full, move || {
                format!(
                    "[plan] exploration completed: {mode_dbg}, {num_optimized}/{total_ops} ops optimized"
                )
            });
        }

        match mode {
            ExecutionMode::Lazy => {
                let next_ops = &operations[num_optimized..operations.len()];

                let trigger = if next_ops.is_empty() {
                    // Happens if the next ops is included in the fused operation, and there is no
                    // way the builder can still continue fusing.
                    ExecutionTrigger::Always
                } else {
                    ExecutionTrigger::OnOperations(next_ops.to_vec())
                };

                match policy.action(store, relative, ExecutionMode::Sync) {
                    Action::Execute(id) => {
                        store.add_trigger(id, trigger);
                        id
                    }
                    _ => {
                        let plan = ExecutionPlan {
                            operations: relative.to_vec(),
                            triggers: vec![trigger],
                            optimization,
                        };
                        store.add(plan)
                    }
                }
            }
            ExecutionMode::Sync => match policy.action(store, relative, ExecutionMode::Sync) {
                Action::Execute(id) => {
                    store.add_trigger(id, ExecutionTrigger::OnSync);
                    id
                }
                _ => {
                    let plan = ExecutionPlan {
                        operations: relative.to_vec(),
                        triggers: vec![ExecutionTrigger::OnSync],
                        optimization,
                    };
                    store.add(plan)
                }
            },
        }
    }
}
