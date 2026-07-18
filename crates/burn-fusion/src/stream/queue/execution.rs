use burn_ir::{HandleContainer, TensorStatus};

use crate::{
    FusionRuntime, UnfusedOp,
    search::BlockOptimization,
    stream::{
        Context, ContextGuard, OperationConverter, OrderedExecution, RelativeOps, StreamId,
        execution::log_execution_table,
        store::{ExecutionPlanId, ExecutionPlanStore, ExecutionStrategy},
    },
};

use super::OperationQueue;

impl<R: FusionRuntime> OperationQueue<R> {
    /// Execute the queue partially following the execution strategy from the plan.
    pub(crate) fn execute(
        &mut self,
        id: ExecutionPlanId,
        handles: &mut HandleContainer<R::FusionHandle>,
        store: &mut ExecutionPlanStore<R::Optimization>,
        stream_id: StreamId,
    ) {
        let plan = store.get_mut_unchecked(id);
        self.execute_block_optimization(&mut plan.optimization, handles, stream_id);
    }

    fn execute_block_optimization(
        &mut self,
        step: &mut BlockOptimization<R::Optimization>,
        handles: &mut HandleContainer<R::FusionHandle>,
        stream_id: StreamId,
    ) {
        log_execution_table(stream_id, &step.strategy, &self.global);

        // FORENSICS for the long-standing ordering.rs index-OOB panic ("len is
        // 1 but the index is 1", deterministic on some LoRA-training datasets):
        // a (possibly cached) execution plan whose orderings reference indices
        // past the live queue is about to OOB in execute_operations. Dump the
        // full plan/queue shape BEFORE the panic so the mismatch is diagnosable
        // from production logs. Cost when healthy: one recursive max() walk.
        let max_idx = strategy_max_index(&step.strategy);
        if let Some(m) = max_idx {
            if m >= self.operations.len() {
                // Full dump only for the first few occurrences per process —
                // the first capture of this dump repeated on EVERY register of
                // a corrupted stream and flooded the journal (the queue reset
                // above should make repeats impossible, but never assume).
                use core::sync::atomic::{AtomicU32, Ordering};
                static DUMPS: AtomicU32 = AtomicU32::new(0);
                let n = DUMPS.fetch_add(1, Ordering::Relaxed);
                if n < 5 {
                    log::error!(
                        "FUSION PLAN/QUEUE MISMATCH stream={stream_id:?}: strategy max index {m} >= \
                         operations len {} (global len {}, relative len {}); strategy shape: {}; \
                         global ops: [{}]",
                        self.operations.len(),
                        self.global.len(),
                        self.relative.len(),
                        render_strategy_shape(&step.strategy),
                        self.global
                            .iter()
                            .map(|op| {
                                let d = format!("{op:?}");
                                d.chars().take(80).collect::<String>()
                            })
                            .collect::<Vec<_>>()
                            .join(" | "),
                    );
                } else {
                    log::error!(
                        "FUSION PLAN/QUEUE MISMATCH (dump {n} suppressed): max index {m} >= \
                         operations len {} (global len {})",
                        self.operations.len(),
                        self.global.len(),
                    );
                }
            }
        }

        let mut operations = Vec::new();
        core::mem::swap(&mut operations, &mut self.operations);

        // PATCH (diffusion-app): panic-safe execution (upstream tracel-ai/burn
        // #4827, open). `operations` was just moved out of `self`; if the
        // strategy panics mid-execution (observed: a cubecl device-server error
        // latched by a failed autotune allocation makes a tensor read panic),
        // the unwind used to leave this queue INCONSISTENT — `self.operations`
        // empty while `self.global`/`self.relative` keep their (and future)
        // entries. Every subsequent register on the stream then hits the
        // ordering index-OOB, an unbounded panic storm that floods the journal
        // and eventually kills the GPU worker thread. Instead: catch the
        // panic, reset the queue to a CONSISTENT empty state (leaking the
        // in-flight ops' handles — the device pool drain on the training/
        // generation error path reclaims them), and resume the original panic
        // so the caller still sees one honest failure.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_strategy(step, &mut self.converter, handles, operations)
        }));
        let (operations, num_drained) = match result {
            Ok(r) => r,
            Err(payload) => {
                log::error!(
                    "fusion strategy execution panicked; resetting stream queue to a consistent \
                     empty state ({} global ops dropped) and propagating the panic",
                    self.global.len()
                );
                self.global.clear();
                self.reset_relative();
                // global is now empty, so every pending out-of-band drop is
                // unreferenced — free them here rather than leaking (consistent
                // with the in-flight ops' handles being reclaimed by the device
                // pool drain on the error path).
                self.sweep_dropped(handles);
                std::panic::resume_unwind(payload);
            }
        };

        self.operations = operations;
        self.drain_queue(num_drained, handles);
    }

    /// Bookkeeping after executing `num_drained` operations from the queue.
    fn drain_queue(&mut self, num_drained: usize, handles: &mut HandleContainer<R::FusionHandle>) {
        self.global[0..num_drained]
            .iter()
            .flat_map(|desc| desc.nodes())
            .for_each(|tensor| {
                if tensor.status == TensorStatus::ReadWrite {
                    self.variables.remove(&tensor.id);
                };
                handles.free(tensor)
            });

        self.global.drain(0..num_drained);

        self.reset_relative();
        self.sweep_dropped(handles);
    }

    /// Free the handles of tensors recorded for out-of-band drop
    /// (`BURN_FUSION_DROP_OOB=1`; see [`OperationQueue::dropped`]) whose last
    /// referencing op has now drained. A dropped tensor is freed only once no
    /// remaining op in `global` references it, which guarantees we never free a
    /// handle a pending op still needs (the handle.rs "Should have handle" panic
    /// class). `remove_handle` is an idempotent map removal, so if the drain
    /// above already freed a ReadWrite last-use, this is a no-op. No-op entirely
    /// when the flag is off (the list stays empty).
    fn sweep_dropped(&mut self, handles: &mut HandleContainer<R::FusionHandle>) {
        if self.dropped.is_empty() {
            return;
        }
        self.dropped.retain(|id| {
            let still_referenced = self
                .global
                .iter()
                .flat_map(|op| op.nodes())
                .any(|node| node.id == *id);
            if still_referenced {
                true
            } else {
                handles.remove_handle(*id);
                false
            }
        });
    }

    fn reset_relative(&mut self) {
        self.relative.clear();
        self.converter.clear();

        for node in self.global.iter() {
            let relative = node.to_relative(&mut self.converter);
            self.relative.push(relative);
        }
    }
}

/// Drive one block's execution strategy.
///
/// Wraps the converter's per-block fields and the handle container into a single owned
/// [`Context`] via [`ContextGuard`] for the duration of this call, then threads `&mut Context`
/// through the recursive strategy walk. Operations-only strategies just grab
/// `&mut ctx.handles`; optimization strategies hand `&mut ctx` to the fused op.
fn run_strategy<R: FusionRuntime>(
    optimization: &mut BlockOptimization<R::Optimization>,
    converter: &mut OperationConverter,
    handles: &mut HandleContainer<R::FusionHandle>,
    operations: Vec<UnfusedOp<R>>,
) -> (Vec<UnfusedOp<R>>, usize) {
    let mut execution = OrderedExecution::new(operations);
    {
        let mut guard = ContextGuard::new(converter, handles);
        execute_strategy::<R>(&mut optimization.strategy, &mut guard, &mut execution);
    }
    execution.finish()
}

/// Largest operation index referenced anywhere in a strategy tree (None = the
/// strategy references no operations). Forensics helper for the ordering-OOB
/// panic; see `execute_block_optimization`.
fn strategy_max_index<O>(strategy: &ExecutionStrategy<O>) -> Option<usize> {
    match strategy {
        ExecutionStrategy::Optimization { ordering, .. } => ordering.iter().copied().max(),
        ExecutionStrategy::Operations { ordering } => ordering.iter().copied().max(),
        ExecutionStrategy::Composed(items) => {
            items.iter().filter_map(|i| strategy_max_index(i)).max()
        }
    }
}

/// Compact one-line rendering of a strategy tree with its orderings.
fn render_strategy_shape<O>(strategy: &ExecutionStrategy<O>) -> String {
    match strategy {
        ExecutionStrategy::Optimization { ordering, score, .. } => {
            format!("Opt(score={score}, ordering={ordering:?})")
        }
        ExecutionStrategy::Operations { ordering } => format!("Ops(ordering={ordering:?})"),
        ExecutionStrategy::Composed(items) => format!(
            "Composed[{}]",
            items
                .iter()
                .map(|i| render_strategy_shape(i))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn execute_strategy<R: FusionRuntime>(
    strategy: &mut ExecutionStrategy<R::Optimization>,
    context: &mut Context<R::FusionHandle>,
    execution: &mut OrderedExecution<R>,
) {
    match strategy {
        ExecutionStrategy::Optimization { ordering, opt, .. } => {
            execution.execute_optimization(opt, context, ordering.clone());
        }
        ExecutionStrategy::Operations { ordering } => {
            execution.execute_operations(&mut context.handles, ordering);
        }
        ExecutionStrategy::Composed(items) => {
            for item in items.iter_mut() {
                execute_strategy::<R>(item, context, execution);
            }
        }
    }
}
