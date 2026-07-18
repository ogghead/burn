use super::{
    StreamId,
    execution::{ExecutionMode, Processor, StreamSegment},
    queue::OperationQueue,
    store::{ExecutionPlanId, ExecutionPlanStore},
};
use crate::{FusionRuntime, UnfusedOp};
use burn_ir::{HandleContainer, OperationIr, TensorId};
use hashbrown::{HashMap, HashSet};

/// Out-of-band Drop handling (EXPERIMENTAL, DEFAULT OFF).
///
/// When enabled, `Drop` operations are never enqueued into a stream's fusion
/// queue. Instead the dropped tensor's handle is freed out-of-band — immediately
/// if no pending op still references it, otherwise deferred to the home stream's
/// next drain (see [`OperationQueue::dropped`](super::queue::OperationQueue)).
///
/// # Why
///
/// Autodiff temporary deallocations register `Drop` ops that land at *varying*
/// positions in the relative op stream every training step. Because plan identity
/// is the relative op subsequence, those position-varying drops fragment
/// otherwise-identical compute graphs into distinct plans → per-step NVRTC
/// recompilation storm. Keeping drops out of the stream makes plan identity
/// drop-free and stable, and — crucially — means no drop membership is ever baked
/// into a cacheable plan (the earlier absorb-into-block approach did that and
/// crashed with a stale cached free: burn-ir/handle.rs "Should have handle").
///
/// Read once and cached. Opt in with `BURN_FUSION_DROP_OOB=1`.
///
/// Public so regression tests in dependent crates can assert the mode is active.
#[cfg(feature = "std")]
pub fn drop_out_of_band() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("BURN_FUSION_DROP_OOB").as_deref(),
            Ok("1") | Ok("true")
        )
    })
}

#[cfg(not(feature = "std"))]
pub fn drop_out_of_band() -> bool {
    false
}

/// Keep track of multiple concurrent lazy streams of operations.
///
/// # Why this exists
///
/// Each `Stream` holds a lazy queue of [`OperationIr`]s whose inputs are assumed
/// to live on that stream. That makes single-stream execution simple — every
/// `TensorId` in a queue is resolvable from the same handle map and the same
/// pending op chain. But a [`FusionTensor`](crate::FusionTensor) is `Send + Clone`,
/// so user code can move or clone a tensor from one thread (= one [`StreamId`]) to
/// another. The receiving thread will then submit ops whose inputs reference a
/// tensor whose home is a *different* stream's queue. This struct is what makes
/// that case behave correctly without giving up the stream-locality invariant.
///
/// # Strategy: shared views
///
/// We never let a foreign-stream tensor id appear in another stream's queue.
/// Instead, when [`FusionTensor::clone`](crate::FusionTensor::clone) or
/// [`FusionTensor::into_ir`](crate::FusionTensor::into_ir) detects that
/// `self.stream != StreamId::current()`, it allocates a fresh id (`dst`) and calls
/// `tag_shared_view` with `(src_stream, src, dst)`. That call does two
/// things, in order:
///
/// 1. **Materialise `src`.** The id `src` might be the output of an op still
///    pending on `src_stream`. We need its backing handle to actually exist before
///    we can alias it. If [`HandleContainer::get_handle_ref`] returns `None` —
///    meaning no op has produced a handle for `src` yet — we drain `src_stream`
///    synchronously, forcing every pending op to run (and thus the handle to be
///    registered). We also record `src` in `shared_sources` so that any
///    *next* share of the same `src` can skip the drain: once registered, a
///    handle stays put (see invariants below).
///
/// 2. **Alias the handle under `dst`.** [`HandleContainer::register_handle`] is
///    called with `dst` and a `clone()` of `src`'s backend handle. Cubecl handles
///    are `Arc`-style reference counters over a backing buffer, so `clone()` is
///    cheap and the buffer survives until the last alias drops. After this call,
///    `handles[src]` and `handles[dst]` are two distinct map entries that both
///    point at the same allocation.
///
/// `shared_view` then returns a new `FusionTensor` carrying `(id = dst, stream =
/// current)`. Every subsequent op on that tensor enqueues on `current` like any
/// other local tensor — the rest of the fusion engine sees no special case.
///
/// # Freeing
///
/// Each `FusionTensor::drop` enqueues an `OperationIr::Drop(ir)` on **its own**
/// `stream` field (the home stream of that particular alias), not the calling
/// thread's stream. So:
///
/// - The original tensor's drop targets `src_stream` and removes `handles[src]`.
/// - The alias tensor's drop targets the stream that minted it and removes
///   `handles[dst]`.
///
/// Each removal decrements the backend handle's `Arc` refcount; the underlying
/// buffer is freed only after the last side drops. No cross-stream coordination
/// is needed.
///
/// # Bounding `shared_sources`
///
/// Naively the set would grow forever, since `tag_shared_view` only ever inserts.
/// Cleanup happens in `register`: as soon as we see an
/// `OperationIr::Drop(ir)` come through, we remove `ir.id` from
/// `shared_sources` immediately — without waiting for the queued `Drop`
/// to actually execute. This is safe because a `Drop` op is registered only
/// after the last live `FusionTensor` with that id has been dropped, so no
/// future `tag_shared_view` can possibly receive that id as a `src`. Removing
/// the entry therefore cannot trigger a redundant drain on any subsequent call.
///
/// # The SSA-like invariant
///
/// Skipping the drain on subsequent shares of the same `src` relies on a
/// property of the fusion IR: **every op output uses a fresh `TensorId`
/// allocated by [`crate::Client::create_empty_handle`], never the id of an
/// input.** Once `handles[src]` is set, no later op overwrites it; the data
/// behind `src` is effectively immutable from the IR's point of view.
///
/// The cubecl-fusion engine *does* sometimes reuse the backing buffer of an
/// input for an output (in-place fusion), but that path is gated by
/// `handle.can_mut()`, which returns false the moment another reference exists.
/// Calling `handle.clone()` in step 2 above is precisely that extra reference,
/// so aliased sources are never eligible for in-place reuse — the engine
/// allocates a fresh output buffer instead.
///
/// # The chained-share fast path
///
/// When a share is itself re-shared (owner → peer → grandpeer), the second
/// `tag_shared_view` call has `src = peer's id`. That id was set up directly by
/// the previous call (via `register_handle`), not by an op enqueued on the peer
/// stream, so `handles.get_handle_ref(&src)` is already `Some` the moment we
/// look. The drain check therefore short-circuits — even though `peer's id` was
/// never added to `shared_sources` (only sources that *required* a
/// drain are tracked there), the handle-existence test alone is sufficient.
pub struct MultiStream<R: FusionRuntime> {
    /// Tensor ids that have been the source of a cross-stream share *and*
    /// required a drain when first shared. Used by `tag_shared_view` to
    /// skip the drain on subsequent shares of the same source. Bounded by
    /// pruning in `register` when a `Drop` op for the id is enqueued —
    /// see the struct-level docs for the full strategy.
    shared_sources: HashSet<TensorId>,
    streams: HashMap<StreamId, Stream<R>>,
    optimizations: ExecutionPlanStore<R::Optimization>,
    device: R::FusionDevice,
    #[cfg(feature = "memory-checks")]
    memory_checks: super::memory_checks::MemoryChecks,
}

impl<R: FusionRuntime> MultiStream<R> {
    pub(crate) fn new(device: R::FusionDevice) -> Self {
        Self {
            shared_sources: HashSet::new(),
            streams: HashMap::new(),
            optimizations: ExecutionPlanStore::new(),
            device,
            #[cfg(feature = "memory-checks")]
            memory_checks: super::memory_checks::MemoryChecks::default(),
        }
    }

    /// Register a new tensor operation on the given `stream`.
    pub(crate) fn register(
        &mut self,
        stream: StreamId,
        repr: OperationIr,
        operation: UnfusedOp<R>,
        handles: &mut HandleContainer<R::FusionHandle>,
    ) {
        // Bound `shared_sources` (see struct-level docs). When the last `FusionTensor`
        // for an id is dropped, a `Drop` op is registered here. At that point no live
        // `FusionTensor` holds this id, so no future `tag_shared_view` can use it as
        // a source — it is safe to drop the entry immediately, without waiting for
        // the queued `Drop` op to actually execute.
        if let OperationIr::Drop(ir) = &repr {
            self.shared_sources.remove(&ir.id);

            // Out-of-band drop (BURN_FUSION_DROP_OOB=1): keep the Drop out of the
            // fusion stream so it can't fragment plan identity (the per-step
            // recompilation storm). Free the handle now if no pending op on the
            // home stream still references the tensor, otherwise defer to that
            // stream's next drain.
            //
            // Safety — two load-bearing invariants make the single-queue guard
            // (`references` scans only `stream`'s own queue) complete:
            //
            // 1. No FUTURE op can reference this id. A `Drop` is registered by
            //    `FusionTensor::drop` only when the frontend refcount hits 0
            //    (tensor.rs), so no live `FusionTensor` for this id remains to be
            //    submitted into any op on any stream.
            //
            // 2. No op on ANOTHER stream can reference this id, now or pending.
            //    A tensor id is local to its home stream: when a `FusionTensor`
            //    crosses a stream boundary (`Clone`/`into_ir` seeing
            //    `self.stream != current`), `shared_view` mints a FRESH id and
            //    `tag_shared_view` aliases the handle under it via
            //    `register_handle(dst, handle.clone())` — an independent map entry
            //    holding its own Arc-cloned handle. So foreign streams only ever
            //    reference the alias `dst`, never this id; and freeing this id's
            //    entry cannot disturb `dst`'s clone (the backing buffer survives on
            //    `dst`'s refcount until its own Drop). `stream` here is the Drop's
            //    home stream (FusionTensor::drop targets `self.stream`), so the
            //    queue we scan is the only one this id can appear in.
            //
            // Memory timing (vs the rejected batched-drop option): this is a
            // deferred free, but bounded — the id is freed at the first drain after
            // its last referencing op drains, i.e. worst-case one drain window of
            // extra lifetime. That matches the original in-stream Drop, which also
            // only freed when its own drain reached it. `sweep_dropped` runs on
            // EVERY drain and retains only still-referenced ids, so the deferred
            // list never survives a drain unswept and cannot compound.
            if drop_out_of_band() {
                match self.streams.get_mut(&stream) {
                    Some(s) if s.queue.references(ir.id) => s.queue.defer_drop(ir.id),
                    _ => {
                        handles.remove_handle(ir.id);
                    }
                }
                return;
            }
        }

        self.enqueue_operation(stream, repr, operation, handles);

        #[cfg(feature = "memory-checks")]
        self.memory_checks.check(&self.streams, handles);
        #[cfg(feature = "test-util")]
        crate::inspect::emit_handle_snapshot(stream, handles.handle_ids().copied());
    }

    /// Set up a cross-stream alias `dst` for the foreign tensor `src` that lives on
    /// `src_stream`. Called when [`FusionTensor::clone`](crate::FusionTensor::clone)
    /// or [`FusionTensor::into_ir`](crate::FusionTensor::into_ir) detects that the
    /// tensor's home stream is not the current stream.
    ///
    /// See the [`MultiStream`] struct-level docs for the full strategy. In short:
    ///
    /// - If `src`'s handle isn't materialised yet, drain `src_stream` so the
    ///   producing op runs and registers it. Skip the drain on subsequent shares
    ///   of the same source by remembering it in `shared_sources` *or* by
    ///   observing that the handle is already in the container (which covers the
    ///   chained-share case where `src` is itself a previously-aliased view).
    /// - Then alias the backing handle under `dst`. `register_handle` clones the
    ///   cubecl handle (`Arc`-style), so both ids share refcount on the buffer
    ///   until each side's own `Drop` op runs.
    pub(crate) fn tag_shared_view(
        &mut self,
        src_stream: StreamId,
        src: TensorId,
        dst: TensorId,
        handles: &mut HandleContainer<R::FusionHandle>,
    ) {
        // Drain only when neither short-circuit applies: `shared_sources` records ids
        // we already drained for, and a `Some` handle means `src` is materialised
        // (e.g., it was itself set up by an earlier `tag_shared_view` call). We
        // record `src` only when we actually drain — the handle-existence path is
        // naturally idempotent on later calls.
        if !self.shared_sources.contains(&src) && handles.get_handle_ref(&src).is_none() {
            self.shared_sources.insert(src);
            self.drain(handles, src_stream);
        }

        if let Some(handle) = handles.get_handle_ref(&src) {
            handles.register_handle(dst, handle.clone());
        }
    }

    /// Enqueue an operation on the queue for `stream` and run the lazy processor.
    fn enqueue_operation(
        &mut self,
        stream: StreamId,
        repr: OperationIr,
        operation: UnfusedOp<R>,
        handles: &mut HandleContainer<R::FusionHandle>,
    ) {
        let s = self
            .streams
            .entry(stream)
            .or_insert_with(|| Stream::new(self.device.clone()));
        s.queue.add(repr, operation);

        let len_before = s.queue.global.len();
        s.processor.process(
            Segment::new(&mut s.queue, handles, stream),
            &mut self.optimizations,
            ExecutionMode::Lazy,
        );
        let len_after = s.queue.global.len();
        s.cursor += (len_before - len_after) as u64;
    }

    /// Mark a tensor as read.
    #[allow(unused_variables)]
    pub fn mark_read(
        &mut self,
        id: StreamId,
        ir: &burn_ir::TensorIr,
        handles: &HandleContainer<R::FusionHandle>,
    ) {
        if !matches!(ir.status, burn_ir::TensorStatus::ReadWrite) {
            return;
        };

        let stream = match self.streams.get_mut(&id) {
            Some(val) => val,
            None => return,
        };

        stream.queue.variables.remove(&ir.id);

        // Keep the stream alive while it still owes out-of-band drop frees
        // (BURN_FUSION_DROP_OOB=1); dropping the queue here would leak those
        // handles. The next drain on this stream sweeps them, after which a
        // later mark_read can remove it.
        if stream.queue.variables.is_empty() && stream.queue.dropped.is_empty() {
            self.streams.remove(&id);
        }

        #[cfg(feature = "memory-checks")]
        self.memory_checks.check(&self.streams, handles);
        #[cfg(feature = "test-util")]
        crate::inspect::emit_handle_snapshot(id, handles.handle_ids().copied());
    }

    /// Drain a stream.
    pub fn drain(&mut self, handles: &mut HandleContainer<R::FusionHandle>, id: StreamId) {
        id.executes(|| {
            if let Some(stream) = self.streams.get_mut(&id) {
                let num_executed = stream.queue.global.len();
                stream.processor.process(
                    Segment::new(&mut stream.queue, handles, id),
                    &mut self.optimizations,
                    ExecutionMode::Sync,
                );
                stream.cursor += num_executed as u64;
            }
        });
        #[cfg(feature = "test-util")]
        crate::inspect::emit_handle_snapshot(id, handles.handle_ids().copied());
    }
}

pub(crate) struct Stream<R: FusionRuntime> {
    pub(crate) queue: OperationQueue<R>,
    processor: Processor<R::Optimization>,
    pub(crate) cursor: u64,
}

#[derive(new)]
struct Segment<'a, R: FusionRuntime> {
    queue: &'a mut OperationQueue<R>,
    handles: &'a mut HandleContainer<R::FusionHandle>,
    id: StreamId,
}

impl<R: FusionRuntime> StreamSegment<R::Optimization> for Segment<'_, R> {
    fn operations(&self) -> &[OperationIr] {
        &self.queue.relative
    }

    fn execute(&mut self, id: ExecutionPlanId, store: &mut ExecutionPlanStore<R::Optimization>) {
        self.queue.execute(id, self.handles, store, self.id)
    }
}

impl<R: FusionRuntime> Stream<R> {
    fn new(device: R::FusionDevice) -> Self {
        Self {
            processor: Processor::new(R::fusers(device)),
            queue: OperationQueue::new(),
            cursor: 0,
        }
    }
}
