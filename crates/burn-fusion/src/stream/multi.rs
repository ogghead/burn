use super::{
    StreamId,
    execution::{ExecutionMode, Processor, StreamSegment},
    queue::OperationQueue,
    store::{ExecutionPlanId, ExecutionPlanStore},
};
use crate::{FusionRuntime, UnfusedOp};
use burn_ir::{HandleContainer, OperationIr, TensorId};
use hashbrown::{HashMap, HashSet};

/// Out-of-band Drop handling (DEFAULT ON — identity-verified).
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
/// DEFAULT ON as of the identity-verified rollout (in-app: storm collapsed
/// 530→frozen, pool flat, LoRA identity 0.5746 vs 0.5239 baseline). The
/// kill-switch is `BURN_FUSION_DROP_OOB=0` (or `false`), which restores the
/// legacy in-stream Drop path. Read once and cached.
///
/// Public so regression tests in dependent crates can assert the mode is active.
#[cfg(feature = "std")]
pub fn drop_out_of_band() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("BURN_FUSION_DROP_OOB").as_deref(),
            Ok("0") | Ok("false")
        )
    })
}

#[cfg(not(feature = "std"))]
pub fn drop_out_of_band() -> bool {
    false
}

/// Diagnostic (env-gated by `BURN_FUSION_ARITH_DUMP=N`) for the residual-bf16
/// step-time round: catalogue elementwise / cast float operations by
/// `(variant, dtype, input dtypes, output shape)` as they are registered —
/// i.e. BEFORE any fusion decision, so it sees the UNFUSED f32 binops (the
/// census population this round targets) as well as the fused ones, and its
/// per-signature counts equal real per-op launch counts.
///
/// Purpose: classify the f32 elementwise census (binop / scalar / unary /
/// cast) as residual-stream (bf16-convertible: last-dim ~3072 / img-txt token
/// shapes) vs intrinsically-f32 (loss / v_target / sigma: scalar / tiny
/// shapes). Shape / movement / init ops are excluded on purpose — those are
/// the ops that drowned the `ops == 1` `EXPLORE_DUMP` and are unfusable by
/// design regardless.
///
/// Prints each new signature once on first sight (capped at `N`) and a full
/// count-sorted census every `BURN_FUSION_ARITH_CENSUS` registrations
/// (default 20000). Zero cost when the env var is unset.
#[cfg(feature = "std")]
fn arith_dump(repr: &OperationIr) {
    use burn_ir::BaseOperationIr;
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    static N: OnceLock<u64> = OnceLock::new();
    let n_cap = *N.get_or_init(|| {
        std::env::var("BURN_FUSION_ARITH_DUMP")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0)
    });
    if n_cap == 0 {
        return;
    }

    // Variant name only: the leading identifier of the inner op's Debug, before
    // any `(`/`{`/space — avoids hand-maintaining a match over every variant.
    fn variant(dbg: String) -> String {
        dbg.split(['(', ' ', '{'])
            .next()
            .unwrap_or("?")
            .to_string()
    }

    let sig = match repr {
        OperationIr::NumericFloat(dtype, op) => {
            Some(format!("NumF/{} dtype={dtype:?}", variant(format!("{op:?}"))))
        }
        OperationIr::Float(dtype, op) => {
            Some(format!("Float/{} dtype={dtype:?}", variant(format!("{op:?}"))))
        }
        OperationIr::BaseFloat(BaseOperationIr::Cast(c)) => {
            Some(format!("Cast {:?}->{:?}", c.input.dtype, c.out.dtype))
        }
        _ => None,
    };
    let Some(mut sig) = sig else {
        return;
    };

    // Input dtypes reveal mixed f32×bf16 populations; output shape is the
    // residual-vs-intrinsic discriminator.
    let ins: Vec<String> = repr.inputs().map(|t| format!("{:?}", t.dtype)).collect();
    if !ins.is_empty() {
        sig.push_str(&format!(" in=[{}]", ins.join(",")));
    }
    if let Some(out) = repr.outputs().next() {
        sig.push_str(&format!(" out={:?}", out.shape));
    }

    static STATE: OnceLock<Mutex<(HashMap<String, u64>, u64)>> = OnceLock::new();
    let state = STATE.get_or_init(|| Mutex::new((HashMap::new(), 0)));
    let mut guard = state.lock().unwrap();
    let (counts, total) = &mut *guard;
    *total += 1;
    let total_now = *total;
    let distinct_before = counts.len() as u64;
    let entry = counts.entry(sig.clone()).or_insert(0);
    let first_seen = *entry == 0;
    *entry += 1;
    if first_seen && distinct_before < n_cap {
        eprintln!("[ARITH_DUMP #{} {sig}]", distinct_before + 1);
    }

    static CENSUS_EVERY: OnceLock<u64> = OnceLock::new();
    let every = *CENSUS_EVERY.get_or_init(|| {
        std::env::var("BURN_FUSION_ARITH_CENSUS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(20000)
    });
    if every > 0 && total_now % every == 0 {
        let mut rows: Vec<(&String, &u64)> = counts.iter().collect();
        rows.sort_by(|a, b| b.1.cmp(a.1));
        eprintln!(
            "[ARITH_CENSUS total={total_now} distinct={}]",
            counts.len()
        );
        for (i, (s, c)) in rows.iter().take(80).enumerate() {
            eprintln!("  [{i:>3}] {c:>8}  {s}");
        }
    }
}

#[cfg(not(feature = "std"))]
fn arith_dump(_repr: &OperationIr) {}

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

        arith_dump(&repr);

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

#[cfg(all(test, feature = "std"))]
mod arith_dump_tests {
    use super::arith_dump;
    use burn_backend::{DType, Shape};
    use burn_ir::{
        BaseOperationIr, BinaryOpIr, CastOpIr, NumericOperationIr, OperationIr, ShapeOpIr,
        TensorId, TensorIr, TensorStatus,
    };

    fn tensor(id: u64, shape: &[usize], dtype: DType) -> TensorIr {
        TensorIr {
            id: TensorId::new(id),
            shape: Shape::from(shape.to_vec()),
            status: TensorStatus::ReadOnly,
            dtype,
        }
    }

    // Not an oracle — a format smoke check so the one GPU-1 capture window is
    // not spent discovering a broken diagnostic. Run with --nocapture to eyeball.
    #[test]
    fn arith_dump_classifies_and_ignores() {
        // SAFETY: single-threaded test setting env before the first arith_dump
        // call in this process; OnceLock reads it exactly once.
        unsafe {
            std::env::set_var("BURN_FUSION_ARITH_DUMP", "50");
            std::env::set_var("BURN_FUSION_ARITH_CENSUS", "3");
        }

        // Residual-class f32 add (last-dim 3072) — must be catalogued.
        let residual_add = OperationIr::NumericFloat(
            DType::F32,
            NumericOperationIr::Add(BinaryOpIr {
                lhs: tensor(0, &[1, 4608, 3072], DType::F32),
                rhs: tensor(1, &[1, 4608, 3072], DType::F32),
                out: tensor(2, &[1, 4608, 3072], DType::F32),
            }),
        );
        // Boundary cast f32 -> bf16 — must be catalogued with direction.
        let cast = OperationIr::BaseFloat(BaseOperationIr::Cast(CastOpIr {
            input: tensor(3, &[1, 4608, 3072], DType::F32),
            out: tensor(4, &[1, 4608, 3072], DType::BF16),
        }));
        // A shape op — must be IGNORED (returns None).
        let reshape = OperationIr::BaseFloat(BaseOperationIr::Reshape(ShapeOpIr {
            input: tensor(5, &[1, 4608, 3072], DType::F32),
            out: tensor(6, &[4608, 3072], DType::F32),
        }));

        // Feed enough to trip the count-sorted census (CENSUS=3).
        arith_dump(&residual_add);
        arith_dump(&cast);
        arith_dump(&reshape);
        arith_dump(&residual_add);
        arith_dump(&reshape);
    }
}
