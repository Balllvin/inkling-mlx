"""Continuous-batching Inkling 2 inference engine with prefix/turn KV cache and streaming."""

from __future__ import annotations

import os
import queue
import secrets
import threading
import time

_LOG_TURNS = os.environ.get("INKLING_LOG_TURNS", "") == "1"
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx
from mlx_vlm.models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchRotatingKVCache,
    CacheList,
    KVCache,
    RotatingKVCache,
    cache_nbytes,
)

from . import fused
from .model import GrowingBatchKVCache, GrowingKVCache, LanguageModel
from .mtp import (
    InklingMTPDrafter,
    clone_draft_cache,
    eval_draft_cache,
    reject_sample_mtp,
    rollback_single_cache,
    snapshot_rotating_kv,
)
from .native_decode import try_create_native_decode_runner
from .parser import STOP_IDS, InklingStreamParser


def _clone_cache_list(caches: list[CacheList]) -> list[CacheList]:
    """Deep-copy a single-sequence CacheList stack so mutations never alias snapshots."""
    out: list[CacheList] = []
    for cl in caches:
        kv, conv = cl.caches[0], cl.caches[1]
        if isinstance(kv, RotatingKVCache):
            nkv = RotatingKVCache(max_size=kv.max_size, keep=kv.keep)
            if kv.keys is not None:
                nkv.keys = mx.array(kv.keys)
                nkv.values = mx.array(kv.values)
            nkv.offset = kv.offset
            nkv._idx = kv._idx
        else:
            nkv = GrowingKVCache() if isinstance(kv, GrowingKVCache) else KVCache()
            if kv.keys is not None:
                nkv.keys = mx.contiguous(kv.keys[..., : kv.offset, :])
                nkv.values = mx.contiguous(kv.values[..., : kv.offset, :])
            nkv.offset = kv.offset
        nconv = ArraysCache(len(conv.cache))
        for i, arr in enumerate(conv.cache):
            if arr is not None:
                nconv.cache[i] = mx.array(arr)
        out.append(CacheList(nkv, nconv))
    return out


def _eval_cache(caches: list[CacheList]) -> None:
    arrs = []
    for cl in caches:
        kv, conv = cl.caches[0], cl.caches[1]
        if kv.keys is not None:
            arrs.append(kv.keys)
        if kv.values is not None:
            arrs.append(kv.values)
        for a in conv.cache:
            if a is not None:
                arrs.append(a)
    if arrs:
        mx.eval(*arrs)


def _clear_conv_padding(batch_cache: list[CacheList]) -> None:
    """Clear exhausted left-padding metadata on ArraysCache after prefill finishes."""
    for cl in batch_cache:
        conv = cl.caches[1]
        conv._left_padding = None
        conv._left_padding_advance = 0
        conv._lengths = None


def _extend_batch_cache(dst: list[CacheList], src: list[CacheList]) -> list[CacheList]:
    """O(1) in-place extend of a batched CacheList stack with another batched CacheList stack."""
    for cl_d, cl_s in zip(dst, src):
        cl_d.caches[0].extend(cl_s.caches[0])
        cl_d.caches[1].extend(cl_s.caches[1])
    _clear_conv_padding(dst)
    return dst


def _merge_single_caches(caches_per_req: list[list[CacheList]]) -> list[CacheList]:
    """Merge B single-sequence caches into one batched CacheList stack."""
    n_visits = len(caches_per_req[0])
    merged: list[CacheList] = []
    for v in range(n_visits):
        kv_list = [req_c[v].caches[0] for req_c in caches_per_req]
        conv_list = [req_c[v].caches[1] for req_c in caches_per_req]
        if isinstance(kv_list[0], RotatingKVCache):
            bkv = BatchRotatingKVCache.merge(kv_list)
        elif isinstance(kv_list[0], GrowingKVCache):
            bkv = GrowingBatchKVCache.merge(kv_list)
        else:
            bkv = BatchKVCache.merge(kv_list)
        bconv = ArraysCache.merge(conv_list)
        merged.append(CacheList(bkv, bconv))
    return merged


def _extract_row_cache(batch_cache: list[CacheList], row: int) -> list[CacheList]:
    out: list[CacheList] = []
    for cl in batch_cache:
        kv = cl.caches[0].extract(row)
        conv = cl.caches[1].extract(row)
        conv._left_padding = None
        conv._left_padding_advance = 0
        conv._lengths = None
        out.append(CacheList(kv, conv))
    return out


class PrefixCache:
    """LRU cache of (token_tuple -> (cloned CacheList stack, optional cloned draft cache)) bounded by byte budget."""

    def __init__(self, max_bytes: int = 4 * (1 << 30), max_entries: int = 64):
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple[int, ...], tuple[list[CacheList], Optional[list], int]] = OrderedDict()
        self.total_bytes = 0
        self.hits = 0
        self.misses = 0

    def lookup(self, ids: list[int]) -> tuple[int, Optional[list[CacheList]], Optional[list]]:
        """Find longest cached strict prefix of `ids` (leaving at least 1 token to evaluate)."""
        if self.max_bytes <= 0 or len(ids) < 2 or not self._entries:
            self.misses += 1
            return 0, None, None
        best_key = None
        best_len = 0
        for key in reversed(self._entries):
            kl = len(key)
            if kl < len(ids) and kl > best_len and tuple(ids[:kl]) == key:
                best_key = key
                best_len = kl
        if best_key is None:
            self.misses += 1
            return 0, None, None
        self._entries.move_to_end(best_key)
        self.hits += 1
        snap, dsnap, _ = self._entries[best_key]
        return best_len, _clone_cache_list(snap), (clone_draft_cache(dsnap) if dsnap is not None else None)

    def _est_trimmed_bytes(self, caches: list[CacheList], draft_cache: Optional[list] = None) -> int:
        total = 0
        for cl in caches:
            kv, conv = cl.caches[0], cl.caches[1]
            if kv.keys is not None:
                if isinstance(kv, RotatingKVCache):
                    total += kv.keys.nbytes + kv.values.nbytes
                else:
                    s = kv.keys.shape
                    total += 2 * s[0] * s[1] * int(kv.offset) * s[3] * kv.keys.dtype.size
            for a in conv.cache:
                if a is not None:
                    total += a.nbytes
        if draft_cache is not None:
            for c in draft_cache:
                if c.keys is not None:
                    if isinstance(c, RotatingKVCache):
                        total += c.keys.nbytes + c.values.nbytes
                    else:
                        s = c.keys.shape
                        total += 2 * s[0] * s[1] * int(c.offset) * s[3] * c.keys.dtype.size
        return total

    def put(
        self,
        ids: tuple[int, ...],
        caches: list[CacheList],
        draft_cache: Optional[list] = None,
        *,
        clone_draft: bool = True,
    ) -> None:
        if self.max_bytes <= 0 or len(ids) < 32:
            return
        if ids in self._entries:
            self._entries.move_to_end(ids)
            return
        if self._est_trimmed_bytes(caches, draft_cache) > self.max_bytes:
            return
        snap = _clone_cache_list(caches)
        _eval_cache(snap)
        dsnap = None
        dnbytes = 0
        if draft_cache is not None:
            dsnap = clone_draft_cache(draft_cache) if clone_draft else draft_cache
            eval_draft_cache(dsnap)
            dnbytes = sum(cache_nbytes(c) for c in dsnap)
        nbytes = sum(cache_nbytes(c) for c in snap) + dnbytes
        if nbytes > self.max_bytes:
            return
        while self._entries and (
            self.total_bytes + nbytes > self.max_bytes or len(self._entries) >= self.max_entries
        ):
            _, (_, _, evicted_bytes) = self._entries.popitem(last=False)
            self.total_bytes -= evicted_bytes
        self._entries[ids] = (snap, dsnap, nbytes)
        self.total_bytes += nbytes

    def clear(self) -> None:
        self._entries.clear()
        self.total_bytes = 0


def sample_token(logits_1d: mx.array, temperature: float, top_p: float) -> int:
    if temperature <= 0.0:
        return int(mx.argmax(logits_1d, axis=-1).item())
    scaled = logits_1d.astype(mx.float32) / float(temperature)
    if 0.0 < top_p < 1.0:
        probs = mx.softmax(scaled, axis=-1)
        sorted_idx = mx.argsort(-probs)
        sorted_probs = probs[sorted_idx]
        cumprobs = mx.cumsum(sorted_probs, axis=-1)
        keep = (cumprobs - sorted_probs) < top_p
        masked = mx.where(keep, sorted_probs, 0.0)
        masked = masked / mx.sum(masked)
        chosen_pos = int(mx.random.categorical(mx.log(mx.maximum(masked, 1e-20))).item())
        return int(sorted_idx[chosen_pos].item())
    return int(mx.random.categorical(scaled).item())


def sample_batch_arr(logits: mx.array, temps: list[float], top_ps: list[float]) -> Optional[mx.array]:
    """Return (B,) int32 token IDs on GPU without host sync when uniform GPU sampling applies."""
    if all(t <= 0.0 for t in temps):
        return mx.argmax(logits, axis=-1).astype(mx.int32)
    if all(t > 0.0 for t in temps) and all(p <= 0.0 or p >= 1.0 for p in top_ps):
        inv_t = mx.array([1.0 / float(t) for t in temps], dtype=mx.float32)[:, None]
        return mx.random.categorical(logits.astype(mx.float32) * inv_t).astype(mx.int32)
    return None


def sample_batch_tokens(logits: mx.array, temps: list[float], top_ps: list[float]) -> list[int]:
    gpu_arr = sample_batch_arr(logits, temps, top_ps)
    if gpu_arr is not None:
        mx.eval(gpu_arr)
        return [int(v) for v in gpu_arr.tolist()]
    mx.eval(logits)
    return [sample_token(logits[i], temps[i], top_ps[i]) for i in range(len(temps))]


_DEFAULT_SAMPLE_TOKEN = sample_token
_DEFAULT_SAMPLE_BATCH_TOKENS = sample_batch_tokens


@dataclass
class GenRequest:
    prompt_ids: list[int]
    max_tokens: int
    temperature: float
    top_p: float
    stop_strings: list[str]
    use_prefix_cache: bool = True
    stream_queue: Optional[queue.Queue] = None
    done_event: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActiveSeq:
    req: GenRequest
    cache: Optional[list[CacheList]]
    parser: InklingStreamParser
    generated_ids: list[int] = field(default_factory=list)
    cached_tokens: int = 0
    prefill_pos: int = 0
    t_start: float = 0.0
    t_first_tok: Optional[float] = None
    next_token: Optional[int] = None
    finish_reason: Optional[str] = None
    prefix_saved: bool = False
    raw_text: str = ""
    pending_utf8_ids: list[int] = field(default_factory=list)
    draft_cache: Optional[list] = None
    draft_hidden: Optional[mx.array] = None
    mtp_k: int = 6
    mtp_steps: int = 0
    mtp_accepted: int = 0
    mtp_ema_acc: float = 4.5
    mtp_cooldown: int = 0
    mtp_cap_hist: list[mx.array] = field(default_factory=list)
    mtp_rng_seed: int = 0
    mtp_rng_step: int = 0
    mtp_hidden_buf: Optional[mx.array] = None


class InklingEngine:
    """Single-GPU-thread continuous-batching engine for Inkling 2 Mini."""

    def __init__(
        self,
        model: LanguageModel,
        tokenizer: Any,
        *,
        max_batch: int = 32,
        prefill_chunk: int = 512,
        prefix_cache_gb: float = 4.0,
        max_context: int = 262144,
        drafter: Optional[InklingMTPDrafter] = None,
        spec_draft_k: int = 0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch = max_batch
        self.prefill_chunk = prefill_chunk
        self.max_context = max_context
        self.drafter = drafter
        self.spec_draft_k = int(spec_draft_k)
        self.mtp_total_steps = 0
        self.mtp_total_accepted = 0
        self.user_message_token_id: int = 200000
        self.prefix_cache = PrefixCache(max_bytes=int(prefix_cache_gb * (1 << 30)))
        self.native_decode = try_create_native_decode_runner(model, max_m=32)
        if self.native_decode is not None and self.drafter is not None:
            if getattr(self.drafter, "_target", None) is model:
                self.native_decode.attach_mtp(self.drafter)

        self._q: queue.Queue[GenRequest] = queue.Queue()
        self._stop_flag = False
        self._worker = threading.Thread(target=self._run_loop, name="inkling-gpu-worker", daemon=True)
        self._worker.start()

    def submit(self, req: GenRequest) -> GenRequest:
        self._q.put(req)
        return req

    def stop(self) -> None:
        self._stop_flag = True
        self._worker.join(timeout=5.0)
        if self.native_decode is not None:
            self.native_decode.close()
            self.native_decode = None
        self.prefix_cache.clear()

    def _mtp_batch_ready(self) -> bool:
        return (
            self.drafter is not None
            and self.native_decode is not None
            and self.native_decode.mtp_attached
            and sample_token is _DEFAULT_SAMPLE_TOKEN
            and sample_batch_tokens is _DEFAULT_SAMPLE_BATCH_TOKENS
        )

    def _release_seq_bindings(self, seq: ActiveSeq) -> None:
        if self.native_decode is not None:
            self.native_decode.release_stream_cache(seq.cache, seq.draft_cache, seq.mtp_hidden_buf)

    def _merge_pool_to_batch(self, decoding: list[ActiveSeq]) -> list[CacheList]:
        """Merge unmerged active streams into one batched CacheList stack (`B >= 2` or plain fallback)."""
        if self.native_decode is not None:
            self.native_decode.clear_cache_binding()
        for s in decoding:
            s.draft_cache = None
            s.draft_hidden = None
            s.mtp_hidden_buf = None
            s.mtp_cap_hist.clear()
            s.mtp_cooldown = 0
        merged = _merge_single_caches([s.cache for s in decoding])
        for s in decoding:
            s.cache = None
        return merged

    def _drain_burst_queue(
        self, pending_prefill: list[ActiveSeq], max_admit: int, window_s: float = 0.008
    ) -> list[ActiveSeq]:
        """Briefly coalesce concurrent burst arrivals up to `max_admit` without starving latency."""
        deadline = time.perf_counter() + window_s
        while len(pending_prefill) < max_admit:
            rem_t = deadline - time.perf_counter()
            if rem_t <= 0:
                break
            try:
                req = self._q.get(timeout=min(rem_t, 0.002))
                pending_prefill.append(
                    ActiveSeq(req=req, cache=None, parser=InklingStreamParser(), t_start=time.perf_counter())
                )
            except queue.Empty:
                break
        take = min(max_admit, len(pending_prefill))
        return [pending_prefill.pop(0) for _ in range(take)]

    def _save_prompt_prefix(self, seq: ActiveSeq) -> None:
        """Snapshot full prompt KV + aligned draft KV into PrefixCache if within budget."""
        ids = seq.req.prompt_ids
        if (
            not seq.req.use_prefix_cache
            or len(ids) < 128
            or seq.cache is None
            or self.prefix_cache._est_trimmed_bytes(seq.cache, seq.draft_cache) > self.prefix_cache.max_bytes
        ):
            return
        dc_snap = None
        if self.drafter is not None and seq.draft_cache is not None and seq.draft_hidden is not None:
            dc_snap = clone_draft_cache(seq.draft_cache)
            self.drafter.prefill_context(seq.draft_hidden, dc_snap)
        self.prefix_cache.put(tuple(ids), seq.cache, draft_cache=dc_snap, clone_draft=False)

    def _enqueue_decode_step(
        self,
        tok_arr: mx.array,
        decoding: list[ActiveSeq],
        batch_cache: list[CacheList],
    ) -> tuple[Optional[mx.array], Optional[mx.array]]:
        temps = [s.req.temperature for s in decoding]
        top_ps = [s.req.top_p for s in decoding]
        use_fast_greedy = (
            sample_batch_tokens is _DEFAULT_SAMPLE_BATCH_TOKENS
            and all(t <= 0.0 for t in temps)
            and len(decoding) <= 8
        )
        if use_fast_greedy:
            out = self.model(tok_arr[:, None], cache=batch_cache, greedy_argmax=True)
            next_arr = out["argmax"][:, -1]
            logits = None
        else:
            out = self.model(tok_arr[:, None], cache=batch_cache)
            logits = out["logits"][:, -1, :]
            next_arr = (
                sample_batch_arr(logits, temps, top_ps)
                if sample_batch_tokens is _DEFAULT_SAMPLE_BATCH_TOKENS
                else None
            )
            if next_arr is None:
                return None, logits
        kv_offsets = [
            cl.caches[0].offset
            for cl in batch_cache
            if isinstance(getattr(cl.caches[0], "offset", None), mx.array)
        ]
        mx.async_eval(next_arr, *kv_offsets)
        return next_arr, logits

    def _run_loop(self) -> None:
        with mx.stream(mx.gpu):
            pending_prefill: list[ActiveSeq] = []
            decoding: list[ActiveSeq] = []
            batch_cache: Optional[list[CacheList]] = None
            inflight_arr: Optional[mx.array] = None

            while not self._stop_flag:
                timeout = 0.0 if (pending_prefill or decoding or inflight_arr is not None) else 0.05
                try:
                    while True:
                        req = self._q.get(timeout=timeout)
                        timeout = 0.0
                        pending_prefill.append(
                            ActiveSeq(
                                req=req,
                                cache=None,
                                parser=InklingStreamParser(),
                                t_start=time.perf_counter(),
                            )
                        )
                except queue.Empty:
                    pass

                if not pending_prefill and not decoding:
                    continue

                if pending_prefill and len(decoding) < self.max_batch and inflight_arr is None:
                    if not decoding and len(pending_prefill) == 1:
                        try:
                            req2 = self._q.get(timeout=0.002)
                            pending_prefill.append(
                                ActiveSeq(req=req2, cache=None, parser=InklingStreamParser(), t_start=time.perf_counter())
                            )
                        except queue.Empty:
                            pass
                    mtp_ready = self._mtp_batch_ready()
                    try:
                        if len(pending_prefill) >= 2:
                            can_admit_mtp = (
                                batch_cache is None
                                and mtp_ready
                                and (not decoding or all(s.draft_cache is not None and s.draft_hidden is not None for s in decoding))
                                and (len(decoding) + len(pending_prefill)) <= 4
                                and all(s.req.max_tokens >= 3 for s in pending_prefill)
                            )
                            if can_admit_mtp:
                                admitted = self._prefill_burst_mtp(pending_prefill, 4 - len(decoding))
                                decoding.extend(admitted)
                                if (
                                    len(decoding) >= 2
                                    and not all(s.draft_cache is not None and s.draft_hidden is not None for s in decoding)
                                ):
                                    batch_cache = self._merge_pool_to_batch(decoding)
                            else:
                                if batch_cache is None and decoding:
                                    batch_cache = self._merge_pool_to_batch(decoding)
                                cap = self.max_batch - len(decoding)
                                admitted_seqs, admitted_cache = self._prefill_burst(pending_prefill, cap)
                                if admitted_seqs and admitted_cache is not None:
                                    if not decoding:
                                        if len(admitted_seqs) == 1:
                                            admitted_seqs[0].cache = admitted_cache
                                            decoding.extend(admitted_seqs)
                                        else:
                                            decoding.extend(admitted_seqs)
                                            batch_cache = admitted_cache
                                    else:
                                        if len(admitted_seqs) == 1:
                                            admitted_cache = _merge_single_caches([admitted_cache])
                                        batch_cache = _extend_batch_cache(batch_cache, admitted_cache)
                                        for s in admitted_seqs:
                                            s.cache = None
                                            s.draft_cache = None
                                            s.draft_hidden = None
                                            s.mtp_hidden_buf = None
                                        decoding.extend(admitted_seqs)
                        else:
                            seq = pending_prefill[0]
                            done_prefill = self._step_prefill_chunk(seq)
                            if done_prefill:
                                pending_prefill.pop(0)
                                if seq.finish_reason is not None:
                                    self._release_seq_bindings(seq)
                                    self._finish_seq(seq)
                                else:
                                    if not decoding:
                                        decoding.append(seq)
                                    else:
                                        stay_mtp_pool = (
                                            batch_cache is None
                                            and mtp_ready
                                            and len(decoding) < 4
                                            and seq.draft_cache is not None
                                            and seq.draft_hidden is not None
                                            and all(s.draft_cache is not None and s.draft_hidden is not None for s in decoding)
                                        )
                                        if stay_mtp_pool:
                                            decoding.append(seq)
                                        else:
                                            if batch_cache is None:
                                                batch_cache = self._merge_pool_to_batch([*decoding, seq])
                                            else:
                                                seq_bcache = _merge_single_caches([seq.cache])
                                                batch_cache = _extend_batch_cache(batch_cache, seq_bcache)
                                                seq.cache = None
                                                seq.draft_cache = None
                                                seq.draft_hidden = None
                                                seq.mtp_hidden_buf = None
                                            decoding.append(seq)
                    except Exception as exc:
                        if pending_prefill:
                            bad = pending_prefill.pop(0)
                            self._release_seq_bindings(bad)
                            self._error_seq(bad, exc)
                    if not decoding:
                        continue

                if not self._q.empty() and len(decoding) < self.max_batch and inflight_arr is None:
                    continue

                can_step_now = (self._q.empty() or len(decoding) >= self.max_batch)
                if (
                    len(decoding) == 1
                    and batch_cache is None
                    and inflight_arr is None
                    and self.drafter is not None
                    and decoding[0].draft_cache is not None
                    and decoding[0].draft_hidden is not None
                    and can_step_now
                    and (decoding[0].req.max_tokens - len(decoding[0].generated_ids)) >= 3
                ):
                    seq = decoding[0]
                    try:
                        assert seq.cache is not None
                        if (
                            seq.mtp_cooldown > 0
                            and self.native_decode is not None
                            and seq.req.temperature <= 0.0
                            and sample_token is _DEFAULT_SAMPLE_TOKEN
                        ):
                            assert seq.next_token is not None
                            tok_id, cap1 = self.native_decode.step_m1_greedy_capture(
                                seq.next_token,
                                seq.cache,
                                self.drafter.config.target_layer_ids,
                            )
                            seq.mtp_cap_hist.append(cap1)
                            seq.mtp_cooldown -= 1
                            seq.next_token = tok_id
                            self._on_token(seq, tok_id)
                        else:
                            self._step_mtp_single(seq, seq.cache)
                        if seq.finish_reason is not None:
                            self._release_seq_bindings(seq)
                            self._finish_seq(seq)
                            decoding.clear()
                            batch_cache = None
                            if self.native_decode is not None:
                                self.native_decode.clear_cache_binding()
                            if self.drafter is not None and getattr(self.drafter, "_native_runner", None) is not None:
                                self.drafter._native_runner.clear_cache_binding()
                    except Exception as exc:
                        self._release_seq_bindings(seq)
                        self._error_seq(seq, exc)
                        decoding.clear()
                        batch_cache = None
                        if self.native_decode is not None:
                            self.native_decode.clear_cache_binding()
                        if self.drafter is not None and getattr(self.drafter, "_native_runner", None) is not None:
                            self.drafter._native_runner.clear_cache_binding()
                    continue

                if (
                    2 <= len(decoding) <= 4
                    and batch_cache is None
                    and inflight_arr is None
                    and self._mtp_batch_ready()
                    and can_step_now
                    and all(s.draft_cache is not None and s.draft_hidden is not None for s in decoding)
                    and any((s.req.max_tokens - len(s.generated_ids)) >= 3 for s in decoding)
                ):
                    try:
                        self._step_mtp_batched(decoding)
                        survivors: list[ActiveSeq] = []
                        for seq in decoding:
                            if seq.finish_reason is not None:
                                self._release_seq_bindings(seq)
                                self._finish_seq(seq)
                            else:
                                survivors.append(seq)
                        decoding = survivors
                        if not decoding:
                            if self.native_decode is not None:
                                self.native_decode.clear_cache_binding()
                    except Exception as exc:
                        for seq in decoding:
                            self._release_seq_bindings(seq)
                            self._error_seq(seq, exc)
                        decoding.clear()
                        batch_cache = None
                        if self.native_decode is not None:
                            self.native_decode.clear_cache_binding()
                    continue

                if decoding:
                    if len(decoding) == 1 and decoding[0].draft_cache is not None:
                        decoding[0].draft_cache = None
                        decoding[0].draft_hidden = None
                        decoding[0].mtp_hidden_buf = None
                        decoding[0].mtp_cap_hist.clear()
                        decoding[0].mtp_cooldown = 0
                    if (
                        len(decoding) == 1
                        and inflight_arr is None
                        and self.native_decode is not None
                        and sample_batch_tokens is _DEFAULT_SAMPLE_BATCH_TOKENS
                        and sample_token is _DEFAULT_SAMPLE_TOKEN
                    ):
                        seq = decoding[0]
                        try:
                            assert seq.cache is not None and seq.next_token is not None
                            if seq.req.temperature <= 0.0:
                                tok_id = self.native_decode.step_m1_greedy(seq.next_token, seq.cache)
                            else:
                                pre_n = self.native_decode.step_m1_prenorm(seq.next_token, seq.cache)
                                logits = self.model._logits_from_norm(self.model.model.norm(pre_n))[0, -1, :]
                                mx.eval(logits)
                                tok_id = sample_token(logits, seq.req.temperature, seq.req.top_p)
                            seq.next_token = tok_id
                            self._on_token(seq, tok_id)
                            if seq.finish_reason is not None:
                                self._release_seq_bindings(seq)
                                self._finish_seq(seq)
                                decoding.clear()
                                batch_cache = None
                                if self.native_decode is not None:
                                    self.native_decode.clear_cache_binding()
                        except Exception as exc:
                            self._release_seq_bindings(seq)
                            self._error_seq(seq, exc)
                            decoding.clear()
                            batch_cache = None
                            if self.native_decode is not None:
                                self.native_decode.clear_cache_binding()
                        continue
                    try:
                        if len(decoding) >= 2 and batch_cache is None:
                            batch_cache = self._merge_pool_to_batch(decoding)
                        active_cache = decoding[0].cache if len(decoding) == 1 else batch_cache
                        assert active_cache is not None
                        next_inflight: Optional[mx.array] = None
                        if inflight_arr is not None:
                            cur_arr = inflight_arr
                            inflight_arr = None
                        else:
                            cur_toks = mx.array([s.next_token for s in decoding], dtype=mx.int32)
                            cur_arr, raw_logits = self._enqueue_decode_step(cur_toks, decoding, active_cache)
                            if cur_arr is None:
                                temps = [s.req.temperature for s in decoding]
                                top_ps = [s.req.top_p for s in decoding]
                                next_ids = sample_batch_tokens(raw_logits, temps, top_ps)
                                cur_arr = None

                        if cur_arr is not None:
                            can_lookahead = (
                                not pending_prefill
                                and self._q.empty()
                                and all((s.req.max_tokens - len(s.generated_ids)) > 1 for s in decoding)
                            )
                            if can_lookahead:
                                next_inflight, _ = self._enqueue_decode_step(cur_arr, decoding, active_cache)
                            next_ids = [int(v) for v in cur_arr.tolist()]

                        keep_indices = []
                        finished_seqs: list[ActiveSeq] = []
                        for i, (seq, tok_id) in enumerate(zip(decoding, next_ids)):
                            seq.next_token = tok_id
                            self._on_token(seq, tok_id)
                            if seq.finish_reason is not None:
                                finished_seqs.append(seq)
                            else:
                                keep_indices.append(i)

                        if finished_seqs:
                            if next_inflight is not None:
                                mx.eval(next_inflight)
                            for fseq in finished_seqs:
                                self._release_seq_bindings(fseq)
                                self._finish_seq(fseq)
                            if not keep_indices:
                                decoding.clear()
                                batch_cache = None
                                inflight_arr = None
                            elif len(keep_indices) == 1 and len(decoding) > 1:
                                idx_arr = mx.array(keep_indices, dtype=mx.int32)
                                if next_inflight is not None:
                                    inflight_arr = mx.contiguous(next_inflight[idx_arr])
                                survivor = decoding[keep_indices[0]]
                                survivor.cache = _extract_row_cache(batch_cache, keep_indices[0])
                                batch_cache = None
                                decoding = [survivor]
                            else:
                                idx_arr = mx.array(keep_indices, dtype=mx.int32)
                                if next_inflight is not None:
                                    inflight_arr = mx.contiguous(next_inflight[idx_arr])
                                for cl in batch_cache:
                                    cl.filter(idx_arr)
                                decoding = [decoding[i] for i in keep_indices]
                        else:
                            inflight_arr = next_inflight
                    except Exception as exc:
                        inflight_arr = None
                        for s in decoding:
                            self._release_seq_bindings(s)
                            self._error_seq(s, exc)
                        decoding.clear()
                        batch_cache = None

    def _prepare_seq_mtp(self, seq: ActiveSeq, *, batched: bool = False, batch_size: int = 1) -> int:
        """Flush any deferred cooldown captures, allocate scratch buffers, and choose draft depth K."""
        drafter = self.drafter
        assert drafter is not None and seq.draft_cache is not None and seq.draft_hidden is not None
        if seq.mtp_cap_hist:
            h_all = mx.concatenate([seq.draft_hidden, *seq.mtp_cap_hist], axis=1)
            seq.mtp_cap_hist.clear()
            max_ctx = drafter.config.block_size
            if h_all.shape[1] > max_ctx:
                drafter.prefill_context(h_all[:, :-max_ctx, :], seq.draft_cache)
                eval_draft_cache(seq.draft_cache)
                h_all = h_all[:, -max_ctx:, :]
            seq.draft_hidden = mx.contiguous(h_all)
            mx.eval(seq.draft_hidden)
        seq.mtp_cooldown = 0
        rem = seq.req.max_tokens - len(seq.generated_ids)
        max_k = drafter.config.block_size - 1
        if batched and batch_size >= 3 and self.spec_draft_k <= 0:
            max_k = min(max_k, 5)
        k_req = self.spec_draft_k if 1 <= self.spec_draft_k <= max_k else seq.mtp_k
        k_cur = max(1, min(k_req, max(1, rem - 1), max_k)) if batched else min(k_req, rem - 1, max_k)
        if self.native_decode is not None and self.native_decode.mtp_attached:
            if seq.mtp_hidden_buf is None:
                n_cap = len(drafter.config.target_layer_ids)
                seq.mtp_hidden_buf = mx.zeros(
                    (1, drafter.config.block_size, n_cap * self.model.config.hidden_size),
                    dtype=mx.bfloat16,
                )
                mx.eval(seq.mtp_hidden_buf)
            if seq.req.temperature > 0.0 and seq.mtp_rng_seed == 0:
                seq.mtp_rng_seed = secrets.randbits(64) or 1
        return k_cur

    def _update_mtp_seq_after_step(
        self,
        seq: ActiveSeq,
        new_toks: list[int],
        next_hidden: mx.array,
        k_cur: int,
        *,
        allow_cooldown: bool,
    ) -> None:
        drafter = self.drafter
        assert drafter is not None
        max_k = drafter.config.block_size - 1
        n_acc = len(new_toks)
        n_match = n_acc - 1
        seq.draft_hidden = next_hidden
        seq.next_token = new_toks[-1]
        seq.mtp_steps += 1
        seq.mtp_accepted += n_acc
        seq.mtp_ema_acc = 0.70 * seq.mtp_ema_acc + 0.30 * float(n_acc)
        self.mtp_total_steps += 1
        self.mtp_total_accepted += n_acc

        if self.spec_draft_k <= 0:
            if n_match >= k_cur - 1:
                seq.mtp_k = min(max_k, seq.mtp_k + 2)
            elif n_match <= 1:
                seq.mtp_k = max(2, seq.mtp_k - 1)
                if (
                    allow_cooldown
                    and seq.req.temperature <= 0.0
                    and self.native_decode is not None
                    and seq.mtp_steps >= 6
                    and seq.mtp_ema_acc < 1.65
                ):
                    seq.mtp_cooldown = 6

        for tid in new_toks:
            self._on_token(seq, tid)
            if seq.finish_reason is not None:
                break

    def _step_mtp_single(self, seq: ActiveSeq, caches: list[CacheList]) -> None:
        """One built-in MTP speculative draft+verify+rollback cycle for a single stream."""
        drafter = self.drafter
        assert drafter is not None and seq.draft_cache is not None and seq.draft_hidden is not None
        k_cur = self._prepare_seq_mtp(seq, batched=False)
        cur = seq.next_token
        assert cur is not None

        if (
            self.native_decode is not None
            and self.native_decode.mtp_attached
            and sample_token is _DEFAULT_SAMPLE_TOKEN
        ):
            softcap = float(getattr(self.model.config, "final_logit_softcapping", 30.0) or 0.0)
            eff_top_p = float(seq.req.top_p if 0.0 < seq.req.top_p <= 1.0 else 1.0)
            new_toks, next_hidden = self.native_decode.step_mtp_single(
                cur,
                seq.draft_hidden,
                seq.draft_cache,
                caches,
                k_cur,
                temperature=seq.req.temperature,
                top_p=eff_top_p,
                logit_softcap=softcap,
                rng_seed=seq.mtp_rng_seed,
                rng_step=seq.mtp_rng_step,
                out_hidden_buf=seq.mtp_hidden_buf,
            )
            seq.mtp_rng_step += 1
        elif seq.req.temperature <= 0.0:
            if self.native_decode is not None:
                d_list = drafter.draft_greedy_list(cur, seq.draft_hidden, seq.draft_cache, k_cur)
                t_list, _, vcap = self.native_decode.step_verify(
                    [cur] + d_list, caches, drafter.config.target_layer_ids, greedy=True
                )
                assert t_list is not None
                n_match = 0
                while n_match < k_cur and d_list[n_match] == t_list[n_match]:
                    n_match += 1
                new_toks = d_list[:n_match] + [t_list[n_match]]
                n_acc = len(new_toks)
                self.native_decode.commit_verify(caches, n_acc, k_cur + 1)
                next_hidden = vcap[:, :n_acc, :]
            else:
                drafts = drafter.draft_greedy(cur, seq.draft_hidden, seq.draft_cache, k_cur)
                verify_in = mx.concatenate([mx.array([cur], mx.int32), drafts], axis=0)[None, :]
                snaps = snapshot_rotating_kv(caches, k_cur + 1)
                vo, vcap, vcs = fused.verify_step(
                    self.model.model,
                    self.model.model.embed(verify_in),
                    caches,
                    capture_layer_ids=drafter.config.target_layer_ids,
                )
                target_toks = self.model.greedy_argmax_from_pre_norm(vo)[0]
                mx.eval(drafts, target_toks, vcap)
                d_list = drafts.tolist()
                t_list = target_toks.tolist()
                n_match = 0
                while n_match < k_cur and d_list[n_match] == t_list[n_match]:
                    n_match += 1
                new_toks = d_list[:n_match] + [t_list[n_match]]
                n_acc = len(new_toks)
                rollback_single_cache(caches, snaps, vcs, n_acc, k_cur + 1)
                next_hidden = vcap[:, :n_acc, :]
        else:
            drafts, q_probs = drafter.draft_sampled(
                cur,
                seq.draft_hidden,
                seq.draft_cache,
                k_cur,
                seq.req.temperature,
                seq.req.top_p,
            )
            if self.native_decode is not None:
                mx.eval(drafts, q_probs)
                d_list = drafts.tolist()
                _, vo, vcap = self.native_decode.step_verify(
                    [cur] + d_list, caches, drafter.config.target_layer_ids, greedy=False
                )
                assert vo is not None
                vlg = self.model._logits_from_norm(self.model.model.norm(vo))[0]
                new_toks, _ = reject_sample_mtp(
                    d_list, q_probs, vlg, seq.req.temperature, seq.req.top_p
                )
                n_acc = len(new_toks)
                self.native_decode.commit_verify(caches, n_acc, k_cur + 1)
                next_hidden = vcap[:, :n_acc, :]
            else:
                verify_in = mx.concatenate([mx.array([cur], mx.int32), drafts], axis=0)[None, :]
                snaps = snapshot_rotating_kv(caches, k_cur + 1)
                vo, vcap, vcs = fused.verify_step(
                    self.model.model,
                    self.model.model.embed(verify_in),
                    caches,
                    capture_layer_ids=drafter.config.target_layer_ids,
                )
                vlg = self.model._logits_from_norm(self.model.model.norm(vo))[0]
                mx.eval(drafts, vcap)
                new_toks, _ = reject_sample_mtp(
                    drafts.tolist(), q_probs, vlg, seq.req.temperature, seq.req.top_p
                )
                n_acc = len(new_toks)
                rollback_single_cache(caches, snaps, vcs, n_acc, k_cur + 1)
                next_hidden = vcap[:, :n_acc, :]

        self._update_mtp_seq_after_step(seq, new_toks, next_hidden, k_cur, allow_cooldown=True)

    def _step_mtp_batched(self, seqs: list[ActiveSeq]) -> None:
        """One built-in MTP draft+verify+accept+commit cycle across B=2..4 unmerged streams in 1 CB."""
        assert self.native_decode is not None and self.drafter is not None
        softcap = float(getattr(self.model.config, "final_logit_softcapping", 30.0) or 0.0)
        items: list[dict[str, Any]] = []
        k_curs: list[int] = []
        for seq in seqs:
            k_cur = self._prepare_seq_mtp(seq, batched=True, batch_size=len(seqs))
            k_curs.append(k_cur)
            assert seq.cache is not None and seq.next_token is not None
            eff_top_p = float(seq.req.top_p if 0.0 < seq.req.top_p <= 1.0 else 1.0)
            items.append(
                {
                    "anchor_token": seq.next_token,
                    "draft_hidden": seq.draft_hidden,
                    "draft_cache": seq.draft_cache,
                    "caches": seq.cache,
                    "num_draft": k_cur,
                    "temperature": seq.req.temperature,
                    "top_p": eff_top_p,
                    "logit_softcap": softcap,
                    "rng_seed": seq.mtp_rng_seed,
                    "rng_step": seq.mtp_rng_step,
                    "out_hidden_buf": seq.mtp_hidden_buf,
                }
            )
        results = self.native_decode.step_mtp_batched(items)
        for seq, k_cur, (new_toks, next_hidden) in zip(seqs, k_curs, results):
            seq.mtp_rng_step += 1
            self._update_mtp_seq_after_step(
                seq, new_toks, next_hidden, k_cur, allow_cooldown=False
            )

    def _init_seq_cache(self, seq: ActiveSeq) -> None:
        if seq.cache is not None:
            return
        ids = seq.req.prompt_ids
        if seq.req.use_prefix_cache:
            hit_len, cached, dcached = self.prefix_cache.lookup(ids)
            if hit_len > 0 and cached is not None:
                seq.cache = cached
                seq.draft_cache = dcached if self.drafter is not None else None
                seq.cached_tokens = hit_len
                seq.prefill_pos = hit_len
                return
        seq.cache = None
        seq.draft_cache = None
        seq.prefill_pos = 0

    def _sample_burst_first_tokens(
        self, out: dict[str, Any], cohort: list[ActiveSeq], all_greedy: bool
    ) -> list[int]:
        if all_greedy:
            tok_arr = out["argmax"][:, -1]
            mx.eval(tok_arr)
            return [int(v) for v in tok_arr.tolist()]
        logits_all = out["logits"][:, -1, :]
        mx.eval(logits_all)
        return [sample_token(logits_all[i], s.req.temperature, s.req.top_p) for i, s in enumerate(cohort)]

    def _prefill_burst_mtp(self, pending_prefill: list[ActiveSeq], max_admit: int) -> list[ActiveSeq]:
        """Batch-prefill `2..4` requests WITH MTP layer capture, keeping unmerged per-stream caches."""
        seqs = self._drain_burst_queue(pending_prefill, min(4, max_admit))
        drafter = self.drafter
        assert drafter is not None
        capture_ids = list(drafter.config.target_layer_ids)
        try:
            for s in seqs:
                self._init_seq_cache(s)

            if self.prefix_cache.max_bytes > 0:
                by_sys: dict[tuple[int, ...], list[ActiveSeq]] = {}
                for s in seqs:
                    if s.cache is None and s.req.use_prefix_cache:
                        ids = s.req.prompt_ids
                        try:
                            upos = ids.index(self.user_message_token_id)
                        except ValueError:
                            upos = -1
                        if 128 <= upos < len(ids):
                            key = tuple(ids[:upos])
                            if key not in self.prefix_cache._entries:
                                by_sys.setdefault(key, []).append(s)
                for pkey, group in by_sys.items():
                    if len(group) < 2:
                        continue
                    p_cache = self.model.make_cache()
                    p_dcache = drafter.make_cache()
                    plen = len(pkey)
                    for pos in range(0, plen, self.prefill_chunk):
                        end = min(plen, pos + self.prefill_chunk)
                        out = self.model(
                            mx.array([list(pkey[pos:end])], dtype=mx.int32),
                            cache=p_cache,
                            skip_logits=True,
                            capture_layer_ids=capture_ids,
                        )
                        h_chunk = mx.concatenate(out["hidden_states"], axis=-1)
                        drafter.prefill_context(h_chunk, p_dcache, skip_local=False)
                        eval_draft_cache(p_dcache)
                        _eval_cache(p_cache)
                    if self.prefix_cache._est_trimmed_bytes(p_cache, p_dcache) <= self.prefix_cache.max_bytes:
                        self.prefix_cache.put(pkey, p_cache, draft_cache=p_dcache, clone_draft=False)
                        for s in group:
                            self._init_seq_cache(s)

            ready_seqs: list[ActiveSeq] = []
            cold = [s for s in seqs if s.cache is None and len(s.req.prompt_ids) >= 2]
            rest = [s for s in seqs if not (s.cache is None and len(s.req.prompt_ids) >= 2)]

            def _emit_and_extract_mtp(
                cohort: list[ActiveSeq],
                bcache: list[CacheList],
                h_last: mx.array,
                first_toks: list[int],
                local_ends: list[int],
                local_starts: list[int],
                has_draft_context: bool,
            ) -> None:
                for i, s in enumerate(cohort):
                    s.prefill_pos = len(s.req.prompt_ids)
                    s.next_token = first_toks[i]
                    self._on_token(s, first_toks[i])
                if has_draft_context:
                    for i, s in enumerate(cohort):
                        ls, le = local_starts[i], local_ends[i]
                        if ls < le:
                            drafter.prefill_context(h_last[i : i + 1, ls:le, :], s.draft_cache, skip_local=False)
                for i, s in enumerate(cohort):
                    s.draft_hidden = mx.contiguous(h_last[i : i + 1, -1:, :])
                _clear_conv_padding(bcache)
                _eval_cache(bcache)
                for i, s in enumerate(cohort):
                    s.cache = _extract_row_cache(bcache, i)
                    _eval_cache(s.cache)
                    if s.draft_cache is not None:
                        eval_draft_cache(s.draft_cache)
                    mx.eval(s.draft_hidden)
                    self._save_prompt_prefix(s)
                    ready_seqs.append(s)

            if len(cold) >= 2:
                lens = [len(s.req.prompt_ids) for s in cold]
                max_L = max(lens)
                left_pad = [max_L - l for l in lens]
                padded = [([0] * lp) + s.req.prompt_ids for s, lp in zip(cold, left_pad)]
                pf_cache = self.model.make_batch_cache(left_pad)
                for s in cold:
                    s.draft_cache = drafter.make_cache()
                chunk_sz = min(self.prefill_chunk, max(1, 4096 // len(cold)))
                all_greedy = sample_token is _DEFAULT_SAMPLE_TOKEN and all(s.req.temperature <= 0.0 for s in cold)
                for pos in range(0, max_L, chunk_sz):
                    end = min(max_L, pos + chunk_sz)
                    is_final = end == max_L
                    chunk_arr = mx.array([row[pos:end] for row in padded], dtype=mx.int32)
                    out = self.model(
                        chunk_arr,
                        cache=pf_cache,
                        skip_logits=(not is_final),
                        last_logits_only=is_final,
                        greedy_argmax=(is_final and all_greedy),
                        capture_layer_ids=capture_ids,
                    )
                    h_chunk = mx.concatenate(out["hidden_states"], axis=-1)
                    l_starts = [max(0, left_pad[i] - pos) for i in range(len(cold))]
                    l_ends = [max(0, min(end, max_L - 1) - pos) for _ in cold]
                    if not is_final:
                        skip_l = (max_L - end) > 1024
                        for i, s in enumerate(cold):
                            if l_starts[i] < l_ends[i]:
                                drafter.prefill_context(
                                    h_chunk[i : i + 1, l_starts[i] : l_ends[i], :],
                                    s.draft_cache,
                                    skip_local=skip_l,
                                )
                                eval_draft_cache(s.draft_cache)
                        _eval_cache(pf_cache)
                    else:
                        first_toks = self._sample_burst_first_tokens(out, cold, all_greedy)
                        _emit_and_extract_mtp(cold, pf_cache, h_chunk, first_toks, l_ends, l_starts, True)
            elif len(cold) == 1:
                rest.append(cold[0])

            by_rem: dict[int, list[ActiveSeq]] = {}
            singletons: list[ActiveSeq] = []
            for s in rest:
                rem = len(s.req.prompt_ids) - s.prefill_pos
                if s.cache is not None and s.draft_cache is not None and rem >= 2:
                    by_rem.setdefault(rem, []).append(s)
                else:
                    singletons.append(s)

            for rem, group in by_rem.items():
                if len(group) < 2:
                    singletons.extend(group)
                    continue
                w_cache = _merge_single_caches([s.cache for s in group])
                for s in group:
                    s.cache = None
                chunk_sz = min(self.prefill_chunk, max(1, 4096 // len(group)))
                all_greedy = sample_token is _DEFAULT_SAMPLE_TOKEN and all(s.req.temperature <= 0.0 for s in group)
                for off in range(0, rem, chunk_sz):
                    end_off = min(rem, off + chunk_sz)
                    is_final = end_off == rem
                    chunk_arr = mx.array(
                        [s.req.prompt_ids[s.prefill_pos + off : s.prefill_pos + end_off] for s in group],
                        dtype=mx.int32,
                    )
                    out = self.model(
                        chunk_arr,
                        cache=w_cache,
                        skip_logits=(not is_final),
                        last_logits_only=is_final,
                        greedy_argmax=(is_final and all_greedy),
                        capture_layer_ids=capture_ids,
                    )
                    h_chunk = mx.concatenate(out["hidden_states"], axis=-1)
                    c_len = end_off - off - (1 if is_final else 0)
                    if not is_final:
                        skip_l = (rem - end_off) > 1024
                        if c_len > 0:
                            for i, s in enumerate(group):
                                drafter.prefill_context(
                                    h_chunk[i : i + 1, :c_len, :],
                                    s.draft_cache,
                                    skip_local=skip_l,
                                )
                                eval_draft_cache(s.draft_cache)
                        _eval_cache(w_cache)
                    else:
                        first_toks = self._sample_burst_first_tokens(out, group, all_greedy)
                        _emit_and_extract_mtp(
                            group,
                            w_cache,
                            h_chunk,
                            first_toks,
                            [c_len] * len(group),
                            [0] * len(group),
                            c_len > 0,
                        )

            for s in singletons:
                while not self._step_prefill_chunk(s):
                    pass
                ready_seqs.append(s)

            unfinished: list[ActiveSeq] = []
            for s in ready_seqs:
                if s.finish_reason is not None:
                    self._release_seq_bindings(s)
                    self._finish_seq(s)
                else:
                    unfinished.append(s)
            return unfinished
        except Exception as exc:
            for s in seqs:
                self._release_seq_bindings(s)
                if s.req.result == {}:
                    self._error_seq(s, exc)
            return []

    def _prefill_burst(
        self, pending_prefill: list[ActiveSeq], max_admit: int
    ) -> tuple[list[ActiveSeq], Optional[list[CacheList]]]:
        """Batch-prefill up to `max_admit` pending requests up to L-1 so all enter decode together."""
        seqs = self._drain_burst_queue(pending_prefill, max_admit)
        try:
            for s in seqs:
                self._init_seq_cache(s)

            ready_seqs: list[ActiveSeq] = []
            combined_cache: Optional[list[CacheList]] = None

            def _append_batch(cohort_seqs: list[ActiveSeq], cohort_cache: list[CacheList]) -> None:
                nonlocal combined_cache
                _clear_conv_padding(cohort_cache)
                if combined_cache is None:
                    combined_cache = cohort_cache
                else:
                    combined_cache = _extend_batch_cache(combined_cache, cohort_cache)
                ready_seqs.extend(cohort_seqs)

            cold = [s for s in seqs if s.cache is None and len(s.req.prompt_ids) >= 2]
            warm = [s for s in seqs if not (s.cache is None and len(s.req.prompt_ids) >= 2)]

            idx = 0
            while idx < len(cold):
                first_len = len(cold[idx].req.prompt_ids) - 1
                max_cohort = min(8, max(2, 4096 // max(1, first_len))) if first_len <= 16384 else 2
                cohort = cold[idx : idx + max_cohort]
                idx += len(cohort)
                lens_m1 = [len(s.req.prompt_ids) - 1 for s in cohort]
                max_m1 = max(lens_m1)
                left_pad = [max_m1 - lm for lm in lens_m1]
                padded = [
                    ([0] * lp) + s.req.prompt_ids[:-1]
                    for s, lp in zip(cohort, left_pad)
                ]
                pf_cache = self.model.make_batch_cache(left_pad)
                k_c = len(cohort)
                chunk_sz = min(self.prefill_chunk, max(1, 4096 // k_c))
                for pos in range(0, max_m1, chunk_sz):
                    end = min(max_m1, pos + chunk_sz)
                    chunk_arr = mx.array([row[pos:end] for row in padded], dtype=mx.int32)
                    self.model(chunk_arr, cache=pf_cache, skip_logits=True)
                    _eval_cache(pf_cache)
                for s in cohort:
                    s.prefill_pos = len(s.req.prompt_ids) - 1
                    s.next_token = s.req.prompt_ids[-1]
                _append_batch(cohort, pf_cache)

            by_rem: dict[int, list[ActiveSeq]] = {}
            for s in warm:
                if s.cache is None:
                    s.cache = self.model.make_cache()
                rem = max(0, len(s.req.prompt_ids) - 1 - s.prefill_pos)
                by_rem.setdefault(rem, []).append(s)

            for rem, group in by_rem.items():
                g_idx = 0
                while g_idx < len(group):
                    max_w = 16 if rem <= 256 else 4
                    cohort = group[g_idx : g_idx + max_w]
                    g_idx += len(cohort)
                    w_cache = _merge_single_caches([s.cache for s in cohort])
                    for s in cohort:
                        s.cache = None
                    if rem > 0:
                        chunk_sz = min(self.prefill_chunk, max(1, 4096 // len(cohort)))
                        for off in range(0, rem, chunk_sz):
                            end_off = min(rem, off + chunk_sz)
                            chunk_arr = mx.array(
                                [s.req.prompt_ids[s.prefill_pos + off : s.prefill_pos + end_off] for s in cohort],
                                dtype=mx.int32,
                            )
                            self.model(chunk_arr, cache=w_cache, skip_logits=True)
                            _eval_cache(w_cache)
                    for s in cohort:
                        s.prefill_pos = len(s.req.prompt_ids) - 1
                        s.next_token = s.req.prompt_ids[-1]
                    _append_batch(cohort, w_cache)

            if len(ready_seqs) == 1 and combined_cache is not None:
                combined_cache = _extract_row_cache(combined_cache, 0)
                ready_seqs[0].cache = combined_cache
            return ready_seqs, combined_cache
        except Exception as exc:
            for s in seqs:
                if s.req.result == {}:
                    self._error_seq(s, exc)
            return [], None

    def _step_prefill_chunk(self, seq: ActiveSeq) -> bool:
        """Advance prefill by one chunk; returns True when prompt prefill completes and first token is sampled."""
        ids = seq.req.prompt_ids
        total = len(ids)
        use_mtp = self.drafter is not None and seq.req.max_tokens >= 3

        if seq.cache is None:
            if seq.req.use_prefix_cache:
                hit_len, cached, dcached = self.prefix_cache.lookup(ids)
                if hit_len > 0 and cached is not None:
                    seq.cache = cached
                    seq.draft_cache = dcached if use_mtp else None
                    seq.cached_tokens = hit_len
                    seq.prefill_pos = hit_len
            if seq.cache is None:
                seq.cache = self.model.make_cache()
                seq.draft_cache = self.drafter.make_cache() if use_mtp else None
                seq.prefill_pos = 0
                if total >= 4096 and mx.default_device() == mx.gpu:
                    cap = min(self.max_context, ((total + 256 + 2047) // 2048) * 2048)
                    kv_dtype = self.model.model.embed_tokens.weight.dtype
                    sched = self.model.model.schedule()
                    for (layer_idx, _, _), cl in zip(sched, seq.cache):
                        kv = cl.caches[0]
                        if isinstance(kv, GrowingKVCache) and kv.keys is None:
                            a = self.model.model.layers[layer_idx].self_attn
                            kv.keys = mx.zeros((1, a.n_kv, cap, a.head_dim), dtype=kv_dtype)
                            kv.values = mx.zeros((1, a.n_kv, cap, a.head_dim), dtype=kv_dtype)
                    if seq.draft_cache is not None and self.drafter is not None:
                        dcfg = self.drafter.config
                        for entry in seq.draft_cache:
                            if not isinstance(entry, RotatingKVCache) and entry.keys is None:
                                entry.keys = mx.zeros(
                                    (1, dcfg.num_key_value_heads, cap, dcfg.head_dim),
                                    dtype=kv_dtype,
                                )
                                entry.values = mx.zeros(
                                    (1, dcfg.num_key_value_heads, cap, dcfg.head_dim),
                                    dtype=kv_dtype,
                                )

        capture_ids = list(self.drafter.config.target_layer_ids) if (use_mtp and seq.draft_cache is not None) else None
        pos = seq.prefill_pos
        end = min(total, pos + self.prefill_chunk)
        split_for_prefix = False
        if not seq.prefix_saved and seq.req.use_prefix_cache and self.prefix_cache.max_bytes > 0:
            try:
                user_pos = ids.index(self.user_message_token_id)
                if 128 <= user_pos and pos < user_pos < total and user_pos <= end and tuple(ids[:user_pos]) not in self.prefix_cache._entries:
                    end = user_pos
                    split_for_prefix = True
            except ValueError:
                pass

        if end < total:
            chunk = mx.array([ids[pos:end]], dtype=mx.int32)
            out = self.model(chunk, cache=seq.cache, skip_logits=True, capture_layer_ids=capture_ids)
            if capture_ids is not None and out["hidden_states"] is not None:
                h_chunk = mx.concatenate(out["hidden_states"], axis=-1)
                self.drafter.prefill_context(
                    h_chunk,
                    seq.draft_cache,
                    skip_local=(not split_for_prefix and (total - end) > 1024),
                )
                eval_draft_cache(seq.draft_cache)
            _eval_cache(seq.cache)
            seq.prefill_pos = end
            if split_for_prefix and seq.req.use_prefix_cache:
                self.prefix_cache.put(tuple(ids[:end]), seq.cache, draft_cache=seq.draft_cache)
                seq.prefix_saved = True
            return False

        chunk = mx.array([ids[pos:total]], dtype=mx.int32)
        if seq.req.temperature <= 0.0 and sample_token is _DEFAULT_SAMPLE_TOKEN:
            out = self.model(
                chunk,
                cache=seq.cache,
                last_logits_only=True,
                greedy_argmax=True,
                capture_layer_ids=capture_ids,
            )
            tok_arr = out["argmax"][0, -1]
            mx.eval(tok_arr)
            first_tok = int(tok_arr.item())
        else:
            out = self.model(
                chunk,
                cache=seq.cache,
                last_logits_only=True,
                capture_layer_ids=capture_ids,
            )
            logits = out["logits"][0, -1, :]
            mx.eval(logits)
            first_tok = sample_token(logits, seq.req.temperature, seq.req.top_p)

        seq.prefill_pos = total
        seq.next_token = first_tok
        self._on_token(seq, first_tok)

        if capture_ids is not None and out["hidden_states"] is not None:
            h_chunk = mx.concatenate(out["hidden_states"], axis=-1)
            if h_chunk.shape[1] > 1:
                self.drafter.prefill_context(h_chunk[:, :-1, :], seq.draft_cache)
                eval_draft_cache(seq.draft_cache)
            seq.draft_hidden = mx.contiguous(h_chunk[:, -1:, :])
            mx.eval(seq.draft_hidden)

        self._save_prompt_prefix(seq)
        return True

    def _on_token(self, seq: ActiveSeq, tok_id: int) -> None:
        if seq.t_first_tok is None:
            seq.t_first_tok = time.perf_counter()

        seq.generated_ids.append(tok_id)
        if tok_id in STOP_IDS:
            seq.finish_reason = "stop"
            return

        if seq.pending_utf8_ids:
            seq.pending_utf8_ids.append(tok_id)
            piece = self.tokenizer.decode(seq.pending_utf8_ids, skip_special_tokens=False)
        else:
            piece = self.tokenizer.decode([tok_id], skip_special_tokens=False)

        if "\ufffd" in piece and len(seq.pending_utf8_ids) < 6:
            if not seq.pending_utf8_ids:
                seq.pending_utf8_ids.append(tok_id)
            if len(seq.pending_utf8_ids) < 6:
                return
        seq.pending_utf8_ids.clear()
        seq.raw_text += piece

        for s in seq.req.stop_strings:
            if s and s in seq.raw_text:
                seq.finish_reason = "stop"
                return

        events = seq.parser.feed(piece)
        if seq.req.stream_queue is not None:
            for ev in events:
                seq.req.stream_queue.put(("delta", ev))

        if seq.parser.stopped:
            seq.finish_reason = "stop"
        elif len(seq.generated_ids) >= seq.req.max_tokens:
            seq.finish_reason = "length"

    def _finish_seq(self, seq: ActiveSeq) -> None:
        t_end = time.perf_counter()
        if seq.pending_utf8_ids:
            piece = self.tokenizer.decode(seq.pending_utf8_ids, skip_special_tokens=False)
            seq.pending_utf8_ids.clear()
            seq.raw_text += piece
            for ev in seq.parser.feed(piece):
                if seq.req.stream_queue is not None:
                    seq.req.stream_queue.put(("delta", ev))
        tail_events = seq.parser.finish()
        if seq.req.stream_queue is not None:
            for ev in tail_events:
                seq.req.stream_queue.put(("delta", ev))

        finish = seq.finish_reason or "stop"
        if seq.parser.tool_calls and finish == "stop":
            finish = "tool_calls"

        ttft_s = (seq.t_first_tok or t_end) - seq.t_start
        decode_s = max(1e-6, t_end - (seq.t_first_tok or t_end))
        n_gen = len(seq.generated_ids)

        seq.req.result = {
            "content": seq.parser.content or None,
            "reasoning_content": seq.parser.reasoning_content or None,
            "tool_calls": seq.parser.tool_calls,
            "finish_reason": finish,
            "usage": {
                "prompt_tokens": len(seq.req.prompt_ids),
                "completion_tokens": n_gen,
                "total_tokens": len(seq.req.prompt_ids) + n_gen,
                "prompt_tokens_details": {"cached_tokens": seq.cached_tokens},
            },
            "metrics": {
                "ttft_ms": round(ttft_s * 1e3, 2),
                "decode_tok_s": round(max(0, n_gen - 1) / decode_s, 2) if n_gen > 1 else None,
                "mtp_steps": seq.mtp_steps,
                "mtp_accepted": seq.mtp_accepted,
                "mtp_mean_acc": round(seq.mtp_accepted / seq.mtp_steps, 3) if seq.mtp_steps > 0 else None,
            },
        }
        if _LOG_TURNS:
            print(
                f"[turn] prompt={len(seq.req.prompt_ids)} cached={seq.cached_tokens} gen={n_gen} "
                f"temp={seq.req.temperature} finish={finish} tools={len(seq.parser.tool_calls)} "
                f"tok/s={seq.req.result['metrics']['decode_tok_s']} acc={seq.req.result['metrics']['mtp_mean_acc']} "
                f"reason={repr((seq.parser.reasoning_content or '')[:80])} "
                f"content={repr((seq.parser.content or '')[:80])}",
                flush=True,
            )
        if seq.req.stream_queue is not None:
            seq.req.stream_queue.put(("done", seq.req.result))
        seq.req.done_event.set()

    def _error_seq(self, seq: ActiveSeq, exc: Exception) -> None:
        msg = f"{type(exc).__name__}: {exc}"
        seq.req.result = {"error": msg}
        if seq.req.stream_queue is not None:
            seq.req.stream_queue.put(("error", msg))
        seq.req.done_event.set()
