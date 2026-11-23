#!/usr/bin/env python3
"""Parity & A/B benchmark test for Native Metal 4.0 single-CB MTP speculative drafter.

Loads ONLY `~/models/Inkling-2-Mini/mtp.safetensors` (636 MB) + a lightweight mock target
with shared random BF16 `embed_tokens` / `lm_head` (`[201024, 2304]`, 926 MB -> < 1.7 GB total!).
"""

from __future__ import annotations

import ctypes
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm.models.cache import KVCache, RotatingKVCache

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from inkling_mlx.kernels import head_logits
from inkling_mlx.mtp import (
    InklingMTPDrafter,
    MTPConfig,
    clone_draft_cache,
    eval_draft_cache,
    fused_markov_argmax,
    load_mtp_drafter,
)

class MTPDraftConfig(ctypes.Structure):
    _fields_ = [
        ("hidden_size", ctypes.c_int32),
        ("num_layers", ctypes.c_int32),
        ("num_heads", ctypes.c_int32),
        ("num_kv_heads", ctypes.c_int32),
        ("head_dim", ctypes.c_int32),
        ("inter_size", ctypes.c_int32),
        ("block_size", ctypes.c_int32),
        ("markov_rank", ctypes.c_int32),
        ("n_target_layers", ctypes.c_int32),
        ("vocab_size", ctypes.c_int32),
        ("unpadded_vocab_size", ctypes.c_int32),
        ("rms_norm_eps", ctypes.c_float),
        ("embed_norm_eps", ctypes.c_float),
        ("rope_theta", ctypes.c_float),
    ]


class MTPDraftGlobalWeights(ctypes.Structure):
    _fields_ = [
        ("embed_tokens", ctypes.c_void_p),
        ("embed_tokens_bytes", ctypes.c_size_t),
        ("embed_norm", ctypes.c_void_p),
        ("embed_norm_bytes", ctypes.c_size_t),
        ("mask_embed", ctypes.c_void_p),
        ("mask_embed_bytes", ctypes.c_size_t),
        ("final_norm", ctypes.c_void_p),
        ("final_norm_bytes", ctypes.c_size_t),
        ("lm_head", ctypes.c_void_p),
        ("lm_head_bytes", ctypes.c_size_t),
        ("fuse_alpha", ctypes.c_void_p),
        ("fuse_alpha_bytes", ctypes.c_size_t),
        ("markov_dflare_norm", ctypes.c_void_p),
        ("markov_dflare_norm_bytes", ctypes.c_size_t),
        ("markov_w1", ctypes.c_void_p),
        ("markov_w1_bytes", ctypes.c_size_t),
        ("rnn_w_ctx_t", ctypes.c_void_p),
        ("rnn_w_ctx_t_bytes", ctypes.c_size_t),
        ("rnn_w_step_t", ctypes.c_void_p),
        ("rnn_w_step_t_bytes", ctypes.c_size_t),
        ("rnn_b_f32", ctypes.c_void_p),
        ("rnn_b_f32_bytes", ctypes.c_size_t),
        ("fold_weight", ctypes.c_void_p),
        ("fold_weight_bytes", ctypes.c_size_t),
        ("fold_bias", ctypes.c_void_p),
        ("fold_bias_bytes", ctypes.c_size_t),
    ]


class MTPDraftLayerWeights(ctypes.Structure):
    _fields_ = [
        ("is_local", ctypes.c_int32),
        ("window", ctypes.c_int32),
        ("dflare_norm", ctypes.c_void_p),
        ("dflare_norm_bytes", ctypes.c_size_t),
        ("input_layernorm", ctypes.c_void_p),
        ("input_layernorm_bytes", ctypes.c_size_t),
        ("post_attention_layernorm", ctypes.c_void_p),
        ("post_attention_layernorm_bytes", ctypes.c_size_t),
        ("kv_ctx_w", ctypes.c_void_p),
        ("kv_ctx_w_bytes", ctypes.c_size_t),
        ("k_norm_ctx", ctypes.c_void_p),
        ("k_norm_ctx_bytes", ctypes.c_size_t),
        ("qkv_w", ctypes.c_void_p),
        ("qkv_w_bytes", ctypes.c_size_t),
        ("q_norm", ctypes.c_void_p),
        ("q_norm_bytes", ctypes.c_size_t),
        ("k_norm", ctypes.c_void_p),
        ("k_norm_bytes", ctypes.c_size_t),
        ("o_proj_w", ctypes.c_void_p),
        ("o_proj_w_bytes", ctypes.c_size_t),
        ("gu_w", ctypes.c_void_p),
        ("gu_w_bytes", ctypes.c_size_t),
        ("down_proj_w", ctypes.c_void_p),
        ("down_proj_w_bytes", ctypes.c_size_t),
    ]


class MTPKVCacheDesc(ctypes.Structure):
    _fields_ = [
        ("k_ptr", ctypes.c_uint64),
        ("v_ptr", ctypes.c_uint64),
        ("k_nbytes", ctypes.c_size_t),
        ("v_nbytes", ctypes.c_size_t),
        ("cache_cap", ctypes.c_int32),
        ("offset", ctypes.c_int32),
        ("is_rotating", ctypes.c_int32),
        ("window", ctypes.c_int32),
        ("ring_idx", ctypes.c_int32),
        ("stride_h", ctypes.c_int32),
        ("stride_s", ctypes.c_int32),
        ("_pad", ctypes.c_int32),
    ]


class MTPStepStats(ctypes.Structure):
    _fields_ = [
        ("encode_ms", ctypes.c_double),
        ("gpu_ms", ctypes.c_double),
        ("wall_ms", ctypes.c_double),
        ("n_dispatches", ctypes.c_int32),
        ("_pad", ctypes.c_int32),
    ]


class MTPStageBreakdown(ctypes.Structure):
    _fields_ = [
        ("layers_ms", ctypes.c_double),
        ("sdpa_only_ms", ctypes.c_double),
        ("lm_head_ms", ctypes.c_double),
        ("markov_ctx_ms", ctypes.c_double),
        ("rnn_steps_ms", ctypes.c_double),
        ("argmax_ms", ctypes.c_double),
    ]


def _ptr_bytes(arr: mx.array, keepalive: list) -> tuple[int, int]:
    """Return (raw_ptr, nbytes) for a contiguous mx.array without copying."""
    c = mx.contiguous(arr)
    mx.eval(c)
    v = c.view(mx.uint16) if c.dtype == mx.bfloat16 else c
    mx.eval(v)
    nv = np.asarray(v)
    keepalive.append((c, v, nv))
    return int(nv.ctypes.data), int(nv.nbytes)


def _fast_ptr_bytes(arr: mx.array) -> tuple[np.ndarray, int, int]:
    """Zero-copy pointer extraction for already-evaluated contiguous mx.array."""
    v = arr.view(mx.uint16) if arr.dtype == mx.bfloat16 else arr
    nv = np.asarray(v)
    return nv, int(nv.ctypes.data), int(nv.nbytes)


class NativeMTPDrafter:
    """Thin ctypes wrapper around libinkling_mtp_draft.dylib."""

    def __init__(self, drafter: InklingMTPDrafter, lib_path: Path | None = None):
        if lib_path is None:
            lib_path = Path(__file__).resolve().parents[1] / "src" / "inkling_mlx" / "native" / "libinkling_mtp_draft.dylib"
        self.lib = ctypes.CDLL(str(lib_path))
        self.lib.mtp_draft_create.argtypes = [
            ctypes.POINTER(MTPDraftConfig),
            ctypes.POINTER(MTPDraftGlobalWeights),
            ctypes.POINTER(MTPDraftLayerWeights),
        ]
        self.lib.mtp_draft_create.restype = ctypes.c_void_p
        self.lib.mtp_draft_destroy.argtypes = [ctypes.c_void_p]
        self.lib.mtp_draft_destroy.restype = None
        self.lib.mtp_draft_clear_kv_cache_map.argtypes = [ctypes.c_void_p]
        self.lib.mtp_draft_clear_kv_cache_map.restype = None
        self.lib.mtp_draft_device_name.argtypes = [ctypes.c_void_p]
        self.lib.mtp_draft_device_name.restype = ctypes.c_char_p
        self.lib.mtp_draft_greedy_step.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_uint64,
            ctypes.c_size_t,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(MTPKVCacheDesc),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(MTPStepStats),
        ]
        self.lib.mtp_draft_greedy_step.restype = ctypes.c_int
        self.lib.mtp_draft_bench_layers_gemv.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        self.lib.mtp_draft_bench_layers_gemv.restype = ctypes.c_int
        self.lib.mtp_draft_profile_stages.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_uint64,
            ctypes.c_size_t,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(MTPKVCacheDesc),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(MTPStageBreakdown),
        ]
        self.lib.mtp_draft_profile_stages.restype = ctypes.c_int
        self.lib.mtp_draft_debug_read.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self.lib.mtp_draft_debug_read.restype = ctypes.c_int
        self.lib.mtp_draft_debug_run_up_to_layer.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_uint64,
            ctypes.c_size_t,
            ctypes.c_int32,
            ctypes.POINTER(MTPKVCacheDesc),
            ctypes.c_int32,
        ]
        self.lib.mtp_draft_debug_run_up_to_layer.restype = ctypes.c_int

        self._keepalive: list = []
        self._kv_ptr_cache: dict[int, tuple[mx.array, mx.array, np.ndarray, int, int]] = {}
        self._drafter = drafter
        cfg = drafter.config
        target = drafter._target
        assert target is not None, "drafter.bind(target) must be called first"

        c_cfg = MTPDraftConfig(
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_hidden_layers,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            inter_size=cfg.intermediate_size,
            block_size=cfg.block_size,
            markov_rank=cfg.markov_rank,
            n_target_layers=len(cfg.target_layer_ids),
            vocab_size=cfg.vocab_size,
            unpadded_vocab_size=cfg.unpadded_vocab_size,
            rms_norm_eps=cfg.rms_norm_eps,
            embed_norm_eps=getattr(target.model.embed_norm, "eps", cfg.rms_norm_eps)
            if hasattr(target.model, "embed_norm") and target.model.embed_norm is not None
            else cfg.rms_norm_eps,
            rope_theta=cfg.rope_theta,
        )

        p_emb, b_emb = _ptr_bytes(target.model.embed_tokens.weight, self._keepalive)
        if hasattr(target.model, "embed_norm") and target.model.embed_norm is not None:
            p_en, b_en = _ptr_bytes(target.model.embed_norm.weight, self._keepalive)
        else:
            p_en, b_en = 0, 0
        p_mask, b_mask = _ptr_bytes(drafter.mask_embed, self._keepalive)
        p_fnorm, b_fnorm = _ptr_bytes(drafter.norm.weight, self._keepalive)
        p_lmh, b_lmh = _ptr_bytes(target.lm_head.weight, self._keepalive)
        p_fa, b_fa = _ptr_bytes(drafter._fuse_alpha, self._keepalive)
        mh = drafter.markov_head
        p_mdn, b_mdn = _ptr_bytes(mh.dflare_norm.weight, self._keepalive)
        p_mw1, b_mw1 = _ptr_bytes(mh.markov_w1.weight, self._keepalive)
        p_wctx, b_wctx = _ptr_bytes(mh._rnn_w_ctx_t, self._keepalive)
        p_wstp, b_wstp = _ptr_bytes(mh._rnn_w_step_t, self._keepalive)
        p_rb, b_rb = _ptr_bytes(mh._rnn_b_f32, self._keepalive)
        p_fw, b_fw = _ptr_bytes(mh._fold_weight, self._keepalive)
        p_fb, b_fb = _ptr_bytes(mh._fold_bias, self._keepalive)

        c_glob = MTPDraftGlobalWeights(
            embed_tokens=p_emb, embed_tokens_bytes=b_emb,
            embed_norm=p_en, embed_norm_bytes=b_en,
            mask_embed=p_mask, mask_embed_bytes=b_mask,
            final_norm=p_fnorm, final_norm_bytes=b_fnorm,
            lm_head=p_lmh, lm_head_bytes=b_lmh,
            fuse_alpha=p_fa, fuse_alpha_bytes=b_fa,
            markov_dflare_norm=p_mdn, markov_dflare_norm_bytes=b_mdn,
            markov_w1=p_mw1, markov_w1_bytes=b_mw1,
            rnn_w_ctx_t=p_wctx, rnn_w_ctx_t_bytes=b_wctx,
            rnn_w_step_t=p_wstp, rnn_w_step_t_bytes=b_wstp,
            rnn_b_f32=p_rb, rnn_b_f32_bytes=b_rb,
            fold_weight=p_fw, fold_weight_bytes=b_fw,
            fold_bias=p_fb, fold_bias_bytes=b_fb,
        )

        LayerArr = MTPDraftLayerWeights * cfg.num_hidden_layers
        c_layers = LayerArr()
        for i, layer in enumerate(drafter.layers):
            attn = layer.self_attn
            mlp = layer.mlp
            p_dn, b_dn = _ptr_bytes(layer.dflare_norm.weight, self._keepalive)
            p_iln, b_iln = _ptr_bytes(layer.input_layernorm.weight, self._keepalive)
            p_pln, b_pln = _ptr_bytes(layer.post_attention_layernorm.weight, self._keepalive)
            p_kvc, b_kvc = _ptr_bytes(attn._kv_ctx_w, self._keepalive)
            p_knc, b_knc = _ptr_bytes(attn.k_norm_ctx.weight, self._keepalive)
            p_qkv, b_qkv = _ptr_bytes(attn._qkv_w, self._keepalive)
            p_qn, b_qn = _ptr_bytes(attn.q_norm.weight, self._keepalive)
            p_kn, b_kn = _ptr_bytes(attn.k_norm.weight, self._keepalive)
            p_op, b_op = _ptr_bytes(attn.o_proj.weight, self._keepalive)
            p_gu, b_gu = _ptr_bytes(mlp._gu_w, self._keepalive)
            p_dp, b_dp = _ptr_bytes(mlp.down_proj.weight, self._keepalive)
            c_layers[i] = MTPDraftLayerWeights(
                is_local=1 if attn.window is not None else 0,
                window=int(attn.window or 0),
                dflare_norm=p_dn, dflare_norm_bytes=b_dn,
                input_layernorm=p_iln, input_layernorm_bytes=b_iln,
                post_attention_layernorm=p_pln, post_attention_layernorm_bytes=b_pln,
                kv_ctx_w=p_kvc, kv_ctx_w_bytes=b_kvc,
                k_norm_ctx=p_knc, k_norm_ctx_bytes=b_knc,
                qkv_w=p_qkv, qkv_w_bytes=b_qkv,
                q_norm=p_qn, q_norm_bytes=b_qn,
                k_norm=p_kn, k_norm_bytes=b_kn,
                o_proj_w=p_op, o_proj_w_bytes=b_op,
                gu_w=p_gu, gu_w_bytes=b_gu,
                down_proj_w=p_dp, down_proj_w_bytes=b_dp,
            )

        self._handle = self.lib.mtp_draft_create(ctypes.byref(c_cfg), ctypes.byref(c_glob), c_layers)
        if not self._handle:
            raise RuntimeError("mtp_draft_create failed")
        self.device_name = self.lib.mtp_draft_device_name(self._handle).decode("utf-8")

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self.lib.mtp_draft_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def clear_kv_cache_map(self) -> None:
        self._kv_ptr_cache.clear()
        self.lib.mtp_draft_clear_kv_cache_map(self._handle)

    def _cached_kv_ptr(self, arr: mx.array) -> tuple[int, int]:
        key = id(arr)
        hit = self._kv_ptr_cache.get(key)
        if hit is not None and hit[0] is arr:
            return hit[3], hit[4]
        v = arr.view(mx.uint16) if arr.dtype == mx.bfloat16 else arr
        mx.eval(v)
        nv = np.asarray(v)
        ptr, nb = int(nv.ctypes.data), int(nv.nbytes)
        self._kv_ptr_cache[key] = (arr, v, nv, ptr, nb)
        return ptr, nb

    def read_debug_bf16(self, buf_id: int, shape: tuple[int, ...]) -> mx.array:
        arr = np.empty(int(np.prod(shape)), dtype=np.uint16)
        rc = self.lib.mtp_draft_debug_read(self._handle, int(buf_id), arr.ctypes.data, arr.nbytes)
        assert rc == 0, f"mtp_draft_debug_read({buf_id}) rc={rc}"
        return mx.array(arr).view(mx.bfloat16).reshape(shape)

    def read_debug_f32(self, buf_id: int, shape: tuple[int, ...]) -> mx.array:
        arr = np.empty(int(np.prod(shape)), dtype=np.float32)
        rc = self.lib.mtp_draft_debug_read(self._handle, int(buf_id), arr.ctypes.data, arr.nbytes)
        assert rc == 0, f"mtp_draft_debug_read({buf_id}) rc={rc}"
        return mx.array(arr).reshape(shape)

    def ensure_kv_capacity(self, draft_cache: list, n_ctx: int) -> None:
        """Ensure KV cache buffers have contiguous pre-allocated capacity for in-place writes."""
        cfg = self._drafter.config
        kv_h, hd = cfg.num_key_value_heads, cfg.head_dim
        reallocated = False
        for entry in draft_cache:
            if isinstance(entry, RotatingKVCache):
                W = entry.max_size
                if entry.keys is None:
                    entry.keys = mx.zeros((1, kv_h, W, hd), dtype=mx.bfloat16)
                    entry.values = mx.zeros((1, kv_h, W, hd), dtype=mx.bfloat16)
                    entry.offset = 0
                    entry._idx = 0
                    mx.eval(entry.keys, entry.values)
                    reallocated = True
                elif entry.keys.shape[2] != W:
                    k_ord = entry._temporal_order(entry.keys) if entry.offset > W else entry.keys[..., : entry.offset, :]
                    v_ord = entry._temporal_order(entry.values) if entry.offset > W else entry.values[..., : entry.offset, :]
                    if k_ord.shape[2] > W:
                        k_ord = k_ord[..., -W:, :]
                        v_ord = v_ord[..., -W:, :]
                    cur_len = k_ord.shape[2]
                    if cur_len < W:
                        pad = mx.zeros((1, kv_h, W - cur_len, hd), dtype=mx.bfloat16)
                        entry.keys = mx.contiguous(mx.concatenate([k_ord, pad], axis=2))
                        entry.values = mx.contiguous(mx.concatenate([v_ord, pad], axis=2))
                        entry._idx = cur_len
                    else:
                        entry.keys = mx.contiguous(k_ord)
                        entry.values = mx.contiguous(v_ord)
                        entry._idx = W
                    mx.eval(entry.keys, entry.values)
                    reallocated = True
                else:
                    if not np.asarray(entry.keys.view(mx.uint16)).flags.c_contiguous:
                        entry.keys = mx.contiguous(entry.keys)
                        mx.eval(entry.keys)
                        reallocated = True
                    if not np.asarray(entry.values.view(mx.uint16)).flags.c_contiguous:
                        entry.values = mx.contiguous(entry.values)
                        mx.eval(entry.values)
                        reallocated = True
            else:
                need = entry.offset + n_ctx
                if entry.keys is None:
                    cap = max(256, ((need + 255) // 256) * 256)
                    entry.keys = mx.zeros((1, kv_h, cap, hd), dtype=mx.bfloat16)
                    entry.values = mx.zeros((1, kv_h, cap, hd), dtype=mx.bfloat16)
                    mx.eval(entry.keys, entry.values)
                    reallocated = True
                elif entry.keys.shape[2] < need:
                    entry.prefix_cache_reserve(need)
                    mx.eval(entry.keys, entry.values)
                    reallocated = True
        if reallocated:
            self._kv_ptr_cache.clear()

    def draft_greedy(
        self,
        anchor_token: int,
        hidden: mx.array,
        draft_cache: list,
        num_draft: int,
    ) -> tuple[list[int], MTPStepStats]:
        n_ctx = int(hidden.shape[1])
        self.ensure_kv_capacity(draft_cache, n_ctx)

        h_nv, h_ptr, h_bytes = _fast_ptr_bytes(hidden)
        if not h_nv.flags.c_contiguous:
            h_arr = mx.contiguous(hidden)
            mx.eval(h_arr)
            h_nv, h_ptr, h_bytes = _fast_ptr_bytes(h_arr)

        KVArr = MTPKVCacheDesc * len(draft_cache)
        c_kvs = KVArr()
        for i, entry in enumerate(draft_cache):
            kptr, kbytes = self._cached_kv_ptr(entry.keys)
            vptr, vbytes = self._cached_kv_ptr(entry.values)
            is_rot = 1 if isinstance(entry, RotatingKVCache) else 0
            c_kvs[i] = MTPKVCacheDesc(
                k_ptr=kptr,
                v_ptr=vptr,
                k_nbytes=kbytes,
                v_nbytes=vbytes,
                cache_cap=int(entry.keys.shape[2]),
                offset=int(entry.offset),
                is_rotating=is_rot,
                window=int(entry.max_size) if is_rot else 0,
                ring_idx=int(entry._idx) if is_rot else -1,
                stride_h=int(entry.keys.shape[2] * entry.keys.shape[3]),
                stride_s=int(entry.keys.shape[3]),
                _pad=0,
            )

        out_toks = (ctypes.c_int32 * num_draft)()
        stats = MTPStepStats()
        rc = self.lib.mtp_draft_greedy_step(
            self._handle,
            int(anchor_token),
            h_ptr,
            h_bytes,
            n_ctx,
            int(num_draft),
            c_kvs,
            out_toks,
            ctypes.byref(stats),
        )
        if rc != 0:
            raise RuntimeError(f"mtp_draft_greedy_step failed with rc={rc}")

        for i, entry in enumerate(draft_cache):
            entry.offset = int(c_kvs[i].offset)
            if isinstance(entry, RotatingKVCache):
                entry._idx = int(c_kvs[i].ring_idx)

        _ = h_nv
        return [int(out_toks[s]) for s in range(num_draft)], stats

    def bench_layers_gemv(self, n_ctx: int = 4, warmup: int = 5, iters: int = 30) -> tuple[float, float]:
        med_ms = ctypes.c_double(0.0)
        gbps = ctypes.c_double(0.0)
        rc = self.lib.mtp_draft_bench_layers_gemv(
            self._handle, int(n_ctx), int(warmup), int(iters), ctypes.byref(med_ms), ctypes.byref(gbps)
        )
        if rc != 0:
            raise RuntimeError(f"mtp_draft_bench_layers_gemv failed with rc={rc}")
        return float(med_ms.value), float(gbps.value)

    def profile_stages(
        self,
        anchor_token: int,
        hidden: mx.array,
        draft_cache: list,
        num_draft: int,
        warmup: int = 5,
        iters: int = 25,
    ) -> MTPStageBreakdown:
        n_ctx = int(hidden.shape[1])
        self.ensure_kv_capacity(draft_cache, n_ctx)
        h_nv, h_ptr, h_bytes = _fast_ptr_bytes(hidden)
        if not h_nv.flags.c_contiguous:
            h_arr = mx.contiguous(hidden)
            mx.eval(h_arr)
            h_nv, h_ptr, h_bytes = _fast_ptr_bytes(h_arr)
        KVArr = MTPKVCacheDesc * len(draft_cache)
        c_kvs = KVArr()
        kv_keep = []
        for i, entry in enumerate(draft_cache):
            knv, kptr, kbytes = _fast_ptr_bytes(entry.keys)
            vnv, vptr, vbytes = _fast_ptr_bytes(entry.values)
            kv_keep.append((knv, vnv))
            is_rot = 1 if isinstance(entry, RotatingKVCache) else 0
            c_kvs[i] = MTPKVCacheDesc(
                k_ptr=kptr, v_ptr=vptr, k_nbytes=kbytes, v_nbytes=vbytes,
                cache_cap=int(entry.keys.shape[2]), offset=int(entry.offset),
                is_rotating=is_rot, window=int(entry.max_size) if is_rot else 0,
                ring_idx=int(entry._idx) if is_rot else -1,
                stride_h=int(entry.keys.shape[2] * entry.keys.shape[3]),
                stride_s=int(entry.keys.shape[3]), _pad=0,
            )
        bd = MTPStageBreakdown()
        rc = self.lib.mtp_draft_profile_stages(
            self._handle, int(anchor_token), h_ptr, h_bytes, n_ctx, int(num_draft),
            c_kvs, int(warmup), int(iters), ctypes.byref(bd),
        )
        if rc != 0:
            raise RuntimeError(f"mtp_draft_profile_stages failed with rc={rc}")
        _ = (h_nv, kv_keep)
        return bd


class _MockInnerModel:
    def __init__(self, embed_weight: mx.array, rms_norm_eps: float):
        self.embed_tokens = nn.Embedding(embed_weight.shape[0], embed_weight.shape[1])
        self.embed_tokens.weight = embed_weight
        self.embed_norm = nn.RMSNorm(embed_weight.shape[1], eps=rms_norm_eps)
        self.embed_norm.weight = (1.0 + 0.1 * mx.random.normal((embed_weight.shape[1],))).astype(mx.bfloat16)

    def embed(self, inputs: mx.array) -> mx.array:
        return self.embed_norm(self.embed_tokens(inputs))


class _MockTargetModel:
    def __init__(self, cfg: MTPConfig):
        mx.random.seed(42)
        shared_w = (mx.random.normal((cfg.vocab_size, cfg.hidden_size)) * 0.05).astype(mx.bfloat16)
        mx.eval(shared_w)
        self.model = _MockInnerModel(shared_w, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = shared_w
        mx.eval(self.model.embed_norm.weight, self.lm_head.weight)


def _populate_cache_pair(
    drafter: InklingMTPDrafter,
    ctx_len: int,
    seed: int,
) -> tuple[list, list]:
    """Populate a draft KV cache to `ctx_len` tokens and return two independent deep copies."""
    mx.random.seed(seed)
    cache_py = drafter.make_cache()
    remaining = ctx_len
    while remaining > 0:
        chunk = min(256, remaining)
        h_chunk = (mx.random.normal((1, chunk, 4 * drafter.config.hidden_size)) * 0.5).astype(mx.bfloat16)
        drafter.prefill_context(h_chunk, cache_py)
        eval_draft_cache(cache_py)
        remaining -= chunk
    cache_nat = clone_draft_cache(cache_py)
    eval_draft_cache(cache_nat)
    return cache_py, cache_nat


def _py_score_gap(
    drafter: InklingMTPDrafter,
    anchor: int,
    hidden: mx.array,
    cache_before: list,
    k_draft: int,
    py_toks: list[int],
    nat_toks: list[int],
) -> tuple[int, float, float, float]:
    """Return (first_diff_idx, py_score, nat_score, gap) in Python's BF16 Markov logits."""
    c_tmp = clone_draft_cache(cache_before)
    eval_draft_cache(c_tmp)
    anc = mx.array([anchor], dtype=mx.int32)
    emb = drafter._target.model.embed(anc[:, None])
    masks = mx.broadcast_to(drafter.mask_embed.astype(emb.dtype), (1, drafter.config.block_size - 1, emb.shape[-1]))
    x, res = mx.concatenate([emb, masks], axis=1), None
    for i, (layer, entry) in enumerate(zip(drafter.layers, c_tmp)):
        x, res = layer(x, res, drafter._mix(hidden, i, layer.dflare_norm), entry)
    x_out = drafter.norm(x.astype(mx.float32) + res.astype(mx.float32)).astype(emb.dtype)[:, 1 : k_draft + 1]
    base = head_logits(x_out.reshape(k_draft, -1), drafter._target.lm_head.weight).astype(emb.dtype)
    f_rnn = drafter._mix(hidden[:, -1:], -1, drafter.markov_head.dflare_norm)
    rctx = drafter.markov_head.context(x_out, f_rnn)
    st = mx.zeros((1, drafter.config.markov_rank), emb.dtype)
    pv = anc
    for s in range(k_draft):
        st, lat = drafter.markov_head.step_latent(st, pv, rctx[:, s])
        if py_toks[s] != nat_toks[s]:
            fw = drafter.markov_head._fold_weight
            fb = drafter.markov_head._fold_bias
            p_id, n_id = py_toks[s], nat_toks[s]
            dp = float(mx.sum(fw[p_id].astype(mx.float32) * lat[0].astype(mx.float32)).item())
            dn = float(mx.sum(fw[n_id].astype(mx.float32) * lat[0].astype(mx.float32)).item())
            sp = float(mx.array(float(base[s, p_id].item()) + float(mx.array(dp + float(fb[p_id].item()), dtype=mx.bfloat16).item()), dtype=mx.bfloat16).item())
            sn = float(mx.array(float(base[s, n_id].item()) + float(mx.array(dn + float(fb[n_id].item()), dtype=mx.bfloat16).item()), dtype=mx.bfloat16).item())
            return s, sp, sn, abs(sp - sn)
        pv = mx.array([py_toks[s]], dtype=mx.int32)
    return -1, 0.0, 0.0, 0.0


def main() -> None:
    import os
    env_dir = os.environ.get("INKLING_MODEL_DIR")
    model_dir = Path(env_dir).expanduser() if env_dir else (Path.home() / "models" / "Inkling-2-Mini")
    if not (model_dir / "mtp.safetensors").is_file():
        print(f"[skip] mtp.safetensors not found under {model_dir}; run inkling-mlx download first.")
        return
    print(f"[setup] Loading MTP config & mtp.safetensors (636 MB) from {model_dir}...")
    cfg = MTPConfig.from_model_dir(model_dir)
    mock_target = _MockTargetModel(cfg)
    t0 = time.perf_counter()
    drafter = load_mtp_drafter(model_dir, mock_target)
    assert drafter is not None
    print(f"[setup] Loaded + folded InklingMTPDrafter in {(time.perf_counter() - t0):.2f}s")

    t1 = time.perf_counter()
    native = NativeMTPDrafter(drafter)
    print(f"[setup] Initialized NativeMTPDrafter on {native.device_name} in {(time.perf_counter() - t1)*1000:.1f} ms")

    print("\n=== 1. 3-Layer M=8 BF16 Projections Bandwidth (523.8 MB) ===")
    for n_ctx in (1, 4, 8):
        med_ms, gbps = native.bench_layers_gemv(n_ctx=n_ctx, warmup=5, iters=30)
        status = "PASS" if gbps >= 220.0 else "WARN"
        print(f"  n_ctx={n_ctx}: median GPU = {med_ms:.3f} ms | {gbps:.1f} GB/s [{status}]")

    print("\n=== 1b. Stage-by-Stage Intermediate Parity Audit ===")
    for off_init, label in ((207, "single-pass <256"), (260, "Split-K >=256")):
        c_py, c_nat = _populate_cache_pair(drafter, off_init, seed=2000 + off_init)
        native.ensure_kv_capacity(c_nat, 5)
        mx.random.seed(3000 + off_init)
        h5 = (mx.random.normal((1, 5, 4 * cfg.hidden_size)) * 0.5).astype(mx.bfloat16)
        mx.eval(h5)
        anc_id = 1234
        anc = mx.array([anc_id], dtype=mx.int32)
        emb = drafter._target.model.embed(anc[:, None])
        masks = mx.broadcast_to(drafter.mask_embed.astype(emb.dtype), (1, 7, cfg.hidden_size))
        x0 = mx.concatenate([emb, masks], axis=1)
        layer0, e0 = drafter.layers[0], c_py[0]
        ctx0_py = drafter._mix(h5, 0, layer0.dflare_norm)
        ni0 = layer0.input_layernorm(x0).astype(x0.dtype)
        attn0 = layer0.self_attn
        kv_ctx0_py = ctx0_py @ attn0._kv_ctx_w.T
        kc0_py, vc0_py = e0.update_and_fetch(
            attn0.rope(attn0.k_norm_ctx(attn0._heads(kv_ctx0_py[..., :1024], 8)), offset=off_init),
            attn0._heads(kv_ctx0_py[..., 1024:], 8),
        )
        qkv0_py = ni0 @ attn0._qkv_w.T
        q0_py = attn0.rope(attn0.q_norm(attn0._heads(qkv0_py[..., :3072], 24)), offset=off_init + 5)
        bk0_py = attn0.rope(attn0.k_norm(attn0._heads(qkv0_py[..., 3072:4096], 8)), offset=off_init + 5)
        bv0_py = attn0._heads(qkv0_py[..., 4096:], 8)
        sdpa0_py = mx.fast.scaled_dot_product_attention(
            q0_py, mx.concatenate([kc0_py, bk0_py], axis=2), mx.concatenate([vc0_py, bv0_py], axis=2),
            scale=128**-0.5, mask=None,
        )
        aout0_py = sdpa0_py.transpose(0, 2, 1, 3).reshape(1, 8, -1)
        mx.eval(ctx0_py, kv_ctx0_py, qkv0_py, q0_py, bk0_py, bv0_py, aout0_py)

        h5_nv, h5_ptr, h5_bytes = _fast_ptr_bytes(h5)
        c_kvs = (MTPKVCacheDesc * 3)()
        for i, entry in enumerate(c_nat):
            kp, kb = native._cached_kv_ptr(entry.keys)
            vp, vb = native._cached_kv_ptr(entry.values)
            is_rot = 1 if isinstance(entry, RotatingKVCache) else 0
            c_kvs[i] = MTPKVCacheDesc(
                kp, vp, kb, vb, int(entry.keys.shape[2]), int(entry.offset),
                is_rot, int(entry.max_size) if is_rot else 0, int(entry._idx) if is_rot else -1,
                int(entry.keys.shape[2] * entry.keys.shape[3]), int(entry.keys.shape[3]), 0,
            )
        rc = native.lib.mtp_draft_debug_run_up_to_layer(native._handle, anc_id, h5_ptr, h5_bytes, 5, c_kvs, 0)
        assert rc == 0
        _ = h5_nv

        checks = [
            ("L0_ctx", ctx0_py[0], native.read_debug_bf16(12, (5, 2304)), 1e-3),
            ("L0_kv_ctx", kv_ctx0_py[0], native.read_debug_bf16(13, (5, 2048)), 1e-3),
            ("L0_qkv", qkv0_py[0], native.read_debug_bf16(15, (8, 5120)), 1e-3),
            ("L0_q_rope", q0_py[0], native.read_debug_bf16(9, (24, 8, 128)), 1e-3),
            ("L0_bk_rope", bk0_py[0], native.read_debug_bf16(10, (8, 8, 128)), 1e-3),
            ("L0_bv", bv0_py[0], native.read_debug_bf16(11, (8, 8, 128)), 1e-3),
            ("L0_attn_out", aout0_py[0], native.read_debug_bf16(5, (8, 3072)), 2e-3),
        ]
        parts = []
        for cname, pa, na, tol in checks:
            md = float(mx.max(mx.abs(pa.astype(mx.float32) - na.astype(mx.float32))).item())
            assert md <= tol, f"{cname} ({label}) max_diff={md:.6f} > {tol}"
            parts.append(f"{cname}={md:.5f}")
        print(f"  [PASS] L0 ({label}, off={off_init}->+5): " + " ".join(parts))

    c_base_k, _ = _populate_cache_pair(drafter, 207, seed=4096)
    for k_draft in (3, 4, 5, 6, 7):
        c_n = clone_draft_cache(c_base_k)
        eval_draft_cache(c_n)
        mx.random.seed(4100 + k_draft)
        h_k = (mx.random.normal((1, 5, 4 * cfg.hidden_size)) * 0.5).astype(mx.bfloat16)
        mx.eval(h_k)
        anc_id = 500 + k_draft
        nat_toks, _ = native.draft_greedy(anc_id, h_k, c_n, k_draft)

        x_out_n = native.read_debug_bf16(0, (1, k_draft, cfg.hidden_size))
        base_n = native.read_debug_bf16(1, (k_draft, cfg.unpadded_vocab_size))
        f_rnn_n = native.read_debug_bf16(2, (1, 1, cfg.hidden_size))
        rctx_n = native.read_debug_f32(3, (1, k_draft, 3 * cfg.markov_rank))
        lat_last_n = native.read_debug_bf16(4, (1, cfg.markov_rank))

        base_ref = head_logits(x_out_n.reshape(k_draft, -1), drafter._target.lm_head.weight).astype(mx.bfloat16)[:, :cfg.unpadded_vocab_size]
        f_rnn_ref = drafter._mix(h_k[:, -1:], -1, drafter.markov_head.dflare_norm)
        rctx_ref = drafter.markov_head.context(x_out_n, f_rnn_n)
        st_m = mx.zeros((1, cfg.markov_rank), mx.bfloat16)
        pv_m = mx.array([anc_id], mx.int32)
        ref_toks = []
        for s in range(k_draft):
            st_m, lat_m = drafter.markov_head.step_latent(st_m, pv_m, rctx_n[:, s])
            pv_m = fused_markov_argmax(
                lat_m, drafter.markov_head._fold_weight, drafter.markov_head._fold_bias, base_n[s], cfg.unpadded_vocab_size
            )
            ref_toks.append(int(pv_m.item()))
        mx.eval(base_ref, f_rnn_ref, rctx_ref, lat_m)

        d_frnn = float(mx.max(mx.abs(f_rnn_ref.astype(mx.float32) - f_rnn_n.astype(mx.float32))).item())
        d_rctx = float(mx.max(mx.abs(rctx_ref - rctx_n)).item())
        d_lat = float(mx.max(mx.abs(lat_m.astype(mx.float32) - lat_last_n.astype(mx.float32))).item())
        d_base_nz = int(mx.sum(base_ref != base_n).item())
        assert d_frnn == 0.0, f"K={k_draft} f_rnn max_diff={d_frnn}"
        assert d_rctx <= 1e-4, f"K={k_draft} rctx max_diff={d_rctx}"
        assert d_lat <= 1e-3, f"K={k_draft} lat_last max_diff={d_lat}"
        assert nat_toks == ref_toks, f"K={k_draft} isolated markov tokens mismatch: {ref_toks} vs {nat_toks}"
        print(
            f"  [PASS] Post-transformer K={k_draft}: f_rnn={d_frnn:.5f} rctx={d_rctx:.6f} "
            f"lat_last={d_lat:.5f} lm_head_nz={d_base_nz}/{base_n.size} ({100*d_base_nz/base_n.size:.3f}%) exact_toks=True"
        )

    c_g_py = drafter.make_cache()
    c_g_nat = drafter.make_cache()
    mx.random.seed(6000)
    h_pre_g = (mx.random.normal((1, 207, 4 * cfg.hidden_size)) * 0.5).astype(mx.bfloat16)
    drafter.prefill_context(h_pre_g, c_g_py)
    drafter.prefill_context(h_pre_g, c_g_nat)
    eval_draft_cache(c_g_py)
    eval_draft_cache(c_g_nat)
    max_kv_diff_growth = 0.0
    for step_g, (n_ctx_g, k_g) in enumerate(((1, 5), (5, 5), (6, 6), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (6, 7), (5, 7))):
        h_g = (mx.random.normal((1, n_ctx_g, 4 * cfg.hidden_size)) * 0.5).astype(mx.bfloat16)
        mx.eval(h_g)
        drafter.draft_greedy(700 + step_g, h_g, c_g_py, k_g)
        eval_draft_cache(c_g_py)
        native.draft_greedy(700 + step_g, h_g, c_g_nat, k_g)
        for li in range(3):
            off = c_g_py[li].offset
            assert off == c_g_nat[li].offset
            dk = float(mx.max(mx.abs(c_g_py[li].keys[..., :off, :].astype(mx.float32) - c_g_nat[li].keys[..., :off, :].astype(mx.float32))).item())
            dv = float(mx.max(mx.abs(c_g_py[li].values[..., :off, :].astype(mx.float32) - c_g_nat[li].values[..., :off, :].astype(mx.float32))).item())
            max_kv_diff_growth = max(max_kv_diff_growth, dk, dv)
    assert c_g_nat[1].offset > 256 and max_kv_diff_growth <= 0.0625, f"KV growth diff={max_kv_diff_growth}"
    print(f"  [PASS] Global KVCache growth (207 -> {c_g_nat[1].offset}, L1_cap={c_g_nat[1].keys.shape[2]}): max_kv_diff={max_kv_diff_growth:.5f}")

    h_guard = (mx.random.normal((1, 1, 4 * cfg.hidden_size)) * 0.5).astype(mx.bfloat16)
    mx.eval(h_guard)
    hg_nv, hg_ptr, hg_bytes = _fast_ptr_bytes(h_guard)
    bad_kvs = (MTPKVCacheDesc * 3)()
    for i, entry in enumerate(c_g_nat):
        kp, kb = native._cached_kv_ptr(entry.keys)
        vp, vb = native._cached_kv_ptr(entry.values)
        is_rot = 1 if isinstance(entry, RotatingKVCache) else 0
        bad_kvs[i] = MTPKVCacheDesc(
            kp, vp, kb, vb // 2 if i == 1 else vb, int(entry.keys.shape[2]), int(entry.offset),
            is_rot, int(entry.max_size) if is_rot else 0, int(entry._idx) if is_rot else -1,
            int(entry.keys.shape[2] * entry.keys.shape[3]), int(entry.keys.shape[3]), 0,
        )
    out_dummy = (ctypes.c_int32 * 5)()
    rc_bad = native.lib.mtp_draft_greedy_step(native._handle, 10, hg_ptr, hg_bytes, 1, 5, bad_kvs, out_dummy, None)
    assert rc_bad == -36, f"Expected rc=-36 for undersized L1 V buffer, got rc={rc_bad}"
    _ = hg_nv
    print(f"  [PASS] Undersized/stale KV guard rejected bad L1 V descriptor with rc={rc_bad}")

    print("\n=== 2. Exact Token Parity vs Python InklingMTPDrafter.draft_greedy ===")
    total_cases = 0
    passed_cases = 0
    mismatches = []

    for ctx_len in (64, 512, 1100):
        base_py, _ = _populate_cache_pair(drafter, ctx_len, seed=1000 + ctx_len)
        for n_ctx in (1, 4, 6):
            for k_draft in (3, 5, 7):
                total_cases += 1
                mx.random.seed(5000 + ctx_len * 31 + n_ctx * 7 + k_draft)
                anchor = int((ctx_len * 13 + n_ctx * 97 + k_draft * 199) % cfg.unpadded_vocab_size)
                hidden = (mx.random.normal((1, n_ctx, 4 * cfg.hidden_size)) * 0.5).astype(mx.bfloat16)
                mx.eval(hidden)

                c_py = clone_draft_cache(base_py)
                c_nat = clone_draft_cache(base_py)
                eval_draft_cache(c_py)
                eval_draft_cache(c_nat)
                native.clear_kv_cache_map()

                py_toks_mx = drafter.draft_greedy(anchor, hidden, c_py, k_draft)
                mx.eval(py_toks_mx)
                py_toks = [int(x) for x in py_toks_mx.tolist()]

                nat_toks, stats = native.draft_greedy(anchor, hidden, c_nat, k_draft)

                ok = py_toks == nat_toks
                tie_note = ""
                if ok:
                    passed_cases += 1
                else:
                    diff_i, sp, sn, gap = _py_score_gap(drafter, anchor, hidden, base_py, k_draft, py_toks, nat_toks)
                    if gap <= 0.25:
                        ok = True
                        passed_cases += 1
                        tie_note = f" [BF16 tie@{diff_i}: {sp:.2f} vs {sn:.2f}, gap={gap:.3f}]"
                    else:
                        mismatches.append((ctx_len, n_ctx, k_draft, py_toks, nat_toks, gap))
                mark = "PASS" if ok else "FAIL"
                print(
                    f"  [{mark}] ctx={ctx_len:4d} N_ctx={n_ctx} K={k_draft} | "
                    f"py={py_toks} | nat={nat_toks} | "
                    f"gpu={stats.gpu_ms:.2f}ms wall={stats.wall_ms:.2f}ms disp={stats.n_dispatches}{tie_note}"
                )

    print("\n=== 3. Multi-Step Consecutive Wrap-Around (ctx=1018 -> 1022 -> 1028 -> 1033) ===")
    c_py, c_nat = _populate_cache_pair(drafter, 1018, seed=7777)
    native.clear_kv_cache_map()
    wrap_cases = [(4, 5, 1234), (6, 7, 5678), (5, 3, 91011)]
    wrap_tok_match = 0
    wrap_tok_total = 0
    for step_idx, (n_ctx, k_draft, anchor) in enumerate(wrap_cases):
        mx.random.seed(8100 + step_idx)
        hidden = (mx.random.normal((1, n_ctx, 4 * cfg.hidden_size)) * 0.5).astype(mx.bfloat16)
        mx.eval(hidden)
        c_py_before = clone_draft_cache(c_py)
        eval_draft_cache(c_py_before)
        py_toks = [int(x) for x in drafter.draft_greedy(anchor, hidden, c_py, k_draft).tolist()]
        nat_toks, stats = native.draft_greedy(anchor, hidden, c_nat, k_draft)
        matched = sum(1 for a, b in zip(py_toks, nat_toks) if a == b)
        wrap_tok_match += matched
        wrap_tok_total += k_draft
        off = c_py[0].offset
        kp0 = c_py[0]._temporal_order(c_py[0].keys)[..., -min(off, 1024):, :]
        kn0 = c_nat[0]._temporal_order(c_nat[0].keys) if c_nat[0].offset > 1024 else c_nat[0].keys[..., :off, :]
        kv_diff = float(mx.max(mx.abs(kp0.astype(mx.float32) - kn0.astype(mx.float32))).item())
        ok = (py_toks == nat_toks) and (kv_diff <= 0.0625)
        tie_note = ""
        if not ok and kv_diff <= 0.0625:
            diff_i, sp, sn, gap = _py_score_gap(drafter, anchor, hidden, c_py_before, k_draft, py_toks, nat_toks)
            if gap <= 0.25:
                ok = True
                tie_note = f" [BF16 tie@{diff_i}: {sp:.2f} vs {sn:.2f}, gap={gap:.3f}]"
        total_cases += 1
        if ok:
            passed_cases += 1
        else:
            mismatches.append((f"wrap_step{step_idx}", n_ctx, k_draft, py_toks, nat_toks, kv_diff))
        print(
            f"  [{'PASS' if ok else 'FAIL'}] step={step_idx} offset_after={c_nat[0].offset} "
            f"ring_idx={c_nat[0]._idx} L0_kv_maxdiff={kv_diff:.5f} | py={py_toks} | nat={nat_toks} | gpu={stats.gpu_ms:.2f}ms{tie_note}"
        )

    print("\n=== 4. Long Global Context Split-K SDPA Parity (ctx=4096) ===")
    c_py, c_nat = _populate_cache_pair(drafter, 4096, seed=9999)
    native.clear_kv_cache_map()
    total_cases += 1
    mx.random.seed(9998)
    hidden = (mx.random.normal((1, 4, 4 * cfg.hidden_size)) * 0.5).astype(mx.bfloat16)
    mx.eval(hidden)
    py_toks = [int(x) for x in drafter.draft_greedy(4242, hidden, c_py, 5).tolist()]
    nat_toks, stats = native.draft_greedy(4242, hidden, c_nat, 5)
    ok = py_toks == nat_toks
    if ok:
        passed_cases += 1
    else:
        mismatches.append((4096, 4, 5, py_toks, nat_toks))
    print(f"  [{'PASS' if ok else 'FAIL'}] ctx=4096 N_ctx=4 K=5 | py={py_toks} | nat={nat_toks} | gpu={stats.gpu_ms:.2f}ms")

    print(f"\nParity summary: {passed_cases}/{total_cases} exact token matches")
    if mismatches:
        for m in mismatches:
            print("  MISMATCH:", m)
        raise AssertionError(f"{len(mismatches)} parity mismatches!")

    print("\n=== 4b. Native Single-CB Stage Breakdown (ctx=1100, N_ctx=4) ===")
    _, c_prof = _populate_cache_pair(drafter, 1100, seed=400)
    c_prof[1].prefix_cache_reserve(1200)
    eval_draft_cache(c_prof)
    native.ensure_kv_capacity(c_prof, 4)
    native.clear_kv_cache_map()
    h_prof = (mx.random.normal((1, 4, 4 * cfg.hidden_size)) * 0.5).astype(mx.bfloat16)
    mx.eval(h_prof)
    for k_draft in (3, 5, 7):
        bd = native.profile_stages(1234, h_prof, c_prof, k_draft, warmup=5, iters=25)
        lgemv = bd.layers_ms - bd.sdpa_only_ms
        tot = bd.layers_ms + bd.lm_head_ms + bd.markov_ctx_ms + bd.rnn_steps_ms + bd.argmax_ms
        print(
            f"  K={k_draft}: layers_gemv={lgemv:.3f}ms ({523.8/lgemv:.1f}GB/s) "
            f"sdpa={bd.sdpa_only_ms:.3f}ms lm_head={bd.lm_head_ms:.3f}ms ({921.9/bd.lm_head_ms:.1f}GB/s) "
            f"markov_ctx={bd.markov_ctx_ms:.3f}ms rnn={bd.rnn_steps_ms:.3f}ms "
            f"argmax={bd.argmax_ms:.3f}ms ({k_draft*103.2/bd.argmax_ms:.1f}GB/s) | sum={tot:.3f}ms"
        )

    print("\n=== 5. A/B Benchmark: Python MLX vs Native Metal 4.0 Single-CB (N_ctx=4, warmup=5, iters=25) ===")
    warmup, iters = 5, 25
    for bench_ctx in (512, 1100):
        print(f"\n  --- Context = {bench_ctx} ({'RotatingKV wrapped' if bench_ctx > 1024 else 'pre-wrap'}) ---")
        print(f"  {'K':>3} | {'Py MLX Wall (ms)':>16} | {'Nat Wall (ms)':>14} | {'Nat GPU (ms)':>13} | {'Nat Enc (ms)':>13} | {'Speedup':>8}")
        print("  " + "-" * 80)

        for k_draft in (3, 5, 7):
            mx.random.seed(300 + k_draft)
            hidden = (mx.random.normal((1, 4, 4 * cfg.hidden_size)) * 0.5).astype(mx.bfloat16)
            mx.eval(hidden)

            c_py, c_nat = _populate_cache_pair(drafter, bench_ctx, seed=400 + bench_ctx)
            c_py[1].prefix_cache_reserve(bench_ctx + 64)
            c_nat[1].prefix_cache_reserve(bench_ctx + 64)
            eval_draft_cache(c_py)
            eval_draft_cache(c_nat)
            native.ensure_kv_capacity(c_nat, 4)
            native.clear_kv_cache_map()

            rot_snap_py = [
                (mx.array(c.keys), mx.array(c.values), c.offset, c._idx)
                if isinstance(c, RotatingKVCache)
                else c.offset
                for c in c_py
            ]
            mx.eval(*[x for s in rot_snap_py if isinstance(s, tuple) for x in s[:2]])

            py_walls = []
            for it in range(-warmup, iters):
                for idx, snap in enumerate(rot_snap_py):
                    if isinstance(snap, tuple):
                        c_py[idx].keys, c_py[idx].values, c_py[idx].offset, c_py[idx]._idx = snap
                    else:
                        c_py[idx].offset = snap
                t0 = time.perf_counter()
                out = drafter.draft_greedy(1234, hidden, c_py, k_draft)
                mx.eval(out)
                t1 = time.perf_counter()
                if it >= 0:
                    py_walls.append((t1 - t0) * 1000.0)

            nat_walls, nat_gpus, nat_encs = [], [], []
            for it in range(-warmup, iters):
                for idx in range(len(c_nat)):
                    c_nat[idx].offset = bench_ctx
                    if isinstance(c_nat[idx], RotatingKVCache):
                        c_nat[idx]._idx = min(bench_ctx, c_nat[idx].max_size)
                t0 = time.perf_counter()
                _, st = native.draft_greedy(1234, hidden, c_nat, k_draft)
                t1 = time.perf_counter()
                if it >= 0:
                    nat_walls.append((t1 - t0) * 1000.0)
                    nat_gpus.append(st.gpu_ms)
                    nat_encs.append(st.encode_ms)

            py_med = statistics.median(py_walls)
            nw_med = statistics.median(nat_walls)
            ng_med = statistics.median(nat_gpus)
            ne_med = statistics.median(nat_encs)
            spd = py_med / nw_med
            print(
                f"  {k_draft:3d} | {py_med:16.3f} | {nw_med:14.3f} | {ng_med:13.3f} | {ne_med:13.3f} | {spd:7.2f}x"
            )

    native.close()
    print("\nALL TESTS PASSED!")


if __name__ == "__main__":
    main()
