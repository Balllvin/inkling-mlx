"""Production zero-copy Python wrapper for `libinkling_mtp_draft.dylib`.

Executes an entire Inkling 2 Mini MTP `draft_greedy(anchor, hidden, draft_cache, K)`
pass (3 BF16 DraftLayers w/ ring-indexed rotating KV + final norm + single-read
`lm_head` + Markov context + K sequential GPU-side RNN/argmax steps) inside ONE
native Metal 4.0 `MTLCommandBuffer` (~7.2–8.3 ms at 280+ GB/s).
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import numpy as np
from mlx_vlm.models.cache import RotatingKVCache

_LIB_PATH = Path(__file__).resolve().parent / "native" / "libinkling_mtp_draft.dylib"
_CACHED_LIB: Optional[ctypes.CDLL] = None
_LIB_TRIED = False


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


def _load_lib() -> Optional[ctypes.CDLL]:
    global _CACHED_LIB, _LIB_TRIED
    if _LIB_TRIED:
        return _CACHED_LIB
    _LIB_TRIED = True
    if not _LIB_PATH.is_file():
        return None
    try:
        lib = ctypes.CDLL(str(_LIB_PATH))
        lib.mtp_draft_create.argtypes = [
            ctypes.POINTER(MTPDraftConfig),
            ctypes.POINTER(MTPDraftGlobalWeights),
            ctypes.POINTER(MTPDraftLayerWeights),
        ]
        lib.mtp_draft_create.restype = ctypes.c_void_p
        lib.mtp_draft_destroy.argtypes = [ctypes.c_void_p]
        lib.mtp_draft_destroy.restype = None
        lib.mtp_draft_clear_kv_cache_map.argtypes = [ctypes.c_void_p]
        lib.mtp_draft_clear_kv_cache_map.restype = None
        lib.mtp_draft_greedy_step.argtypes = [
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
        lib.mtp_draft_greedy_step.restype = ctypes.c_int
        _CACHED_LIB = lib
    except Exception:
        _CACHED_LIB = None
    return _CACHED_LIB


def _ptr_bytes(arr: mx.array, keepalive: list[Any]) -> tuple[int, int]:
    c = mx.contiguous(arr)
    v = c.view(mx.uint16) if c.dtype == mx.bfloat16 else c
    mx.eval(v)
    nv = np.asarray(v)
    keepalive.append((c, v, nv))
    return int(nv.ctypes.data), int(nv.nbytes)


def build_mtp_c_descriptors(
    drafter,
    keepalive: list[Any],
) -> tuple[MTPDraftConfig, MTPDraftGlobalWeights, ctypes.Array, tuple[int, ...]]:
    """Build pinned zero-copy C descriptor structs for `InklingMTPDrafter`."""
    cfg = drafter.config
    target = drafter._target
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
        embed_norm_eps=(
            float(getattr(target.model.embed_norm, "eps", cfg.rms_norm_eps))
            if hasattr(target.model, "embed_norm") and target.model.embed_norm is not None
            else cfg.rms_norm_eps
        ),
        rope_theta=cfg.rope_theta,
    )

    p_emb, b_emb = _ptr_bytes(target.model.embed_tokens.weight, keepalive)
    if hasattr(target.model, "embed_norm") and target.model.embed_norm is not None:
        p_en, b_en = _ptr_bytes(target.model.embed_norm.weight, keepalive)
    else:
        p_en, b_en = 0, 0
    p_mask, b_mask = _ptr_bytes(drafter.mask_embed, keepalive)
    p_fnorm, b_fnorm = _ptr_bytes(drafter.norm.weight, keepalive)
    p_lmh, b_lmh = _ptr_bytes(target.lm_head.weight, keepalive)
    p_fa, b_fa = _ptr_bytes(drafter._fuse_alpha, keepalive)
    mh = drafter.markov_head
    p_mdn, b_mdn = _ptr_bytes(mh.dflare_norm.weight, keepalive)
    p_mw1, b_mw1 = _ptr_bytes(mh.markov_w1.weight, keepalive)
    p_wctx, b_wctx = _ptr_bytes(mh._rnn_w_ctx_t, keepalive)
    p_wstp, b_wstp = _ptr_bytes(mh._rnn_w_step_t, keepalive)
    p_rb, b_rb = _ptr_bytes(mh._rnn_b_f32, keepalive)
    p_fw, b_fw = _ptr_bytes(mh._fold_weight, keepalive)
    p_fb, b_fb = _ptr_bytes(mh._fold_bias, keepalive)

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

    c_layers = (MTPDraftLayerWeights * cfg.num_hidden_layers)()
    for i, layer in enumerate(drafter.layers):
        attn = layer.self_attn
        mlp = layer.mlp
        p_dn, b_dn = _ptr_bytes(layer.dflare_norm.weight, keepalive)
        p_iln, b_iln = _ptr_bytes(layer.input_layernorm.weight, keepalive)
        p_pln, b_pln = _ptr_bytes(layer.post_attention_layernorm.weight, keepalive)
        p_kvc, b_kvc = _ptr_bytes(attn._kv_ctx_w, keepalive)
        p_knc, b_knc = _ptr_bytes(attn.k_norm_ctx.weight, keepalive)
        p_qkv, b_qkv = _ptr_bytes(attn._qkv_w, keepalive)
        p_qn, b_qn = _ptr_bytes(attn.q_norm.weight, keepalive)
        p_kn, b_kn = _ptr_bytes(attn.k_norm.weight, keepalive)
        p_op, b_op = _ptr_bytes(attn.o_proj.weight, keepalive)
        p_gu, b_gu = _ptr_bytes(mlp._gu_w, keepalive)
        p_dp, b_dp = _ptr_bytes(mlp.down_proj.weight, keepalive)
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
    return c_cfg, c_glob, c_layers, tuple(int(x) for x in cfg.target_layer_ids)


class NativeMTPDraftRunner:
    """Zero-copy single-CB native Metal 4.0 runner for `InklingMTPDrafter.draft_greedy`."""

    def __init__(self, lib: ctypes.CDLL, drafter):
        self.lib = lib
        self._drafter = drafter
        self._keepalive: list[Any] = []
        self._kv_ptr_cache: dict[int, tuple[mx.array, np.ndarray, int, int]] = {}
        self._last_cache_id: int = 0
        self._out_toks = (ctypes.c_int32 * (drafter.config.block_size - 1))()
        self._c_kvs = (MTPKVCacheDesc * drafter.config.num_hidden_layers)()

        c_cfg, c_glob, c_layers, _ = build_mtp_c_descriptors(drafter, self._keepalive)

        self._handle = lib.mtp_draft_create(ctypes.byref(c_cfg), ctypes.byref(c_glob), c_layers)
        if not self._handle:
            raise RuntimeError("mtp_draft_create failed")

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self.lib.mtp_draft_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def clear_cache_binding(self) -> None:
        self._last_cache_obj = None
        self._kv_ptr_cache.clear()
        if getattr(self, "_handle", None):
            self.lib.mtp_draft_clear_kv_cache_map(self._handle)

    def _cached_kv_ptr(self, arr: mx.array) -> tuple[int, int, bool]:
        key = id(arr)
        hit = self._kv_ptr_cache.get(key)
        if hit is not None and hit[0] is arr:
            return hit[3], hit[4], False
        v = arr.view(mx.uint16)
        mx.eval(v)
        nv = np.asarray(v)
        if not nv.flags.c_contiguous:
            arr_c = mx.contiguous(arr)
            v = arr_c.view(mx.uint16)
            mx.eval(v)
            nv = np.asarray(v)
        ptr, nb = int(nv.ctypes.data), int(nv.nbytes)
        self._kv_ptr_cache[key] = (arr, v, nv, ptr, nb)
        return ptr, nb, True

    def _ensure_kv_capacity(self, draft_cache: list, n_ctx: int) -> None:
        cfg = self._drafter.config
        kv_h, hd = cfg.num_key_value_heads, cfg.head_dim
        to_eval = []
        for entry in draft_cache:
            if isinstance(entry, RotatingKVCache):
                w = entry.max_size
                if entry.keys is None:
                    entry.keys = mx.zeros((1, kv_h, w, hd), dtype=mx.bfloat16)
                    entry.values = mx.zeros((1, kv_h, w, hd), dtype=mx.bfloat16)
                    entry.offset = 0
                    entry._idx = 0
                    to_eval.extend((entry.keys, entry.values))
                elif entry.keys.shape[2] != w:
                    k_ord = entry._temporal_order(entry.keys) if entry.offset > w else entry.keys[..., : entry.offset, :]
                    v_ord = entry._temporal_order(entry.values) if entry.offset > w else entry.values[..., : entry.offset, :]
                    if k_ord.shape[2] > w:
                        k_ord = k_ord[..., -w:, :]
                        v_ord = v_ord[..., -w:, :]
                    cur_len = k_ord.shape[2]
                    if cur_len < w:
                        pad = mx.zeros((1, kv_h, w - cur_len, hd), dtype=mx.bfloat16)
                        entry.keys = mx.contiguous(mx.concatenate([k_ord, pad], axis=2))
                        entry.values = mx.contiguous(mx.concatenate([v_ord, pad], axis=2))
                        entry._idx = cur_len
                    else:
                        entry.keys = mx.contiguous(k_ord)
                        entry.values = mx.contiguous(v_ord)
                        entry._idx = w
                    to_eval.extend((entry.keys, entry.values))
                else:
                    if id(entry.keys) not in self._kv_ptr_cache or id(entry.values) not in self._kv_ptr_cache:
                        kc = mx.contiguous(entry.keys)
                        vc = mx.contiguous(entry.values)
                        if kc is not entry.keys or vc is not entry.values:
                            entry.keys = kc
                            entry.values = vc
                            to_eval.extend((entry.keys, entry.values))
            else:
                need = entry.offset + n_ctx
                if entry.keys is None:
                    cap = max(2048, ((need + 2047) // 2048) * 2048)
                    entry.keys = mx.zeros((1, kv_h, cap, hd), dtype=mx.bfloat16)
                    entry.values = mx.zeros((1, kv_h, cap, hd), dtype=mx.bfloat16)
                    to_eval.extend((entry.keys, entry.values))
                elif entry.keys.shape[2] < need:
                    step = 2048
                    new_cap = ((need + step - 1) // step) * step
                    pad = new_cap - entry.keys.shape[2]
                    entry.keys = mx.contiguous(
                        mx.concatenate([entry.keys, mx.zeros((1, kv_h, pad, hd), dtype=mx.bfloat16)], axis=2)
                    )
                    entry.values = mx.contiguous(
                        mx.concatenate([entry.values, mx.zeros((1, kv_h, pad, hd), dtype=mx.bfloat16)], axis=2)
                    )
                    to_eval.extend((entry.keys, entry.values))
        if to_eval:
            mx.eval(*to_eval)
            self._kv_ptr_cache.clear()

    def draft_greedy_list(
        self,
        anchor_token: int,
        hidden: mx.array,
        draft_cache: list,
        num_draft: int,
    ) -> list[int]:
        if getattr(self, "_last_cache_obj", None) is not draft_cache:
            self._kv_ptr_cache.clear()
            self.lib.mtp_draft_clear_kv_cache_map(self._handle)
            self._last_cache_obj = draft_cache

        n_ctx = int(hidden.shape[1])
        self._ensure_kv_capacity(draft_cache, n_ctx)

        h_view = hidden.view(mx.uint16)
        mx.eval(h_view)
        h_nv = np.asarray(h_view)
        if not h_nv.flags.c_contiguous:
            h_c = mx.contiguous(hidden)
            h_view = h_c.view(mx.uint16)
            mx.eval(h_view)
            h_nv = np.asarray(h_view)
        h_ptr = int(h_nv.ctypes.data)
        h_bytes = int(h_nv.nbytes)

        c_kvs = self._c_kvs
        any_new_kv = False
        for i, entry in enumerate(draft_cache):
            kptr, kbytes, k_new = self._cached_kv_ptr(entry.keys)
            vptr, vbytes, v_new = self._cached_kv_ptr(entry.values)
            if k_new or v_new:
                any_new_kv = True
            is_rot = 1 if isinstance(entry, RotatingKVCache) else 0
            cap = int(entry.keys.shape[2])
            hd = int(entry.keys.shape[3])
            c_kvs[i].k_ptr = kptr
            c_kvs[i].v_ptr = vptr
            c_kvs[i].k_nbytes = kbytes
            c_kvs[i].v_nbytes = vbytes
            c_kvs[i].cache_cap = cap
            c_kvs[i].offset = int(entry.offset)
            c_kvs[i].is_rotating = is_rot
            c_kvs[i].window = int(entry.max_size) if is_rot else 0
            c_kvs[i].ring_idx = int(entry._idx) if is_rot else -1
            c_kvs[i].stride_h = cap * hd
            c_kvs[i].stride_s = hd
            c_kvs[i]._pad = 0

        if any_new_kv:
            self.lib.mtp_draft_clear_kv_cache_map(self._handle)

        stats = MTPStepStats()
        rc = self.lib.mtp_draft_greedy_step(
            self._handle,
            int(anchor_token),
            h_ptr,
            h_bytes,
            n_ctx,
            int(num_draft),
            c_kvs,
            self._out_toks,
            ctypes.byref(stats),
        )
        if rc != 0:
            raise RuntimeError(f"mtp_draft_greedy_step failed with rc={rc}")

        for i, entry in enumerate(draft_cache):
            entry.offset = int(c_kvs[i].offset)
            if isinstance(entry, RotatingKVCache):
                entry._idx = int(c_kvs[i].ring_idx)

        return [int(self._out_toks[i]) for i in range(num_draft)]


def try_create_native_mtp_runner(drafter) -> Optional[NativeMTPDraftRunner]:
    if mx.default_device() != mx.gpu:
        return None
    target = getattr(drafter, "_target", None)
    if target is None:
        return None
    if target.lm_head.weight.dtype != mx.bfloat16 or drafter.mask_embed.dtype != mx.bfloat16:
        return None
    lib = _load_lib()
    if lib is None:
        return None
    try:
        return NativeMTPDraftRunner(lib, drafter)
    except Exception:
        return None
