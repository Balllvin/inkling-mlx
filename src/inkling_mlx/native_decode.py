"""Production zero-copy Python wrapper for `libinkling_decode.dylib`.

Executes an entire 48-visit Inkling 2 Mini single-stream decode (`M=1`) or MTP
speculative verify (`M=2..8`) step — including embedding, all 48 visits (3 loop
passes), final mix+norm, greedy `head_argmax`, MTP layer captures, and staged
rotating KV + conv-state rollback — inside ONE native Metal `MTLCommandBuffer`.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import numpy as np
from mlx_vlm.models.cache import BatchRotatingKVCache, CacheList, RotatingKVCache

from . import fused
from .native_mtp import (
    MTPDraftConfig,
    MTPDraftGlobalWeights,
    MTPDraftLayerWeights,
    MTPKVCacheDesc,
    build_mtp_c_descriptors,
)

_LIB_PATH = Path(__file__).resolve().parent / "native" / "libinkling_decode.dylib"
_CACHED_LIB: Optional[ctypes.CDLL] = None
_LIB_TRIED = False


class InklingModelConfig(ctypes.Structure):
    _fields_ = [
        ("n_layers", ctypes.c_int32),
        ("loop_start", ctypes.c_int32),
        ("loop_end", ctypes.c_int32),
        ("loop_passes", ctypes.c_int32),
        ("hidden_size", ctypes.c_int32),
        ("inter_size", ctypes.c_int32),
        ("n_experts", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("vocab_size", ctypes.c_int32),
        ("unpadded_vocab_size", ctypes.c_int32),
        ("d_rel", ctypes.c_int32),
        ("loop_gate_rank", ctypes.c_int32),
        ("max_m", ctypes.c_int32),
        ("max_n_heads", ctypes.c_int32),
        ("max_n_kv", ctypes.c_int32),
        ("max_head_dim", ctypes.c_int32),
        ("max_rel_extent", ctypes.c_int32),
        ("rms_norm_eps", ctypes.c_float),
        ("qk_l2_clip_threshold", ctypes.c_float),
        ("rel_l2_clip_threshold", ctypes.c_float),
        ("route_scale", ctypes.c_float),
        ("mlp_gate_softcap", ctypes.c_float),
        ("mlp_up_softcap", ctypes.c_float),
        ("log_scaling_n_floor", ctypes.c_float),
        ("log_scaling_alpha", ctypes.c_float),
    ]


class InklingGlobalWeights(ctypes.Structure):
    _fields_ = [
        ("embed_tokens", ctypes.c_void_p),
        ("embed_tokens_bytes", ctypes.c_size_t),
        ("embed_norm", ctypes.c_void_p),
        ("embed_norm_bytes", ctypes.c_size_t),
        ("message_queries", ctypes.c_void_p),
        ("message_queries_bytes", ctypes.c_size_t),
        ("final_query", ctypes.c_void_p),
        ("final_query_bytes", ctypes.c_size_t),
        ("final_norm", ctypes.c_void_p),
        ("final_norm_bytes", ctypes.c_size_t),
        ("lm_head", ctypes.c_void_p),
        ("lm_head_bytes", ctypes.c_size_t),
        ("embed_norm_eps", ctypes.c_float),
        ("final_norm_eps", ctypes.c_float),
    ]


class InklingLayerWeights(ctypes.Structure):
    _fields_ = [
        ("layer_idx", ctypes.c_int32),
        ("is_sliding", ctypes.c_int32),
        ("is_block_boundary", ctypes.c_int32),
        ("has_attn_input", ctypes.c_int32),
        ("has_message_gates", ctypes.c_int32),
        ("n_heads", ctypes.c_int32),
        ("n_kv", ctypes.c_int32),
        ("head_dim", ctypes.c_int32),
        ("sliding_window", ctypes.c_int32),
        ("rel_extent", ctypes.c_int32),
        ("qkv_at", ctypes.c_int32 * 5),
        ("qkv_total_rows", ctypes.c_int32),
        ("moe_mode", ctypes.c_int32),
        ("attn_eps", ctypes.c_float),
        ("mlp_eps", ctypes.c_float),
        ("stack", ctypes.c_void_p),
        ("stack_bytes", ctypes.c_size_t),
        ("o_proj", ctypes.c_void_p),
        ("o_proj_bytes", ctypes.c_size_t),
        ("attn_query", ctypes.c_void_p),
        ("attn_query_bytes", ctypes.c_size_t),
        ("attn_norm", ctypes.c_void_p),
        ("attn_norm_bytes", ctypes.c_size_t),
        ("mlp_query", ctypes.c_void_p),
        ("mlp_query_bytes", ctypes.c_size_t),
        ("mlp_norm", ctypes.c_void_p),
        ("mlp_norm_bytes", ctypes.c_size_t),
        ("attn_taps", ctypes.c_void_p),
        ("attn_taps_bytes", ctypes.c_size_t),
        ("mlp_taps", ctypes.c_void_p),
        ("mlp_taps_bytes", ctypes.c_size_t),
        ("k_taps", ctypes.c_void_p),
        ("k_taps_bytes", ctypes.c_size_t),
        ("v_taps", ctypes.c_void_p),
        ("v_taps_bytes", ctypes.c_size_t),
        ("kdyn_w", ctypes.c_void_p),
        ("kdyn_w_bytes", ctypes.c_size_t),
        ("vdyn_w", ctypes.c_void_p),
        ("vdyn_w_bytes", ctypes.c_size_t),
        ("attn_gate_down", ctypes.c_void_p),
        ("attn_gate_down_bytes", ctypes.c_size_t),
        ("attn_gate_forget", ctypes.c_void_p),
        ("attn_gate_forget_bytes", ctypes.c_size_t),
        ("attn_gate_gain", ctypes.c_void_p),
        ("attn_gate_gain_bytes", ctypes.c_size_t),
        ("mlp_gate_down", ctypes.c_void_p),
        ("mlp_gate_down_bytes", ctypes.c_size_t),
        ("mlp_gate_forget", ctypes.c_void_p),
        ("mlp_gate_forget_bytes", ctypes.c_size_t),
        ("mlp_gate_gain", ctypes.c_void_p),
        ("mlp_gate_gain_bytes", ctypes.c_size_t),
        ("router_w", ctypes.c_void_p),
        ("router_w_bytes", ctypes.c_size_t),
        ("router_b", ctypes.c_void_p),
        ("router_b_bytes", ctypes.c_size_t),
        ("sink_w", ctypes.c_void_p),
        ("sink_w_bytes", ctypes.c_size_t),
        ("q_scale", ctypes.c_void_p),
        ("q_scale_bytes", ctypes.c_size_t),
        ("rel_scale", ctypes.c_void_p),
        ("rel_scale_bytes", ctypes.c_size_t),
        ("rel_proj", ctypes.c_void_p),
        ("rel_proj_bytes", ctypes.c_size_t),
        ("w13_weight", ctypes.c_void_p),
        ("w13_weight_bytes", ctypes.c_size_t),
        ("w13_scales", ctypes.c_void_p),
        ("w13_scales_bytes", ctypes.c_size_t),
        ("w13_gscale", ctypes.c_void_p),
        ("w13_gscale_bytes", ctypes.c_size_t),
        ("w2_weight", ctypes.c_void_p),
        ("w2_weight_bytes", ctypes.c_size_t),
        ("w2_scales", ctypes.c_void_p),
        ("w2_scales_bytes", ctypes.c_size_t),
        ("w2_gscale", ctypes.c_void_p),
        ("w2_gscale_bytes", ctypes.c_size_t),
        ("attn_gate_fg_bf16", ctypes.c_void_p),
        ("attn_gate_fg_bf16_bytes", ctypes.c_size_t),
        ("mlp_gate_fg_bf16", ctypes.c_void_p),
        ("mlp_gate_fg_bf16_bytes", ctypes.c_size_t),
        ("router_w_bf16", ctypes.c_void_p),
        ("router_w_bf16_bytes", ctypes.c_size_t),
    ]


class InklingKVMeta(ctypes.Structure):
    _fields_ = [
        ("offset", ctypes.c_int32),
        ("idx", ctypes.c_int32),
        ("capacity", ctypes.c_int32),
        ("max_size", ctypes.c_int32),
        ("is_rotating", ctypes.c_int32),
        ("stride_h", ctypes.c_int32),
        ("stride_s", ctypes.c_int32),
        ("_pad", ctypes.c_int32),
    ]


class InklingBatchKVMeta(ctypes.Structure):
    _fields_ = [
        ("idx", ctypes.c_int32),
        ("offset", ctypes.c_int32),
        ("capacity", ctypes.c_int32),
        ("max_size", ctypes.c_int32),
        ("is_rotating", ctypes.c_int32),
        ("stride_b", ctypes.c_int32),
        ("stride_h", ctypes.c_int32),
        ("stride_s", ctypes.c_int32),
    ]


class InklingStepStats(ctypes.Structure):
    _fields_ = [
        ("encode_ms", ctypes.c_double),
        ("gpu_ms", ctypes.c_double),
        ("wall_ms", ctypes.c_double),
        ("n_dispatches", ctypes.c_int32),
    ]


class InklingMTPStreamDesc(ctypes.Structure):
    _fields_ = [
        ("anchor_token", ctypes.c_int32),
        ("n_ctx", ctypes.c_int32),
        ("num_draft", ctypes.c_int32),
        ("_pad0", ctypes.c_int32),
        ("in_hidden_ptr", ctypes.c_void_p),
        ("in_hidden_bytes", ctypes.c_size_t),
        ("out_hidden_ptr", ctypes.c_void_p),
        ("out_hidden_bytes", ctypes.c_size_t),
        ("temperature", ctypes.c_float),
        ("top_p", ctypes.c_float),
        ("logit_softcap", ctypes.c_float),
        ("_pad1", ctypes.c_int32),
        ("rng_seed", ctypes.c_uint64),
        ("rng_step", ctypes.c_uint64),
        ("kv_ptrs", ctypes.POINTER(ctypes.c_void_p)),
        ("conv_ptrs", ctypes.POINTER(ctypes.c_void_p)),
        ("kv_meta", ctypes.POINTER(InklingKVMeta)),
        ("draft_kv_descs", ctypes.POINTER(MTPKVCacheDesc)),
        ("out_tokens", ctypes.c_int32 * 8),
        ("out_n_acc", ctypes.c_int32),
        ("_pad2", ctypes.c_int32),
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
        lib.inkling_ctx_create.argtypes = [ctypes.POINTER(InklingModelConfig)]
        lib.inkling_ctx_create.restype = ctypes.c_void_p
        lib.inkling_ctx_destroy.argtypes = [ctypes.c_void_p]
        lib.inkling_ctx_destroy.restype = None
        lib.inkling_ctx_register_globals.argtypes = [ctypes.c_void_p, ctypes.POINTER(InklingGlobalWeights)]
        lib.inkling_ctx_register_globals.restype = ctypes.c_int
        lib.inkling_ctx_register_layer.argtypes = [ctypes.c_void_p, ctypes.POINTER(InklingLayerWeights)]
        lib.inkling_ctx_register_layer.restype = ctypes.c_int
        lib.inkling_ctx_num_visits.argtypes = [ctypes.c_void_p]
        lib.inkling_ctx_num_visits.restype = ctypes.c_int
        lib.inkling_ctx_scratch_bytes.argtypes = [ctypes.c_void_p]
        lib.inkling_ctx_scratch_bytes.restype = ctypes.c_size_t
        lib.inkling_ctx_clear_ptr_cache.argtypes = [ctypes.c_void_p]
        lib.inkling_ctx_clear_ptr_cache.restype = None
        lib.inkling_ctx_step.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(InklingKVMeta),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.POINTER(InklingStepStats),
        ]
        lib.inkling_ctx_step.restype = ctypes.c_int
        lib.inkling_ctx_commit_verify.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(InklingKVMeta),
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        lib.inkling_ctx_commit_verify.restype = ctypes.c_int
        lib.inkling_ctx_step_batched.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(InklingBatchKVMeta),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_void_p,
            ctypes.POINTER(InklingStepStats),
        ]
        lib.inkling_ctx_step_batched.restype = ctypes.c_int
        lib.inkling_ctx_attach_mtp.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(MTPDraftConfig),
            ctypes.POINTER(MTPDraftGlobalWeights),
            ctypes.POINTER(MTPDraftLayerWeights),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
        ]
        lib.inkling_ctx_attach_mtp.restype = ctypes.c_int
        lib.inkling_ctx_mtp_step.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(InklingKVMeta),
            ctypes.POINTER(MTPKVCacheDesc),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(InklingStepStats),
        ]
        lib.inkling_ctx_mtp_step.restype = ctypes.c_int
        lib.inkling_ctx_mtp_step_batched.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(InklingMTPStreamDesc),
            ctypes.c_int32,
            ctypes.POINTER(InklingStepStats),
        ]
        lib.inkling_ctx_mtp_step_batched.restype = ctypes.c_int
        _CACHED_LIB = lib
    except Exception:
        _CACHED_LIB = None
    return _CACHED_LIB


def _arr_ptr(arr: Optional[mx.array], keepalive: list[Any]) -> tuple[int, int]:
    if arr is None:
        return 0, 0
    a = mx.contiguous(arr)
    view = a.view(mx.uint16) if a.dtype == mx.bfloat16 else a
    mx.eval(view)
    nv = np.asarray(view)
    keepalive.append((a, view, nv))
    return int(nv.ctypes.data), int(nv.nbytes)


class NativeDecodeRunner:
    """Zero-copy single-CB native Metal runner for single/batched decode (B=1..64) & MTP verify (M=2..8)."""

    def __init__(self, lib: ctypes.CDLL, lm, max_m: int = 64):
        self.lib = lib
        self.lm = lm
        self.tm = lm.model
        self.cfg = lm.config
        self._max_m = max_m
        self._keepalive: list[Any] = []
        self._cap_bufs: dict[int, tuple[mx.array, mx.array, np.ndarray]] = {}
        self._pre_norm_mx = mx.zeros((1, max_m, self.cfg.hidden_size), dtype=mx.float32)
        mx.eval(self._pre_norm_mx)
        self._pre_norm_np = np.asarray(self._pre_norm_mx)
        self._toks_c = (ctypes.c_int32 * max_m)()
        self._toks_out_c = (ctypes.c_int32 * max_m)()
        self._cap_ids_cache: dict[tuple[int, ...], Any] = {}
        self._schedule = self.tm.schedule()
        self._visit_head_dims = [self.tm.layers[li].self_attn.head_dim for li, _, _ in self._schedule]
        self._cache_binding: Optional[dict[str, Any]] = None
        self._batch_cache_binding: Optional[dict[str, Any]] = None
        self._last_verify_packed: Optional[tuple] = None
        self.mtp_attached: bool = False
        self._mtp_keepalive: list[Any] = []
        self._mtp_cfg: Optional[MTPDraftConfig] = None
        self._mtp_drafter: Any = None
        self._draft_kv_binding: Optional[dict[str, Any]] = None
        self._mtp_out_hidden_view: Optional[tuple[mx.array, mx.array, np.ndarray]] = None
        self._mtp_out_toks_c = (ctypes.c_int32 * max(8, max_m))()
        self._mtp_out_n_acc_c = ctypes.c_int32(0)
        self._stream_cache_bindings: dict[int, dict[str, Any]] = {}
        self._stream_draft_bindings: dict[int, dict[str, Any]] = {}
        self._stream_out_hidden_views: dict[int, tuple[mx.array, mx.array, np.ndarray]] = {}
        self._mtp_batch_descs = (InklingMTPStreamDesc * 4)()

        consts = getattr(self.tm, "_fused_consts", None)
        if consts is None:
            consts = [fused._LayerConsts(layer, self.cfg) for layer in self.tm.layers]
            object.__setattr__(self.tm, "_fused_consts", consts)
        self.consts = consts

        c = self.cfg
        mcfg = InklingModelConfig(
            n_layers=c.num_hidden_layers,
            loop_start=c.loop_layer_index_start,
            loop_end=c.loop_layer_index_end,
            loop_passes=c.loop_num_passes,
            hidden_size=c.hidden_size,
            inter_size=c.intermediate_size,
            n_experts=c.n_routed_experts,
            top_k=c.num_experts_per_tok,
            vocab_size=c.vocab_size,
            unpadded_vocab_size=c.unpadded_vocab_size or c.vocab_size,
            d_rel=c.d_rel,
            loop_gate_rank=c.loop_gate_rank,
            max_m=max_m,
            max_n_heads=max(c.num_attention_heads, c.swa_num_attention_heads),
            max_n_kv=max(c.num_key_value_heads, c.swa_num_key_value_heads),
            max_head_dim=max(c.head_dim, c.swa_head_dim),
            max_rel_extent=max(c.rel_extent, c.sliding_window_size),
            rms_norm_eps=c.rms_norm_eps,
            qk_l2_clip_threshold=c.qk_l2_clip_threshold,
            rel_l2_clip_threshold=c.rel_l2_clip_threshold,
            route_scale=c.route_scale,
            mlp_gate_softcap=c.mlp_gate_softcap,
            mlp_up_softcap=c.mlp_up_softcap,
            log_scaling_n_floor=float(c.log_scaling_n_floor or 0.0),
            log_scaling_alpha=float(c.log_scaling_alpha),
        )
        self.ctx = lib.inkling_ctx_create(ctypes.byref(mcfg))
        if not self.ctx:
            raise RuntimeError("inkling_ctx_create failed")
        self.n_visits = lib.inkling_ctx_num_visits(self.ctx)
        self.scratch_bytes = lib.inkling_ctx_scratch_bytes(self.ctx)

        mq_f32 = mx.contiguous(self.tm.message_input.query.astype(mx.float32))
        fq_f32 = mx.contiguous(self.tm.final_input.query.reshape(-1).astype(mx.float32))
        mx.eval(mq_f32, fq_f32)
        gw = InklingGlobalWeights()
        gw.embed_tokens, gw.embed_tokens_bytes = _arr_ptr(self.tm.embed_tokens.weight, self._keepalive)
        gw.embed_norm, gw.embed_norm_bytes = _arr_ptr(self.tm.embed_norm.weight, self._keepalive)
        gw.message_queries, gw.message_queries_bytes = _arr_ptr(mq_f32, self._keepalive)
        gw.final_query, gw.final_query_bytes = _arr_ptr(fq_f32, self._keepalive)
        gw.final_norm, gw.final_norm_bytes = _arr_ptr(self.tm.norm.weight, self._keepalive)
        gw.lm_head, gw.lm_head_bytes = _arr_ptr(lm.lm_head.weight, self._keepalive)
        gw.embed_norm_eps = float(self.tm.embed_norm.eps)
        gw.final_norm_eps = float(self.tm.norm.eps)
        if lib.inkling_ctx_register_globals(self.ctx, ctypes.byref(gw)) != 0:
            self.close()
            raise RuntimeError("inkling_ctx_register_globals failed")

        for idx, (layer, lc) in enumerate(zip(self.tm.layers, self.consts)):
            a = layer.self_attn
            moe = layer.mlp
            lw = InklingLayerWeights()
            lw.layer_idx = idx
            lw.is_sliding = 1 if layer.is_sliding else 0
            lw.is_block_boundary = 1 if layer.is_block_boundary else 0
            lw.has_attn_input = 1 if layer.attention_input is not None else 0
            lw.has_message_gates = 1 if lc.attn_gates is not None else 0
            lw.n_heads = a.n_heads
            lw.n_kv = a.n_kv
            lw.head_dim = a.head_dim
            lw.sliding_window = a.sliding
            lw.rel_extent = a.rel_extent
            for j in range(5):
                lw.qkv_at[j] = lc.at[j]
            lw.qkv_total_rows = int(lc.stack.shape[0])
            lw.moe_mode = 0 if moe.w13_proj.mode == "nvfp4" else 1
            lw.attn_eps = float(lc.attn_eps)
            lw.mlp_eps = float(lc.mlp_eps)

            lw.stack, lw.stack_bytes = _arr_ptr(lc.stack, self._keepalive)
            lw.o_proj, lw.o_proj_bytes = _arr_ptr(a.o_proj.weight, self._keepalive)
            lw.attn_query, lw.attn_query_bytes = _arr_ptr(lc.attn_query, self._keepalive)
            lw.attn_norm, lw.attn_norm_bytes = _arr_ptr(lc.attn_norm, self._keepalive)
            lw.mlp_query, lw.mlp_query_bytes = _arr_ptr(lc.mlp_query, self._keepalive)
            lw.mlp_norm, lw.mlp_norm_bytes = _arr_ptr(lc.mlp_norm, self._keepalive)
            lw.attn_taps, lw.attn_taps_bytes = _arr_ptr(lc.attn_taps, self._keepalive)
            lw.mlp_taps, lw.mlp_taps_bytes = _arr_ptr(lc.mlp_taps, self._keepalive)
            lw.k_taps, lw.k_taps_bytes = _arr_ptr(lc.k_taps, self._keepalive)
            lw.v_taps, lw.v_taps_bytes = _arr_ptr(lc.v_taps, self._keepalive)
            lw.kdyn_w, lw.kdyn_w_bytes = _arr_ptr(lc.kdyn_w, self._keepalive)
            lw.vdyn_w, lw.vdyn_w_bytes = _arr_ptr(lc.vdyn_w, self._keepalive)
            if lc.attn_gates is not None:
                lw.attn_gate_down, lw.attn_gate_down_bytes = _arr_ptr(lc.attn_gates[0], self._keepalive)
                lw.attn_gate_forget, lw.attn_gate_forget_bytes = _arr_ptr(lc.attn_gates[1], self._keepalive)
                lw.attn_gate_gain, lw.attn_gate_gain_bytes = _arr_ptr(lc.attn_gates[2], self._keepalive)
                lw.mlp_gate_down, lw.mlp_gate_down_bytes = _arr_ptr(lc.mlp_gates[0], self._keepalive)
                lw.mlp_gate_forget, lw.mlp_gate_forget_bytes = _arr_ptr(lc.mlp_gates[1], self._keepalive)
                lw.mlp_gate_gain, lw.mlp_gate_gain_bytes = _arr_ptr(lc.mlp_gates[2], self._keepalive)
            lw.router_w, lw.router_w_bytes = _arr_ptr(lc.router_w, self._keepalive)
            lw.router_b, lw.router_b_bytes = _arr_ptr(lc.router_b, self._keepalive)
            lw.sink_w, lw.sink_w_bytes = _arr_ptr(lc.sink_w, self._keepalive)
            lw.q_scale, lw.q_scale_bytes = _arr_ptr(a.q_scale, self._keepalive)
            lw.rel_scale, lw.rel_scale_bytes = _arr_ptr(a.rel_scale, self._keepalive)
            lw.rel_proj, lw.rel_proj_bytes = _arr_ptr(a.rel_proj, self._keepalive)
            lw.w13_weight, lw.w13_weight_bytes = _arr_ptr(moe.w13_proj.weight, self._keepalive)
            lw.w13_scales, lw.w13_scales_bytes = _arr_ptr(moe.w13_proj.scales, self._keepalive)
            lw.w13_gscale, lw.w13_gscale_bytes = _arr_ptr(
                getattr(moe.w13_proj, "global_scale", None), self._keepalive
            )
            lw.w2_weight, lw.w2_weight_bytes = _arr_ptr(moe.down_proj.weight, self._keepalive)
            lw.w2_scales, lw.w2_scales_bytes = _arr_ptr(moe.down_proj.scales, self._keepalive)
            lw.w2_gscale, lw.w2_gscale_bytes = _arr_ptr(
                getattr(moe.down_proj, "global_scale", None), self._keepalive
            )
            if lib.inkling_ctx_register_layer(self.ctx, ctypes.byref(lw)) != 0:
                self.close()
                raise RuntimeError(f"inkling_ctx_register_layer failed for layer {idx}")

    def close(self) -> None:
        if getattr(self, "ctx", None):
            self.lib.inkling_ctx_destroy(self.ctx)
            self.ctx = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _ensure_cache_ready(self, caches: list[CacheList], m_tokens: int) -> None:
        to_eval = []
        d = self.cfg.hidden_size
        for (layer_idx, _, _), cl in zip(self._schedule, caches):
            a = self.tm.layers[layer_idx].self_attn
            kv, conv = cl.caches[0], cl.caches[1]
            ch = a.n_kv * a.head_dim

            for idx_c, width in enumerate((ch, ch, d, d)):
                st = conv[idx_c]
                if st is None:
                    conv[idx_c] = mx.zeros((1, 3, width), dtype=mx.float32)
                    to_eval.append(conv[idx_c])
                elif st.dtype != mx.float32:
                    conv[idx_c] = mx.contiguous(st.astype(mx.float32))
                    to_eval.append(conv[idx_c])

            if isinstance(kv, RotatingKVCache):
                target_cap = kv.max_size
                if kv.keys is None:
                    kv.keys = mx.zeros((1, a.n_kv, target_cap, a.head_dim), dtype=mx.bfloat16)
                    kv.values = mx.zeros((1, a.n_kv, target_cap, a.head_dim), dtype=mx.bfloat16)
                    to_eval.extend((kv.keys, kv.values))
                elif kv.keys.shape[2] > target_cap:
                    k_ord = kv._temporal_order(kv.keys) if kv._idx < kv.keys.shape[2] else kv.keys
                    v_ord = kv._temporal_order(kv.values) if kv._idx < kv.values.shape[2] else kv.values
                    kv.keys = mx.contiguous(k_ord[..., -target_cap:, :])
                    kv.values = mx.contiguous(v_ord[..., -target_cap:, :])
                    kv._idx = target_cap
                    to_eval.extend((kv.keys, kv.values))
                elif kv.keys.shape[2] < target_cap:
                    pad = target_cap - kv.keys.shape[2]
                    kv.keys = mx.concatenate(
                        [mx.contiguous(kv.keys), mx.zeros((1, a.n_kv, pad, a.head_dim), dtype=mx.bfloat16)],
                        axis=2,
                    )
                    kv.values = mx.concatenate(
                        [mx.contiguous(kv.values), mx.zeros((1, a.n_kv, pad, a.head_dim), dtype=mx.bfloat16)],
                        axis=2,
                    )
                    to_eval.extend((kv.keys, kv.values))
            else:
                need = kv.offset + m_tokens
                if kv.keys is None:
                    cap = max(kv.step, ((need + kv.step - 1) // kv.step) * kv.step)
                    kv.keys = mx.zeros((1, a.n_kv, cap, a.head_dim), dtype=mx.bfloat16)
                    kv.values = mx.zeros((1, a.n_kv, cap, a.head_dim), dtype=mx.bfloat16)
                    to_eval.extend((kv.keys, kv.values))
                elif kv.keys.shape[2] < need:
                    pad = ((need - kv.keys.shape[2] + kv.step - 1) // kv.step) * kv.step
                    kv.keys = mx.concatenate(
                        [mx.contiguous(kv.keys), mx.zeros((1, a.n_kv, pad, a.head_dim), dtype=mx.bfloat16)],
                        axis=2,
                    )
                    kv.values = mx.concatenate(
                        [mx.contiguous(kv.values), mx.zeros((1, a.n_kv, pad, a.head_dim), dtype=mx.bfloat16)],
                        axis=2,
                    )
                    to_eval.extend((kv.keys, kv.values))

        if to_eval:
            mx.eval(*to_eval)
            self.lib.inkling_ctx_clear_ptr_cache(self.ctx)

    def clear_cache_binding(self) -> None:
        """Drop held KV/conv references and clear native MTLBuffer VA cache (call when a stream ends)."""
        self._cache_binding = None
        self._batch_cache_binding = None
        self._draft_kv_binding = None
        self._mtp_out_hidden_view = None
        self._last_verify_packed = None
        self._stream_cache_bindings.clear()
        self._stream_draft_bindings.clear()
        self._stream_out_hidden_views.clear()
        if getattr(self, "ctx", None):
            self.lib.inkling_ctx_clear_ptr_cache(self.ctx)

    def release_stream_cache(
        self,
        caches: Optional[list[CacheList]] = None,
        draft_cache: Optional[list] = None,
        out_hidden_buf: Optional[mx.array] = None,
    ) -> None:
        """Release per-stream cached bindings and invalidate native VA cache on stream completion."""
        if caches is not None:
            b = self._stream_cache_bindings.get(id(caches))
            if b is not None and b["caches"] is caches:
                self._stream_cache_bindings.pop(id(caches), None)
            if self._cache_binding is not None and self._cache_binding["caches"] is caches:
                self._cache_binding = None
        if draft_cache is not None:
            db = self._stream_draft_bindings.get(id(draft_cache))
            if db is not None and db["draft_cache"] is draft_cache:
                self._stream_draft_bindings.pop(id(draft_cache), None)
            if self._draft_kv_binding is not None and self._draft_kv_binding["draft_cache"] is draft_cache:
                self._draft_kv_binding = None
        if out_hidden_buf is not None:
            ov = self._stream_out_hidden_views.get(id(out_hidden_buf))
            if ov is not None and ov[0] is out_hidden_buf:
                self._stream_out_hidden_views.pop(id(out_hidden_buf), None)
            if self._mtp_out_hidden_view is not None and self._mtp_out_hidden_view[0] is out_hidden_buf:
                self._mtp_out_hidden_view = None
        if getattr(self, "ctx", None) and (caches is not None or draft_cache is not None):
            self.lib.inkling_ctx_clear_ptr_cache(self.ctx)

    def _new_single_cache_binding(self, caches: list[CacheList]) -> dict[str, Any]:
        nv = self.n_visits
        return {
            "caches": caches,
            "kv_ptrs": (ctypes.c_void_p * (2 * nv))(),
            "conv_ptrs": (ctypes.c_void_p * (4 * nv))(),
            "kv_meta": (InklingKVMeta * nv)(),
            "kv_ids": [None] * nv,
            "conv_ids": [None] * (4 * nv),
            "kv_views": [None] * nv,
            "conv_views": [None] * (4 * nv),
        }

    def _populate_single_cache_binding(
        self, b: dict[str, Any], caches: list[CacheList]
    ) -> tuple[Any, Any, Any, dict[str, Any], bool]:
        kv_ptrs = b["kv_ptrs"]
        conv_ptrs = b["conv_ptrs"]
        kv_meta = b["kv_meta"]
        kv_ids = b["kv_ids"]
        conv_ids = b["conv_ids"]
        kv_views = b["kv_views"]
        conv_views = b["conv_views"]
        hds = self._visit_head_dims
        any_rebound = False

        for v, cl in enumerate(caches):
            kv, conv = cl.caches[0], cl.caches[1]
            is_rot = 1 if isinstance(kv, RotatingKVCache) else 0
            k_id = (id(kv.keys), id(kv.values))
            prev_kv = kv_views[v]
            if kv_ids[v] != k_id or prev_kv is None or prev_kv[0] is not kv.keys or prev_kv[1] is not kv.values:
                any_rebound = True
                kv_k_view = kv.keys.view(mx.uint16)
                kv_v_view = kv.values.view(mx.uint16)
                mx.eval(kv_k_view, kv_v_view)
                nk = np.asarray(kv_k_view)
                nv_arr = np.asarray(kv_v_view)
                if not nk.flags.c_contiguous or not nv_arr.flags.c_contiguous:
                    kv.keys = mx.contiguous(kv.keys)
                    kv.values = mx.contiguous(kv.values)
                    kv_k_view = kv.keys.view(mx.uint16)
                    kv_v_view = kv.values.view(mx.uint16)
                    mx.eval(kv_k_view, kv_v_view)
                    nk = np.asarray(kv_k_view)
                    nv_arr = np.asarray(kv_v_view)
                    k_id = (id(kv.keys), id(kv.values))
                kv_views[v] = (kv.keys, kv.values, kv_k_view, kv_v_view, nk, nv_arr)
                kv_ids[v] = k_id
                kv_ptrs[2 * v + 0] = int(nk.ctypes.data)
                kv_ptrs[2 * v + 1] = int(nv_arr.ctypes.data)
                cap = int(kv.keys.shape[2])
                km = kv_meta[v]
                km.capacity = cap
                km.max_size = int(kv.max_size) if is_rot else 0
                km.is_rotating = is_rot
                km.stride_h = cap * hds[v]
                km.stride_s = hds[v]
                km._pad = 0

            km = kv_meta[v]
            km.offset = int(kv.offset)
            km.idx = int(kv._idx) if is_rot else int(kv.offset)

            base_c = 4 * v
            for c in range(4):
                st = conv[c]
                cid = id(st)
                prev_c = conv_views[base_c + c]
                if conv_ids[base_c + c] != cid or prev_c is None or prev_c[0] is not st:
                    any_rebound = True
                    mx.eval(st)
                    ns = np.asarray(st)
                    if not ns.flags.c_contiguous:
                        st = mx.contiguous(st)
                        conv[c] = st
                        mx.eval(st)
                        ns = np.asarray(st)
                        cid = id(st)
                    conv_views[base_c + c] = (st, ns)
                    conv_ids[base_c + c] = cid
                    conv_ptrs[base_c + c] = int(ns.ctypes.data)

        return kv_ptrs, conv_ptrs, kv_meta, b, any_rebound

    def _pack_cache_args(self, caches: list[CacheList]):
        b = self._cache_binding
        switched = b is None or b["caches"] is not caches
        if switched:
            if b is not None:
                self.lib.inkling_ctx_clear_ptr_cache(self.ctx)
            b = self._new_single_cache_binding(caches)
            self._cache_binding = b
        kv_ptrs, conv_ptrs, kv_meta, b, any_rebound = self._populate_single_cache_binding(b, caches)
        if any_rebound and not switched:
            self.lib.inkling_ctx_clear_ptr_cache(self.ctx)
        return kv_ptrs, conv_ptrs, kv_meta, b

    def _pack_stream_cache_args(self, caches: list[CacheList]):
        cid = id(caches)
        b = self._stream_cache_bindings.get(cid)
        switched = b is None or b["caches"] is not caches
        if switched:
            b = self._new_single_cache_binding(caches)
            self._stream_cache_bindings[cid] = b
        kv_ptrs, conv_ptrs, kv_meta, b, any_rebound = self._populate_single_cache_binding(b, caches)
        return kv_ptrs, conv_ptrs, kv_meta, b, (switched or any_rebound)

    def step_m1_greedy(self, token_id: int, caches: list[CacheList]) -> int:
        """Single-stream plain greedy decode step (M=1) in one native Metal CB."""
        self._ensure_cache_ready(caches, 1)
        kv_ptrs, conv_ptrs, kv_meta, _keep = self._pack_cache_args(caches)
        self._toks_c[0] = int(token_id)
        stats = InklingStepStats()
        rc = self.lib.inkling_ctx_step(
            self.ctx,
            self._toks_c,
            1,
            0,
            kv_ptrs,
            conv_ptrs,
            kv_meta,
            None,
            0,
            None,
            self._toks_out_c,
            None,
            None,
            1,
            ctypes.byref(stats),
        )
        if rc != 0:
            raise RuntimeError(f"inkling_ctx_step(M=1) failed with rc={rc}")
        for v, cl in enumerate(caches):
            kv, conv = cl.caches[0], cl.caches[1]
            kv.offset = int(kv_meta[v].offset)
            if isinstance(kv, RotatingKVCache):
                kv._idx = int(kv_meta[v].idx)
            conv.advance(1)
        return int(self._toks_out_c[0])

    def step_m1_greedy_capture(
        self,
        token_id: int,
        caches: list[CacheList],
        capture_layer_ids: tuple[int, ...],
    ) -> tuple[int, mx.array]:
        """Single-stream plain M=1 greedy decode with MTP layer capture (<0.02 ms extra)."""
        self._ensure_cache_ready(caches, 1)
        kv_ptrs, conv_ptrs, kv_meta, _keep = self._pack_cache_args(caches)
        self._toks_c[0] = int(token_id)

        n_cap = len(capture_layer_ids)
        cap_ids_c = self._cap_ids_cache.get(capture_layer_ids)
        if cap_ids_c is None:
            cap_ids_c = (ctypes.c_int32 * n_cap)(*capture_layer_ids)
            self._cap_ids_cache[capture_layer_ids] = cap_ids_c
        cbuf = self._cap_bufs.get(n_cap)
        if cbuf is None:
            b_mx = mx.zeros((1, self._max_m, n_cap * self.cfg.hidden_size), dtype=mx.bfloat16)
            b_v = b_mx.view(mx.uint16)
            mx.eval(b_v)
            b_np = np.asarray(b_v)
            cbuf = (b_mx, b_v, b_np)
            self._cap_bufs[n_cap] = cbuf

        stats = InklingStepStats()
        rc = self.lib.inkling_ctx_step(
            self.ctx,
            self._toks_c,
            1,
            0,
            kv_ptrs,
            conv_ptrs,
            kv_meta,
            cap_ids_c,
            n_cap,
            None,
            self._toks_out_c,
            ctypes.c_void_p(int(cbuf[2].ctypes.data)),
            None,
            1,
            ctypes.byref(stats),
        )
        if rc != 0:
            raise RuntimeError(f"inkling_ctx_step(M=1 capture) failed with rc={rc}")
        for v, cl in enumerate(caches):
            kv, conv = cl.caches[0], cl.caches[1]
            kv.offset = int(kv_meta[v].offset)
            if isinstance(kv, RotatingKVCache):
                kv._idx = int(kv_meta[v].idx)
            conv.advance(1)
        cap_copy = mx.array(cbuf[0][:, :1, :])
        mx.eval(cap_copy)
        return int(self._toks_out_c[0]), cap_copy

    def step_m1_prenorm(self, token_id: int, caches: list[CacheList]) -> mx.array:
        """Single-stream sampled decode step (M=1): returns pre-norm FP32 [1, 1, D]."""
        self._ensure_cache_ready(caches, 1)
        kv_ptrs, conv_ptrs, kv_meta, _keep = self._pack_cache_args(caches)
        self._toks_c[0] = int(token_id)
        stats = InklingStepStats()
        rc = self.lib.inkling_ctx_step(
            self.ctx,
            self._toks_c,
            1,
            0,
            kv_ptrs,
            conv_ptrs,
            kv_meta,
            None,
            0,
            ctypes.c_void_p(int(self._pre_norm_np.ctypes.data)),
            None,
            None,
            None,
            1,
            ctypes.byref(stats),
        )
        if rc != 0:
            raise RuntimeError(f"inkling_ctx_step(M=1 prenorm) failed with rc={rc}")
        for v, cl in enumerate(caches):
            kv, conv = cl.caches[0], cl.caches[1]
            kv.offset = int(kv_meta[v].offset)
            if isinstance(kv, RotatingKVCache):
                kv._idx = int(kv_meta[v].idx)
            conv.advance(1)
        return self._pre_norm_mx[:, :1, :]

    def step_verify(
        self,
        token_ids: list[int],
        caches: list[CacheList],
        capture_layer_ids: tuple[int, ...],
        *,
        greedy: bool = True,
    ) -> tuple[Optional[list[int]], Optional[mx.array], mx.array]:
        """Single-CB sequential MTP verify (M=2..8) with staged rotating KV & conv history.

        Call `commit_verify(caches, n_accepted, len(token_ids))` immediately after matching!
        Returns `(argmax_list_or_None, pre_norm_or_None, captures_bf16 [1, M, n_cap*D])`.
        """
        m = len(token_ids)
        self._ensure_cache_ready(caches, m)
        kv_ptrs, conv_ptrs, kv_meta, _keep = self._pack_cache_args(caches)
        for i in range(m):
            self._toks_c[i] = int(token_ids[i])

        n_cap = len(capture_layer_ids)
        cap_ids_c = self._cap_ids_cache.get(capture_layer_ids)
        if cap_ids_c is None:
            cap_ids_c = (ctypes.c_int32 * n_cap)(*capture_layer_ids)
            self._cap_ids_cache[capture_layer_ids] = cap_ids_c

        cbuf = self._cap_bufs.get(n_cap)
        if cbuf is None:
            b_mx = mx.zeros((1, self._max_m, n_cap * self.cfg.hidden_size), dtype=mx.bfloat16)
            b_v = b_mx.view(mx.uint16)
            mx.eval(b_v)
            b_np = np.asarray(b_v)
            cbuf = (b_mx, b_v, b_np)
            self._cap_bufs[n_cap] = cbuf
        cap_ptr = int(cbuf[2].ctypes.data)

        pre_norm_ptr = 0 if greedy else int(self._pre_norm_np.ctypes.data)
        toks_out_c = self._toks_out_c if greedy else None

        stats = InklingStepStats()
        rc = self.lib.inkling_ctx_step(
            self.ctx,
            self._toks_c,
            m,
            1,
            kv_ptrs,
            conv_ptrs,
            kv_meta,
            cap_ids_c,
            n_cap,
            ctypes.c_void_p(pre_norm_ptr) if pre_norm_ptr else None,
            toks_out_c,
            ctypes.c_void_p(cap_ptr),
            None,
            0,
            ctypes.byref(stats),
        )
        if rc != 0:
            raise RuntimeError(f"inkling_ctx_step(verify M={m}) failed with rc={rc}")

        self._last_verify_packed = (caches, kv_ptrs, conv_ptrs, kv_meta, _keep)
        argmax = [int(self._toks_out_c[i]) for i in range(m)] if greedy else None
        pre_norm = None if greedy else self._pre_norm_mx[:, :m, :]
        return argmax, pre_norm, cbuf[0][:, :m, :]

    def commit_verify(self, caches: list[CacheList], n_accepted: int, m_tokens: int) -> None:
        """Commit `n_accepted` tokens in-place (KV offsets/ring + conv states, ~0.09 ms)."""
        packed = self._last_verify_packed
        self._last_verify_packed = None
        if packed is not None and packed[0] is caches:
            _, kv_ptrs, conv_ptrs, kv_meta, _keep = packed
        else:
            kv_ptrs, conv_ptrs, kv_meta, _keep = self._pack_cache_args(caches)
        rc = self.lib.inkling_ctx_commit_verify(
            self.ctx, kv_ptrs, conv_ptrs, kv_meta, int(n_accepted), int(m_tokens)
        )
        if rc != 0:
            raise RuntimeError(f"inkling_ctx_commit_verify failed with rc={rc}")
        for v, cl in enumerate(caches):
            kv, conv = cl.caches[0], cl.caches[1]
            kv.offset = int(kv_meta[v].offset)
            if isinstance(kv, RotatingKVCache):
                kv._idx = int(kv_meta[v].idx)
            if n_accepted > 0:
                conv.advance(n_accepted)

    def attach_mtp(self, drafter) -> bool:
        """Attach zero-copy built-in MTP drafter weights onto this unified native context."""
        if self.mtp_attached and self._mtp_drafter is drafter:
            return True
        if getattr(drafter, "_target", None) is not self.lm:
            return False
        keep: list[Any] = []
        c_cfg, c_glob, c_layers, cap_ids = build_mtp_c_descriptors(drafter, keep)
        cap_c = (ctypes.c_int32 * len(cap_ids))(*cap_ids)
        rc = self.lib.inkling_ctx_attach_mtp(
            self.ctx,
            ctypes.byref(c_cfg),
            ctypes.byref(c_glob),
            c_layers,
            cap_c,
            len(cap_ids),
        )
        if rc != 0:
            raise RuntimeError(f"inkling_ctx_attach_mtp failed with rc={rc}")
        keep.append((c_cfg, c_glob, c_layers, cap_c))
        self._mtp_keepalive = keep
        self._mtp_cfg = c_cfg
        self._mtp_drafter = drafter
        self._mtp_d_descs = (MTPKVCacheDesc * c_cfg.num_layers)()
        self.mtp_attached = True
        return True

    def _ensure_draft_kv_ready(self, draft_cache: list, n_ctx: int) -> None:
        mc = self._mtp_cfg
        assert mc is not None
        kv_h, hd = int(mc.num_kv_heads), int(mc.head_dim)
        to_eval = []
        for entry in draft_cache:
            if isinstance(entry, RotatingKVCache):
                w = int(entry.max_size)
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
                    cur_len = int(k_ord.shape[2])
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
                need = int(entry.offset) + int(n_ctx)
                if entry.keys is None:
                    cap = max(2048, ((need + 2047) // 2048) * 2048)
                    entry.keys = mx.zeros((1, kv_h, cap, hd), dtype=mx.bfloat16)
                    entry.values = mx.zeros((1, kv_h, cap, hd), dtype=mx.bfloat16)
                    to_eval.extend((entry.keys, entry.values))
                elif entry.keys.shape[2] < need:
                    new_cap = ((need + 2047) // 2048) * 2048
                    pad = int(new_cap) - int(entry.keys.shape[2])
                    entry.keys = mx.contiguous(
                        mx.concatenate([entry.keys, mx.zeros((1, kv_h, pad, hd), dtype=mx.bfloat16)], axis=2)
                    )
                    entry.values = mx.contiguous(
                        mx.concatenate([entry.values, mx.zeros((1, kv_h, pad, hd), dtype=mx.bfloat16)], axis=2)
                    )
                    to_eval.extend((entry.keys, entry.values))
        if to_eval:
            mx.eval(*to_eval)

    def _populate_draft_kv_binding(self, b: dict[str, Any], draft_cache: list) -> tuple[Any, dict[str, Any]]:
        mc = self._mtp_cfg
        assert mc is not None
        hd = int(mc.head_dim)
        descs = b["descs"]
        kv_views = b["kv_views"]
        for i, dc in enumerate(draft_cache):
            prev = kv_views[i]
            if prev is None or prev[0] is not dc.keys or prev[1] is not dc.values:
                kv = dc.keys.view(mx.uint16)
                vv = dc.values.view(mx.uint16)
                mx.eval(kv, vv)
                nk = np.asarray(kv)
                nv = np.asarray(vv)
                if not nk.flags.c_contiguous or not nv.flags.c_contiguous:
                    dc.keys = mx.contiguous(dc.keys)
                    dc.values = mx.contiguous(dc.values)
                    kv = dc.keys.view(mx.uint16)
                    vv = dc.values.view(mx.uint16)
                    mx.eval(kv, vv)
                    nk = np.asarray(kv)
                    nv = np.asarray(vv)
                prev = (dc.keys, dc.values, kv, vv, nk, nv)
                kv_views[i] = prev
            cap = int(dc.keys.shape[2])
            is_rot = 1 if isinstance(dc, RotatingKVCache) else 0
            descs[i].k_ptr = int(prev[4].ctypes.data)
            descs[i].v_ptr = int(prev[5].ctypes.data)
            descs[i].k_nbytes = int(prev[4].nbytes)
            descs[i].v_nbytes = int(prev[5].nbytes)
            descs[i].cache_cap = cap
            descs[i].offset = int(dc.offset)
            descs[i].is_rotating = is_rot
            descs[i].window = int(dc.max_size) if is_rot else 0
            descs[i].ring_idx = int(dc._idx) if is_rot else int(dc.offset)
            descs[i].stride_h = cap * hd
            descs[i].stride_s = hd
            descs[i]._pad = 0
        return descs, b

    def _pack_draft_kv_args(self, draft_cache: list):
        mc = self._mtp_cfg
        assert mc is not None
        nl = int(mc.num_layers)
        b = self._draft_kv_binding
        if b is None or b["draft_cache"] is not draft_cache:
            b = {
                "draft_cache": draft_cache,
                "descs": self._mtp_d_descs,
                "kv_views": [None] * nl,
            }
            self._draft_kv_binding = b
        return self._populate_draft_kv_binding(b, draft_cache)

    def _pack_stream_draft_kv_args(self, draft_cache: list):
        mc = self._mtp_cfg
        assert mc is not None
        nl = int(mc.num_layers)
        did = id(draft_cache)
        b = self._stream_draft_bindings.get(did)
        if b is None or b["draft_cache"] is not draft_cache:
            b = {
                "draft_cache": draft_cache,
                "descs": (MTPKVCacheDesc * nl)(),
                "kv_views": [None] * nl,
            }
            self._stream_draft_bindings[did] = b
        return self._populate_draft_kv_binding(b, draft_cache)

    def step_mtp_single(
        self,
        anchor_token: int,
        draft_hidden: mx.array,
        draft_cache: list,
        caches: list[CacheList],
        num_draft: int,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        logit_softcap: float = 30.0,
        rng_seed: int = 0,
        rng_step: int = 0,
        out_hidden_buf: Optional[mx.array] = None,
    ) -> tuple[list[int], mx.array]:
        """Execute 1-CB B=1 GPU-accept MTP step (draft + verify + accept/reject + commit + capture copy)."""
        mc = self._mtp_cfg
        assert self.mtp_attached and mc is not None
        n_ctx = int(draft_hidden.shape[1])
        d = int(mc.hidden_size)
        n_cap = int(mc.n_target_layers)
        blk = int(mc.block_size)

        self._ensure_cache_ready(caches, num_draft + 1)
        kv_ptrs, conv_ptrs, kv_meta, _t_keep = self._pack_cache_args(caches)
        self._ensure_draft_kv_ready(draft_cache, n_ctx)
        d_descs, _d_keep = self._pack_draft_kv_args(draft_cache)

        in_v = draft_hidden.view(mx.uint16)
        mx.eval(in_v)
        in_np = np.asarray(in_v)
        if not in_np.flags.c_contiguous:
            in_c = mx.contiguous(draft_hidden)
            in_v = in_c.view(mx.uint16)
            mx.eval(in_v)
            in_np = np.asarray(in_v)
        else:
            in_c = draft_hidden

        if out_hidden_buf is None:
            out_hidden_buf = mx.zeros((1, blk, n_cap * d), dtype=mx.bfloat16)
            mx.eval(out_hidden_buf)
        ov_cache = self._mtp_out_hidden_view
        if ov_cache is not None and ov_cache[0] is out_hidden_buf:
            out_v, out_np = ov_cache[1], ov_cache[2]
        else:
            out_v = out_hidden_buf.view(mx.uint16)
            mx.eval(out_v)
            out_np = np.asarray(out_v)
            self._mtp_out_hidden_view = (out_hidden_buf, out_v, out_np)

        stats = InklingStepStats()
        rc = self.lib.inkling_ctx_mtp_step(
            self.ctx,
            int(anchor_token),
            ctypes.c_void_p(int(in_np.ctypes.data)),
            int(in_np.nbytes),
            ctypes.c_void_p(int(out_np.ctypes.data)),
            int(out_np.nbytes),
            n_ctx,
            int(num_draft),
            float(temperature),
            float(top_p),
            float(logit_softcap),
            int(rng_seed),
            int(rng_step),
            kv_ptrs,
            conv_ptrs,
            kv_meta,
            d_descs,
            self._mtp_out_toks_c,
            ctypes.byref(self._mtp_out_n_acc_c),
            ctypes.byref(stats),
        )
        if rc != 0:
            raise RuntimeError(f"inkling_ctx_mtp_step(K={num_draft}, T={temperature}) failed with rc={rc}")

        n_acc = int(self._mtp_out_n_acc_c.value)
        for v, cl in enumerate(caches):
            kv, conv = cl.caches[0], cl.caches[1]
            kv.offset = int(kv_meta[v].offset)
            if isinstance(kv, RotatingKVCache):
                kv._idx = int(kv_meta[v].idx)
            conv.advance(n_acc)
        for i, dc in enumerate(draft_cache):
            dc.offset = int(d_descs[i].offset)
            if d_descs[i].is_rotating:
                dc._idx = int(d_descs[i].ring_idx)

        new_toks = [int(self._mtp_out_toks_c[j]) for j in range(n_acc)]
        del in_c, in_v, in_np
        return new_toks, out_hidden_buf[:, :n_acc, :]

    def step_mtp_batched(self, items: list[dict[str, Any]]) -> list[tuple[list[int], mx.array]]:
        """Execute 1-CB B=1..4 batched GPU-accept MTP step across independent unmerged streams."""
        mc = self._mtp_cfg
        assert self.mtp_attached and mc is not None
        bsz = len(items)
        if bsz < 1 or bsz > 4:
            raise ValueError(f"step_mtp_batched batch_size={bsz} out of [1, 4]")
        total_v = sum(int(it["num_draft"]) + 1 for it in items)
        if total_v > self._max_m or total_v > 32:
            raise ValueError(f"step_mtp_batched total verify tokens={total_v} exceeds max_m={self._max_m}")

        d = int(mc.hidden_size)
        n_cap = int(mc.n_target_layers)
        blk = int(mc.block_size)

        for it in items:
            self._ensure_cache_ready(it["caches"], int(it["num_draft"]) + 1)
            self._ensure_draft_kv_ready(it["draft_cache"], int(it["draft_hidden"].shape[1]))

        any_rebound = False
        keep_views: list[Any] = []
        packed_meta: list[tuple[list[CacheList], Any, list, Any, mx.array]] = []
        descs = self._mtp_batch_descs

        for b, it in enumerate(items):
            caches = it["caches"]
            draft_cache = it["draft_cache"]
            draft_hidden = it["draft_hidden"]
            num_draft = int(it["num_draft"])
            n_ctx = int(draft_hidden.shape[1])

            kv_ptrs, conv_ptrs, kv_meta, t_keep, rebound = self._pack_stream_cache_args(caches)
            any_rebound = any_rebound or rebound
            d_descs, d_keep = self._pack_stream_draft_kv_args(draft_cache)

            in_v = draft_hidden.view(mx.uint16)
            mx.eval(in_v)
            in_np = np.asarray(in_v)
            if not in_np.flags.c_contiguous:
                in_c = mx.contiguous(draft_hidden)
                in_v = in_c.view(mx.uint16)
                mx.eval(in_v)
                in_np = np.asarray(in_v)
            else:
                in_c = draft_hidden

            out_hidden_buf = it.get("out_hidden_buf")
            if out_hidden_buf is None:
                out_hidden_buf = mx.zeros((1, blk, n_cap * d), dtype=mx.bfloat16)
                mx.eval(out_hidden_buf)
            ov = self._stream_out_hidden_views.get(id(out_hidden_buf))
            if ov is not None and ov[0] is out_hidden_buf:
                out_v, out_np = ov[1], ov[2]
            else:
                out_v = out_hidden_buf.view(mx.uint16)
                mx.eval(out_v)
                out_np = np.asarray(out_v)
                self._stream_out_hidden_views[id(out_hidden_buf)] = (out_hidden_buf, out_v, out_np)

            keep_views.append((t_keep, d_keep, in_c, in_v, in_np, out_hidden_buf, out_v, out_np))
            packed_meta.append((caches, kv_meta, draft_cache, d_descs, out_hidden_buf))

            sd = descs[b]
            sd.anchor_token = int(it["anchor_token"])
            sd.n_ctx = n_ctx
            sd.num_draft = num_draft
            sd._pad0 = 0
            sd.in_hidden_ptr = ctypes.c_void_p(int(in_np.ctypes.data))
            sd.in_hidden_bytes = int(in_np.nbytes)
            sd.out_hidden_ptr = ctypes.c_void_p(int(out_np.ctypes.data))
            sd.out_hidden_bytes = int(out_np.nbytes)
            sd.temperature = float(it.get("temperature", 0.0))
            sd.top_p = float(it.get("top_p", 1.0))
            sd.logit_softcap = float(it.get("logit_softcap", 30.0))
            sd._pad1 = 0
            sd.rng_seed = int(it.get("rng_seed", 0))
            sd.rng_step = int(it.get("rng_step", 0))
            sd.kv_ptrs = kv_ptrs
            sd.conv_ptrs = conv_ptrs
            sd.kv_meta = kv_meta
            sd.draft_kv_descs = d_descs
            sd.out_n_acc = 0
            sd._pad2 = 0

        if any_rebound:
            self.lib.inkling_ctx_clear_ptr_cache(self.ctx)

        stats = InklingStepStats()
        rc = self.lib.inkling_ctx_mtp_step_batched(self.ctx, descs, bsz, ctypes.byref(stats))
        if rc != 0:
            raise RuntimeError(f"inkling_ctx_mtp_step_batched(B={bsz}) failed with rc={rc}")

        results: list[tuple[list[int], mx.array]] = []
        for b in range(bsz):
            sd = descs[b]
            caches, kv_meta, draft_cache, d_descs, out_hidden_buf = packed_meta[b]
            n_acc = int(sd.out_n_acc)
            for v, cl in enumerate(caches):
                kv, conv = cl.caches[0], cl.caches[1]
                kv.offset = int(kv_meta[v].offset)
                if isinstance(kv, RotatingKVCache):
                    kv._idx = int(kv_meta[v].idx)
                conv.advance(n_acc)
            for i, dc in enumerate(draft_cache):
                dc.offset = int(d_descs[i].offset)
                if d_descs[i].is_rotating:
                    dc._idx = int(d_descs[i].ring_idx)
            new_toks = [int(sd.out_tokens[j]) for j in range(n_acc)]
            results.append((new_toks, out_hidden_buf[:, :n_acc, :]))

        keep_views.clear()
        return results

    def _ensure_batch_cache_ready(self, caches: list[CacheList], batch_size: int) -> None:
        to_eval = []
        d = self.cfg.hidden_size
        for (layer_idx, _, _), cl in zip(self._schedule, caches):
            a = self.tm.layers[layer_idx].self_attn
            kv, conv = cl.caches[0], cl.caches[1]
            ch = a.n_kv * a.head_dim

            for idx_c, width in enumerate((ch, ch, d, d)):
                st = conv[idx_c]
                if st is None:
                    conv[idx_c] = mx.zeros((batch_size, 3, width), dtype=mx.float32)
                    to_eval.append(conv[idx_c])
                elif st.dtype != mx.float32:
                    conv[idx_c] = mx.contiguous(st.astype(mx.float32))
                    to_eval.append(conv[idx_c])

            if isinstance(kv, BatchRotatingKVCache):
                if kv.keys is not None and kv.keys.shape[2] > kv.max_size:
                    trim = kv.keys.shape[2] - kv.max_size
                    kv.keys = mx.contiguous(kv.keys[..., trim:, :])
                    kv.values = mx.contiguous(kv.values[..., trim:, :])
                    kv._idx = kv.max_size
                    kv.left_padding = kv.left_padding - trim
                    to_eval.extend((kv.keys, kv.values, kv.left_padding))
                target_cap = kv.max_size
                if kv.keys is None:
                    kv.keys = mx.zeros((batch_size, a.n_kv, target_cap, a.head_dim), dtype=mx.bfloat16)
                    kv.values = mx.zeros((batch_size, a.n_kv, target_cap, a.head_dim), dtype=mx.bfloat16)
                    to_eval.extend((kv.keys, kv.values))
                elif kv.keys.shape[2] < target_cap:
                    pad = target_cap - kv.keys.shape[2]
                    kv.keys = mx.concatenate(
                        [mx.contiguous(kv.keys), mx.zeros((batch_size, a.n_kv, pad, a.head_dim), dtype=mx.bfloat16)],
                        axis=2,
                    )
                    kv.values = mx.concatenate(
                        [mx.contiguous(kv.values), mx.zeros((batch_size, a.n_kv, pad, a.head_dim), dtype=mx.bfloat16)],
                        axis=2,
                    )
                    to_eval.extend((kv.keys, kv.values))
            else:
                need = kv._idx + 1
                if kv.keys is None:
                    cap = max(kv.step, ((need + kv.step - 1) // kv.step) * kv.step)
                    kv.keys = mx.zeros((batch_size, a.n_kv, cap, a.head_dim), dtype=mx.bfloat16)
                    kv.values = mx.zeros((batch_size, a.n_kv, cap, a.head_dim), dtype=mx.bfloat16)
                    to_eval.extend((kv.keys, kv.values))
                elif kv.keys.shape[2] < need:
                    pad = ((need - kv.keys.shape[2] + kv.step - 1) // kv.step) * kv.step
                    kv.keys = mx.concatenate(
                        [mx.contiguous(kv.keys), mx.zeros((batch_size, a.n_kv, pad, a.head_dim), dtype=mx.bfloat16)],
                        axis=2,
                    )
                    kv.values = mx.concatenate(
                        [mx.contiguous(kv.values), mx.zeros((batch_size, a.n_kv, pad, a.head_dim), dtype=mx.bfloat16)],
                        axis=2,
                    )
                    to_eval.extend((kv.keys, kv.values))

        if to_eval:
            mx.eval(*to_eval)
            self.lib.inkling_ctx_clear_ptr_cache(self.ctx)

    def sync_batch_cache(self, caches: Optional[list[CacheList]] = None) -> None:
        """Materialize deferred native offset & left_padding updates onto MLX BatchKVCache objects."""
        b = self._batch_cache_binding
        if b is None or not b["dirty"]:
            return
        if caches is not None and b["caches"] is not caches:
            return
        rot_steps = b["rot_steps"]
        row_offsets = b["row_offsets"]
        off_refs = b["off_refs"]
        to_eval = []
        for v, cl in enumerate(b["caches"]):
            kv = cl.caches[0]
            rs = rot_steps[v]
            if isinstance(kv, BatchRotatingKVCache) and rs > 0:
                kv.left_padding = kv.left_padding - rs
                to_eval.append(kv.left_padding)
                rot_steps[v] = 0
            new_off = mx.array(row_offsets[v], dtype=mx.int32)
            kv.offset = new_off
            off_refs[v] = new_off
            to_eval.append(new_off)
        if to_eval:
            mx.eval(*to_eval)
        b["dirty"] = False

    def _pack_batch_cache_args(self, caches: list[CacheList], batch_size: int):
        nv = self.n_visits
        b = self._batch_cache_binding
        if b is None or b["caches"] is not caches or b["nv"] != nv or b["batch_size"] != batch_size:
            if b is not None:
                self.sync_batch_cache(b["caches"])
                self.lib.inkling_ctx_clear_ptr_cache(self.ctx)
            b = {
                "caches": caches,
                "nv": nv,
                "batch_size": batch_size,
                "kv_ptrs": (ctypes.c_void_p * (2 * nv))(),
                "conv_ptrs": (ctypes.c_void_p * (4 * nv))(),
                "kv_meta": (InklingBatchKVMeta * nv)(),
                "row_offsets": np.zeros((nv, batch_size), dtype=np.int32),
                "rot_steps": [0] * nv,
                "kv_views": [None] * nv,
                "conv_views": [None] * (4 * nv),
                "off_refs": [None] * nv,
                "dirty": False,
            }
            self._batch_cache_binding = b

        kv_ptrs = b["kv_ptrs"]
        conv_ptrs = b["conv_ptrs"]
        kv_meta = b["kv_meta"]
        row_offsets = b["row_offsets"]
        kv_views = b["kv_views"]
        conv_views = b["conv_views"]
        off_refs = b["off_refs"]
        hds = self._visit_head_dims
        any_rebound = False

        for v, cl in enumerate(caches):
            kv, conv = cl.caches[0], cl.caches[1]
            is_rot = 1 if isinstance(kv, BatchRotatingKVCache) else 0
            prev_kv = kv_views[v]
            if prev_kv is None or prev_kv[0] is not kv.keys or prev_kv[1] is not kv.values:
                any_rebound = True
                kv_k_view = kv.keys.view(mx.uint16)
                kv_v_view = kv.values.view(mx.uint16)
                mx.eval(kv_k_view, kv_v_view)
                nk = np.asarray(kv_k_view)
                nv_arr = np.asarray(kv_v_view)
                if not nk.flags.c_contiguous or not nv_arr.flags.c_contiguous:
                    kv.keys = mx.contiguous(kv.keys)
                    kv.values = mx.contiguous(kv.values)
                    kv_k_view = kv.keys.view(mx.uint16)
                    kv_v_view = kv.values.view(mx.uint16)
                    mx.eval(kv_k_view, kv_v_view)
                    nk = np.asarray(kv_k_view)
                    nv_arr = np.asarray(kv_v_view)
                kv_views[v] = (kv.keys, kv.values, kv_k_view, kv_v_view, nk, nv_arr)
                kv_ptrs[2 * v + 0] = int(nk.ctypes.data)
                kv_ptrs[2 * v + 1] = int(nv_arr.ctypes.data)
                cap = int(kv.keys.shape[2])
                n_kv = int(kv.keys.shape[1])
                km = kv_meta[v]
                km.capacity = cap
                km.max_size = int(kv.max_size) if is_rot else 0
                km.is_rotating = is_rot
                km.stride_s = hds[v]
                km.stride_h = cap * hds[v]
                km.stride_b = n_kv * cap * hds[v]

            km = kv_meta[v]
            km.idx = int(kv._idx)
            km.offset = int(kv._offset) if is_rot else 0

            if off_refs[v] is not kv.offset:
                off_arr = mx.contiguous(kv.offset.astype(mx.int32))
                mx.eval(off_arr)
                row_offsets[v] = np.asarray(off_arr)
                off_refs[v] = kv.offset
                b["rot_steps"][v] = 0

            base_c = 4 * v
            for c in range(4):
                st = conv[c]
                prev_c = conv_views[base_c + c]
                if prev_c is None or prev_c[0] is not st:
                    any_rebound = True
                    mx.eval(st)
                    ns = np.asarray(st)
                    if not ns.flags.c_contiguous:
                        st = mx.contiguous(st)
                        conv[c] = st
                        mx.eval(st)
                        ns = np.asarray(st)
                    conv_views[base_c + c] = (st, ns)
                    conv_ptrs[base_c + c] = int(ns.ctypes.data)

        if any_rebound:
            self.lib.inkling_ctx_clear_ptr_cache(self.ctx)

        return kv_ptrs, conv_ptrs, kv_meta, b

    def _step_batch_raw(
        self,
        token_ids: list[int],
        caches: list[CacheList],
        *,
        want_pre_norm: bool,
        want_argmax: bool,
    ):
        bsz = len(token_ids)
        self._ensure_batch_cache_ready(caches, bsz)
        kv_ptrs, conv_ptrs, kv_meta, b_state = self._pack_batch_cache_args(caches, bsz)
        for i in range(bsz):
            self._toks_c[i] = int(token_ids[i])
        pre_norm_ptr = int(self._pre_norm_np.ctypes.data) if want_pre_norm else 0
        toks_out_c = self._toks_out_c if want_argmax else None
        row_c = b_state["row_offsets"].ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        stats = InklingStepStats()
        rc = self.lib.inkling_ctx_step_batched(
            self.ctx,
            self._toks_c,
            bsz,
            kv_ptrs,
            conv_ptrs,
            kv_meta,
            row_c,
            ctypes.c_void_p(pre_norm_ptr) if pre_norm_ptr else None,
            toks_out_c,
            None,
            ctypes.byref(stats),
        )
        if rc != 0:
            raise RuntimeError(f"inkling_ctx_step_batched(B={bsz}) failed with rc={rc}")

        rot_steps = b_state["rot_steps"]
        for v, cl in enumerate(caches):
            kv, conv = cl.caches[0], cl.caches[1]
            if isinstance(kv, BatchRotatingKVCache):
                if int(kv._idx) == kv.max_size:
                    kv.rotated = True
                if kv.rotated:
                    rot_steps[v] += 1
                kv._idx = int(kv_meta[v].idx)
                kv._offset = int(kv_meta[v].offset)
            else:
                kv._idx = int(kv_meta[v].idx)
            conv.advance(1)
        b_state["dirty"] = True

    def step_batch_greedy(self, token_ids: list[int], caches: list[CacheList]) -> list[int]:
        """Batched greedy decode step (B=1..64) in one native Metal CB."""
        self._step_batch_raw(token_ids, caches, want_pre_norm=False, want_argmax=True)
        return [int(self._toks_out_c[i]) for i in range(len(token_ids))]

    def step_batch_prenorm(self, token_ids: list[int], caches: list[CacheList]) -> mx.array:
        """Batched sampled decode step (B=1..64): returns pre-norm FP32 [B, 1, D]."""
        self._step_batch_raw(token_ids, caches, want_pre_norm=True, want_argmax=False)
        d = self.cfg.hidden_size
        return self._pre_norm_mx.reshape(self._max_m, 1, d)[: len(token_ids)]


def try_create_native_decode_runner(lm, max_m: int = 64) -> Optional[NativeDecodeRunner]:
    """Instantiate NativeDecodeRunner if GPU BF16 NVFP4/MXFP8 model + dylib are available."""
    if mx.default_device() != mx.gpu:
        return None
    if lm.model.embed_tokens.weight.dtype != mx.bfloat16 or lm.lm_head.weight.dtype != mx.bfloat16:
        return None
    for layer in lm.model.layers:
        if getattr(layer.mlp.w13_proj, "mode", None) not in ("nvfp4", "mxfp8"):
            return None
    lib = _load_lib()
    if lib is None:
        return None
    try:
        return NativeDecodeRunner(lib, lm, max_m=max_m)
    except Exception:
        return None
