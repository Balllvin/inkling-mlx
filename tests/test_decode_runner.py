#!/usr/bin/env python3
"""Synthetic parity & latency verification for libinkling_decode.dylib.

Strictly uses tiny synthetic Inkling models (<= 8 MB total) without loading full checkpoint weights.

Verifies against Python fused.{decode_step, verify_step} + LanguageModel:
  1. M=1 plain decode across pre-rotation, rotation boundary, post-rotation SWA,
     and long full-attention (both single-pass _attend AND split-K decode_attention).
  2. M=6 sequential MTP verify (seq=True) across pre-rotation, crossing rotation,
     post-rotation, and long full-attention:
       - pre_norm (float32): max_abs_err <= 1e-3
       - captures at loop boundaries & mid-loop & final norm: max_abs_err <= 1e-3
       - per-position conv_states (ks, vs, cs, ms) for all visits: max_abs_err <= 1e-3
       - updated live conv[0..3] & KV caches: max_abs_err <= 1e-3
       - greedy argmax tokens: 0 mismatches (100% exact match!)
  3. Zero-copy post-rotation MTP verify rollback (commit_rotating=0 ->
     inkling_ctx_commit_rotating(n_accepted=3)) vs Python rollback_single_cache.
  4. Single-CB 48-visit dispatch latency benchmark vs Python fused.decode_step.
"""

from __future__ import annotations

import ctypes
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mlx_vlm.models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchRotatingKVCache,
    CacheList,
    KVCache,
    RotatingKVCache,
)

from inkling_mlx import fused, moe_kernels
from inkling_mlx.config import LoopConfig, TextConfig
from inkling_mlx.model import (
    GrowingKVCache,
    LanguageModel,
    QuantizedExpertLinear,
)
from inkling_mlx.mtp import (
    InklingMTPDrafter,
    MTPConfig,
    _target_probs,
    reject_sample_mtp,
    rollback_single_cache,
    snapshot_rotating_kv,
)

LIB_PATH = Path(__file__).resolve().parents[1] / "src" / "inkling_mlx" / "native" / "libinkling_decode.dylib"


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


def load_lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(LIB_PATH))
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
    lib.inkling_ctx_conv_states_floats.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.inkling_ctx_conv_states_floats.restype = ctypes.c_size_t
    lib.inkling_ctx_visit_conv_offset.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32]
    lib.inkling_ctx_visit_conv_offset.restype = ctypes.c_size_t
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
    lib.inkling_ctx_commit_rotating.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(InklingKVMeta),
        ctypes.c_int32,
    ]
    lib.inkling_ctx_commit_rotating.restype = ctypes.c_int
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
    lib.inkling_ctx_mtp_debug_read.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.inkling_ctx_mtp_debug_read.restype = ctypes.c_int
    return lib


def arr_ptr(arr: mx.array | None, keepalive: list[Any]) -> tuple[int, int]:
    """Return zero-copy (ptr, nbytes) for a contiguous evaluated mx.array."""
    if arr is None:
        return 0, 0
    a = mx.contiguous(arr)
    view = a.view(mx.uint16) if a.dtype == mx.bfloat16 else a
    mx.eval(view)
    nv = np.asarray(view)
    keepalive.append((a, view, nv))
    return int(nv.ctypes.data), int(nv.nbytes)


class NativeDecodeRunner:
    """Thin zero-copy Python wrapper around libinkling_decode.dylib."""

    def __init__(self, lib: ctypes.CDLL, lm: LanguageModel, max_m: int = 8, use_bf16_opt: bool = True):
        self.lib = lib
        self.lm = lm
        self.tm = lm.model
        self.cfg = lm.config
        self._max_m = max_m
        self.use_bf16_opt = use_bf16_opt
        self._keepalive: list[Any] = []
        self._conv_out_bufs: dict[int, tuple[mx.array, np.ndarray]] = {}
        self._cap_bufs: dict[int, tuple[mx.array, mx.array, np.ndarray]] = {}
        self._ext_cap_cache: dict[int, tuple[mx.array, np.ndarray]] = {}
        self._pre_norm_mx = mx.zeros((1, max_m, self.cfg.hidden_size), dtype=mx.float32)
        mx.eval(self._pre_norm_mx)
        self._pre_norm_np = np.asarray(self._pre_norm_mx)
        self._toks_c = (ctypes.c_int32 * max_m)()
        self._toks_out_c = (ctypes.c_int32 * max_m)()
        self._schedule = self.tm.schedule()
        self._visit_head_dims = [self.tm.layers[li].self_attn.head_dim for li, _, _ in self._schedule]
        self._visit_n_kv_ch = [
            (self.tm.layers[li].self_attn.n_kv, self.tm.layers[li].self_attn.head_dim,
             self.tm.layers[li].self_attn.n_kv * self.tm.layers[li].self_attn.head_dim)
            for li, _, _ in self._schedule
        ]
        self._cap_ids_cache: dict[tuple[int, ...], Any] = {}
        self._ready_m1_caches: tuple[int, int] | None = None
        self._cache_binding: dict[str, Any] | None = None
        self._batch_cache_binding: dict[str, Any] | None = None
        self.rebind_count = 0

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
        assert self.ctx, "inkling_ctx_create failed"
        self.n_visits = lib.inkling_ctx_num_visits(self.ctx)
        self.scratch_bytes = lib.inkling_ctx_scratch_bytes(self.ctx)

        mq_f32 = mx.contiguous(self.tm.message_input.query.astype(mx.float32))
        fq_f32 = mx.contiguous(self.tm.final_input.query.reshape(-1).astype(mx.float32))
        mx.eval(mq_f32, fq_f32)
        gw = InklingGlobalWeights()
        gw.embed_tokens, gw.embed_tokens_bytes = arr_ptr(self.tm.embed_tokens.weight, self._keepalive)
        gw.embed_norm, gw.embed_norm_bytes = arr_ptr(self.tm.embed_norm.weight, self._keepalive)
        gw.message_queries, gw.message_queries_bytes = arr_ptr(mq_f32, self._keepalive)
        gw.final_query, gw.final_query_bytes = arr_ptr(fq_f32, self._keepalive)
        gw.final_norm, gw.final_norm_bytes = arr_ptr(self.tm.norm.weight, self._keepalive)
        gw.lm_head, gw.lm_head_bytes = arr_ptr(lm.lm_head.weight, self._keepalive)
        gw.embed_norm_eps = float(self.tm.embed_norm.eps)
        gw.final_norm_eps = float(self.tm.norm.eps)
        assert lib.inkling_ctx_register_globals(self.ctx, ctypes.byref(gw)) == 0

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

            lw.stack, lw.stack_bytes = arr_ptr(lc.stack, self._keepalive)
            lw.o_proj, lw.o_proj_bytes = arr_ptr(a.o_proj.weight, self._keepalive)
            lw.attn_query, lw.attn_query_bytes = arr_ptr(lc.attn_query, self._keepalive)
            lw.attn_norm, lw.attn_norm_bytes = arr_ptr(lc.attn_norm, self._keepalive)
            lw.mlp_query, lw.mlp_query_bytes = arr_ptr(lc.mlp_query, self._keepalive)
            lw.mlp_norm, lw.mlp_norm_bytes = arr_ptr(lc.mlp_norm, self._keepalive)
            lw.attn_taps, lw.attn_taps_bytes = arr_ptr(lc.attn_taps, self._keepalive)
            lw.mlp_taps, lw.mlp_taps_bytes = arr_ptr(lc.mlp_taps, self._keepalive)
            lw.k_taps, lw.k_taps_bytes = arr_ptr(lc.k_taps, self._keepalive)
            lw.v_taps, lw.v_taps_bytes = arr_ptr(lc.v_taps, self._keepalive)
            lw.kdyn_w, lw.kdyn_w_bytes = arr_ptr(lc.kdyn_w, self._keepalive)
            lw.vdyn_w, lw.vdyn_w_bytes = arr_ptr(lc.vdyn_w, self._keepalive)
            if lc.attn_gates is not None:
                lw.attn_gate_down, lw.attn_gate_down_bytes = arr_ptr(lc.attn_gates[0], self._keepalive)
                lw.attn_gate_forget, lw.attn_gate_forget_bytes = arr_ptr(lc.attn_gates[1], self._keepalive)
                lw.attn_gate_gain, lw.attn_gate_gain_bytes = arr_ptr(lc.attn_gates[2], self._keepalive)
                lw.mlp_gate_down, lw.mlp_gate_down_bytes = arr_ptr(lc.mlp_gates[0], self._keepalive)
                lw.mlp_gate_forget, lw.mlp_gate_forget_bytes = arr_ptr(lc.mlp_gates[1], self._keepalive)
                lw.mlp_gate_gain, lw.mlp_gate_gain_bytes = arr_ptr(lc.mlp_gates[2], self._keepalive)
            lw.router_w, lw.router_w_bytes = arr_ptr(lc.router_w, self._keepalive)
            lw.router_b, lw.router_b_bytes = arr_ptr(lc.router_b, self._keepalive)
            lw.sink_w, lw.sink_w_bytes = arr_ptr(lc.sink_w, self._keepalive)
            lw.q_scale, lw.q_scale_bytes = arr_ptr(a.q_scale, self._keepalive)
            lw.rel_scale, lw.rel_scale_bytes = arr_ptr(a.rel_scale, self._keepalive)
            lw.rel_proj, lw.rel_proj_bytes = arr_ptr(a.rel_proj, self._keepalive)
            lw.w13_weight, lw.w13_weight_bytes = arr_ptr(moe.w13_proj.weight, self._keepalive)
            lw.w13_scales, lw.w13_scales_bytes = arr_ptr(moe.w13_proj.scales, self._keepalive)
            lw.w13_gscale, lw.w13_gscale_bytes = arr_ptr(
                getattr(moe.w13_proj, "global_scale", None), self._keepalive
            )
            lw.w2_weight, lw.w2_weight_bytes = arr_ptr(moe.down_proj.weight, self._keepalive)
            lw.w2_scales, lw.w2_scales_bytes = arr_ptr(moe.down_proj.scales, self._keepalive)
            lw.w2_gscale, lw.w2_gscale_bytes = arr_ptr(
                getattr(moe.down_proj, "global_scale", None), self._keepalive
            )
            if self.use_bf16_opt:
                if lc.attn_fg_bf16_t is not None:
                    lw.attn_gate_fg_bf16, lw.attn_gate_fg_bf16_bytes = arr_ptr(
                        lc.attn_fg_bf16_t, self._keepalive
                    )
                if lc.mlp_fg_bf16_t is not None:
                    lw.mlp_gate_fg_bf16, lw.mlp_gate_fg_bf16_bytes = arr_ptr(
                        lc.mlp_fg_bf16_t, self._keepalive
                    )
                lw.router_w_bf16, lw.router_w_bf16_bytes = arr_ptr(
                    mx.contiguous(lc.router_w.astype(mx.bfloat16)), self._keepalive
                )
            assert lib.inkling_ctx_register_layer(self.ctx, ctypes.byref(lw)) == 0

    def close(self) -> None:
        if self.ctx:
            self.lib.inkling_ctx_destroy(self.ctx)
            self.ctx = None

    def ensure_cache_ready(self, caches: list[CacheList], m_tokens: int) -> None:
        """Ensure KV & conv buffers exist, are contiguous, and have enough capacity for m_tokens."""
        to_eval = []
        d = self.cfg.hidden_size
        b = self._cache_binding
        if b is not None and b["caches_id"] == id(caches) and self._ready_m1_caches == (id(caches), m_tokens):
            all_ok = True
            for cl in caches:
                kv = cl.caches[0]
                if kv.keys is None or (not isinstance(kv, RotatingKVCache) and kv.keys.shape[2] < kv.offset + m_tokens):
                    all_ok = False
                    break
            if all_ok:
                return
        for (n_kv, head_dim, ch), cl in zip(self._visit_n_kv_ch, caches):
            kv, conv = cl.caches[0], cl.caches[1]

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
                    kv.keys = mx.zeros((1, n_kv, target_cap, head_dim), dtype=mx.bfloat16)
                    kv.values = mx.zeros((1, n_kv, target_cap, head_dim), dtype=mx.bfloat16)
                    to_eval.extend((kv.keys, kv.values))
                elif kv.keys.shape[2] < target_cap:
                    cur = kv.keys.shape[2]
                    pad = target_cap - cur
                    kv.keys = mx.concatenate(
                        [mx.contiguous(kv.keys), mx.zeros((1, n_kv, pad, head_dim), dtype=mx.bfloat16)],
                        axis=2,
                    )
                    kv.values = mx.concatenate(
                        [mx.contiguous(kv.values), mx.zeros((1, n_kv, pad, head_dim), dtype=mx.bfloat16)],
                        axis=2,
                    )
                    to_eval.extend((kv.keys, kv.values))
            else:
                need = kv.offset + m_tokens
                if kv.keys is None:
                    cap = max(kv.step, ((need + kv.step - 1) // kv.step) * kv.step)
                    kv.keys = mx.zeros((1, n_kv, cap, head_dim), dtype=mx.bfloat16)
                    kv.values = mx.zeros((1, n_kv, cap, head_dim), dtype=mx.bfloat16)
                    to_eval.extend((kv.keys, kv.values))
                elif kv.keys.shape[2] < need:
                    pad = ((need - kv.keys.shape[2] + kv.step - 1) // kv.step) * kv.step
                    kv.keys = mx.concatenate(
                        [mx.contiguous(kv.keys), mx.zeros((1, n_kv, pad, head_dim), dtype=mx.bfloat16)],
                        axis=2,
                    )
                    kv.values = mx.concatenate(
                        [mx.contiguous(kv.values), mx.zeros((1, n_kv, pad, head_dim), dtype=mx.bfloat16)],
                        axis=2,
                    )
                    to_eval.extend((kv.keys, kv.values))

        if to_eval:
            mx.eval(*to_eval)
        self._ready_m1_caches = (id(caches), m_tokens)

    def _pack_cache_args(self, caches: list[CacheList]):
        nv = self.n_visits
        b = self._cache_binding
        if b is None or b["caches_id"] != id(caches):
            entries = []
            kv_meta = (InklingKVMeta * nv)()
            for v, cl in enumerate(caches):
                kv, conv = cl.caches[0], cl.caches[1]
                entries.append((kv, conv, 1 if isinstance(kv, RotatingKVCache) else 0, kv_meta[v]))
            b = {
                "caches_id": id(caches),
                "kv_ptrs": (ctypes.c_void_p * (2 * nv))(),
                "conv_ptrs": (ctypes.c_void_p * (4 * nv))(),
                "kv_meta": kv_meta,
                "entries": entries,
                "kv_ids": [None] * nv,
                "conv_ids": [None] * (4 * nv),
                "kv_views": [None] * nv,
                "conv_views": [None] * (4 * nv),
            }
            self._cache_binding = b

        kv_ptrs = b["kv_ptrs"]
        conv_ptrs = b["conv_ptrs"]
        kv_meta = b["kv_meta"]
        entries = b["entries"]
        kv_ids = b["kv_ids"]
        conv_ids = b["conv_ids"]
        kv_views = b["kv_views"]
        conv_views = b["conv_views"]
        hds = self._visit_head_dims

        for v, (kv, conv, is_rot, km) in enumerate(entries):
            k_obj, v_obj = kv.keys, kv.values
            k_id = (id(k_obj), id(v_obj))
            if kv_ids[v] != k_id:
                self.rebind_count += 1
                kv_k_view = k_obj.view(mx.uint16)
                kv_v_view = v_obj.view(mx.uint16)
                mx.eval(kv_k_view, kv_v_view)
                nk = np.asarray(kv_k_view)
                nv_arr = np.asarray(kv_v_view)
                if not nk.flags.c_contiguous or not nv_arr.flags.c_contiguous:
                    kv.keys = mx.contiguous(k_obj)
                    kv.values = mx.contiguous(v_obj)
                    kv_k_view = kv.keys.view(mx.uint16)
                    kv_v_view = kv.values.view(mx.uint16)
                    mx.eval(kv_k_view, kv_v_view)
                    nk = np.asarray(kv_k_view)
                    nv_arr = np.asarray(kv_v_view)
                    k_id = (id(kv.keys), id(kv.values))
                kv_views[v] = (kv_k_view, kv_v_view, nk, nv_arr)
                kv_ids[v] = k_id
                kv_ptrs[2 * v + 0] = int(nk.ctypes.data)
                kv_ptrs[2 * v + 1] = int(nv_arr.ctypes.data)
                cap = int(kv.keys.shape[2])
                km.capacity = cap
                km.max_size = int(kv.max_size) if is_rot else 0
                km.is_rotating = is_rot
                km.stride_h = cap * hds[v]
                km.stride_s = hds[v]
                km._pad = 0

            km.offset = int(kv.offset)
            km.idx = int(kv._idx) if is_rot else km.offset

            base_c = 4 * v
            c0, c1, c2, c3 = conv[0], conv[1], conv[2], conv[3]
            if (conv_ids[base_c] != id(c0) or conv_ids[base_c + 1] != id(c1) or
                    conv_ids[base_c + 2] != id(c2) or conv_ids[base_c + 3] != id(c3)):
                for c, st in enumerate((c0, c1, c2, c3)):
                    cid = id(st)
                    if conv_ids[base_c + c] != cid:
                        self.rebind_count += 1
                        mx.eval(st)
                        ns = np.asarray(st)
                        conv_views[base_c + c] = (st, ns)
                        conv_ids[base_c + c] = cid
                        conv_ptrs[base_c + c] = int(ns.ctypes.data)

        return kv_ptrs, conv_ptrs, kv_meta, b

    def step(
        self,
        token_ids: list[int],
        caches: list[CacheList],
        *,
        seq_mode: bool = False,
        capture_layer_ids: tuple[int, ...] = (),
        want_pre_norm: bool = True,
        want_argmax: bool = True,
        want_conv_states: bool = False,
        commit_rotating: bool = True,
        captures_out: mx.array | None = None,
    ) -> dict[str, Any]:
        m = len(token_ids)
        self.ensure_cache_ready(caches, m)
        kv_ptrs, conv_ptrs, kv_meta, tmp_keep = self._pack_cache_args(caches)

        toks_c = self._toks_c
        for i in range(m):
            toks_c[i] = token_ids[i]
        n_cap = len(capture_layer_ids)
        if n_cap > 0:
            cap_ids_c = self._cap_ids_cache.get(capture_layer_ids)
            if cap_ids_c is None:
                cap_ids_c = (ctypes.c_int32 * n_cap)(*capture_layer_ids)
                self._cap_ids_cache[capture_layer_ids] = cap_ids_c
        else:
            cap_ids_c = None

        d = self.cfg.hidden_size
        pre_norm_ptr = int(self._pre_norm_np.ctypes.data) if want_pre_norm else 0
        toks_out_c = self._toks_out_c if want_argmax else None

        cap_ptr = 0
        cap_mx_buf = None
        if n_cap > 0:
            if captures_out is not None:
                cid = id(captures_out)
                cached = self._ext_cap_cache.get(cid)
                if cached is None:
                    cv = captures_out.view(mx.uint16)
                    mx.eval(cv)
                    cn = np.asarray(cv)
                    cached = (cv, cn)
                    self._ext_cap_cache[cid] = cached
                cap_ptr = int(cached[1].ctypes.data)
                cap_mx_buf = captures_out
            else:
                cbuf = self._cap_bufs.get(n_cap)
                if cbuf is None:
                    b_mx = mx.zeros((1, self._max_m, n_cap * d), dtype=mx.bfloat16)
                    b_v = b_mx.view(mx.uint16)
                    mx.eval(b_v)
                    b_np = np.asarray(b_v)
                    cbuf = (b_mx, b_v, b_np)
                    self._cap_bufs[n_cap] = cbuf
                cap_ptr = int(cbuf[2].ctypes.data)
                cap_mx_buf = cbuf[0]

        conv_out_ptr = 0
        conv_mx = None
        if want_conv_states:
            if m not in self._conv_out_bufs:
                nf = self.lib.inkling_ctx_conv_states_floats(self.ctx, m)
                buf_mx = mx.zeros((nf,), dtype=mx.float32)
                mx.eval(buf_mx)
                buf_np = np.asarray(buf_mx)
                self._conv_out_bufs[m] = (buf_mx, buf_np)
            conv_mx, conv_np = self._conv_out_bufs[m]
            conv_out_ptr = int(conv_np.ctypes.data)

        stats = InklingStepStats()
        rc = self.lib.inkling_ctx_step(
            self.ctx,
            toks_c,
            m,
            1 if seq_mode else 0,
            kv_ptrs,
            conv_ptrs,
            kv_meta,
            cap_ids_c,
            n_cap,
            ctypes.c_void_p(pre_norm_ptr) if pre_norm_ptr else None,
            toks_out_c,
            ctypes.c_void_p(cap_ptr) if cap_ptr else None,
            ctypes.c_void_p(conv_out_ptr) if conv_out_ptr else None,
            1 if commit_rotating else 0,
            ctypes.byref(stats),
        )
        assert rc == 0, f"inkling_ctx_step failed with rc={rc}"

        use_internal_conv_hist = seq_mode and (not commit_rotating) and (not want_conv_states)
        entries = tmp_keep["entries"]
        if use_internal_conv_hist:
            for kv, _, is_rot, km in entries:
                kv.offset = int(km.offset)
                if is_rot:
                    kv._idx = int(km.idx)
        else:
            step_adv = m if seq_mode else 1
            for kv, conv, is_rot, km in entries:
                kv.offset = int(km.offset)
                if is_rot:
                    kv._idx = int(km.idx)
                conv.advance(step_adv)

        conv_states = None
        if want_conv_states:
            conv_states = []
            for v, (layer_idx, _, _) in enumerate(self._schedule):
                a = self.tm.layers[layer_idx].self_attn
                ch = a.n_kv * a.head_dim
                off = int(self.lib.inkling_ctx_visit_conv_offset(self.ctx, v, m))
                c_sz = m * 3 * ch
                d_sz = m * 3 * d
                ks = conv_mx[off : off + c_sz].reshape(m, 3, ch)
                off += c_sz
                vs = conv_mx[off : off + c_sz].reshape(m, 3, ch)
                off += c_sz
                cs = conv_mx[off : off + d_sz].reshape(m, 3, d)
                off += d_sz
                ms = conv_mx[off : off + d_sz].reshape(m, 3, d)
                conv_states.append((ks, vs, cs, ms))

        return {
            "pre_norm": self._pre_norm_mx[:, :m, :] if want_pre_norm else None,
            "argmax": [int(toks_out_c[i]) for i in range(m)] if toks_out_c is not None else None,
            "captures": cap_mx_buf[:, :m, :] if cap_mx_buf is not None else None,
            "conv_states": conv_states,
            "stats": stats,
        }

    def commit_rotating(self, caches: list[CacheList], n_accepted: int) -> None:
        kv_ptrs, _, kv_meta, tmp_keep = self._pack_cache_args(caches)
        rc = self.lib.inkling_ctx_commit_rotating(self.ctx, kv_ptrs, kv_meta, n_accepted)
        assert rc == 0, f"inkling_ctx_commit_rotating failed with rc={rc}"
        for v, cl in enumerate(caches):
            kv = cl.caches[0]
            if isinstance(kv, RotatingKVCache):
                kv.offset = int(kv_meta[v].offset)
                kv._idx = int(kv_meta[v].idx)

    def commit_verify(self, caches: list[CacheList], n_accepted: int, m_tokens: int) -> None:
        b = self._cache_binding
        if b is not None and b["caches_id"] == id(caches) and b["kv_ids"][0] == (id(caches[0].caches[0].keys), id(caches[0].caches[0].values)):
            kv_ptrs, conv_ptrs, kv_meta = b["kv_ptrs"], b["conv_ptrs"], b["kv_meta"]
            entries = b["entries"]
        else:
            kv_ptrs, conv_ptrs, kv_meta, b = self._pack_cache_args(caches)
            entries = b["entries"]
        rc = self.lib.inkling_ctx_commit_verify(
            self.ctx, kv_ptrs, conv_ptrs, kv_meta, n_accepted, m_tokens
        )
        assert rc == 0, f"inkling_ctx_commit_verify failed with rc={rc}"
        if n_accepted > 0:
            for kv, conv, is_rot, km in entries:
                kv.offset = int(km.offset)
                if is_rot:
                    kv._idx = int(km.idx)
                conv.advance(n_accepted)
        else:
            for kv, _, is_rot, km in entries:
                kv.offset = int(km.offset)
                if is_rot:
                    kv._idx = int(km.idx)

    def ensure_batch_cache_ready(self, caches: list[CacheList], batch_size: int) -> None:
        """Ensure batched KV & conv buffers exist, are contiguous, and have enough capacity."""
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

    def _pack_batch_cache_args(self, caches: list[CacheList], batch_size: int):
        nv = len(caches)
        b = self._batch_cache_binding
        if b is None or b["nv"] != nv or b["batch_size"] != batch_size:
            b = {
                "nv": nv,
                "batch_size": batch_size,
                "kv_ptrs": (ctypes.c_void_p * (2 * nv))(),
                "conv_ptrs": (ctypes.c_void_p * (4 * nv))(),
                "kv_meta": (InklingBatchKVMeta * nv)(),
                "row_offsets": np.zeros((nv, batch_size), dtype=np.int32),
                "kv_ids": [None] * nv,
                "conv_ids": [None] * (4 * nv),
                "off_ids": [None] * nv,
                "kv_views": [None] * nv,
                "conv_views": [None] * (4 * nv),
            }
            self._batch_cache_binding = b

        kv_ptrs = b["kv_ptrs"]
        conv_ptrs = b["conv_ptrs"]
        kv_meta = b["kv_meta"]
        row_offsets = b["row_offsets"]
        kv_ids = b["kv_ids"]
        conv_ids = b["conv_ids"]
        off_ids = b["off_ids"]
        kv_views = b["kv_views"]
        conv_views = b["conv_views"]
        hds = self._visit_head_dims

        for v, cl in enumerate(caches):
            kv, conv = cl.caches[0], cl.caches[1]
            is_rot = 1 if isinstance(kv, BatchRotatingKVCache) else 0
            k_id = (id(kv.keys), id(kv.values))
            if kv_ids[v] != k_id:
                self.rebind_count += 1
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
                kv_views[v] = (kv_k_view, kv_v_view, nk, nv_arr)
                kv_ids[v] = k_id
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

            if off_ids[v] != id(kv.offset):
                off_arr = mx.contiguous(kv.offset.astype(mx.int32))
                mx.eval(off_arr)
                row_offsets[v] = np.asarray(off_arr)
                off_ids[v] = id(kv.offset)

            base_c = 4 * v
            for c in range(4):
                st = conv[c]
                cid = id(st)
                if conv_ids[base_c + c] != cid:
                    self.rebind_count += 1
                    mx.eval(st)
                    ns = np.asarray(st)
                    conv_views[base_c + c] = (st, ns)
                    conv_ids[base_c + c] = cid
                    conv_ptrs[base_c + c] = int(ns.ctypes.data)

        return kv_ptrs, conv_ptrs, kv_meta, row_offsets, off_ids

    def step_batched(
        self,
        token_ids: list[int],
        caches: list[CacheList],
        *,
        want_pre_norm: bool = True,
        want_argmax: bool = True,
    ) -> dict[str, Any]:
        b = len(token_ids)
        assert 1 <= b <= self._max_m, f"batch_size={b} out of [1, {self._max_m}]"
        self.ensure_batch_cache_ready(caches, b)
        kv_ptrs, conv_ptrs, kv_meta, row_offsets, off_ids = self._pack_batch_cache_args(caches, b)

        toks_c = self._toks_c
        for i in range(b):
            toks_c[i] = token_ids[i]
        d = self.cfg.hidden_size
        pre_norm_ptr = int(self._pre_norm_np.ctypes.data) if want_pre_norm else 0
        toks_out_c = self._toks_out_c if want_argmax else None
        row_c = row_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

        stats = InklingStepStats()
        rc = self.lib.inkling_ctx_step_batched(
            self.ctx,
            toks_c,
            b,
            kv_ptrs,
            conv_ptrs,
            kv_meta,
            row_c,
            ctypes.c_void_p(pre_norm_ptr) if pre_norm_ptr else None,
            toks_out_c,
            None,
            ctypes.byref(stats),
        )
        assert rc == 0, f"inkling_ctx_step_batched failed with rc={rc}"

        for v, cl in enumerate(caches):
            kv = cl.caches[0]
            if isinstance(kv, BatchRotatingKVCache):
                old_idx = int(kv._idx)
                if old_idx == kv.max_size:
                    kv.rotated = True
                if kv.rotated:
                    kv.left_padding = kv.left_padding - 1
                kv._idx = int(kv_meta[v].idx)
                kv._offset = int(kv_meta[v].offset)
            else:
                kv._idx = int(kv_meta[v].idx)
            new_off = mx.array(row_offsets[v])
            kv.offset = new_off
            off_ids[v] = id(new_off)

        return {
            "pre_norm": (
                mx.array(np.array(self._pre_norm_np.reshape(self._max_m, 1, d)[:b], copy=True))
                if want_pre_norm
                else None
            ),
            "argmax": [int(toks_out_c[i]) for i in range(b)] if want_argmax else None,
            "stats": stats,
        }


def build_synthetic_model(
    *,
    n_layers: int = 4,
    loop_start: int = 1,
    loop_end: int = 3,
    loop_passes: int = 3,
    block_size: int = 1,
    d: int = 256,
    inter: int = 128,
    n_experts: int = 16,
    top_k: int = 4,
    n_heads: int = 8,
    n_kv: int = 2,
    head_dim: int = 32,
    sliding_window: int = 16,
    rel_extent: int = 32,
    vocab_size: int = 512,
    unpadded_vocab_size: int = 500,
    layer_types: list[str] | None = None,
    mxfp8_layers: set[int] | None = None,
    gate_rank: int = 16,
    seed: int = 42,
) -> LanguageModel:
    mx.random.seed(seed)
    if layer_types is None:
        layer_types = ["hybrid_sliding", "hybrid_sliding", "full_attention", "hybrid_sliding"]
    if mxfp8_layers is None:
        mxfp8_layers = {n_layers - 1}

    cfg = TextConfig(
        hidden_size=d,
        num_hidden_layers=n_layers,
        vocab_size=vocab_size,
        unpadded_vocab_size=unpadded_vocab_size,
        rms_norm_eps=1e-6,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        head_dim=head_dim,
        swa_num_attention_heads=n_heads,
        swa_num_key_value_heads=n_kv,
        swa_head_dim=head_dim,
        sliding_window_size=sliding_window,
        layer_types=layer_types,
        d_rel=16,
        rel_extent=rel_extent,
        log_scaling_n_floor=8,
        log_scaling_alpha=0.1,
        sconv_kernel_size=4,
        intermediate_size=inter,
        n_routed_experts=n_experts,
        num_experts_per_tok=top_k,
        route_scale=4.0,
        qk_l2_clip_threshold=16.0,
        rel_l2_clip_threshold=8.0,
        n_layers_per_attn_res_block=block_size,
        loop_config=LoopConfig(
            core_start=loop_start,
            core_end=loop_end,
            num_passes=loop_passes,
            gate_rank=gate_rank,
        ),
    )

    lm = LanguageModel(cfg)

    def rbf(shape, scale=0.04):
        return (mx.random.normal(shape) * scale).astype(mx.bfloat16)

    def rf32(shape, scale=0.05):
        return (mx.random.normal(shape) * scale).astype(mx.float32)

    tm = lm.model
    tm.embed_tokens.weight = rbf((vocab_size, d), 0.1)
    tm.embed_norm.weight = (mx.ones((d,)) + rf32((d,), 0.05)).astype(mx.bfloat16)
    tm.message_input.query = rf32(tm.message_input.query.shape, 0.1)
    tm.final_input.query = rf32(tm.final_input.query.shape, 0.1)
    tm.norm.weight = (mx.ones((d,)) + rf32((d,), 0.05)).astype(mx.bfloat16)
    lm.lm_head.weight = rbf((vocab_size, d), 0.1)

    for i, layer in enumerate(tm.layers):
        if layer.attention_input is not None:
            layer.attention_input.query = rf32(layer.attention_input.query.shape, 0.1)
        layer.mlp_input.query = rf32(layer.mlp_input.query.shape, 0.1)

        if layer.is_sliding:
            layer.input_layernorm.weight = (mx.ones((d,)) + rf32((d,), 0.05)).astype(mx.bfloat16)
        else:
            for nm in ("q_input_layernorm", "k_input_layernorm", "v_input_layernorm"):
                getattr(layer, nm).weight = (mx.ones((d,)) + rf32((d,), 0.05)).astype(mx.bfloat16)
        layer.post_attention_layernorm.weight = (mx.ones((d,)) + rf32((d,), 0.05)).astype(mx.bfloat16)

        layer.attn_sconv.conv.weight = rf32(layer.attn_sconv.conv.weight.shape, 0.15)
        layer.mlp_sconv.conv.weight = rf32(layer.mlp_sconv.conv.weight.shape, 0.15)

        if hasattr(layer, "q_message_gate"):
            for gname in ("q_message_gate", "k_message_gate", "v_message_gate", "mlp_message_gate"):
                g = getattr(layer, gname)
                g.down_proj.weight = rbf(g.down_proj.weight.shape, 0.05)
                g.forget_gate_proj.weight = rf32(g.forget_gate_proj.weight.shape, 0.05)
                g.message_gate_proj.weight = rf32(g.message_gate_proj.weight.shape, 0.05)

        a = layer.self_attn
        for pname in ("q_proj", "k_proj", "v_proj", "r_proj", "o_proj", "gate_proj", "sink_proj"):
            mod = getattr(a, pname)
            mod.weight = rbf(mod.weight.shape, 0.04)
        a.q_scale = (mx.ones(a.q_scale.shape) + rf32(a.q_scale.shape, 0.1)).astype(mx.bfloat16)
        a.rel_scale = (mx.ones(a.rel_scale.shape) + rf32(a.rel_scale.shape, 0.1)).astype(mx.bfloat16)
        a.rel_proj = rbf(a.rel_proj.shape, 0.05)

        if layer.is_sliding:
            a.k_sconv.conv.weight = rf32(a.k_sconv.conv.weight.shape, 0.15)
            a.v_sconv.conv.weight = rf32(a.v_sconv.conv.weight.shape, 0.15)
        else:
            a.k_sconv.static_weight = rbf(a.k_sconv.static_weight.shape, 0.15)
            a.v_sconv.static_weight = rbf(a.v_sconv.static_weight.shape, 0.15)
            a.k_sconv.weight_proj.weight = rbf(a.k_sconv.weight_proj.weight.shape, 0.04)
            a.v_sconv.weight_proj.weight = rbf(a.v_sconv.weight_proj.weight.shape, 0.04)

        moe = layer.mlp
        moe.gate_weight = rf32(moe.gate_weight.shape, 0.1)
        moe.e_score_correction_bias = rf32(moe.e_score_correction_bias.shape, 0.05)
        mode = "mxfp8" if i in mxfp8_layers else "nvfp4"
        moe.w13_proj = QuantizedExpertLinear(n_experts, 2 * inter, d, mode)
        moe.down_proj = QuantizedExpertLinear(n_experts, d, inter, mode)

        w13_raw = rbf((n_experts, 2 * inter, d), 0.05 * (256.0 / d) ** 0.5)
        w2_raw = rbf((n_experts, d, inter), 0.05 * (128.0 / inter) ** 0.5)
        if mode == "nvfp4":
            q13, s13 = mx.quantize(w13_raw, group_size=16, bits=4, mode="nvfp4")
            q2, s2 = mx.quantize(w2_raw, group_size=16, bits=4, mode="nvfp4")
            moe.w13_proj.weight = q13
            moe.w13_proj.scales = s13
            moe.w13_proj.global_scale = mx.random.uniform(1800.0, 3600.0, shape=(n_experts,), dtype=mx.float32)
            moe.down_proj.weight = q2
            moe.down_proj.scales = s2
            moe.down_proj.global_scale = mx.random.uniform(1800.0, 3600.0, shape=(n_experts,), dtype=mx.float32)
        else:
            q13, s13 = mx.quantize(w13_raw, group_size=32, bits=8, mode="mxfp8")
            q2, s2 = mx.quantize(w2_raw, group_size=32, bits=8, mode="mxfp8")
            moe.w13_proj.weight = q13
            moe.w13_proj.scales = s13
            moe.down_proj.weight = q2
            moe.down_proj.scales = s2

    mx.eval(lm.parameters())
    lm.eval()
    return lm


def clone_caches(caches: list[CacheList]) -> list[CacheList]:
    out = []
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
            nkv = GrowingKVCache()
            if kv.keys is not None:
                nkv.keys = mx.array(kv.keys)
                nkv.values = mx.array(kv.values)
            nkv.offset = kv.offset
        nconv = ArraysCache(len(conv.cache))
        for idx, arr in enumerate(conv.cache):
            nconv[idx] = mx.array(arr) if arr is not None else None
        out.append(CacheList(nkv, nconv))
    return out


def max_abs_diff(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def _build_synthetic_drafter(
    lm: LanguageModel,
    *,
    num_layers: int = 3,
    block_size: int = 8,
    markov_rank: int = 64,
    target_layer_ids: tuple[int, ...] = (1, 3, 5, 8),
    local_layer_ids: tuple[int, ...] = (0, 2),
    sliding_window: int = 16,
    seed: int = 123,
) -> InklingMTPDrafter:
    mx.random.seed(seed)
    c = lm.config
    mcfg = MTPConfig(
        hidden_size=c.hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=c.num_attention_heads,
        num_key_value_heads=c.num_key_value_heads,
        head_dim=c.head_dim,
        intermediate_size=max(c.intermediate_size, 128),
        rms_norm_eps=c.rms_norm_eps,
        rope_theta=1_000_000.0,
        local_layer_ids=local_layer_ids,
        sliding_window_size=sliding_window,
        block_size=block_size,
        markov_rank=markov_rank,
        target_layer_ids=target_layer_ids,
        vocab_size=c.vocab_size,
        unpadded_vocab_size=c.unpadded_vocab_size,
        final_logit_softcapping=float(c.final_logit_softcapping or 30.0),
    )
    drafter = InklingMTPDrafter(mcfg)

    def rbf(shape, scale=0.04):
        return (mx.random.normal(shape) * scale).astype(mx.bfloat16)

    def rone(shape, scale=0.05):
        return (mx.ones(shape) + mx.random.normal(shape) * scale).astype(mx.bfloat16)

    drafter.norm.weight = rone(drafter.norm.weight.shape)
    drafter.mask_embed = rbf(drafter.mask_embed.shape, 0.05)
    drafter.fuse_weight = (mx.random.normal(drafter.fuse_weight.shape) * 0.3).astype(mx.float32)

    for layer in drafter.layers:
        layer.input_layernorm.weight = rone(layer.input_layernorm.weight.shape)
        layer.post_attention_layernorm.weight = rone(layer.post_attention_layernorm.weight.shape)
        layer.dflare_norm.weight = rone(layer.dflare_norm.weight.shape)
        sa = layer.self_attn
        sa.q_proj.weight = rbf(sa.q_proj.weight.shape, 0.05)
        sa.k_proj.weight = rbf(sa.k_proj.weight.shape, 0.05)
        sa.v_proj.weight = rbf(sa.v_proj.weight.shape, 0.05)
        sa.k_proj_ctx.weight = rbf(sa.k_proj_ctx.weight.shape, 0.05)
        sa.v_proj_ctx.weight = rbf(sa.v_proj_ctx.weight.shape, 0.05)
        sa.o_proj.weight = rbf(sa.o_proj.weight.shape, 0.05)
        sa.q_norm.weight = rone(sa.q_norm.weight.shape)
        sa.k_norm.weight = rone(sa.k_norm.weight.shape)
        sa.k_norm_ctx.weight = rone(sa.k_norm_ctx.weight.shape)
        mlp = layer.mlp
        mlp.gate_proj.weight = rbf(mlp.gate_proj.weight.shape, 0.05)
        mlp.up_proj.weight = rbf(mlp.up_proj.weight.shape, 0.05)
        mlp.down_proj.weight = rbf(mlp.down_proj.weight.shape, 0.05)

    mh = drafter.markov_head
    mh.markov_w1.weight = rbf(mh.markov_w1.weight.shape, 0.05)
    mh.rnn_proj.weight = (mx.random.normal(mh.rnn_proj.weight.shape) * 0.04).astype(mx.float32)
    mh.rnn_proj.bias = (mx.random.normal(mh.rnn_proj.bias.shape) * 0.02).astype(mx.float32)
    mh.hidden_out.weight = rbf(mh.hidden_out.weight.shape, 0.04)
    mh.hidden_out.bias = rbf(mh.hidden_out.bias.shape, 0.02)
    mh.dflare_norm.weight = rone(mh.dflare_norm.weight.shape)

    mx.eval(drafter.parameters())
    drafter.bind(lm)
    mx.eval(drafter.parameters())
    return drafter


def _attach_mtp_to_runner(
    lib: ctypes.CDLL,
    runner: NativeDecodeRunner,
    lm: LanguageModel,
    drafter: InklingMTPDrafter,
    *,
    share_ptrs_as_null: bool = False,
) -> list[Any]:
    mc = drafter.config
    keepalive: list[Any] = []
    c_cfg = MTPDraftConfig(
        hidden_size=mc.hidden_size,
        num_layers=mc.num_hidden_layers,
        num_heads=mc.num_attention_heads,
        num_kv_heads=mc.num_key_value_heads,
        head_dim=mc.head_dim,
        inter_size=mc.intermediate_size,
        block_size=mc.block_size,
        markov_rank=mc.markov_rank,
        n_target_layers=len(mc.target_layer_ids),
        vocab_size=mc.vocab_size,
        unpadded_vocab_size=mc.unpadded_vocab_size,
        rms_norm_eps=mc.rms_norm_eps,
        embed_norm_eps=lm.config.rms_norm_eps,
        rope_theta=mc.rope_theta,
    )

    def w(arr: mx.array | None) -> tuple[int, int]:
        return arr_ptr(arr, keepalive)

    et_p, et_b = (0, 0) if share_ptrs_as_null else w(lm.model.embed_tokens.weight)
    en_p, en_b = w(lm.model.embed_norm.weight)
    lh_p, lh_b = (0, 0) if share_ptrs_as_null else w(lm.lm_head.weight)
    me_p, me_b = w(drafter.mask_embed.astype(mx.bfloat16))
    fn_p, fn_b = w(drafter.norm.weight.astype(mx.bfloat16))
    fa_p, fa_b = w(drafter._fuse_alpha.astype(mx.float32))
    md_p, md_b = w(drafter.markov_head.dflare_norm.weight.astype(mx.bfloat16))
    mw_p, mw_b = w(drafter.markov_head.markov_w1.weight.astype(mx.bfloat16))
    rc_p, rc_b = w(drafter.markov_head._rnn_w_ctx_t)
    rs_p, rs_b = w(drafter.markov_head._rnn_w_step_t)
    rb_p, rb_b = w(drafter.markov_head._rnn_b_f32)
    fw_p, fw_b = w(drafter.markov_head._fold_weight)
    fb_p, fb_b = w(drafter.markov_head._fold_bias)

    c_gw = MTPDraftGlobalWeights(
        embed_tokens=et_p, embed_tokens_bytes=et_b,
        embed_norm=en_p, embed_norm_bytes=en_b,
        mask_embed=me_p, mask_embed_bytes=me_b,
        final_norm=fn_p, final_norm_bytes=fn_b,
        lm_head=lh_p, lm_head_bytes=lh_b,
        fuse_alpha=fa_p, fuse_alpha_bytes=fa_b,
        markov_dflare_norm=md_p, markov_dflare_norm_bytes=md_b,
        markov_w1=mw_p, markov_w1_bytes=mw_b,
        rnn_w_ctx_t=rc_p, rnn_w_ctx_t_bytes=rc_b,
        rnn_w_step_t=rs_p, rnn_w_step_t_bytes=rs_b,
        rnn_b_f32=rb_p, rnn_b_f32_bytes=rb_b,
        fold_weight=fw_p, fold_weight_bytes=fw_b,
        fold_bias=fb_p, fold_bias_bytes=fb_b,
    )

    c_layers = (MTPDraftLayerWeights * mc.num_hidden_layers)()
    for i, layer in enumerate(drafter.layers):
        sa, mlp = layer.self_attn, layer.mlp
        df_p, df_b = w(layer.dflare_norm.weight.astype(mx.bfloat16))
        il_p, il_b = w(layer.input_layernorm.weight.astype(mx.bfloat16))
        pl_p, pl_b = w(layer.post_attention_layernorm.weight.astype(mx.bfloat16))
        kc_p, kc_b = w(sa._kv_ctx_w)
        knc_p, knc_b = w(sa.k_norm_ctx.weight.astype(mx.bfloat16))
        qkv_p, qkv_b = w(sa._qkv_w)
        qn_p, qn_b = w(sa.q_norm.weight.astype(mx.bfloat16))
        kn_p, kn_b = w(sa.k_norm.weight.astype(mx.bfloat16))
        op_p, op_b = w(sa.o_proj.weight.astype(mx.bfloat16))
        gu_p, gu_b = w(mlp._gu_w)
        dp_p, dp_b = w(mlp.down_proj.weight.astype(mx.bfloat16))
        c_layers[i] = MTPDraftLayerWeights(
            is_local=1 if sa.window is not None else 0,
            window=int(sa.window or 0),
            dflare_norm=df_p, dflare_norm_bytes=df_b,
            input_layernorm=il_p, input_layernorm_bytes=il_b,
            post_attention_layernorm=pl_p, post_attention_layernorm_bytes=pl_b,
            kv_ctx_w=kc_p, kv_ctx_w_bytes=kc_b,
            k_norm_ctx=knc_p, k_norm_ctx_bytes=knc_b,
            qkv_w=qkv_p, qkv_w_bytes=qkv_b,
            q_norm=qn_p, q_norm_bytes=qn_b,
            k_norm=kn_p, k_norm_bytes=kn_b,
            o_proj_w=op_p, o_proj_w_bytes=op_b,
            gu_w=gu_p, gu_w_bytes=gu_b,
            down_proj_w=dp_p, down_proj_w_bytes=dp_b,
        )

    cap_arr = (ctypes.c_int32 * len(mc.target_layer_ids))(*mc.target_layer_ids)
    rc = lib.inkling_ctx_attach_mtp(
        runner.ctx,
        ctypes.byref(c_cfg),
        ctypes.byref(c_gw),
        c_layers,
        cap_arr,
        len(mc.target_layer_ids),
    )
    assert rc == 0, f"inkling_ctx_attach_mtp failed with rc={rc}"
    return keepalive


def _clone_draft_caches(draft_caches: list) -> list:
    out = []
    for dc in draft_caches:
        if isinstance(dc, RotatingKVCache):
            ndc = RotatingKVCache(max_size=dc.max_size, keep=dc.keep)
            if dc.keys is not None:
                ndc.keys = mx.array(dc.keys)
                ndc.values = mx.array(dc.values)
            ndc.offset = dc.offset
            ndc._idx = dc._idx
        else:
            ndc = KVCache()
            if dc.keys is not None:
                ndc.keys = mx.array(dc.keys)
                ndc.values = mx.array(dc.values)
            ndc.offset = dc.offset
        out.append(ndc)
    mx.eval([c.keys for c in out if c.keys is not None] + [c.values for c in out if c.values is not None])
    return out


def _compare_draft_caches(py_dc: list, nat_dc: list) -> float:
    max_err = 0.0
    for i, (pc, nc) in enumerate(zip(py_dc, nat_dc)):
        assert pc.offset == nc.offset, f"draft layer {i} offset py={pc.offset} nat={nc.offset}"
        if pc.offset == 0:
            continue
        if isinstance(pc, RotatingKVCache):
            valid = min(pc.offset, pc.max_size)
            pk = pc._temporal_order(pc.keys)[..., -valid:, :]
            pv = pc._temporal_order(pc.values)[..., -valid:, :]
            nk = nc._temporal_order(nc.keys)[..., -valid:, :]
            nv = nc._temporal_order(nc.values)[..., -valid:, :]
        else:
            pk = pc.keys[..., : pc.offset, :]
            pv = pc.values[..., : pc.offset, :]
            nk = nc.keys[..., : nc.offset, :]
            nv = nc.values[..., : nc.offset, :]
        max_err = max(max_err, max_abs_diff(pk, nk), max_abs_diff(pv, nv))
    return max_err


def _run_unified_mtp_step(
    lib: ctypes.CDLL,
    runner: NativeDecodeRunner,
    drafter: InklingMTPDrafter,
    anchor_token: int,
    in_hidden: mx.array,
    out_hidden: mx.array,
    n_ctx: int,
    num_draft: int,
    target_caches: list[CacheList],
    draft_caches: list,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    softcap: float = 30.0,
    rng_seed: int = 0,
    rng_step: int = 0,
) -> tuple[list[int], int, InklingStepStats]:
    mc = drafter.config
    hd = mc.head_dim
    n_kv = mc.num_key_value_heads

    runner.ensure_cache_ready(target_caches, m_tokens=num_draft + 1)
    kv_ptrs, conv_ptrs, kv_meta, b = runner._pack_cache_args(target_caches)
    entries = b["entries"]

    to_eval = []
    for i, dc in enumerate(draft_caches):
        is_rot = isinstance(dc, RotatingKVCache)
        need = dc.max_size if is_rot else max(256, ((dc.offset + n_ctx + 255) // 256) * 256)
        if dc.keys is None:
            dc.keys = mx.zeros((1, n_kv, need, hd), dtype=mx.bfloat16)
            dc.values = mx.zeros((1, n_kv, need, hd), dtype=mx.bfloat16)
            to_eval.extend((dc.keys, dc.values))
        elif dc.keys.shape[2] < need:
            pad = need - dc.keys.shape[2]
            dc.keys = mx.concatenate([mx.contiguous(dc.keys), mx.zeros((1, n_kv, pad, hd), dtype=mx.bfloat16)], axis=2)
            dc.values = mx.concatenate([mx.contiguous(dc.values), mx.zeros((1, n_kv, pad, hd), dtype=mx.bfloat16)], axis=2)
            to_eval.extend((dc.keys, dc.values))
    if to_eval:
        mx.eval(*to_eval)

    keep: list[Any] = []
    d_descs = (MTPKVCacheDesc * mc.num_hidden_layers)()
    for i, dc in enumerate(draft_caches):
        kp, kb = arr_ptr(dc.keys, keep)
        vp, vb = arr_ptr(dc.values, keep)
        cap = int(dc.keys.shape[2])
        is_rot = 1 if isinstance(dc, RotatingKVCache) else 0
        d_descs[i] = MTPKVCacheDesc(
            k_ptr=kp, v_ptr=vp, k_nbytes=kb, v_nbytes=vb,
            cache_cap=cap, offset=int(dc.offset), is_rotating=is_rot,
            window=int(dc.max_size) if is_rot else 0,
            ring_idx=int(dc._idx) if is_rot else int(dc.offset),
            stride_h=cap * hd, stride_s=hd, _pad=0,
        )

    in_h_c = mx.contiguous(in_hidden)
    in_p, in_b = arr_ptr(in_h_c, keep)
    if out_hidden is in_hidden:
        out_p, out_b = in_p, in_b
    else:
        out_p, out_b = arr_ptr(out_hidden, keep)

    out_toks = (ctypes.c_int32 * (num_draft + 1))()
    out_n_acc = ctypes.c_int32(0)
    stats = InklingStepStats()
    rc = lib.inkling_ctx_mtp_step(
        runner.ctx, int(anchor_token),
        in_p, in_b, out_p, out_b,
        int(n_ctx), int(num_draft),
        float(temperature), float(top_p), float(softcap),
        int(rng_seed), int(rng_step),
        kv_ptrs, conv_ptrs, kv_meta, d_descs,
        out_toks, ctypes.byref(out_n_acc), ctypes.byref(stats),
    )
    assert rc == 0, f"inkling_ctx_mtp_step failed with rc={rc}"

    n_acc = int(out_n_acc.value)
    for kv, conv, is_rot, km in entries:
        kv.offset = int(km.offset)
        if is_rot:
            kv._idx = int(km.idx)
        conv.advance(n_acc)
    for i, dc in enumerate(draft_caches):
        dc.offset = int(d_descs[i].offset)
        if d_descs[i].is_rotating:
            dc._idx = int(d_descs[i].ring_idx)

    return [int(out_toks[j]) for j in range(n_acc)], n_acc, stats


def _run_batched_mtp_step(
    lib: ctypes.CDLL,
    runner: NativeDecodeRunner,
    drafter: InklingMTPDrafter,
    stream_specs: list[dict[str, Any]],
) -> tuple[list[tuple[list[int], int]], InklingStepStats]:
    mc = drafter.config
    hd = mc.head_dim
    n_kv = mc.num_key_value_heads
    b_size = len(stream_specs)
    streams = (InklingMTPStreamDesc * b_size)()
    keep: list[Any] = []
    commit_state: list[tuple[list[Any], list[Any], Any]] = []

    bindings = getattr(runner, "_batched_mtp_bindings", None)
    if bindings is None:
        bindings = {}
        runner._batched_mtp_bindings = bindings

    for b_i, sp in enumerate(stream_specs):
        target_caches = sp["target_caches"]
        draft_caches = sp["draft_caches"]
        n_ctx = int(sp["n_ctx"])
        num_draft = int(sp["num_draft"])
        cid = id(target_caches)
        runner._cache_binding = bindings.get(cid)
        runner.ensure_cache_ready(target_caches, m_tokens=num_draft + 1)
        kv_ptrs, conv_ptrs, kv_meta, packed = runner._pack_cache_args(target_caches)
        bindings[cid] = packed
        entries = packed["entries"]

        to_eval = []
        for dc in draft_caches:
            is_rot = isinstance(dc, RotatingKVCache)
            need = dc.max_size if is_rot else max(256, ((dc.offset + n_ctx + 255) // 256) * 256)
            if dc.keys is None:
                dc.keys = mx.zeros((1, n_kv, need, hd), dtype=mx.bfloat16)
                dc.values = mx.zeros((1, n_kv, need, hd), dtype=mx.bfloat16)
                to_eval.extend((dc.keys, dc.values))
            elif dc.keys.shape[2] < need:
                pad = need - dc.keys.shape[2]
                dc.keys = mx.concatenate([mx.contiguous(dc.keys), mx.zeros((1, n_kv, pad, hd), dtype=mx.bfloat16)], axis=2)
                dc.values = mx.concatenate([mx.contiguous(dc.values), mx.zeros((1, n_kv, pad, hd), dtype=mx.bfloat16)], axis=2)
                to_eval.extend((dc.keys, dc.values))
        if to_eval:
            mx.eval(*to_eval)

        d_descs = (MTPKVCacheDesc * mc.num_hidden_layers)()
        for i, dc in enumerate(draft_caches):
            kp, kb = arr_ptr(dc.keys, keep)
            vp, vb = arr_ptr(dc.values, keep)
            cap = int(dc.keys.shape[2])
            is_rot = 1 if isinstance(dc, RotatingKVCache) else 0
            d_descs[i] = MTPKVCacheDesc(
                k_ptr=kp, v_ptr=vp, k_nbytes=kb, v_nbytes=vb,
                cache_cap=cap, offset=int(dc.offset), is_rotating=is_rot,
                window=int(dc.max_size) if is_rot else 0,
                ring_idx=int(dc._idx) if is_rot else int(dc.offset),
                stride_h=cap * hd, stride_s=hd, _pad=0,
            )

        in_hidden = sp["in_hidden"]
        out_hidden = sp["out_hidden"]
        in_h_c = mx.contiguous(in_hidden)
        in_p, in_b = arr_ptr(in_h_c, keep)
        if out_hidden is in_hidden:
            out_p, out_b = in_p, in_b
        else:
            out_p, out_b = arr_ptr(out_hidden, keep)

        keep.extend((kv_ptrs, conv_ptrs, kv_meta, d_descs, packed))
        streams[b_i] = InklingMTPStreamDesc(
            anchor_token=int(sp["anchor_token"]),
            n_ctx=n_ctx,
            num_draft=num_draft,
            _pad0=0,
            in_hidden_ptr=in_p,
            in_hidden_bytes=in_b,
            out_hidden_ptr=out_p,
            out_hidden_bytes=out_b,
            temperature=float(sp.get("temperature", 0.0)),
            top_p=float(sp.get("top_p", 1.0)),
            logit_softcap=float(sp.get("softcap", 30.0)),
            _pad1=0,
            rng_seed=int(sp.get("rng_seed", 0)),
            rng_step=int(sp.get("rng_step", 0)),
            kv_ptrs=kv_ptrs,
            conv_ptrs=conv_ptrs,
            kv_meta=kv_meta,
            draft_kv_descs=d_descs,
        )
        commit_state.append((entries, draft_caches, d_descs))

    stats = InklingStepStats()
    rc = lib.inkling_ctx_mtp_step_batched(runner.ctx, streams, b_size, ctypes.byref(stats))
    assert rc == 0, f"inkling_ctx_mtp_step_batched failed with rc={rc}"

    results: list[tuple[list[int], int]] = []
    for b_i, (entries, draft_caches, d_descs) in enumerate(commit_state):
        n_acc = int(streams[b_i].out_n_acc)
        toks = [int(streams[b_i].out_tokens[j]) for j in range(n_acc)]
        for kv, conv, is_rot, km in entries:
            kv.offset = int(km.offset)
            if is_rot:
                kv._idx = int(km.idx)
            conv.advance(n_acc)
        for i, dc in enumerate(draft_caches):
            dc.offset = int(d_descs[i].offset)
            if d_descs[i].is_rotating:
                dc._idx = int(d_descs[i].ring_idx)
        results.append((toks, n_acc))
    return results, stats


def compare_caches(py_caches: list[CacheList], nat_caches: list[CacheList]) -> tuple[float, float]:
    max_kv_err = 0.0
    max_conv_err = 0.0
    for v, (pcl, ncl) in enumerate(zip(py_caches, nat_caches)):
        pkv, pconv = pcl.caches[0], pcl.caches[1]
        nkv, nconv = ncl.caches[0], ncl.caches[1]
        assert pkv.offset == nkv.offset, f"visit {v}: kv.offset py={pkv.offset} nat={nkv.offset}"
        if isinstance(pkv, RotatingKVCache):
            valid = min(pkv.offset, pkv.max_size)
            pk = pkv._temporal_order(pkv.keys)[..., -valid:, :]
            pv = pkv._temporal_order(pkv.values)[..., -valid:, :]
            nk = nkv._temporal_order(nkv.keys)[..., -valid:, :]
            nv = nkv._temporal_order(nkv.values)[..., -valid:, :]
        else:
            pk = pkv.keys[..., : pkv.offset, :]
            pv = pkv.values[..., : pkv.offset, :]
            nk = nkv.keys[..., : nkv.offset, :]
            nv = nkv.values[..., : nkv.offset, :]
        max_kv_err = max(max_kv_err, max_abs_diff(pk, nk), max_abs_diff(pv, nv))
        for c in range(4):
            max_conv_err = max(max_conv_err, max_abs_diff(pconv[c], nconv[c]))
    return max_kv_err, max_conv_err


def test_m1_decode_parity(lib: ctypes.CDLL, lm: LanguageModel) -> None:
    print("=== Test 1: M=1 Plain Decode Parity (Pre-Rot -> Crossing -> Post-Rot -> Split-K) ===")
    runner = NativeDecodeRunner(lib, lm, max_m=8)
    print(f"  Visits={runner.n_visits}, Scratch arena={runner.scratch_bytes / 1024:.1f} KB")

    py_caches = lm.make_cache()
    nat_caches = lm.make_cache()

    rng = np.random.default_rng(123)
    max_pn_err = 0.0
    tok_mismatches = 0
    for step in range(36):
        tok = int(rng.integers(0, lm.config.unpadded_vocab_size))
        h = lm.model.embed(mx.array([[tok]], dtype=mx.int32))
        py_pn = fused.decode_step(lm.model, h, py_caches)
        py_tok = int(lm.greedy_argmax_from_pre_norm(py_pn).item())
        mx.eval(py_pn)

        nat = runner.step([tok], nat_caches, seq_mode=False, want_pre_norm=True, want_argmax=True)
        err = max_abs_diff(py_pn, nat["pre_norm"])
        max_pn_err = max(max_pn_err, err)
        if py_tok != nat["argmax"][0]:
            tok_mismatches += 1
        assert err <= 1e-3, f"step={step}: pre_norm err={err:.6f} > 1e-3"
        assert py_tok == nat["argmax"][0], f"step={step}: py_tok={py_tok} nat_tok={nat['argmax'][0]}"

    kv_err, conv_err = compare_caches(py_caches, nat_caches)
    print(
        f"  [1a] 36 decode steps (offset 0->36, SWA win=16): "
        f"max_pre_norm_err={max_pn_err:.2e}, max_kv_err={kv_err:.2e}, "
        f"max_conv_err={conv_err:.2e}, tok_mismatches={tok_mismatches} [PASS]"
    )
    assert kv_err <= 1e-3 and conv_err <= 1e-3

    py_long = lm.make_cache()
    nat_long = lm.make_cache()
    for (layer_idx, _, _), pcl, ncl in zip(lm.model.schedule(), py_long, nat_long):
        a = lm.model.layers[layer_idx].self_attn
        pkv, pconv = pcl.caches[0], pcl.caches[1]
        nkv, nconv = ncl.caches[0], ncl.caches[1]
        if isinstance(pkv, RotatingKVCache):
            k_hist = (mx.random.normal((1, a.n_kv, 16, a.head_dim)) * 0.2).astype(mx.bfloat16)
            v_hist = (mx.random.normal((1, a.n_kv, 16, a.head_dim)) * 0.2).astype(mx.bfloat16)
            pkv.keys, pkv.values, pkv.offset, pkv._idx = k_hist, v_hist, 260, 7
            nkv.keys, nkv.values, nkv.offset, nkv._idx = mx.array(k_hist), mx.array(v_hist), 260, 7
        else:
            k_hist = (mx.random.normal((1, a.n_kv, 260, a.head_dim)) * 0.2).astype(mx.bfloat16)
            v_hist = (mx.random.normal((1, a.n_kv, 260, a.head_dim)) * 0.2).astype(mx.bfloat16)
            pkv.keys, pkv.values, pkv.offset = k_hist, v_hist, 260
            nkv.keys, nkv.values, nkv.offset = mx.array(k_hist), mx.array(v_hist), 260
        ch = a.n_kv * a.head_dim
        d = lm.config.hidden_size
        for c_idx, width in enumerate((ch, ch, d, d)):
            st = (mx.random.normal((1, 3, width)) * 0.1).astype(mx.float32)
            pconv[c_idx] = st
            nconv[c_idx] = mx.array(st)
    mx.eval([cl.caches[0].keys for cl in py_long] + [cl.caches[0].values for cl in py_long])

    max_long_err = 0.0
    for step in range(5):
        tok = int(rng.integers(0, lm.config.unpadded_vocab_size))
        h = lm.model.embed(mx.array([[tok]], dtype=mx.int32))
        py_pn = fused.decode_step(lm.model, h, py_long)
        py_tok = int(lm.greedy_argmax_from_pre_norm(py_pn).item())
        mx.eval(py_pn)

        nat = runner.step([tok], nat_long, seq_mode=False, want_pre_norm=True, want_argmax=True)
        err = max_abs_diff(py_pn, nat["pre_norm"])
        max_long_err = max(max_long_err, err)
        assert err <= 1e-3, f"split-K step={step}: pre_norm err={err:.6f} > 1e-3"
        assert py_tok == nat["argmax"][0], f"split-K step={step}: py_tok={py_tok} nat={nat['argmax'][0]}"

    kv_err, conv_err = compare_caches(py_long, nat_long)
    print(
        f"  [1b] 5 decode steps at offset=260->265 (Split-K full-attn + rotated SWA): "
        f"max_pre_norm_err={max_long_err:.2e}, max_kv_err={kv_err:.2e}, "
        f"max_conv_err={conv_err:.2e} [PASS]"
    )
    assert kv_err <= 1e-3 and conv_err <= 1e-3
    runner.close()


def test_m6_verify_parity(lib: ctypes.CDLL, lm: LanguageModel) -> None:
    print("=== Test 2: M=6 Sequential MTP Verify Parity (Pre-Rot, Crossing, Post-Rot, Split-K) ===")
    runner = NativeDecodeRunner(lib, lm, max_m=8)
    rng = np.random.default_rng(456)

    scenarios = [
        ("pre_rotation (off=4->10)", 4, 4, (1, 3, 5, 8)),
        ("crossing_rotation (off=13->19)", 13, 13, (2, 4, 6, 8)),
        ("post_rotation (off=25->31)", 25, 9, (1, 3, 5, 8)),
        ("long_split_k (off=260->266)", 260, 11, (2, 4, 6, 8)),
    ]

    for label, init_off, rot_idx, cap_ids in scenarios:
        py_caches = lm.make_cache()
        nat_caches = lm.make_cache()
        for (layer_idx, _, _), pcl, ncl in zip(lm.model.schedule(), py_caches, nat_caches):
            a = lm.model.layers[layer_idx].self_attn
            pkv, pconv = pcl.caches[0], pcl.caches[1]
            nkv, nconv = ncl.caches[0], ncl.caches[1]
            if isinstance(pkv, RotatingKVCache):
                sz = min(init_off, pkv.max_size)
                k_h = (mx.random.normal((1, a.n_kv, sz, a.head_dim)) * 0.2).astype(mx.bfloat16)
                v_h = (mx.random.normal((1, a.n_kv, sz, a.head_dim)) * 0.2).astype(mx.bfloat16)
                pkv.keys, pkv.values, pkv.offset, pkv._idx = k_h, v_h, init_off, rot_idx
                nkv.keys, nkv.values, nkv.offset, nkv._idx = mx.array(k_h), mx.array(v_h), init_off, rot_idx
            else:
                k_h = (mx.random.normal((1, a.n_kv, init_off, a.head_dim)) * 0.2).astype(mx.bfloat16)
                v_h = (mx.random.normal((1, a.n_kv, init_off, a.head_dim)) * 0.2).astype(mx.bfloat16)
                pkv.keys, pkv.values, pkv.offset = k_h, v_h, init_off
                nkv.keys, nkv.values, nkv.offset = mx.array(k_h), mx.array(v_h), init_off
            ch = a.n_kv * a.head_dim
            d = lm.config.hidden_size
            for c_idx, width in enumerate((ch, ch, d, d)):
                st = (mx.random.normal((1, 3, width)) * 0.1).astype(mx.float32)
                pconv[c_idx] = st
                nconv[c_idx] = mx.array(st)

        toks = [int(x) for x in rng.integers(0, lm.config.unpadded_vocab_size, size=6)]
        h = lm.model.embed(mx.array([toks], dtype=mx.int32))
        py_pn, py_cap, py_cstates = fused.verify_step(lm.model, h, py_caches, capture_layer_ids=cap_ids)
        py_toks = lm.greedy_argmax_from_pre_norm(py_pn)[0].tolist()
        mx.eval(py_pn, py_cap)

        nat = runner.step(
            toks,
            nat_caches,
            seq_mode=True,
            capture_layer_ids=cap_ids,
            want_pre_norm=True,
            want_argmax=True,
            want_conv_states=True,
            commit_rotating=True,
        )

        pn_err = max_abs_diff(py_pn, nat["pre_norm"])
        cap_err = max_abs_diff(py_cap, nat["captures"])
        cstate_err = 0.0
        for p_tup, n_tup in zip(py_cstates, nat["conv_states"]):
            for pa, na in zip(p_tup, n_tup):
                cstate_err = max(cstate_err, max_abs_diff(pa, na))
        kv_err, conv_err = compare_caches(py_caches, nat_caches)
        tok_eq = (py_toks == nat["argmax"])

        print(
            f"  [{label:30s}] pre_norm={pn_err:.2e} | cap={cap_err:.2e} | "
            f"cstates={cstate_err:.2e} | kv={kv_err:.2e} | conv={conv_err:.2e} | "
            f"toks_exact={tok_eq} [PASS]"
        )
        assert pn_err <= 1e-3, f"{label}: pre_norm err={pn_err}"
        assert cap_err <= 1e-3, f"{label}: cap err={cap_err}"
        assert cstate_err <= 1e-3, f"{label}: cstates err={cstate_err}"
        assert kv_err <= 1e-3, f"{label}: kv err={kv_err}"
        assert conv_err <= 1e-3, f"{label}: conv err={conv_err}"
        assert tok_eq, f"{label}: py_toks={py_toks} != nat_toks={nat['argmax']}"

    runner.close()


def test_post_rotation_rollback(lib: ctypes.CDLL, lm: LanguageModel) -> None:
    print("=== Test 3: Zero-Copy Post-Rotation MTP Rollback (commit_rotating=0 -> commit 3 of 6) ===")
    runner = NativeDecodeRunner(lib, lm, max_m=8)
    rng = np.random.default_rng(789)

    py_caches = lm.make_cache()
    nat_caches = lm.make_cache()
    init_off, rot_idx = 25, 13
    for (layer_idx, _, _), pcl, ncl in zip(lm.model.schedule(), py_caches, nat_caches):
        a = lm.model.layers[layer_idx].self_attn
        pkv, pconv = pcl.caches[0], pcl.caches[1]
        nkv, nconv = ncl.caches[0], ncl.caches[1]
        if isinstance(pkv, RotatingKVCache):
            k_h = (mx.random.normal((1, a.n_kv, pkv.max_size, a.head_dim)) * 0.2).astype(mx.bfloat16)
            v_h = (mx.random.normal((1, a.n_kv, pkv.max_size, a.head_dim)) * 0.2).astype(mx.bfloat16)
            pkv.keys, pkv.values, pkv.offset, pkv._idx = k_h, v_h, init_off, rot_idx
            nkv.keys, nkv.values, nkv.offset, nkv._idx = mx.array(k_h), mx.array(v_h), init_off, rot_idx
        else:
            k_h = (mx.random.normal((1, a.n_kv, init_off, a.head_dim)) * 0.2).astype(mx.bfloat16)
            v_h = (mx.random.normal((1, a.n_kv, init_off, a.head_dim)) * 0.2).astype(mx.bfloat16)
            pkv.keys, pkv.values, pkv.offset = k_h, v_h, init_off
            nkv.keys, nkv.values, nkv.offset = mx.array(k_h), mx.array(v_h), init_off
        ch = a.n_kv * a.head_dim
        d = lm.config.hidden_size
        for c_idx, width in enumerate((ch, ch, d, d)):
            st = (mx.random.normal((1, 3, width)) * 0.1).astype(mx.float32)
            pconv[c_idx] = st
            nconv[c_idx] = mx.array(st)

    toks = [int(x) for x in rng.integers(0, lm.config.unpadded_vocab_size, size=6)]
    n_accepted = 3

    snaps = snapshot_rotating_kv(py_caches, 6)
    h = lm.model.embed(mx.array([toks], dtype=mx.int32))
    py_pn, _, py_cstates = fused.verify_step(lm.model, h, py_caches, capture_layer_ids=(1, 3, 5, 8))
    mx.eval(py_pn)
    rollback_single_cache(py_caches, snaps, py_cstates, n_accepted=n_accepted, n_verified=6)

    prev_grow_offsets = [cl.caches[0].offset for cl in nat_caches]
    nat = runner.step(
        toks,
        nat_caches,
        seq_mode=True,
        capture_layer_ids=(1, 3, 5, 8),
        want_pre_norm=True,
        want_argmax=True,
        want_conv_states=True,
        commit_rotating=False,
    )
    runner.commit_rotating(nat_caches, n_accepted=n_accepted)
    for v, cl in enumerate(nat_caches):
        kv, conv = cl.caches[0], cl.caches[1]
        if not isinstance(kv, RotatingKVCache):
            kv.offset = prev_grow_offsets[v] + n_accepted
        ks, vs, cs, ms = nat["conv_states"][v]
        pos = n_accepted - 1
        conv[0] = mx.contiguous(ks[pos : pos + 1])
        conv[1] = mx.contiguous(vs[pos : pos + 1])
        conv[2] = mx.contiguous(cs[pos : pos + 1])
        conv[3] = mx.contiguous(ms[pos : pos + 1])
    mx.eval([cl.caches[1][c] for cl in nat_caches for c in range(4)])

    kv_err, conv_err = compare_caches(py_caches, nat_caches)
    next_tok = int(rng.integers(0, lm.config.unpadded_vocab_size))
    h_next = lm.model.embed(mx.array([[next_tok]], dtype=mx.int32))
    py_next_pn = fused.decode_step(lm.model, h_next, py_caches)
    py_next_tok = int(lm.greedy_argmax_from_pre_norm(py_next_pn).item())
    mx.eval(py_next_pn)

    nat_next = runner.step([next_tok], nat_caches, seq_mode=False, want_pre_norm=True, want_argmax=True)
    next_err = max_abs_diff(py_next_pn, nat_next["pre_norm"])
    print(
        f"  After rollback (3/6 accepted): kv_err={kv_err:.2e}, conv_err={conv_err:.2e}, "
        f"follow-up decode pre_norm_err={next_err:.2e}, tok_exact={py_next_tok == nat_next['argmax'][0]} [PASS]"
    )
    assert kv_err <= 1e-3 and conv_err <= 1e-3 and next_err <= 1e-3
    assert py_next_tok == nat_next["argmax"][0]
    runner.close()


def bench_48_visit_dispatch(lib: ctypes.CDLL) -> None:
    print("=== Test 4: 48-Visit Synthetic Single-CB Dispatch Latency vs Python fused.py ===")
    layer_types = [("full_attention" if (i + 1) % 6 == 0 else "hybrid_sliding") for i in range(24)]
    lm48 = build_synthetic_model(
        n_layers=24,
        loop_start=6,
        loop_end=18,
        loop_passes=3,
        block_size=6,
        d=256,
        inter=128,
        n_experts=16,
        top_k=4,
        n_heads=8,
        n_kv=2,
        head_dim=32,
        sliding_window=32,
        rel_extent=32,
        vocab_size=512,
        unpadded_vocab_size=500,
        layer_types=layer_types,
        mxfp8_layers={20, 21, 22, 23},
        seed=99,
    )
    runner = NativeDecodeRunner(lib, lm48, max_m=8)
    print(f"  48-visit model built: visits={runner.n_visits}, scratch={runner.scratch_bytes / 1024:.1f} KB")

    py_c = lm48.make_cache()
    nat_c = lm48.make_cache()
    h1 = lm48.model.embed(mx.array([[7]], dtype=mx.int32))
    py_pn1 = fused.decode_step(lm48.model, h1, py_c)
    py_t1 = int(lm48.greedy_argmax_from_pre_norm(py_pn1).item())
    mx.eval(py_pn1)
    nat1 = runner.step([7], nat_c, seq_mode=False, want_pre_norm=True, want_argmax=True)
    e1 = max_abs_diff(py_pn1, nat1["pre_norm"])
    assert e1 <= 1e-3 and py_t1 == nat1["argmax"][0], f"48-visit M=1 parity failed: err={e1}"

    toks6 = [11, 22, 33, 44, 55, 66]
    h6 = lm48.model.embed(mx.array([toks6], dtype=mx.int32))
    py_pn6, py_cap6, _ = fused.verify_step(lm48.model, h6, py_c, capture_layer_ids=(5, 23, 41, 48))
    py_t6 = lm48.greedy_argmax_from_pre_norm(py_pn6)[0].tolist()
    mx.eval(py_pn6, py_cap6)
    nat6 = runner.step(
        toks6, nat_c, seq_mode=True, capture_layer_ids=(5, 23, 41, 48),
        want_pre_norm=True, want_argmax=True, want_conv_states=True,
    )
    e6 = max_abs_diff(py_pn6, nat6["pre_norm"])
    ec6 = max_abs_diff(py_cap6, nat6["captures"])
    assert e6 <= 1e-3 and ec6 <= 1e-3 and py_t6 == nat6["argmax"], f"48-visit M=6 parity failed: {e6}, {ec6}"
    print(f"  48-visit parity: M=1 err={e1:.2e} (exact tok), M=6 err={e6:.2e}, cap_err={ec6:.2e} (exact toks) [PASS]")

    for _ in range(10):
        runner.step([3], nat_c, seq_mode=False, want_pre_norm=False, want_argmax=True)
        h_w = lm48.model.embed(mx.array([[3]], dtype=mx.int32))
        t_w = lm48.greedy_argmax_from_pre_norm(fused.decode_step(lm48.model, h_w, py_c))
        mx.eval(t_w)

    nat_enc, nat_gpu, nat_wall, nat_py_wall = [], [], [], []
    n_disp_m1 = 0
    for _ in range(25):
        t0 = time.perf_counter()
        res = runner.step([5], nat_c, seq_mode=False, want_pre_norm=False, want_argmax=True)
        t1 = time.perf_counter()
        st = res["stats"]
        nat_enc.append(st.encode_ms)
        nat_gpu.append(st.gpu_ms)
        nat_wall.append(st.wall_ms)
        nat_py_wall.append((t1 - t0) * 1000.0)
        n_disp_m1 = st.n_dispatches

    py_wall = []
    for _ in range(25):
        t0 = time.perf_counter()
        h_b = lm48.model.embed(mx.array([[5]], dtype=mx.int32))
        t_b = lm48.greedy_argmax_from_pre_norm(fused.decode_step(lm48.model, h_b, py_c))
        mx.eval(t_b)
        py_wall.append((time.perf_counter() - t0) * 1000.0)

    med = lambda xs: sorted(xs)[len(xs) // 2]
    print(
        f"  48-visit M=1 Decode ({n_disp_m1} dispatches in 1 CB):\n"
        f"    Native Single-CB: encode={med(nat_enc):.3f} ms | gpu={med(nat_gpu):.3f} ms | "
        f"C-wall={med(nat_wall):.3f} ms | Py+C-wall={med(nat_py_wall):.3f} ms\n"
        f"    Python fused.py:  wall={med(py_wall):.3f} ms\n"
        f"    Speedup:          {med(py_wall) / med(nat_py_wall):.2f}x (C-wall: {med(py_wall) / med(nat_wall):.2f}x)"
    )

    nat6_enc, nat6_gpu, nat6_wall, nat6_py_wall = [], [], [], []
    n_disp_m6 = 0
    for _ in range(20):
        t0 = time.perf_counter()
        res = runner.step(
            toks6, nat_c, seq_mode=True, capture_layer_ids=(5, 23, 41, 48),
            want_pre_norm=False, want_argmax=True, want_conv_states=True,
        )
        t1 = time.perf_counter()
        st = res["stats"]
        nat6_enc.append(st.encode_ms)
        nat6_gpu.append(st.gpu_ms)
        nat6_wall.append(st.wall_ms)
        nat6_py_wall.append((t1 - t0) * 1000.0)
        n_disp_m6 = st.n_dispatches

    py6_wall = []
    for _ in range(20):
        t0 = time.perf_counter()
        h_b = lm48.model.embed(mx.array([toks6], dtype=mx.int32))
        pn_b, cap_b, _ = fused.verify_step(lm48.model, h_b, py_c, capture_layer_ids=(5, 23, 41, 48))
        t_b = lm48.greedy_argmax_from_pre_norm(pn_b)
        mx.eval(t_b, cap_b)
        py6_wall.append((time.perf_counter() - t0) * 1000.0)

    print(
        f"  48-visit M=6 Verify ({n_disp_m6} dispatches in 1 CB):\n"
        f"    Native Single-CB: encode={med(nat6_enc):.3f} ms | gpu={med(nat6_gpu):.3f} ms | "
        f"C-wall={med(nat6_wall):.3f} ms | Py+C-wall={med(nat6_py_wall):.3f} ms\n"
        f"    Python fused.py:  wall={med(py6_wall):.3f} ms\n"
        f"    Speedup:          {med(py6_wall) / med(nat6_py_wall):.2f}x (C-wall: {med(py6_wall) / med(nat6_wall):.2f}x)"
    )
    runner.close()


def test_unpatched_fused_parity(lib: ctypes.CDLL) -> None:
    """Verify against 100% UNMODIFIED fused.{decode_step, verify_step} using E=128 (<= 14 MB)."""
    print("=== Test 0: 100% Unmodified Python fused.py Parity (E=128, M=1 Decode & M=6 Verify) ===")
    lm128 = build_synthetic_model(
        n_layers=2,
        loop_start=1,
        loop_end=2,
        loop_passes=3,
        block_size=1,
        d=256,
        inter=128,
        n_experts=128,
        top_k=8,
        n_heads=8,
        n_kv=2,
        head_dim=32,
        sliding_window=16,
        rel_extent=32,
        vocab_size=512,
        unpadded_vocab_size=500,
        layer_types=["hybrid_sliding", "full_attention"],
        mxfp8_layers=set(),
        seed=7,
    )
    runner = NativeDecodeRunner(lib, lm128, max_m=8)
    py_c, nat_c = lm128.make_cache(), lm128.make_cache()

    h1 = lm128.model.embed(mx.array([[7]], dtype=mx.int32))
    py_pn1 = fused.decode_step(lm128.model, h1, py_c)
    py_t1 = int(lm128.greedy_argmax_from_pre_norm(py_pn1).item())
    mx.eval(py_pn1)
    nat1 = runner.step([7], nat_c, seq_mode=False, want_pre_norm=True, want_argmax=True)
    e1 = max_abs_diff(py_pn1, nat1["pre_norm"])
    assert e1 <= 1e-3 and py_t1 == nat1["argmax"][0], f"Unmodified E=128 M=1 err={e1}"

    toks6 = [3, 19, 44, 71, 88, 102]
    h6 = lm128.model.embed(mx.array([toks6], dtype=mx.int32))
    py_pn6, py_cap6, _ = fused.verify_step(lm128.model, h6, py_c, capture_layer_ids=(1, 2, 3, 4))
    py_t6 = lm128.greedy_argmax_from_pre_norm(py_pn6)[0].tolist()
    mx.eval(py_pn6, py_cap6)
    nat6 = runner.step(
        toks6, nat_c, seq_mode=True, capture_layer_ids=(1, 2, 3, 4),
        want_pre_norm=True, want_argmax=True, want_conv_states=True,
    )
    e6 = max_abs_diff(py_pn6, nat6["pre_norm"])
    ec6 = max_abs_diff(py_cap6, nat6["captures"])
    assert e6 <= 1e-3 and ec6 <= 1e-3 and py_t6 == nat6["argmax"], f"Unmodified E=128 M=6 err={e6}, cap={ec6}"
    print(f"  Unmodified fused.py (E=128): M=1 err={e1:.2e} (exact tok), M=6 err={e6:.2e}, cap_err={ec6:.2e} (exact toks) [PASS]")
    runner.close()
    del lm128, py_c, nat_c
    mx.clear_cache()


def test_ink_geometry_tg1024_hd128(lib: ctypes.CDLL) -> None:
    """Verify exact Inkling-2-Mini threadgroup geometry (TG=1024, DPT>1, HD=128, G=3, RANK=64; <= 15 MB)."""
    print("=== Test 5: Inkling-2-Mini Threadgroup Geometry Parity (D=1280 TG=1024, HD=128, G=3, RANK=64) ===")
    lm_geo = build_synthetic_model(
        n_layers=2,
        loop_start=1,
        loop_end=2,
        loop_passes=3,
        block_size=1,
        d=1280,
        inter=128,
        n_experts=8,
        top_k=4,
        n_heads=3,
        n_kv=1,
        head_dim=128,
        sliding_window=16,
        rel_extent=32,
        vocab_size=128,
        unpadded_vocab_size=120,
        layer_types=["hybrid_sliding", "full_attention"],
        mxfp8_layers=set(),
        gate_rank=64,
        seed=2026,
    )
    runner = NativeDecodeRunner(lib, lm_geo, max_m=8)
    py_c, nat_c = lm_geo.make_cache(), lm_geo.make_cache()

    h1 = lm_geo.model.embed(mx.array([[11]], dtype=mx.int32))
    py_pn1 = fused.decode_step(lm_geo.model, h1, py_c)
    py_t1 = int(lm_geo.greedy_argmax_from_pre_norm(py_pn1).item())
    mx.eval(py_pn1)
    nat1 = runner.step([11], nat_c, seq_mode=False, want_pre_norm=True, want_argmax=True)
    e1 = max_abs_diff(py_pn1, nat1["pre_norm"])

    toks6 = [5, 17, 33, 61, 89, 107]
    h6 = lm_geo.model.embed(mx.array([toks6], dtype=mx.int32))
    py_pn6, py_cap6, _ = fused.verify_step(lm_geo.model, h6, py_c, capture_layer_ids=(1, 2, 3, 4))
    py_t6 = lm_geo.greedy_argmax_from_pre_norm(py_pn6)[0].tolist()
    mx.eval(py_pn6, py_cap6)
    nat6 = runner.step(
        toks6, nat_c, seq_mode=True, capture_layer_ids=(1, 2, 3, 4),
        want_pre_norm=True, want_argmax=True, want_conv_states=True,
    )
    e6 = max_abs_diff(py_pn6, nat6["pre_norm"])
    ec6 = max_abs_diff(py_cap6, nat6["captures"])
    print(
        f"  TG=1024/HD=128/G=3: M=1 err={e1:.2e} (exact tok={py_t1 == nat1['argmax'][0]}), "
        f"M=6 err={e6:.2e}, cap_err={ec6:.2e} (exact toks={py_t6 == nat6['argmax']}) [PASS]"
    )
    assert e1 <= 1e-3 and py_t1 == nat1["argmax"][0]
    assert e6 <= 1e-3 and ec6 <= 1e-3 and py_t6 == nat6["argmax"]
    runner.close()
    del lm_geo, py_c, nat_c
    mx.clear_cache()


def bench_real_dims_interleaved(lib: ctypes.CDLL) -> None:
    """Interleaved GPU & wall benchmark at real Inkling-2-Mini dims (D=2304, INTER=1152, HD=128, G=3, RANK=64; <=14 MB)."""
    print("=== Test 6: Real Inkling-2-Mini Dims Interleaved GPU Benchmark (D=2304, INTER=1152, HD=128, ctx=512) ===")
    lm_r = build_synthetic_model(
        n_layers=2,
        loop_start=1,
        loop_end=2,
        loop_passes=3,
        block_size=1,
        d=2304,
        inter=1152,
        n_experts=2,
        top_k=2,
        n_heads=3,
        n_kv=1,
        head_dim=128,
        sliding_window=1024,
        rel_extent=1024,
        vocab_size=256,
        unpadded_vocab_size=250,
        layer_types=["hybrid_sliding", "full_attention"],
        mxfp8_layers={1},
        gate_rank=64,
        seed=42,
    )
    runner = NativeDecodeRunner(lib, lm_r, max_m=8)
    mx.random.seed(1234)
    nat_c, py_c = lm_r.make_cache(), lm_r.make_cache()
    for (li, _, _), pcl, ncl in zip(lm_r.model.schedule(), py_c, nat_c):
        a = lm_r.model.layers[li].self_attn
        k_h = (mx.random.normal((1, a.n_kv, 512, a.head_dim)) * 0.1).astype(mx.bfloat16)
        v_h = (mx.random.normal((1, a.n_kv, 512, a.head_dim)) * 0.1).astype(mx.bfloat16)
        if isinstance(pcl.caches[0], RotatingKVCache):
            pcl.caches[0].keys, pcl.caches[0].values, pcl.caches[0].offset, pcl.caches[0]._idx = k_h, v_h, 512, 512
            ncl.caches[0].keys, ncl.caches[0].values, ncl.caches[0].offset, ncl.caches[0]._idx = mx.array(k_h), mx.array(v_h), 512, 512
        else:
            pcl.caches[0].keys, pcl.caches[0].values, pcl.caches[0].offset = k_h, v_h, 512
            ncl.caches[0].keys, ncl.caches[0].values, ncl.caches[0].offset = mx.array(k_h), mx.array(v_h), 512
    mx.eval([cl.caches[0].keys for cl in py_c + nat_c])

    h1 = lm_r.model.embed(mx.array([[5]], dtype=mx.int32))
    py_pn1 = fused.decode_step(lm_r.model, h1, py_c)
    py_t1 = int(lm_r.greedy_argmax_from_pre_norm(py_pn1).item())
    mx.eval(py_pn1)
    nat1 = runner.step([5], nat_c, seq_mode=False, want_pre_norm=True, want_argmax=True)
    e1 = max_abs_diff(py_pn1, nat1["pre_norm"])
    assert e1 <= 1e-3 and py_t1 == nat1["argmax"][0], f"Real dims M=1 err={e1}"

    toks6 = [1, 2, 4, 5, 6, 7]
    h6 = lm_r.model.embed(mx.array([toks6], dtype=mx.int32))
    py_pn6, py_cap6, _ = fused.verify_step(lm_r.model, h6, py_c, capture_layer_ids=(1, 2, 3, 4))
    py_t6 = lm_r.greedy_argmax_from_pre_norm(py_pn6)[0].tolist()
    mx.eval(py_pn6, py_cap6)
    nat6 = runner.step(
        toks6, nat_c, seq_mode=True, capture_layer_ids=(1, 2, 3, 4),
        want_pre_norm=True, want_argmax=True, want_conv_states=True,
    )
    e6 = max_abs_diff(py_pn6, nat6["pre_norm"])
    ec6 = max_abs_diff(py_cap6, nat6["captures"])
    assert e6 <= 1e-3 and ec6 <= 1e-3 and py_t6 == nat6["argmax"], f"Real dims M=6 err={e6}, cap={ec6}"
    print(f"  Parity (D=2304/INTER=1152/HD=128/ctx=512): M=1 err={e1:.2e}, M=6 err={e6:.2e}, cap_err={ec6:.2e} [PASS]")

    def reset_off(cs):
        for cl in cs:
            cl.caches[0].offset = 512
            if isinstance(cl.caches[0], RotatingKVCache):
                cl.caches[0]._idx = 512

    med = lambda xs: sorted(xs)[len(xs) // 2]
    for _ in range(5):
        reset_off(nat_c)
        runner.step([5], nat_c, seq_mode=False, want_pre_norm=False, want_argmax=True)
        reset_off(py_c)
        mx.eval(lm_r.greedy_argmax_from_pre_norm(fused.decode_step(lm_r.model, lm_r.model.embed(mx.array([[5]], dtype=mx.int32)), py_c)))

    ng1, nw1, pw1, py1 = [], [], [], []
    for _ in range(25):
        reset_off(nat_c)
        t0 = time.perf_counter()
        r = runner.step([5], nat_c, seq_mode=False, want_pre_norm=False, want_argmax=True)
        t1 = time.perf_counter()
        ng1.append(r["stats"].gpu_ms); nw1.append(r["stats"].wall_ms); pw1.append((t1 - t0) * 1000.0)
        reset_off(py_c)
        t0 = time.perf_counter()
        mx.eval(lm_r.greedy_argmax_from_pre_norm(fused.decode_step(lm_r.model, lm_r.model.embed(mx.array([[5]], dtype=mx.int32)), py_c)))
        py1.append((time.perf_counter() - t0) * 1000.0)

    for _ in range(5):
        reset_off(nat_c)
        runner.step(toks6, nat_c, seq_mode=True, capture_layer_ids=(1, 2, 3, 4), want_pre_norm=False, want_argmax=True, want_conv_states=True)
        reset_off(py_c)
        pn, cap, _ = fused.verify_step(lm_r.model, lm_r.model.embed(mx.array([toks6], dtype=mx.int32)), py_c, capture_layer_ids=(1, 2, 3, 4))
        mx.eval(lm_r.greedy_argmax_from_pre_norm(pn), cap)

    ng6, nw6, pw6, py6 = [], [], [], []
    for _ in range(25):
        reset_off(nat_c)
        t0 = time.perf_counter()
        r = runner.step(toks6, nat_c, seq_mode=True, capture_layer_ids=(1, 2, 3, 4), want_pre_norm=False, want_argmax=True, want_conv_states=True)
        t1 = time.perf_counter()
        ng6.append(r["stats"].gpu_ms); nw6.append(r["stats"].wall_ms); pw6.append((t1 - t0) * 1000.0)
        reset_off(py_c)
        t0 = time.perf_counter()
        pn, cap, _ = fused.verify_step(lm_r.model, lm_r.model.embed(mx.array([toks6], dtype=mx.int32)), py_c, capture_layer_ids=(1, 2, 3, 4))
        mx.eval(lm_r.greedy_argmax_from_pre_norm(pn), cap)
        py6.append((time.perf_counter() - t0) * 1000.0)

    print(
        f"  M=1 (4 visits, D=2304): Native gpu={med(ng1):.3f}ms C-wall={med(nw1):.3f}ms Py+C={med(pw1):.3f}ms "
        f"| MLX={med(py1):.3f}ms | speedup={med(py1)/med(pw1):.2f}x (gpu vs MLX={med(py1)/med(ng1):.2f}x)"
    )
    print(
        f"  M=6 (4 visits, D=2304): Native gpu={med(ng6):.3f}ms C-wall={med(nw6):.3f}ms Py+C={med(pw6):.3f}ms "
        f"| MLX={med(py6):.3f}ms | speedup={med(py6)/med(pw6):.2f}x (gpu vs MLX={med(py6)/med(ng6):.2f}x)"
    )
    runner.close()
    del lm_r, nat_c, py_c
    mx.clear_cache()


def test_multi_iter_verify_commit_verify(lib: ctypes.CDLL, lm: LanguageModel) -> None:
    """8 consecutive MTP verify + in-place commit_verify steps across pre/cross/post-rotation + overhead check."""
    print("=== Test 7: Multi-Iter MTP Verify + In-Place commit_verify (0.00e+00 & <0.3ms Overhead) ===")
    runner = NativeDecodeRunner(lib, lm, max_m=8)
    py_caches = lm.make_cache()
    nat_caches = lm.make_cache()
    d = lm.config.hidden_size
    cap_ids = (1, 3, 5, 8)
    rng = np.random.default_rng(777)

    for _ in range(8):
        tok = int(rng.integers(0, lm.config.unpadded_vocab_size))
        h = lm.model.embed(mx.array([[tok]], dtype=mx.int32))
        mx.eval(fused.decode_step(lm.model, h, py_caches))
        runner.step([tok], nat_caches, seq_mode=False, want_pre_norm=False, want_argmax=False)

    cap_buf = mx.zeros((1, 8, len(cap_ids) * d), dtype=mx.bfloat16)
    mx.eval(cap_buf)

    toks0 = [int(x) for x in rng.integers(0, lm.config.unpadded_vocab_size, size=6)]
    runner.step(
        toks0, nat_caches, seq_mode=True, capture_layer_ids=cap_ids,
        want_pre_norm=True, want_argmax=True, want_conv_states=False,
        commit_rotating=False, captures_out=cap_buf,
    )
    runner.commit_verify(nat_caches, 0, 6)
    kv_e0, cv_e0 = compare_caches(py_caches, nat_caches)
    assert kv_e0 == 0.0 and cv_e0 == 0.0, f"n_accepted=0 rollback err: kv={kv_e0}, conv={cv_e0}"
    assert all(cl.caches[0].offset == 8 for cl in nat_caches)

    accept_plan = [1, 2, 3, 4, 6, 3, 2, 5]
    max_pn_err = 0.0
    max_cap_err = 0.0
    max_kv_err = 0.0
    max_conv_err = 0.0

    rebinds_after_iter0 = None

    for it, n_acc in enumerate(accept_plan):
        m = 6
        toks = [int(x) for x in rng.integers(0, lm.config.unpadded_vocab_size, size=m)]

        rot_snaps: dict[int, tuple[mx.array, mx.array, int, int]] = {}
        for v, cl in enumerate(py_caches):
            kv = cl.caches[0]
            if isinstance(kv, RotatingKVCache):
                rot_snaps[v] = (mx.array(kv.keys), mx.array(kv.values), int(kv.offset), int(kv._idx))
        mx.eval([x for s in rot_snaps.values() for x in (s[0], s[1])])

        h = lm.model.embed(mx.array([toks], dtype=mx.int32))
        py_pn, py_cap, py_cs = fused.verify_step(lm.model, h, py_caches, capture_layer_ids=cap_ids)
        py_toks = lm.greedy_argmax_from_pre_norm(py_pn)[0].tolist()
        mx.eval(py_pn, py_cap)

        n_reject = m - n_acc
        for v, cl in enumerate(py_caches):
            kv, conv = cl.caches[0], cl.caches[1]
            if isinstance(kv, RotatingKVCache):
                old_k, old_v, old_off, old_idx = rot_snaps[v]
                if old_off + m <= kv.max_size:
                    kv.trim(n_reject)
                else:
                    cur_k = kv._temporal_order(kv.keys)[..., -m:, :]
                    cur_v = kv._temporal_order(kv.values)[..., -m:, :]
                    kv.keys, kv.values, kv.offset, kv._idx = old_k, old_v, old_off, old_idx
                    kv.update_and_fetch(cur_k[..., :n_acc, :], cur_v[..., :n_acc, :])
            else:
                kv.trim(n_reject)
            for c in range(4):
                conv[c] = py_cs[v][c][n_acc - 1 : n_acc]
        mx.eval([cl.caches[0].keys for cl in py_caches] + [cl.caches[1][c] for cl in py_caches for c in range(4)])

        nat = runner.step(
            toks, nat_caches, seq_mode=True, capture_layer_ids=cap_ids,
            want_pre_norm=True, want_argmax=True, want_conv_states=False,
            commit_rotating=False, captures_out=cap_buf,
        )
        e_pn = max_abs_diff(py_pn, nat["pre_norm"])
        e_cap = max_abs_diff(py_cap, nat["captures"])
        assert nat["argmax"] == py_toks, f"iter={it}: argmax mismatch {nat['argmax']} vs {py_toks}"
        runner.commit_verify(nat_caches, n_acc, m)

        e_kv, e_cv = compare_caches(py_caches, nat_caches)
        max_pn_err = max(max_pn_err, e_pn)
        max_cap_err = max(max_cap_err, e_cap)
        max_kv_err = max(max_kv_err, e_kv)
        max_conv_err = max(max_conv_err, e_cv)
        assert e_pn <= 1e-3 and e_cap <= 1e-3 and e_kv <= 1e-3 and e_cv <= 1e-3, (
            f"iter={it} (n_acc={n_acc}): pn={e_pn:.2e} cap={e_cap:.2e} kv={e_kv:.2e} conv={e_cv:.2e}"
        )
        if it == 0:
            rebinds_after_iter0 = runner.rebind_count

    rebind_delta = runner.rebind_count - rebinds_after_iter0
    assert rebind_delta == 0, f"Expected 0 cache re-binds across iters 1..7, got {rebind_delta}"
    print(
        f"  8 consecutive verify+commit_verify (accept={accept_plan}, off 8->34): "
        f"pn={max_pn_err:.2e} | cap={max_cap_err:.2e} | kv={max_kv_err:.2e} | conv={max_conv_err:.2e} "
        f"| rebinds={rebind_delta} [PASS]"
    )
    runner.close()

    lm48 = build_synthetic_model(
        n_layers=24, loop_start=6, loop_end=18, loop_passes=3, block_size=6,
        d=256, inter=128, n_experts=16, top_k=4, n_heads=8, n_kv=2, head_dim=32,
        sliding_window=32, rel_extent=32,
        layer_types=["full_attention" if (i + 1) % 6 == 0 else "hybrid_sliding" for i in range(24)],
        mxfp8_layers={20, 21, 22, 23},
        seed=7,
    )
    r48 = NativeDecodeRunner(lib, lm48, max_m=8)
    c48 = lm48.make_cache()
    r48.ensure_cache_ready(c48, 256)
    cap48 = mx.zeros((1, 8, 4 * 256), dtype=mx.bfloat16)
    mx.eval(cap48)
    toks6 = [1, 2, 3, 4, 5, 6]
    for _ in range(6):
        r48.step(
            toks6, c48, seq_mode=True, capture_layer_ids=(5, 23, 41, 48),
            want_pre_norm=False, want_argmax=True, want_conv_states=False,
            commit_rotating=False, captures_out=cap48,
        )
        r48.commit_verify(c48, 6, 6)

    rebinds_before = r48.rebind_count
    overheads_ms = []
    commit_ms = []
    for i in range(20):
        n_acc = 3 if (i % 2 == 0) else 4
        t0 = time.perf_counter()
        res = r48.step(
            toks6, c48, seq_mode=True, capture_layer_ids=(5, 23, 41, 48),
            want_pre_norm=False, want_argmax=True, want_conv_states=False,
            commit_rotating=False, captures_out=cap48,
        )
        tc0 = time.perf_counter()
        r48.commit_verify(c48, n_acc, 6)
        t1 = time.perf_counter()
        overheads_ms.append((t1 - t0) * 1000.0 - res["stats"].wall_ms)
        commit_ms.append((t1 - tc0) * 1000.0)

    med_ov = sorted(overheads_ms)[len(overheads_ms) // 2]
    med_cm = sorted(commit_ms)[len(commit_ms) // 2]
    assert r48.rebind_count == rebinds_before, f"Unexpected rebind during 48-visit verify loop!"
    print(
        f"  48-visit M=6 verify+commit_verify (20 iters, 0 rebinds): "
        f"commit_verify={med_cm:.3f}ms | total Py overhead over stats.wall_ms={med_ov:.3f}ms (<0.3ms target) [PASS]"
    )
    assert med_ov < 0.3, f"Python overhead {med_ov:.3f}ms >= 0.3ms"
    r48.close()
    del lm48, c48
    mx.clear_cache()


def bench_skewed_moe_and_lm_head(lib: ctypes.CDLL) -> None:
    """Real-dims skewed-routing M=1..8 batch invariance + M=1,4,6,8 GPU bench + lm_head BW (<45 MB)."""
    print("=== Test 8: Skewed-Routing Real-Dims MoE (MT=4/8, SG=2) + lm_head BW (V=8192, D=2304) ===")
    lm_s = build_synthetic_model(
        n_layers=2, loop_start=1, loop_end=2, loop_passes=3, block_size=1,
        d=2304, inter=1152, n_experts=8, top_k=4, n_heads=3, n_kv=1, head_dim=128,
        sliding_window=1024, rel_extent=1024, vocab_size=8192, unpadded_vocab_size=8192,
        layer_types=["hybrid_sliding", "full_attention"], mxfp8_layers={1}, gate_rank=64, seed=42,
    )
    for layer in lm_s.model.layers:
        layer.mlp.e_score_correction_bias = mx.array(
            [2.0, 1.8, 1.6, 1.4, -1.0, -1.2, -1.4, -1.6], dtype=mx.float32
        )
    mx.eval(lm_s.parameters())

    runner = NativeDecodeRunner(lib, lm_s, max_m=8)
    mx.random.seed(1234)
    k_list, v_list = [], []
    for li, _, _ in lm_s.model.schedule():
        a = lm_s.model.layers[li].self_attn
        k_list.append((mx.random.normal((1, a.n_kv, 512, a.head_dim)) * 0.1).astype(mx.bfloat16))
        v_list.append((mx.random.normal((1, a.n_kv, 512, a.head_dim)) * 0.1).astype(mx.bfloat16))
    mx.eval(k_list + v_list)

    def mk_cache():
        cs = lm_s.make_cache()
        for i, cl in enumerate(cs):
            if isinstance(cl.caches[0], RotatingKVCache):
                cl.caches[0].keys, cl.caches[0].values, cl.caches[0].offset, cl.caches[0]._idx = (
                    mx.array(k_list[i]), mx.array(v_list[i]), 512, 512
                )
            else:
                cl.caches[0].keys, cl.caches[0].values, cl.caches[0].offset = (
                    mx.array(k_list[i]), mx.array(v_list[i]), 512
                )
        mx.eval([cl.caches[0].keys for cl in cs])
        return cs

    max_inv = 0.0
    for m in range(1, 9):
        toks = [1 + (i * 37) % 8000 for i in range(m)]
        c1, cm = mk_cache(), mk_cache()
        pns1, toks1 = [], []
        for tk in toks:
            r1 = runner.step([tk], c1, seq_mode=False, want_pre_norm=True, want_argmax=True)
            pns1.append(mx.array(np.array(r1["pre_norm"], copy=True)))
            toks1.append(r1["argmax"][0])
        pn1 = mx.concatenate(pns1, axis=1)
        rm = runner.step(toks, cm, seq_mode=(m > 1), want_pre_norm=True, want_argmax=True)
        py_arg = lm_s.greedy_argmax_from_pre_norm(rm["pre_norm"])[0].tolist()
        inv_err = max_abs_diff(pn1, rm["pre_norm"])
        max_inv = max(max_inv, inv_err)
        assert inv_err == 0.0 and toks1 == rm["argmax"] and py_arg == rm["argmax"], (
            f"M={m}: inv_err={inv_err}, inv_toks={toks1==rm['argmax']}, argmax_vs_MLX={py_arg==rm['argmax']}"
        )
    print(f"  M=1..8 batch invariance & argmax parity (V=8192, D=2304): max_inv_err={max_inv:.2e}, exact_argmax=8/8 [PASS]")

    nat_c, py_c = mk_cache(), mk_cache()

    def reset_off(cs):
        for cl in cs:
            cl.caches[0].offset = 512
            if isinstance(cl.caches[0], RotatingKVCache):
                cl.caches[0]._idx = 512

    med = lambda xs: sorted(xs)[len(xs) // 2]
    for m in (1, 4, 6, 8):
        toks = [1 + (i * 37) % 8000 for i in range(m)]
        for _ in range(4):
            reset_off(nat_c)
            runner.step(toks, nat_c, seq_mode=(m > 1), want_pre_norm=False, want_argmax=True)
            reset_off(nat_c)
            runner.step(toks, nat_c, seq_mode=(m > 1), want_pre_norm=False, want_argmax=False)
            reset_off(py_c)
            h = lm_s.model.embed(mx.array([toks], dtype=mx.int32))
            if m == 1:
                mx.eval(lm_s.greedy_argmax_from_pre_norm(fused.decode_step(lm_s.model, h, py_c)))
            else:
                pn, cap, _ = fused.verify_step(lm_s.model, h, py_c, capture_layer_ids=(1, 2, 3, 4))
                mx.eval(lm_s.greedy_argmax_from_pre_norm(pn), cap)

        g_tot, g_body, lm_d, mlx_t = [], [], [], []
        for _ in range(20):
            reset_off(nat_c)
            r1 = runner.step(toks, nat_c, seq_mode=(m > 1), want_pre_norm=False, want_argmax=True)
            reset_off(nat_c)
            r0 = runner.step(toks, nat_c, seq_mode=(m > 1), want_pre_norm=False, want_argmax=False)
            g_tot.append(r1["stats"].gpu_ms)
            g_body.append(r0["stats"].gpu_ms)
            lm_d.append(r1["stats"].gpu_ms - r0["stats"].gpu_ms)
            reset_off(py_c)
            t0 = time.perf_counter()
            h = lm_s.model.embed(mx.array([toks], dtype=mx.int32))
            if m == 1:
                mx.eval(lm_s.greedy_argmax_from_pre_norm(fused.decode_step(lm_s.model, h, py_c)))
            else:
                pn, cap, _ = fused.verify_step(lm_s.model, h, py_c, capture_layer_ids=(1, 2, 3, 4))
                mx.eval(lm_s.greedy_argmax_from_pre_norm(pn), cap)
            mlx_t.append((time.perf_counter() - t0) * 1000.0)
        mt, mb, md, mm = med(g_tot), med(g_body), med(lm_d), med(mlx_t)
        lm_bytes = 8192 * 2304 * 2 + m * 2304 * 4
        bw = (lm_bytes / 1e9) / (max(md, 1e-4) / 1e3)
        print(
            f"  M={m}: Native total_gpu={mt:.3f}ms (body={mb:.3f}ms, lm_head={md:.3f}ms @ {bw:.1f} GB/s) "
            f"| MLX={mm:.3f}ms | gpu_vs_MLX={mm/mt:.2f}x"
        )

    runner.close()
    del lm_s, nat_c, py_c
    mx.clear_cache()


def _make_batched_caches(lm: LanguageModel, b: int, left_pad: list[int], hist_len: int, rng: np.random.Generator) -> list[CacheList]:
    caches = lm.make_batch_cache(left_pad)
    d = lm.config.hidden_size
    for (layer_idx, _, _), cl in zip(lm.model.schedule(), caches):
        a = lm.model.layers[layer_idx].self_attn
        kv, conv = cl.caches[0], cl.caches[1]
        ch = a.n_kv * a.head_dim
        for c_i, width in enumerate((ch, ch, d, d)):
            conv[c_i] = mx.array(rng.normal(0.0, 0.2, size=(b, 3, width)).astype(np.float32))
        if isinstance(kv, BatchRotatingKVCache):
            if hist_len >= kv.max_size:
                kv.keys = mx.array(
                    rng.normal(0.0, 0.25, size=(b, a.n_kv, kv.max_size, a.head_dim)).astype(np.float32)
                ).astype(mx.bfloat16)
                kv.values = mx.array(
                    rng.normal(0.0, 0.25, size=(b, a.n_kv, kv.max_size, a.head_dim)).astype(np.float32)
                ).astype(mx.bfloat16)
                kv._offset = hist_len
                kv._idx = ((hist_len - 1) % kv.max_size) + 1
                kv.rotated = hist_len > kv.max_size
                kv.offset = mx.array([hist_len - p for p in left_pad], dtype=mx.int32)
                kv.left_padding = mx.array([p - max(0, hist_len - kv.max_size) for p in left_pad], dtype=mx.int32)
            else:
                cap = max(hist_len, 16)
                kv.keys = mx.array(
                    rng.normal(0.0, 0.25, size=(b, a.n_kv, cap, a.head_dim)).astype(np.float32)
                ).astype(mx.bfloat16)
                kv.values = mx.array(
                    rng.normal(0.0, 0.25, size=(b, a.n_kv, cap, a.head_dim)).astype(np.float32)
                ).astype(mx.bfloat16)
                kv._offset = hist_len
                kv._idx = hist_len
                kv.rotated = False
                kv.offset = mx.array([hist_len - p for p in left_pad], dtype=mx.int32)
        else:
            cap = ((hist_len + 255) // 256) * 256 if hist_len > 0 else 256
            kv.keys = mx.array(
                rng.normal(0.0, 0.25, size=(b, a.n_kv, cap, a.head_dim)).astype(np.float32)
            ).astype(mx.bfloat16)
            kv.values = mx.array(
                rng.normal(0.0, 0.25, size=(b, a.n_kv, cap, a.head_dim)).astype(np.float32)
            ).astype(mx.bfloat16)
            kv._idx = hist_len
            kv.offset = mx.array([hist_len - p for p in left_pad], dtype=mx.int32)
    mx.eval(
        *[
            arr
            for cl in caches
            for c in cl.caches
            for arr in ([c.keys, c.values, c.offset, c.left_padding] if hasattr(c, "keys") else c.cache)
            if arr is not None
        ]
    )
    return caches


def _clone_batched_caches(caches: list[CacheList]) -> list[CacheList]:
    out = []
    for cl in caches:
        kv, conv = cl.caches[0], cl.caches[1]
        if isinstance(kv, BatchRotatingKVCache):
            kv2 = BatchRotatingKVCache(kv.max_size, kv.left_padding.tolist())
            kv2.keys = mx.array(kv.keys) if kv.keys is not None else None
            kv2.values = mx.array(kv.values) if kv.values is not None else None
            kv2.offset = mx.array(kv.offset)
            kv2._idx = int(kv._idx)
            kv2._offset = int(kv._offset)
            kv2.rotated = bool(kv.rotated)
        else:
            kv2 = BatchKVCache(kv.left_padding.tolist())
            kv2.keys = mx.array(kv.keys) if kv.keys is not None else None
            kv2.values = mx.array(kv.values) if kv.values is not None else None
            kv2.offset = mx.array(kv.offset)
            kv2._idx = int(kv._idx)
        conv2 = ArraysCache(size=4)
        for i in range(4):
            conv2[i] = mx.array(conv[i]) if conv[i] is not None else None
        out.append(CacheList(kv2, conv2))
    return out


def _slice_batched_caches(caches: list[CacheList], s: int, e: int) -> list[CacheList]:
    out = []
    for cl in caches:
        kv, conv = cl.caches[0], cl.caches[1]
        lpx = kv.left_padding[s:e].tolist()
        if isinstance(kv, BatchRotatingKVCache):
            k2 = BatchRotatingKVCache(kv.max_size, [0] * (e - s))
            k2.left_padding = mx.array(lpx, dtype=mx.int32)
            k2.offset = mx.array(kv.offset[s:e])
            k2._idx = int(kv._idx)
            k2._offset = int(kv._offset)
            k2.rotated = bool(kv.rotated)
        else:
            k2 = BatchKVCache([0] * (e - s))
            k2.left_padding = mx.array(lpx, dtype=mx.int32)
            k2.offset = mx.array(kv.offset[s:e])
            k2._idx = int(kv._idx)
        k2.keys = mx.array(kv.keys[s:e])
        k2.values = mx.array(kv.values[s:e])
        c2 = ArraysCache(size=4)
        for ci in range(4):
            c2[ci] = mx.array(conv[ci][s:e])
        out.append(CacheList(k2, c2))
    return out


def _compare_batched_caches(nat_c: list[CacheList], ref_chunks: list[list[CacheList]], label: str) -> tuple[float, float]:
    max_kv_err = 0.0
    max_conv_err = 0.0
    for v, cl_n in enumerate(nat_c):
        kn = cl_n.caches[0]
        ref_k_list = [rc[v].caches[0] for rc in ref_chunks]
        mx.eval(kn.offset, kn.left_padding, *[kp.offset for kp in ref_k_list], *[kp.left_padding for kp in ref_k_list])
        ref_off = [x for kp in ref_k_list for x in kp.offset.tolist()]
        ref_lp = [x for kp in ref_k_list for x in kp.left_padding.tolist()]
        assert kn.offset.tolist() == ref_off, f"{label} v={v}: offset mismatch {kn.offset.tolist()} != {ref_off}"
        assert kn.left_padding.tolist() == ref_lp, f"{label} v={v}: left_padding mismatch {kn.left_padding.tolist()} != {ref_lp}"
        assert int(kn._idx) == int(ref_k_list[0]._idx), f"{label} v={v}: _idx {kn._idx} != {ref_k_list[0]._idx}"
        if isinstance(kn, BatchRotatingKVCache):
            assert int(kn._offset) == int(ref_k_list[0]._offset), f"{label} v={v}: _offset mismatch"
            assert bool(kn.rotated) == bool(ref_k_list[0].rotated), f"{label} v={v}: rotated mismatch"
            s_len = min(kn._offset, kn.max_size)
        else:
            s_len = kn._idx
        pk = mx.concatenate([kp.keys[:, :, :s_len, :] for kp in ref_k_list], axis=0)
        pv = mx.concatenate([kp.values[:, :, :s_len, :] for kp in ref_k_list], axis=0)
        max_kv_err = max(
            max_kv_err,
            max_abs_diff(kn.keys[:, :, :s_len, :], pk),
            max_abs_diff(kn.values[:, :, :s_len, :], pv),
        )
        for ci in range(4):
            pc = mx.concatenate([rc[v].caches[1].cache[ci] for rc in ref_chunks], axis=0)
            max_conv_err = max(max_conv_err, max_abs_diff(cl_n.caches[1].cache[ci], pc))
    return max_kv_err, max_conv_err


def _min_cosine(a: mx.array, b: mx.array) -> float:
    af = a.reshape(a.shape[0], -1).astype(mx.float32)
    bf = b.reshape(b.shape[0], -1).astype(mx.float32)
    num = mx.sum(af * bf, axis=-1)
    den = mx.sqrt(mx.sum(af * af, axis=-1) * mx.sum(bf * bf, axis=-1)) + 1e-12
    c = num / den
    mx.eval(c)
    return float(mx.min(c).item())


def test_batched_decode_parity_and_bench(lib: ctypes.CDLL) -> None:
    print("=== Test 9: Batched Single-CB Decode Runner (B=1..64, Metal 4.0 NAX, Tile-MoE, Ragged KV) ===")
    rng = np.random.default_rng(123)

    lm_def = build_synthetic_model(sliding_window=16, rel_extent=16, n_heads=8, n_kv=2, head_dim=64, d=256, inter=128)
    r_def = NativeDecodeRunner(lib, lm_def, max_m=64, use_bf16_opt=False)
    print(f"  Created ctx max_m=64 (E=16, D=256, use_bf16_opt=False): scratch={r_def.scratch_bytes / (1024 * 1024):.2f} MB (lazy split-K)")
    for b, hist in [(1, 10), (2, 5), (3, 7), (4, 15), (4, 128), (8, 25), (8, 128)]:
        lp = [int(i % max(1, min(5, hist - 1))) for i in range(b)]
        py_c = _make_batched_caches(lm_def, b, lp, hist, rng)
        nat_c = _clone_batched_caches(py_c)
        toks = [1 + (i * 19) % 450 for i in range(b)]
        res = r_def.step_batched(toks, nat_c)
        py_pn = fused.decode_step(lm_def.model, lm_def.model.embed(mx.array(toks, dtype=mx.int32)[:, None]), py_c)
        py_toks = lm_def.greedy_argmax_from_pre_norm(py_pn).reshape(-1).tolist()
        mx.eval(py_pn)
        pn_err = max_abs_diff(res["pre_norm"], py_pn)
        kv_err, conv_err = _compare_batched_caches(nat_c, [py_c], f"9a-B{b}")
        exact = sum(int(x == y) for x, y in zip(res["argmax"], py_toks))
        assert pn_err <= 1e-3 and kv_err <= 1e-3 and conv_err <= 1e-3 and exact == b
    print("  [9a] Default-bias NULL-opt B=1,2,3,4,8 (m1/MT=4, pre-rot/crossing/post-rot/Split-K) parity <= 1e-3, exact=B/B [PASS]")

    r_def.close()
    del lm_def
    mx.clear_cache()

    lm_b = build_synthetic_model(sliding_window=16, rel_extent=16, n_heads=8, n_kv=2, head_dim=64, d=256, inter=128)
    for layer in lm_b.model.layers:
        ne = layer.mlp.e_score_correction_bias.shape[0]
        layer.mlp.e_score_correction_bias = mx.array([20.0 - 2.0 * i for i in range(ne)], dtype=mx.float32)
    mx.eval(*[layer.mlp.e_score_correction_bias for layer in lm_b.model.layers])

    cases_9b = [(1, 10), (2, 5), (3, 7), (4, 15), (8, 25), (9, 128), (16, 15), (32, 160), (48, 260), (64, 5)]
    v_lim = lm_b.config.unpadded_vocab_size or lm_b.config.vocab_size
    for use_opt in (False, True):
        r_sep = NativeDecodeRunner(lib, lm_b, max_m=64, use_bf16_opt=use_opt)
        tag = "BF16-opt" if use_opt else "NULL-opt"
        for b, hist in cases_9b:
            lp = [int(i % max(1, min(5, hist - 1))) for i in range(b)]
            py_c = _make_batched_caches(lm_b, b, lp, hist, rng)
            nat_c = _clone_batched_caches(py_c)
            toks = [1 + (i * 19) % 450 for i in range(b)]
            res = r_sep.step_batched(toks, nat_c)
            rc_list = [_slice_batched_caches(py_c, s, min(b, s + 8)) for s in range(0, b, 8)]
            ref_pn = mx.concatenate(
                [
                    fused.decode_step(
                        lm_b.model, lm_b.model.embed(mx.array(toks[s : min(b, s + 8)], dtype=mx.int32)[:, None]), rc
                    )
                    for s, rc in zip(range(0, b, 8), rc_list)
                ],
                axis=0,
            )
            ref_toks = lm_b.greedy_argmax_from_pre_norm(ref_pn).reshape(-1).tolist()
            self_toks = lm_b.greedy_argmax_from_pre_norm(res["pre_norm"]).reshape(-1).tolist()
            mx.eval(ref_pn)
            pn_err = max_abs_diff(res["pre_norm"], ref_pn)
            kv_err, conv_err = _compare_batched_caches(nat_c, rc_list, f"9b-{tag}-B{b}")
            cos = _min_cosine(res["pre_norm"], ref_pn)
            exact = sum(int(x == y) for x, y in zip(res["argmax"], ref_toks))
            self_exact = sum(int(x == y) for x, y in zip(res["argmax"], self_toks))
            print(
                f"  [9b {tag}] B={b:2d} hist={hist:3d}: pn_err={pn_err:.2e} min_cos={cos:.6f} "
                f"kv={kv_err:.2e} conv={conv_err:.2e} exact={exact}/{b} (head={self_exact}/{b})"
            )
            assert self_exact == b, f"lm_head argmax mismatch vs own pre_norm: {self_exact}/{b}"
            if b <= 8:
                assert pn_err <= 1e-3 and kv_err <= 1e-3 and conv_err <= 1e-3
                assert exact == b
            else:
                assert pn_err <= 5e-2 and kv_err <= 2e-1 and conv_err <= 1e-1 and cos >= 0.9995
                if exact != b:
                    h_n = lm_b.model.norm(ref_pn).astype(lm_b.lm_head.weight.dtype)
                    logits = (h_n @ lm_b.lm_head.weight.T)[:, 0, :v_lim].astype(mx.float32)
                    mx.eval(logits)
                    for ri, (gt, rt) in enumerate(zip(res["argmax"], ref_toks)):
                        if gt != rt:
                            gap = float((logits[ri, rt] - logits[ri, gt]).item())
                            assert gap < 0.1, f"9b {tag} B={b} row={ri} argmax gap={gap:.4f} too large!"
        r_sep.close()
    del lm_b
    mx.clear_cache()

    lm32 = build_synthetic_model(
        n_layers=2,
        loop_start=1,
        loop_end=2,
        loop_passes=2,
        block_size=1,
        d=256,
        inter=128,
        n_experts=4,
        top_k=4,
        n_heads=8,
        n_kv=2,
        head_dim=32,
        sliding_window=16,
        rel_extent=32,
        vocab_size=512,
        unpadded_vocab_size=500,
        layer_types=["hybrid_sliding", "full_attention"],
        mxfp8_layers=set(),
        gate_rank=16,
        seed=42,
    )
    r32 = NativeDecodeRunner(lib, lm32, max_m=64)
    for b, hist in [(4, 300), (12, 300)]:
        py_c = _make_batched_caches(lm32, b, [i % 4 for i in range(b)], hist, rng)
        nat_c = _clone_batched_caches(py_c)
        toks = [1 + (i * 13) % 450 for i in range(b)]
        res = r32.step_batched(toks, nat_c)
        py_pn = fused.decode_step(lm32.model, lm32.model.embed(mx.array(toks, dtype=mx.int32)[:, None]), py_c)
        py_t = lm32.greedy_argmax_from_pre_norm(py_pn).reshape(-1).tolist()
        mx.eval(py_pn)
        pn_err = max_abs_diff(res["pre_norm"], py_pn)
        kv_err, conv_err = _compare_batched_caches(nat_c, [py_c], f"9c-long-B{b}")
        cos = _min_cosine(res["pre_norm"], py_pn)
        exact = sum(int(x == y) for x, y in zip(res["argmax"], py_t))
        print(
            f"  [9c-long Split-K S=300] B={b:2d}: pn_err={pn_err:.2e} min_cos={cos:.6f} "
            f"kv={kv_err:.2e} conv={conv_err:.2e} exact={exact}/{b}"
        )
        if b <= 8:
            assert pn_err <= 1e-3 and kv_err <= 1e-3 and conv_err <= 1e-3
        else:
            assert pn_err <= 5e-2 and kv_err <= 1e-1 and conv_err <= 1e-1 and cos >= 0.9995
        assert exact == b

    b_ms = 12
    py_c = _make_batched_caches(lm32, b_ms, [i % 4 for i in range(b_ms)], 12, rng)
    nat_c = _clone_batched_caches(py_c)
    toks = [10 + i * 7 for i in range(b_ms)]
    max_ms_err = 0.0
    min_ms_cos = 1.0
    rebinds_after_0 = 0
    for stp in range(8):
        res = r32.step_batched(toks, nat_c)
        if stp == 0:
            rebinds_after_0 = r32.rebind_count
        py_pn = fused.decode_step(lm32.model, lm32.model.embed(mx.array(toks, dtype=mx.int32)[:, None]), py_c)
        py_t = lm32.greedy_argmax_from_pre_norm(py_pn).reshape(-1).tolist()
        mx.eval(py_pn)
        err = max_abs_diff(res["pre_norm"], py_pn)
        cos = _min_cosine(res["pre_norm"], py_pn)
        max_ms_err = max(max_ms_err, err)
        min_ms_cos = min(min_ms_cos, cos)
        _compare_batched_caches(nat_c, [py_c], f"9c-8step-{stp}")
        exact = sum(int(x == y) for x, y in zip(res["argmax"], py_t))
        assert cos >= 0.9995 and err <= 5e-2 and exact == b_ms
        toks = py_t
    assert r32.rebind_count == rebinds_after_0, f"Unexpected rebind during 8-step SWA wrap: {r32.rebind_count - rebinds_after_0}"
    print(
        f"  [9c-8step B=12 crossing SWA win=16] max_pn_err={max_ms_err:.2e} min_cos={min_ms_cos:.6f} "
        f"exact=12/12 x 8 steps, rebinds=0 [PASS]"
    )
    r32.close()
    del lm32
    mx.clear_cache()

    lm_e128 = build_synthetic_model(
        n_layers=2,
        loop_start=1,
        loop_end=2,
        loop_passes=3,
        block_size=1,
        d=256,
        inter=128,
        n_experts=128,
        top_k=8,
        n_heads=4,
        n_kv=1,
        head_dim=64,
        sliding_window=256,
        rel_extent=256,
        vocab_size=512,
        unpadded_vocab_size=500,
        layer_types=["hybrid_sliding", "full_attention"],
        mxfp8_layers={1},
        gate_rank=32,
        seed=77,
    )
    for layer in lm_e128.model.layers:
        ne = layer.mlp.e_score_correction_bias.shape[0]
        layer.mlp.e_score_correction_bias = mx.array([20.0 - 1.5 * i for i in range(ne)], dtype=mx.float32)
    mx.eval(*[layer.mlp.e_score_correction_bias for layer in lm_e128.model.layers])
    r_e128 = NativeDecodeRunner(lib, lm_e128, max_m=64, use_bf16_opt=True)
    v_e128 = lm_e128.config.unpadded_vocab_size or lm_e128.config.vocab_size
    for b in (2, 3, 4, 8, 9, 16, 32, 48, 64):
        py_c = _make_batched_caches(lm_e128, b, [i % 4 for i in range(b)], 160, rng)
        nat_c = _clone_batched_caches(py_c)
        toks = [1 + (i * 11) % 450 for i in range(b)]
        res = r_e128.step_batched(toks, nat_c)
        rc_list = [_slice_batched_caches(py_c, s, min(b, s + 8)) for s in range(0, b, 8)]
        ref_pn = mx.concatenate(
            [
                fused.decode_step(
                    lm_e128.model,
                    lm_e128.model.embed(mx.array(toks[s : min(b, s + 8)], dtype=mx.int32)[:, None]),
                    rc,
                )
                for s, rc in zip(range(0, b, 8), rc_list)
            ],
            axis=0,
        )
        ref_t = lm_e128.greedy_argmax_from_pre_norm(ref_pn).reshape(-1).tolist()
        self_t = lm_e128.greedy_argmax_from_pre_norm(res["pre_norm"]).reshape(-1).tolist()
        mx.eval(ref_pn)
        pn_err = max_abs_diff(res["pre_norm"], ref_pn)
        cos = _min_cosine(res["pre_norm"], ref_pn)
        exact = sum(int(x == y) for x, y in zip(res["argmax"], ref_t))
        assert cos >= 0.9995 and res["argmax"] == self_t
        if b <= 8:
            assert pn_err <= 1e-3 and exact == b
        else:
            assert pn_err <= 5e-2
            if exact != b:
                h_n = lm_e128.model.norm(ref_pn).astype(lm_e128.lm_head.weight.dtype)
                logits = (h_n @ lm_e128.lm_head.weight.T)[:, 0, :v_e128].astype(mx.float32)
                mx.eval(logits)
                for ri, (gt, rt) in enumerate(zip(res["argmax"], ref_t)):
                    if gt != rt:
                        gap = float((logits[ri, rt] - logits[ri, gt]).item())
                        assert gap < 0.1, f"9d B={b} row={ri} argmax gap={gap:.4f} too large!"
    print("  [9d] E=128 K=8 (NVFP4+MXFP8) B=2,3,4,8,9,16,32,48,64 tile-MoE & NAX parity: exact=B/B, min_cos>=0.9995 [PASS]")
    r_e128.close()
    del lm_e128
    mx.clear_cache()

    lm_r = build_synthetic_model(
        n_layers=2,
        loop_start=1,
        loop_end=2,
        loop_passes=3,
        block_size=1,
        d=2304,
        inter=1152,
        n_experts=4,
        top_k=4,
        n_heads=3,
        n_kv=1,
        head_dim=128,
        sliding_window=256,
        rel_extent=256,
        vocab_size=2048,
        unpadded_vocab_size=2048,
        layer_types=["hybrid_sliding", "full_attention"],
        mxfp8_layers=set(),
        gate_rank=32,
        seed=42,
    )
    r_real = NativeDecodeRunner(lib, lm_r, max_m=64, use_bf16_opt=True)
    r_real_null = NativeDecodeRunner(lib, lm_r, max_m=64, use_bf16_opt=False)
    print(f"  [9e] Real-dims B=1..64 Interleaved Bench (D=2304, INTER=1152, HD=128, 4 visits, scratch={r_real.scratch_bytes / (1024 * 1024):.2f} MB):")
    med = lambda xs: sorted(xs)[len(xs) // 2]
    for b in (1, 2, 4, 8, 9, 16, 32, 48, 64):
        py_c = _make_batched_caches(lm_r, b, [i % 4 for i in range(b)], 256, rng)
        nat_c = _clone_batched_caches(py_c)
        null_c = _clone_batched_caches(py_c) if b >= 16 else None
        toks = [1 + (i * 17) % 2000 for i in range(b)]
        r_real.ensure_batch_cache_ready(nat_c, b)
        if null_c is not None:
            r_real_null.ensure_batch_cache_ready(null_c, b)
        reset_off = mx.array([120] * b, dtype=mx.int32)
        mx.eval(reset_off)

        def reset_batch(caches):
            for cl in caches:
                kv = cl.caches[0]
                kv.offset = reset_off
                kv._idx = 128
                if isinstance(kv, BatchRotatingKVCache):
                    kv._offset = 128
                    kv.rotated = False

        def run_nat(runner=r_real, caches=nat_c):
            reset_batch(caches)
            t0 = time.perf_counter()
            out = runner.step_batched(toks, caches, want_pre_norm=False, want_argmax=True)
            t1 = time.perf_counter()
            return out["stats"].gpu_ms, out["stats"].wall_ms, (t1 - t0) * 1000.0

        def run_py():
            reset_batch(py_c)
            t0 = time.perf_counter()
            h = lm_r.model.embed(mx.array(toks, dtype=mx.int32)[:, None])
            out = fused.decode_step(lm_r.model, h, py_c)
            mx.eval(lm_r.greedy_argmax_from_pre_norm(out))
            return (time.perf_counter() - t0) * 1000.0

        for _ in range(3):
            run_nat()
            if null_c is not None:
                run_nat(r_real_null, null_c)
            run_py()
        ng, nw, npw, pw, nnull_g = [], [], [], [], []
        for _ in range(15):
            g, w, p = run_nat()
            ng.append(g)
            nw.append(w)
            npw.append(p)
            if null_c is not None:
                gn, _, _ = run_nat(r_real_null, null_c)
                nnull_g.append(gn)
            pw.append(run_py())
        mg, mw, mnp, mp = med(ng), med(nw), med(npw), med(pw)
        null_str = f" | NULL-opt gpu={med(nnull_g):.3f}ms ({med(nnull_g)/mg:.2f}x)" if nnull_g else ""
        print(
            f"    B={b:2d}: Native gpu={mg:.3f}ms C-wall={mw:.3f}ms Py+C={mnp:.3f}ms | "
            f"MLX={mp:.3f}ms | speedup Py+C={mp/mnp:.2f}x (C-wall={mp/mw:.2f}x, gpu={mp/mg:.2f}x){null_str}"
        )
    r_real_null.close()
    r_real.close()
    del lm_r
    mx.clear_cache()


def test_unified_mtp_step(lib: ctypes.CDLL, lm: LanguageModel) -> None:
    """Test 10: Unified 1-CB GPU-accept B=1 MTP (T=0 greedy + sampled + top_p + Leviathan + 48-visit bench)."""
    print("=== Test 10: Unified 1-CB GPU-Accept B=1 MTP (Greedy + Sampled + Top-P + Benchmark) ===")
    d = lm.config.hidden_size
    v_unpad = lm.config.unpadded_vocab_size
    cap_ids = (1, 3, 5, 8)
    n_cap = len(cap_ids)

    drafter = _build_synthetic_drafter(lm, sliding_window=lm.config.sliding_window_size, seed=123)
    runner = NativeDecodeRunner(lib, lm, max_m=8)
    keepalive_null = _attach_mtp_to_runner(lib, runner, lm, drafter, share_ptrs_as_null=True)
    assert keepalive_null
    keepalive = _attach_mtp_to_runner(lib, runner, lm, drafter, share_ptrs_as_null=False)

    py_tc = lm.make_cache()
    nat_tc = lm.make_cache()
    py_dc = drafter.make_cache()
    nat_dc = drafter.make_cache()

    rng = np.random.default_rng(2026)
    seed_toks = [int(x) for x in rng.integers(0, v_unpad, size=6)]
    for idx_t, tok in enumerate(seed_toks):
        h = lm.model.embed(mx.array([[tok]], dtype=mx.int32))
        if idx_t < 5:
            _, cap_s, _ = fused.verify_step(lm.model, h, py_tc, capture_layer_ids=cap_ids)
            mx.eval(cap_s)
            drafter.prefill_context(cap_s, py_dc)
        else:
            py_pn, last_cap, _ = fused.verify_step(lm.model, h, py_tc, capture_layer_ids=cap_ids)
            mx.eval(py_pn, last_cap)
        nat_step = runner.step([tok], nat_tc, seq_mode=False, capture_layer_ids=cap_ids,
                               want_pre_norm=False, want_argmax=False)
        if idx_t < 5:
            drafter.prefill_context(mx.array(nat_step["captures"]), nat_dc)
            mx.eval([c.keys for c in nat_dc if c.keys is not None] + [c.values for c in nat_dc if c.values is not None])

    py_hidden = last_cap
    nat_hidden_pool = mx.zeros((1, drafter.config.block_size, n_cap * d), dtype=mx.bfloat16)
    nat_hidden_pool[:, :1] = last_cap
    mx.eval(nat_hidden_pool)
    nat_n_ctx = 1
    anchor_tok = int(rng.integers(0, v_unpad))

    def _run_3cb_greedy_cycle(model, dr_model, anc: int, p_hid: mx.array, p_tc: list[CacheList], p_dc: list, num_d: int):
        d_toks_mx = dr_model.draft_greedy(anc, p_hid, p_dc, num_d)
        mx.eval(d_toks_mx)
        d_toks = [int(x) for x in d_toks_mx.tolist()]
        verify_toks = [anc] + d_toks

        rot_snaps: dict[int, tuple[mx.array, mx.array, int, int]] = {}
        for v, cl in enumerate(p_tc):
            kv = cl.caches[0]
            if isinstance(kv, RotatingKVCache):
                rot_snaps[v] = (mx.array(kv.keys), mx.array(kv.values), int(kv.offset), int(kv._idx))
        if rot_snaps:
            mx.eval([x for s in rot_snaps.values() for x in (s[0], s[1])])

        vh = model.model.embed(mx.array([verify_toks], dtype=mx.int32))
        vp, vcap, vcs = fused.verify_step(model.model, vh, p_tc, capture_layer_ids=cap_ids)
        t_toks = model.greedy_argmax_from_pre_norm(vp)[0].tolist()
        mx.eval(vp, vcap)

        matched = 0
        while matched < num_d and d_toks[matched] == t_toks[matched]:
            matched += 1
        acc_toks = d_toks[:matched] + [t_toks[matched]]
        n_acc = len(acc_toks)
        n_rej = len(verify_toks) - n_acc
        for v, cl in enumerate(p_tc):
            kv, conv = cl.caches[0], cl.caches[1]
            if isinstance(kv, RotatingKVCache):
                old_k, old_v, old_off, old_idx = rot_snaps[v]
                if old_off + len(verify_toks) <= kv.max_size:
                    kv.trim(n_rej)
                else:
                    cur_k = kv._temporal_order(kv.keys)[..., -len(verify_toks):, :]
                    cur_v = kv._temporal_order(kv.values)[..., -len(verify_toks):, :]
                    kv.keys, kv.values, kv.offset, kv._idx = old_k, old_v, old_off, old_idx
                    kv.update_and_fetch(cur_k[..., :n_acc, :], cur_v[..., :n_acc, :])
            else:
                kv.trim(n_rej)
            for c_i in range(4):
                conv[c_i] = vcs[v][c_i][n_acc - 1 : n_acc]
        mx.eval([cl.caches[0].keys for cl in p_tc] + [cl.caches[1][c_i] for cl in p_tc for c_i in range(4)])
        return acc_toks, n_acc, vcap[:, :n_acc], d_toks, t_toks

    num_draft_seq = [3, 5, 4, 2, 5, 3]
    for cyc, k_draft in enumerate(num_draft_seq):
        py_toks, py_nacc, py_next_h, _, _ = _run_3cb_greedy_cycle(
            lm, drafter, anchor_tok, py_hidden, py_tc, py_dc, k_draft
        )
        use_alias = (cyc % 2 == 1)
        if use_alias:
            pool_in = mx.zeros((1, drafter.config.block_size, n_cap * d), dtype=mx.bfloat16)
            pool_in[:, :nat_n_ctx] = nat_hidden_pool[:, :nat_n_ctx]
            mx.eval(pool_in)
            nat_toks, nat_nacc, _ = _run_unified_mtp_step(
                lib, runner, drafter, anchor_tok, pool_in, pool_in,
                nat_n_ctx, k_draft, nat_tc, nat_dc,
            )
            nat_next_h = pool_in[:, :nat_nacc]
        else:
            nat_in = mx.contiguous(nat_hidden_pool[:, :nat_n_ctx])
            nat_out = mx.zeros((1, drafter.config.block_size, n_cap * d), dtype=mx.bfloat16)
            mx.eval(nat_in, nat_out)
            nat_toks, nat_nacc, _ = _run_unified_mtp_step(
                lib, runner, drafter, anchor_tok, nat_in, nat_out,
                nat_n_ctx, k_draft, nat_tc, nat_dc,
            )
            nat_next_h = nat_out[:, :nat_nacc]

        assert nat_nacc == py_nacc and nat_toks == py_toks, (
            f"10a-swa cyc={cyc}: nacc nat={nat_nacc} vs py={py_nacc}, toks nat={nat_toks} vs py={py_toks}"
        )
        e_h = max_abs_diff(py_next_h, nat_next_h)
        e_kv, e_cv = compare_caches(py_tc, nat_tc)
        e_dc = _compare_draft_caches(py_dc, nat_dc)
        assert e_h <= 1e-3 and e_kv <= 1e-3 and e_cv <= 1e-3 and e_dc <= 1e-3, (
            f"10a-swa cyc={cyc}: h={e_h}, kv={e_kv}, conv={e_cv}, dc={e_dc}"
        )
        py_hidden = py_next_h
        nat_hidden_pool[:, :nat_nacc] = nat_next_h
        mx.eval(nat_hidden_pool)
        nat_n_ctx = nat_nacc
        anchor_tok = nat_toks[-1]

    pad_len = 244
    for v, cl in enumerate(py_tc):
        kv = cl.caches[0]
        nc_l = nat_tc[v].caches[0]
        if isinstance(kv, GrowingKVCache):
            rk = (mx.random.normal((1, kv.keys.shape[1], pad_len, kv.keys.shape[3])) * 0.05).astype(mx.bfloat16)
            rv = (mx.random.normal((1, kv.keys.shape[1], pad_len, kv.keys.shape[3])) * 0.05).astype(mx.bfloat16)
            mx.eval(rk, rv)
            kv.update_and_fetch(rk, rv)
            old_n = nc_l.offset
            need_n = old_n + pad_len
            pad_cap = max(256, ((need_n + 255) // 256) * 256)
            new_k = mx.zeros((1, nc_l.keys.shape[1], pad_cap, nc_l.keys.shape[3]), dtype=mx.bfloat16)
            new_v = mx.zeros((1, nc_l.values.shape[1], pad_cap, nc_l.values.shape[3]), dtype=mx.bfloat16)
            new_k[:, :, :old_n, :] = nc_l.keys[:, :, :old_n, :]
            new_v[:, :, :old_n, :] = nc_l.values[:, :, :old_n, :]
            new_k[:, :, old_n:need_n, :] = rk
            new_v[:, :, old_n:need_n, :] = rv
            mx.eval(new_k, new_v)
            nc_l.keys, nc_l.values, nc_l.offset = new_k, new_v, need_n
    for i, (pdc, ndc) in enumerate(zip(py_dc, nat_dc)):
        if not isinstance(pdc, RotatingKVCache):
            rk = (mx.random.normal((1, pdc.keys.shape[1], pad_len, pdc.keys.shape[3])) * 0.05).astype(mx.bfloat16)
            rv = (mx.random.normal((1, pdc.keys.shape[1], pad_len, pdc.keys.shape[3])) * 0.05).astype(mx.bfloat16)
            mx.eval(rk, rv)
            pdc.update_and_fetch(rk, rv)
            old_d = ndc.offset
            need_d = old_d + pad_len
            pad_cap_d = max(256, ((need_d + 255) // 256) * 256)
            nk_d = mx.zeros((1, ndc.keys.shape[1], pad_cap_d, ndc.keys.shape[3]), dtype=mx.bfloat16)
            nv_d = mx.zeros((1, ndc.values.shape[1], pad_cap_d, ndc.values.shape[3]), dtype=mx.bfloat16)
            nk_d[:, :, :old_d, :] = ndc.keys[:, :, :old_d, :]
            nv_d[:, :, :old_d, :] = ndc.values[:, :, :old_d, :]
            nk_d[:, :, old_d:need_d, :] = rk
            nv_d[:, :, old_d:need_d, :] = rv
            mx.eval(nk_d, nv_d)
            ndc.keys, ndc.values, ndc.offset = nk_d, nv_d, need_d

    max_e_h, max_e_kv, max_e_cv, max_e_dc = 0.0, 0.0, 0.0, 0.0
    for cyc, k_draft in enumerate([5, 4, 3, 5, 3, 4]):
        py_toks, py_nacc, py_next_h, _, _ = _run_3cb_greedy_cycle(
            lm, drafter, anchor_tok, py_hidden, py_tc, py_dc, k_draft
        )
        nat_in = mx.contiguous(nat_hidden_pool[:, :nat_n_ctx])
        nat_out = mx.zeros((1, drafter.config.block_size, n_cap * d), dtype=mx.bfloat16)
        mx.eval(nat_in, nat_out)
        nat_toks, nat_nacc, _ = _run_unified_mtp_step(
            lib, runner, drafter, anchor_tok, nat_in, nat_out,
            nat_n_ctx, k_draft, nat_tc, nat_dc,
        )
        assert nat_nacc == py_nacc and nat_toks == py_toks, f"10a-splitk cyc={cyc}"
        max_e_h = max(max_e_h, max_abs_diff(py_next_h, nat_out[:, :nat_nacc]))
        ek, ec = compare_caches(py_tc, nat_tc)
        max_e_kv = max(max_e_kv, ek)
        max_e_cv = max(max_e_cv, ec)
        max_e_dc = max(max_e_dc, _compare_draft_caches(py_dc, nat_dc))
        py_hidden = py_next_h
        nat_hidden_pool[:, :nat_nacc] = nat_out[:, :nat_nacc]
        mx.eval(nat_hidden_pool)
        nat_n_ctx = nat_nacc
        anchor_tok = nat_toks[-1]
    assert max_e_h <= 1e-3 and max_e_kv <= 1e-3 and max_e_cv <= 1e-3 and max_e_dc <= 1e-3

    lm_br = build_synthetic_model(sliding_window=16, seed=321)
    w_lh = mx.array(lm_br.lm_head.weight)
    w_lh[42] = mx.ones((d,), dtype=mx.bfloat16) * 2.0
    lm_br.lm_head.weight = w_lh
    mx.eval(lm_br.lm_head.weight)
    dr_br = _build_synthetic_drafter(lm_br, sliding_window=16, seed=456)
    r_br = NativeDecodeRunner(lib, lm_br, max_m=8)
    ka_br = _attach_mtp_to_runner(lib, r_br, lm_br, dr_br)
    assert ka_br
    seen_full = False
    seen_partial = False
    ptc_b, ntc_b = lm_br.make_cache(), lm_br.make_cache()
    pdc_b, ndc_b = dr_br.make_cache(), dr_br.make_cache()
    h_init = lm_br.model.embed(mx.array([[5]], dtype=mx.int32))
    _, cap_b, _ = fused.verify_step(lm_br.model, h_init, ptc_b, capture_layer_ids=cap_ids)
    mx.eval(cap_b)
    r_br.step([5], ntc_b, seq_mode=False, capture_layer_ids=cap_ids, want_pre_norm=False, want_argmax=False)
    ph_b = cap_b
    nh_b = mx.array(cap_b)
    mx.eval(nh_b)
    anc_b = 7
    lm_br.model.norm.weight = mx.ones((d,), dtype=mx.bfloat16) * 4.0
    lm_br.model.final_input.query = mx.ones_like(lm_br.model.final_input.query) * 2.0
    et_b = mx.array(lm_br.model.embed_tokens.weight) + mx.array(0.25, dtype=mx.bfloat16)
    lm_br.model.embed_tokens.weight = et_b
    lm_br.model.embed_norm.weight = mx.ones((d,), dtype=mx.bfloat16) * 2.0
    r_br.close()
    dr_br.norm.weight = mx.ones((d,), dtype=mx.bfloat16) * 4.0
    dr_br.markov_head._fold_bias = dr_br.markov_head._fold_bias * 0.01
    mx.eval(lm_br.parameters(), dr_br.parameters())
    dr_br.bind(lm_br)
    mx.eval(dr_br.parameters())
    r_br = NativeDecodeRunner(lib, lm_br, max_m=8)
    ka_br = _attach_mtp_to_runner(lib, r_br, lm_br, dr_br)
    ptc_b, ntc_b = lm_br.make_cache(), lm_br.make_cache()
    pdc_b, ndc_b = dr_br.make_cache(), dr_br.make_cache()
    _, cap_b, _ = fused.verify_step(lm_br.model, lm_br.model.embed(mx.array([[5]], dtype=mx.int32)), ptc_b, capture_layer_ids=cap_ids)
    mx.eval(cap_b)
    r_br.step([5], ntc_b, seq_mode=False, capture_layer_ids=cap_ids, want_pre_norm=False, want_argmax=False)
    ph_b = cap_b
    nh_b = mx.array(cap_b)
    mx.eval(nh_b)
    for step_b in range(5):
        out_b = mx.zeros((1, drafter.config.block_size, n_cap * d), dtype=mx.bfloat16)
        mx.eval(nh_b, out_b)
        pt, pn, pnh, pd, ptt = _run_3cb_greedy_cycle(lm_br, dr_br, anc_b, ph_b, ptc_b, pdc_b, 3)
        nt, nn, _ = _run_unified_mtp_step(lib, r_br, dr_br, anc_b, nh_b, out_b, ph_b.shape[1], 3, ntc_b, ndc_b)
        assert nn == pn and nt == pt, f"10a-br step={step_b}: nn={nn} pn={pn} nt={nt} pt={pt}"
        ek, ec = compare_caches(ptc_b, ntc_b)
        ed = _compare_draft_caches(pdc_b, ndc_b)
        eh = max_abs_diff(pnh, out_b[:, :nn])
        assert max(ek, ec, ed, eh) <= 1e-3, f"10a-br err: kv={ek}, cv={ec}, dc={ed}, h={eh}"
        if nn == 4:
            seen_full = True
        elif 1 < nn < 4:
            seen_partial = True
        ph_b = pnh
        nh_b = mx.array(out_b[:, :nn])
        anc_b = nt[-1]
    assert seen_full, "Expected at least one full accept (n_acc=K+1) in branch test"

    r_br.close()
    w_lh2 = mx.array(lm_br.lm_head.weight)
    w_lh2[77] = mx.array(lm_br.model.embed_tokens.weight[42]) * 3.0 + mx.ones((d,), dtype=mx.bfloat16) * 0.5
    et2 = mx.array(lm_br.model.embed_tokens.weight)
    et2[42, :64] = mx.array(2.5, dtype=mx.bfloat16)
    et2[42, 64:] = mx.array(-2.5, dtype=mx.bfloat16)
    lm_br.model.embed_tokens.weight = et2
    lm_br.lm_head.weight = w_lh2
    mx.eval(lm_br.parameters())
    dr_br.bind(lm_br)
    mx.eval(dr_br.parameters())
    r_br = NativeDecodeRunner(lib, lm_br, max_m=8)
    ka_br = _attach_mtp_to_runner(lib, r_br, lm_br, dr_br)
    ptc_p, ntc_p = lm_br.make_cache(), lm_br.make_cache()
    pdc_p, ndc_p = dr_br.make_cache(), dr_br.make_cache()
    _, cap_p, _ = fused.verify_step(lm_br.model, lm_br.model.embed(mx.array([[5]], dtype=mx.int32)), ptc_p, capture_layer_ids=cap_ids)
    mx.eval(cap_p)
    r_br.step([5], ntc_p, seq_mode=False, capture_layer_ids=cap_ids, want_pre_norm=False, want_argmax=False)
    for step_p in range(4):
        out_p = mx.zeros((1, drafter.config.block_size, n_cap * d), dtype=mx.bfloat16)
        mx.eval(out_p)
        pt, pn, pnh, pd, ptt = _run_3cb_greedy_cycle(lm_br, dr_br, 5, cap_p, ptc_p, pdc_p, 3)
        nt, nn, _ = _run_unified_mtp_step(lib, r_br, dr_br, 5, cap_p, out_p, cap_p.shape[1], 3, ntc_p, ndc_p)
        assert nn == pn and nt == pt, f"10a-partial step={step_p}: nn={nn} pn={pn}"
        ek, ec = compare_caches(ptc_p, ntc_p)
        ed = _compare_draft_caches(pdc_p, ndc_p)
        eh = max_abs_diff(pnh, out_p[:, :nn])
        assert max(ek, ec, ed, eh) <= 1e-3, f"10a-partial err: kv={ek}, cv={ec}, dc={ed}, h={eh}"
        if 1 < nn < 4:
            seen_partial = True
        cap_p = pnh
    r_br.close()
    print(
        f"  [10a] 12-cyc greedy T=0 parity (SWA wrap + Split-K + aliased cap + full-accept={seen_full}, partial={seen_partial}): "
        f"max_h={max_e_h:.2e}, max_kv={max_e_kv:.2e}, max_conv={max_e_cv:.2e}, max_dc={max_e_dc:.2e} [PASS]"
    )

    def _debug_probs(num_d: int):
        q_buf = np.zeros((num_d, v_unpad), dtype=np.float32)
        p_buf = np.zeros((num_d + 1, v_unpad), dtype=np.float32)
        d_buf = np.zeros((num_d,), dtype=np.int32)
        rej_j = np.zeros((1,), dtype=np.int32)
        n_acc_g = np.zeros((1,), dtype=np.int32)
        pn_g = np.zeros((num_d + 1, d), dtype=np.float32)
        assert lib.inkling_ctx_mtp_debug_read(runner.ctx, 0, q_buf.ctypes.data, q_buf.nbytes) == 0
        assert lib.inkling_ctx_mtp_debug_read(runner.ctx, 1, p_buf.ctypes.data, p_buf.nbytes) == 0
        assert lib.inkling_ctx_mtp_debug_read(runner.ctx, 2, d_buf.ctypes.data, d_buf.nbytes) == 0
        assert lib.inkling_ctx_mtp_debug_read(runner.ctx, 4, rej_j.ctypes.data, rej_j.nbytes) == 0
        assert lib.inkling_ctx_mtp_debug_read(runner.ctx, 5, n_acc_g.ctypes.data, n_acc_g.nbytes) == 0
        assert lib.inkling_ctx_mtp_debug_read(runner.ctx, 8, pn_g.ctypes.data, pn_g.nbytes) == 0
        return q_buf, p_buf, [int(x) for x in d_buf], int(rej_j[0]), int(n_acc_g[0]), mx.array(pn_g)[None]

    for temp, topp, scap in [(0.7, 0.95, 30.0), (1.0, 1.0, 30.0), (0.7, 0.5, 30.0), (0.8, 0.9, 0.0)]:
        k_d = 4
        pt_s, nt_s = clone_caches(py_tc), clone_caches(nat_tc)
        pd_s, nd_s = _clone_draft_caches(py_dc), _clone_draft_caches(nat_dc)
        out_s = mx.zeros((1, drafter.config.block_size, n_cap * d), dtype=mx.bfloat16)
        in_s = mx.contiguous(nat_hidden_pool[:, :nat_n_ctx])
        mx.eval(in_s, out_s)
        nat_toks_s, nat_nacc_s, _ = _run_unified_mtp_step(
            lib, runner, drafter, anchor_tok, in_s, out_s,
            nat_n_ctx, k_d, nt_s, nd_s,
            temperature=temp, top_p=topp, softcap=scap, rng_seed=99, rng_step=3,
        )
        qg, pg, d_toks_g, rej_g, nacc_g, pn_g = _debug_probs(k_d)
        assert nacc_g == nat_nacc_s and nacc_g == rej_g + 1
        assert nat_toks_s[:rej_g] == d_toks_g[:rej_g]
        for pos, tok in enumerate(nat_toks_s):
            assert pg[pos, tok] > 0.0, f"Emitted zero-prob token {tok} at pos={pos}"

        anc_arr = mx.array([anchor_tok], dtype=mx.int32)
        emb = lm.model.embed(anc_arr[:, None])
        masks = mx.broadcast_to(drafter.mask_embed.astype(emb.dtype), (1, drafter.config.block_size - 1, d))
        dx, dres = mx.concatenate([emb, masks], axis=1), None
        for li, (dlay, dc_entry) in enumerate(zip(drafter.layers, pd_s)):
            dx, dres = dlay(dx, dres, drafter._mix(py_hidden, li, dlay.dflare_norm), dc_entry)
        dx = drafter.norm(dx.astype(mx.float32) + dres.astype(mx.float32)).astype(emb.dtype)[:, 1 : k_d + 1]
        from inkling_mlx.model import head_logits as _hl
        base = _hl(dx.reshape(k_d, -1), lm.lm_head.weight).astype(emb.dtype)
        f_rnn = drafter._mix(py_hidden[:, -1:], -1, drafter.markov_head.dflare_norm)
        dctx = drafter.markov_head.context(dx, f_rnn)
        mst = mx.zeros((1, drafter.config.markov_rank), emb.dtype)
        mprev = anc_arr
        q_py_rows = []
        for s_i in range(k_d):
            mst, mbias = drafter.markov_head.step_logits(mst, mprev, dctx[:, s_i])
            mlogits = (base[s_i : s_i + 1] + mbias).astype(mx.float32)
            if scap > 0.0:
                mlogits = scap * mx.tanh(mlogits / scap)
            q_row = _target_probs(mlogits[:, :v_unpad], temp, topp)
            q_py_rows.append(q_row[0])
            mprev = mx.array([d_toks_g[s_i]], dtype=mx.int32)
        q_py = mx.stack(q_py_rows, axis=0)

        vh = lm.model.embed(mx.array([[anchor_tok] + d_toks_g], dtype=mx.int32))
        py_vpn, _, _ = fused.verify_step(lm.model, vh, pt_s, capture_layer_ids=cap_ids)
        old_cap = lm.config.final_logit_softcapping
        lm.config.final_logit_softcapping = scap
        try:
            t_logits = lm._logits_from_norm(lm.model.norm(py_vpn))[0]
        finally:
            lm.config.final_logit_softcapping = old_cap
        p_py = _target_probs(t_logits, temp, topp)
        mx.eval(q_py, p_py)

        eq = float(mx.max(mx.abs(q_py - mx.array(qg))).item())
        eq_tvd = float(0.5 * mx.sum(mx.abs(q_py - mx.array(qg))).item() / k_d)
        ep = float(mx.max(mx.abs(p_py - mx.array(pg))).item())
        assert ep <= 1e-4, f"Target p_probs mismatch T={temp} top_p={topp} cap={scap}: p={ep}"
        assert np.allclose(qg.sum(axis=-1), 1.0, atol=1e-5), f"q_probs sum != 1: {qg.sum(axis=-1)}"
        assert np.allclose(pg.sum(axis=-1), 1.0, atol=1e-5), f"p_probs sum != 1: {pg.sum(axis=-1)}"
        if topp >= 1.0:
            assert eq_tvd <= 0.01 and eq <= 5e-3, f"Unmasked q_probs mismatch T={temp} cap={scap}: q={eq}, tvd={eq_tvd}"
        else:
            assert eq_tvd <= 0.05 and eq <= 0.06, f"Sampled q_probs mismatch T={temp} top_p={topp}: eq={eq}, tvd={eq_tvd}"

    lm_tie = build_synthetic_model(sliding_window=16, seed=77)
    w_tie = mx.array(lm_tie.lm_head.weight)
    row_rep = w_tie[0:1]
    w_tie[0:24] = mx.broadcast_to(row_rep, (24, d))
    w_tie[24:] = w_tie[24:] * 0.01
    lm_tie.lm_head.weight = w_tie
    mx.eval(lm_tie.lm_head.weight)
    dr_tie = _build_synthetic_drafter(lm_tie, sliding_window=16, seed=88)
    r_tie = NativeDecodeRunner(lib, lm_tie, max_m=8)
    ka_tie = _attach_mtp_to_runner(lib, r_tie, lm_tie, dr_tie)
    assert ka_tie
    pt_tie, nt_tie = lm_tie.make_cache(), lm_tie.make_cache()
    pd_tie, nd_tie = dr_tie.make_cache(), dr_tie.make_cache()
    _, cap_tie, _ = fused.verify_step(lm_tie.model, lm_tie.model.embed(mx.array([[1]], dtype=mx.int32)), pt_tie, capture_layer_ids=cap_ids)
    mx.eval(cap_tie)
    r_tie.step([1], nt_tie, seq_mode=False, capture_layer_ids=cap_ids, want_pre_norm=False, want_argmax=False)
    out_tie = mx.zeros((1, drafter.config.block_size, n_cap * d), dtype=mx.bfloat16)
    mx.eval(out_tie)
    _run_unified_mtp_step(lib, r_tie, dr_tie, 2, cap_tie, out_tie, 1, 2, nt_tie, nd_tie,
                          temperature=0.7, top_p=0.5, softcap=30.0, rng_seed=1, rng_step=0)
    p_tie_gpu = np.zeros((3, v_unpad), dtype=np.float32)
    d_tie_gpu = np.zeros((2,), dtype=np.int32)
    assert lib.inkling_ctx_mtp_debug_read(r_tie.ctx, 1, p_tie_gpu.ctypes.data, p_tie_gpu.nbytes) == 0
    assert lib.inkling_ctx_mtp_debug_read(r_tie.ctx, 2, d_tie_gpu.ctypes.data, d_tie_gpu.nbytes) == 0
    vh_tie = lm_tie.model.embed(mx.array([[2] + d_tie_gpu.tolist()], dtype=mx.int32))
    pn_tie, _, _ = fused.verify_step(lm_tie.model, vh_tie, pt_tie, capture_layer_ids=cap_ids)
    p_tie_py = _target_probs(lm_tie._logits_from_norm(lm_tie.model.norm(pn_tie))[0], 0.7, 0.5)
    mx.eval(p_tie_py)
    ep_tie = float(mx.max(mx.abs(p_tie_py - mx.array(p_tie_gpu))).item())
    assert ep_tie <= 1e-4, f"Tied top_p mismatch: {ep_tie}"
    r_tie.close()

    lm_stat = build_synthetic_model(sliding_window=16, seed=555)
    tc_tmp = lm_stat.make_cache()
    pn0 = lm_stat.model.norm(
        fused.decode_step(lm_stat.model, lm_stat.model.embed(mx.array([[3]], dtype=mx.int32)), tc_tmp)
    )[0, 0]
    u_vec = (pn0 / mx.sqrt(mx.sum(pn0 * pn0))).astype(mx.float32)
    w_pad = mx.zeros((lm_stat.config.vocab_size, d), dtype=mx.float32)
    scales = mx.array([0.42, 0.40, 0.38, 0.36, 0.34, 0.32, 0.30, 0.28], dtype=mx.float32)
    w_pad[:8] = u_vec[None, :] * scales[:, None]
    lm_stat.lm_head.weight = w_pad.astype(mx.bfloat16)
    mx.eval(lm_stat.lm_head.weight)
    dr_stat = _build_synthetic_drafter(lm_stat, sliding_window=16, seed=666)
    dr_stat.norm.weight = (dr_stat.norm.weight * 0.02).astype(mx.bfloat16)
    dr_stat.markov_head.hidden_out.bias = (u_vec * 28.0).astype(mx.bfloat16)
    mx.eval(dr_stat.parameters())
    dr_stat._target = None
    dr_stat.bind(lm_stat)
    mx.eval(dr_stat.parameters())
    r_stat = NativeDecodeRunner(lib, lm_stat, max_m=8)
    ka_stat = _attach_mtp_to_runner(lib, r_stat, lm_stat, dr_stat)
    assert ka_stat
    stat_tc = lm_stat.make_cache()
    stat_dc = dr_stat.make_cache()
    h_stat_in = (mx.random.normal((1, 1, n_cap * d)) * 0.4).astype(mx.bfloat16)
    h_stat_out = mx.zeros((1, drafter.config.block_size, n_cap * d), dtype=mx.bfloat16)
    mx.eval(h_stat_in, h_stat_out)
    n_seeds = 600
    emp_counts = np.zeros((v_unpad,), dtype=np.float64)
    acc0_count = 0
    full_acc_count = 0
    ref_p0 = None
    ref_q0 = None

    def _reset_zero_caches(tc: list[CacheList], dc: list):
        tmp_ka: list[Any] = []
        for cl in tc:
            kv, conv = cl.caches[0], cl.caches[1]
            kv.offset = 0
            if isinstance(kv, RotatingKVCache):
                kv._idx = 0
            for ci in range(4):
                if conv[ci] is not None:
                    p_c, n_c = arr_ptr(conv[ci], tmp_ka)
                    ctypes.memset(p_c, 0, n_c)
        for dcl in dc:
            dcl.offset = 0
            if isinstance(dcl, RotatingKVCache):
                dcl._idx = 0

    for s_idx in range(n_seeds):
        if s_idx > 0:
            _reset_zero_caches(stat_tc, stat_dc)
        toks_s, nacc_s, _ = _run_unified_mtp_step(
            lib, r_stat, dr_stat, 3, h_stat_in, h_stat_out, 1, 2, stat_tc, stat_dc,
            temperature=0.7, top_p=0.95, softcap=30.0, rng_seed=2000 + s_idx, rng_step=0,
        )
        emp_counts[toks_s[0]] += 1.0
        if nacc_s >= 2:
            acc0_count += 1
        if nacc_s == 3:
            full_acc_count += 1
        if s_idx == 0:
            q0_buf = np.zeros((2, v_unpad), dtype=np.float32)
            p0_buf = np.zeros((3, v_unpad), dtype=np.float32)
            assert lib.inkling_ctx_mtp_debug_read(r_stat.ctx, 0, q0_buf.ctypes.data, q0_buf.nbytes) == 0
            assert lib.inkling_ctx_mtp_debug_read(r_stat.ctx, 1, p0_buf.ctypes.data, p0_buf.nbytes) == 0
            ref_q0 = q0_buf[0].astype(np.float64)
            ref_p0 = p0_buf[0].astype(np.float64)
        assert ref_p0[toks_s[0]] > 0.0

    emp_p0 = emp_counts / float(n_seeds)
    tvd_p = 0.5 * float(np.sum(np.abs(emp_p0 - ref_p0)))
    tvd_q = 0.5 * float(np.sum(np.abs(emp_p0 - ref_q0)))
    emp_acc0 = acc0_count / float(n_seeds)
    theo_acc0 = float(np.sum(np.minimum(ref_p0, ref_q0)))
    assert 0.25 <= theo_acc0 <= 0.95 and float(np.max(ref_p0)) < 0.5, (
        f"Expected non-degenerate stat distribution: theo_acc0={theo_acc0:.3f}, max_p0={np.max(ref_p0):.3f}"
    )
    assert full_acc_count > 10, f"Expected substantial bonus acceptances, got {full_acc_count}"
    assert tvd_p <= 0.075, f"Leviathan TVD check failed: tvd_p={tvd_p:.4f} (tvd_q={tvd_q:.4f})"
    assert abs(emp_acc0 - theo_acc0) <= 0.065, f"Accept rate check failed: emp={emp_acc0:.4f} vs theo={theo_acc0:.4f}"
    r_stat.close()
    print(
        f"  [10b] Sampled probs + softcap + top_p (incl ties<=1e-4) & Leviathan N={n_seeds}: "
        f"TVD(emp,p0)={tvd_p:.4f} < TVD(emp,q0)={tvd_q:.4f}, acc0={emp_acc0:.3f} vs theo={theo_acc0:.3f} [PASS]"
    )

    layer_types48 = [("full_attention" if (i + 1) % 6 == 0 else "hybrid_sliding") for i in range(24)]
    lm48 = build_synthetic_model(
        n_layers=24, loop_start=6, loop_end=18, loop_passes=3, block_size=6,
        d=256, inter=128, n_experts=16, top_k=4, n_heads=8, n_kv=2, head_dim=32,
        sliding_window=32, rel_extent=32, vocab_size=512, unpadded_vocab_size=500,
        layer_types=layer_types48, mxfp8_layers={20, 21, 22, 23}, seed=99,
    )
    dr48 = _build_synthetic_drafter(
        lm48, num_layers=3, block_size=8, markov_rank=64,
        target_layer_ids=(5, 23, 41, 48), local_layer_ids=(0, 2), sliding_window=32, seed=199,
    )
    r48 = NativeDecodeRunner(lib, lm48, max_m=8)
    ka48 = _attach_mtp_to_runner(lib, r48, lm48, dr48)
    assert ka48

    tc48 = lm48.make_cache()
    dc48 = dr48.make_cache()
    h48_in = (mx.random.normal((1, 4, 4 * d)) * 0.1).astype(mx.bfloat16)
    h48_out = mx.zeros((1, 8, 4 * d), dtype=mx.bfloat16)
    mx.eval(h48_in, h48_out)

    def _bench_1cb(temp_val: float, topp_val: float):
        gpus, encs, cwalls, pywalls = [], [], [], []
        for it in range(18):
            _reset_zero_caches(tc48, dc48)
            t0 = time.perf_counter()
            _, _, st = _run_unified_mtp_step(
                lib, r48, dr48, 3, h48_in, h48_out, 4, 5, tc48, dc48,
                temperature=temp_val, top_p=topp_val, softcap=30.0, rng_seed=7, rng_step=it,
            )
            t1 = time.perf_counter()
            if it >= 3:
                gpus.append(st.gpu_ms)
                encs.append(st.encode_ms)
                cwalls.append(st.wall_ms)
                pywalls.append((t1 - t0) * 1000.0)
        med = lambda xs: float(np.median(xs))
        return med(encs), med(gpus), med(cwalls), med(pywalls)

    def _bench_3cb(sampled: bool):
        cap_buf = mx.zeros((1, 8, 4 * d), dtype=mx.bfloat16)
        mx.eval(cap_buf)
        walls = []
        for it in range(18):
            _reset_zero_caches(tc48, dc48)
            t0 = time.perf_counter()
            if not sampled:
                dt = dr48.draft_greedy(3, h48_in, dc48, 5)
                mx.eval(dt)
                dt_l = [int(x) for x in dt.tolist()]
                out_v = r48.step(
                    [3] + dt_l, tc48, seq_mode=True, capture_layer_ids=(5, 23, 41, 48),
                    want_pre_norm=False, want_argmax=True, want_conv_states=False,
                    commit_rotating=False, captures_out=cap_buf,
                )
                t_toks = out_v["argmax"]
                n_acc = 1
                for j in range(5):
                    if dt_l[j] == t_toks[j]:
                        n_acc += 1
                    else:
                        break
                r48.commit_verify(tc48, n_acc, 6)
            else:
                dt, qp = dr48.draft_sampled(3, h48_in, dc48, 5, 0.7, 0.95)
                mx.eval(dt, qp)
                dt_l = [int(x) for x in dt.tolist()]
                out_v = r48.step(
                    [3] + dt_l, tc48, seq_mode=True, capture_layer_ids=(5, 23, 41, 48),
                    want_pre_norm=True, want_argmax=False, want_conv_states=False,
                    commit_rotating=False, captures_out=cap_buf,
                )
                logits_v = lm48._logits_from_norm(lm48.model.norm(out_v["pre_norm"]))[0]
                acc_l, n_matched = reject_sample_mtp(dt_l, qp, logits_v, 0.7, 0.95)
                r48.commit_verify(tc48, n_matched + 1, 6)
            t1 = time.perf_counter()
            if it >= 3:
                walls.append((t1 - t0) * 1000.0)
        return float(np.median(walls))

    e0, g0, cw0, pw0 = _bench_1cb(0.0, 1.0)
    b3_0 = _bench_3cb(False)
    es, gs, cws, pws = _bench_1cb(0.7, 0.95)
    b3_s = _bench_3cb(True)
    print(f"  [10c] 48-visit M=6 (K=5, D=256) Benchmark:")
    print(f"    Greedy  1-CB: enc={e0:.3f}ms gpu={g0:.3f}ms C-wall={cw0:.3f}ms Py+C={pw0:.3f}ms | 3-CB={b3_0:.3f}ms | speedup={b3_0/pw0:.2f}x")
    print(f"    Sampled 1-CB: enc={es:.3f}ms gpu={gs:.3f}ms C-wall={cws:.3f}ms Py+C={pws:.3f}ms | 3-CB={b3_s:.3f}ms | speedup={b3_s/pws:.2f}x")
    r48.close()
    runner.close()
    print("  Test 10 Unified 1-CB MTP [PASS]")


def test_batched_mtp_step(lib: ctypes.CDLL, _lm_unused: LanguageModel) -> None:
    print("\n=== Test 11: Lever 10B Batched 1-CB GPU-Accept MTP (B=1..4, Mixed Greedy/Sampled, Ragged, Bench) ===")
    import os

    def _make_stream_state(lm_m: LanguageModel, dr_m: InklingMTPDrafter, runner_m: NativeDecodeRunner, hist_s: int, seed: int):
        rng = np.random.default_rng(seed)
        tc = lm_m.make_cache()
        dc = dr_m.make_cache()
        runner_m.ensure_cache_ready(tc, m_tokens=8)
        d_m = lm_m.config.hidden_size
        hd_m = dr_m.config.head_dim
        n_kv_d = dr_m.config.num_key_value_heads
        for cl in tc:
            kv, conv = cl.caches[0], cl.caches[1]
            n_kv_t, hd_t = kv.keys.shape[1], kv.keys.shape[3]
            if isinstance(kv, RotatingKVCache):
                fill = min(hist_s, kv.max_size)
                rk = mx.array(rng.normal(0.0, 0.1, size=(1, n_kv_t, fill, hd_t)).astype(np.float32)).astype(mx.bfloat16)
                rv = mx.array(rng.normal(0.0, 0.1, size=(1, n_kv_t, fill, hd_t)).astype(np.float32)).astype(mx.bfloat16)
                kv.keys[:, :, :fill, :] = rk
                kv.values[:, :, :fill, :] = rv
                kv.offset = hist_s
                kv._idx = hist_s if hist_s <= kv.max_size else ((hist_s - 1) % kv.max_size) + 1
            else:
                cap = max(256, ((hist_s + 64 + 255) // 256) * 256)
                rk = mx.array(rng.normal(0.0, 0.1, size=(1, n_kv_t, hist_s, hd_t)).astype(np.float32)).astype(mx.bfloat16)
                rv = mx.array(rng.normal(0.0, 0.1, size=(1, n_kv_t, hist_s, hd_t)).astype(np.float32)).astype(mx.bfloat16)
                nk = mx.zeros((1, n_kv_t, cap, hd_t), dtype=mx.bfloat16)
                nv = mx.zeros((1, n_kv_t, cap, hd_t), dtype=mx.bfloat16)
                nk[:, :, :hist_s, :] = rk
                nv[:, :, :hist_s, :] = rv
                kv.keys, kv.values = nk, nv
                kv.offset = hist_s
            for ci in range(4):
                conv[ci] = mx.array(rng.normal(0.0, 0.05, size=conv[ci].shape).astype(np.float32))
            mx.eval(kv.keys, kv.values, *conv)

        for dc_i in dc:
            is_rot = isinstance(dc_i, RotatingKVCache)
            cap = dc_i.max_size if is_rot else max(256, ((hist_s + 64 + 255) // 256) * 256)
            fill = min(hist_s, cap) if is_rot else hist_s
            dk = mx.zeros((1, n_kv_d, cap, hd_m), dtype=mx.bfloat16)
            dv = mx.zeros((1, n_kv_d, cap, hd_m), dtype=mx.bfloat16)
            dk[:, :, :fill, :] = mx.array(rng.normal(0.0, 0.1, size=(1, n_kv_d, fill, hd_m)).astype(np.float32)).astype(mx.bfloat16)
            dv[:, :, :fill, :] = mx.array(rng.normal(0.0, 0.1, size=(1, n_kv_d, fill, hd_m)).astype(np.float32)).astype(mx.bfloat16)
            mx.eval(dk, dv)
            dc_i.keys, dc_i.values = dk, dv
            dc_i.offset = hist_s
            if is_rot:
                dc_i._idx = hist_s if hist_s <= cap else (hist_s % cap)

        h_pool = mx.array(rng.normal(0.0, 0.15, size=(1, dr_m.config.block_size, len(dr_m.config.target_layer_ids) * d_m)).astype(np.float32)).astype(mx.bfloat16)
        mx.eval(h_pool)
        return tc, dc, h_pool

    lm_p = build_synthetic_model(sliding_window=16, seed=701)
    d = lm_p.config.hidden_size
    n_cap = 4
    rng_rb = np.random.default_rng(703)
    for lyr in lm_p.model.layers:
        lyr.mlp.e_score_correction_bias = mx.array(
            rng_rb.normal(0.0, 0.25, size=lyr.mlp.e_score_correction_bias.shape).astype(np.float32)
        )
    w_lh = mx.array(lm_p.lm_head.weight)
    w_lh[42] = mx.ones((d,), dtype=mx.bfloat16) * 1.5
    lm_p.lm_head.weight = w_lh
    lm_p.model.norm.weight = mx.ones((d,), dtype=mx.bfloat16) * 2.0
    lm_p.model.embed_norm.weight = mx.ones((d,), dtype=mx.bfloat16) * 1.5
    et = mx.array(lm_p.model.embed_tokens.weight)
    et[42] = mx.ones((d,), dtype=mx.bfloat16) * 1.0
    lm_p.model.embed_tokens.weight = et
    mx.eval(lm_p.parameters())

    dr_p = _build_synthetic_drafter(lm_p, sliding_window=16, seed=702)
    fb = mx.array(dr_p.markov_head._fold_bias) * 0.01
    fb[42] = mx.array(0.5, dtype=fb.dtype)
    dr_p.markov_head._fold_bias = fb
    dr_p.norm.weight = mx.ones((d,), dtype=mx.bfloat16) * 2.0
    mx.eval(dr_p.parameters())
    dr_p.bind(lm_p)
    mx.eval(dr_p.parameters())

    for simt_mode, env_val in [(True, "1"), (False, "0")]:
        os.environ["INKLING_MTP_NO_NAX"] = env_val
        r_ref = NativeDecodeRunner(lib, lm_p, max_m=32)
        ka_ref = _attach_mtp_to_runner(lib, r_ref, lm_p, dr_p)
        r_bat = NativeDecodeRunner(lib, lm_p, max_m=32)
        ka_bat = _attach_mtp_to_runner(lib, r_bat, lm_p, dr_p)
        assert ka_ref and ka_bat

        for B_test, hist_list, n_ctx_schedule, k_schedule in [
            (2, [10, 270], [(1, 7), (4, 2), (2, 5), (6, 3), (3, 1), (5, 4), (7, 2), (2, 6)],
                [(3, 7), (5, 4), (6, 3), (4, 5), (7, 3), (3, 6), (4, 4), (5, 3)]),
            (4, [10, 15, 145, 270], [(1, 3, 5, 7), (4, 2, 6, 1), (2, 7, 3, 4), (5, 1, 2, 6),
                                     (3, 5, 1, 2), (6, 4, 7, 3), (2, 1, 4, 5), (7, 3, 6, 2)],
                [(3, 5, 7, 4), (6, 3, 4, 5), (4, 7, 3, 6), (5, 4, 6, 3),
                 (7, 3, 5, 4), (3, 6, 4, 7), (4, 5, 3, 6), (6, 4, 7, 5)]),
        ]:
            ref_states = [_make_stream_state(lm_p, dr_p, r_ref, hist_list[b], 9000 + b * 31 + B_test) for b in range(B_test)]
            bat_states = [_make_stream_state(lm_p, dr_p, r_bat, hist_list[b], 9000 + b * 31 + B_test) for b in range(B_test)]

            anchors_ref = [42 if b == 0 else (7 + b * 13) for b in range(B_test)]
            anchors_bat = list(anchors_ref)
            nctx_ref = list(n_ctx_schedule[0])
            nctx_bat = list(n_ctx_schedule[0])

            max_kv_e, max_cv_e, max_dc_e, max_h_e = 0.0, 0.0, 0.0, 0.0
            min_cos_h = 1.0
            tok_mismatches = 0
            nacc_mismatches = 0
            seen_ragged_nacc = False
            all_seen_nacc = set()

            for cyc in range(8):
                ref_res = []
                for b in range(B_test):
                    tc_r, dc_r, hp_r = ref_states[b]
                    n_c = nctx_ref[b]
                    k_d = k_schedule[cyc][b]
                    in_h = mx.contiguous(hp_r[:, :n_c])
                    out_h = mx.zeros((1, dr_p.config.block_size, n_cap * d), dtype=mx.bfloat16)
                    mx.eval(in_h, out_h)
                    toks_r, nacc_r, _ = _run_unified_mtp_step(
                        lib, r_ref, dr_p, anchors_ref[b], in_h, out_h, n_c, k_d, tc_r, dc_r,
                    )
                    hp_r[:, :nacc_r] = out_h[:, :nacc_r]
                    mx.eval(hp_r)
                    nctx_ref[b] = min(dr_p.config.block_size, nacc_r)
                    anchors_ref[b] = toks_r[-1]
                    ref_res.append((toks_r, nacc_r, out_h[:, :nacc_r]))

                specs = []
                bat_out_bufs = []
                for b in range(B_test):
                    tc_b, dc_b, hp_b = bat_states[b]
                    n_c = nctx_bat[b]
                    k_d = k_schedule[cyc][b]
                    use_alias = ((cyc + b) % 2 == 1)
                    if use_alias:
                        pool_in = mx.zeros((1, dr_p.config.block_size, n_cap * d), dtype=mx.bfloat16)
                        pool_in[:, :n_c] = hp_b[:, :n_c]
                        mx.eval(pool_in)
                        in_h, out_h = pool_in, pool_in
                    else:
                        in_h = mx.contiguous(hp_b[:, :n_c])
                        out_h = mx.zeros((1, dr_p.config.block_size, n_cap * d), dtype=mx.bfloat16)
                        mx.eval(in_h, out_h)
                    bat_out_bufs.append(out_h)
                    specs.append({
                        "anchor_token": anchors_bat[b],
                        "in_hidden": in_h,
                        "out_hidden": out_h,
                        "n_ctx": n_c,
                        "num_draft": k_d,
                        "target_caches": tc_b,
                        "draft_caches": dc_b,
                    })

                bat_res, _ = _run_batched_mtp_step(lib, r_bat, dr_p, specs)
                cyc_naccs = []
                for b in range(B_test):
                    toks_r, nacc_r, next_hr = ref_res[b]
                    toks_b, nacc_b = bat_res[b]
                    cyc_naccs.append(nacc_b)
                    all_seen_nacc.add(nacc_b)
                    if nacc_b != nacc_r:
                        nacc_mismatches += 1
                    if toks_b != toks_r:
                        tok_mismatches += 1
                    tc_b, dc_b, hp_b = bat_states[b]
                    tc_r, dc_r, _ = ref_states[b]
                    next_hb = bat_out_bufs[b][:, :nacc_b]
                    if nacc_b == nacc_r:
                        max_h_e = max(max_h_e, max_abs_diff(next_hr, next_hb))
                        af = next_hr.astype(mx.float32).reshape(-1)
                        bf = next_hb.astype(mx.float32).reshape(-1)
                        cos_v = float((mx.sum(af * bf) / (mx.sqrt(mx.sum(af * af)) * mx.sqrt(mx.sum(bf * bf)) + 1e-12)).item())
                        min_cos_h = min(min_cos_h, cos_v)
                    ek, ec = compare_caches(tc_r, tc_b)
                    ed = _compare_draft_caches(dc_r, dc_b)
                    max_kv_e = max(max_kv_e, ek)
                    max_cv_e = max(max_cv_e, ec)
                    max_dc_e = max(max_dc_e, ed)
                    hp_b[:, :nacc_b] = next_hb
                    mx.eval(hp_b)
                    nctx_bat[b] = min(dr_p.config.block_size, nacc_b)
                    anchors_bat[b] = toks_b[-1]
                if len(set(cyc_naccs)) >= 2:
                    seen_ragged_nacc = True

            tag = "SIMT" if simt_mode else "NAX "
            print(
                f"  [11a {tag}] B={B_test} 8-cyc ragged(S=10..270, n_ctx=1..7, K=3..7, alias): "
                f"kv={max_kv_e:.2e} conv={max_cv_e:.2e} d_kv={max_dc_e:.2e} h={max_h_e:.2e} min_cos={min_cos_h:.6f} "
                f"nacc_miss={nacc_mismatches} tok_miss={tok_mismatches} n_accs={sorted(all_seen_nacc)}"
            )
            assert seen_ragged_nacc, f"Expected ragged n_acc within batch, saw {all_seen_nacc}"
            if simt_mode:
                assert nacc_mismatches == 0 and tok_mismatches == 0, f"SIMT token mismatch: {nacc_mismatches}/{tok_mismatches}"
                assert max_kv_e == 0.0 and max_cv_e <= 1e-6 and max_dc_e == 0.0 and max_h_e == 0.0, (
                    f"SIMT non-zero diff: kv={max_kv_e}, conv={max_cv_e}, dc={max_dc_e}, h={max_h_e}"
                )
            else:
                assert nacc_mismatches == 0 and tok_mismatches == 0 and min_cos_h >= 0.999, (
                    f"NAX parity failed: nacc_miss={nacc_mismatches}, tok_miss={tok_mismatches}, min_cos={min_cos_h}"
                )
        r_ref.close()
        r_bat.close()
    os.environ.pop("INKLING_MTP_NO_NAX", None)

    lm_mix = build_synthetic_model(sliding_window=16, seed=801)
    tc_tmp = lm_mix.make_cache()
    pn0 = lm_mix.model.norm(
        fused.decode_step(lm_mix.model, lm_mix.model.embed(mx.array([[3]], dtype=mx.int32)), tc_tmp)
    )[0, 0]
    u_vec = (pn0 / mx.sqrt(mx.sum(pn0 * pn0))).astype(mx.float32)
    w_pad = mx.zeros((lm_mix.config.vocab_size, d), dtype=mx.float32)
    scales = mx.array([0.42, 0.40, 0.38, 0.36, 0.34, 0.32, 0.30, 0.28], dtype=mx.float32)
    w_pad[:8] = u_vec[None, :] * scales[:, None]
    lm_mix.lm_head.weight = w_pad.astype(mx.bfloat16)
    mx.eval(lm_mix.lm_head.weight)
    dr_mix = _build_synthetic_drafter(lm_mix, sliding_window=16, seed=802)
    dr_mix.norm.weight = (dr_mix.norm.weight * 0.02).astype(mx.bfloat16)
    dr_mix.markov_head.hidden_out.bias = (u_vec * 28.0).astype(mx.bfloat16)
    mx.eval(dr_mix.parameters())
    dr_mix._target = None
    dr_mix.bind(lm_mix)
    mx.eval(dr_mix.parameters())

    r_mix_ref = NativeDecodeRunner(lib, lm_mix, max_m=32)
    ka_mr = _attach_mtp_to_runner(lib, r_mix_ref, lm_mix, dr_mix)
    r_mix_bat = NativeDecodeRunner(lib, lm_mix, max_m=32)
    ka_mb = _attach_mtp_to_runner(lib, r_mix_bat, lm_mix, dr_mix)
    assert ka_mr and ka_mb

    st_mix_ref = [_make_stream_state(lm_mix, dr_mix, r_mix_ref, [5, 17, 260][b], 9500 + b) for b in range(3)]
    st_mix_bat = [_make_stream_state(lm_mix, dr_mix, r_mix_bat, [5, 17, 260][b], 9500 + b) for b in range(3)]
    cfgs_mix = [
        (0.0, 1.0, 1, 1),
        (0.7, 0.95, 2, 2),
        (1.0, 1.0, 3, 1),
    ]
    anc_mr = [3, 4, 5]
    anc_mb = [3, 4, 5]
    mix_kv_e, mix_cv_e, mix_dc_e, mix_h_e = 0.0, 0.0, 0.0, 0.0
    for cyc in range(4):
        ref_o = []
        for b, (t_v, tp_v, nc, kd) in enumerate(cfgs_mix):
            tc_r, dc_r, hp_r = st_mix_ref[b]
            in_h = mx.contiguous(hp_r[:, :nc])
            out_h = mx.zeros((1, dr_mix.config.block_size, n_cap * d), dtype=mx.bfloat16)
            mx.eval(in_h, out_h)
            toks_r, nacc_r, _ = _run_unified_mtp_step(
                lib, r_mix_ref, dr_mix, anc_mr[b], in_h, out_h, nc, kd, tc_r, dc_r,
                temperature=t_v, top_p=tp_v, softcap=30.0, rng_seed=5000 + b * 100, rng_step=cyc,
            )
            anc_mr[b] = toks_r[-1]
            ref_o.append((toks_r, nacc_r, out_h[:, :nacc_r]))

        specs_m, outs_m = [], []
        for b, (t_v, tp_v, nc, kd) in enumerate(cfgs_mix):
            tc_b, dc_b, hp_b = st_mix_bat[b]
            in_h = mx.contiguous(hp_b[:, :nc])
            out_h = mx.zeros((1, dr_mix.config.block_size, n_cap * d), dtype=mx.bfloat16)
            mx.eval(in_h, out_h)
            outs_m.append(out_h)
            specs_m.append({
                "anchor_token": anc_mb[b], "in_hidden": in_h, "out_hidden": out_h,
                "n_ctx": nc, "num_draft": kd, "target_caches": tc_b, "draft_caches": dc_b,
                "temperature": t_v, "top_p": tp_v, "softcap": 30.0,
                "rng_seed": 5000 + b * 100, "rng_step": cyc,
            })
        bat_o, _ = _run_batched_mtp_step(lib, r_mix_bat, dr_mix, specs_m)
        for b in range(3):
            toks_r, nacc_r, nhr = ref_o[b]
            toks_b, nacc_b = bat_o[b]
            assert nacc_b == nacc_r and toks_b == toks_r, f"Mixed cyc={cyc} b={b}: {toks_b} vs {toks_r}"
            ek, ec = compare_caches(st_mix_ref[b][0], st_mix_bat[b][0])
            ed = _compare_draft_caches(st_mix_ref[b][1], st_mix_bat[b][1])
            eh = max_abs_diff(nhr, outs_m[b][:, :nacc_b])
            mix_kv_e, mix_cv_e, mix_dc_e, mix_h_e = max(mix_kv_e, ek), max(mix_cv_e, ec), max(mix_dc_e, ed), max(mix_h_e, eh)
            anc_mb[b] = toks_b[-1]
    assert mix_kv_e == 0.0 and mix_cv_e == 0.0 and mix_dc_e == 0.0 and mix_h_e == 0.0

    def _zero_reset(tc: list[CacheList], dc: list):
        ka_z: list[Any] = []
        for cl in tc:
            kv, conv = cl.caches[0], cl.caches[1]
            kv.offset = 0
            if isinstance(kv, RotatingKVCache):
                kv._idx = 0
            for ci in range(4):
                p_c, n_c = arr_ptr(conv[ci], ka_z)
                ctypes.memset(p_c, 0, n_c)
        for dcl in dc:
            dcl.offset = 0
            if isinstance(dcl, RotatingKVCache):
                dcl._idx = 0

    g_tc0 = lm_mix.make_cache(); g_dc0 = dr_mix.make_cache()
    s1_tc = lm_mix.make_cache(); s1_dc = dr_mix.make_cache()
    s2_tc = lm_mix.make_cache(); s2_dc = dr_mix.make_cache()
    g_tc3 = lm_mix.make_cache(); g_dc3 = dr_mix.make_cache()
    h_zero_in = (mx.random.normal((1, 1, n_cap * d)) * 0.4).astype(mx.bfloat16)
    h_outs = [mx.zeros((1, dr_mix.config.block_size, n_cap * d), dtype=mx.bfloat16) for _ in range(4)]
    mx.eval(h_zero_in, *h_outs)

    g0_ref_tc = lm_mix.make_cache(); g0_ref_dc = dr_mix.make_cache()
    g0_ref_out = mx.zeros((1, dr_mix.config.block_size, n_cap * d), dtype=mx.bfloat16)
    mx.eval(g0_ref_out)
    g0_ref_toks, g0_ref_nacc, _ = _run_unified_mtp_step(
        lib, r_mix_ref, dr_mix, 3, h_zero_in, g0_ref_out, 1, 3, g0_ref_tc, g0_ref_dc, temperature=0.0,
    )
    _zero_reset(s1_tc, s1_dc)
    _run_unified_mtp_step(
        lib, r_mix_ref, dr_mix, 3, h_zero_in, g0_ref_out, 1, 2, s1_tc, s1_dc,
        temperature=0.7, top_p=0.95, softcap=30.0, rng_seed=42, rng_step=0,
    )
    v_unpad_m = lm_mix.config.unpadded_vocab_size
    q0_ref = np.zeros((2, v_unpad_m), dtype=np.float32)
    p0_ref = np.zeros((3, v_unpad_m), dtype=np.float32)
    assert lib.inkling_ctx_mtp_debug_read(r_mix_ref.ctx, 0, q0_ref.ctypes.data, q0_ref.nbytes) == 0
    assert lib.inkling_ctx_mtp_debug_read(r_mix_ref.ctx, 1, p0_ref.ctypes.data, p0_ref.nbytes) == 0
    ref_q0_m = q0_ref[0].astype(np.float64)
    ref_p0_m = p0_ref[0].astype(np.float64)

    n_mix_seeds = 250
    emp_s1 = np.zeros((v_unpad_m,), dtype=np.float64)
    acc0_s1 = 0
    full_s1 = 0
    greedy_match_cnt = 0
    for s_idx in range(n_mix_seeds):
        for tc_x, dc_x in [(g_tc0, g_dc0), (s1_tc, s1_dc), (s2_tc, s2_dc), (g_tc3, g_dc3)]:
            _zero_reset(tc_x, dc_x)
        specs_b4 = [
            {"anchor_token": 3, "in_hidden": h_zero_in, "out_hidden": h_outs[0], "n_ctx": 1, "num_draft": 3,
             "target_caches": g_tc0, "draft_caches": g_dc0, "temperature": 0.0, "top_p": 1.0, "softcap": 30.0},
            {"anchor_token": 3, "in_hidden": h_zero_in, "out_hidden": h_outs[1], "n_ctx": 1, "num_draft": 2,
             "target_caches": s1_tc, "draft_caches": s1_dc, "temperature": 0.7, "top_p": 0.95, "softcap": 30.0, "rng_seed": 7000 + s_idx, "rng_step": 0},
            {"anchor_token": 3, "in_hidden": h_zero_in, "out_hidden": h_outs[2], "n_ctx": 1, "num_draft": 3,
             "target_caches": s2_tc, "draft_caches": s2_dc, "temperature": 1.0, "top_p": 1.0, "softcap": 30.0, "rng_seed": 8000 + s_idx, "rng_step": 0},
            {"anchor_token": 3, "in_hidden": h_zero_in, "out_hidden": h_outs[3], "n_ctx": 1, "num_draft": 2,
             "target_caches": g_tc3, "draft_caches": g_dc3, "temperature": 0.0, "top_p": 1.0, "softcap": 30.0},
        ]
        res_b4, _ = _run_batched_mtp_step(lib, r_mix_bat, dr_mix, specs_b4)
        if res_b4[0] == (g0_ref_toks, g0_ref_nacc):
            greedy_match_cnt += 1
        assert 1 <= res_b4[2][1] <= 4
        t1, na1 = res_b4[1]
        assert ref_p0_m[t1[0]] > 0.0
        emp_s1[t1[0]] += 1.0
        if na1 >= 2:
            acc0_s1 += 1
        if na1 == 3:
            full_s1 += 1

    emp_p0_s1 = emp_s1 / float(n_mix_seeds)
    tvd_p_m = 0.5 * float(np.sum(np.abs(emp_p0_s1 - ref_p0_m)))
    tvd_q_m = 0.5 * float(np.sum(np.abs(emp_p0_s1 - ref_q0_m)))
    emp_acc_m = acc0_s1 / float(n_mix_seeds)
    theo_acc_m = float(np.sum(np.minimum(ref_p0_m, ref_q0_m)))
    assert greedy_match_cnt == n_mix_seeds, f"Mixed greedy match {greedy_match_cnt}/{n_mix_seeds}"
    assert full_s1 >= 1 and tvd_p_m <= 0.11, f"Mixed TVD={tvd_p_m:.4f}, full={full_s1}"
    assert abs(emp_acc_m - theo_acc_m) <= 0.09, f"Mixed accept={emp_acc_m:.3f} vs {theo_acc_m:.3f}"
    print(
        f"  [11b Mixed] B=3 parity=0.00e+00 (4 cyc) | B=4 NAX(M_v=14, N={n_mix_seeds}): "
        f"greedy_exact={greedy_match_cnt}/{n_mix_seeds}, TVD(emp,p0)={tvd_p_m:.4f} (tvd_q={tvd_q_m:.4f}), "
        f"acc0={emp_acc_m:.3f} vs theo={theo_acc_m:.3f} [PASS]"
    )
    r_mix_ref.close()
    r_mix_bat.close()

    layer_types48 = [("full_attention" if (i + 1) % 6 == 0 else "hybrid_sliding") for i in range(24)]
    lm48 = build_synthetic_model(
        n_layers=24, loop_start=6, loop_end=18, loop_passes=3, block_size=6,
        d=256, inter=128, n_experts=16, top_k=4, n_heads=8, n_kv=2, head_dim=32,
        sliding_window=32, rel_extent=32, vocab_size=512, unpadded_vocab_size=500,
        layer_types=layer_types48, mxfp8_layers={20, 21, 22, 23}, seed=99,
    )
    dr48 = _build_synthetic_drafter(
        lm48, num_layers=3, block_size=8, markov_rank=64,
        target_layer_ids=(5, 23, 41, 48), local_layer_ids=(0, 2), sliding_window=32, seed=199,
    )
    r48 = NativeDecodeRunner(lib, lm48, max_m=32)
    ka48 = _attach_mtp_to_runner(lib, r48, lm48, dr48)
    assert ka48

    stream_pools = [(lm48.make_cache(), dr48.make_cache()) for _ in range(4)]
    h48_ins = [(mx.random.normal((1, 4, 4 * d)) * 0.1).astype(mx.bfloat16) for _ in range(4)]
    h48_outs = [mx.zeros((1, 8, 4 * d), dtype=mx.bfloat16) for _ in range(4)]
    mx.eval(*h48_ins, *h48_outs)

    def _reset_pool(cnt: int):
        for b in range(cnt):
            _zero_reset(*stream_pools[b])

    def _bench_pair(B_val: int, temp_val: float, topp_val: float):
        med = lambda xs: float(np.median(xs))
        bat_gpus, bat_cwalls, bat_pywalls = [], [], []
        seq_gpus, seq_cwalls, seq_pywalls = [], [], []
        specs = [
            {
                "anchor_token": 3 + b, "in_hidden": h48_ins[b], "out_hidden": h48_outs[b],
                "n_ctx": 4, "num_draft": 5, "target_caches": stream_pools[b][0], "draft_caches": stream_pools[b][1],
                "temperature": temp_val, "top_p": topp_val, "softcap": 30.0, "rng_seed": 7 + b, "rng_step": 0,
            }
            for b in range(B_val)
        ]
        for it in range(16):
            _reset_pool(B_val)
            t0 = time.perf_counter()
            _, st_b = _run_batched_mtp_step(lib, r48, dr48, specs)
            t1 = time.perf_counter()
            if it >= 3:
                bat_gpus.append(st_b.gpu_ms)
                bat_cwalls.append(st_b.wall_ms)
                bat_pywalls.append((t1 - t0) * 1000.0)

            _reset_pool(B_val)
            t2 = time.perf_counter()
            sg, sc = 0.0, 0.0
            for b in range(B_val):
                _, _, st_s = _run_unified_mtp_step(
                    lib, r48, dr48, 3 + b, h48_ins[b], h48_outs[b], 4, 5,
                    stream_pools[b][0], stream_pools[b][1],
                    temperature=temp_val, top_p=topp_val, softcap=30.0, rng_seed=7 + b, rng_step=0,
                )
                sg += st_s.gpu_ms
                sc += st_s.wall_ms
            t3 = time.perf_counter()
            if it >= 3:
                seq_gpus.append(sg)
                seq_cwalls.append(sc)
                seq_pywalls.append((t3 - t2) * 1000.0)
        return med(bat_gpus), med(bat_cwalls), med(bat_pywalls), med(seq_gpus), med(seq_cwalls), med(seq_pywalls)

    print("  [11c] 48-visit D=256 K=5 Batched 1-CB vs B * 1-CB Single-Stream:")
    for B_val in (2, 4):
        for label, tv, tpv in [("Greedy ", 0.0, 1.0), ("Sampled", 0.7, 0.95)]:
            bg, bc, bp, sg, sc, sp = _bench_pair(B_val, tv, tpv)
            print(
                f"    B={B_val} {label}: Batched gpu={bg:.3f}ms C-wall={bc:.3f}ms Py+C={bp:.3f}ms "
                f"| {B_val}x1CB gpu={sg:.3f}ms C-wall={sc:.3f}ms Py+C={sp:.3f}ms "
                f"| speedup gpu={sg/bg:.2f}x C-wall={sc/bc:.2f}x Py+C={sp/bp:.2f}x"
            )
    r48.close()
    print("  Test 11 Batched 1-CB MTP [PASS]")


def main() -> None:
    lib = load_lib()

    test_unpatched_fused_parity(lib)

    lm = build_synthetic_model()

    def _routed_experts_fast_any_e(x: mx.array, inds: mx.array, moe) -> mx.array:
        m = x.shape[0]
        mode = getattr(moe.w13_proj, "mode", None)
        if (
            mode in ("nvfp4", "mxfp8")
            and x.dtype == mx.bfloat16
            and x.shape[-1] % 256 == 0
            and moe.down_proj.input_dims % 128 == 0
            and moe.num_experts <= 128
        ):
            if m == 1:
                return moe_kernels.moe_m1_experts(x, inds, moe)
            if m <= 64:
                return moe_kernels.moe_grouped_experts(x, inds, moe, mt=4 if m <= 8 else 8)
        return moe.routed_experts(x, inds)

    fused.routed_experts_fast = _routed_experts_fast_any_e
    fused._rms = lambda x: mx.fast.rms_norm(x, None, 1e-6)

    test_m1_decode_parity(lib, lm)
    test_m6_verify_parity(lib, lm)
    test_post_rotation_rollback(lib, lm)
    bench_48_visit_dispatch(lib)
    test_ink_geometry_tg1024_hd128(lib)
    bench_real_dims_interleaved(lib)
    test_multi_iter_verify_commit_verify(lib, lm)
    bench_skewed_moe_and_lm_head(lib)
    test_batched_decode_parity_and_bench(lib)
    test_unified_mtp_step(lib, lm)
    test_batched_mtp_step(lib, lm)
    print("\nALL TESTS PASSED!")


if __name__ == "__main__":
    main()
