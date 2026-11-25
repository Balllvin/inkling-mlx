"""Inkling 2 Mini (3-pass looped hybrid attention + mixed NVFP4/MXFP8 MoE) MLX model."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchRotatingKVCache,
    CacheList,
    KVCache,
    RotatingKVCache,
)

from .config import ModelConfig, TextConfig
from .kernels import (
    bf16_sigmoid_mul,
    decode_attention,
    dynamic_short_convolution,
    head_argmax,
    head_logits,
    prefill_qkvr_norm,
    relative_attention,
    relative_attention_nax,
    relative_attention_nax_native,
    short_convolution,
    swiglu_interleaved,
)

_DECODE_KERNEL_MIN_KEYS = 4096


class GrowingKVCache(KVCache):
    """Global KVCache with 2048-token growth granularity to minimize 256k reallocations."""

    step = 2048

    @classmethod
    def merge(_, caches):
        return GrowingBatchKVCache.merge(caches)


class GrowingBatchKVCache(BatchKVCache):
    """Batched global KVCache preserving 2048-token allocation steps up to 256k."""

    step = 2048

    def extract(self, idx):
        cache = GrowingKVCache()
        batch_size = self.offset.size
        if idx < 0:
            idx += batch_size
        if idx < 0 or idx >= batch_size:
            raise IndexError("GrowingBatchKVCache row index out of range")
        if self.keys is None:
            return cache
        padding = self.left_padding[idx].item()
        cache.keys = mx.contiguous(self.keys[idx : idx + 1, :, padding : self._idx])
        cache.values = mx.contiguous(self.values[idx : idx + 1, :, padding : self._idx])
        cache.offset = cache.keys.shape[2]
        return cache


_DECODE_CACHES = (
    KVCache,
    RotatingKVCache,
    BatchKVCache,
    BatchRotatingKVCache,
    GrowingKVCache,
    GrowingBatchKVCache,
)


def _patch_arrays_cache_advance() -> None:
    """Keep ArraysCache padding metadata evaluated so batched decode never builds lazy chains."""
    if getattr(ArraysCache.advance, "_inkling_patched", False):
        return
    original = ArraysCache.advance

    def advance(self, n):
        original(self, n)
        for name in ("lengths", "left_padding"):
            v = getattr(self, name, None)
            if isinstance(v, mx.array):
                mx.async_eval(v)

    advance._inkling_patched = True
    ArraysCache.advance = advance


_patch_arrays_cache_advance()


def _split_gate_up(v: mx.array) -> tuple[mx.array, mx.array]:
    *lead, two_i, hidden = v.shape
    w = v.reshape(*lead, two_i // 2, 2, hidden)
    return mx.contiguous(w[..., 0, :]), mx.contiguous(w[..., 1, :])


def _to_mlx_conv(v: mx.array) -> mx.array:
    if v.ndim == 3 and v.shape[-1] == 1:
        return v
    return v.transpose(0, 2, 1)


@mx.compile
def l2_clip(x: mx.array, threshold: float, scale: Optional[mx.array] = None) -> mx.array:
    dtype = x.dtype
    x = x.astype(mx.float32)
    if scale is not None:
        x = x * scale.astype(mx.float32)
    norm = mx.sqrt(mx.sum(mx.square(x), axis=-1, keepdims=True))
    return (x * mx.minimum(threshold / (norm + 1e-6), 1.0)).astype(dtype)


@mx.compile
def _softcapped_swiglu_interleaved(gu: mx.array, gate_cap: float, up_cap: float) -> mx.array:
    dtype = gu.dtype
    gate = gu[..., 0::2].astype(mx.float32)
    up = gu[..., 1::2].astype(mx.float32)
    return (
        mx.tanh(gate / gate_cap)
        * gate_cap
        * mx.sigmoid(gate)
        * mx.tanh(up / up_cap)
        * up_cap
    ).astype(dtype)


@mx.compile
def _route(x: mx.array, weight: mx.array, bias: mx.array, top_k: int, route_scale: float):
    scores = mx.sigmoid(x.astype(mx.float32) @ weight.T)
    indices = mx.argpartition(-(scores + bias), top_k - 1, axis=-1)[..., :top_k]
    indices = mx.stop_gradient(indices)
    weights = mx.take_along_axis(scores, indices, axis=-1)
    weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20) * route_scale
    return indices, weights


@mx.compile
def _combine_routes(out: mx.array, weights: mx.array) -> mx.array:
    return (out.astype(mx.float32) * weights[..., None]).sum(axis=-2).astype(out.dtype)


@mx.compile
def _attention_residual(states: mx.array, query: mx.array, normalize_values: bool) -> mx.array:
    states = states.astype(mx.float32)
    keys = mx.fast.rms_norm(states, None, 1e-6)
    scores = (keys @ query.reshape(-1, states.shape[-1]).T) * states.shape[-1] ** -0.5
    weights = mx.softmax(scores, axis=-2).swapaxes(-1, -2)
    return weights @ (keys if normalize_values else states)


@mx.compile
def _message_gate(
    hidden: mx.array,
    message: mx.array,
    down: mx.array,
    forget: mx.array,
    gain: mx.array,
    eps: float,
) -> mx.array:
    hidden = hidden.astype(mx.float32)
    message = message.astype(mx.float32)
    gate_input = mx.concatenate(
        [mx.fast.rms_norm(hidden, None, eps), mx.fast.rms_norm(message, None, eps)],
        axis=-1,
    )
    gate_input = (gate_input.astype(down.dtype) @ down.T).astype(mx.float32)
    return (1.0 - mx.tanh(gate_input @ forget.T)) * hidden + mx.tanh(gate_input @ gain.T) * message


def _cache_padding_mask(cache, length: int) -> Optional[mx.array]:
    if cache is None:
        return None
    positions = mx.arange(length)[None, :]
    mask = None
    left_padding = getattr(cache, "left_padding", None)
    if left_padding is not None:
        mask = positions >= left_padding[:, None]
    lengths = getattr(cache, "lengths", None)
    if lengths is not None:
        length_mask = positions < lengths[:, None]
        mask = length_mask if mask is None else mask & length_mask
    return mask


def _next_conv_state(state: mx.array, inputs: mx.array, mask: Optional[mx.array]) -> mx.array:
    state_size = state.shape[1]
    if state_size == 0:
        return state
    if mask is None:
        if inputs.shape[1] >= state_size:
            return inputs[:, -state_size:].astype(mx.float32)
        return mx.concatenate([state, inputs.astype(mx.float32)], axis=1)[:, -state_size:]
    xf = inputs.astype(mx.float32)
    valid_count = mx.sum(mask, axis=1).astype(mx.int32)
    valid_start = mx.argmax(mask.astype(mx.int32), axis=1)
    logical = mx.arange(state_size)[None, :] + valid_count[:, None] - state_size
    state_indices = state_size + logical
    input_indices = state_size + valid_start[:, None] + mx.maximum(logical, 0)
    indices = mx.where(logical >= 0, input_indices, state_indices)
    combined = mx.concatenate([state, xf], axis=1)
    indices = mx.broadcast_to(indices[..., None], (*indices.shape, inputs.shape[-1]))
    return mx.take_along_axis(combined, indices, axis=1)


class ShortConvolution(nn.Module):
    def __init__(self, channels: int, kernel_size: int, conv_idx: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_idx = conv_idx
        self.conv = nn.Conv1d(channels, channels, kernel_size, groups=channels, bias=False)

    def __call__(self, x: mx.array, cache=None, mask: Optional[mx.array] = None) -> mx.array:
        state = cache[self.conv_idx] if cache is not None else None
        if state is None:
            state = mx.zeros((x.shape[0], self.kernel_size - 1, x.shape[-1]), dtype=mx.float32)
        if (
            x.dtype == mx.bfloat16
            and self.kernel_size == 4
            and x.shape[-1] % 64 == 0
            and mx.default_device() == mx.gpu
        ):
            if cache is not None:
                cache[self.conv_idx] = _next_conv_state(state, x, mask)
            return short_convolution(state, self.conv.weight, x, mask, self.kernel_size)
        xf = x.astype(mx.float32)
        inputs = xf if mask is None else mx.where(mask[..., None], xf, 0)
        context = mx.concatenate([state, inputs], axis=1)
        if cache is not None:
            cache[self.conv_idx] = _next_conv_state(state, inputs, mask)
        out = mx.conv1d(context, self.conv.weight.astype(mx.float32), groups=x.shape[-1])
        return ((out + xf) / math.sqrt(self.kernel_size + 1)).astype(x.dtype)


class DynamicShortConvolution(nn.Module):
    def __init__(self, config: TextConfig, channels: int, num_heads: int, conv_idx: int):
        super().__init__()
        self.kernel_size = config.sconv_kernel_size
        self.num_heads = num_heads
        self.conv_idx = conv_idx
        self.static_weight = mx.zeros((self.kernel_size, channels))
        self.weight_proj = nn.Linear(config.hidden_size, num_heads * self.kernel_size, bias=False)

    def __call__(
        self,
        x: mx.array,
        dynamic_input: mx.array,
        cache=None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        b, length, channels = x.shape
        k = self.kernel_size
        state = cache[self.conv_idx] if cache is not None else None
        if state is None:
            state = mx.zeros((b, k - 1, channels), mx.float32)
        if (
            x.dtype == mx.bfloat16
            and k == 4
            and channels % self.num_heads == 0
            and mx.default_device() == mx.gpu
        ):
            if cache is not None:
                cache[self.conv_idx] = _next_conv_state(state, x, mask)
            dynamic = (
                self.weight_proj(dynamic_input)
                .astype(mx.float32)
                .reshape(b, length, self.num_heads, k)
            )
            return dynamic_short_convolution(
                state,
                self.static_weight,
                dynamic,
                x,
                mask,
                k,
                channels // self.num_heads,
            )
        xf = x.astype(mx.float32)
        inputs = xf if mask is None else mx.where(mask[..., None], xf, 0)
        context = mx.concatenate([state, inputs], axis=1)
        if cache is not None:
            cache[self.conv_idx] = _next_conv_state(state, inputs, mask)
        dynamic = (
            self.weight_proj(dynamic_input)
            .astype(mx.float32)
            .reshape(b, length, self.num_heads, 1, k)
        )
        weights = self.static_weight.T.reshape(self.num_heads, -1, k) + dynamic
        out = mx.zeros_like(xf).reshape(b, length, self.num_heads, -1)
        for lag in range(k):
            values = context[:, k - 1 - lag : k - 1 - lag + length].reshape(out.shape)
            out = out + values * weights[..., lag]
        return ((out.reshape(x.shape) + xf) / math.sqrt(2 * k + 1)).astype(x.dtype)


class Attention(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.is_sliding = config.layer_is_sliding(layer_idx)
        self.head_dim = config.swa_head_dim if self.is_sliding else config.head_dim
        self.n_heads = (
            config.swa_num_attention_heads if self.is_sliding else config.num_attention_heads
        )
        self.n_kv = (
            config.swa_num_key_value_heads if self.is_sliding else config.num_key_value_heads
        )
        self.sliding = config.sliding_window_size if self.is_sliding else 0
        self.rel_extent = self.sliding or config.rel_extent
        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.r_proj = nn.Linear(config.hidden_size, self.n_heads * config.d_rel, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.sink_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.q_scale = mx.ones((self.n_heads, self.head_dim))
        self.rel_scale = mx.ones((self.n_heads, config.d_rel))
        self.rel_proj = mx.zeros((config.d_rel, self.rel_extent))
        channels = self.n_kv * self.head_dim
        for idx, name in enumerate(("k_sconv", "v_sconv")):
            self[name] = (
                ShortConvolution(channels, config.sconv_kernel_size, idx)
                if self.is_sliding
                else DynamicShortConvolution(config, channels, self.n_kv, idx)
            )

    def __call__(
        self,
        x: mx.array,
        key_input: Optional[mx.array] = None,
        value_input: Optional[mx.array] = None,
        cache=None,
        conv_mask: Optional[mx.array] = None,
        fused_const=None,
    ) -> mx.array:
        b, length, _ = x.shape
        cfg = self.config
        kv, conv = cache if cache is not None else (None, None)
        shared_kv = key_input is None and value_input is None
        key_input = x if key_input is None else key_input
        value_input = x if value_input is None else value_input

        if fused_const is not None and shared_kv:
            proj = x @ fused_const.stack.T
            a0, a1, a2, a3, a4 = fused_const.at
            q_raw = proj[..., a0:a1]
            r_raw = proj[..., a1:a2]
            gate_raw = proj[..., a2:a3]
            k = proj[..., a3:a4]
            v = proj[..., a4:]
        elif fused_const is not None:
            proj = x @ fused_const.head.T
            a0, a1, a2, a3 = fused_const.at[:4]
            q_raw = proj[..., a0:a1]
            r_raw = proj[..., a1:a2]
            gate_raw = proj[..., a2:a3]
            k = self.k_proj(key_input)
            v = self.v_proj(value_input)
        else:
            proj = None
            a0 = a1 = a2 = 0
            q_raw = self.q_proj(x)
            r_raw = self.r_proj(x)
            gate_raw = self.gate_proj(x)
            k = self.k_proj(key_input)
            v = self.v_proj(value_input)

        if self.is_sliding:
            k = self.k_sconv(k, conv, conv_mask)
            v = self.v_sconv(v, conv, conv_mask)
        else:
            k = self.k_sconv(k, key_input, conv, conv_mask)
            v = self.v_sconv(v, value_input, conv, conv_mask)

        tau = None
        if not self.is_sliding and cfg.log_scaling_n_floor is not None:
            offset = kv.offset if kv is not None else 0
            positions = mx.arange(length) + mx.array(offset)[..., None] + 1
            tau = 1.0 + cfg.log_scaling_alpha * mx.log(
                mx.maximum(positions / cfg.log_scaling_n_floor, 1.0)
            )
            if tau.ndim == 1 and b > 1:
                tau = mx.broadcast_to(tau[None, :], (b, length))

        use_fused_prep = (
            proj is not None
            and x.dtype == mx.bfloat16
            and mx.default_device() == mx.gpu
            and self.head_dim == 128
            and cfg.d_rel == 16
        )
        if use_fused_prep:
            q, k, v, r = prefill_qkvr_norm(
                proj,
                a0,
                a1,
                k,
                v,
                self.q_scale,
                self.rel_scale,
                tau,
                hq=self.n_heads,
                hkv=self.n_kv,
                hd=self.head_dim,
                dr=cfg.d_rel,
                qk_thr=cfg.qk_l2_clip_threshold,
                rel_thr=cfg.rel_l2_clip_threshold,
                eps=cfg.rms_norm_eps,
                use_vnorm=cfg.use_value_norm_no_gamma,
            )
        else:
            q = l2_clip(
                q_raw.reshape(b, length, self.n_heads, self.head_dim),
                cfg.qk_l2_clip_threshold,
                self.q_scale,
            )
            k = l2_clip(k.reshape(b, length, self.n_kv, self.head_dim), cfg.qk_l2_clip_threshold)
            v = v.reshape(b, length, self.n_kv, self.head_dim)
            if cfg.use_value_norm_no_gamma:
                v = mx.fast.rms_norm(v.astype(mx.float32), None, cfg.rms_norm_eps).astype(x.dtype)
            r = l2_clip(
                r_raw.reshape(b, length, self.n_heads, cfg.d_rel),
                cfg.rel_l2_clip_threshold,
                self.rel_scale,
            )
            if tau is not None:
                q = (q.astype(mx.float32) * tau[..., None, None]).astype(x.dtype)
                r = (r.astype(mx.float32) * tau[..., None, None]).astype(x.dtype)
            q, k, v = (a.transpose(0, 2, 1, 3) for a in (q, k, v))

        gate_fused = False
        if (
            b == 1
            and length == 1
            and type(kv) in _DECODE_CACHES
        ):
            out = self._decode_attention_b1(x, q, k, v, r, kv)
        else:
            out, gate_fused = self._flash_attention(
                x,
                q,
                k,
                v,
                r,
                kv,
                fused_const=fused_const,
                gate_proj=proj if use_fused_prep else None,
                gate_off=a2 if use_fused_prep else 0,
            )
        out = out.transpose(0, 2, 1, 3).reshape(b, length, -1)
        if not gate_fused:
            if use_fused_prep:
                out = bf16_sigmoid_mul(out, proj, a2, proj.shape[-1])
            else:
                out = (
                    out.astype(mx.float32) * mx.sigmoid(gate_raw.astype(mx.float32))
                ).astype(x.dtype)
        return self.o_proj(out)

    def _decode_attention_b1(self, x, q, k, v, r, kv):
        batched = isinstance(kv, (BatchKVCache, BatchRotatingKVCache))
        rotating = isinstance(kv, (RotatingKVCache, BatchRotatingKVCache))
        k, v = kv.update_and_fetch(k, v)
        size = k.shape[2]
        rel = r[0, 0] @ self.rel_proj.astype(x.dtype)
        sinks = self.sink_proj(x)[0, 0]
        start = kv.left_padding[:1] if batched else None
        if (
            not rotating
            and size >= _DECODE_KERNEL_MIN_KEYS
            and q.dtype == k.dtype == v.dtype == mx.bfloat16
            and self.head_dim % 32 == 0
        ):
            return decode_attention(
                q,
                k,
                v,
                rel,
                sinks,
                scale=self.head_dim**-0.5,
                key_start=None if start is None else mx.maximum(start, 0),
            )
        slots = mx.arange(size)
        if rotating:
            slots = (slots - kv._idx) % size
        distance = (size - 1) - slots
        bias = mx.take(rel, mx.minimum(distance, self.rel_extent - 1), axis=-1)
        if size > self.rel_extent:
            bias = mx.where(distance < self.rel_extent, bias, mx.array(0, bias.dtype))
        if start is not None:
            bias = mx.where(slots >= start, bias, mx.array(-mx.inf, bias.dtype))
        return mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.head_dim**-0.5,
            mask=bias[None, :, None, :],
            sinks=sinks.astype(q.dtype),
        )

    def _flash_attention(
        self,
        x,
        q,
        k,
        v,
        r,
        kv,
        fused_const=None,
        gate_proj=None,
        gate_off: int = 0,
    ):
        b, length = x.shape[:2]
        if kv is not None:
            k, v = kv.update_and_fetch(k, v)
            if isinstance(kv, RotatingKVCache):
                if kv.offset > kv.max_size:
                    k, v = kv._temporal_order(k), kv._temporal_order(v)
            elif isinstance(kv, BatchRotatingKVCache) and kv.rotated:
                k, v = mx.roll(k, -kv._idx, axis=2), mx.roll(v, -kv._idx, axis=2)
        padding = getattr(kv, "left_padding", None)
        sinks = self.sink_proj(x)
        key_start = (
            mx.zeros((b,), mx.int32)
            if padding is None
            else mx.maximum(padding, 0).astype(mx.int32)
        )
        if (
            q.dtype == mx.bfloat16
            and r.shape[-1] == 16
            and mx.default_device() == mx.gpu
        ):
            if fused_const is not None:
                rel_proj_t = fused_const.rel_proj_t
                rel_proj_bf16 = fused_const.rel_proj_bf16
            else:
                rel_proj_t = getattr(self, "_rel_proj_t", None)
                rel_proj_bf16 = getattr(self, "_rel_proj_bf16", None)
                if rel_proj_t is None:
                    rel_proj_t = mx.contiguous(self.rel_proj.T.astype(x.dtype))
                    rel_proj_bf16 = mx.contiguous(self.rel_proj.astype(x.dtype))
                    object.__setattr__(self, "_rel_proj_t", rel_proj_t)
                    object.__setattr__(self, "_rel_proj_bf16", rel_proj_bf16)
            if self.head_dim == 128 and (b * length >= 1024 or (not self.is_sliding and k.shape[2] >= 2048)):
                native_out = relative_attention_nax_native(
                    q,
                    k,
                    v,
                    r,
                    rel_proj_t,
                    sinks,
                    scale=self.head_dim**-0.5,
                    window=self.sliding,
                    key_start=key_start,
                    gate_proj=gate_proj,
                    gate_off=gate_off,
                    gate_stride=gate_proj.shape[-1] if gate_proj is not None else 0,
                )
                if native_out is not None:
                    return native_out.transpose(0, 2, 1, 3), (gate_proj is not None)
            out = relative_attention_nax(
                q,
                k,
                v,
                r,
                rel_proj_t,
                sinks,
                scale=self.head_dim**-0.5,
                window=self.sliding,
                key_start=key_start,
                rel_proj=rel_proj_bf16,
            )
            return out.transpose(0, 2, 1, 3), False
        out = relative_attention(
            q,
            k,
            v,
            r @ self.rel_proj.astype(x.dtype),
            sinks,
            scale=self.head_dim**-0.5,
            window=self.sliding,
            key_start=key_start,
        )
        return out.transpose(0, 2, 1, 3), False


class QuantizedExpertLinear(nn.Module):
    """Switch linear holding packed NVFP4 (with outer scale) or MXFP8 experts."""

    def __init__(self, num_experts: int, output_dims: int, input_dims: int, mode: str):
        super().__init__()
        self.num_experts = num_experts
        self.output_dims = output_dims
        self.input_dims = input_dims
        self.mode = mode
        if mode == "nvfp4":
            self.group_size = 16
            self.bits = 4
            w_cols = input_dims // 8
            s_cols = input_dims // 16
            self.global_scale = mx.ones((num_experts,), dtype=mx.float32) * 2688.0
        elif mode == "mxfp8":
            self.group_size = 32
            self.bits = 8
            w_cols = input_dims // 4
            s_cols = input_dims // 32
        else:
            raise ValueError(f"Unsupported expert quant mode: {mode}")
        self.weight = mx.zeros((num_experts, output_dims, w_cols), dtype=mx.uint32)
        self.scales = mx.zeros((num_experts, output_dims, s_cols), dtype=mx.uint8)
        self.freeze()

    def __call__(
        self, x: mx.array, indices: mx.array, *, sorted_indices: bool = False
    ) -> mx.array:
        gs = self.global_scale if self.mode == "nvfp4" else None
        return mx.gather_qmm(
            x,
            self.weight,
            self.scales,
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            global_scale=gs,
            sorted_indices=sorted_indices,
        )


def _gather_sort(x: mx.array, indices: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    n_slots = indices.shape[-1]
    flat_idx = indices.flatten()
    order = mx.argsort(flat_idx)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // n_slots], flat_idx[order], inv_order


def _scatter_unsort(x: mx.array, inv_order: mx.array, shape: tuple[int, ...]) -> mx.array:
    return mx.unflatten(x[inv_order], 0, shape)


class SparseMoE(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.route_scale = config.route_scale
        self.gate_cap = config.mlp_gate_softcap
        self.up_cap = config.mlp_up_softcap
        self.gate_weight = mx.zeros((config.n_routed_experts, config.hidden_size), dtype=mx.float32)
        self.e_score_correction_bias = mx.zeros((config.n_routed_experts,), dtype=mx.float32)
        mode = "nvfp4" if layer_idx < 20 else "mxfp8"
        e, d, inter = config.n_routed_experts, config.hidden_size, config.intermediate_size
        self.w13_proj = QuantizedExpertLinear(e, 2 * inter, d, mode)
        self.down_proj = QuantizedExpertLinear(e, d, inter, mode)

    def routed_experts(self, x: mx.array, indices: mx.array) -> mx.array:
        d = x.shape[-1]
        n_tok = x.size // d
        idx2d = indices.reshape(n_tok, self.top_k)
        do_sort = idx2d.size >= 4 * self.num_experts
        inv_order = None
        if do_sort:
            flat_idx = idx2d.flatten()
            order = mx.argsort(flat_idx)
            inv_order = mx.argsort(order)
            xe = x.reshape(n_tok, 1, d)[order // self.top_k]
            idx_use = flat_idx[order]
        else:
            xe = x.reshape(n_tok, 1, 1, d)
            idx_use = idx2d
        gu = self.w13_proj(xe, idx_use, sorted_indices=do_sort)
        if gu.dtype == mx.bfloat16 and mx.default_device() == mx.gpu:
            act = swiglu_interleaved(gu, self.gate_cap, self.up_cap)
        else:
            act = _softcapped_swiglu_interleaved(gu, self.gate_cap, self.up_cap)
        out = self.down_proj(act, idx_use, sorted_indices=do_sort)
        if do_sort:
            out = out[inv_order]
        return out.reshape(n_tok, self.top_k, d)

    def routed_experts_sorted(self, x: mx.array, indices: mx.array) -> tuple[mx.array, mx.array]:
        """Run Metal 4.0 NAX grouped prefill MoE (22-27 TF/s) or sorted gather_qmm fallback."""
        from .moe_kernels import bucket_experts_full, prefill_moe_nax_sorted

        d = x.shape[-1]
        n_tok = x.size // d
        if n_tok >= 128:
            res = prefill_moe_nax_sorted(x, indices, self)
            if res is not None:
                return res
        flat_idx = indices.reshape(n_tok * self.top_k).astype(mx.uint32)
        _offsets, sorted_slots, _active, inv_order = bucket_experts_full(flat_idx, num_experts=self.num_experts)
        idx_use = flat_idx[sorted_slots]
        xe = x.reshape(n_tok, 1, d)[sorted_slots // self.top_k]
        gu = self.w13_proj(xe, idx_use, sorted_indices=True)
        act = swiglu_interleaved(gu, self.gate_cap, self.up_cap)
        ys_sorted = self.down_proj(act, idx_use, sorted_indices=True)
        return ys_sorted.reshape(n_tok * self.top_k, d), inv_order

    def __call__(self, x: mx.array) -> mx.array:
        indices, weights = _route(
            x,
            self.gate_weight,
            self.e_score_correction_bias,
            self.top_k,
            self.route_scale,
        )
        out = self.routed_experts(x, indices).reshape(*x.shape[:-1], self.top_k, x.shape[-1])
        return _combine_routes(out, weights)


class AttentionResidual(nn.Module):
    def __init__(self, config: TextConfig, num_queries: int = 1):
        super().__init__()
        shape = (config.hidden_size,) if num_queries == 1 else (num_queries, config.hidden_size)
        self.query = mx.zeros(shape, dtype=mx.float32)
        self.normalize_values = config.attn_res_normalize_values

    def __call__(self, blocks: list[mx.array], partial: mx.array) -> mx.array:
        return _attention_residual(
            mx.stack([*blocks, partial], axis=-2), self.query, self.normalize_values
        )


class MessageGate(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.eps = config.rms_norm_eps
        rank = config.loop_gate_rank
        self.down_proj = nn.Linear(2 * config.hidden_size, rank, bias=False)
        self.forget_gate_proj = nn.Linear(rank, config.hidden_size, bias=False)
        self.message_gate_proj = nn.Linear(rank, config.hidden_size, bias=False)

    def __call__(self, hidden: mx.array, message: mx.array) -> mx.array:
        return _message_gate(
            hidden,
            message,
            self.down_proj.weight,
            self.forget_gate_proj.weight,
            self.message_gate_proj.weight,
            self.eps,
        )


class MessageAttentionResidual(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.query = mx.zeros((config.num_loop_layers, 4, config.hidden_size), dtype=mx.float32)
        self.normalize_values = config.attn_res_normalize_values

    def __call__(self, blocks: list[mx.array], partial: mx.array, loop_layer_idx: int) -> mx.array:
        states = mx.stack([*blocks, partial], axis=-2)
        return _attention_residual(states, self.query[loop_layer_idx], self.normalize_values)


class DecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.is_block_boundary = layer_idx % config.n_layers_per_attn_res_block == 0
        self.is_sliding = config.layer_is_sliding(layer_idx)
        self.attention_input = (
            AttentionResidual(config, 1 if self.is_sliding else 3) if layer_idx else None
        )
        self.mlp_input = AttentionResidual(config)
        self.self_attn = Attention(config, layer_idx)
        self.mlp = SparseMoE(config, layer_idx)
        names = (
            ("input_layernorm",)
            if self.is_sliding
            else ("q_input_layernorm", "k_input_layernorm", "v_input_layernorm")
        )
        for name in (*names, "post_attention_layernorm"):
            self[name] = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_sconv = ShortConvolution(config.hidden_size, config.sconv_kernel_size, 2)
        self.mlp_sconv = ShortConvolution(config.hidden_size, config.sconv_kernel_size, 3)

    def __call__(self, x: mx.array, blocks: list[mx.array], cache=None) -> mx.array:
        conv = cache[1] if cache is not None else None
        mask = _cache_padding_mask(conv, x.shape[1])
        mixed = self.attention_input(blocks, x) if self.attention_input is not None else x[..., None, :]
        if self.is_sliding:
            attn = self.self_attn(
                self.input_layernorm(mixed[..., 0, :]).astype(x.dtype),
                cache=cache,
                conv_mask=mask,
            )
        else:
            parts = [
                norm(mixed[..., i, :]).astype(x.dtype)
                for i, norm in enumerate(
                    (self.q_input_layernorm, self.k_input_layernorm, self.v_input_layernorm)
                )
            ]
            attn = self.self_attn(*parts, cache=cache, conv_mask=mask)
        residual = self.attn_sconv(attn, conv, mask).astype(mx.float32)
        if self.is_block_boundary:
            blocks.append(x)
        else:
            residual = residual + x.astype(mx.float32)
        hidden = self.mlp_input(blocks, residual)[..., 0, :]
        hidden = self.mlp(self.post_attention_layernorm(hidden).astype(x.dtype))
        out = (residual + self.mlp_sconv(hidden, conv, mask).astype(mx.float32)).astype(x.dtype)
        if conv is not None:
            conv.advance(x.shape[1])
        return out


class LoopDecoderLayer(DecoderLayer):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_message_gate = MessageGate(config)
        self.k_message_gate = MessageGate(config)
        self.v_message_gate = MessageGate(config)
        self.mlp_message_gate = MessageGate(config)

    def __call__(
        self,
        x: mx.array,
        blocks: list[mx.array],
        cache=None,
        messages: Optional[mx.array] = None,
    ) -> mx.array:
        if messages is None:
            return super().__call__(x, blocks, cache)
        conv = cache[1] if cache is not None else None
        mask = _cache_padding_mask(conv, x.shape[1])
        mixed = self.attention_input(blocks, x)
        q_msg, k_msg, v_msg, mlp_msg = (messages[..., i, :] for i in range(4))
        if self.is_sliding:
            shared = mixed[..., 0, :]
            parts = (shared, shared, shared)
            norms = (self.input_layernorm,) * 3
        else:
            parts = tuple(mixed[..., i, :] for i in range(3))
            norms = (self.q_input_layernorm, self.k_input_layernorm, self.v_input_layernorm)
        gated_parts = [
            norm(gate(part, msg)).astype(x.dtype)
            for norm, gate, part, msg in zip(
                norms,
                (self.q_message_gate, self.k_message_gate, self.v_message_gate),
                parts,
                (q_msg, k_msg, v_msg),
            )
        ]
        attn = self.self_attn(
            gated_parts[0],
            key_input=gated_parts[1],
            value_input=gated_parts[2],
            cache=cache,
            conv_mask=mask,
        )
        residual = self.attn_sconv(attn, conv, mask).astype(mx.float32)
        if self.is_block_boundary:
            blocks.append(x)
        else:
            residual = residual + x.astype(mx.float32)
        hidden = self.mlp_input(blocks, residual)[..., 0, :]
        hidden = self.mlp_message_gate(hidden, mlp_msg)
        hidden = self.mlp(self.post_attention_layernorm(hidden).astype(x.dtype))
        out = (residual + self.mlp_sconv(hidden, conv, mask).astype(mx.float32)).astype(x.dtype)
        if conv is not None:
            conv.advance(x.shape[1])
        return out


class TextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        start, end = config.loop_layer_index_start, config.loop_layer_index_end
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [
            (LoopDecoderLayer if start <= i < end else DecoderLayer)(config, i)
            for i in range(config.num_hidden_layers)
        ]
        self.message_input = MessageAttentionResidual(config)
        self.final_input = AttentionResidual(config)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed(self, inputs: mx.array) -> mx.array:
        return self.embed_norm(self.embed_tokens(inputs))

    def schedule(self) -> list[tuple[int, int, int]]:
        """Return (physical_layer_idx, visit_cache_idx, loop_pass_idx) for all 48 visits."""
        cfg = self.config
        start, end = cfg.loop_layer_index_start, cfg.loop_layer_index_end
        out = [(i, i, 0) for i in range(start)]
        visit = start
        for p in range(cfg.loop_num_passes):
            for i in range(start, end):
                out.append((i, visit, p))
                visit += 1
        for i in range(end, cfg.num_hidden_layers):
            out.append((i, visit, 0))
            visit += 1
        return out

    def __call__(
        self,
        inputs: Optional[mx.array],
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        skip_final_norm: bool = False,
        capture_layer_ids: Optional[list[int]] = None,
    ):
        cfg = self.config
        start = cfg.loop_layer_index_start
        requested = set(capture_layer_ids or [])
        captures: dict[int, mx.array] = {}
        x = self.embed(inputs) if input_embeddings is None else input_embeddings
        if capture_layer_ids is None:
            from . import fused

            if fused.supported(self, x, cache):
                h = fused.decode_step(self, x, cache)
                return h if skip_final_norm else self.norm(h).astype(x.dtype)
        if (
            x.ndim == 3
            and x.dtype == mx.bfloat16
            and mx.metal.is_available()
            and mx.default_device() == mx.gpu
        ):
            from . import fused

            res = fused.prefill_step(self, x, cache, capture_layer_ids=capture_layer_ids)
            if capture_layer_ids is not None:
                h, caps = res
                return (h if skip_final_norm else self.norm(h).astype(x.dtype)), caps
            return res if skip_final_norm else self.norm(res).astype(x.dtype)
        blocks: list[mx.array] = []
        pre_loop = None
        previous = None
        for step, (layer_idx, cache_idx, pass_idx) in enumerate(self.schedule()):
            layer = self.layers[layer_idx]
            if layer_idx == start:
                if pass_idx == 0:
                    pre_loop = (x, list(blocks))
                else:
                    previous = (blocks[len(pre_loop[1]) + 1 :], x)
                    x, blocks = pre_loop[0], list(pre_loop[1])
            if step in requested:
                captures[step] = x
            c = cache[cache_idx] if cache is not None else None
            if pass_idx > 0:
                messages = self.message_input(previous[0], previous[1], layer_idx - start)
                x = layer(x, blocks, c, messages)
            else:
                x = layer(x, blocks, c)
        h = self.final_input(blocks, x)[..., 0, :]
        normalized = self.norm(h).astype(x.dtype)
        total = cfg.total_num_hidden_layers
        if total in requested:
            captures[total] = normalized
        result = h if skip_final_norm else normalized
        if capture_layer_ids is not None:
            return result, [captures[i] for i in capture_layer_ids]
        return result


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def _logits_from_norm(self, h: mx.array) -> mx.array:
        h = h.astype(self.model.embed_norm.weight.dtype)
        if h.ndim >= 2 and h.shape[-1] % 8 == 0:
            lead = h.shape[:-1]
            m = 1
            for dim in lead:
                m *= dim
            flat = h.reshape(m, h.shape[-1])
            if m <= 64:
                rows = head_logits(flat, self.lm_head.weight)
            else:
                rows = mx.concatenate(
                    [head_logits(flat[i : i + 64], self.lm_head.weight) for i in range(0, m, 64)],
                    axis=0,
                )
            logits = rows.reshape(*lead, -1)
        else:
            logits = h.astype(mx.float32) @ self.lm_head.weight.astype(mx.float32).T
        cap = self.config.final_logit_softcapping
        if cap:
            logits = mx.tanh(logits / cap) * cap
        return logits[..., : self.config.unpadded_vocab_size]

    def greedy_argmax_from_pre_norm(self, pre_norm: mx.array) -> mx.array:
        """Fused FP32 lm_head + argmax over unpadded vocab for M=1..8 (exact greedy, skips 5MB logits & tanh)."""
        normed = self.model.norm(pre_norm)
        lead = normed.shape[:-1]
        m = 1
        for dim in lead:
            m *= dim
        if (
            1 <= m <= 8
            and normed.shape[-1] % 4 == 0
            and self.lm_head.weight.dtype == mx.bfloat16
            and mx.metal.is_available()
            and mx.default_device() == mx.gpu
        ):
            toks = head_argmax(
                normed.reshape(m, normed.shape[-1]),
                self.lm_head.weight,
                self.config.unpadded_vocab_size,
            )
            return toks.reshape(*lead)
        return mx.argmax(self._logits_from_norm(normed), axis=-1).astype(mx.int32)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
        skip_logits: bool = False,
        last_logits_only: bool = False,
        greedy_argmax: bool = False,
        return_hidden: bool = False,
        capture_layer_ids: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        res = self.model(
            inputs,
            cache=cache,
            input_embeddings=inputs_embeds,
            skip_final_norm=True,
            capture_layer_ids=capture_layer_ids,
        )
        pre_norm, captured = res if capture_layer_ids is not None else (res, None)
        if skip_logits:
            logits = None
            argmax_toks = None
        elif greedy_argmax:
            logits = None
            src = pre_norm[:, -1:, :] if last_logits_only else pre_norm
            argmax_toks = self.greedy_argmax_from_pre_norm(src)
        elif last_logits_only:
            logits = self._logits_from_norm(self.model.norm(pre_norm[:, -1:, :]))
            argmax_toks = None
        else:
            logits = self._logits_from_norm(self.model.norm(pre_norm))
            argmax_toks = None
        return {
            "logits": logits,
            "argmax": argmax_toks,
            "hidden_states": captured if capture_layer_ids is not None else ([pre_norm] if return_hidden else None),
        }

    def make_cache(self) -> list[CacheList]:
        return [
            CacheList(
                (
                    RotatingKVCache(max_size=self.config.sliding_window_size, keep=0)
                    if self.model.layers[layer_idx].is_sliding
                    else GrowingKVCache()
                ),
                ArraysCache(4),
            )
            for layer_idx, _, _ in self.model.schedule()
        ]

    def make_batch_cache(self, left_padding: list[int]) -> list[CacheList]:
        return [
            CacheList(
                (
                    BatchRotatingKVCache(
                        max_size=self.config.sliding_window_size, left_padding=left_padding
                    )
                    if self.model.layers[layer_idx].is_sliding
                    else GrowingBatchKVCache(left_padding=left_padding)
                ),
                ArraysCache(4, left_padding=left_padding),
            )
            for layer_idx, _, _ in self.model.schedule()
        ]


_GATES = {"attn": "q", "k": "k", "v": "v", "mlp": "mlp"}
_GATE_WEIGHTS = {
    "down_weight": "down_proj",
    "forget_weight": "forget_gate_proj",
    "gain_weight": "message_gate_proj",
}
_ATTN_NAMES = {
    "wq_du": "q_proj",
    "wk_dv": "k_proj",
    "wv_dv": "v_proj",
    "wr_du": "r_proj",
    "wo_ud": "o_proj",
    "gate_attn": "gate_proj",
    "attn_sink_proj": "sink_proj",
}
_NORMS = {
    "attn_norm": "input_layernorm",
    "mlp_norm": "post_attention_layernorm",
    "q_input_norm": "q_input_layernorm",
    "k_input_norm": "k_input_layernorm",
    "v_input_norm": "v_input_layernorm",
}


def sanitize_text_weights(raw_weights: dict[str, mx.array]) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    experts: dict[str, dict[str, mx.array]] = {}
    loop_q: dict[str, mx.array] = {}

    for key, value in raw_weights.items():
        if key.startswith("model.visual.") or key.startswith("model.mtp.") or key.startswith("mtp."):
            continue
        if key in (
            "model.llm.middle_cycle_cross_pass_queries",
            "model.llm.middle_cycle_cross_pass_kv_queries",
        ):
            loop_q[key.rsplit(".", 1)[1]] = value.astype(mx.float32)
            continue
        if key.startswith("model.llm.layers."):
            layer, sub = key[len("model.llm.layers.") :].split(".", 1)
            base = f"model.layers.{layer}."
            if sub.startswith("mlp.experts."):
                experts.setdefault(layer, {})[sub[len("mlp.experts.") :]] = value
                continue
            if ".controllers." in key or sub.startswith("controllers."):
                _, gate, weight_name = sub.split(".")
                if weight_name in ("forget_weight", "gain_weight"):
                    value = value.astype(mx.float32)
                out[f"{base}{_GATES[gate]}_message_gate.{_GATE_WEIGHTS[weight_name]}.weight"] = value
                continue
            if sub.startswith("attn."):
                sub = sub[len("attn.") :]
                name, _, leaf = sub.partition(".")
                sub = _ATTN_NAMES.get(name, name) + ("." + leaf if leaf else "")
                sub = {
                    "q_l2_clip.scale": "q_scale",
                    "rel_l2_clip.scale": "rel_scale",
                    "rel_logits_proj.proj": "rel_proj",
                }.get(sub, sub)
                for conv in ("k_sconv", "v_sconv"):
                    if sub == conv + ".weight":
                        sub = conv + ".conv.weight"
                        value = _to_mlx_conv(value)
                    elif sub == conv + ".static_w":
                        sub = conv + ".static_weight"
                    elif sub == conv + ".weight_proj":
                        sub += ".weight"
                sub = "self_attn." + sub
            elif sub in ("attn_sconv.weight", "mlp_sconv.weight"):
                sub = sub.replace(".weight", ".conv.weight")
                value = _to_mlx_conv(value)
            elif sub == "mlp.gate.weight":
                sub = "mlp.gate_weight"
                value = value.astype(mx.float32)
            elif sub == "mlp.gate.bias":
                sub = "mlp.e_score_correction_bias"
                value = value.astype(mx.float32)
            else:
                name, _, leaf = sub.partition(".")
                sub = _NORMS.get(name, name) + "." + leaf
            out[base + sub] = value
        elif key.startswith("model.llm.attn_res."):
            sub = key[len("model.llm.attn_res.") :]
            if sub == "output_query":
                target = "final_input.query"
            else:
                kind, index = sub.split(".")
                layer = int(index) + (kind == "attn_queries")
                target = f"layers.{layer}.{'attention_input' if kind == 'attn_queries' else 'mlp_input'}.query"
            out["model." + target] = value.astype(mx.float32)
        elif key.startswith("model.llm."):
            sub = key[len("model.llm.") :]
            if sub == "unembed.weight":
                out["lm_head.weight"] = value
            else:
                out["model." + sub.replace("embed.weight", "embed_tokens.weight")] = value

    for layer, tensors in experts.items():
        base = f"model.layers.{layer}.mlp."
        for name, proj in (("w13_weight", "w13_proj"), ("w2_weight", "down_proj")):
            weight = tensors[name]
            packed = (name + "_scale") in tensors
            out[f"{base}{proj}.weight"] = weight.view(mx.uint32) if packed else weight
            if packed:
                out[f"{base}{proj}.scales"] = tensors[name + "_scale"]
            if (name + "_amax") in tensors:
                scale_2 = (
                    tensors[name + "_amax"].astype(mx.float32)
                    / tensors[name + "_outer_scale_denominator"].astype(mx.float32)
                )
                out[f"{base}{proj}.global_scale"] = 2688.0 * scale_2

    if "middle_cycle_cross_pass_queries" in loop_q:
        qm = loop_q["middle_cycle_cross_pass_queries"]
        kv = loop_q["middle_cycle_cross_pass_kv_queries"]
        n = qm.shape[0] // 2
        qm = qm.reshape(n, 2, -1)
        kv = kv.reshape(n, 2, -1)
        out["model.message_input.query"] = mx.stack(
            [qm[:, 0], kv[:, 0], kv[:, 1], qm[:, 1]], axis=1
        )
    return out


_PEFT_ATTN_MAP = {
    "self_attn.q_proj.linear": "self_attn.q_proj",
    "self_attn.k_proj.linear": "self_attn.k_proj",
    "self_attn.v_proj.linear": "self_attn.v_proj",
    "self_attn.o_proj.linear": "self_attn.o_proj",
    "self_attn.o_proj.gate_proj": "self_attn.gate_proj",
    "self_attn.relative_position_bias.linear": "self_attn.r_proj",
    "self_attn.sink_proj": "self_attn.sink_proj",
}
_PEFT_GATE_PREFIXES = tuple(
    f"{gate}.{proj}"
    for gate in ("q_message_gate", "k_message_gate", "v_message_gate", "mlp_message_gate")
    for proj in ("down_proj", "forget_gate_proj", "message_gate_proj")
)


def _locate_peft_adapter_dir(adapter_dir: str | Path) -> Path:
    root = Path(adapter_dir).expanduser()
    for candidate in (root, root / "adapter"):
        if (candidate / "adapter_config.json").is_file() and (candidate / "adapter_model.safetensors").is_file():
            return candidate
    raise FileNotFoundError(
        f"PEFT LoRA adapter not found in {root} or {root / 'adapter'} "
        "(expected adapter_config.json and adapter_model.safetensors)."
    )


def _map_peft_key_to_mlx(key: str) -> tuple[str | None, str | None, bool]:
    """Return `(mlx_weight_key, 'A'|'B', skipped_vision)` for one PEFT tensor key."""
    if ".lora_A." in key:
        which = "A"
        mod = key.split(".lora_A.", 1)[0]
    elif ".lora_B." in key:
        which = "B"
        mod = key.split(".lora_B.", 1)[0]
    else:
        return None, None, False

    for prefix in ("base_model.model.model.", "base_model.model.", "model."):
        if mod.startswith(prefix):
            mod = mod[len(prefix) :]
            break

    if mod.startswith("vision_tower.") or ".vision_tower." in mod:
        return None, which, True

    if mod.startswith("language_model.layers."):
        mod = mod[len("language_model.") :]
    if not mod.startswith("layers."):
        raise ValueError(f"Unsupported PEFT LoRA target module: {key}")

    parts = mod.split(".", 2)
    if len(parts) != 3:
        raise ValueError(f"Malformed PEFT layer module path: {key}")
    layer_idx, rest = parts[1], parts[2]
    if rest in _PEFT_ATTN_MAP:
        mapped = _PEFT_ATTN_MAP[rest]
    elif rest in _PEFT_GATE_PREFIXES:
        mapped = rest
    else:
        raise ValueError(f"Unsupported Inkling PEFT LoRA suffix '{rest}' in key: {key}")
    return f"model.layers.{layer_idx}.{mapped}.weight", which, False


def apply_peft_lora_adapter(clean_weights: dict[str, mx.array], adapter_dir: str | Path) -> dict[str, int]:
    """Merge a PEFT LoRA adapter (`lora_A`/`lora_B`) into sanitized Inkling-2-Mini text weights."""
    adir = _locate_peft_adapter_dir(adapter_dir)
    cfg = json.loads((adir / "adapter_config.json").read_text())
    r = float(cfg["r"])
    alpha = float(cfg.get("lora_alpha", r))
    if r <= 0:
        raise ValueError(f"Adapter rank r must be positive, got {r}")
    scale = alpha / r

    raw_lora = mx.load(str(adir / "adapter_model.safetensors"))
    if raw_lora:
        mx.eval(*raw_lora.values())

    grouped: dict[str, dict[str, mx.array]] = {}
    skipped_vision = 0
    for key, tensor in raw_lora.items():
        target, which, is_vision = _map_peft_key_to_mlx(key)
        if is_vision:
            skipped_vision += 1
            continue
        if target is None or which is None:
            continue
        grouped.setdefault(target, {})[which] = tensor

    merged = 0
    for target, pair in grouped.items():
        if "A" not in pair or "B" not in pair:
            raise ValueError(f"Incomplete LoRA pair for {target}: got keys {sorted(pair.keys())}")
        if target not in clean_weights:
            raise ValueError(f"Target weight '{target}' not present in sanitized base model weights")
        base_w = clean_weights[target]
        a = pair["A"]
        b = pair["B"]
        if b.shape[0] != base_w.shape[0] or a.shape[1] != base_w.shape[1] or a.shape[0] != b.shape[1]:
            raise ValueError(
                f"Shape mismatch merging LoRA into {target}: base={base_w.shape}, A={a.shape}, B={b.shape}"
            )
        delta = (b.astype(mx.float32) @ a.astype(mx.float32)) * scale
        updated = (base_w.astype(mx.float32) + delta).astype(base_w.dtype)
        clean_weights[target] = updated
        merged += 1

    if merged == 0:
        raise ValueError(f"No text LoRA modules merged from {adir}")
    mx.eval(*(clean_weights[t] for t in grouped))
    return {"merged_modules": merged, "skipped_vision_tensors": skipped_vision}


def load_model(
    model_dir: str | Path,
    adapter_dir: Optional[str | Path] = None,
) -> tuple[LanguageModel, Any, ModelConfig]:
    """Load Inkling 2 Mini natively into MLX without weight requantization or FP32 lm_head copy."""
    from transformers import AutoTokenizer

    path = Path(model_dir).expanduser()
    raw_cfg = json.loads((path / "config.json").read_text())
    cfg = ModelConfig.from_dict(raw_cfg)

    lm = LanguageModel(cfg.text_config)
    raw_weights = mx.load(str(path / "model-00001-of-00001.safetensors"))
    clean_weights = sanitize_text_weights(raw_weights)
    del raw_weights

    if adapter_dir is not None:
        stats = apply_peft_lora_adapter(clean_weights, adapter_dir)
        print(
            f"[inkling] Merged {stats['merged_modules']} LoRA text modules from {adapter_dir} "
            f"(skipped {stats['skipped_vision_tensors']} vision tensors)",
            flush=True,
        )

    lm.load_weights(list(clean_weights.items()), strict=True)
    mx.eval(lm.parameters())
    lm.eval()
    mx.clear_cache()

    tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=False, local_files_only=True)
    return lm, tokenizer, cfg
