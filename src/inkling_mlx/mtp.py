"""Built-in Inkling 2 Mini MTP (DSpark + DFlare + RNN MarkovHead) speculative drafter.

Loads ONLY the official released `mtp.safetensors` (636 MB BF16, 3 layers, block_size=8,
markov_rank=256, target_layer_ids=[5, 23, 41, 48]) and implements:
- One-time FP64 Markov head folding against target `lm_head` (bounded 4096-row tiles).
- Packed QKV / KVCtx / GateUp BF16 projections for single-pass bidirectional drafting.
- Fused Metal `markov_argmax` kernel (zero [201024] intermediate writes -> ~0.41 ms/draft tok).
- Exact zero-copy O(1) rollback helpers for KVCache, RotatingKVCache, and ArraysCache.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_unflatten
from mlx_vlm.models.cache import CacheList, KVCache, RotatingKVCache

from .kernels import head_logits


@dataclass
class MTPConfig:
    hidden_size: int = 2304
    num_hidden_layers: int = 3
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 9216
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    local_layer_ids: tuple[int, ...] = (0, 2)
    sliding_window_size: int = 1024
    block_size: int = 8
    markov_rank: int = 256
    target_layer_ids: tuple[int, ...] = (5, 23, 41, 48)
    vocab_size: int = 201024
    unpadded_vocab_size: int = 200058
    final_logit_softcapping: float = 30.0

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "MTPConfig":
        raw = json.loads((Path(model_dir) / "config.json").read_text())
        m = raw["mtp_config"]
        t = raw["text_config"]
        return cls(
            hidden_size=m["hidden_size"],
            num_hidden_layers=m["num_hidden_layers"],
            num_attention_heads=m["num_attention_heads"],
            num_key_value_heads=m["num_key_value_heads"],
            head_dim=m["head_dim"],
            intermediate_size=m["intermediate_size"],
            rms_norm_eps=m["rms_norm_eps"],
            rope_theta=m["rope_theta"],
            local_layer_ids=tuple(m["local_layer_ids"]),
            sliding_window_size=m["sliding_window_size"],
            block_size=m["block_size"],
            markov_rank=m["markov_rank"],
            target_layer_ids=tuple(m["target_layer_ids"]),
            vocab_size=t["vocab_size"],
            unpadded_vocab_size=t["unpadded_vocab_size"],
            final_logit_softcapping=t.get("final_logit_softcapping", 30.0),
        )


_MARKOV_ARGMAX_P1_SRC = r"""
    constexpr int SG = 8;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int tg_id = int(threadgroup_position_in_grid.x);
    const int v_base = tg_id * (SG * ROWS_PER_SG);

    threadgroup float lat[RANK];
    for (uint i = thread_position_in_threadgroup.x; i < RANK; i += SG * 32) {
        lat[i] = float(latent[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float best_val = -INFINITY;
    uint best_idx = 0u;

    for (int ri = 0; ri < ROWS_PER_SG; ++ri) {
        const int v = v_base + int(sg) * ROWS_PER_SG + ri;
        if (v < V_LIMIT) {
            const int k = int(lane) * 8;
            auto wp = (const device vec<bfloat, 4>*)(fold_w + size_t(v) * RANK + k);
            const vec<bfloat, 4> wa = wp[0], wb = wp[1];
            float dot = float(wa[0]) * lat[k + 0] + float(wa[1]) * lat[k + 1]
                      + float(wa[2]) * lat[k + 2] + float(wa[3]) * lat[k + 3]
                      + float(wb[0]) * lat[k + 4] + float(wb[1]) * lat[k + 5]
                      + float(wb[2]) * lat[k + 6] + float(wb[3]) * lat[k + 7];
            const float total = simd_sum(dot);
            if (lane == 0) {
                const float score = float(bfloat(float(base_l[v]) + float(bfloat(total + float(fold_b[v])))));
                if (score > best_val) {
                    best_val = score;
                    best_idx = uint(v);
                }
            }
        }
    }

    threadgroup float sg_val[SG];
    threadgroup uint sg_idx[SG];
    if (lane == 0) {
        sg_val[sg] = best_val;
        sg_idx[sg] = best_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        float bv = sg_val[0];
        uint bi = sg_idx[0];
        for (int s = 1; s < SG; ++s) {
            if (sg_val[s] > bv) {
                bv = sg_val[s];
                bi = sg_idx[s];
            }
        }
        part_val[tg_id] = bv;
        part_idx[tg_id] = bi;
    }
"""

_MARKOV_ARGMAX_P2_SRC = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int n = part_val_shape[0];
    float bv = -INFINITY;
    uint bi = 0u;
    for (int i = int(tid); i < n; i += 256) {
        const float v = part_val[i];
        const uint idx = part_idx[i];
        if (v > bv || (v == bv && idx < bi)) {
            bv = v;
            bi = idx;
        }
    }
    for (ushort off = 16; off > 0; off >>= 1) {
        const float ov = simd_shuffle_xor(bv, off);
        const uint oi = simd_shuffle_xor(bi, off);
        if (ov > bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
        }
    }
    threadgroup float sg_v[8];
    threadgroup uint sg_i[8];
    if (lane == 0) { sg_v[sg] = bv; sg_i[sg] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        float fv = sg_v[0];
        uint fi = sg_i[0];
        for (int s = 1; s < 8; ++s) {
            if (sg_v[s] > fv || (sg_v[s] == fv && sg_i[s] < fi)) {
                fv = sg_v[s];
                fi = sg_i[s];
            }
        }
        out_tok[0] = int(fi);
    }
"""


@lru_cache(maxsize=1)
def _markov_argmax_p1():
    return mx.fast.metal_kernel(
        name="inkling_mtp_markov_argmax_p1",
        input_names=["latent", "fold_w", "fold_b", "base_l"],
        output_names=["part_val", "part_idx"],
        source=_MARKOV_ARGMAX_P1_SRC,
    )


@lru_cache(maxsize=1)
def _markov_argmax_p2():
    return mx.fast.metal_kernel(
        name="inkling_mtp_markov_argmax_p2",
        input_names=["part_val", "part_idx"],
        output_names=["out_tok"],
        source=_MARKOV_ARGMAX_P2_SRC,
    )


def fused_markov_argmax(
    latent_1d: mx.array,
    fold_w: mx.array,
    fold_b: mx.array,
    base_1d: mx.array,
    v_limit: int,
) -> mx.array:
    """Single-read fused Markov bias + argmax returning (1,) int32 token (~0.41ms)."""
    rank = fold_w.shape[1]
    rows_per_sg = 8
    rows_per_tg = 8 * rows_per_sg
    ntg = (v_limit + rows_per_tg - 1) // rows_per_tg
    pv, pi = _markov_argmax_p1()(
        inputs=[latent_1d.reshape(rank), fold_w, fold_b, base_1d.reshape(-1)],
        template=[("RANK", rank), ("V_LIMIT", v_limit), ("ROWS_PER_SG", rows_per_sg)],
        grid=(ntg * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(ntg,), (ntg,)],
        output_dtypes=[mx.float32, mx.uint32],
    )
    return _markov_argmax_p2()(
        inputs=[pv, pi],
        grid=(256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1,)],
        output_dtypes=[mx.int32],
    )[0]


class DraftAttention(nn.Module):
    def __init__(self, config: MTPConfig, local: bool):
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.window = config.sliding_window_size if local else None
        self.q_proj = nn.Linear(config.hidden_size, self.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.k_proj_ctx = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.v_proj_ctx = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm_ctx = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)
        self._qkv_w: Optional[mx.array] = None
        self._kv_ctx_w: Optional[mx.array] = None

    def pack_weights(self) -> None:
        self._qkv_w = mx.concatenate([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], axis=0)
        self._kv_ctx_w = mx.concatenate([self.k_proj_ctx.weight, self.v_proj_ctx.weight], axis=0)
        mx.eval(self._qkv_w, self._kv_ctx_w)

    def _heads(self, x: mx.array, heads: int) -> mx.array:
        return x.reshape(*x.shape[:2], heads, self.head_dim).transpose(0, 2, 1, 3)

    def __call__(self, x: mx.array, context: mx.array, cache) -> mx.array:
        if self.window and context.shape[1] > self.window:
            skip = context.shape[1] - self.window
            context = context[:, skip:]
            cache.offset += skip
        offset = cache.offset
        kv_dim = self.kv_heads * self.head_dim
        if self._kv_ctx_w is not None:
            kv_ctx = context @ self._kv_ctx_w.T
            k_ctx_raw, v_ctx_raw = kv_ctx[..., :kv_dim], kv_ctx[..., kv_dim:]
        else:
            k_ctx_raw, v_ctx_raw = self.k_proj_ctx(context), self.v_proj_ctx(context)
        k = self.k_norm_ctx(self._heads(k_ctx_raw, self.kv_heads))
        v = self._heads(v_ctx_raw, self.kv_heads)
        k, v = cache.update_and_fetch(self.rope(k, offset=offset), v)
        if isinstance(cache, RotatingKVCache):
            if cache.offset > cache.max_size:
                k, v = cache._temporal_order(k), cache._temporal_order(v)
            if k.shape[2] > self.window:
                k, v = k[..., -self.window :, :], v[..., -self.window :, :]
        offset += context.shape[1]

        q_dim = self.heads * self.head_dim
        if self._qkv_w is not None:
            qkv = x @ self._qkv_w.T
            q_raw = qkv[..., :q_dim]
            bk_raw = qkv[..., q_dim : q_dim + kv_dim]
            bv_raw = qkv[..., q_dim + kv_dim :]
        else:
            q_raw, bk_raw, bv_raw = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        q = self.q_norm(self._heads(q_raw, self.heads))
        block_k = self.k_norm(self._heads(bk_raw, self.kv_heads))
        k = mx.concatenate([k, self.rope(block_k, offset=offset)], axis=2)
        v = mx.concatenate([v, self._heads(bv_raw, self.kv_heads)], axis=2)
        out = mx.fast.scaled_dot_product_attention(
            self.rope(q, offset=offset), k, v, scale=self.head_dim**-0.5, mask=None
        )
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(*x.shape[:2], -1))

    def prefill_context(self, context: mx.array, cache) -> None:
        """Update only draft KV cache with prompt context hidden states (used for intermediate prefill chunks)."""
        if self.window and context.shape[1] > self.window:
            skip = context.shape[1] - self.window
            context = context[:, skip:]
            cache.offset += skip
        offset = cache.offset
        kv_dim = self.kv_heads * self.head_dim
        if self._kv_ctx_w is not None:
            kv_ctx = context @ self._kv_ctx_w.T
            k_ctx_raw, v_ctx_raw = kv_ctx[..., :kv_dim], kv_ctx[..., kv_dim:]
        else:
            k_ctx_raw, v_ctx_raw = self.k_proj_ctx(context), self.v_proj_ctx(context)
        k = self.k_norm_ctx(self._heads(k_ctx_raw, self.kv_heads))
        v = self._heads(v_ctx_raw, self.kv_heads)
        cache.update_and_fetch(self.rope(k, offset=offset), v)


class DraftMLP(nn.Module):
    def __init__(self, config: MTPConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self._gu_w: Optional[mx.array] = None
        self._inter = config.intermediate_size

    def pack_weights(self) -> None:
        self._gu_w = mx.concatenate([self.gate_proj.weight, self.up_proj.weight], axis=0)
        mx.eval(self._gu_w)

    def __call__(self, x: mx.array) -> mx.array:
        if self._gu_w is not None:
            gu = x @ self._gu_w.T
            gate = gu[..., : self._inter].astype(mx.float32)
            up = gu[..., self._inter :].astype(mx.float32)
        else:
            gate = self.gate_proj(x).astype(mx.float32)
            up = self.up_proj(x).astype(mx.float32)
        return self.down_proj((nn.silu(gate) * up).astype(x.dtype))


class DraftLayer(nn.Module):
    def __init__(self, config: MTPConfig, index: int):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dflare_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = DraftAttention(config, index in config.local_layer_ids)
        self.mlp = DraftMLP(config)

    def pack_weights(self) -> None:
        self.self_attn.pack_weights()
        self.mlp.pack_weights()

    def __call__(self, x: mx.array, residual: Optional[mx.array], context: mx.array, cache) -> tuple[mx.array, mx.array]:
        summed = x if residual is None else x.astype(mx.float32) + residual.astype(mx.float32)
        residual = summed.astype(x.dtype)
        x = self.self_attn(self.input_layernorm(summed).astype(x.dtype), context, cache)
        summed = x.astype(mx.float32) + residual.astype(mx.float32)
        return self.mlp(self.post_attention_layernorm(summed).astype(x.dtype)), summed.astype(x.dtype)


class MarkovHead(nn.Module):
    def __init__(self, config: MTPConfig):
        super().__init__()
        rank, hidden = config.markov_rank, config.hidden_size
        self.markov_w1 = nn.Embedding(config.vocab_size, rank)
        self.rnn_proj = nn.Linear(2 * rank + 2 * hidden, 3 * rank)
        self.hidden_out = nn.Linear(rank, hidden)
        self.dflare_norm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        self._fold_weight: Optional[mx.array] = None
        self._fold_bias: Optional[mx.array] = None
        self._rnn_w_ctx_t: Optional[mx.array] = None
        self._rnn_w_step_t: Optional[mx.array] = None
        self._rnn_b_f32: Optional[mx.array] = None

    def fold(self, lm_head) -> None:
        """One-time FP64 folding of hidden_out into lm_head (bounded 4096-row tiles)."""
        projection = np.array(self.hidden_out.weight.astype(mx.float32), dtype=np.float64)
        bias = np.array(self.hidden_out.bias.astype(mx.float32), dtype=np.float64)
        weights, biases = [], []
        dtype = self.hidden_out.weight.dtype
        for start in range(0, lm_head.weight.shape[0], 4096):
            rows = slice(start, start + 4096)
            head = np.array(lm_head.weight[rows].astype(mx.float32), dtype=np.float64)
            weights.append(mx.array((head @ projection).astype(np.float32)).astype(dtype))
            biases.append(mx.array((head @ bias).astype(np.float32)).astype(dtype))
        self._fold_weight = mx.contiguous(mx.concatenate(weights))
        self._fold_bias = mx.contiguous(mx.concatenate(biases))
        rank = self.markov_w1.weight.shape[1]
        self._rnn_w_ctx_t = mx.contiguous(self.rnn_proj.weight[:, 2 * rank :].astype(mx.float32).T)
        self._rnn_w_step_t = mx.contiguous(self.rnn_proj.weight[:, : 2 * rank].astype(mx.float32).T)
        self._rnn_b_f32 = self.rnn_proj.bias.astype(mx.float32)
        mx.eval(self._fold_weight, self._fold_bias, self._rnn_w_ctx_t, self._rnn_w_step_t, self._rnn_b_f32)

    def context(self, hidden: mx.array, f_rnn: mx.array) -> mx.array:
        inputs = mx.concatenate([hidden, mx.broadcast_to(f_rnn, hidden.shape)], axis=-1)
        return inputs.astype(mx.float32) @ self._rnn_w_ctx_t + self._rnn_b_f32

    def step_latent(self, state: mx.array, previous: mx.array, context_i: mx.array) -> tuple[mx.array, mx.array]:
        inputs = mx.concatenate([state, self.markov_w1(previous)], axis=-1)
        raw = (inputs.astype(mx.float32) @ self._rnn_w_step_t + context_i).astype(state.dtype)
        gate, candidate, output = mx.split(raw, 3, axis=-1)
        gate = mx.sigmoid(gate.astype(mx.float32)).astype(state.dtype)
        candidate = mx.tanh(candidate.astype(mx.float32)).astype(state.dtype)
        new_state = gate * state + (1 - gate) * candidate
        latent = mx.tanh(output.astype(mx.float32)).astype(state.dtype)
        return new_state, latent

    def step_logits(self, state: mx.array, previous: mx.array, context_i: mx.array) -> tuple[mx.array, mx.array]:
        new_state, latent = self.step_latent(state, previous, context_i)
        bias = mx.addmm(self._fold_bias, latent, self._fold_weight.T)
        return new_state, bias


class InklingMTPDrafter(nn.Module):
    """Built-in Inkling 2 Mini MTP DSpark drafter (3 BF16 layers + folded MarkovHead)."""

    def __init__(self, config: MTPConfig):
        super().__init__()
        self.config = config
        self.layers = [DraftLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.markov_head = MarkovHead(config)
        self.fuse_weight = mx.zeros((config.num_hidden_layers + 1, len(config.target_layer_ids)))
        self.mask_embed = mx.zeros((config.hidden_size,))
        self._fuse_alpha: Optional[mx.array] = None
        self._target = None
        self._native_runner = None

    def bind(self, target_model) -> "InklingMTPDrafter":
        if self._target is not target_model:
            self.markov_head.fold(target_model.lm_head)
            for layer in self.layers:
                layer.pack_weights()
            self._fuse_alpha = mx.softmax(self.fuse_weight.astype(mx.float32), axis=-1)
            mx.eval(self._fuse_alpha)
            self._target = target_model
            from .native_mtp import try_create_native_mtp_runner

            self._native_runner = try_create_native_mtp_runner(self)
        return self

    def make_cache(self) -> list[KVCache | RotatingKVCache]:
        return [
            (
                KVCache()
                if layer.self_attn.window is None
                else RotatingKVCache(max_size=layer.self_attn.window, keep=0)
            )
            for layer in self.layers
        ]

    def _mix(self, hidden: mx.array, row: int, norm: nn.RMSNorm) -> mx.array:
        features = hidden.reshape(*hidden.shape[:2], -1, self.config.hidden_size).astype(mx.float32)
        alpha = self._fuse_alpha[row].astype(hidden.dtype).astype(mx.float32)
        mixed = mx.sum(features * alpha[None, None, :, None], axis=2)
        return norm(mixed).astype(hidden.dtype)

    def prefill_context(self, hidden: mx.array, draft_cache: list, *, skip_local: bool = False) -> None:
        """Ingest prompt-chunk captured hidden states into draft KV caches without drafting."""
        n_tok = hidden.shape[1]
        for i, (layer, entry) in enumerate(zip(self.layers, draft_cache)):
            win = layer.self_attn.window
            if win is not None:
                if skip_local:
                    entry.keys = None
                    entry.values = None
                    entry._idx = 0
                    entry.offset += n_tok
                    continue
                if n_tok >= win:
                    entry.keys = None
                    entry.values = None
                    entry._idx = 0
                    skip = n_tok - win
                    entry.offset += skip
                    layer.self_attn.prefill_context(
                        self._mix(hidden[:, skip:, :], i, layer.dflare_norm), entry
                    )
                    continue
            layer.self_attn.prefill_context(self._mix(hidden, i, layer.dflare_norm), entry)

    def draft_greedy(
        self,
        anchor_token: int | mx.array,
        hidden: mx.array,
        draft_cache: list,
        num_draft: int,
    ) -> mx.array:
        """Draft `num_draft` tokens (1 <= num_draft <= block_size - 1) greedily in one fused pass."""
        if not 1 <= num_draft < self.config.block_size:
            raise ValueError(f"num_draft={num_draft} out of range 1..{self.config.block_size - 1}")
        anchor = (
            mx.array([anchor_token], dtype=mx.int32)
            if isinstance(anchor_token, int)
            else anchor_token.reshape(1).astype(mx.int32)
        )
        embed = self._target.model.embed(anchor[:, None])
        masks = mx.broadcast_to(
            self.mask_embed.astype(embed.dtype),
            (1, self.config.block_size - 1, embed.shape[-1]),
        )
        x, residual = mx.concatenate([embed, masks], axis=1), None
        for i, (layer, entry) in enumerate(zip(self.layers, draft_cache)):
            x, residual = layer(x, residual, self._mix(hidden, i, layer.dflare_norm), entry)
        x = self.norm(x.astype(mx.float32) + residual.astype(mx.float32)).astype(embed.dtype)[:, 1 : num_draft + 1]
        base = head_logits(x.reshape(num_draft, -1), self._target.lm_head.weight).astype(embed.dtype)
        f_rnn = self._mix(hidden[:, -1:], -1, self.markov_head.dflare_norm)
        context = self.markov_head.context(x, f_rnn)

        state = mx.zeros((1, self.config.markov_rank), embed.dtype)
        previous = anchor
        tokens = []
        v_lim = self.config.unpadded_vocab_size
        use_fused = mx.default_device() == mx.gpu and embed.dtype == mx.bfloat16
        for i in range(num_draft):
            if use_fused:
                state, latent = self.markov_head.step_latent(state, previous, context[:, i])
                previous = fused_markov_argmax(
                    latent, self.markov_head._fold_weight, self.markov_head._fold_bias, base[i], v_lim
                )
            else:
                state, bias = self.markov_head.step_logits(state, previous, context[:, i])
                logits = base[i : i + 1] + bias
                previous = mx.argmax(logits[:, :v_lim], axis=-1).astype(mx.int32)
            tokens.append(previous)
        return mx.concatenate(tokens, axis=0)

    def draft_greedy_list(
        self,
        anchor_token: int,
        hidden: mx.array,
        draft_cache: list,
        num_draft: int,
    ) -> list[int]:
        """Fast single-CB native Metal 4.0 greedy draft returning Python `list[int]`."""
        if (
            self._native_runner is not None
            and hidden.dtype == mx.bfloat16
            and 1 <= int(hidden.shape[1]) <= self.config.block_size
            and 1 <= num_draft < self.config.block_size
        ):
            return self._native_runner.draft_greedy_list(anchor_token, hidden, draft_cache, num_draft)
        drafts = self.draft_greedy(anchor_token, hidden, draft_cache, num_draft)
        mx.eval(drafts)
        return drafts.tolist()

    def draft_sampled(
        self,
        anchor_token: int | mx.array,
        hidden: mx.array,
        draft_cache: list,
        num_draft: int,
        temperature: float,
        top_p: float,
    ) -> tuple[mx.array, mx.array]:
        """Draft `num_draft` tokens with sampling, returning (tokens[K], probs[K, V]) for exact rejection sampling."""
        anchor = (
            mx.array([anchor_token], dtype=mx.int32)
            if isinstance(anchor_token, int)
            else anchor_token.reshape(1).astype(mx.int32)
        )
        embed = self._target.model.embed(anchor[:, None])
        masks = mx.broadcast_to(
            self.mask_embed.astype(embed.dtype),
            (1, self.config.block_size - 1, embed.shape[-1]),
        )
        x, residual = mx.concatenate([embed, masks], axis=1), None
        for i, (layer, entry) in enumerate(zip(self.layers, draft_cache)):
            x, residual = layer(x, residual, self._mix(hidden, i, layer.dflare_norm), entry)
        x = self.norm(x.astype(mx.float32) + residual.astype(mx.float32)).astype(embed.dtype)[:, 1 : num_draft + 1]
        base = head_logits(x.reshape(num_draft, -1), self._target.lm_head.weight).astype(embed.dtype)
        f_rnn = self._mix(hidden[:, -1:], -1, self.markov_head.dflare_norm)
        context = self.markov_head.context(x, f_rnn)

        state = mx.zeros((1, self.config.markov_rank), embed.dtype)
        previous = anchor
        tokens, prob_rows = [], []
        v_lim = self.config.unpadded_vocab_size
        cap = self.config.final_logit_softcapping
        for i in range(num_draft):
            state, bias = self.markov_head.step_logits(state, previous, context[:, i])
            logits = (base[i : i + 1] + bias).astype(mx.float32)
            if cap:
                logits = cap * mx.tanh(logits / cap)
            logits = logits[:, :v_lim] / float(temperature)
            probs = mx.softmax(logits, axis=-1)
            if 0.0 < top_p < 1.0:
                s_idx = mx.argsort(-probs, axis=-1)
                s_p = mx.take_along_axis(probs, s_idx, axis=-1)
                cum = mx.cumsum(s_p, axis=-1)
                keep = (cum - s_p) < top_p
                masked_s = mx.where(keep, s_p, 0.0)
                masked_s = masked_s / mx.sum(masked_s, axis=-1, keepdims=True)
                inv = mx.argsort(s_idx, axis=-1)
                probs = mx.take_along_axis(masked_s, inv, axis=-1)
            tok = mx.random.categorical(mx.log(mx.maximum(probs, 1e-20))).astype(mx.int32)
            previous = tok.reshape(1)
            tokens.append(previous)
            prob_rows.append(probs[0])
        return mx.concatenate(tokens, axis=0), mx.stack(prob_rows, axis=0)


def load_mtp_drafter(model_dir: str | Path, target_model) -> Optional[InklingMTPDrafter]:
    """Load and bind the built-in Inkling 2 Mini MTP drafter from `mtp.safetensors`."""
    mdir = Path(model_dir)
    weights_path = mdir / "mtp.safetensors"
    if not weights_path.is_file():
        return None
    cfg = MTPConfig.from_model_dir(mdir)
    drafter = InklingMTPDrafter(cfg)
    raw_weights = mx.load(str(weights_path))
    drafter.update(tree_unflatten(list(raw_weights.items())))
    mx.eval(drafter.parameters())
    drafter.bind(target_model)
    return drafter


def snapshot_rotating_kv(caches: list[CacheList], n_slots: int) -> list[Any]:
    """Zero-copy O(1) reference snapshot before a multi-token verify pass (L > 1 uses _update_concat)."""
    snaps: list[Any] = []
    for cl in caches:
        kv = cl.caches[0]
        if isinstance(kv, RotatingKVCache):
            snaps.append((kv.keys, kv.values, kv.offset, kv._idx))
        else:
            snaps.append(kv.offset)
    return snaps


def rollback_single_cache(
    caches: list[CacheList],
    snaps: list[Any],
    conv_states_per_visit: list[tuple[mx.array, mx.array, mx.array, mx.array]],
    n_accepted: int,
    n_verified: int,
) -> None:
    """Restore target KV and conv caches to exact state after `n_accepted` of `n_verified` tokens."""
    n_reject = n_verified - n_accepted
    for cl, snap, cstates in zip(caches, snaps, conv_states_per_visit):
        kv, conv = cl.caches[0], cl.caches[1]
        if n_reject > 0:
            if isinstance(kv, RotatingKVCache):
                old_k, old_v, old_off, old_idx = snap
                if old_k is None or old_off + n_verified <= kv.max_size:
                    keep = kv.keys.shape[2] - n_reject
                    kv.keys = kv.keys[..., :keep, :]
                    kv.values = kv.values[..., :keep, :]
                    kv.offset = old_off + n_accepted
                    kv._idx = keep
                else:
                    acc_k = kv.keys[..., -n_verified : -n_reject, :]
                    acc_v = kv.values[..., -n_verified : -n_reject, :]
                    kv.keys = old_k
                    kv.values = old_v
                    kv.offset = old_off
                    kv._idx = old_idx
                    kv.update_and_fetch(acc_k, acc_v)
            else:
                kv.offset = snap + n_accepted
        elif isinstance(kv, RotatingKVCache) and kv.keys is not None and kv.keys.shape[2] > kv.max_size:
            trim = kv.keys.shape[2] - kv.max_size
            kv.keys = kv.keys[..., trim:, :]
            kv.values = kv.values[..., trim:, :]
            kv._idx = kv.keys.shape[2]
        ks, vs, cs, ms = cstates
        pos = n_accepted - 1
        conv[0] = ks[pos : pos + 1]
        conv[1] = vs[pos : pos + 1]
        conv[2] = cs[pos : pos + 1]
        conv[3] = ms[pos : pos + 1]


def clone_draft_cache(draft_cache: list) -> list:
    """Deep-copy a 3-layer MTP draft KV cache for PrefixCache snapshots."""
    out = []
    for kv in draft_cache:
        if isinstance(kv, RotatingKVCache):
            nkv = RotatingKVCache(max_size=kv.max_size, keep=kv.keep)
            if kv.keys is not None:
                nkv.keys = mx.array(kv.keys)
                nkv.values = mx.array(kv.values)
            nkv.offset = kv.offset
            nkv._idx = kv._idx
        else:
            nkv = KVCache()
            if kv.keys is not None:
                nkv.keys = mx.contiguous(kv.keys[..., : kv.offset, :])
                nkv.values = mx.contiguous(kv.values[..., : kv.offset, :])
            nkv.offset = kv.offset
        out.append(nkv)
    return out


def eval_draft_cache(draft_cache: list) -> None:
    arrs = []
    for kv in draft_cache:
        if kv.keys is not None:
            arrs.append(kv.keys)
        if kv.values is not None:
            arrs.append(kv.values)
    if arrs:
        mx.eval(*arrs)


def _target_probs(logits_2d: mx.array, temperature: float, top_p: float) -> mx.array:
    """Convert target logits (M, V) into exact normalized probabilities matching `sample_token`."""
    scaled = logits_2d.astype(mx.float32) / float(temperature)
    probs = mx.softmax(scaled, axis=-1)
    if 0.0 < top_p < 1.0:
        s_idx = mx.argsort(-probs, axis=-1)
        s_p = mx.take_along_axis(probs, s_idx, axis=-1)
        cum = mx.cumsum(s_p, axis=-1)
        keep = (cum - s_p) < top_p
        masked_s = mx.where(keep, s_p, 0.0)
        masked_s = masked_s / mx.sum(masked_s, axis=-1, keepdims=True)
        inv = mx.argsort(s_idx, axis=-1)
        probs = mx.take_along_axis(masked_s, inv, axis=-1)
    return probs


def reject_sample_mtp(
    draft_toks: list[int],
    q_probs: mx.array,
    target_logits: mx.array,
    temperature: float,
    top_p: float,
) -> tuple[list[int], int]:
    """Exact lossless speculative rejection sampling (Leviathan et al.).

    Returns `(accepted_tokens, n_matched_drafts)` where `len(accepted_tokens) == n_matched_drafts + 1`.
    """
    k = len(draft_toks)
    p_probs = _target_probs(target_logits, temperature, top_p)
    u = mx.random.uniform(shape=(k,), dtype=mx.float32)
    mx.eval(p_probs, q_probs, u)
    u_list = u.tolist()

    accepted: list[int] = []
    for j in range(k):
        d = draft_toks[j]
        p_d = float(p_probs[j, d].item())
        q_d = float(q_probs[j, d].item())
        ratio = 1.0 if q_d <= 0.0 else min(1.0, p_d / q_d)
        if u_list[j] <= ratio:
            accepted.append(d)
        else:
            diff = mx.maximum(p_probs[j] - q_probs[j], 0.0)
            s = float(mx.sum(diff).item())
            if s > 1e-12:
                res = diff / s
                corr = int(mx.random.categorical(mx.log(mx.maximum(res, 1e-20))).item())
            else:
                corr = int(mx.random.categorical(mx.log(mx.maximum(p_probs[j], 1e-20))).item())
            accepted.append(corr)
            return accepted, j

    bonus = int(mx.random.categorical(mx.log(mx.maximum(p_probs[k], 1e-20))).item())
    accepted.append(bonus)
    return accepted, k
