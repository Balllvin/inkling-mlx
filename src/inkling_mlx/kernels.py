"""Metal inference kernels for Inkling 2 Loop (FP32-accurate prefill, decode, and BF16 head logits)."""

from __future__ import annotations

import math
import ctypes
from functools import lru_cache
from pathlib import Path

import mlx.core as mx
import numpy as np


@lru_cache(maxsize=1)
def _short_convolution():
    return mx.fast.metal_kernel(
        name="inkling2_virtual_short_convolution",
        input_names=["state", "weight", "x", "mask", "divisor"],
        output_names=["out"],
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
        source=r"""
        uint i = thread_position_in_grid.x;
        int B = x_shape[0], L = x_shape[1], C = x_shape[2];
        if (i >= B * L * C) return;
        int c = i % C, t = (i / C) % L, b = i / (C * L);
        float value = 0;
        for (int j = 0; j < W; ++j) {
            int p = t + j - W + 1;
            float xv = p < 0
                ? float(state[b * state_strides[0] + (p + W - 1) * state_strides[1]
                              + c * state_strides[2]])
                : ((HAS_MASK && !mask[b * mask_strides[0] + p * mask_strides[1]])
                    ? 0.0f
                    : float(x[b * x_strides[0] + p * x_strides[1] + c * x_strides[2]]));
            float w = weight[c * weight_strides[0] + j * weight_strides[1]];
            value = value + xv * w;
        }
        out[i] = T((value + float(x[b * x_strides[0] + t * x_strides[1]
                                   + c * x_strides[2]])) / divisor);
        """,
    )


def short_convolution(state: mx.array, weight: mx.array, x: mx.array, mask: mx.array | None, width: int) -> mx.array:
    """Depthwise causal conv with residual add and FP32 accumulation without FP32 input copy."""
    return _short_convolution()(
        inputs=[
            state,
            weight,
            x,
            mask if mask is not None else mx.zeros((1, 1), mx.bool_),
            mx.array(math.sqrt(width + 1), mx.float32),
        ],
        template=[("T", x.dtype), ("W", width), ("HAS_MASK", mask is not None)],
        grid=(x.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


@lru_cache(maxsize=1)
def _short_convolution_resid_f32():
    return mx.fast.metal_kernel(
        name="inkling2_short_conv_resid_f32",
        input_names=["state", "weight", "x", "resid", "mask", "divisor"],
        output_names=["out"],
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
        source=r"""
        uint i = thread_position_in_grid.x;
        int B = x_shape[0], L = x_shape[1], C = x_shape[2];
        if (i >= uint(B * L * C)) return;
        int c = int(i % uint(C));
        int t = int((i / uint(C)) % uint(L));
        int b = int(i / uint(C * L));
        float value = 0.0f;
        for (int j = 0; j < W; ++j) {
            int p = t + j - W + 1;
            float xv = p < 0
                ? float(state[b * state_strides[0] + (p + W - 1) * state_strides[1]
                              + c * state_strides[2]])
                : ((HAS_MASK && !mask[b * mask_strides[0] + p * mask_strides[1]])
                    ? 0.0f
                    : float(x[b * x_strides[0] + p * x_strides[1] + c * x_strides[2]]));
            float w = weight[c * weight_strides[0] + j * weight_strides[1]];
            value = value + xv * w;
        }
        float x0 = float(x[b * x_strides[0] + t * x_strides[1] + c * x_strides[2]]);
        float sc = float(T((value + x0) / divisor));
        out[i] = ADD_RESID ? (sc + float(resid[i])) : sc;
        """,
    )


def short_convolution_resid_f32(
    state: mx.array,
    weight: mx.array,
    x: mx.array,
    resid: mx.array | None,
    mask: mx.array | None,
    width: int,
) -> mx.array:
    """Fused ShortConvolution + optional residual add directly to FP32 [B*L, C] (bit-exact BF16 rounding)."""
    add_resid = resid is not None
    return _short_convolution_resid_f32()(
        inputs=[
            state,
            weight,
            x,
            resid.reshape(-1) if add_resid else mx.zeros((1,), x.dtype),
            mask if mask is not None else mx.zeros((1, 1), mx.bool_),
            mx.array(math.sqrt(width + 1), mx.float32),
        ],
        template=[
            ("T", x.dtype),
            ("W", width),
            ("HAS_MASK", mask is not None),
            ("ADD_RESID", add_resid),
        ],
        grid=(x.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(x.shape[0] * x.shape[1], x.shape[2])],
        output_dtypes=[mx.float32],
    )[0]


@lru_cache(maxsize=1)
def _dynamic_short_convolution():
    return mx.fast.metal_kernel(
        name="inkling2_dynamic_short_convolution",
        input_names=["state", "static_w", "dyn_w", "x", "mask", "divisor"],
        output_names=["out"],
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
        source=r"""
        uint i = thread_position_in_grid.x;
        int B = x_shape[0], L = x_shape[1], C = x_shape[2];
        if (i >= uint(B * L * C)) return;
        int c = int(i % uint(C));
        int t = int((i / uint(C)) % uint(L));
        int b = int(i / uint(C * L));
        int h = c / HD;
        float acc = 0.0f;
        for (int lag = 0; lag < W; ++lag) {
            int p = t - lag;
            float xv = p < 0
                ? float(state[b * state_strides[0] + (p + W - 1) * state_strides[1] + c * state_strides[2]])
                : ((HAS_MASK && !mask[b * mask_strides[0] + p * mask_strides[1]])
                    ? 0.0f
                    : float(x[b * x_strides[0] + p * x_strides[1] + c * x_strides[2]]));
            float sw = float(static_w[lag * static_w_strides[0] + c * static_w_strides[1]]);
            float dw = float(dyn_w[b * dyn_w_strides[0] + t * dyn_w_strides[1]
                                   + h * dyn_w_strides[2] + lag * dyn_w_strides[3]]);
            acc += xv * (sw + dw);
        }
        float x0 = float(x[b * x_strides[0] + t * x_strides[1] + c * x_strides[2]]);
        out[i] = T((acc + x0) / divisor);
        """,
    )


def dynamic_short_convolution(
    state: mx.array,
    static_weight: mx.array,
    dynamic_weights: mx.array,
    x: mx.array,
    mask: mx.array | None,
    width: int,
    head_dim: int,
) -> mx.array:
    """Fused DynamicShortConvolution with FP32 lag accumulation and zero full-sequence FP32 copy."""
    return _dynamic_short_convolution()(
        inputs=[
            state,
            static_weight,
            dynamic_weights,
            x,
            mask if mask is not None else mx.zeros((1, 1), mx.bool_),
            mx.array(math.sqrt(2 * width + 1), mx.float32),
        ],
        template=[("T", x.dtype), ("W", width), ("HD", head_dim), ("HAS_MASK", mask is not None)],
        grid=(x.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


@lru_cache(maxsize=1)
def _swiglu_interleaved_kernel():
    return mx.fast.metal_kernel(
        name="inkling2_swiglu_interleaved",
        input_names=["gu", "caps"],
        output_names=["out"],
        source=r"""
        uint i = thread_position_in_grid.x;
        uint n = gu_shape[0] * (gu_shape[1] / 2);
        if (i >= n) return;
        const float g = float(gu[size_t(i) * 2]);
        const float u = float(gu[size_t(i) * 2 + 1]);
        const float gc = caps[0], uc = caps[1];
        const float act = metal::precise::tanh(g / gc) * gc
            * (1.0f / (1.0f + metal::precise::exp(-g)))
            * metal::precise::tanh(u / uc) * uc;
        out[i] = T(act);
        """,
    )


_SWIGLU_CAPS: dict[tuple[float, float], mx.array] = {}


def swiglu_interleaved(gu: mx.array, gate_cap: float, up_cap: float) -> mx.array:
    """Contiguous single-pass interleaved softcapped SwiGLU for prefill MoE."""
    out_shape = (*gu.shape[:-1], gu.shape[-1] // 2)
    n_pairs = gu.size // 2
    key = (float(gate_cap), float(up_cap))
    caps = _SWIGLU_CAPS.get(key)
    if caps is None:
        caps = _SWIGLU_CAPS[key] = mx.array([gate_cap, up_cap], dtype=mx.float32)
    out = _swiglu_interleaved_kernel()(
        inputs=[mx.contiguous(gu.reshape(-1, gu.shape[-1])), caps],
        template=[("T", gu.dtype)],
        grid=(n_pairs, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n_pairs,)],
        output_dtypes=[gu.dtype],
    )[0]
    return out.reshape(out_shape)


_HEAD_SIMDGROUPS = 8


@lru_cache(maxsize=1)
def _head_logits():
    return mx.fast.metal_kernel(
        name="inkling2_bf16_head_fp32_logits",
        input_names=["h", "w"],
        output_names=["out"],
        source=r"""
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int row0 = (int(threadgroup_position_in_grid.x) * SG + int(sg)) * R;
        if (row0 >= N) return;
        float acc[R][M];
        for (int r = 0; r < R; ++r)
            for (int m = 0; m < M; ++m) acc[r][m] = 0.0f;
        for (int c = lane; c < K / 8; c += 32) {
            const int k = c * 8;
            float wv[R][8];
            for (int r = 0; r < R; ++r) {
                const int row = min(row0 + r, N - 1);
                auto wp = (const device vec<T, 4>*)(w + size_t(row) * K + k);
                const vec<T, 4> a = wp[0], b = wp[1];
                for (int j = 0; j < 4; ++j) {
                    wv[r][j] = float(a[j]);
                    wv[r][j + 4] = float(b[j]);
                }
            }
            for (int m = 0; m < M; ++m) {
                auto hp = (const device float4*)(h + size_t(m) * K + k);
                const float4 x0 = hp[0], x1 = hp[1];
                for (int r = 0; r < R; ++r) {
                    acc[r][m] += wv[r][0] * x0.x + wv[r][1] * x0.y + wv[r][2] * x0.z
                        + wv[r][3] * x0.w + wv[r][4] * x1.x + wv[r][5] * x1.y
                        + wv[r][6] * x1.z + wv[r][7] * x1.w;
                }
            }
        }
        for (int r = 0; r < R; ++r) {
            for (int m = 0; m < M; ++m) {
                const float total = simd_sum(acc[r][m]);
                if (lane == 0 && row0 + r < N) out[size_t(m) * N + row0 + r] = total;
            }
        }
        """,
    )


@lru_cache(maxsize=1)
def _head_logits_gemm():
    return mx.fast.metal_kernel(
        name="inkling2_bf16_head_fp32_gemm",
        input_names=["h", "w"],
        output_names=["out"],
        source=r"""
        const int n0 = int(threadgroup_position_in_grid.x) * BN;
        const uint tid = thread_position_in_threadgroup.x;
        const uint sg = simdgroup_index_in_threadgroup;
        constexpr int MT_SG = MT <= 4 ? MT : (MT + 1) / 2;
        constexpr int MT_A = MT <= 4 ? MT : MT_SG * 2;
        constexpr int NT_SG = MT <= 4 ? (BN / 64) : (BN / 32);
        const int sg_m = (MT <= 4) ? 0 : int(sg / 4);
        const int sg_n = (MT <= 4) ? int(sg) : int(sg % 4);
        constexpr int LDH = BK + 4;
        constexpr int LDW = BK + 4;
        threadgroup bfloat Hs[MT_A * 8 * LDH];
        threadgroup bfloat Ws[BN * LDW];

        simdgroup_float8x8 acc[MT_SG][NT_SG];
        for (int im = 0; im < MT_SG; ++im)
            for (int in_ = 0; in_ < NT_SG; ++in_)
                acc[im][in_] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

        for (int k0 = 0; k0 < K; k0 += BK) {
            for (int idx = int(tid); idx < (MT * 8) * (BK / 4); idx += 256) {
                const int r = idx / (BK / 4);
                const int c = (idx % (BK / 4)) * 4;
                vec<bfloat, 4> hv = (r < M) ? *(const device vec<bfloat, 4>*)(h + size_t(r) * K + k0 + c) : vec<bfloat, 4>(0);
                for (int j = 0; j < 4; ++j) Hs[r * LDH + c + j] = hv[j];
            }
            for (int idx = int(tid); idx < BN * (BK / 4); idx += 256) {
                const int r = idx / (BK / 4);
                const int c = (idx % (BK / 4)) * 4;
                const int gn = n0 + r;
                vec<bfloat, 4> wv = (gn < N) ? *(const device vec<bfloat, 4>*)(w + size_t(gn) * K + k0 + c) : vec<bfloat, 4>(0);
                for (int j = 0; j < 4; ++j) Ws[r * LDW + c + j] = wv[j];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int kk = 0; kk < BK; kk += 8) {
                simdgroup_bfloat8x8 a[MT_SG];
                for (int im = 0; im < MT_SG; ++im) {
                    const int m_off = (sg_m * MT_SG + im) * 8;
                    simdgroup_load(a[im], Hs + m_off * LDH + kk, LDH);
                }
                for (int in_ = 0; in_ < NT_SG; ++in_) {
                    simdgroup_bfloat8x8 b;
                    const int n_off = (sg_n * NT_SG + in_) * 8;
                    simdgroup_load(b, Ws + n_off * LDW + kk, LDW, ulong2(0, 0), true);
                    for (int im = 0; im < MT_SG; ++im) {
                        simdgroup_multiply_accumulate(acc[im][in_], a[im], b, acc[im][in_]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        threadgroup float OutS[MT_A * 8 * BN];
        for (int im = 0; im < MT_SG; ++im) {
            for (int in_ = 0; in_ < NT_SG; ++in_) {
                const int m_off = (sg_m * MT_SG + im) * 8;
                const int n_off = (sg_n * NT_SG + in_) * 8;
                if (m_off < M && n0 + n_off < N) {
                    simdgroup_store(acc[im][in_], OutS + m_off * BN + n_off, BN);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx = int(tid); idx < M * BN; idx += 256) {
            const int r = idx / BN;
            const int c = idx % BN;
            const int gn = n0 + c;
            if (gn < N) out[size_t(r) * N + gn] = OutS[r * BN + c];
        }
        """,
    )


@lru_cache(maxsize=1)
def _bf16_linear_32lane():
    """Zero-barrier 32-lane 512B-coalesced batch-invariant (inv_diff = 0.0 across M=1..64) BF16 linear."""
    return mx.fast.metal_kernel(
        name="inkling2_bf16_linear_32lane",
        input_names=["x", "w"],
        output_names=["out"],
        source=r"""
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int n0 = (int(threadgroup_position_in_grid.x) * SG + int(sg)) * R;
        const int t0 = int(threadgroup_position_in_grid.y) * MT;
        if (n0 >= N || t0 >= M) return;
        const int tn = min(MT, M - t0);

        constexpr int K8 = K / 8;
        float acc[R][MT];
        for (int ri = 0; ri < R; ++ri)
            for (int ti = 0; ti < MT; ++ti) acc[ri][ti] = 0.0f;

        for (int c = int(lane); c < K8; c += 32) {
            const int k = c * 8;
            if (R == 1) {
                auto wp = (const device vec<bfloat, 4>*)(w + size_t(n0) * K + k);
                const vec<bfloat, 4> wa = wp[0], wb = wp[1];
                const float w0 = float(wa[0]), w1 = float(wa[1]), w2 = float(wa[2]), w3 = float(wa[3]);
                const float w4 = float(wb[0]), w5 = float(wb[1]), w6 = float(wb[2]), w7 = float(wb[3]);
                for (int ti = 0; ti < tn; ++ti) {
                    auto xp = (const device vec<bfloat, 4>*)(x + size_t(t0 + ti) * K + k);
                    const vec<bfloat, 4> xa = xp[0], xb = xp[1];
                    acc[0][ti] += float(xa[0]) * w0 + float(xa[1]) * w1 + float(xa[2]) * w2 + float(xa[3]) * w3
                                + float(xb[0]) * w4 + float(xb[1]) * w5 + float(xb[2]) * w6 + float(xb[3]) * w7;
                }
            } else {
                float x0_buf[MT][4], x1_buf[MT][4];
                for (int ti = 0; ti < tn; ++ti) {
                    auto xp = (const device vec<bfloat, 4>*)(x + size_t(t0 + ti) * K + k);
                    const vec<bfloat, 4> xa = xp[0], xb = xp[1];
                    x0_buf[ti][0] = float(xa[0]); x0_buf[ti][1] = float(xa[1]);
                    x0_buf[ti][2] = float(xa[2]); x0_buf[ti][3] = float(xa[3]);
                    x1_buf[ti][0] = float(xb[0]); x1_buf[ti][1] = float(xb[1]);
                    x1_buf[ti][2] = float(xb[2]); x1_buf[ti][3] = float(xb[3]);
                }
                for (int ri = 0; ri < R; ++ri) {
                    const int n = min(n0 + ri, N - 1);
                    auto wp = (const device vec<bfloat, 4>*)(w + size_t(n) * K + k);
                    const vec<bfloat, 4> wa = wp[0], wb = wp[1];
                    const float w0 = float(wa[0]), w1 = float(wa[1]), w2 = float(wa[2]), w3 = float(wa[3]);
                    const float w4 = float(wb[0]), w5 = float(wb[1]), w6 = float(wb[2]), w7 = float(wb[3]);
                    for (int ti = 0; ti < tn; ++ti) {
                        acc[ri][ti] += x0_buf[ti][0] * w0 + x0_buf[ti][1] * w1 + x0_buf[ti][2] * w2 + x0_buf[ti][3] * w3
                                     + x1_buf[ti][0] * w4 + x1_buf[ti][1] * w5 + x1_buf[ti][2] * w6 + x1_buf[ti][3] * w7;
                    }
                }
            }
        }

        for (int ri = 0; ri < R; ++ri) {
            const int n = n0 + ri;
            if (n < N) {
                for (int ti = 0; ti < tn; ++ti) {
                    const float total = simd_sum(acc[ri][ti]);
                    if (lane == 0) {
                        out[size_t(t0 + ti) * N + n] = OUT_T(total);
                    }
                }
            }
        }
""",
    )


def bf16_linear(x: mx.array, weight: mx.array, *, out_dtype=None) -> mx.array:
    """Batch-invariant (0.0 diff across M=1..64) 32-lane 512B-coalesced BF16 linear y = x @ weight.T."""
    m, k = x.shape
    n = weight.shape[0]
    target_dtype = x.dtype if out_dtype is None else out_dtype
    if 1 <= m <= 64 and k % 256 == 0 and x.dtype == mx.bfloat16 and weight.dtype == mx.bfloat16:
        sg = 8 if n >= 2048 else 4
        if m == 1:
            r, mt = 2, 1
        elif m <= 4:
            r, mt = 2, 4
        elif m <= 8:
            r, mt = 1, 8
        else:
            r, mt = 1, 16
        nx = (n + sg * r - 1) // (sg * r)
        ny = (m + mt - 1) // mt
        return _bf16_linear_32lane()(
            inputs=[x, weight],
            template=[("OUT_T", target_dtype), ("M", m), ("K", k), ("N", n), ("MT", mt), ("R", r), ("SG", sg)],
            grid=(nx * sg * 32, ny, 1),
            threadgroup=(sg * 32, 1, 1),
            output_shapes=[(m, n)],
            output_dtypes=[target_dtype],
        )[0]
    out = x @ weight.T
    return out.astype(target_dtype) if out.dtype != target_dtype else out


@lru_cache(maxsize=1)
def _qkv_wide_32lane():
    """Single-launch 32-lane exact-order BF16 GEMV/GEMM for M=1..8 (0.0 diff M=1 vs M=2..8).

    Supports both shared input x (M, K) with S=1 and 3-way (xq, xk, xv) input (M, 3, K)
    with S=3 against packed c.stack (picking src=0 for row<AT3, 1 for AT3<=row<AT4, 2 else).
    """
    return mx.fast.metal_kernel(
        name="inkling2_qkv_wide_32lane",
        input_names=["x", "w"],
        output_names=["out"],
        source=r"""
        constexpr int UNROLL = 4;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int row = int(threadgroup_position_in_grid.y) * SG + int(sg);
        if (row >= N) return;

        const int src = (S == 1) ? 0 : (row < AT3 ? 0 : (row < AT4 ? 1 : 2));
        const device vec<bfloat, 4>* w4 = (const device vec<bfloat, 4>*)(w + size_t(row) * K);
        const device vec<bfloat, 4>* x4 = (const device vec<bfloat, 4>*)(x);
        constexpr int K4 = K / 4;
        constexpr int K_MAIN = K4 - (K4 % (32 * UNROLL));
        const int x_row_stride4 = (S * K) / 4;
        const int x_src_off4 = (src * K) / 4;

        float acc[M];
        for (int m = 0; m < M; ++m) acc[m] = 0.0f;

        for (int base = 0; base < K_MAIN; base += 32 * UNROLL) {
            vec<float, 4> wf[UNROLL];
            #pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                wf[u] = vec<float, 4>(w4[base + u * 32 + lane]);
            }
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                const int xoff = m * x_row_stride4 + x_src_off4 + base + int(lane);
                float a = 0.0f;
                #pragma unroll
                for (int u = 0; u < UNROLL; ++u) {
                    a += dot(wf[u], vec<float, 4>(x4[xoff + u * 32]));
                }
                acc[m] += a;
            }
        }
        for (int idx = K_MAIN + int(lane); idx < K4; idx += 32) {
            const vec<float, 4> wf = vec<float, 4>(w4[idx]);
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                acc[m] += dot(wf, vec<float, 4>(x4[m * x_row_stride4 + x_src_off4 + idx]));
            }
        }

        #pragma unroll
        for (int m = 0; m < M; ++m) {
            float v = simd_sum(acc[m]);
            if (lane == 0) out[size_t(m) * N + row] = bfloat(v);
        }
        """,
    )


def qkv_wide_linear(x: mx.array, weight: mx.array, at3: int = 0, at4: int = 0) -> mx.array:
    """Single-launch 32-lane BF16 linear for M=1..8 with exact M=1..8 invariance (diff=0.0)."""
    if x.ndim == 2:
        m, k = x.shape
        s = 1
    else:
        m, s, k = x.shape
    n = weight.shape[0]
    sg = 8 if n >= 4096 else 4
    ny = (n + sg - 1) // sg
    return _qkv_wide_32lane()(
        inputs=[x, weight],
        template=[("M", m), ("S", s), ("K", k), ("N", n), ("AT3", at3), ("AT4", at4), ("SG", sg)],
        grid=(32, ny * sg, 1),
        threadgroup=(32, sg, 1),
        output_shapes=[(m, n)],
        output_dtypes=[mx.bfloat16],
    )[0]


def head_logits(h: mx.array, weight: mx.array) -> mx.array:
    """Compute FP32 logits directly from BF16 head weight reading weight ONCE up to M=64."""
    m, k = h.shape
    n = weight.shape[0]
    if m > 8 and k % 32 == 0 and weight.dtype == mx.bfloat16:
        mt = (m + 7) // 8
        bn, bk = 64, 32
        nb = (n + bn - 1) // bn
        return _head_logits_gemm()(
            inputs=[mx.contiguous(h.astype(mx.bfloat16)), weight],
            template=[("M", m), ("K", k), ("N", n), ("MT", mt), ("BN", bn), ("BK", bk)],
            grid=(nb * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(m, n)],
            output_dtypes=[mx.float32],
        )[0]
    head_rows = 1
    rows = head_rows * _HEAD_SIMDGROUPS
    return _head_logits()(
        inputs=[mx.contiguous(h.astype(mx.float32)), weight],
        template=[
            ("T", weight.dtype),
            ("M", m),
            ("K", k),
            ("N", n),
            ("R", head_rows),
            ("SG", _HEAD_SIMDGROUPS),
        ],
        grid=((n + rows - 1) // rows * 32 * _HEAD_SIMDGROUPS, 1, 1),
        threadgroup=(32 * _HEAD_SIMDGROUPS, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[mx.float32],
    )[0]


@lru_cache(maxsize=1)
def _head_argmax_p1():
    return mx.fast.metal_kernel(
        name="inkling2_head_argmax_p1",
        input_names=["h", "w"],
        output_names=["part_val", "part_idx"],
        source=r"""
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int tg_id = int(threadgroup_position_in_grid.x);
        const int v_base = tg_id * (SG * R);

        float bv[M];
        uint bi[M];
        for (int m = 0; m < M; ++m) { bv[m] = -INFINITY; bi[m] = 0u; }

        for (int ri = 0; ri < R; ++ri) {
            const int v = v_base + int(sg) * R + ri;
            if (v < V_LIMIT) {
                const device vec<bfloat, 4>* w4 = (const device vec<bfloat, 4>*)(w + size_t(v) * K);
                const device vec<float, 4>* h4 = (const device vec<float, 4>*)(h);
                constexpr int K4 = K / 4;
                float acc[M];
                for (int m = 0; m < M; ++m) acc[m] = 0.0f;
                for (int idx = int(lane); idx < K4; idx += 32) {
                    const vec<float, 4> wf = vec<float, 4>(w4[idx]);
                    #pragma unroll
                    for (int m = 0; m < M; ++m) {
                        acc[m] += dot(wf, h4[size_t(m) * K4 + idx]);
                    }
                }
                for (int m = 0; m < M; ++m) {
                    const float tot = simd_sum(acc[m]);
                    if (lane == 0 && tot > bv[m]) {
                        bv[m] = tot;
                        bi[m] = uint(v);
                    }
                }
            }
        }

        threadgroup float sg_v[SG];
        threadgroup uint sg_i[SG];
        for (int m = 0; m < M; ++m) {
            if (lane == 0) { sg_v[sg] = bv[m]; sg_i[sg] = bi[m]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0 && lane == 0) {
                float fv = sg_v[0];
                uint fi = sg_i[0];
                for (int s = 1; s < SG; ++s) {
                    if (sg_v[s] > fv || (sg_v[s] == fv && sg_i[s] < fi)) {
                        fv = sg_v[s];
                        fi = sg_i[s];
                    }
                }
                part_val[size_t(m) * NTG + tg_id] = fv;
                part_idx[size_t(m) * NTG + tg_id] = fi;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        """,
    )


@lru_cache(maxsize=1)
def _head_argmax_p2():
    return mx.fast.metal_kernel(
        name="inkling2_head_argmax_p2",
        input_names=["part_val", "part_idx"],
        output_names=["out_tok"],
        source=r"""
        const uint tid = thread_position_in_threadgroup.x;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int m = int(threadgroup_position_in_grid.y);
        const int n = part_val_shape[1];
        const size_t base = size_t(m) * n;
        float bv = -INFINITY;
        uint bi = 0u;
        for (int i = int(tid); i < n; i += 256) {
            const float v = part_val[base + i];
            const uint idx = part_idx[base + i];
            if (v > bv || (v == bv && idx < bi)) { bv = v; bi = idx; }
        }
        for (ushort off = 16; off > 0; off >>= 1) {
            const float ov = simd_shuffle_xor(bv, off);
            const uint oi = simd_shuffle_xor(bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        threadgroup float sg_v[8];
        threadgroup uint sg_i[8];
        if (lane == 0) { sg_v[sg] = bv; sg_i[sg] = bi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0 && lane == 0) {
            float fv = sg_v[0]; uint fi = sg_i[0];
            for (int s = 1; s < 8; ++s) {
                if (sg_v[s] > fv || (sg_v[s] == fv && sg_i[s] < fi)) { fv = sg_v[s]; fi = sg_i[s]; }
            }
            out_tok[m] = int(fi);
        }
        """,
    )


def head_argmax(h: mx.array, weight: mx.array, v_limit: int) -> mx.array:
    """Fused FP32 lm_head matmul + argmax over [0, v_limit) for M=1..8 (skips 5MB logits & tanh!)."""
    m, k = h.shape
    sg = 8
    r = 4 if m == 1 else 2
    ntg = (v_limit + sg * r - 1) // (sg * r)
    pv, pi = _head_argmax_p1()(
        inputs=[mx.contiguous(h.astype(mx.float32)), weight],
        template=[("M", m), ("K", k), ("V_LIMIT", v_limit), ("R", r), ("SG", sg), ("NTG", ntg)],
        grid=(ntg * 32 * sg, 1, 1),
        threadgroup=(32 * sg, 1, 1),
        output_shapes=[(m, ntg), (m, ntg)],
        output_dtypes=[mx.float32, mx.uint32],
    )
    return _head_argmax_p2()(
        inputs=[pv, pi],
        grid=(256, m, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(m,)],
        output_dtypes=[mx.int32],
    )[0]


_DECODE_SIMDGROUPS = 8


@lru_cache(maxsize=1)
def _decode_partials():
    return mx.fast.metal_kernel(
        name="inkling2_decode_attention_partials_batched",
        input_names=["q", "k", "v", "rel", "key_start", "sizes", "scale"],
        output_names=["part_max", "part_sum", "part_acc"],
        ensure_row_contiguous=False,
        source=r"""
        constexpr int G = HQ / HKV;
        constexpr int E = D / 32;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int block = threadgroup_position_in_grid.x;
        const int g = threadgroup_position_in_grid.y;
        const int batch = threadgroup_position_in_grid.z;
        const int S = sizes[0];
        const int nblocks = sizes[1];
        const int ring_idx = sizes[2];
        const int n_q = sizes[3];
        const int k_batch = SEQ ? 0 : batch;
        const int last = SEQ ? (S - n_q + batch) : (S - 1);
        const int ks_raw = key_start[k_batch];
        const int ks = WINDOW > 0 ? max(ks_raw, max(0, last - WINDOW + 1)) : ks_raw;
        const int start = ks + block * BS;
        const int end = min(start + BS, last + 1);
        const int first = start;
        const float sc = scale;

        float qv[G][E];
        const size_t q_base = (size_t(batch) * HQ + g * G) * D + lane * E;
        for (int h = 0; h < G; ++h)
            for (int e = 0; e < E; ++e)
                qv[h][e] = float(q[q_base + size_t(h) * D + e]) * sc;

        float m[G], l[G], acc[G][E];
        for (int h = 0; h < G; ++h) {
            m[h] = -INFINITY;
            l[h] = 0.0f;
            for (int e = 0; e < E; ++e) acc[h][e] = 0.0f;
        }

        const device T* k_head = k + size_t(k_batch) * k_strides[0] + size_t(g) * k_strides[1];
        const device T* v_head = v + size_t(k_batch) * v_strides[0] + size_t(g) * v_strides[1];
        const size_t rel_base = (size_t(batch) * HQ + g * G) * REL;

        for (int j = first + int(sg); j < end; j += SG) {
            const int phys = (ring_idx == 0) ? j : ((j + ring_idx) % S);
            const device T* kp = k_head + size_t(phys) * k_strides[2] + lane * E;
            const device T* vp = v_head + size_t(phys) * v_strides[2] + lane * E;
            float kv[E], vv[E];
            for (int e = 0; e < E; ++e) {
                kv[e] = float(kp[e]);
                vv[e] = float(vp[e]);
            }
            const int dist = last - j;
            for (int h = 0; h < G; ++h) {
                float s = 0.0f;
                for (int e = 0; e < E; ++e) s += qv[h][e] * kv[e];
                s = simd_sum(s);
                if (dist < REL) s += float(rel[rel_base + size_t(h) * REL + dist]);
                const float mn = max(m[h], s);
                const float f = metal::exp(m[h] - mn);
                const float p = metal::exp(s - mn);
                l[h] = l[h] * f + p;
                for (int e = 0; e < E; ++e) acc[h][e] = acc[h][e] * f + p * vv[e];
                m[h] = mn;
            }
        }

        threadgroup float tm[SG][G];
        threadgroup float tl[SG][G];
        threadgroup float ta[SG][G][D];
        for (int h = 0; h < G; ++h) {
            if (lane == 0) {
                tm[sg][h] = m[h];
                tl[sg][h] = l[h];
            }
            for (int e = 0; e < E; ++e) ta[sg][h][lane * E + e] = acc[h][e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint t = thread_position_in_threadgroup.x; t < uint(G * D); t += SG * 32) {
            const int h = t / D, d = t % D;
            float M = -INFINITY;
            for (int s = 0; s < SG; ++s) M = max(M, tm[s][h]);
            float L = 0.0f, A = 0.0f;
            if (M > -INFINITY) {
                for (int s = 0; s < SG; ++s) {
                    const float w = metal::exp(tm[s][h] - M);
                    L += tl[s][h] * w;
                    A += ta[s][h][d] * w;
                }
            }
            const size_t row = (size_t(batch) * nblocks + block) * HQ + g * G + h;
            part_acc[row * D + d] = A;
            if (d == 0) {
                part_max[row] = M;
                part_sum[row] = L;
            }
        }
        """,
    )


@lru_cache(maxsize=1)
def _decode_combine():
    return mx.fast.metal_kernel(
        name="inkling2_decode_attention_combine_batched",
        input_names=["part_max", "part_sum", "part_acc", "sinks", "gate", "sizes"],
        output_names=["out"],
        source=r"""
        const int h = threadgroup_position_in_grid.x;
        const int batch = threadgroup_position_in_grid.y;
        const int d = thread_position_in_threadgroup.x;
        const int nblocks = sizes[1];
        const float sk = float(sinks[batch * HQ + h]);
        float M = sk;
        const size_t base = size_t(batch) * nblocks * HQ + h;
        for (int b = 0; b < nblocks; ++b) {
            M = max(M, part_max[base + size_t(b) * HQ]);
        }
        float L = metal::exp(sk - M), A = 0.0f;
        for (int b = 0; b < nblocks; ++b) {
            const size_t idx = base + size_t(b) * HQ;
            const float pm = part_max[idx];
            if (pm > -INFINITY) {
                const float w = metal::exp(pm - M);
                L += part_sum[idx] * w;
                A += part_acc[idx * D + d] * w;
            }
        }
        float v = float(T(A / L));
        if (HAS_GATE) {
            const float gv = float(gate[size_t(batch) * GW + GOFF + h * D + d]);
            v = v * (1.0f / (1.0f + metal::precise::exp(-gv)));
        }
        out[(size_t(batch) * HQ + h) * D + d] = T(v);
        """,
    )


def decode_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    rel: mx.array,
    sinks: mx.array,
    *,
    scale: float,
    key_start: mx.array | None = None,
    ring_idx: int = 0,
    window: int = 0,
    seq: bool = False,
    gate: mx.array | None = None,
    gate_at: int = 0,
) -> mx.array:
    """Split-K decode/verify attention over chronological or rotating KV cache (B>=1, or B=1 seq L<=8)."""
    b, hq, _, d = q.shape
    hkv, s = k.shape[1], k.shape[2]
    bs = 256 if s < 16384 else 512
    nblocks = (s + bs - 1) // bs
    kb = 1 if seq else b
    if key_start is None:
        key_start = mx.zeros((kb,), mx.int32)
    else:
        key_start = mx.contiguous(key_start.reshape(kb).astype(mx.int32))
    rel_c = mx.contiguous(rel.reshape(b, hq, -1))
    sinks_c = mx.contiguous(sinks.reshape(b, hq).astype(mx.float32))
    sizes = mx.array([s, nblocks, int(ring_idx) % s if s > 0 else 0, b], dtype=mx.int32)
    common = [("T", q.dtype), ("D", d), ("HQ", hq), ("HKV", hkv)]
    part_max, part_sum, part_acc = _decode_partials()(
        inputs=[
            mx.contiguous(q.reshape(b, hq, d)),
            k,
            v,
            rel_c,
            key_start,
            sizes,
            mx.array(scale, dtype=mx.float32),
        ],
        template=[*common, ("REL", rel_c.shape[-1]), ("BS", bs), ("SG", _DECODE_SIMDGROUPS), ("SEQ", seq), ("WINDOW", window)],
        grid=(nblocks * 32 * _DECODE_SIMDGROUPS, hkv, b),
        threadgroup=(32 * _DECODE_SIMDGROUPS, 1, 1),
        output_shapes=[(b, nblocks, hq), (b, nblocks, hq), (b, nblocks, hq, d)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    has_gate = gate is not None
    gw = gate.shape[-1] if has_gate else 1
    out = _decode_combine()(
        inputs=[
            part_max,
            part_sum,
            part_acc,
            sinks_c,
            gate if has_gate else mx.zeros((1,), dtype=q.dtype),
            sizes,
        ],
        template=[*common, ("HAS_GATE", has_gate), ("GW", gw), ("GOFF", gate_at if has_gate else 0)],
        grid=(hq * d, b, 1),
        threadgroup=(d, 1, 1),
        output_shapes=[(b, hq, d)],
        output_dtypes=[q.dtype],
    )[0]
    if has_gate:
        return out.reshape(b, hq * d)
    return out.reshape(b, hq, 1, d)


_BQ = 32
_TG = 128

_RELATIVE_ATTENTION_SOURCE = """
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr int BK = 16;
    constexpr int LDK = BK + 8;
    constexpr int LDV = D + 8;
    const int L = sizes[0];
    const int S = sizes[1];
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int batch = threadgroup_position_in_grid.z;
    const int head = threadgroup_position_in_grid.y;
    const int kv_head = head / (HQ / HKV);
    const int q0 = threadgroup_position_in_grid.x * BQ;
    const int start = key_start[batch];
    const float qscale = scale * LOG2E;

    const short fm = ((lane / 4) & 4) + (lane / 2) % 4;
    const short fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    const int row = q0 + int(sg) * 8 + fm;
    const bool row_ok = row < L;
    const int diag = S - L + row;

    threadgroup bfloat Ks[D * LDK];
    threadgroup bfloat Vs[BK * LDV];

    simdgroup_bfloat8x8 q_frag[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        bfloat q0v = bfloat(0.0f), q1v = bfloat(0.0f);
        if (row_ok) {
            const device T* qp = q + batch * q_strides[0] + head * q_strides[1]
                + row * q_strides[2] + db * 8 + fn;
            q0v = bfloat(float(qp[0]));
            q1v = bfloat(float(qp[1]));
        }
        q_frag[db].thread_elements()[0] = q0v;
        q_frag[db].thread_elements()[1] = q1v;
    }
    simdgroup_float8x8 acc[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        acc[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float row_max = row_ok ? float(sinks[(batch * L + row) * HQ + head]) * LOG2E : 0.0f;
    float row_sum = 1.0f;
    const int bias_row = batch * L + (row_ok ? row : 0);
    const device T* bias = rel + (size_t(bias_row) * HQ + head) * R;

    const int first_diag = S - L + q0;
    const int last_diag = S - L + min(q0 + BQ, L) - 1;
    int first_key = WINDOW > 0 ? max(0, first_diag - WINDOW + 1) : 0;
    first_key = max(first_key, start);
    const device T* k_head = k + batch * k_strides[0] + kv_head * k_strides[1];
    const device T* v_head = v + batch * v_strides[0] + kv_head * v_strides[1];

    for (int k0 = first_key / BK * BK; k0 <= last_diag; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int c = tid; c < BK * D / 8; c += TG) {
            const int r = c / (D / 8);
            const int col = (c % (D / 8)) * 8;
            const int key = k0 + r;
            vec<T, 4> kv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
            vec<T, 4> vv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
            if (key < S) {
                auto kp = (const device vec<T, 4>*)(k_head + key * k_strides[2] + col);
                auto vp = (const device vec<T, 4>*)(v_head + key * v_strides[2] + col);
                kv[0] = kp[0];
                kv[1] = kp[1];
                vv[0] = vp[0];
                vv[1] = vp[1];
            }
            for (int j = 0; j < 8; ++j) {
                Ks[(col + j) * LDK + r] = bfloat(float(kv[j / 4][j % 4]));
                Vs[r * LDV + col + j] = bfloat(float(vv[j / 4][j % 4]));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (q0 + int(sg) * 8 >= L) {
            continue;
        }

        simdgroup_float8x8 scores[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            scores[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int db = 0; db < D / 8; ++db) {
                simdgroup_bfloat8x8 k_frag;
                simdgroup_load(k_frag, Ks + db * 8 * LDK + kb * 8, LDK);
                simdgroup_multiply_accumulate(scores[kb], q_frag[db], k_frag, scores[kb]);
            }
        }

        float tile_max = -INFINITY;
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const int key = k0 + kb * 8 + fn + e;
                const int d = diag - key;
                const bool visible = row_ok && d >= 0 && key >= start
                    && (WINDOW == 0 || d < WINDOW);
                float s = -INFINITY;
                if (visible) {
                    s = scores[kb].thread_elements()[e] * qscale;
                    if (d < R) {
                        s += float(bias[d]) * LOG2E;
                    }
                }
                scores[kb].thread_elements()[e] = s;
                tile_max = max(tile_max, s);
            }
        }
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 1));
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 8));
        const float new_max = max(row_max, tile_max);
        const float factor = metal::precise::exp2(row_max - new_max);
        float tile_sum = 0.0f;
        simdgroup_bfloat8x8 probs[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const float p = metal::precise::exp2(scores[kb].thread_elements()[e] - new_max);
                probs[kb].thread_elements()[e] = bfloat(p);
                tile_sum += p;
            }
        }
        tile_sum += simd_shuffle_xor(tile_sum, 1);
        tile_sum += simd_shuffle_xor(tile_sum, 8);
        row_sum = row_sum * factor + tile_sum;
        row_max = new_max;
        for (int db = 0; db < D / 8; ++db) {
            acc[db].thread_elements()[0] *= factor;
            acc[db].thread_elements()[1] *= factor;
            for (int kb = 0; kb < BK / 8; ++kb) {
                simdgroup_bfloat8x8 v_frag;
                simdgroup_load(v_frag, Vs + kb * 8 * LDV + db * 8, LDV);
                simdgroup_multiply_accumulate(acc[db], probs[kb], v_frag, acc[db]);
            }
        }
    }

    if (row_ok) {
        device T* op = out + (size_t(batch * L + row) * HQ + head) * D;
        for (int db = 0; db < D / 8; ++db) {
            op[db * 8 + fn] = T(acc[db].thread_elements()[0] / row_sum);
            op[db * 8 + fn + 1] = T(acc[db].thread_elements()[1] / row_sum);
        }
    }
"""

_RELATIVE_ATTENTION_FUSED_REL_SOURCE = """
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr int BK = 32;
    constexpr int LDK = BK + 8;
    constexpr int LDV = D + 8;
    constexpr int G = HQ / HKV;
    constexpr int Q_SG = BQ / 8;
    const int L = sizes[0];
    const int S = sizes[1];
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int batch = threadgroup_position_in_grid.z;
    const int kv_head = threadgroup_position_in_grid.y;
    const int q0 = threadgroup_position_in_grid.x * BQ;
    const int start = key_start[batch];
    const float qscale = scale * LOG2E;

    const int gh = int(sg) / Q_SG;
    const int q_sg = int(sg) % Q_SG;
    const int head = kv_head * G + gh;

    const short fm = ((lane / 4) & 4) + (lane / 2) % 4;
    const short fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    const int row_local = q_sg * 8 + fm;
    const int row = q0 + row_local;
    const bool row_ok = row < L;
    const int diag = S - L + row;

    threadgroup bfloat Ks[D * LDK];
    threadgroup bfloat Vs[BK * LDV];
    threadgroup bfloat Rs[G * BQ * 16];

    for (int idx = int(tid); idx < G * BQ * 4; idx += TG) {
        const int h_l = idx / (BQ * 4);
        const int r_l = (idx / 4) % BQ;
        const int c4 = (idx % 4) * 4;
        const int gr = q0 + r_l;
        vec<T, 4> rv4 = vec<T, 4>(0);
        if (gr < L) {
            const device T* rp = r_vec + (size_t(batch * L + gr) * HQ + (kv_head * G + h_l)) * 16 + c4;
            rv4 = *(const device vec<T, 4>*)rp;
        }
        const int base = (h_l * BQ + r_l) * 16 + c4;
        Rs[base] = bfloat(rv4[0]);
        Rs[base + 1] = bfloat(rv4[1]);
        Rs[base + 2] = bfloat(rv4[2]);
        Rs[base + 3] = bfloat(rv4[3]);
    }

    simdgroup_bfloat8x8 q_frag[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        bfloat q0v = bfloat(0.0f), q1v = bfloat(0.0f);
        if (row_ok) {
            const device T* qp = q + batch * q_strides[0] + head * q_strides[1]
                + row * q_strides[2] + db * 8 + fn;
            q0v = bfloat(qp[0]);
            q1v = bfloat(qp[1]);
        }
        q_frag[db].thread_elements()[0] = q0v;
        q_frag[db].thread_elements()[1] = q1v;
    }

    simdgroup_float8x8 acc[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        acc[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float row_max = row_ok ? float(sinks[(batch * L + row) * HQ + head]) * LOG2E : 0.0f;
    float row_sum = 1.0f;

    const int first_diag = S - L + q0;
    const int last_diag = S - L + min(q0 + BQ, L) - 1;
    int first_key = WINDOW > 0 ? max(0, first_diag - WINDOW + 1) : 0;
    first_key = max(first_key, start);
    const device T* k_head = k + batch * k_strides[0] + kv_head * k_strides[1];
    const device T* v_head = v + batch * v_strides[0] + kv_head * v_strides[1];
    const threadgroup bfloat* r_row = Rs + (gh * BQ + row_local) * 16;

    for (int k0 = first_key / BK * BK; k0 <= last_diag; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int c = int(tid); c < BK * D / 8; c += TG) {
            const int r = c / (D / 8);
            const int col = (c % (D / 8)) * 8;
            const int key = k0 + r;
            vec<T, 4> kv0 = vec<T, 4>(0), kv1 = vec<T, 4>(0);
            vec<T, 4> vv0 = vec<T, 4>(0), vv1 = vec<T, 4>(0);
            if (key < S) {
                auto kp = (const device vec<T, 4>*)(k_head + key * k_strides[2] + col);
                auto vp = (const device vec<T, 4>*)(v_head + key * v_strides[2] + col);
                kv0 = kp[0]; kv1 = kp[1];
                vv0 = vp[0]; vv1 = vp[1];
            }
            for (int j = 0; j < 4; ++j) {
                Ks[(col + j) * LDK + r] = bfloat(kv0[j]);
                Ks[(col + 4 + j) * LDK + r] = bfloat(kv1[j]);
                Vs[r * LDV + col + j] = bfloat(vv0[j]);
                Vs[r * LDV + col + 4 + j] = bfloat(vv1[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (q0 + q_sg * 8 >= L) {
            continue;
        }

        simdgroup_float8x8 scores[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            scores[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int db = 0; db < D / 8; ++db) {
                simdgroup_bfloat8x8 k_frag;
                simdgroup_load(k_frag, Ks + db * 8 * LDK + kb * 8, LDK);
                simdgroup_multiply_accumulate(scores[kb], q_frag[db], k_frag, scores[kb]);
            }
        }

        const bool tile_has_rel = (first_diag - (k0 + BK - 1)) < R;
        const bool tile_full_vis = row_ok && (k0 >= start) && (k0 + BK - 1 <= diag)
            && (WINDOW == 0 || (diag - k0) < WINDOW);
        float tile_max = -INFINITY;
        if (tile_full_vis && !tile_has_rel) {
            for (int kb = 0; kb < BK / 8; ++kb) {
                for (int e = 0; e < 2; ++e) {
                    const float s = scores[kb].thread_elements()[e] * qscale;
                    scores[kb].thread_elements()[e] = s;
                    tile_max = max(tile_max, s);
                }
            }
        } else {
            for (int kb = 0; kb < BK / 8; ++kb) {
                for (int e = 0; e < 2; ++e) {
                    const int key = k0 + kb * 8 + fn + e;
                    const int d = diag - key;
                    const bool visible = row_ok && d >= 0 && key >= start
                        && (WINDOW == 0 || d < WINDOW);
                    float s = -INFINITY;
                    if (visible) {
                        s = scores[kb].thread_elements()[e] * qscale;
                        if (tile_has_rel && d < R) {
                            auto pp = (const device vec<T, 4>*)(rel_proj_t + size_t(d) * 16);
                            const vec<T, 4> p0 = pp[0], p1 = pp[1], p2 = pp[2], p3 = pp[3];
                            const float b = float(r_row[0])*float(p0[0]) + float(r_row[1])*float(p0[1])
                                          + float(r_row[2])*float(p0[2]) + float(r_row[3])*float(p0[3])
                                          + float(r_row[4])*float(p1[0]) + float(r_row[5])*float(p1[1])
                                          + float(r_row[6])*float(p1[2]) + float(r_row[7])*float(p1[3])
                                          + float(r_row[8])*float(p2[0]) + float(r_row[9])*float(p2[1])
                                          + float(r_row[10])*float(p2[2]) + float(r_row[11])*float(p2[3])
                                          + float(r_row[12])*float(p3[0]) + float(r_row[13])*float(p3[1])
                                          + float(r_row[14])*float(p3[2]) + float(r_row[15])*float(p3[3]);
                            s += float(T(b)) * LOG2E;
                        }
                    }
                    scores[kb].thread_elements()[e] = s;
                    tile_max = max(tile_max, s);
                }
            }
        }
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 1));
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 8));
        const float new_max = max(row_max, tile_max);
        const float factor = metal::precise::exp2(row_max - new_max);
        float tile_sum = 0.0f;
        simdgroup_bfloat8x8 probs[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const float p = metal::precise::exp2(scores[kb].thread_elements()[e] - new_max);
                probs[kb].thread_elements()[e] = bfloat(p);
                tile_sum += p;
            }
        }
        tile_sum += simd_shuffle_xor(tile_sum, 1);
        tile_sum += simd_shuffle_xor(tile_sum, 8);
        row_sum = row_sum * factor + tile_sum;
        row_max = new_max;
        for (int db = 0; db < D / 8; ++db) {
            acc[db].thread_elements()[0] *= factor;
            acc[db].thread_elements()[1] *= factor;
            for (int kb = 0; kb < BK / 8; ++kb) {
                simdgroup_bfloat8x8 v_frag;
                simdgroup_load(v_frag, Vs + kb * 8 * LDV + db * 8, LDV);
                simdgroup_multiply_accumulate(acc[db], probs[kb], v_frag, acc[db]);
            }
        }
    }

    if (row_ok) {
        device T* op = out + (size_t(batch * L + row) * HQ + head) * D;
        for (int db = 0; db < D / 8; ++db) {
            op[db * 8 + fn] = T(acc[db].thread_elements()[0] / row_sum);
            op[db * 8 + fn + 1] = T(acc[db].thread_elements()[1] / row_sum);
        }
    }
"""

_relative_attention_fused_rel_kernel = mx.fast.metal_kernel(
    name="inkling2_relative_attention_fused_rel_gqa",
    input_names=["q", "k", "v", "r_vec", "rel_proj_t", "sinks", "key_start", "sizes", "scale"],
    output_names=["out"],
    source=_RELATIVE_ATTENTION_FUSED_REL_SOURCE,
    ensure_row_contiguous=False,
)


def relative_attention_fused_rel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    r_vec: mx.array,
    rel_proj_t: mx.array,
    sinks: mx.array,
    *,
    scale: float,
    window: int,
    key_start: mx.array,
) -> mx.array:
    """Shared-KV GQA (BQ=64, BK=32) causal flash attention fusing r @ rel_proj with zero register spill."""
    B, Hq, L, D = q.shape
    Hkv, S = k.shape[1], k.shape[2]
    R = rel_proj_t.shape[0]
    g = Hq // Hkv
    q_sg = 8 if g <= 3 else max(1, 32 // g)
    bq = q_sg * 8
    tg = g * q_sg * 32
    return _relative_attention_fused_rel_kernel(
        inputs=[
            q,
            k,
            v,
            mx.contiguous(r_vec),
            rel_proj_t,
            mx.contiguous(sinks.astype(mx.float32)),
            mx.contiguous(key_start.astype(mx.int32)),
            mx.array([L, S], dtype=mx.int32),
            mx.array(scale, dtype=mx.float32),
        ],
        template=[
            ("T", q.dtype),
            ("D", D),
            ("HQ", Hq),
            ("HKV", Hkv),
            ("R", R),
            ("WINDOW", window),
            ("BQ", bq),
            ("TG", tg),
        ],
        grid=((L + bq - 1) // bq * tg, Hkv, B),
        threadgroup=(tg, 1, 1),
        output_shapes=[(B, L, Hq, D)],
        output_dtypes=[q.dtype],
    )[0]


_BUILD_REL_MASK_FAST_SRC = r"""
    const int k = int(thread_position_in_grid.x);     // 0..S-1
    const int q_idx = int(thread_position_in_grid.y); // 0..QL-1
    const int bh = int(thread_position_in_grid.z);    // batch * HQ + head
    const int QL = sizes[0], S = sizes[1], diag_base = sizes[2];
    if (k >= S || q_idx >= QL) return;
    const int batch = bh / HQ;
    const int head = bh % HQ;
    const size_t out_i = (size_t(bh) * QL + q_idx) * S + k;
    const int diag = diag_base + q_idx;
    const int d = diag - k;
    const int start = key_start[batch];
    if (d < 0 || k < start || (WINDOW > 0 && d >= WINDOW)) {
        mask[out_i] = T(-INFINITY);
        return;
    }
    const float sk = sinks[(size_t(batch) * QL + q_idx) * HQ + head];
    if (d >= R) {
        mask[out_i] = T(-sk);
        return;
    }
    const float rb = float(rel_tbl[(size_t(batch * QL + q_idx) * HQ + head) * R + d]);
    mask[out_i] = T(rb - sk);
"""

_build_rel_mask_fast_kernel = mx.fast.metal_kernel(
    name="inkling2_build_rel_mask_fast",
    input_names=["rel_tbl", "sinks", "key_start", "sizes"],
    output_names=["mask"],
    source=_BUILD_REL_MASK_FAST_SRC,
)


class _FlashAttnDesc(ctypes.Structure):
    _fields_ = [
        ("B", ctypes.c_int32),
        ("HQ", ctypes.c_int32),
        ("HKV", ctypes.c_int32),
        ("L", ctypes.c_int32),
        ("S", ctypes.c_int32),
        ("D", ctypes.c_int32),
        ("R", ctypes.c_int32),
        ("window", ctypes.c_int32),
        ("scale", ctypes.c_float),
        ("q_s0", ctypes.c_int64),
        ("q_s1", ctypes.c_int64),
        ("q_s2", ctypes.c_int64),
        ("k_s0", ctypes.c_int64),
        ("k_s1", ctypes.c_int64),
        ("k_s2", ctypes.c_int64),
        ("v_s0", ctypes.c_int64),
        ("v_s1", ctypes.c_int64),
        ("v_s2", ctypes.c_int64),
        ("sinks_is_bf16", ctypes.c_int32),
        ("has_gate", ctypes.c_int32),
        ("gate_off", ctypes.c_int32),
        ("gate_stride", ctypes.c_int32),
        ("variant", ctypes.c_int32),
    ]


_FLASH_ATTN_LIB = None
_FLASH_ATTN_TRIED = False


def _get_flash_attn_lib():
    global _FLASH_ATTN_LIB, _FLASH_ATTN_TRIED
    if not _FLASH_ATTN_TRIED:
        _FLASH_ATTN_TRIED = True
        lib_path = Path(__file__).resolve().parent / "native" / "libinkling_flash_attn.dylib"
        if lib_path.is_file():
            try:
                lib = ctypes.CDLL(str(lib_path))
                lib.inkling_flash_attn_init.restype = ctypes.c_int
                lib.inkling_flash_attn_nax.argtypes = [
                    ctypes.POINTER(_FlashAttnDesc),
                    ctypes.c_uint64, ctypes.c_size_t,
                    ctypes.c_uint64, ctypes.c_size_t,
                    ctypes.c_uint64, ctypes.c_size_t,
                    ctypes.c_uint64, ctypes.c_size_t,
                    ctypes.c_uint64, ctypes.c_size_t,
                    ctypes.c_uint64, ctypes.c_size_t,
                    ctypes.c_uint64, ctypes.c_size_t,
                    ctypes.c_uint64, ctypes.c_size_t,
                    ctypes.c_uint64, ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_double),
                ]
                lib.inkling_flash_attn_nax.restype = ctypes.c_int
                if lib.inkling_flash_attn_init() == 0:
                    _FLASH_ATTN_LIB = lib
            except Exception:
                _FLASH_ATTN_LIB = None
    return _FLASH_ATTN_LIB


def _np_ptr_span_strides(arr: mx.array):
    view = arr.view(mx.uint16) if arr.dtype == mx.bfloat16 else arr
    nv = np.asarray(view)
    item = nv.itemsize
    strides = tuple(s // item for s in nv.strides)
    if nv.size == 0:
        span = 0
    else:
        span = sum((d - 1) * s for d, s in zip(nv.shape, nv.strides) if d > 0) + item
    return nv, int(nv.ctypes.data), int(span), strides


_FLASH_ATTN_OUT_SCRATCH: dict[tuple[int, int, int, int], tuple[mx.array, np.ndarray, int, int]] = {}


def _get_flash_attn_out(B: int, L: int, HQ: int, D: int):
    key = (B, L, HQ, D)
    entry = _FLASH_ATTN_OUT_SCRATCH.get(key)
    if entry is None:
        if len(_FLASH_ATTN_OUT_SCRATCH) >= 4:
            _FLASH_ATTN_OUT_SCRATCH.clear()
        out = mx.zeros((B, L, HQ, D), dtype=mx.bfloat16)
        mx.eval(out)
        vo, op, on, _ = _np_ptr_span_strides(out)
        entry = (out, vo, op, on)
        _FLASH_ATTN_OUT_SCRATCH[key] = entry
    return entry


def relative_attention_nax_native(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    r_vec: mx.array,
    rel_proj_t: mx.array,
    sinks: mx.array,
    *,
    scale: float,
    window: int,
    key_start: mx.array,
    gate_proj: mx.array | None = None,
    gate_off: int = 0,
    gate_stride: int = 0,
) -> mx.array | None:
    """Zero-mask Metal 4.0 NAX FlashAttention with fused NAX Toeplitz rel bias, sinks & optional gate."""
    lib = _get_flash_attn_lib()
    if lib is None or q.dtype != mx.bfloat16 or q.shape[-1] != 128 or r_vec.shape[-1] != 16:
        return None
    B, HQ, L, D = q.shape
    HKV, S = k.shape[1], k.shape[2]
    R = rel_proj_t.shape[0]
    ks = mx.contiguous(key_start.astype(mx.int32))
    if window > 0 and S > L + window - 1:
        k_off = S - L - window + 1
        k_use = k[:, :, k_off:, :]
        v_use = v[:, :, k_off:, :]
        s_use = S - k_off
        ks = mx.maximum(ks - k_off, 0)
    else:
        k_use, v_use, s_use = k, v, S

    r_c = mx.contiguous(r_vec)
    sinks_c = mx.contiguous(sinks)
    to_eval = [q, k_use, v_use, r_c, rel_proj_t, sinks_c, ks]
    if gate_proj is not None:
        to_eval.append(gate_proj)
    mx.eval(*to_eval)
    out, vo, op, on = _get_flash_attn_out(B, L, HQ, D)

    vq, qp, qn, qs = _np_ptr_span_strides(q)
    vk, kp, kn, kst = _np_ptr_span_strides(k_use)
    vv, vp, vn, vst = _np_ptr_span_strides(v_use)
    vr, rp, rn, _ = _np_ptr_span_strides(r_c)
    vpt, ptp, ptn, _ = _np_ptr_span_strides(rel_proj_t)
    vsk, skp, skn, _ = _np_ptr_span_strides(sinks_c)
    vks, ksp, ksn, _ = _np_ptr_span_strides(ks)
    if gate_proj is not None:
        vg, gp, gn, _ = _np_ptr_span_strides(gate_proj)
    else:
        vg, gp, gn = None, 0, 0

    desc = _FlashAttnDesc(
        B=B, HQ=HQ, HKV=HKV, L=L, S=s_use, D=D, R=R, window=window, scale=float(scale),
        q_s0=qs[0], q_s1=qs[1], q_s2=qs[2],
        k_s0=kst[0], k_s1=kst[1], k_s2=kst[2],
        v_s0=vst[0], v_s1=vst[1], v_s2=vst[2],
        sinks_is_bf16=1 if sinks_c.dtype == mx.bfloat16 else 0,
        has_gate=1 if gate_proj is not None else 0,
        gate_off=gate_off,
        gate_stride=gate_stride,
        variant=0,
    )
    rc = lib.inkling_flash_attn_nax(
        ctypes.byref(desc),
        qp, qn, kp, kn, vp, vn, rp, rn, ptp, ptn, skp, skn, ksp, ksn, gp, gn, op, on, None,
    )
    _ = (vq, vk, vv, vr, vpt, vsk, vks, vo, vg)
    if rc != 0:
        return None
    return out


def relative_attention_nax(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    r_vec: mx.array,
    rel_proj_t: mx.array,
    sinks: mx.array,
    *,
    scale: float,
    window: int,
    key_start: mx.array,
    rel_proj: mx.array | None = None,
) -> mx.array:
    """M5 NAX SDPA with exact sink-shift identity (zero KV concat), NAX rel GEMM, and SWA q_blk=256 tiling."""
    B, Hq, L, D = q.shape
    Hkv, S = k.shape[1], k.shape[2]
    R = rel_proj_t.shape[0]
    ks = mx.contiguous(key_start.astype(mx.int32))

    if window > 0 and S > L + window - 1:
        k_off = S - L - window + 1
        k_use = k[:, :, k_off:, :]
        v_use = v[:, :, k_off:, :]
        s_use = S - k_off
        ks_use = mx.maximum(ks - k_off, 0)
    else:
        k_use, v_use, s_use, ks_use = k, v, S, ks

    if s_use > 32768:
        return relative_attention_fused_rel(
            q, k_use, v_use, r_vec, rel_proj_t, sinks, scale=scale, window=window, key_start=ks_use
        )

    rp = rel_proj if rel_proj is not None else rel_proj_t.T
    rel_tbl = (r_vec.reshape(B * L * Hq, r_vec.shape[-1]) @ rp).reshape(B, L, Hq, R)
    sinks_f32 = mx.contiguous(sinks.astype(mx.float32))
    zero_sinks = mx.zeros((Hq,), dtype=q.dtype)

    if window > 0 and L > 256:
        q_blk = 256
        outs = []
        diag0 = s_use - L
        for q0 in range(0, L, q_blk):
            q1 = min(L, q0 + q_blk)
            ql = q1 - q0
            k_lo = max(0, diag0 + q0 - window + 1)
            k_hi = diag0 + q1
            s_tile = k_hi - k_lo
            ks_tile = mx.maximum(ks_use - k_lo, 0)
            mask = _build_rel_mask_fast_kernel(
                inputs=[
                    mx.contiguous(rel_tbl[:, q0:q1]),
                    mx.contiguous(sinks_f32[:, q0:q1]),
                    ks_tile,
                    mx.array([ql, s_tile, diag0 + q0 - k_lo], dtype=mx.int32),
                ],
                template=[("T", q.dtype), ("HQ", Hq), ("R", R), ("WINDOW", window)],
                grid=(s_tile, ql, B * Hq),
                threadgroup=(min(256, s_tile), 1, 1),
                output_shapes=[(B, Hq, ql, s_tile)],
                output_dtypes=[q.dtype],
            )[0]
            outs.append(
                mx.fast.scaled_dot_product_attention(
                    q[:, :, q0:q1, :],
                    k_use[:, :, k_lo:k_hi, :],
                    v_use[:, :, k_lo:k_hi, :],
                    scale=scale,
                    mask=mask,
                    sinks=zero_sinks,
                ).transpose(0, 2, 1, 3)
            )
        return mx.concatenate(outs, axis=1)

    max_elems = 128 * 1024 * 1024
    q_blk = max(64, min(L, max_elems // max(1, B * Hq * s_use)))
    if q_blk >= L:
        mask = _build_rel_mask_fast_kernel(
            inputs=[
                rel_tbl,
                sinks_f32,
                ks_use,
                mx.array([L, s_use, s_use - L], dtype=mx.int32),
            ],
            template=[("T", q.dtype), ("HQ", Hq), ("R", R), ("WINDOW", window)],
            grid=(s_use, L, B * Hq),
            threadgroup=(min(256, s_use), 1, 1),
            output_shapes=[(B, Hq, L, s_use)],
            output_dtypes=[q.dtype],
        )[0]
        return mx.fast.scaled_dot_product_attention(
            q, k_use, v_use, scale=scale, mask=mask, sinks=zero_sinks
        ).transpose(0, 2, 1, 3)

    outs = []
    for q0 in range(0, L, q_blk):
        q1 = min(L, q0 + q_blk)
        ql = q1 - q0
        k_hi = s_use - L + q1
        mask = _build_rel_mask_fast_kernel(
            inputs=[
                mx.contiguous(rel_tbl[:, q0:q1]),
                mx.contiguous(sinks_f32[:, q0:q1]),
                ks_use,
                mx.array([ql, k_hi, s_use - L + q0], dtype=mx.int32),
            ],
            template=[("T", q.dtype), ("HQ", Hq), ("R", R), ("WINDOW", window)],
            grid=(k_hi, ql, B * Hq),
            threadgroup=(min(256, k_hi), 1, 1),
            output_shapes=[(B, Hq, ql, k_hi)],
            output_dtypes=[q.dtype],
        )[0]
        outs.append(
            mx.fast.scaled_dot_product_attention(
                q[:, :, q0:q1, :],
                k_use[:, :, :k_hi, :],
                v_use[:, :, :k_hi, :],
                scale=scale,
                mask=mask,
                sinks=zero_sinks,
            ).transpose(0, 2, 1, 3)
        )
    return mx.concatenate(outs, axis=1)


_BF16_SIGMOID_MUL_SRC = r"""
    const int idx = int(thread_position_in_grid.x) * 4;
    const int row = int(thread_position_in_grid.y);
    if (idx >= D) return;
    const device vec<bfloat, 4>* op = (const device vec<bfloat, 4>*)(out_in + size_t(row) * D + idx);
    const device vec<bfloat, 4>* gp = (const device vec<bfloat, 4>*)(proj + size_t(row) * GSTRIDE + GOFF + idx);
    const vec<bfloat, 4> ov = op[0];
    const vec<bfloat, 4> gv = gp[0];
    vec<bfloat, 4> rv;
    rv[0] = bfloat(float(ov[0]) / (1.0f + metal::precise::exp(-float(gv[0]))));
    rv[1] = bfloat(float(ov[1]) / (1.0f + metal::precise::exp(-float(gv[1]))));
    rv[2] = bfloat(float(ov[2]) / (1.0f + metal::precise::exp(-float(gv[2]))));
    rv[3] = bfloat(float(ov[3]) / (1.0f + metal::precise::exp(-float(gv[3]))));
    *(device vec<bfloat, 4>*)(out_res + size_t(row) * D + idx) = rv;
"""

_bf16_sigmoid_mul_kernel = mx.fast.metal_kernel(
    name="inkling2_bf16_sigmoid_mul",
    input_names=["out_in", "proj"],
    output_names=["out_res"],
    source=_BF16_SIGMOID_MUL_SRC,
    ensure_row_contiguous=False,
)


def bf16_sigmoid_mul(out_in: mx.array, proj: mx.array, gate_off: int, gate_stride: int) -> mx.array:
    """Fused out * sigmoid(gate_raw) reading strided BF16 gate directly from proj (saves 4GB/1k)."""
    d = out_in.shape[-1]
    m = out_in.size // d
    return _bf16_sigmoid_mul_kernel(
        inputs=[mx.contiguous(out_in.reshape(m, d)), proj],
        template=[("D", d), ("GOFF", gate_off), ("GSTRIDE", gate_stride)],
        grid=(d // 4, m, 1),
        threadgroup=(min(256, d // 4), 1, 1),
        output_shapes=[out_in.shape],
        output_dtypes=[mx.bfloat16],
    )[0]


_PREFILL_QKVR_NORM_SRC = r"""
    const uint h = threadgroup_position_in_grid.x;
    const uint l = threadgroup_position_in_grid.y;
    const uint b = threadgroup_position_in_grid.z;
    const uint t = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const size_t row = size_t(b) * L + l;
    const float qk_thr = prm[0], rel_thr = prm[1], eps = prm[2];
    const float tau_ = HAS_TAU ? tau[row] : 1.0f;

    const float qy = float(proj[row * PSTRIDE + QOFF + h * HD + t]) * float(qscale[h * HD + t]);
    const float ry = (t < DR) ? (float(proj[row * PSTRIDE + ROFF + h * DR + t]) * float(rscale[h * DR + t])) : 0.0f;
    float ky = 0.0f, vy = 0.0f;
    if (h < HKV) {
        ky = float(k_in[row * (HKV * HD) + h * HD + t]);
        vy = float(v_in[row * (HKV * HD) + h * HD + t]);
    }

    threadgroup float sbuf[4][4];
    float q2 = simd_sum(qy * qy);
    float r2 = simd_sum(ry * ry);
    float k2 = simd_sum(ky * ky);
    float v2 = simd_sum(vy * vy);
    if (lane == 0) {
        sbuf[0][sg] = q2; sbuf[1][sg] = r2; sbuf[2][sg] = k2; sbuf[3][sg] = v2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float tot_q = sbuf[0][0] + sbuf[0][1] + sbuf[0][2] + sbuf[0][3];
    const float tot_r = sbuf[1][0];
    const float tot_k = sbuf[2][0] + sbuf[2][1] + sbuf[2][2] + sbuf[2][3];
    const float tot_v = sbuf[3][0] + sbuf[3][1] + sbuf[3][2] + sbuf[3][3];

    float qv = float(bfloat(qy * min(qk_thr / (metal::precise::sqrt(tot_q) + 1e-6f), 1.0f)));
    if (HAS_TAU) qv = float(bfloat(qv * tau_));
    q_out[(((size_t(b) * HQ + h) * L) + l) * HD + t] = bfloat(qv);

    if (t < DR) {
        float rv = float(bfloat(ry * min(rel_thr / (metal::precise::sqrt(tot_r) + 1e-6f), 1.0f)));
        if (HAS_TAU) rv = float(bfloat(rv * tau_));
        r_out[(row * HQ + h) * DR + t] = bfloat(rv);
    }
    if (h < HKV) {
        const size_t kvo = (((size_t(b) * HKV + h) * L) + l) * HD + t;
        k_out[kvo] = bfloat(ky * min(qk_thr / (metal::precise::sqrt(tot_k) + 1e-6f), 1.0f));
        v_out[kvo] = HAS_VNORM ? bfloat(vy * metal::precise::rsqrt(tot_v / float(HD) + eps)) : bfloat(vy);
    }
"""

_prefill_qkvr_norm_kernel = mx.fast.metal_kernel(
    name="inkling2_prefill_qkvr_norm",
    input_names=["proj", "k_in", "v_in", "qscale", "rscale", "tau", "prm"],
    output_names=["q_out", "k_out", "v_out", "r_out"],
    source=_PREFILL_QKVR_NORM_SRC,
    ensure_row_contiguous=False,
)


def prefill_qkvr_norm(
    proj: mx.array,
    q_off: int,
    r_off: int,
    k_in: mx.array,
    v_in: mx.array,
    q_scale: mx.array,
    rel_scale: mx.array,
    tau: mx.array | None,
    *,
    hq: int,
    hkv: int,
    hd: int,
    dr: int,
    qk_thr: float,
    rel_thr: float,
    eps: float,
    use_vnorm: bool = True,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Fused prefill l2_clip(q,k,r) + rms_norm(v) + tau + [B,H,L,D] transpose reading strided proj."""
    b, length = k_in.shape[:2]
    pstride = proj.shape[-1]
    has_tau = tau is not None
    tau_arr = mx.contiguous(tau.reshape(b * length).astype(mx.float32)) if has_tau else mx.zeros((1,), mx.float32)
    prm = mx.array([qk_thr, rel_thr, eps], dtype=mx.float32)
    return _prefill_qkvr_norm_kernel(
        inputs=[proj, mx.contiguous(k_in), mx.contiguous(v_in), q_scale, rel_scale, tau_arr, prm],
        template=[
            ("HQ", hq),
            ("HKV", hkv),
            ("HD", hd),
            ("DR", dr),
            ("L", length),
            ("PSTRIDE", pstride),
            ("QOFF", q_off),
            ("ROFF", r_off),
            ("HAS_TAU", has_tau),
            ("HAS_VNORM", use_vnorm),
        ],
        grid=(hq * hd, length, b),
        threadgroup=(hd, 1, 1),
        output_shapes=[(b, hq, length, hd), (b, hkv, length, hd), (b, hkv, length, hd), (b, length, hq, dr)],
        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16, mx.bfloat16],
    )


_FUSED_UNSORT_COMBINE_SCONV_RESID_SRC = r"""
    const int d = int(threadgroup_position_in_grid.x) * 32 + int(thread_position_in_threadgroup.x);
    const int t0 = int(threadgroup_position_in_grid.y) * 64;
    const int batch = int(threadgroup_position_in_grid.z);
    const int lx = int(thread_position_in_threadgroup.x);
    const int ly = int(thread_position_in_threadgroup.y);
    if (d >= D) return;

    threadgroup float ybuf[67][32];
    const int row_base = batch * L;
    for (int r = ly; r < 67; r += 8) {
        const int t = t0 + r - 3;
        float val = 0.0f;
        if (t < 0) {
            val = state[(size_t(batch) * 3 + (t + 3)) * D + d];
        } else if (t < L) {
            const int gtok = row_base + t;
            const bool valid = !HAS_MASK || (mask[gtok] != 0);
            if (valid) {
                float acc = 0.0f;
                const size_t wbase = size_t(gtok) * TOP_K;
                for (int k = 0; k < TOP_K; ++k) {
                    const uint sp = inv_order[wbase + k];
                    acc += float(ys_sorted[size_t(sp) * D + d]) * weights[wbase + k];
                }
                val = float(bfloat(acc));
            }
        }
        ybuf[r][lx] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float t0_w = taps[size_t(d) * 4 + 0];
    const float t1_w = taps[size_t(d) * 4 + 1];
    const float t2_w = taps[size_t(d) * 4 + 2];
    const float t3_w = taps[size_t(d) * 4 + 3];
    for (int r = ly; r < 64; r += 8) {
        const int t = t0 + r;
        if (t < L) {
            const int gtok = row_base + t;
            const float v0 = ybuf[r][lx];
            const float v1 = ybuf[r + 1][lx];
            const float v2 = ybuf[r + 2][lx];
            const float v3 = ybuf[r + 3][lx];
            const float sc = (t0_w * v0 + t1_w * v1 + t2_w * v2 + t3_w * v3 + v3) * 0.4472135955f;
            out[size_t(gtok) * D + d] = bfloat(resid[size_t(gtok) * D + d] + float(bfloat(sc)));
            if (t >= L - 3) {
                new_state[(size_t(batch) * 3 + (t - (L - 3))) * D + d] = v3;
            }
        }
    }
    if (L < 3 && t0 == 0 && ly < (3 - L)) {
        const int s_dst = ly;
        new_state[(size_t(batch) * 3 + s_dst) * D + d] = state[(size_t(batch) * 3 + s_dst + L) * D + d];
    }
"""

_fused_unsort_combine_sconv_resid_kernel = mx.fast.metal_kernel(
    name="inkling2_fused_unsort_combine_sconv_resid",
    input_names=["ys_sorted", "inv_order", "weights", "state", "taps", "resid", "mask"],
    output_names=["out", "new_state"],
    source=_FUSED_UNSORT_COMBINE_SCONV_RESID_SRC,
)


def fused_unsort_combine_sconv_resid(
    ys_sorted: mx.array,
    inv_order: mx.array,
    weights: mx.array,
    state: mx.array,
    taps: mx.array,
    resid: mx.array,
    *,
    b: int,
    length: int,
    top_k: int,
    mask: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Fused MoE unsort + weighted combine + mlp_sconv + residual add (saves ~6.3 GB/1k toks)."""
    d = resid.shape[-1]
    has_mask = mask is not None
    mask_arr = mx.contiguous(mask.reshape(b * length).astype(mx.uint8)) if has_mask else mx.zeros((1,), mx.uint8)
    nx = (d + 31) // 32
    ny = (length + 63) // 64
    return _fused_unsort_combine_sconv_resid_kernel(
        inputs=[
            ys_sorted.reshape(-1, d),
            inv_order.reshape(-1).astype(mx.uint32),
            weights.reshape(-1).astype(mx.float32),
            mx.contiguous(state.astype(mx.float32)),
            mx.contiguous(taps.astype(mx.float32)),
            mx.contiguous(resid.reshape(b * length, d).astype(mx.float32)),
            mask_arr,
        ],
        template=[("D", d), ("L", length), ("TOP_K", top_k), ("HAS_MASK", has_mask)],
        grid=(nx * 32, ny * 8, b),
        threadgroup=(32, 8, 1),
        output_shapes=[(b * length, d), (b, 3, d)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )


_MSG_BLEND_NORM_SRC = r"""
    const int j = int(threadgroup_position_in_grid.x);
    const int row = int(threadgroup_position_in_grid.y);
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    constexpr int DPT = (D + 255) / 256;
    const int h_idx = H_SHARED ? 0 : j;
    const size_t hb = (size_t(row) * HQ_IN + h_idx) * D;
    const size_t mb = (size_t(row) * MS_IN + MOFF + j) * D;
    const size_t fb = (size_t(j) * M + row) * (2 * D);

    float o[DPT];
    float n2 = 0.0f;
    for (int c = 0; c < DPT; ++c) {
        const int d = int(tid) + c * 256;
        if (d < D) {
            const float f = float(fg[fb + d]);
            const float g = float(fg[fb + D + d]);
            const float ov = (1.0f - metal::precise::tanh(f)) * float(hidden[hb + d])
                           + metal::precise::tanh(g) * float(messages[mb + d]);
            o[c] = ov;
            n2 += ov * ov;
        } else {
            o[c] = 0.0f;
        }
    }
    threadgroup float sg_sum[8];
    float s = simd_sum(n2);
    if (lane == 0) sg_sum[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float tot = simd_sum(lane < 8 ? sg_sum[lane] : 0.0f);
        if (lane == 0) sg_sum[0] = tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = metal::precise::rsqrt(sg_sum[0] / float(D) + eps);
    const int n_idx = j % QN;
    for (int c = 0; c < DPT; ++c) {
        const int d = int(tid) + c * 256;
        if (d < D) {
            out[(size_t(row) * QO + j) * D + d] = bfloat(o[c] * inv * float(norm_w[size_t(n_idx) * D + d]));
        }
    }
"""

_msg_blend_norm_kernel = mx.fast.metal_kernel(
    name="inkling2_msg_blend_norm_bf16",
    input_names=["hidden", "messages", "fg", "norm_w", "eps"],
    output_names=["out"],
    source=_MSG_BLEND_NORM_SRC,
)


def msg_blend_norm_bf16(
    hidden: mx.array,
    messages: mx.array,
    fg: mx.array,
    norm_w: mx.array,
    eps: float,
    *,
    moff: int = 0,
    h_shared: bool = False,
) -> mx.array:
    """Fused Stage-2 MessageGate blend + RMSNorm reading NAX-computed fg [QO, M, 2*D] with zero transpose."""
    qo, m = fg.shape[0], fg.shape[1]
    d = hidden.shape[-1]
    hq_in = 1 if h_shared else hidden.shape[1]
    ms_in = messages.shape[1]
    qn = norm_w.shape[0]
    return _msg_blend_norm_kernel(
        inputs=[
            mx.contiguous(hidden.astype(mx.float32)),
            mx.contiguous(messages.astype(mx.float32)),
            fg,
            norm_w,
            mx.array(eps, dtype=mx.float32),
        ],
        template=[
            ("D", d),
            ("M", m),
            ("QO", qo),
            ("HQ_IN", hq_in),
            ("MS_IN", ms_in),
            ("MOFF", moff),
            ("QN", qn),
            ("H_SHARED", h_shared),
        ],
        grid=(qo * 256, m, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(m, qo, d)],
        output_dtypes=[mx.bfloat16],
    )[0]


_relative_attention_kernel = mx.fast.metal_kernel(
    name="inkling2_relative_attention",
    input_names=["q", "k", "v", "rel", "sinks", "key_start", "sizes", "scale"],
    output_names=["out"],
    source=_RELATIVE_ATTENTION_SOURCE,
    ensure_row_contiguous=False,
)


def relative_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    rel: mx.array,
    sinks: mx.array,
    *,
    scale: float,
    window: int,
    key_start: mx.array,
) -> mx.array:
    """Causal flash attention with fused relative bias, sliding window, and query-dependent sink."""
    B, Hq, L, D = q.shape
    Hkv, S = k.shape[1], k.shape[2]
    return _relative_attention_kernel(
        inputs=[
            q,
            k,
            v,
            mx.contiguous(rel),
            mx.contiguous(sinks.astype(mx.float32)),
            mx.contiguous(key_start.astype(mx.int32)),
            mx.array([L, S], dtype=mx.int32),
            mx.array(scale, dtype=mx.float32),
        ],
        template=[
            ("T", q.dtype),
            ("D", D),
            ("HQ", Hq),
            ("HKV", Hkv),
            ("R", rel.shape[-1]),
            ("WINDOW", window),
            ("BQ", _BQ),
            ("TG", _TG),
        ],
        grid=((L + _BQ - 1) // _BQ * _TG, Hq, B),
        threadgroup=(_TG, 1, 1),
        output_shapes=[(B, L, Hq, D)],
        output_dtypes=[q.dtype],
    )[0]
