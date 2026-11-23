"""Fused Metal decode kernels for Inkling 2 Mini across all batch sizes B >= 1.

Collapses ~125 per-visit MLX ops (norm, depth AttentionResidual, Loop MessageGate,
ShortConvolution, DynamicShortConvolution, L2 clip, value norm, log tau scaling,
relative bias projection, windowed attention w/ sink & gate, top-8 sigmoid+bias router,
and weighted MoE combine + sconv + residual) into 5 batched Metal kernel dispatches per
layer visit, while packing Q/R/Gate/K/V weights into one contiguous BF16 matrix.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Optional

import mlx.core as mx

from mlx_vlm.models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchRotatingKVCache,
    KVCache,
    RotatingKVCache,
)

from .kernels import (
    decode_attention,
    fused_unsort_combine_sconv_resid,
    msg_blend_norm_bf16,
    qkv_wide_linear,
    relative_attention,
    short_convolution_resid_f32,
)
from .model import GrowingBatchKVCache, GrowingKVCache, _next_conv_state
from .moe_kernels import prefill_moe_nax_fused_combine, routed_experts_fast

_F32 = mx.float32
_KV_CACHES = (
    KVCache,
    RotatingKVCache,
    BatchKVCache,
    BatchRotatingKVCache,
    GrowingKVCache,
    GrowingBatchKVCache,
)

_HEADER = """
#define REDUCE(n, vals, tots) \\
    threadgroup_barrier(mem_flags::mem_threadgroup); \\
    for (int i_ = 0; i_ < (n); i_++) { const float s_ = simd_sum((vals)[i_]); if (lane == 0) buf[i_][sg] = s_; } \\
    threadgroup_barrier(mem_flags::mem_threadgroup); \\
    for (int i_ = 0; i_ < (n); i_++) { float t_ = 0.0f; for (int g_ = 0; g_ < NS; g_++) t_ += buf[i_][g_]; (tots)[i_] = t_; }
#define REDUCE2(n, vals, tots) \\
    threadgroup_barrier(mem_flags::mem_threadgroup); \\
    for (int i_ = 0; i_ < (n); i_++) { const float s_ = simd_sum((vals)[i_]); if (lane == 0) buf[i_][sg] = s_; } \\
    threadgroup_barrier(mem_flags::mem_threadgroup); \\
    if (sg == 0) for (int i_ = 0; i_ < (n); i_++) { \\
        const float s_ = simd_sum(lane < NS ? buf[i_][lane] : 0.0f); if (lane == 0) red[i_] = s_; } \\
    threadgroup_barrier(mem_flags::mem_threadgroup); \\
    for (int i_ = 0; i_ < (n); i_++) (tots)[i_] = red[i_];
inline float sigmoid_(float x) { return 1.0f / (1.0f + metal::precise::exp(-x)); }
"""

_MIX_SRC = """
    constexpr int NS = TG / 32;
    constexpr int DPT = (D + TG - 1) / TG;
    constexpr int NSRC = K + 1;
    constexpr int NCH = TG / RANK;
    constexpr int CD = (D + NCH - 1) / NCH;
    const uint j = threadgroup_position_in_grid.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int M = x_shape[0];
    const float eps = prm[0];
    const int jq = Q == 1 ? 0 : j;
    threadgroup float buf[NRED][NS];
    threadgroup float red[NRED];

    float v[DPT];
    for (int c = 0; c < DPT; c++) {
        const int d = tid + c * TG;
        float val = 0.0f;
        if (d < D) {
            val = float(x[size_t(row) * D + d]);
            if (CONV) {
                const size_t so = size_t(row) * 3 * D + d, to = size_t(d) * 4;
                float s0, s1, s2;
                if (SEQ) {
                    const int r = int(row);
                    s0 = r >= 3 ? float(x[size_t(r - 3) * D + d]) : state[size_t(r) * D + d];
                    s1 = r >= 2 ? float(x[size_t(r - 2) * D + d]) : state[size_t(r + 1) * D + d];
                    s2 = r >= 1 ? float(x[size_t(r - 1) * D + d]) : state[size_t(r + 2) * D + d];
                } else {
                    s0 = state[so]; s1 = state[so + D]; s2 = state[so + 2 * D];
                }
                float h = (taps[to] * s0 + taps[to + 1] * s1 + taps[to + 2] * s2 + taps[to + 3] * val + val) * 0.4472135955f;
                h = float(T(h));
                new_state[so] = s1; new_state[so + D] = s2; new_state[so + 2 * D] = val;
                val = ADD ? float(res[size_t(row) * D + d]) + h : h;
                resid[size_t(row) * D + d] = val;
            }
        }
        v[c] = val;
    }
    float one[1], tot1[1];
    one[0] = 0.0f;
    for (int c = 0; c < DPT; c++) one[0] += v[c] * v[c];
    REDUCE2(1, one, tot1)
    const float inv = metal::precise::rsqrt(tot1[0] / D + 1e-6f);
    for (int c = 0; c < DPT; c++) {
        const int d = tid + c * TG;
        v[c] *= inv;
        if (WRITE_PARTIAL && d < D && j == 0) partial[size_t(row) * D + d] = v[c];
    }

    float acc[NSRC], w[NSRC];
    for (int i = 0; i < NSRC; i++) acc[i] = 0.0f;
    for (int c = 0; c < DPT; c++) {
        const int d = tid + c * TG;
        if (d >= D) continue;
        const float qv = queries[size_t(jq) * D + d];
        for (int i = 0; i < NSRC; i++) acc[i] += (i < K ? blocks[(size_t(i) * M + row) * D + d] : v[c]) * qv;
    }
    REDUCE2(NSRC, acc, w)
    const float sc = metal::precise::rsqrt(float(D));
    float mw = -INFINITY, zw = 0.0f;
    for (int i = 0; i < NSRC; i++) mw = max(mw, w[i] * sc);
    for (int i = 0; i < NSRC; i++) { w[i] = metal::precise::exp(w[i] * sc - mw); zw += w[i]; }
    float o[DPT];
    for (int c = 0; c < DPT; c++) {
        const int d = tid + c * TG;
        float s = 0.0f;
        if (d < D) for (int i = 0; i < NSRC; i++) s += w[i] / zw * (i < K ? blocks[(size_t(i) * M + row) * D + d] : v[c]);
        o[c] = s;
    }

    if (GATE) {
        threadgroup float on[D];
        threadgroup float mn[D];
        threadgroup float pg[TG];
        threadgroup float low[RANK];
        float mm[DPT], two[2], tt[2];
        two[0] = 0.0f; two[1] = 0.0f;
        for (int c = 0; c < DPT; c++) {
            const int d = tid + c * TG;
            mm[c] = d < D ? messages[(size_t(row) * 4 + MOFF + j) * D + d] : 0.0f;
            two[0] += o[c] * o[c]; two[1] += mm[c] * mm[c];
        }
        REDUCE2(2, two, tt)
        const float io = metal::precise::rsqrt(tt[0] / D + 1e-6f), im = metal::precise::rsqrt(tt[1] / D + 1e-6f);
        for (int c = 0; c < DPT; c++) {
            const int d = tid + c * TG;
            if (d < D) { on[d] = float(GT(o[c] * io)); mn[d] = float(GT(mm[c] * im)); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int r = tid % RANK, chunk = tid / RANK;
        const size_t base = (size_t(j) * RANK + r) * 2 * D;
        float gp = 0.0f;
        for (int d = chunk * CD; d < min(D, (chunk + 1) * CD); d++)
            gp += float(down[base + d]) * on[d] + float(down[base + D + d]) * mn[d];
        pg[tid] = gp;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (chunk == 0) {
            for (int c = 1; c < NCH; c++) gp += pg[c * RANK + r];
            low[r] = float(GT(gp));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int c = 0; c < DPT; c++) {
            const int d = tid + c * TG;
            if (d >= D) continue;
            const size_t fb = (size_t(j) * D + d) * RANK;
            float f = 0.0f, gn = 0.0f;
            for (int k = 0; k < RANK; k++) { f += forget[fb + k] * low[k]; gn += gain[fb + k] * low[k]; }
            o[c] = (1.0f - metal::precise::tanh(f)) * o[c] + metal::precise::tanh(gn) * mm[c];
        }
    }

    if (NORM) {
        float n2[1], nt[1];
        n2[0] = 0.0f;
        for (int c = 0; c < DPT; c++) n2[0] += o[c] * o[c];
        REDUCE2(1, n2, nt)
        const float invj = metal::precise::rsqrt(nt[0] / D + eps);
        for (int c = 0; c < DPT; c++) {
            const int d = tid + c * TG;
            if (d < D) out[(size_t(row) * QO + j) * D + d] = OT(o[c] * invj * float(norm_w[size_t(j % QN) * D + d]));
        }
    } else {
        for (int c = 0; c < DPT; c++) {
            const int d = tid + c * TG;
            if (d < D) out[(size_t(row) * QO + j) * D + d] = OT(o[c]);
        }
    }
"""

_KV_SRC = """
    constexpr int NS = TG / 32;
    constexpr int C = HKV * HD;
    const uint h = threadgroup_position_in_grid.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint t = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const float qk_thr = prm[0], rel_thr = prm[1], eps = prm[2];
    const float tau_ = tau[row];
    threadgroup float buf[9][NS];
    threadgroup float rr[DR];

    float nine[9], dots[9];
    for (int i = 0; i < 9; i++) nine[i] = 0.0f;
    for (int d = t; d < D; d += TG) {
        nine[0] += float(xq[size_t(row) * D + d]) * float(sink_w[size_t(h) * D + d]);
        if (DYN && h < HKV) {
            const float kx = float(xk[size_t(row) * D + d]), vx = float(xv[size_t(row) * D + d]);
            for (int lag = 0; lag < 4; lag++) {
                nine[1 + lag] += kx * float(kdyn_w[size_t(h * 4 + lag) * D + d]);
                nine[5 + lag] += vx * float(vdyn_w[size_t(h * 4 + lag) * D + d]);
            }
        }
    }
    REDUCE(DYN ? 9 : 1, nine, dots)
    if (t == 0) sinks[size_t(row) * HQ + h] = T(dots[0]);

    float four[4], tot[4];
    const float qy = t < HD ? float(q[size_t(row) * QW + QOFF + h * HD + t]) * float(qscale[h * HD + t]) : 0.0f;
    const float ry = t < DR ? float(r[size_t(row) * RW + ROFF + h * DR + t]) * float(rscale[h * DR + t]) : 0.0f;
    float ky = 0.0f, vy = 0.0f;
    if (h < HKV && t < HD) {
        const int c = h * HD + t;
        const size_t so = size_t(row) * 3 * C + c;
        const float kx = float(k[size_t(row) * KW + KOFF + c]), vx = float(v[size_t(row) * VW + VOFF + c]);
        float k0, k1, k2, v0, v1, v2;
        if (SEQ) {
            const int r_ = int(row);
            k0 = r_ >= 3 ? float(k[size_t(r_ - 3) * KW + KOFF + c]) : kstate[size_t(r_) * C + c];
            k1 = r_ >= 2 ? float(k[size_t(r_ - 2) * KW + KOFF + c]) : kstate[size_t(r_ + 1) * C + c];
            k2 = r_ >= 1 ? float(k[size_t(r_ - 1) * KW + KOFF + c]) : kstate[size_t(r_ + 2) * C + c];
            v0 = r_ >= 3 ? float(v[size_t(r_ - 3) * VW + VOFF + c]) : vstate[size_t(r_) * C + c];
            v1 = r_ >= 2 ? float(v[size_t(r_ - 2) * VW + VOFF + c]) : vstate[size_t(r_ + 1) * C + c];
            v2 = r_ >= 1 ? float(v[size_t(r_ - 1) * VW + VOFF + c]) : vstate[size_t(r_ + 2) * C + c];
        } else {
            k0 = kstate[so]; k1 = kstate[so + C]; k2 = kstate[so + 2 * C];
            v0 = vstate[so]; v1 = vstate[so + C]; v2 = vstate[so + 2 * C];
        }
        float kc, vc;
        if (DYN) {
            kc = ((float(ktaps[c]) + float(T(dots[1]))) * kx + (float(ktaps[C + c]) + float(T(dots[2]))) * k2
                + (float(ktaps[2 * C + c]) + float(T(dots[3]))) * k1 + (float(ktaps[3 * C + c]) + float(T(dots[4]))) * k0
                + kx) * 0.3333333333f;
            vc = ((float(vtaps[c]) + float(T(dots[5]))) * vx + (float(vtaps[C + c]) + float(T(dots[6]))) * v2
                + (float(vtaps[2 * C + c]) + float(T(dots[7]))) * v1 + (float(vtaps[3 * C + c]) + float(T(dots[8]))) * v0
                + vx) * 0.3333333333f;
        } else {
            const size_t to = size_t(c) * 4;
            kc = (float(ktaps[to]) * k0 + float(ktaps[to + 1]) * k1 + float(ktaps[to + 2]) * k2 + float(ktaps[to + 3]) * kx + kx)
                * 0.4472135955f;
            vc = (float(vtaps[to]) * v0 + float(vtaps[to + 1]) * v1 + float(vtaps[to + 2]) * v2 + float(vtaps[to + 3]) * vx + vx)
                * 0.4472135955f;
        }
        ky = float(T(kc)); vy = float(T(vc));
        kstate_new[so] = k1; kstate_new[so + C] = k2; kstate_new[so + 2 * C] = kx;
        vstate_new[so] = v1; vstate_new[so + C] = v2; vstate_new[so + 2 * C] = vx;
    }
    four[0] = qy * qy; four[1] = ry * ry; four[2] = ky * ky; four[3] = vy * vy;
    REDUCE(4, four, tot)

    if (t < HD) {
        float qv = float(T(qy * min(qk_thr / (metal::precise::sqrt(tot[0]) + 1e-6f), 1.0f)));
        if (TAU) qv = float(T(qv * tau_));
        qo[(size_t(row) * HQ + h) * HD + t] = T(qv);
        if (h < HKV) {
            const size_t xo = (size_t(row) * HKV + h) * HD + t;
            ko[xo] = T(ky * min(qk_thr / (metal::precise::sqrt(tot[2]) + 1e-6f), 1.0f));
            vo[xo] = T(vy * metal::precise::rsqrt(tot[3] / HD + eps));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < DR) {
        float rv = float(T(ry * min(rel_thr / (metal::precise::sqrt(tot[1]) + 1e-6f), 1.0f)));
        if (TAU) rv = float(T(rv * tau_));
        rr[t] = rv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int d = t; d < R; d += TG) {
        float b = 0.0f;
        for (int i = 0; i < DR; i++) b += rr[i] * float(proj[size_t(i) * R + d]);
        bias[(size_t(row) * HQ + h) * R + d] = T(b);
    }
"""

_ATTEND_SRC = """
    constexpr int NS = TG / 32;
    constexpr int G = HQ / HKV;
    constexpr int CH = TG / HD;
    constexpr int CK = (TG + CH - 1) / CH;
    const uint kvh = threadgroup_position_in_grid.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint t = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int S = keys_shape[2], M = q_shape[0];
    const int brow = SEQ ? 0 : int(row);
    const int idx = meta[0], count = counts[brow];
    const float scale = prm[0];
    threadgroup float p[G][TG];
    threadgroup float part[G][TG];
    threadgroup float buf[G][NS];
    threadgroup float red[G];
    const device T* kp = keys + size_t(brow) * keys_strides[0] + size_t(kvh) * keys_strides[1];
    const device T* vp = values + size_t(brow) * values_strides[0] + size_t(kvh) * values_strides[1];
    const size_t qb = (size_t(row) * HQ + kvh * G) * HD;

    const int last = SEQ ? S - M + int(row) : S - 1;
    const int first_valid = max(S - count, WINDOW > 0 ? max(0, last - WINDOW + 1) : 0);
    const int n_valid = max(0, last - first_valid + 1);
    const int place = first_valid + int(t);
    const bool seen = int(t) < n_valid;
    float sc[G], m[G], z[G];
    for (int g = 0; g < G; g++) sc[g] = -INFINITY;
    if (seen) {
        const device T* ks = kp + size_t((place + idx) % S) * keys_strides[2];
        for (int g = 0; g < G; g++) sc[g] = 0.0f;
        for (int d = 0; d < HD; d++) {
            const float kd = float(ks[d]);
            for (int g = 0; g < G; g++) sc[g] += float(q[qb + g * HD + d]) * kd;
        }
        const int dist = last - place;
        for (int g = 0; g < G; g++)
            sc[g] = sc[g] * scale + (dist < R ? float(bias[(size_t(row) * HQ + kvh * G + g) * R + dist]) : 0.0f);
    }
    for (int g = 0; g < G; g++) { const float v = simd_max(sc[g]); if (lane == 0) buf[g][sg] = v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) for (int g = 0; g < G; g++) {
        const float v = simd_max(lane < NS ? buf[g][lane] : -INFINITY);
        if (lane == 0) red[g] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int g = 0; g < G; g++) {
        m[g] = max(red[g], float(sinks[size_t(row) * HQ + kvh * G + g]));
        const float e = seen ? metal::precise::exp(sc[g] - m[g]) : 0.0f;
        p[g][t] = e;
        z[g] = simd_sum(e);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int g = 0; g < G; g++) if (lane == 0) buf[g][sg] = z[g];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) for (int g = 0; g < G; g++) {
        const float v = simd_sum(lane < NS ? buf[g][lane] : 0.0f);
        if (lane == 0) red[g] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int g = 0; g < G; g++) z[g] = red[g] + metal::precise::exp(float(sinks[size_t(row) * HQ + kvh * G + g]) - m[g]);

    const int d = t % HD, ch = t / HD;
    float o[G];
    for (int g = 0; g < G; g++) o[g] = 0.0f;
    for (int i = ch * CK; i < min(n_valid, (ch + 1) * CK); i++) {
        const float vd = float(vp[size_t((first_valid + i + idx) % S) * values_strides[2] + d]);
        for (int g = 0; g < G; g++) o[g] += p[g][i] * vd;
    }
    for (int g = 0; g < G; g++) part[g][t] = o[g];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ch == 0) for (int g = 0; g < G; g++) {
        float v = o[g];
        for (int c = 1; c < CH; c++) v += part[g][c * HD + d];
        v = float(T(v / z[g]));
        const size_t at = (kvh * G + g) * HD + d;
        out[size_t(row) * HQ * HD + at] = T(v * sigmoid_(float(gate[size_t(row) * GW + GOFF + at])));
    }
"""

_ROUTE_SRC = """
    constexpr int CHUNK = (D + PARTS - 1) / PARTS;
    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const int e = tid / PARTS, part = tid % PARTS;
    threadgroup float partial[E * PARTS];
    threadgroup float score[E];
    threadgroup float chosen[K];
    float dot = 0.0f;
    for (int d = part * CHUNK; d < min(D, (part + 1) * CHUNK); d++) dot += float(x[size_t(row) * D + d]) * w[size_t(e) * D + d];
    partial[tid] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (part == 0) {
        for (int i = 1; i < PARTS; i++) dot += partial[e * PARTS + i];
        score[e] = sigmoid_(dot);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (part == 0) {
        const float mine = score[e] + bias[e];
        int rank = 0;
        for (int i = 0; i < E; i++) {
            const float other = score[i] + bias[i];
            rank += (other > mine || (other == mine && i < e)) ? 1 : 0;
        }
        if (rank < K) { inds[size_t(row) * K + rank] = uint(e); chosen[rank] = score[e]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < K) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) sum += chosen[k];
        weights[size_t(row) * K + tid] = chosen[tid] / (sum + 1e-20f) * prm[0];
    }
"""

_COMBINE_SRC = """
    const uint d = thread_position_in_grid.x;
    const uint row = thread_position_in_grid.y;
    float ysum[4];
    for (int j = 0; j < 4; j++) {
        const int r = int(row) - 3 + j;
        ysum[j] = 0.0f;
        if (r < 0 || (j < 3 && !SEQ)) continue;
        float y = 0.0f;
        for (int k = 0; k < K; k++) y += float(ys[(size_t(r) * K + k) * D + d]) * weights[size_t(r) * K + k];
        ysum[j] = float(T(y));
    }
    const float y = ysum[3];
    const size_t so = size_t(row) * 3 * D + d, to = size_t(d) * 4;
    float s0, s1, s2;
    if (SEQ) {
        const int r = int(row);
        s0 = r >= 3 ? ysum[0] : state[size_t(r) * D + d];
        s1 = r >= 2 ? ysum[1] : state[size_t(r + 1) * D + d];
        s2 = r >= 1 ? ysum[2] : state[size_t(r + 2) * D + d];
    } else {
        s0 = state[so]; s1 = state[so + D]; s2 = state[so + 2 * D];
    }
    float h = (taps[to] * s0 + taps[to + 1] * s1 + taps[to + 2] * s2 + taps[to + 3] * y + y) * 0.4472135955f;
    h = float(T(h));
    new_state[so] = s1; new_state[so + D] = s2; new_state[so + 2 * D] = y;
    out[size_t(row) * D + d] = T(resid[size_t(row) * D + d] + h);
"""

_KERNELS: dict[str, Any] = {}
_FCONSTS: dict[tuple[float, ...], mx.array] = {}
_SMALL: dict[tuple[Any, tuple[Any, ...]], mx.array] = {}
_ZERO: dict[Any, mx.array] = {}


def _kernel(name: str, inputs: list[str], outputs: list[str], source: str, contiguous: bool = True):
    k = _KERNELS.get(name)
    if k is None:
        k = _KERNELS[name] = mx.fast.metal_kernel(
            name="inkling_" + name,
            input_names=inputs,
            output_names=outputs,
            source=source,
            header=_HEADER,
            ensure_row_contiguous=contiguous,
        )
    return k


def _f(*vals: float) -> mx.array:
    out = _FCONSTS.get(vals)
    if out is None:
        out = _FCONSTS[vals] = mx.array(vals, dtype=_F32)
    return out


def _small(dtype: Any, vals: tuple[Any, ...]) -> mx.array:
    key = (dtype, vals)
    out = _SMALL.get(key)
    if out is None:
        if len(_SMALL) > 8192:
            _SMALL.clear()
        out = _SMALL[key] = mx.array(vals, dtype=dtype)
    return out


def _zero(dtype: Any) -> mx.array:
    out = _ZERO.get(dtype)
    if out is None:
        out = _ZERO[dtype] = mx.zeros((1,), dtype=dtype)
    return out


def _mix(
    x: mx.array,
    blocks: mx.array,
    queries: mx.array,
    *,
    norm_w: Optional[mx.array] = None,
    eps: float = 1e-6,
    conv: Optional[tuple] = None,
    messages: Optional[mx.array] = None,
    gates: Optional[tuple] = None,
    moff: int = 0,
    outputs: int = 0,
    shape: Optional[tuple[int, ...]] = None,
    seq: bool = False,
    write_partial: bool = True,
):
    m, d = x.shape
    k, q = blocks.shape[0], queries.shape[0]
    qo = outputs or q
    gate = messages is not None
    rank = gates[0].shape[1] if gate else 1
    tg = 1024 if d >= 1024 else 256 if d >= 256 else 32
    tg = min(tg, max(rank, 32)) if tg % rank != 0 else tg
    state, taps, res = conv if conv is not None else (_zero(_F32), _zero(_F32), None)
    down, forget, gain = gates if gate else (_zero(x.dtype), _zero(_F32), _zero(_F32))
    t_in = x.dtype
    ot = norm_w.dtype if norm_w is not None else _F32
    out = _kernel(
        "mix",
        ["x", "blocks", "queries", "state", "taps", "res", "messages", "down", "forget", "gain", "norm_w", "prm"],
        ["out", "partial", "resid", "new_state"],
        _MIX_SRC,
    )(
        inputs=[
            x,
            blocks,
            queries,
            state,
            taps,
            res if res is not None else _zero(t_in),
            messages if gate else _zero(_F32),
            down,
            forget,
            gain,
            norm_w if norm_w is not None else _zero(ot),
            _f(eps),
        ],
        template=[
            ("T", t_in),
            ("OT", ot),
            ("GT", down.dtype),
            ("D", d),
            ("K", k),
            ("Q", q),
            ("QO", qo),
            ("QN", norm_w.shape[0] if norm_w is not None else 1),
            ("TG", tg),
            ("CONV", conv is not None),
            ("ADD", res is not None),
            ("GATE", gate),
            ("NORM", norm_w is not None),
            ("MOFF", moff),
            ("RANK", rank),
            ("NRED", max(k + 1, 2)),
            ("SEQ", seq),
            ("WRITE_PARTIAL", write_partial),
        ],
        grid=(qo * tg, m, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[
            shape or (m, qo, d),
            (m, d) if write_partial else (1,),
            (m, d) if conv is not None else (1,),
            (m, 3, d) if conv is not None else (1,),
        ],
        output_dtypes=[ot, _F32, _F32, _F32],
    )
    return out if conv is not None else out[:2]


_MIX_PRESCORED_SRC = """
    constexpr int NS = TG / 32;
    constexpr int DPT = (D + TG - 1) / TG;
    const uint row = threadgroup_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int M = x_shape[0];
    const float eps = prm[0];
    threadgroup float buf[Q][NS];
    threadgroup float red[Q];

    float v[DPT];
    float one[1], tot1[1];
    one[0] = 0.0f;
    for (int c = 0; c < DPT; c++) {
        const int d = tid + c * TG;
        const float val = d < D ? float(x[size_t(row) * D + d]) : 0.0f;
        v[c] = val;
        one[0] += val * val;
    }
    REDUCE2(1, one, tot1)
    const float inv = metal::precise::rsqrt(tot1[0] / D + 1e-6f);
    for (int c = 0; c < DPT; c++) {
        const int d = tid + c * TG;
        v[c] *= inv;
        if (WRITE_PARTIAL && d < D) partial[size_t(row) * D + d] = v[c];
    }

    float dot_cur[Q], tot_cur[Q];
    for (int jq = 0; jq < Q; jq++) dot_cur[jq] = 0.0f;
    for (int c = 0; c < DPT; c++) {
        const int d = tid + c * TG;
        if (d < D) {
            for (int jq = 0; jq < Q; jq++) dot_cur[jq] += v[c] * queries[size_t(jq) * D + d];
        }
    }
    REDUCE2(Q, dot_cur, tot_cur)
    const float sc = metal::precise::rsqrt(float(D));

    float w[Q][K + 1];
    for (int jq = 0; jq < Q; jq++) {
        float s_arr[K + 1];
        float mw = -INFINITY;
        for (int i = 0; i < K; i++) {
            const float si = hist_scores[(size_t(i) * M + row) * Q + jq];
            s_arr[i] = si;
            mw = max(mw, si);
        }
        const float s_last = tot_cur[jq] * sc;
        s_arr[K] = s_last;
        mw = max(mw, s_last);
        float zw = 0.0f;
        for (int i = 0; i <= K; i++) {
            s_arr[i] = metal::precise::exp(s_arr[i] - mw);
            zw += s_arr[i];
        }
        const float inv_z = 1.0f / zw;
        for (int i = 0; i <= K; i++) w[jq][i] = s_arr[i] * inv_z;
    }

    float o[Q][DPT];
    for (int jq = 0; jq < Q; jq++)
        for (int c = 0; c < DPT; c++)
            o[jq][c] = w[jq][K] * v[c];

    const size_t rbase = size_t(row) * D;
    #define ACC_BLK_(IDX, BNAME) \\
        if (K > IDX) { \\
            for (int c = 0; c < DPT; c++) { \\
                const int d = tid + c * TG; \\
                if (d < D) { \\
                    const float bv = BNAME[rbase + d]; \\
                    for (int jq = 0; jq < Q; jq++) o[jq][c] += w[jq][IDX] * bv; \\
                } \\
            } \\
        }
    ACC_BLK_(0, b0) ACC_BLK_(1, b1) ACC_BLK_(2, b2) ACC_BLK_(3, b3)
    ACC_BLK_(4, b4) ACC_BLK_(5, b5) ACC_BLK_(6, b6) ACC_BLK_(7, b7)
    #undef ACC_BLK_

    if (NORM) {
        float oo[Q], tot_o[Q];
        for (int jq = 0; jq < Q; jq++) {
            float s = 0.0f;
            for (int c = 0; c < DPT; c++) s += o[jq][c] * o[jq][c];
            oo[jq] = s;
        }
        REDUCE2(Q, oo, tot_o)
        for (int jq = 0; jq < Q; jq++) {
            const float io = metal::precise::rsqrt(tot_o[jq] / D + eps);
            const size_t nbase = size_t(jq % QN) * D;
            const size_t obase = (size_t(row) * Q + jq) * D;
            for (int c = 0; c < DPT; c++) {
                const int d = tid + c * TG;
                if (d < D) out[obase + d] = OT(o[jq][c] * io * float(norm_w[nbase + d]));
            }
        }
    } else {
        for (int jq = 0; jq < Q; jq++) {
            const size_t obase = (size_t(row) * Q + jq) * D;
            for (int c = 0; c < DPT; c++) {
                const int d = tid + c * TG;
                if (d < D) out[obase + d] = OT(o[jq][c]);
            }
        }
    }
"""


def _mix_prescored(
    x: mx.array,
    blocks_list: list[mx.array],
    hist_scores: mx.array,
    queries: mx.array,
    *,
    norm_w: Optional[mx.array] = None,
    eps: float = 1e-6,
    shape: Optional[tuple[int, ...]] = None,
    write_partial: bool = False,
):
    """Single-read pre-scored AttentionResidual combine across all Q outputs (1 <= K <= 8, ZERO blocks concat)."""
    m, d = x.shape
    k = len(blocks_list)
    q = queries.shape[0]
    tg = min(256, max(32, -(-d // 32) * 32))
    ot = norm_w.dtype if norm_w is not None else _F32
    qn = norm_w.shape[0] if norm_w is not None else 1
    zf = _zero(_F32)
    b_in = [blocks_list[i] if i < k else zf for i in range(8)]
    return _kernel(
        "mix_prescored",
        ["x", "queries", "hist_scores", "norm_w", "prm", "b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"],
        ["out", "partial"],
        _MIX_PRESCORED_SRC,
    )(
        inputs=[
            x,
            queries,
            hist_scores,
            norm_w if norm_w is not None else _zero(ot),
            _f(eps),
            *b_in,
        ],
        template=[
            ("T", x.dtype),
            ("OT", ot),
            ("D", d),
            ("K", k),
            ("Q", q),
            ("QN", qn),
            ("TG", tg),
            ("NORM", norm_w is not None),
            ("WRITE_PARTIAL", write_partial),
        ],
        grid=(tg, m, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[shape or (m, q, d), (m, d) if write_partial else (1,)],
        output_dtypes=[ot, _F32],
    )


def _kv_prep(
    q, r, k, v, at, xq, xk, xv, sink_w, kdyn_w, vdyn_w, kstate, vstate,
    ktaps, vtaps, qscale, rscale, proj, tau, *,
    heads: int, kv_heads: int, hd: int, qk_threshold: float, rel_threshold: float, eps: float, seq: bool = False,
):
    m, d = xq.shape
    dr, reg_r = rscale.shape[-1], proj.shape[-1]
    t_in = xq.dtype
    c = kv_heads * hd
    tg = max(32, -(-hd // 32) * 32)
    dyn = kdyn_w is not None
    z = _zero(t_in)
    return _kernel(
        "kv",
        ["q", "r", "k", "v", "xq", "xk", "xv", "sink_w", "kdyn_w", "vdyn_w",
         "kstate", "vstate", "ktaps", "vtaps", "qscale", "rscale", "proj", "tau", "prm"],
        ["qo", "bias", "ko", "vo", "sinks", "kstate_new", "vstate_new"],
        _KV_SRC,
    )(
        inputs=[
            q, r, k, v, xq, xk if dyn else z, xv if dyn else z, sink_w,
            kdyn_w if dyn else z, vdyn_w if dyn else z,
            kstate, vstate, ktaps, vtaps, qscale, rscale, proj,
            tau if tau is not None else _zero(_F32),
            _f(qk_threshold, rel_threshold, eps),
        ],
        template=[
            ("T", t_in), ("D", d), ("HQ", heads), ("HKV", kv_heads), ("HD", hd),
            ("DR", dr), ("R", reg_r), ("TG", tg), ("DYN", dyn), ("TAU", tau is not None),
            ("QW", q.shape[-1]), ("RW", r.shape[-1]), ("KW", k.shape[-1]), ("VW", v.shape[-1]),
            ("QOFF", at[0]), ("ROFF", at[1]), ("KOFF", at[2]), ("VOFF", at[3]), ("SEQ", seq),
        ],
        grid=(heads * tg, m, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[
            (m, heads, hd), (m, heads, reg_r), (m, kv_heads, 1, hd), (m, kv_heads, 1, hd),
            (m, heads), (m, 3, c), (m, 3, c),
        ],
        output_dtypes=[t_in, t_in, t_in, t_in, t_in, _F32, _F32],
    )


def _window_tg(hd: int) -> int:
    return min(1024, -(-hd * 8 // 32) * 32)


def _attend(q, keys, values, bias, sinks, gate, idx: int, counts, scale: float, gate_at: int = 0, window_size: int = 0, seq: bool = False):
    m, heads, hd = q.shape
    t_in = q.dtype
    tg = _window_tg(hd)
    return _kernel(
        "attend",
        ["q", "keys", "values", "bias", "sinks", "gate", "meta", "counts", "prm"],
        ["out"],
        _ATTEND_SRC,
        contiguous=False,
    )(
        inputs=[q, keys, values, bias, sinks, gate, _small(mx.int32, (idx,)), counts, _f(scale)],
        template=[
            ("T", t_in), ("HQ", heads), ("HKV", keys.shape[1]), ("HD", hd),
            ("R", bias.shape[-1]), ("TG", tg), ("GW", gate.shape[-1]),
            ("GOFF", gate_at), ("WINDOW", window_size), ("SEQ", seq),
        ],
        grid=(keys.shape[1] * tg, m, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(m, heads * hd)],
        output_dtypes=[t_in],
    )[0]


def _route(x: mx.array, weight: mx.array, bias: mx.array, k: int, route_scale: float):
    e, d = weight.shape
    m = x.size // d
    parts = max(1, min(8, 1024 // e))
    return _kernel("route", ["x", "w", "bias", "prm"], ["inds", "weights"], _ROUTE_SRC)(
        inputs=[x, weight, bias, _f(route_scale)],
        template=[("T", x.dtype), ("D", d), ("E", e), ("K", k), ("PARTS", parts)],
        grid=(m * e * parts, 1, 1),
        threadgroup=(e * parts, 1, 1),
        output_shapes=[(m, k), (m, k)],
        output_dtypes=[mx.uint32, _F32],
    )


def _route_prefill(x: mx.array, weight_f32: mx.array, bias_f32: mx.array, k: int, route_scale: float):
    """GEMM-based exact FP32 top-k router for large prefill M >= 64 (avoids GEMV router 200ms cliff!)."""
    logits = x.astype(_F32) @ weight_f32.T
    scores = mx.sigmoid(logits)
    biased = scores + bias_f32
    inds = mx.argpartition(-biased, kth=k - 1, axis=-1)[:, :k].astype(mx.uint32)
    weights = mx.take_along_axis(scores, inds, axis=-1)
    weights = (weights / mx.sum(weights, axis=-1, keepdims=True)) * route_scale
    return inds, weights


def _combine(ys: mx.array, weights: mx.array, state: mx.array, taps: mx.array, resid: mx.array, seq: bool = False):
    m, d = resid.shape
    k = ys.size // (m * d)
    t_in = ys.dtype
    return _kernel("combine", ["ys", "weights", "state", "taps", "resid"], ["out", "new_state"], _COMBINE_SRC)(
        inputs=[ys, weights, state, taps, resid],
        template=[("T", t_in), ("K", k), ("D", d), ("SEQ", seq)],
        grid=(d, m, 1),
        threadgroup=(min(256, d), 1, 1),
        output_shapes=[(m, d), (m, 3, d)],
        output_dtypes=[t_in, _F32],
    )


@lru_cache(maxsize=64)
def _rms_w_f32(x: mx.array) -> mx.array:
    return x


class _LayerConsts:
    """Packs per-layer weights & contiguous Q/R/Gate/K/V stack once with zero extra weight copies."""

    def __init__(self, layer, cfg):
        a = layer.self_attn
        d = cfg.hidden_size
        f32 = lambda t: t.astype(_F32)
        self.attn_query = f32(layer.attention_input.query).reshape(-1, d) if layer.attention_input is not None else None
        norms = (
            (layer.input_layernorm,)
            if layer.is_sliding
            else (layer.q_input_layernorm, layer.k_input_layernorm, layer.v_input_layernorm)
        )
        self.attn_norm = mx.stack([n.weight for n in norms])
        self.attn_eps = norms[0].eps
        self.mlp_query = f32(layer.mlp_input.query).reshape(1, d)
        self.mlp_norm = layer.post_attention_layernorm.weight.reshape(1, d)
        self.mlp_eps = layer.post_attention_layernorm.eps
        self.attn_taps = f32(layer.attn_sconv.conv.weight).reshape(d, -1)
        self.mlp_taps = f32(layer.mlp_sconv.conv.weight).reshape(d, -1)
        if layer.is_sliding:
            self.k_taps = f32(a.k_sconv.conv.weight).reshape(-1, a.k_sconv.kernel_size)
            self.v_taps = f32(a.v_sconv.conv.weight).reshape(-1, a.v_sconv.kernel_size)
            self.kdyn_w = self.vdyn_w = None
        else:
            self.k_taps, self.v_taps = a.k_sconv.static_weight, a.v_sconv.static_weight
            self.kdyn_w, self.vdyn_w = a.k_sconv.weight_proj.weight, a.v_sconv.weight_proj.weight
        self.attn_gates = self.mlp_gates = None
        self.attn_down_h = self.attn_down_m = None
        self.mlp_down_h = self.mlp_down_m = None
        self.attn_fg_bf16_t = self.mlp_fg_bf16_t = None
        if hasattr(layer, "q_message_gate"):
            def _stack(gates):
                return (
                    mx.stack([g.down_proj.weight for g in gates]),
                    mx.stack([f32(g.forget_gate_proj.weight) for g in gates]),
                    mx.stack([f32(g.message_gate_proj.weight) for g in gates]),
                )
            self.attn_gates = _stack((layer.q_message_gate, layer.k_message_gate, layer.v_message_gate))
            self.mlp_gates = _stack((layer.mlp_message_gate,))
            ad = self.attn_gates[0]
            self.attn_down_h = mx.contiguous(ad[:, :, :d].swapaxes(-1, -2))
            self.attn_down_m = mx.contiguous(ad[:, :, d:].swapaxes(-1, -2))
            self.attn_fg_bf16_t = mx.contiguous(
                mx.concatenate([self.attn_gates[1], self.attn_gates[2]], axis=1).astype(ad.dtype).swapaxes(-1, -2)
            )
            md = self.mlp_gates[0][0]
            self.mlp_down_h = mx.contiguous(md[:, :d].T)
            self.mlp_down_m = mx.contiguous(md[:, d:].T)
            self.mlp_fg_bf16_t = mx.contiguous(
                mx.concatenate([self.mlp_gates[1][0], self.mlp_gates[2][0]], axis=0).astype(md.dtype).T
            )
        self.router_w = f32(layer.mlp.gate_weight)
        self.router_b = f32(layer.mlp.e_score_correction_bias)
        self.sink_w = a.sink_proj.weight

        mods = [a.q_proj, a.r_proj, a.gate_proj, a.k_proj, a.v_proj]
        weights = mx.concatenate([m.weight for m in mods], axis=0)
        mx.eval(weights)
        self.at: list[int] = []
        start = 0
        for m in mods:
            rows = m.weight.shape[0]
            self.at.append(start)
            m.weight = weights[start : start + rows]
            start += rows
        self.stack = weights
        self.head = weights[: self.at[3]]
        self.rel_proj_t = mx.contiguous(a.rel_proj.T.astype(weights.dtype))
        self.rel_proj_bf16 = mx.contiguous(a.rel_proj.astype(weights.dtype))
        mx.eval([v for v in vars(self).values() if isinstance(v, mx.array)], [m.parameters() for m in mods])


def _rms(x: mx.array) -> mx.array:
    return x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + 1e-6)


def _unpadded(conv: ArraysCache) -> bool:
    base = getattr(conv, "_left_padding", None)
    if base is None:
        return True
    seen = getattr(conv, "_fused_padding", None)
    if seen is None or seen[0] is not base:
        seen = (base, int(mx.max(base)))
        conv._fused_padding = seen
    return conv._left_padding_advance >= seen[1]


def supported(text_model, h: mx.array, cache) -> bool:
    """True when this forward step can run fused kernels for ANY batch size B >= 1 (L == 1)."""
    if (
        h.ndim != 3
        or h.shape[1] != 1
        or cache is None
        or not mx.metal.is_available()
        or mx.default_device() != mx.gpu
        or h.dtype not in (mx.bfloat16, mx.float32)
    ):
        return False
    for entry in cache:
        kv, conv = entry.caches[0], entry.caches[1]
        if type(kv) not in _KV_CACHES or not isinstance(conv, ArraysCache):
            return False
        if getattr(conv, "_lengths", None) is not None or not _unpadded(conv):
            return False
    return True


def _long_attention(
    attn_mod,
    qo,
    keys,
    values,
    bias,
    sinks,
    kv,
    gate: Optional[mx.array] = None,
    gate_at: int = 0,
    seq: bool = False,
):
    """Exact split-K attention past the single-threadgroup limit for ALL B >= 1 and B=1 seq L<=8."""
    b, hq, hd = qo.shape
    q = qo[:, :, None, :]
    size = keys.shape[2]
    batched = isinstance(kv, (BatchKVCache, BatchRotatingKVCache))
    rotating = isinstance(kv, (RotatingKVCache, BatchRotatingKVCache))
    ring_idx = int(kv._idx) % size if rotating and size > 0 else 0
    if rotating:
        win_floor = max(0, size - attn_mod.sliding) if (attn_mod.sliding > 0 and not seq) else 0
        if batched:
            valid = mx.minimum(kv.offset, size)
            start = mx.maximum(size - valid, win_floor).astype(mx.int32)
        else:
            start = _small(mx.int32, (win_floor,) * (1 if seq else b)) if win_floor > 0 else None
    else:
        start = mx.maximum(kv.left_padding, 0).astype(mx.int32) if batched else None
    if q.dtype == mx.bfloat16 and keys.dtype == mx.bfloat16 and hd % 32 == 0:
        return decode_attention(
            q,
            keys,
            values,
            bias,
            sinks,
            scale=hd**-0.5,
            key_start=start,
            ring_idx=ring_idx,
            window=attn_mod.sliding if seq else 0,
            seq=seq,
            gate=gate,
            gate_at=gate_at,
        )
    if b == 1:
        distance = (size - 1) - mx.arange(size)
        rel = mx.take(bias[0], mx.minimum(distance, attn_mod.rel_extent - 1), axis=-1)
        rel = mx.where(distance < attn_mod.rel_extent, rel, mx.array(0, rel.dtype))
        if start is not None:
            rel = mx.where(mx.arange(size) >= start[0], rel, mx.array(-mx.inf, rel.dtype))
        out = mx.fast.scaled_dot_product_attention(
            q, keys, values, scale=hd**-0.5, mask=rel[None, :, None, :], sinks=sinks[0].astype(q.dtype)
        )
    else:
        out = relative_attention(
            q,
            keys,
            values,
            bias.reshape(b, 1, hq, -1),
            sinks.reshape(b, 1, hq),
            scale=hd**-0.5,
            window=attn_mod.sliding,
            key_start=start if start is not None else mx.zeros((b,), mx.int32),
        ).transpose(0, 2, 1, 3)
    if gate is not None:
        g = gate[:, gate_at : gate_at + hq * hd]
        return (out.reshape(b, -1).astype(_F32) * mx.sigmoid(g.astype(_F32))).astype(g.dtype)
    return out


def _visit(
    layer,
    c: _LayerConsts,
    cfg,
    x: mx.array,
    blocks: Optional[mx.array],
    kv,
    conv,
    messages: Optional[mx.array],
    *,
    seq: bool = False,
    return_conv_states: bool = False,
):
    a = layer.self_attn
    m, d = x.shape
    ch = a.k_proj.weight.shape[0]
    sb = 1 if seq else m
    st = [
        mx.zeros((sb, 3, n), _F32) if s is None else s.astype(_F32)
        for s, n in zip(conv.cache, (ch, ch, d, d))
    ]
    if layer.attention_input is None:
        normed = _rms(x.astype(_F32))
        xq = xk = xv = layer.input_layernorm(x)
        qkv_in = xq
    elif (
        messages is not None
        and not seq
        and m >= 16
        and x.dtype == mx.bfloat16
        and c.attn_fg_bf16_t is not None
    ):
        mixed, normed = _mix(x, blocks, c.attn_query, write_partial=layer.is_block_boundary)
        m_n = mx.fast.rms_norm(messages[:, :3], None, c.attn_eps).astype(x.dtype).transpose(1, 0, 2)
        low_m = m_n @ c.attn_down_m
        if layer.is_sliding:
            h_n = mx.fast.rms_norm(mixed[:, 0], None, c.attn_eps).astype(x.dtype)
            low_h = mx.broadcast_to(h_n[None, :, :], (3, m, d)) @ c.attn_down_h
            fg = (low_h + low_m).astype(x.dtype) @ c.attn_fg_bf16_t
            out_qkv = msg_blend_norm_bf16(
                mixed[:, :1], messages, fg, c.attn_norm, c.attn_eps, h_shared=True
            )
        else:
            h_n = mx.fast.rms_norm(mixed[:, :3], None, c.attn_eps).astype(x.dtype).transpose(1, 0, 2)
            fg = (h_n @ c.attn_down_h + low_m).astype(x.dtype) @ c.attn_fg_bf16_t
            out_qkv = msg_blend_norm_bf16(
                mixed[:, :3], messages, fg, c.attn_norm, c.attn_eps, h_shared=False
            )
        qkv_in = out_qkv
        xq, xk, xv = out_qkv[:, 0], out_qkv[:, 1], out_qkv[:, 2]
    else:
        gated = messages is not None
        out, normed = _mix(
            x,
            blocks,
            c.attn_query,
            norm_w=c.attn_norm,
            eps=c.attn_eps,
            messages=messages,
            gates=c.attn_gates if gated else None,
            outputs=3 if gated else 0,
            shape=(m, d) if layer.is_sliding and not gated else None,
            write_partial=layer.is_block_boundary,
        )
        qkv_in = out
        xq, xk, xv = (out,) * 3 if out.ndim == 2 else (out[:, 0], out[:, 1], out[:, 2])

    tau = None
    batched = isinstance(kv.offset, mx.array)
    if not layer.is_sliding and cfg.log_scaling_n_floor is not None:
        if batched:
            pos = (kv.offset + 1).astype(_F32)
            tau = 1.0 + cfg.log_scaling_alpha * mx.log(mx.maximum(pos / cfg.log_scaling_n_floor, 1.0))
        elif seq:
            pos = (kv.offset + 1 + mx.arange(m)).astype(_F32)
            tau = 1.0 + cfg.log_scaling_alpha * mx.log(mx.maximum(pos / cfg.log_scaling_n_floor, 1.0))
        else:
            val = 1.0 + cfg.log_scaling_alpha * math.log(max((kv.offset + 1) / cfg.log_scaling_n_floor, 1.0))
            tau = _small(_F32, (val,) * m)

    use_wide = 1 <= m <= 8 and x.dtype == mx.bfloat16 and d % 128 == 0 and c.stack.dtype == mx.bfloat16
    if use_wide:
        q = r = k = v = gate = qkv_wide_linear(qkv_in, c.stack, c.at[3], c.at[4])
        at, gate_at = (c.at[0], c.at[1], c.at[3], c.at[4]), c.at[2]
    elif xk is xq and xv is xq:
        q = r = k = v = gate = xq @ c.stack.T
        at, gate_at = (c.at[0], c.at[1], c.at[3], c.at[4]), c.at[2]
    else:
        q = r = gate = xq @ c.head.T
        k, v, at, gate_at = a.k_proj(xk), a.v_proj(xv), (c.at[0], c.at[1], 0, 0), c.at[2]

    qo, bias, ko, vo, sinks, ks, vs = _kv_prep(
        q, r, k, v, at, xq, xk, xv, c.sink_w, c.kdyn_w, c.vdyn_w, st[0], st[1],
        c.k_taps, c.v_taps, a.q_scale, a.rel_scale, a.rel_proj, tau,
        heads=a.n_heads, kv_heads=a.n_kv, hd=a.head_dim,
        qk_threshold=cfg.qk_l2_clip_threshold, rel_threshold=cfg.rel_l2_clip_threshold,
        eps=cfg.rms_norm_eps,
        seq=seq,
    )
    if seq:
        keys, values = kv.update_and_fetch(ko.transpose(2, 1, 0, 3), vo.transpose(2, 1, 0, 3))
    else:
        keys, values = kv.update_and_fetch(ko, vo)
    s_len = keys.shape[2]
    use_splitk_batched = (not seq) and m >= 8 and s_len >= 128 and qo.dtype == mx.bfloat16
    if s_len <= _window_tg(qo.shape[-1]) and not use_splitk_batched:
        idx = kv._idx if isinstance(kv, (RotatingKVCache, BatchRotatingKVCache)) else 0
        counts = mx.minimum(kv.offset, s_len).astype(mx.int32) if batched else _small(mx.int32, (s_len,) * (1 if seq else m))
        att = _attend(qo, keys, values, bias, sinks, gate, idx, counts, a.head_dim**-0.5, gate_at, a.sliding, seq=seq)
    else:
        att = _long_attention(a, qo, keys, values, bias, sinks, kv, gate=gate, gate_at=gate_at, seq=seq)
    if use_wide and att.shape[-1] % 128 == 0 and a.o_proj.weight.dtype == mx.bfloat16:
        att = qkv_wide_linear(att, a.o_proj.weight)
    else:
        att = a.o_proj(att)

    boundary = layer.is_block_boundary
    if boundary:
        blocks = normed[None] if blocks is None else mx.concatenate([blocks, normed[None]], axis=0)
    gated = messages is not None
    if (
        gated
        and not seq
        and m >= 16
        and x.dtype == mx.bfloat16
        and c.mlp_fg_bf16_t is not None
    ):
        mlp_mixed, _, resid, cs = _mix(
            att,
            blocks,
            c.mlp_query,
            conv=(st[2], c.attn_taps, None if boundary else x),
            write_partial=False,
            seq=seq,
        )
        h_n = mx.fast.rms_norm(mlp_mixed[:, 0], None, c.mlp_eps).astype(x.dtype)
        m_n = mx.fast.rms_norm(messages[:, 3], None, c.mlp_eps).astype(x.dtype)
        fg = ((h_n @ c.mlp_down_h + m_n @ c.mlp_down_m).astype(x.dtype) @ c.mlp_fg_bf16_t)[None, :, :]
        mlp_in = msg_blend_norm_bf16(
            mlp_mixed[:, :1], messages, fg, c.mlp_norm, c.mlp_eps, moff=3, h_shared=True
        )[:, 0]
    else:
        mlp_in, _, resid, cs = _mix(
            att,
            blocks,
            c.mlp_query,
            norm_w=c.mlp_norm,
            eps=c.mlp_eps,
            conv=(st[2], c.attn_taps, None if boundary else x),
            messages=messages,
            gates=c.mlp_gates if gated else None,
            moff=3,
            shape=(m, 1, d),
            seq=seq,
            write_partial=False,
        )

    moe = layer.mlp
    if m >= 16 and not seq:
        inds, weights = _route_prefill(mlp_in.reshape(m, d), c.router_w, c.router_b, moe.top_k, moe.route_scale)
    else:
        inds, weights = _route(mlp_in, c.router_w, c.router_b, moe.top_k, moe.route_scale)
    ys = routed_experts_fast(mlp_in.reshape(m, d), inds, moe)
    x_out, ms = _combine(ys, weights, st[3], c.mlp_taps, resid, seq=seq)
    for idx_c, arr in enumerate((ks, vs, cs, ms)):
        conv[idx_c] = arr[-1:] if seq else arr
    conv.advance(m if seq else 1)
    if return_conv_states:
        return x_out, blocks, (ks, vs, cs, ms)
    return x_out, blocks


def prefill_step(
    text_model,
    h: mx.array,
    cache,
    capture_layer_ids: Optional[list[int]] = None,
):
    """Fused Metal prefill for ANY B >= 1, L >= 1 on GPU BF16 (eliminates 120x mx.stack & 100MB/layer rel)."""
    cfg = text_model.config
    consts = getattr(text_model, "_fused_consts", None)
    if consts is None:
        consts = [_LayerConsts(layer, cfg) for layer in text_model.layers]
        object.__setattr__(text_model, "_fused_consts", consts)

    b, length, d = h.shape
    m = b * length
    x_dtype = h.dtype
    x = h.reshape(m, d)
    start = cfg.loop_layer_index_start
    requested = set(capture_layer_ids or [])
    captures: dict[int, mx.array] = {}
    blocks = pre_loop = bank = None
    bank_norm_t = bank_weights = None
    msg_q_flat = getattr(text_model, "_fused_msg_q_flat", None)
    if msg_q_flat is None and hasattr(text_model, "message_input"):
        msg_q_flat = mx.contiguous(text_model.message_input.query.reshape(-1, d).astype(_F32))
        mx.eval(msg_q_flat)
        object.__setattr__(text_model, "_fused_msg_q_flat", msg_q_flat)

    res_q_pack = getattr(text_model, "_fused_res_q_pack", None)
    if res_q_pack is None:
        q_chunks = []
        attn_q_slices: list[Optional[tuple[int, int]]] = []
        mlp_q_slices: list[tuple[int, int]] = []
        cursor = 0
        for layer_j, cj in zip(text_model.layers, consts):
            if layer_j.attention_input is not None and cj.attn_query is not None:
                qn = int(cj.attn_query.shape[0])
                q_chunks.append(cj.attn_query)
                attn_q_slices.append((cursor, cursor + qn))
                cursor += qn
            else:
                attn_q_slices.append(None)
            q_chunks.append(cj.mlp_query)
            mlp_q_slices.append((cursor, cursor + 1))
            cursor += 1
        final_q = text_model.final_input.query.reshape(1, -1).astype(_F32)
        q_chunks.append(final_q)
        final_q_slice = (cursor, cursor + 1)
        all_res_q_t = mx.contiguous(mx.concatenate(q_chunks, axis=0).T)
        mx.eval(all_res_q_t)
        res_q_pack = (all_res_q_t, attn_q_slices, mlp_q_slices, final_q_slice, d**-0.5)
        object.__setattr__(text_model, "_fused_res_q_pack", res_q_pack)
    all_res_q_t, attn_q_slices, mlp_q_slices, final_q_slice, inv_sqrt_d = res_q_pack

    from .model import _cache_padding_mask

    use_fused_moe_post = (
        m >= 128
        and x_dtype == mx.bfloat16
        and mx.default_device() == mx.gpu
    )
    use_prescored_mix = m > 32 and mx.default_device() == mx.gpu
    blocks_list: list[mx.array] = []
    scores_list: list[mx.array] = []

    def _run_mix(
        cur_x: mx.array,
        queries: mx.array,
        q_slice: tuple[int, int],
        *,
        norm_w: Optional[mx.array] = None,
        eps: float = 1e-6,
        messages_in: Optional[mx.array] = None,
        gates=None,
        moff: int = 0,
        outputs: int = 0,
        shape: Optional[tuple[int, ...]] = None,
        write_partial: bool = False,
    ):
        k_blks = len(blocks_list)
        if use_prescored_mix and gates is None and 1 <= k_blks <= 8:
            qs0, qs1 = q_slice
            hs = mx.stack([s[:, qs0:qs1] for s in scores_list], axis=0)
            return _mix_prescored(
                cur_x,
                blocks_list,
                hs,
                queries,
                norm_w=norm_w,
                eps=eps,
                shape=shape,
                write_partial=write_partial,
            )
        blks_t = mx.stack(blocks_list, axis=0) if k_blks > 0 else None
        return _mix(
            cur_x,
            blks_t,
            queries,
            norm_w=norm_w,
            eps=eps,
            messages=messages_in,
            gates=gates,
            moff=moff,
            outputs=outputs,
            shape=shape,
            write_partial=write_partial,
        )

    for step, (i, cache_idx, pass_idx) in enumerate(text_model.schedule()):
        layer = text_model.layers[i]
        c = consts[i]
        if i == start:
            if pass_idx == 0:
                pre_loop = (x, list(blocks_list), list(scores_list))
            else:
                done = len(pre_loop[1])
                bank_blks = blocks_list[done + 1 :]
                bank = (mx.stack(bank_blks, axis=0) if bank_blks else None, x)
                if m > 32 and msg_q_flat is not None:
                    b_last = _rms(x.astype(_F32))
                    bank_norm = mx.stack([*bank_blks, b_last], axis=0)
                    k_b = bank_norm.shape[0]
                    scores = (bank_norm.reshape(k_b * m, d) @ msg_q_flat.T).reshape(k_b, m, -1) * (d**-0.5)
                    bank_weights = mx.softmax(scores, axis=0)
                    bank_norm_t = mx.contiguous(bank_norm.transpose(1, 0, 2))
                else:
                    bank_norm_t = bank_weights = None
                x, blocks_list, scores_list = pre_loop[0], list(pre_loop[1]), list(pre_loop[2])
        if step in requested:
            captures[step] = x.reshape(b, length, d)

        messages = None
        if pass_idx > 0:
            if bank_norm_t is not None and bank_weights is not None:
                qoff = (i - start) * 4
                w4 = bank_weights[:, :, qoff : qoff + 4]
                messages = w4.transpose(1, 2, 0) @ bank_norm_t
            else:
                messages = _mix(bank[1], bank[0], text_model.message_input.query[i - start].astype(_F32))[0]

        entry = cache[cache_idx] if cache is not None else None
        conv = entry[1] if entry is not None else None
        mask = _cache_padding_mask(conv, length)

        if layer.attention_input is None:
            normed = _rms(x.astype(_F32))
            xq = xk = xv = layer.input_layernorm(x).astype(x_dtype)
        elif messages is not None and m > 32 and c.attn_fg_bf16_t is not None:
            mixed, normed = _run_mix(
                x, c.attn_query, attn_q_slices[i], write_partial=layer.is_block_boundary
            )
            m_n = mx.fast.rms_norm(messages[:, :3], None, c.attn_eps).astype(x_dtype).transpose(1, 0, 2)
            low_m = m_n @ c.attn_down_m
            if layer.is_sliding:
                h_n = mx.fast.rms_norm(mixed[:, 0], None, c.attn_eps).astype(x_dtype)
                low_h = mx.broadcast_to(h_n[None, :, :], (3, m, d)) @ c.attn_down_h
                fg = (low_h + low_m).astype(x_dtype) @ c.attn_fg_bf16_t
                out_qkv = msg_blend_norm_bf16(
                    mixed[:, :1], messages, fg, c.attn_norm, c.attn_eps, h_shared=True
                )
            else:
                h_n = mx.fast.rms_norm(mixed[:, :3], None, c.attn_eps).astype(x_dtype).transpose(1, 0, 2)
                fg = (h_n @ c.attn_down_h + low_m).astype(x_dtype) @ c.attn_fg_bf16_t
                out_qkv = msg_blend_norm_bf16(
                    mixed[:, :3], messages, fg, c.attn_norm, c.attn_eps, h_shared=False
                )
            xq, xk, xv = out_qkv[:, 0], out_qkv[:, 1], out_qkv[:, 2]
        else:
            gated = messages is not None
            out, normed = _run_mix(
                x,
                c.attn_query,
                attn_q_slices[i],
                norm_w=c.attn_norm,
                eps=c.attn_eps,
                messages_in=messages,
                gates=c.attn_gates if gated else None,
                outputs=3 if gated else 0,
                shape=(m, d) if layer.is_sliding and not gated else None,
                write_partial=layer.is_block_boundary,
            )
            xq, xk, xv = (out,) * 3 if out.ndim == 2 else (out[:, 0], out[:, 1], out[:, 2])

        attn = layer.self_attn(
            xq.reshape(b, length, d),
            key_input=None if xk is xq else xk.reshape(b, length, d),
            value_input=None if xv is xq else xv.reshape(b, length, d),
            cache=entry,
            conv_mask=mask,
            fused_const=c,
        )
        if conv is not None and x_dtype == mx.bfloat16 and mx.default_device() == mx.gpu:
            sc_mod = layer.attn_sconv
            a_st = conv[sc_mod.conv_idx]
            if a_st is None:
                a_st = mx.zeros((b, sc_mod.kernel_size - 1, d), _F32)
            residual = short_convolution_resid_f32(
                a_st,
                c.attn_taps,
                attn,
                None if layer.is_block_boundary else x,
                mask,
                sc_mod.kernel_size,
            )
            conv[sc_mod.conv_idx] = _next_conv_state(a_st, attn, mask)
        else:
            residual = layer.attn_sconv(attn, conv, mask).astype(_F32).reshape(m, d)
            if not layer.is_block_boundary:
                residual = residual + x.astype(_F32)

        if layer.is_block_boundary:
            blocks_list.append(normed)
            if use_prescored_mix:
                scores_list.append((normed @ all_res_q_t) * inv_sqrt_d)

        if messages is not None and m > 32 and c.mlp_fg_bf16_t is not None:
            mlp_mixed, _ = _run_mix(residual, c.mlp_query, mlp_q_slices[i], write_partial=False)
            h_n = mx.fast.rms_norm(mlp_mixed[:, 0], None, c.mlp_eps).astype(x_dtype)
            m_n = mx.fast.rms_norm(messages[:, 3], None, c.mlp_eps).astype(x_dtype)
            fg = ((h_n @ c.mlp_down_h + m_n @ c.mlp_down_m).astype(x_dtype) @ c.mlp_fg_bf16_t)[None, :, :]
            mlp_in = msg_blend_norm_bf16(
                mlp_mixed[:, :1], messages, fg, c.mlp_norm, c.mlp_eps, moff=3, h_shared=True
            )[:, 0]
        else:
            gated = messages is not None
            mlp_in, _ = _run_mix(
                residual,
                c.mlp_query,
                mlp_q_slices[i],
                norm_w=c.mlp_norm,
                eps=c.mlp_eps,
                messages_in=messages,
                gates=c.mlp_gates if gated else None,
                moff=3,
                shape=(m, d),
                write_partial=False,
            )

        if use_fused_moe_post and conv is not None:
            moe = layer.mlp
            inds, rweights = _route_prefill(mlp_in, c.router_w, c.router_b, moe.top_k, moe.route_scale)
            m_st = conv[3]
            if m_st is None:
                m_st = mx.zeros((b, 3, d), _F32)
            fused_res = prefill_moe_nax_fused_combine(
                mlp_in,
                inds,
                rweights,
                m_st,
                c.mlp_taps,
                residual,
                moe,
                b=b,
                length=length,
                mask=mask,
            )
            if fused_res is not None:
                x, new_mst = fused_res
            else:
                ys_sorted, inv_order = moe.routed_experts_sorted(mlp_in, inds)
                x, new_mst = fused_unsort_combine_sconv_resid(
                    ys_sorted,
                    inv_order,
                    rweights,
                    m_st,
                    c.mlp_taps,
                    residual,
                    b=b,
                    length=length,
                    top_k=moe.top_k,
                    mask=mask,
                )
            conv[3] = new_mst
            conv.advance(length)
        else:
            moe = layer.mlp
            if (
                1 <= m <= 112
                and x_dtype == mx.bfloat16
                and mx.default_device() == mx.gpu
                and getattr(moe.w13_proj, "mode", None) in ("nvfp4", "mxfp8")
            ):
                inds, rweights = _route(mlp_in, c.router_w, c.router_b, moe.top_k, moe.route_scale)
                ys = routed_experts_fast(mlp_in, inds, moe)
                hidden = mx.sum(ys.astype(_F32) * rweights[:, :, None], axis=1).astype(x_dtype).reshape(b, length, d)
            else:
                hidden = moe(mlp_in.reshape(b, length, d))
            x = (residual + layer.mlp_sconv(hidden, conv, mask).astype(_F32).reshape(m, d)).astype(x_dtype)
            if conv is not None:
                conv.advance(length)
        if (step + 1) % 4 == 0:
            mx.async_eval(x)

    out = _run_mix(
        x,
        text_model.final_input.query.reshape(1, -1).astype(_F32),
        final_q_slice,
        write_partial=False,
    )[0].reshape(b, length, d)
    if capture_layer_ids is not None:
        total = cfg.total_num_hidden_layers
        if total in requested:
            captures[total] = text_model.norm(out).astype(x_dtype)
        return out, [captures[k] for k in capture_layer_ids]
    return out


def verify_step(
    text_model,
    h: mx.array,
    cache,
    capture_layer_ids: tuple[int, ...] = (5, 23, 41, 48),
) -> tuple[mx.array, mx.array, list[tuple[mx.array, mx.array, mx.array, mx.array]]]:
    """Fused B=1 sequential verify step (1 <= L <= 8) for MTP speculative decoding.

    Returns `(pre_norm_out [1, L, D], captured_concat [1, L, 4*D], conv_states_per_visit)`
    with zero fallback to unoptimized prefill ops and exact per-position conv states for rollback.
    """
    cfg = text_model.config
    consts = getattr(text_model, "_fused_consts", None)
    if consts is None:
        consts = [_LayerConsts(layer, cfg) for layer in text_model.layers]
        object.__setattr__(text_model, "_fused_consts", consts)

    _, length, d = h.shape
    seq = length > 1
    x = h.reshape(length, d)
    x_dtype = h.dtype
    start = cfg.loop_layer_index_start
    requested = set(capture_layer_ids)
    captures: dict[int, mx.array] = {}
    conv_states: list[tuple[mx.array, mx.array, mx.array, mx.array]] = []
    blocks = pre_loop = bank = None

    for step, (i, cache_idx, pass_idx) in enumerate(text_model.schedule()):
        if i == start:
            if pass_idx == 0:
                pre_loop = (x, blocks)
            else:
                done = 0 if pre_loop[1] is None else pre_loop[1].shape[0]
                bank = (blocks[done + 1 :], x)
                x, blocks = pre_loop
        if step in requested:
            captures[step] = x.reshape(1, length, d)
        messages = None
        if pass_idx > 0:
            messages = _mix(bank[1], bank[0], text_model.message_input.query[i - start].astype(_F32))[0]
        entry = cache[cache_idx]
        x, blocks, cstates = _visit(
            text_model.layers[i],
            consts[i],
            cfg,
            x,
            blocks,
            entry.caches[0],
            entry.caches[1],
            messages,
            seq=seq,
            return_conv_states=True,
        )
        conv_states.append(cstates)
        if (step + 1) % 4 == 0:
            mx.async_eval(x)

    out = _mix(x, blocks, text_model.final_input.query.reshape(1, -1).astype(_F32))[0].reshape(1, length, d)
    total = cfg.total_num_hidden_layers
    if total in requested:
        captures[total] = text_model.norm(out).astype(x_dtype)
    captured = mx.concatenate([captures[k] for k in capture_layer_ids], axis=-1)
    return out, captured, conv_states


def decode_step(text_model, h: mx.array, cache) -> mx.array:
    """Execute one decode token across all 48 layer visits in fused Metal kernels (pre-norm output)."""
    cfg = text_model.config
    consts = getattr(text_model, "_fused_consts", None)
    if consts is None:
        consts = [_LayerConsts(layer, cfg) for layer in text_model.layers]
        object.__setattr__(text_model, "_fused_consts", consts)

    x = h[:, 0]
    m, d = x.shape
    start = cfg.loop_layer_index_start
    blocks = pre_loop = bank = None
    bank_norm_t = bank_weights = None
    msg_q_flat = getattr(text_model, "_fused_msg_q_flat", None)
    if msg_q_flat is None and m >= 16 and hasattr(text_model, "message_input"):
        msg_q_flat = mx.contiguous(text_model.message_input.query.reshape(-1, d).astype(_F32))
        mx.eval(msg_q_flat)
        object.__setattr__(text_model, "_fused_msg_q_flat", msg_q_flat)
    for step, (i, cache_idx, pass_idx) in enumerate(text_model.schedule()):
        if i == start:
            if pass_idx == 0:
                pre_loop = (x, blocks)
            else:
                done = 0 if pre_loop[1] is None else pre_loop[1].shape[0]
                bank = (blocks[done + 1 :], x)
                if m >= 16 and msg_q_flat is not None:
                    b_last = _rms(x.astype(_F32))[None]
                    bank_norm = mx.concatenate([blocks[done + 1 :], b_last], axis=0)
                    k_b = bank_norm.shape[0]
                    scores = (bank_norm.reshape(k_b * m, d) @ msg_q_flat.T).reshape(k_b, m, -1) * (d**-0.5)
                    bank_weights = mx.softmax(scores, axis=0)
                    bank_norm_t = mx.contiguous(bank_norm.transpose(1, 0, 2))
                else:
                    bank_norm_t = bank_weights = None
                x, blocks = pre_loop
        messages = None
        if pass_idx > 0:
            if bank_norm_t is not None and bank_weights is not None:
                qoff = (i - start) * 4
                w4 = bank_weights[:, :, qoff : qoff + 4]
                messages = w4.transpose(1, 2, 0) @ bank_norm_t
            else:
                messages = _mix(bank[1], bank[0], text_model.message_input.query[i - start].astype(_F32))[0]
        entry = cache[cache_idx]
        x, blocks = _visit(text_model.layers[i], consts[i], cfg, x, blocks, entry.caches[0], entry.caches[1], messages)
        if (step + 1) % 4 == 0:
            mx.async_eval(x)
    out = _mix(x, blocks, text_model.final_input.query.reshape(1, -1).astype(_F32), write_partial=False)[0]
    return out.reshape(h.shape[0], 1, -1)
