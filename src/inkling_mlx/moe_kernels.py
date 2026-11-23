"""Custom 8-lane 100%-utilization NVFP4 & MXFP8 MoE Metal kernels (M=1..48).

Key design invariants:
- `2304 = 9 * 256` and `1152 = 9 * 128`, so 8 lanes per output row divides EVERY K/INTER
  group count evenly (9 or 18 iterations -> 0% idle lanes, 64B/128B coalesced DRAM bursts).
- 3-step `simd_shuffle_xor` (`reduce8`) matches identically between `M=1` and grouped `M=2..48`
  -> exact `0.00000` batch invariance across `B=1..64`.
- `m1_w2_combine`: 8 simdgroups per threadgroup execute all `TOP_K=8` experts in PARALLEL,
  reduce in 128B shared memory in fixed `k=0..7` order, and fuse `mlp_sconv + resid` (zero
  `[M, 8, 2304]` DRAM traffic).
- Grouped kernels (`M=2..48`): GPU 128-bin radix bucket -> each unique active expert's weights
  read ONCE from DRAM across all assigned tokens (`MT=4` or `MT=8`), with 8-float pre-decode
  amortized across `tn > 1` tokens.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

_DECODERS_HEADER = r"""
constant float E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
   -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
};

inline float decode_fp8_e4m3(uint8_t value) {
    half decoded = as_type<half>(ushort((value & 127) << 7));
    decoded *= 256.0f;
    return float(value & 128 ? -decoded : decoded);
}

inline float decode_fp8_e8m0(uint8_t value) {
    uint32_t decoded = value == 0 ? 0x400000 : uint32_t(value) << 23;
    return as_type<float>(decoded);
}

inline float reduce8(float v) {
    v += simd_shuffle_xor(v, 4);
    v += simd_shuffle_xor(v, 2);
    v += simd_shuffle_xor(v, 1);
    return v;
}

inline float softcapped_swiglu(float gate, float up, float gate_cap, float up_cap) {
    float g = metal::precise::tanh(gate / gate_cap) * gate_cap
              * (1.0f / (1.0f + metal::precise::exp(-gate)));
    float u = metal::precise::tanh(up / up_cap) * up_cap;
    return g * u;
}

inline float dot8_nvfp4(uint w, vec<bfloat, 4> x0, vec<bfloat, 4> x1) {
    return float(x0[0]) * E2M1_LUT[w & 0xFu]
         + float(x0[1]) * E2M1_LUT[(w >> 4) & 0xFu]
         + float(x0[2]) * E2M1_LUT[(w >> 8) & 0xFu]
         + float(x0[3]) * E2M1_LUT[(w >> 12) & 0xFu]
         + float(x1[0]) * E2M1_LUT[(w >> 16) & 0xFu]
         + float(x1[1]) * E2M1_LUT[(w >> 20) & 0xFu]
         + float(x1[2]) * E2M1_LUT[(w >> 24) & 0xFu]
         + float(x1[3]) * E2M1_LUT[(w >> 28) & 0xFu];
}

inline float dot4_fp8(uint w, vec<bfloat, 4> x0) {
    return float(x0[0]) * decode_fp8_e4m3(w & 0xFFu)
         + float(x0[1]) * decode_fp8_e4m3((w >> 8) & 0xFFu)
         + float(x0[2]) * decode_fp8_e4m3((w >> 16) & 0xFFu)
         + float(x0[3]) * decode_fp8_e4m3((w >> 24) & 0xFFu);
}

inline void decode8_nvfp4_scaled(uint w, float s, thread float f[8]) {
    f[0] = E2M1_LUT[w & 0xFu] * s;
    f[1] = E2M1_LUT[(w >> 4) & 0xFu] * s;
    f[2] = E2M1_LUT[(w >> 8) & 0xFu] * s;
    f[3] = E2M1_LUT[(w >> 12) & 0xFu] * s;
    f[4] = E2M1_LUT[(w >> 16) & 0xFu] * s;
    f[5] = E2M1_LUT[(w >> 20) & 0xFu] * s;
    f[6] = E2M1_LUT[(w >> 24) & 0xFu] * s;
    f[7] = E2M1_LUT[(w >> 28) & 0xFu] * s;
}

inline void decode4_fp8_scaled(uint w, float s, thread float f[4]) {
    f[0] = decode_fp8_e4m3(w & 0xFFu) * s;
    f[1] = decode_fp8_e4m3((w >> 8) & 0xFFu) * s;
    f[2] = decode_fp8_e4m3((w >> 16) & 0xFFu) * s;
    f[3] = decode_fp8_e4m3((w >> 24) & 0xFFu) * s;
}

inline float decode_fp8_e8m0_x256(uint8_t sc) {
    if (sc <= 246u && sc >= 1u) {
        return as_type<float>(uint32_t(sc + 8u) << 23);
    }
    return decode_fp8_e8m0(sc) * 256.0f;
}

inline half4 unpack4_fp8_h_div256(uint w) {
    const ushort4 b = ushort4(as_type<uchar4>(w));
    const ushort4 h = ((b & ushort4(0x7Fu)) << 7) | ((b & ushort4(0x80u)) << 8);
    return as_type<half4>(h);
}

inline float dot8_h(half4 lo, half4 hi, vec<bfloat, 4> x0, vec<bfloat, 4> x1) {
    return float(x0[0]) * float(lo.x)
         + float(x0[1]) * float(lo.y)
         + float(x0[2]) * float(lo.z)
         + float(x0[3]) * float(lo.w)
         + float(x1[0]) * float(hi.x)
         + float(x1[1]) * float(hi.y)
         + float(x1[2]) * float(hi.z)
         + float(x1[3]) * float(hi.w);
}

inline float dot8_hf(half4 lo, half4 hi, float4 fx0, float4 fx1) {
    return fx0.x * float(lo.x)
         + fx0.y * float(lo.y)
         + fx0.z * float(lo.z)
         + fx0.w * float(lo.w)
         + fx1.x * float(hi.x)
         + fx1.y * float(hi.y)
         + fx1.z * float(hi.z)
         + fx1.w * float(hi.w);
}

inline float dot4_h(half4 w, vec<bfloat, 4> x0) {
    return float(x0[0]) * float(w.x)
         + float(x0[1]) * float(w.y)
         + float(x0[2]) * float(w.z)
         + float(x0[3]) * float(w.w);
}

inline float dot4_hf(half4 w, float4 fx0) {
    return fx0.x * float(w.x)
         + fx0.y * float(w.y)
         + fx0.z * float(w.z)
         + fx0.w * float(w.w);
}
"""


def _build_e2m1_pair_h_table() -> str:
    lut = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]
    entries = [f"half2({lut[b & 0xF]:.1f}h,{lut[(b >> 4) & 0xF]:.1f}h)" for b in range(256)]
    rows = [", ".join(entries[i : i + 8]) for i in range(0, 256, 8)]
    return (
        "constant half2 E2M1_PAIR_H[256] = {\n  "
        + ",\n  ".join(rows)
        + "\n};\n"
        + "inline void unpack8_nvfp4_h(uint w, thread half4& lo, thread half4& hi) {\n"
        + "    half2 p0 = E2M1_PAIR_H[w & 0xFFu];\n"
        + "    half2 p1 = E2M1_PAIR_H[(w >> 8) & 0xFFu];\n"
        + "    half2 p2 = E2M1_PAIR_H[(w >> 16) & 0xFFu];\n"
        + "    half2 p3 = E2M1_PAIR_H[w >> 24];\n"
        + "    lo = half4(p0, p1);\n"
        + "    hi = half4(p2, p3);\n"
        + "}\n"
    )


_DECODERS_HEADER = _DECODERS_HEADER + "\n" + _build_e2m1_pair_h_table()


_BUCKET_TILES_SRC = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const int T = inds_shape[0];
    threadgroup atomic_uint curs[128];
    atomic_store_explicit(&curs[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = int(tid); i < T; i += 128) {
        uint e = inds[i];
        atomic_fetch_add_explicit(&curs[e], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint c = atomic_load_explicit(&curs[tid], memory_order_relaxed);
    uint n_tiles = (c + uint(MT) - 1u) / uint(MT);
    uint sg_inc = simd_prefix_inclusive_sum(c);
    uint sg_exc = sg_inc - c;
    uint t_inc = simd_prefix_inclusive_sum(n_tiles);
    uint t_exc = t_inc - n_tiles;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    threadgroup uint sg_tots[4];
    threadgroup uint t_tots[4];
    if (lane == 31) { sg_tots[sg] = sg_inc; t_tots[sg] = t_inc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = 0u, tbase = 0u;
    for (uint s = 0u; s < sg; ++s) { base += sg_tots[s]; tbase += t_tots[s]; }
    uint off = base + sg_exc;
    uint toff = tbase + t_exc;
    offsets[tid] = int(off);
    if (tid == 127) offsets[128] = int(off + c);
    for (int i = int(tid); i < MAX_TILES; i += 128) {
        tile_expert[i] = -1;
        tile_t0[i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0u; k < n_tiles; ++k) {
        tile_expert[toff + k] = int(tid);
        tile_t0[toff + k] = int(k * uint(MT));
    }
    atomic_store_explicit(&curs[tid], off, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = int(tid); i < T; i += 128) {
        uint e = inds[i];
        uint pos = atomic_fetch_add_explicit(&curs[e], 1u, memory_order_relaxed);
        sorted_slots[pos] = uint(i);
    }
"""

_BUCKET_SRC = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const int T = inds_shape[0];
    threadgroup atomic_uint curs[128];
    atomic_store_explicit(&curs[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = int(tid); i < T; i += 128) {
        uint e = inds[i];
        atomic_fetch_add_explicit(&curs[e], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint c = atomic_load_explicit(&curs[tid], memory_order_relaxed);
    uint sg_inc = simd_prefix_inclusive_sum(c);
    uint sg_exc = sg_inc - c;
    uint act = c > 0u ? 1u : 0u;
    uint act_inc = simd_prefix_inclusive_sum(act);
    uint act_exc = act_inc - act;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    threadgroup uint sg_tots[4];
    threadgroup uint act_tots[4];
    if (lane == 31) { sg_tots[sg] = sg_inc; act_tots[sg] = act_inc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = 0, abase = 0;
    for (uint s = 0; s < sg; ++s) { base += sg_tots[s]; abase += act_tots[s]; }
    uint off = base + sg_exc;
    uint aoff = abase + act_exc;
    offsets[tid] = int(off);
    if (tid == 127) offsets[128] = int(off + c);
    if (int(tid) < MAX_ACT) active_list[tid] = -1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (c > 0u) active_list[aoff] = int(tid);
    atomic_store_explicit(&curs[tid], off, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = int(tid); i < T; i += 128) {
        uint e = inds[i];
        uint pos = atomic_fetch_add_explicit(&curs[e], 1u, memory_order_relaxed);
        sorted_slots[pos] = uint(i);
        inv_order[i] = pos;
    }
"""


@lru_cache(maxsize=1)
def _bucket_kernel():
    return mx.fast.metal_kernel(
        name="inkling_moe3_bucket_128_compact_inv",
        input_names=["inds"],
        output_names=["offsets", "sorted_slots", "active_list", "inv_order"],
        source=_BUCKET_SRC,
    )


def bucket_experts(flat_inds: mx.array, num_experts: int = 128) -> tuple[mx.array, mx.array, mx.array]:
    t = flat_inds.shape[0]
    max_act = min(num_experts, t)
    offsets, sorted_slots, active_list, _ = _bucket_kernel()(
        inputs=[mx.contiguous(flat_inds.astype(mx.uint32))],
        template=[("MAX_ACT", max_act)],
        grid=(128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(129,), (t,), (max_act,), (t,)],
        output_dtypes=[mx.int32, mx.uint32, mx.int32, mx.uint32],
    )
    return offsets, sorted_slots, active_list


def bucket_experts_full(flat_inds: mx.array, num_experts: int = 128) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Return (offsets, sorted_slots, active_list, inv_order) in ONE 128-thread GPU radix pass."""
    t = flat_inds.shape[0]
    max_act = min(num_experts, t)
    return _bucket_kernel()(
        inputs=[mx.contiguous(flat_inds.astype(mx.uint32))],
        template=[("MAX_ACT", max_act)],
        grid=(128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(129,), (t,), (max_act,), (t,)],
        output_dtypes=[mx.int32, mx.uint32, mx.int32, mx.uint32],
    )


_M1_W13_NVFP4_SRC = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sub = lane >> 3;
    const uint slane = lane & 7u;
    const uint sg = simdgroup_index_in_threadgroup;
    const int r = (int(threadgroup_position_in_grid.x) * SG + int(sg)) * 4 + int(sub);
    const int slot = int(threadgroup_position_in_grid.y);
    const int m = int(threadgroup_position_in_grid.z);
    if (r >= INTER) return;

    const uint e = inds[m * TOP_K + slot];
    const float oscale = gscale[e] * (1.0f / 2688.0f);
    const float gate_cap = caps[0];
    const float up_cap = caps[1];

    constexpr int PAIRS = K / 32;
    constexpr int GROUPS = K / 16;
    constexpr int W_STRIDE = K / 8;
    const device bfloat* xm = x + size_t(m) * K;
    const size_t w_base_g = (size_t(e) * (2 * INTER) + size_t(2 * r)) * W_STRIDE;
    const size_t w_base_u = w_base_g + W_STRIDE;
    const size_t s_base_g = (size_t(e) * (2 * INTER) + size_t(2 * r)) * GROUPS;
    const size_t s_base_u = s_base_g + GROUPS;

    float acc_g = 0.0f;
    float acc_u = 0.0f;
    for (int p = int(slane); p < PAIRS; p += 8) {
        const uint4 wg = *(const device uint4*)(w + w_base_g + p * 4);
        const uchar2 sg2 = *(const device uchar2*)(scales + s_base_g + p * 2);
        const uint4 wu = *(const device uint4*)(w + w_base_u + p * 4);
        const uchar2 su2 = *(const device uchar2*)(scales + s_base_u + p * 2);
        auto xp = (const device vec<bfloat, 4>*)(xm + p * 32);
        {
            const vec<bfloat, 4> x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
            acc_g += (dot8_nvfp4(wg.x, x0, x1) + dot8_nvfp4(wg.y, x2, x3)) * decode_fp8_e4m3(sg2.x);
            acc_u += (dot8_nvfp4(wu.x, x0, x1) + dot8_nvfp4(wu.y, x2, x3)) * decode_fp8_e4m3(su2.x);
        }
        {
            const vec<bfloat, 4> x0 = xp[4], x1 = xp[5], x2 = xp[6], x3 = xp[7];
            acc_g += (dot8_nvfp4(wg.z, x0, x1) + dot8_nvfp4(wg.w, x2, x3)) * decode_fp8_e4m3(sg2.y);
            acc_u += (dot8_nvfp4(wu.z, x0, x1) + dot8_nvfp4(wu.w, x2, x3)) * decode_fp8_e4m3(su2.y);
        }
    }

    const float gate = float(bfloat(reduce8(acc_g) * oscale));
    const float up = float(bfloat(reduce8(acc_u) * oscale));
    if (slane == 0) {
        out[(size_t(m) * TOP_K + slot) * INTER + r] = bfloat(softcapped_swiglu(gate, up, gate_cap, up_cap));
    }
"""

_M1_W13_MXFP8_SRC = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sub = lane >> 3;
    const uint slane = lane & 7u;
    const uint sg = simdgroup_index_in_threadgroup;
    const int r = (int(threadgroup_position_in_grid.x) * SG + int(sg)) * 4 + int(sub);
    const int slot = int(threadgroup_position_in_grid.y);
    const int m = int(threadgroup_position_in_grid.z);
    if (r >= INTER) return;

    const uint e = inds[m * TOP_K + slot];
    const float gate_cap = caps[0];
    const float up_cap = caps[1];

    constexpr int HGROUPS = K / 16;
    constexpr int SGROUPS = K / 32;
    constexpr int W_STRIDE = K / 4;
    const device bfloat* xm = x + size_t(m) * K;
    const size_t w_base_g = (size_t(e) * (2 * INTER) + size_t(2 * r)) * W_STRIDE;
    const size_t w_base_u = w_base_g + W_STRIDE;
    const size_t s_base_g = (size_t(e) * (2 * INTER) + size_t(2 * r)) * SGROUPS;
    const size_t s_base_u = s_base_g + SGROUPS;

    float acc_g = 0.0f;
    float acc_u = 0.0f;
    for (int hg = int(slane); hg < HGROUPS; hg += 8) {
        const uint4 wg = *(const device uint4*)(w + w_base_g + hg * 4);
        const float sg_val = decode_fp8_e8m0(scales[s_base_g + (hg >> 1)]);
        const uint4 wu = *(const device uint4*)(w + w_base_u + hg * 4);
        const float su_val = decode_fp8_e8m0(scales[s_base_u + (hg >> 1)]);
        auto xp = (const device vec<bfloat, 4>*)(xm + hg * 16);
        const vec<bfloat, 4> x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
        acc_g += (dot4_fp8(wg.x, x0) + dot4_fp8(wg.y, x1) + dot4_fp8(wg.z, x2) + dot4_fp8(wg.w, x3)) * sg_val;
        acc_u += (dot4_fp8(wu.x, x0) + dot4_fp8(wu.y, x1) + dot4_fp8(wu.z, x2) + dot4_fp8(wu.w, x3)) * su_val;
    }

    const float gate = float(bfloat(reduce8(acc_g)));
    const float up = float(bfloat(reduce8(acc_u)));
    if (slane == 0) {
        out[(size_t(m) * TOP_K + slot) * INTER + r] = bfloat(softcapped_swiglu(gate, up, gate_cap, up_cap));
    }
"""

_M1_W2_SPLIT_NVFP4_SRC = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sub = lane >> 3;
    const uint slane = lane & 7u;
    const uint sg = simdgroup_index_in_threadgroup;
    const int d = (int(threadgroup_position_in_grid.x) * SG + int(sg)) * 4 + int(sub);
    const int slot = int(threadgroup_position_in_grid.y);
    const int m = int(threadgroup_position_in_grid.z);
    if (d >= D) return;

    const uint e = inds[m * TOP_K + slot];
    const float oscale = gscale[e] * (1.0f / 2688.0f);
    const device bfloat* hm = h + (size_t(m) * TOP_K + slot) * INTER;

    constexpr int GROUPS = INTER / 16;
    constexpr int W_STRIDE = INTER / 8;
    const size_t w_base = (size_t(e) * D + size_t(d)) * W_STRIDE;
    const size_t s_base = (size_t(e) * D + size_t(d)) * GROUPS;

    float acc = 0.0f;
    for (int g = int(slane); g < GROUPS; g += 8) {
        const uint2 ww = *(const device uint2*)(w + w_base + g * 2);
        const float sv = decode_fp8_e4m3(scales[s_base + g]);
        auto hp = (const device vec<bfloat, 4>*)(hm + g * 16);
        acc += (dot8_nvfp4(ww.x, hp[0], hp[1]) + dot8_nvfp4(ww.y, hp[2], hp[3])) * sv;
    }
    const bfloat yk = bfloat(reduce8(acc) * oscale);
    if (slane == 0) {
        ys[(size_t(m) * TOP_K + slot) * D + d] = yk;
    }
"""

_M1_W2_SPLIT_MXFP8_SRC = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sub = lane >> 3;
    const uint slane = lane & 7u;
    const uint sg = simdgroup_index_in_threadgroup;
    const int d = (int(threadgroup_position_in_grid.x) * SG + int(sg)) * 4 + int(sub);
    const int slot = int(threadgroup_position_in_grid.y);
    const int m = int(threadgroup_position_in_grid.z);
    if (d >= D) return;

    const uint e = inds[m * TOP_K + slot];
    const device bfloat* hm = h + (size_t(m) * TOP_K + slot) * INTER;

    constexpr int HGROUPS = INTER / 16;
    constexpr int SGROUPS = INTER / 32;
    constexpr int W_STRIDE = INTER / 4;
    const size_t w_base = (size_t(e) * D + size_t(d)) * W_STRIDE;
    const size_t s_base = (size_t(e) * D + size_t(d)) * SGROUPS;

    float acc = 0.0f;
    for (int hg = int(slane); hg < HGROUPS; hg += 8) {
        const uint4 ww = *(const device uint4*)(w + w_base + hg * 4);
        const float sv = decode_fp8_e8m0(scales[s_base + (hg >> 1)]);
        auto hp = (const device vec<bfloat, 4>*)(hm + hg * 16);
        acc += (dot4_fp8(ww.x, hp[0]) + dot4_fp8(ww.y, hp[1]) + dot4_fp8(ww.z, hp[2]) + dot4_fp8(ww.w, hp[3])) * sv;
    }
    const bfloat yk = bfloat(reduce8(acc));
    if (slane == 0) {
        ys[(size_t(m) * TOP_K + slot) * D + d] = yk;
    }
"""

_GRP_W13_NVFP4_SRC = r"""
    const int e = active_list[int(threadgroup_position_in_grid.y)];
    if (e < 0) return;
    const int beg = offsets[e];
    const int cnt = offsets[e + 1] - beg;
    if (cnt == 0) return;

    const uint lane = thread_index_in_simdgroup;
    const uint sub = lane >> 3;
    const uint slane = lane & 7u;
    const uint sg = simdgroup_index_in_threadgroup;
    const int r = (int(threadgroup_position_in_grid.x) * SG + int(sg)) * 4 + int(sub);
    if (r >= INTER) return;

    const float oscale = gscale[e] * (1.0f / 2688.0f);
    const float gate_cap = caps[0];
    const float up_cap = caps[1];

    constexpr int PAIRS = K / 32;
    constexpr int GROUPS = K / 16;
    constexpr int W_STRIDE = K / 8;
    const size_t w_base_g = (size_t(e) * (2 * INTER) + size_t(2 * r)) * W_STRIDE;
    const size_t w_base_u = w_base_g + W_STRIDE;
    const size_t s_base_g = (size_t(e) * (2 * INTER) + size_t(2 * r)) * GROUPS;
    const size_t s_base_u = s_base_g + GROUPS;

    const int z_id = int(threadgroup_position_in_grid.z);
    for (int t0 = z_id * MT; t0 < cnt; t0 += ZSTEP) {
        const int tn = min(MT, cnt - t0);
        int tok_ids[MT];
        float acc_g[MT];
        float acc_u[MT];
        for (int ti = 0; ti < MT; ++ti) {
            acc_g[ti] = 0.0f;
            acc_u[ti] = 0.0f;
            tok_ids[ti] = (ti < tn) ? int(sorted_slots[beg + t0 + ti] / TOP_K) : 0;
        }

        if (tn == 1) {
            const device bfloat* xm = x + size_t(tok_ids[0]) * K;
            float ag = 0.0f, au = 0.0f;
            for (int p = int(slane); p < PAIRS; p += 8) {
                const uint4 wg = *(const device uint4*)(w + w_base_g + p * 4);
                const uchar2 sg2 = *(const device uchar2*)(scales + s_base_g + p * 2);
                const uint4 wu = *(const device uint4*)(w + w_base_u + p * 4);
                const uchar2 su2 = *(const device uchar2*)(scales + s_base_u + p * 2);
                auto xp = (const device vec<bfloat, 4>*)(xm + p * 32);
                {
                    const vec<bfloat, 4> x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
                    ag += (dot8_nvfp4(wg.x, x0, x1) + dot8_nvfp4(wg.y, x2, x3)) * decode_fp8_e4m3(sg2.x);
                    au += (dot8_nvfp4(wu.x, x0, x1) + dot8_nvfp4(wu.y, x2, x3)) * decode_fp8_e4m3(su2.x);
                }
                {
                    const vec<bfloat, 4> x0 = xp[4], x1 = xp[5], x2 = xp[6], x3 = xp[7];
                    ag += (dot8_nvfp4(wg.z, x0, x1) + dot8_nvfp4(wg.w, x2, x3)) * decode_fp8_e4m3(sg2.y);
                    au += (dot8_nvfp4(wu.z, x0, x1) + dot8_nvfp4(wu.w, x2, x3)) * decode_fp8_e4m3(su2.y);
                }
            }
            const float gate = float(bfloat(reduce8(ag) * oscale));
            const float up = float(bfloat(reduce8(au) * oscale));
            if (slane == 0) {
                h_sorted[size_t(beg + t0) * INTER + r] = bfloat(softcapped_swiglu(gate, up, gate_cap, up_cap));
            }
        } else {
            for (int p = int(slane); p < PAIRS; p += 8) {
                const uint4 wg = *(const device uint4*)(w + w_base_g + p * 4);
                const uchar2 sg2 = *(const device uchar2*)(scales + s_base_g + p * 2);
                const float sg0 = decode_fp8_e4m3(sg2.x), sg1 = decode_fp8_e4m3(sg2.y);
                const uint4 wu = *(const device uint4*)(w + w_base_u + p * 4);
                const uchar2 su2 = *(const device uchar2*)(scales + s_base_u + p * 2);
                const float su0 = decode_fp8_e4m3(su2.x), su1 = decode_fp8_e4m3(su2.y);

                for (int ti = 0; ti < tn; ++ti) {
                    auto xp = (const device vec<bfloat, 4>*)(x + size_t(tok_ids[ti]) * K + p * 32);
                    {
                        const vec<bfloat, 4> x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
                        acc_g[ti] += (dot8_nvfp4(wg.x, x0, x1) + dot8_nvfp4(wg.y, x2, x3)) * sg0;
                        acc_u[ti] += (dot8_nvfp4(wu.x, x0, x1) + dot8_nvfp4(wu.y, x2, x3)) * su0;
                    }
                    {
                        const vec<bfloat, 4> x0 = xp[4], x1 = xp[5], x2 = xp[6], x3 = xp[7];
                        acc_g[ti] += (dot8_nvfp4(wg.z, x0, x1) + dot8_nvfp4(wg.w, x2, x3)) * sg1;
                        acc_u[ti] += (dot8_nvfp4(wu.z, x0, x1) + dot8_nvfp4(wu.w, x2, x3)) * su1;
                    }
                }
            }
            for (int ti = 0; ti < tn; ++ti) {
                const float gate = float(bfloat(reduce8(acc_g[ti]) * oscale));
                const float up = float(bfloat(reduce8(acc_u[ti]) * oscale));
                if (slane == 0) {
                    h_sorted[size_t(beg + t0 + ti) * INTER + r] = bfloat(softcapped_swiglu(gate, up, gate_cap, up_cap));
                }
            }
        }
    }
"""

_GRP_W13_MXFP8_SRC = r"""
    const int e = active_list[int(threadgroup_position_in_grid.y)];
    if (e < 0) return;
    const int beg = offsets[e];
    const int cnt = offsets[e + 1] - beg;
    if (cnt == 0) return;

    const uint lane = thread_index_in_simdgroup;
    const uint sub = lane >> 3;
    const uint slane = lane & 7u;
    const uint sg = simdgroup_index_in_threadgroup;
    const int r = (int(threadgroup_position_in_grid.x) * SG + int(sg)) * 4 + int(sub);
    if (r >= INTER) return;

    const float gate_cap = caps[0];
    const float up_cap = caps[1];

    constexpr int HGROUPS = K / 16;
    constexpr int SGROUPS = K / 32;
    constexpr int W_STRIDE = K / 4;
    const size_t w_base_g = (size_t(e) * (2 * INTER) + size_t(2 * r)) * W_STRIDE;
    const size_t w_base_u = w_base_g + W_STRIDE;
    const size_t s_base_g = (size_t(e) * (2 * INTER) + size_t(2 * r)) * SGROUPS;
    const size_t s_base_u = s_base_g + SGROUPS;

    const int z_id = int(threadgroup_position_in_grid.z);
    for (int t0 = z_id * MT; t0 < cnt; t0 += ZSTEP) {
        const int tn = min(MT, cnt - t0);
        int tok_ids[MT];
        float acc_g[MT];
        float acc_u[MT];
        for (int ti = 0; ti < MT; ++ti) {
            acc_g[ti] = 0.0f;
            acc_u[ti] = 0.0f;
            tok_ids[ti] = (ti < tn) ? int(sorted_slots[beg + t0 + ti] / TOP_K) : 0;
        }

        if (tn == 1) {
            const device bfloat* xm = x + size_t(tok_ids[0]) * K;
            float ag = 0.0f, au = 0.0f;
            for (int hg = int(slane); hg < HGROUPS; hg += 8) {
                const uint4 wg = *(const device uint4*)(w + w_base_g + hg * 4);
                const float sg_val = decode_fp8_e8m0(scales[s_base_g + (hg >> 1)]);
                const uint4 wu = *(const device uint4*)(w + w_base_u + hg * 4);
                const float su_val = decode_fp8_e8m0(scales[s_base_u + (hg >> 1)]);
                auto xp = (const device vec<bfloat, 4>*)(xm + hg * 16);
                const vec<bfloat, 4> x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
                ag += (dot4_fp8(wg.x, x0) + dot4_fp8(wg.y, x1) + dot4_fp8(wg.z, x2) + dot4_fp8(wg.w, x3)) * sg_val;
                au += (dot4_fp8(wu.x, x0) + dot4_fp8(wu.y, x1) + dot4_fp8(wu.z, x2) + dot4_fp8(wu.w, x3)) * su_val;
            }
            const float gate = float(bfloat(reduce8(ag)));
            const float up = float(bfloat(reduce8(au)));
            if (slane == 0) {
                h_sorted[size_t(beg + t0) * INTER + r] = bfloat(softcapped_swiglu(gate, up, gate_cap, up_cap));
            }
        } else {
            for (int hg = int(slane); hg < HGROUPS; hg += 8) {
                const uint4 wg = *(const device uint4*)(w + w_base_g + hg * 4);
                const float sg_val = decode_fp8_e8m0(scales[s_base_g + (hg >> 1)]);
                const uint4 wu = *(const device uint4*)(w + w_base_u + hg * 4);
                const float su_val = decode_fp8_e8m0(scales[s_base_u + (hg >> 1)]);
                for (int ti = 0; ti < tn; ++ti) {
                    auto xp = (const device vec<bfloat, 4>*)(x + size_t(tok_ids[ti]) * K + hg * 16);
                    const vec<bfloat, 4> x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
                    acc_g[ti] += (dot4_fp8(wg.x, x0) + dot4_fp8(wg.y, x1) + dot4_fp8(wg.z, x2) + dot4_fp8(wg.w, x3)) * sg_val;
                    acc_u[ti] += (dot4_fp8(wu.x, x0) + dot4_fp8(wu.y, x1) + dot4_fp8(wu.z, x2) + dot4_fp8(wu.w, x3)) * su_val;
                }
            }
            for (int ti = 0; ti < tn; ++ti) {
                const float gate = float(bfloat(reduce8(acc_g[ti])));
                const float up = float(bfloat(reduce8(acc_u[ti])));
                if (slane == 0) {
                    h_sorted[size_t(beg + t0 + ti) * INTER + r] = bfloat(softcapped_swiglu(gate, up, gate_cap, up_cap));
                }
            }
        }
    }
"""

_GRP_W2_NVFP4_SRC = r"""
    const int e = active_list[int(threadgroup_position_in_grid.y)];
    if (e < 0) return;
    const int beg = offsets[e];
    const int cnt = offsets[e + 1] - beg;
    if (cnt == 0) return;

    const uint lane = thread_index_in_simdgroup;
    const uint sub = lane >> 3;
    const uint slane = lane & 7u;
    const uint sg = simdgroup_index_in_threadgroup;
    const int d = (int(threadgroup_position_in_grid.x) * SG + int(sg)) * 4 + int(sub);
    if (d >= D) return;

    const float oscale = gscale[e] * (1.0f / 2688.0f);
    constexpr int GROUPS = INTER / 16;
    constexpr int W_STRIDE = INTER / 8;
    const size_t w_base = (size_t(e) * D + size_t(d)) * W_STRIDE;
    const size_t s_base = (size_t(e) * D + size_t(d)) * GROUPS;

    const int z_id = int(threadgroup_position_in_grid.z);
    for (int t0 = z_id * MT; t0 < cnt; t0 += ZSTEP) {
        const int tn = min(MT, cnt - t0);
        uint slot_ids[MT];
        float acc[MT];
        for (int ti = 0; ti < MT; ++ti) {
            acc[ti] = 0.0f;
            slot_ids[ti] = (ti < tn) ? sorted_slots[beg + t0 + ti] : 0;
        }

        if (tn == 1) {
            const device bfloat* hm = h_sorted + size_t(beg + t0) * INTER;
            float a = 0.0f;
            for (int g = int(slane); g < GROUPS; g += 8) {
                const uint2 ww = *(const device uint2*)(w + w_base + g * 2);
                const float sv = decode_fp8_e4m3(scales[s_base + g]);
                auto hp = (const device vec<bfloat, 4>*)(hm + g * 16);
                a += (dot8_nvfp4(ww.x, hp[0], hp[1]) + dot8_nvfp4(ww.y, hp[2], hp[3])) * sv;
            }
            const bfloat yk = bfloat(reduce8(a) * oscale);
            if (slane == 0) ys[size_t(slot_ids[0]) * D + d] = yk;
        } else {
            for (int g = int(slane); g < GROUPS; g += 8) {
                const uint2 ww = *(const device uint2*)(w + w_base + g * 2);
                const float sv = decode_fp8_e4m3(scales[s_base + g]);
                for (int ti = 0; ti < tn; ++ti) {
                    auto hp = (const device vec<bfloat, 4>*)(h_sorted + size_t(beg + t0 + ti) * INTER + g * 16);
                    acc[ti] += (dot8_nvfp4(ww.x, hp[0], hp[1]) + dot8_nvfp4(ww.y, hp[2], hp[3])) * sv;
                }
            }
            for (int ti = 0; ti < tn; ++ti) {
                const bfloat yk = bfloat(reduce8(acc[ti]) * oscale);
                if (slane == 0) ys[size_t(slot_ids[ti]) * D + d] = yk;
            }
        }
    }
"""

_GRP_W2_MXFP8_SRC = r"""
    const int e = active_list[int(threadgroup_position_in_grid.y)];
    if (e < 0) return;
    const int beg = offsets[e];
    const int cnt = offsets[e + 1] - beg;
    if (cnt == 0) return;

    const uint lane = thread_index_in_simdgroup;
    const uint sub = lane >> 3;
    const uint slane = lane & 7u;
    const uint sg = simdgroup_index_in_threadgroup;
    const int d = (int(threadgroup_position_in_grid.x) * SG + int(sg)) * 4 + int(sub);
    if (d >= D) return;

    constexpr int HGROUPS = INTER / 16;
    constexpr int SGROUPS = INTER / 32;
    constexpr int W_STRIDE = INTER / 4;
    const size_t w_base = (size_t(e) * D + size_t(d)) * W_STRIDE;
    const size_t s_base = (size_t(e) * D + size_t(d)) * SGROUPS;

    const int z_id = int(threadgroup_position_in_grid.z);
    for (int t0 = z_id * MT; t0 < cnt; t0 += ZSTEP) {
        const int tn = min(MT, cnt - t0);
        uint slot_ids[MT];
        float acc[MT];
        for (int ti = 0; ti < MT; ++ti) {
            acc[ti] = 0.0f;
            slot_ids[ti] = (ti < tn) ? sorted_slots[beg + t0 + ti] : 0;
        }

        if (tn == 1) {
            const device bfloat* hm = h_sorted + size_t(beg + t0) * INTER;
            float a = 0.0f;
            for (int hg = int(slane); hg < HGROUPS; hg += 8) {
                const uint4 ww = *(const device uint4*)(w + w_base + hg * 4);
                const float sv = decode_fp8_e8m0(scales[s_base + (hg >> 1)]);
                auto hp = (const device vec<bfloat, 4>*)(hm + hg * 16);
                a += (dot4_fp8(ww.x, hp[0]) + dot4_fp8(ww.y, hp[1]) + dot4_fp8(ww.z, hp[2]) + dot4_fp8(ww.w, hp[3])) * sv;
            }
            const bfloat yk = bfloat(reduce8(a));
            if (slane == 0) ys[size_t(slot_ids[0]) * D + d] = yk;
        } else {
            for (int hg = int(slane); hg < HGROUPS; hg += 8) {
                const uint4 ww = *(const device uint4*)(w + w_base + hg * 4);
                const float sv = decode_fp8_e8m0(scales[s_base + (hg >> 1)]);
                for (int ti = 0; ti < tn; ++ti) {
                    auto hp = (const device vec<bfloat, 4>*)(h_sorted + size_t(beg + t0 + ti) * INTER + hg * 16);
                    acc[ti] += (dot4_fp8(ww.x, hp[0]) + dot4_fp8(ww.y, hp[1]) + dot4_fp8(ww.z, hp[2]) + dot4_fp8(ww.w, hp[3])) * sv;
                }
            }
            for (int ti = 0; ti < tn; ++ti) {
                const bfloat yk = bfloat(reduce8(acc[ti]));
                if (slane == 0) ys[size_t(slot_ids[ti]) * D + d] = yk;
            }
        }
    }
"""

_MK: dict[str, Any] = {}
_CAPS: dict[tuple[float, float], mx.array] = {}


def _get_caps(gate_cap: float, up_cap: float) -> mx.array:
    key = (float(gate_cap), float(up_cap))
    arr = _CAPS.get(key)
    if arr is None:
        arr = _CAPS[key] = mx.array([gate_cap, up_cap], dtype=mx.float32)
    return arr


def _mk(name: str, inputs: list[str], outputs: list[str], source: str):
    k = _MK.get(name)
    if k is None:
        k = _MK[name] = mx.fast.metal_kernel(
            name="inkling_moe3_" + name,
            input_names=inputs,
            output_names=outputs,
            source=source,
            header=_DECODERS_HEADER,
        )
    return k


def moe_m1_experts(
    x: mx.array,
    inds: mx.array,
    moe,
    *,
    sg_w13: int | None = None,
    sg_w2: int | None = None,
) -> mx.array:
    """Direct M=1..2 MoE (zero bucket, zero empty experts, 1 expert per TG) returning ys (M, TOP_K, D)."""
    m = x.shape[0]
    k_dim = x.shape[-1]
    inter = moe.down_proj.input_dims
    d_out = moe.down_proj.output_dims
    top_k = moe.top_k
    mode = moe.w13_proj.mode
    if sg_w13 is None:
        sg_w13 = 8 if mode == "nvfp4" else 4
    if sg_w2 is None:
        sg_w2 = 8
    caps = _get_caps(moe.gate_cap, moe.up_cap)
    inds_u = inds.reshape(m, top_k).astype(mx.uint32)

    nx13 = (inter + sg_w13 * 4 - 1) // (sg_w13 * 4)
    nx2 = (d_out + sg_w2 * 4 - 1) // (sg_w2 * 4)

    if mode == "nvfp4":
        h = _mk(
            "m1_w13_nvfp4",
            ["x", "w", "scales", "gscale", "inds", "caps"],
            ["out"],
            _M1_W13_NVFP4_SRC,
        )(
            inputs=[x.reshape(m, k_dim), moe.w13_proj.weight, moe.w13_proj.scales, moe.w13_proj.global_scale, inds_u, caps],
            template=[("K", k_dim), ("INTER", inter), ("TOP_K", top_k), ("SG", sg_w13)],
            grid=(nx13 * 32 * sg_w13, top_k, m),
            threadgroup=(32 * sg_w13, 1, 1),
            output_shapes=[(m, top_k, inter)],
            output_dtypes=[mx.bfloat16],
        )[0]
        ys = _mk(
            "m1_w2_split_nvfp4",
            ["h", "w", "scales", "gscale", "inds"],
            ["ys"],
            _M1_W2_SPLIT_NVFP4_SRC,
        )(
            inputs=[h, moe.down_proj.weight, moe.down_proj.scales, moe.down_proj.global_scale, inds_u],
            template=[("D", d_out), ("INTER", inter), ("TOP_K", top_k), ("SG", sg_w2)],
            grid=(nx2 * 32 * sg_w2, top_k, m),
            threadgroup=(32 * sg_w2, 1, 1),
            output_shapes=[(m, top_k, d_out)],
            output_dtypes=[mx.bfloat16],
        )[0]
        return ys
    else:
        h = _mk(
            "m1_w13_mxfp8",
            ["x", "w", "scales", "inds", "caps"],
            ["out"],
            _M1_W13_MXFP8_SRC,
        )(
            inputs=[x.reshape(m, k_dim), moe.w13_proj.weight, moe.w13_proj.scales, inds_u, caps],
            template=[("K", k_dim), ("INTER", inter), ("TOP_K", top_k), ("SG", sg_w13)],
            grid=(nx13 * 32 * sg_w13, top_k, m),
            threadgroup=(32 * sg_w13, 1, 1),
            output_shapes=[(m, top_k, inter)],
            output_dtypes=[mx.bfloat16],
        )[0]
        ys = _mk(
            "m1_w2_split_mxfp8",
            ["h", "w", "scales", "inds"],
            ["ys"],
            _M1_W2_SPLIT_MXFP8_SRC,
        )(
            inputs=[h, moe.down_proj.weight, moe.down_proj.scales, inds_u],
            template=[("D", d_out), ("INTER", inter), ("TOP_K", top_k), ("SG", sg_w2)],
            grid=(nx2 * 32 * sg_w2, top_k, m),
            threadgroup=(32 * sg_w2, 1, 1),
            output_shapes=[(m, top_k, d_out)],
            output_dtypes=[mx.bfloat16],
        )[0]
        return ys


_GRP_W13_NVFP4_TILES_SRC = r"""
    const int tile_idx = int(threadgroup_position_in_grid.y);
    const int row_blk  = int(threadgroup_position_in_grid.x);
    const int e = tile_expert[tile_idx];
    if (e < 0) return;
    const int t0 = tile_t0[tile_idx];
    const int beg = offsets[e];
    const int cnt = offsets[e + 1] - beg;
    const int tn = min(MT, cnt - t0);

    threadgroup int tg_tok[MT];
    const uint tid_tg = thread_index_in_threadgroup;
    if (int(tid_tg) < tn) tg_tok[tid_tg] = int(sorted_slots[beg + t0 + tid_tg] / TOP_K);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint lane = thread_index_in_simdgroup;
    const uint sub = lane >> 3;
    const uint slane = lane & 7u;
    const uint sg = simdgroup_index_in_threadgroup;
    const int r = (row_blk * SG + int(sg)) * 4 + int(sub);
    if (r >= INTER) return;

    const float oscale = gscale[e] * (1.0f / 2688.0f);
    const float gate_cap = caps[0];
    const float up_cap = caps[1];
    constexpr int PAIRS = K / 32;
    constexpr int GROUPS = K / 16;
    constexpr int W_STRIDE = K / 8;
    const size_t w_base_g = (size_t(e) * (2 * INTER) + size_t(2 * r)) * W_STRIDE;
    const size_t w_base_u = w_base_g + W_STRIDE;
    const size_t s_base_g = (size_t(e) * (2 * INTER) + size_t(2 * r)) * GROUPS;
    const size_t s_base_u = s_base_g + GROUPS;

    if (tn == 1) {
        const device bfloat* xm = x + size_t(tg_tok[0]) * K;
        float ag = 0.0f, au = 0.0f;
        for (int p = int(slane); p < PAIRS; p += 8) {
            const uint4 wg = *(const device uint4*)(w + w_base_g + p * 4);
            const uchar2 sg2 = *(const device uchar2*)(scales + s_base_g + p * 2);
            const uint4 wu = *(const device uint4*)(w + w_base_u + p * 4);
            const uchar2 su2 = *(const device uchar2*)(scales + s_base_u + p * 2);
            const device uint4* xq = (const device uint4*)(xm + p * 32);
            {
                const uint4 q0 = xq[0], q1 = xq[1];
                const vec<bfloat, 4> x0 = as_type<vec<bfloat, 4>>(q0.xy), x1 = as_type<vec<bfloat, 4>>(q0.zw);
                const vec<bfloat, 4> x2 = as_type<vec<bfloat, 4>>(q1.xy), x3 = as_type<vec<bfloat, 4>>(q1.zw);
                ag += (dot8_nvfp4(wg.x, x0, x1) + dot8_nvfp4(wg.y, x2, x3)) * decode_fp8_e4m3(sg2.x);
                au += (dot8_nvfp4(wu.x, x0, x1) + dot8_nvfp4(wu.y, x2, x3)) * decode_fp8_e4m3(su2.x);
            }
            {
                const uint4 q2 = xq[2], q3 = xq[3];
                const vec<bfloat, 4> x0 = as_type<vec<bfloat, 4>>(q2.xy), x1 = as_type<vec<bfloat, 4>>(q2.zw);
                const vec<bfloat, 4> x2 = as_type<vec<bfloat, 4>>(q3.xy), x3 = as_type<vec<bfloat, 4>>(q3.zw);
                ag += (dot8_nvfp4(wg.z, x0, x1) + dot8_nvfp4(wg.w, x2, x3)) * decode_fp8_e4m3(sg2.y);
                au += (dot8_nvfp4(wu.z, x0, x1) + dot8_nvfp4(wu.w, x2, x3)) * decode_fp8_e4m3(su2.y);
            }
        }
        const float gate = float(bfloat(reduce8(ag) * oscale));
        const float up = float(bfloat(reduce8(au) * oscale));
        if (slane == 0) h_sorted[size_t(beg + t0) * INTER + r] = bfloat(softcapped_swiglu(gate, up, gate_cap, up_cap));
    } else if (tn == MT) {
        float acc_g[MT];
        float acc_u[MT];
        #pragma unroll
        for (int ti = 0; ti < MT; ++ti) { acc_g[ti] = 0.0f; acc_u[ti] = 0.0f; }
        for (int p = int(slane); p < PAIRS; p += 8) {
            const uint4 wg = *(const device uint4*)(w + w_base_g + p * 4);
            const uchar2 sg2 = *(const device uchar2*)(scales + s_base_g + p * 2);
            const float sg0 = decode_fp8_e4m3(sg2.x), sg1 = decode_fp8_e4m3(sg2.y);
            const uint4 wu = *(const device uint4*)(w + w_base_u + p * 4);
            const uchar2 su2 = *(const device uchar2*)(scales + s_base_u + p * 2);
            const float su0 = decode_fp8_e4m3(su2.x), su1 = decode_fp8_e4m3(su2.y);
            {
                half4 g0, g1, g2, g3, u0, u1, u2, u3;
                unpack8_nvfp4_h(wg.x, g0, g1); unpack8_nvfp4_h(wg.y, g2, g3);
                unpack8_nvfp4_h(wu.x, u0, u1); unpack8_nvfp4_h(wu.y, u2, u3);
                #pragma unroll
                for (int ti = 0; ti < MT; ++ti) {
                    const device uint4* xq = (const device uint4*)(x + size_t(tg_tok[ti]) * K + p * 32);
                    const uint4 q0 = xq[0], q1 = xq[1];
                    const float4 fx0 = float4(as_type<vec<bfloat, 4>>(q0.xy));
                    const float4 fx1 = float4(as_type<vec<bfloat, 4>>(q0.zw));
                    const float4 fx2 = float4(as_type<vec<bfloat, 4>>(q1.xy));
                    const float4 fx3 = float4(as_type<vec<bfloat, 4>>(q1.zw));
                    acc_g[ti] += (dot8_hf(g0, g1, fx0, fx1) + dot8_hf(g2, g3, fx2, fx3)) * sg0;
                    acc_u[ti] += (dot8_hf(u0, u1, fx0, fx1) + dot8_hf(u2, u3, fx2, fx3)) * su0;
                }
            }
            {
                half4 g0, g1, g2, g3, u0, u1, u2, u3;
                unpack8_nvfp4_h(wg.z, g0, g1); unpack8_nvfp4_h(wg.w, g2, g3);
                unpack8_nvfp4_h(wu.z, u0, u1); unpack8_nvfp4_h(wu.w, u2, u3);
                #pragma unroll
                for (int ti = 0; ti < MT; ++ti) {
                    const device uint4* xq = (const device uint4*)(x + size_t(tg_tok[ti]) * K + p * 32 + 16);
                    const uint4 q0 = xq[0], q1 = xq[1];
                    const float4 fx0 = float4(as_type<vec<bfloat, 4>>(q0.xy));
                    const float4 fx1 = float4(as_type<vec<bfloat, 4>>(q0.zw));
                    const float4 fx2 = float4(as_type<vec<bfloat, 4>>(q1.xy));
                    const float4 fx3 = float4(as_type<vec<bfloat, 4>>(q1.zw));
                    acc_g[ti] += (dot8_hf(g0, g1, fx0, fx1) + dot8_hf(g2, g3, fx2, fx3)) * sg1;
                    acc_u[ti] += (dot8_hf(u0, u1, fx0, fx1) + dot8_hf(u2, u3, fx2, fx3)) * su1;
                }
            }
        }
        #pragma unroll
        for (int ti = 0; ti < MT; ++ti) {
            const float gate = float(bfloat(reduce8(acc_g[ti]) * oscale));
            const float up = float(bfloat(reduce8(acc_u[ti]) * oscale));
            if (slane == 0) h_sorted[size_t(beg + t0 + ti) * INTER + r] = bfloat(softcapped_swiglu(gate, up, gate_cap, up_cap));
        }
    } else {
        float acc_g[MT];
        float acc_u[MT];
        #pragma unroll
        for (int ti = 0; ti < MT; ++ti) { acc_g[ti] = 0.0f; acc_u[ti] = 0.0f; }
        for (int p = int(slane); p < PAIRS; p += 8) {
            const uint4 wg = *(const device uint4*)(w + w_base_g + p * 4);
            const uchar2 sg2 = *(const device uchar2*)(scales + s_base_g + p * 2);
            const float sg0 = decode_fp8_e4m3(sg2.x), sg1 = decode_fp8_e4m3(sg2.y);
            const uint4 wu = *(const device uint4*)(w + w_base_u + p * 4);
            const uchar2 su2 = *(const device uchar2*)(scales + s_base_u + p * 2);
            const float su0 = decode_fp8_e4m3(su2.x), su1 = decode_fp8_e4m3(su2.y);
            {
                half4 g0, g1, g2, g3, u0, u1, u2, u3;
                unpack8_nvfp4_h(wg.x, g0, g1); unpack8_nvfp4_h(wg.y, g2, g3);
                unpack8_nvfp4_h(wu.x, u0, u1); unpack8_nvfp4_h(wu.y, u2, u3);
                #pragma unroll
                for (int ti = 0; ti < MT; ++ti) {
                    if (ti < tn) {
                        const device uint4* xq = (const device uint4*)(x + size_t(tg_tok[ti]) * K + p * 32);
                        const uint4 q0 = xq[0], q1 = xq[1];
                        const float4 fx0 = float4(as_type<vec<bfloat, 4>>(q0.xy));
                        const float4 fx1 = float4(as_type<vec<bfloat, 4>>(q0.zw));
                        const float4 fx2 = float4(as_type<vec<bfloat, 4>>(q1.xy));
                        const float4 fx3 = float4(as_type<vec<bfloat, 4>>(q1.zw));
                        acc_g[ti] += (dot8_hf(g0, g1, fx0, fx1) + dot8_hf(g2, g3, fx2, fx3)) * sg0;
                        acc_u[ti] += (dot8_hf(u0, u1, fx0, fx1) + dot8_hf(u2, u3, fx2, fx3)) * su0;
                    }
                }
            }
            {
                half4 g0, g1, g2, g3, u0, u1, u2, u3;
                unpack8_nvfp4_h(wg.z, g0, g1); unpack8_nvfp4_h(wg.w, g2, g3);
                unpack8_nvfp4_h(wu.z, u0, u1); unpack8_nvfp4_h(wu.w, u2, u3);
                #pragma unroll
                for (int ti = 0; ti < MT; ++ti) {
                    if (ti < tn) {
                        const device uint4* xq = (const device uint4*)(x + size_t(tg_tok[ti]) * K + p * 32 + 16);
                        const uint4 q0 = xq[0], q1 = xq[1];
                        const float4 fx0 = float4(as_type<vec<bfloat, 4>>(q0.xy));
                        const float4 fx1 = float4(as_type<vec<bfloat, 4>>(q0.zw));
                        const float4 fx2 = float4(as_type<vec<bfloat, 4>>(q1.xy));
                        const float4 fx3 = float4(as_type<vec<bfloat, 4>>(q1.zw));
                        acc_g[ti] += (dot8_hf(g0, g1, fx0, fx1) + dot8_hf(g2, g3, fx2, fx3)) * sg1;
                        acc_u[ti] += (dot8_hf(u0, u1, fx0, fx1) + dot8_hf(u2, u3, fx2, fx3)) * su1;
                    }
                }
            }
        }
        #pragma unroll
        for (int ti = 0; ti < MT; ++ti) {
            if (ti < tn) {
                const float gate = float(bfloat(reduce8(acc_g[ti]) * oscale));
                const float up = float(bfloat(reduce8(acc_u[ti]) * oscale));
                if (slane == 0) h_sorted[size_t(beg + t0 + ti) * INTER + r] = bfloat(softcapped_swiglu(gate, up, gate_cap, up_cap));
            }
        }
    }
"""

_GRP_W2_NVFP4_TILES_SRC = r"""
    const int tile_idx = int(threadgroup_position_in_grid.y);
    const int row_blk  = int(threadgroup_position_in_grid.x);
    const int e = tile_expert[tile_idx];
    if (e < 0) return;
    const int t0 = tile_t0[tile_idx];
    const int beg = offsets[e];
    const int cnt = offsets[e + 1] - beg;
    const int tn = min(MT, cnt - t0);

    threadgroup uint tg_slot[MT];
    const uint tid_tg = thread_index_in_threadgroup;
    if (int(tid_tg) < tn) tg_slot[tid_tg] = sorted_slots[beg + t0 + tid_tg];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint lane = thread_index_in_simdgroup;
    const uint sub = lane >> 3;
    const uint slane = lane & 7u;
    const uint sg = simdgroup_index_in_threadgroup;
    const int d = (row_blk * SG + int(sg)) * 4 + int(sub);
    if (d >= D) return;

    const float oscale = gscale[e] * (1.0f / 2688.0f);
    constexpr int GROUPS = INTER / 16;
    constexpr int W_STRIDE = INTER / 8;
    const size_t w_base = (size_t(e) * D + size_t(d)) * W_STRIDE;
    const size_t s_base = (size_t(e) * D + size_t(d)) * GROUPS;

    if (tn == 1) {
        const device bfloat* hm = h_sorted + size_t(beg + t0) * INTER;
        float a = 0.0f;
        for (int g = int(slane); g < GROUPS; g += 8) {
            const uint2 ww = *(const device uint2*)(w + w_base + g * 2);
            const float sv = decode_fp8_e4m3(scales[s_base + g]);
            const device uint4* hq = (const device uint4*)(hm + g * 16);
            const uint4 q0 = hq[0], q1 = hq[1];
            a += (dot8_nvfp4(ww.x, as_type<vec<bfloat, 4>>(q0.xy), as_type<vec<bfloat, 4>>(q0.zw))
                + dot8_nvfp4(ww.y, as_type<vec<bfloat, 4>>(q1.xy), as_type<vec<bfloat, 4>>(q1.zw))) * sv;
        }
        const bfloat yk = bfloat(reduce8(a) * oscale);
        if (slane == 0) ys[size_t(tg_slot[0]) * D + d] = yk;
    } else if (tn == MT) {
        float acc[MT];
        #pragma unroll
        for (int ti = 0; ti < MT; ++ti) acc[ti] = 0.0f;
        for (int g = int(slane); g < GROUPS; g += 8) {
            const uint2 ww = *(const device uint2*)(w + w_base + g * 2);
            const float sv = decode_fp8_e4m3(scales[s_base + g]);
            half4 f0, f1, f2, f3;
            unpack8_nvfp4_h(ww.x, f0, f1); unpack8_nvfp4_h(ww.y, f2, f3);
            #pragma unroll
            for (int ti = 0; ti < MT; ++ti) {
                const device uint4* hq = (const device uint4*)(h_sorted + size_t(beg + t0 + ti) * INTER + g * 16);
                const uint4 q0 = hq[0], q1 = hq[1];
                acc[ti] += (dot8_h(f0, f1, as_type<vec<bfloat, 4>>(q0.xy), as_type<vec<bfloat, 4>>(q0.zw))
                          + dot8_h(f2, f3, as_type<vec<bfloat, 4>>(q1.xy), as_type<vec<bfloat, 4>>(q1.zw))) * sv;
            }
        }
        #pragma unroll
        for (int ti = 0; ti < MT; ++ti) {
            const bfloat yk = bfloat(reduce8(acc[ti]) * oscale);
            if (slane == 0) ys[size_t(tg_slot[ti]) * D + d] = yk;
        }
    } else {
        float acc[MT];
        #pragma unroll
        for (int ti = 0; ti < MT; ++ti) acc[ti] = 0.0f;
        for (int g = int(slane); g < GROUPS; g += 8) {
            const uint2 ww = *(const device uint2*)(w + w_base + g * 2);
            const float sv = decode_fp8_e4m3(scales[s_base + g]);
            half4 f0, f1, f2, f3;
            unpack8_nvfp4_h(ww.x, f0, f1); unpack8_nvfp4_h(ww.y, f2, f3);
            #pragma unroll
            for (int ti = 0; ti < MT; ++ti) {
                if (ti < tn) {
                    const device uint4* hq = (const device uint4*)(h_sorted + size_t(beg + t0 + ti) * INTER + g * 16);
                    const uint4 q0 = hq[0], q1 = hq[1];
                    acc[ti] += (dot8_h(f0, f1, as_type<vec<bfloat, 4>>(q0.xy), as_type<vec<bfloat, 4>>(q0.zw))
                              + dot8_h(f2, f3, as_type<vec<bfloat, 4>>(q1.xy), as_type<vec<bfloat, 4>>(q1.zw))) * sv;
                }
            }
        }
        #pragma unroll
        for (int ti = 0; ti < MT; ++ti) {
            if (ti < tn) {
                const bfloat yk = bfloat(reduce8(acc[ti]) * oscale);
                if (slane == 0) ys[size_t(tg_slot[ti]) * D + d] = yk;
            }
        }
    }
"""

_GRP_W13_MXFP8_TILES_SRC = r"""
    const int tile_idx = int(threadgroup_position_in_grid.y);
    const int row_blk  = int(threadgroup_position_in_grid.x);
    const int e = tile_expert[tile_idx];
    if (e < 0) return;
    const int t0 = tile_t0[tile_idx];
    const int beg = offsets[e];
    const int cnt = offsets[e + 1] - beg;
    const int tn = min(MT, cnt - t0);

    threadgroup int tg_tok[MT];
    const uint tid_tg = thread_index_in_threadgroup;
    if (int(tid_tg) < tn) tg_tok[tid_tg] = int(sorted_slots[beg + t0 + tid_tg] / TOP_K);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint lane = thread_index_in_simdgroup;
    const uint sub = lane >> 3;
    const uint slane = lane & 7u;
    const uint sg = simdgroup_index_in_threadgroup;
    const int r = (row_blk * SG + int(sg)) * 4 + int(sub);
    if (r >= INTER) return;

    const float gate_cap = caps[0];
    const float up_cap = caps[1];
    constexpr int HGROUPS = K / 16;
    constexpr int SGROUPS = K / 32;
    constexpr int W_STRIDE = K / 4;
    const size_t w_base_g = (size_t(e) * (2 * INTER) + size_t(2 * r)) * W_STRIDE;
    const size_t w_base_u = w_base_g + W_STRIDE;
    const size_t s_base_g = (size_t(e) * (2 * INTER) + size_t(2 * r)) * SGROUPS;
    const size_t s_base_u = s_base_g + SGROUPS;

    if (tn == 1) {
        const device bfloat* xm = x + size_t(tg_tok[0]) * K;
        float ag = 0.0f, au = 0.0f;
        for (int hg = int(slane); hg < HGROUPS; hg += 8) {
            const uint4 wg = *(const device uint4*)(w + w_base_g + hg * 4);
            const float sg_val = decode_fp8_e8m0_x256(scales[s_base_g + (hg >> 1)]);
            const uint4 wu = *(const device uint4*)(w + w_base_u + hg * 4);
            const float su_val = decode_fp8_e8m0_x256(scales[s_base_u + (hg >> 1)]);
            const device uint4* xq = (const device uint4*)(xm + hg * 16);
            const uint4 q0 = xq[0], q1 = xq[1];
            const float4 fx0 = float4(as_type<vec<bfloat, 4>>(q0.xy));
            const float4 fx1 = float4(as_type<vec<bfloat, 4>>(q0.zw));
            const float4 fx2 = float4(as_type<vec<bfloat, 4>>(q1.xy));
            const float4 fx3 = float4(as_type<vec<bfloat, 4>>(q1.zw));
            ag += (dot4_hf(unpack4_fp8_h_div256(wg.x), fx0) + dot4_hf(unpack4_fp8_h_div256(wg.y), fx1)
                 + dot4_hf(unpack4_fp8_h_div256(wg.z), fx2) + dot4_hf(unpack4_fp8_h_div256(wg.w), fx3)) * sg_val;
            au += (dot4_hf(unpack4_fp8_h_div256(wu.x), fx0) + dot4_hf(unpack4_fp8_h_div256(wu.y), fx1)
                 + dot4_hf(unpack4_fp8_h_div256(wu.z), fx2) + dot4_hf(unpack4_fp8_h_div256(wu.w), fx3)) * su_val;
        }
        const float gate = float(bfloat(reduce8(ag)));
        const float up = float(bfloat(reduce8(au)));
        if (slane == 0) h_sorted[size_t(beg + t0) * INTER + r] = bfloat(softcapped_swiglu(gate, up, gate_cap, up_cap));
    } else if (tn == MT) {
        float acc_g[MT];
        float acc_u[MT];
        #pragma unroll
        for (int ti = 0; ti < MT; ++ti) { acc_g[ti] = 0.0f; acc_u[ti] = 0.0f; }
        for (int hg = int(slane); hg < HGROUPS; hg += 8) {
            const uint4 wg = *(const device uint4*)(w + w_base_g + hg * 4);
            const float sg_val = decode_fp8_e8m0_x256(scales[s_base_g + (hg >> 1)]);
            const uint4 wu = *(const device uint4*)(w + w_base_u + hg * 4);
            const float su_val = decode_fp8_e8m0_x256(scales[s_base_u + (hg >> 1)]);
            const half4 g0 = unpack4_fp8_h_div256(wg.x), g1 = unpack4_fp8_h_div256(wg.y);
            const half4 g2 = unpack4_fp8_h_div256(wg.z), g3 = unpack4_fp8_h_div256(wg.w);
            const half4 u0 = unpack4_fp8_h_div256(wu.x), u1 = unpack4_fp8_h_div256(wu.y);
            const half4 u2 = unpack4_fp8_h_div256(wu.z), u3 = unpack4_fp8_h_div256(wu.w);
            #pragma unroll
            for (int ti = 0; ti < MT; ++ti) {
                const device uint4* xq = (const device uint4*)(x + size_t(tg_tok[ti]) * K + hg * 16);
                const uint4 q0 = xq[0], q1 = xq[1];
                const float4 fx0 = float4(as_type<vec<bfloat, 4>>(q0.xy));
                const float4 fx1 = float4(as_type<vec<bfloat, 4>>(q0.zw));
                const float4 fx2 = float4(as_type<vec<bfloat, 4>>(q1.xy));
                const float4 fx3 = float4(as_type<vec<bfloat, 4>>(q1.zw));
                acc_g[ti] += (dot4_hf(g0, fx0) + dot4_hf(g1, fx1) + dot4_hf(g2, fx2) + dot4_hf(g3, fx3)) * sg_val;
                acc_u[ti] += (dot4_hf(u0, fx0) + dot4_hf(u1, fx1) + dot4_hf(u2, fx2) + dot4_hf(u3, fx3)) * su_val;
            }
        }
        #pragma unroll
        for (int ti = 0; ti < MT; ++ti) {
            const float gate = float(bfloat(reduce8(acc_g[ti])));
            const float up = float(bfloat(reduce8(acc_u[ti])));
            if (slane == 0) h_sorted[size_t(beg + t0 + ti) * INTER + r] = bfloat(softcapped_swiglu(gate, up, gate_cap, up_cap));
        }
    } else {
        float acc_g[MT];
        float acc_u[MT];
        #pragma unroll
        for (int ti = 0; ti < MT; ++ti) { acc_g[ti] = 0.0f; acc_u[ti] = 0.0f; }
        for (int hg = int(slane); hg < HGROUPS; hg += 8) {
            const uint4 wg = *(const device uint4*)(w + w_base_g + hg * 4);
            const float sg_val = decode_fp8_e8m0_x256(scales[s_base_g + (hg >> 1)]);
            const uint4 wu = *(const device uint4*)(w + w_base_u + hg * 4);
            const float su_val = decode_fp8_e8m0_x256(scales[s_base_u + (hg >> 1)]);
            const half4 g0 = unpack4_fp8_h_div256(wg.x), g1 = unpack4_fp8_h_div256(wg.y);
            const half4 g2 = unpack4_fp8_h_div256(wg.z), g3 = unpack4_fp8_h_div256(wg.w);
            const half4 u0 = unpack4_fp8_h_div256(wu.x), u1 = unpack4_fp8_h_div256(wu.y);
            const half4 u2 = unpack4_fp8_h_div256(wu.z), u3 = unpack4_fp8_h_div256(wu.w);
            #pragma unroll
            for (int ti = 0; ti < MT; ++ti) {
                if (ti < tn) {
                    const device uint4* xq = (const device uint4*)(x + size_t(tg_tok[ti]) * K + hg * 16);
                    const uint4 q0 = xq[0], q1 = xq[1];
                    const float4 fx0 = float4(as_type<vec<bfloat, 4>>(q0.xy));
                    const float4 fx1 = float4(as_type<vec<bfloat, 4>>(q0.zw));
                    const float4 fx2 = float4(as_type<vec<bfloat, 4>>(q1.xy));
                    const float4 fx3 = float4(as_type<vec<bfloat, 4>>(q1.zw));
                    acc_g[ti] += (dot4_hf(g0, fx0) + dot4_hf(g1, fx1) + dot4_hf(g2, fx2) + dot4_hf(g3, fx3)) * sg_val;
                    acc_u[ti] += (dot4_hf(u0, fx0) + dot4_hf(u1, fx1) + dot4_hf(u2, fx2) + dot4_hf(u3, fx3)) * su_val;
                }
            }
        }
        #pragma unroll
        for (int ti = 0; ti < MT; ++ti) {
            if (ti < tn) {
                const float gate = float(bfloat(reduce8(acc_g[ti])));
                const float up = float(bfloat(reduce8(acc_u[ti])));
                if (slane == 0) h_sorted[size_t(beg + t0 + ti) * INTER + r] = bfloat(softcapped_swiglu(gate, up, gate_cap, up_cap));
            }
        }
    }
"""

_GRP_W2_MXFP8_TILES_SRC = r"""
    const int tile_idx = int(threadgroup_position_in_grid.y);
    const int row_blk  = int(threadgroup_position_in_grid.x);
    const int e = tile_expert[tile_idx];
    if (e < 0) return;
    const int t0 = tile_t0[tile_idx];
    const int beg = offsets[e];
    const int cnt = offsets[e + 1] - beg;
    const int tn = min(MT, cnt - t0);

    threadgroup uint tg_slot[MT];
    const uint tid_tg = thread_index_in_threadgroup;
    if (int(tid_tg) < tn) tg_slot[tid_tg] = sorted_slots[beg + t0 + tid_tg];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint lane = thread_index_in_simdgroup;
    const uint sub = lane >> 3;
    const uint slane = lane & 7u;
    const uint sg = simdgroup_index_in_threadgroup;
    const int d = (row_blk * SG + int(sg)) * 4 + int(sub);
    if (d >= D) return;

    constexpr int HGROUPS = INTER / 16;
    constexpr int SGROUPS = INTER / 32;
    constexpr int W_STRIDE = INTER / 4;
    const size_t w_base = (size_t(e) * D + size_t(d)) * W_STRIDE;
    const size_t s_base = (size_t(e) * D + size_t(d)) * SGROUPS;

    if (tn == 1) {
        const device bfloat* hm = h_sorted + size_t(beg + t0) * INTER;
        float a = 0.0f;
        for (int hg = int(slane); hg < HGROUPS; hg += 8) {
            const uint4 ww = *(const device uint4*)(w + w_base + hg * 4);
            const float sv = decode_fp8_e8m0_x256(scales[s_base + (hg >> 1)]);
            const device uint4* hq = (const device uint4*)(hm + hg * 16);
            const uint4 q0 = hq[0], q1 = hq[1];
            a += (dot4_h(unpack4_fp8_h_div256(ww.x), as_type<vec<bfloat, 4>>(q0.xy))
                + dot4_h(unpack4_fp8_h_div256(ww.y), as_type<vec<bfloat, 4>>(q0.zw))
                + dot4_h(unpack4_fp8_h_div256(ww.z), as_type<vec<bfloat, 4>>(q1.xy))
                + dot4_h(unpack4_fp8_h_div256(ww.w), as_type<vec<bfloat, 4>>(q1.zw))) * sv;
        }
        const bfloat yk = bfloat(reduce8(a));
        if (slane == 0) ys[size_t(tg_slot[0]) * D + d] = yk;
    } else if (tn == MT) {
        float acc[MT];
        #pragma unroll
        for (int ti = 0; ti < MT; ++ti) acc[ti] = 0.0f;
        for (int hg = int(slane); hg < HGROUPS; hg += 8) {
            const uint4 ww = *(const device uint4*)(w + w_base + hg * 4);
            const float sv = decode_fp8_e8m0_x256(scales[s_base + (hg >> 1)]);
            const half4 f0 = unpack4_fp8_h_div256(ww.x), f1 = unpack4_fp8_h_div256(ww.y);
            const half4 f2 = unpack4_fp8_h_div256(ww.z), f3 = unpack4_fp8_h_div256(ww.w);
            #pragma unroll
            for (int ti = 0; ti < MT; ++ti) {
                const device uint4* hq = (const device uint4*)(h_sorted + size_t(beg + t0 + ti) * INTER + hg * 16);
                const uint4 q0 = hq[0], q1 = hq[1];
                acc[ti] += (dot4_h(f0, as_type<vec<bfloat, 4>>(q0.xy))
                          + dot4_h(f1, as_type<vec<bfloat, 4>>(q0.zw))
                          + dot4_h(f2, as_type<vec<bfloat, 4>>(q1.xy))
                          + dot4_h(f3, as_type<vec<bfloat, 4>>(q1.zw))) * sv;
            }
        }
        #pragma unroll
        for (int ti = 0; ti < MT; ++ti) {
            const bfloat yk = bfloat(reduce8(acc[ti]));
            if (slane == 0) ys[size_t(tg_slot[ti]) * D + d] = yk;
        }
    } else {
        float acc[MT];
        #pragma unroll
        for (int ti = 0; ti < MT; ++ti) acc[ti] = 0.0f;
        for (int hg = int(slane); hg < HGROUPS; hg += 8) {
            const uint4 ww = *(const device uint4*)(w + w_base + hg * 4);
            const float sv = decode_fp8_e8m0_x256(scales[s_base + (hg >> 1)]);
            const half4 f0 = unpack4_fp8_h_div256(ww.x), f1 = unpack4_fp8_h_div256(ww.y);
            const half4 f2 = unpack4_fp8_h_div256(ww.z), f3 = unpack4_fp8_h_div256(ww.w);
            #pragma unroll
            for (int ti = 0; ti < MT; ++ti) {
                if (ti < tn) {
                    const device uint4* hq = (const device uint4*)(h_sorted + size_t(beg + t0 + ti) * INTER + hg * 16);
                    const uint4 q0 = hq[0], q1 = hq[1];
                    acc[ti] += (dot4_h(f0, as_type<vec<bfloat, 4>>(q0.xy))
                              + dot4_h(f1, as_type<vec<bfloat, 4>>(q0.zw))
                              + dot4_h(f2, as_type<vec<bfloat, 4>>(q1.xy))
                              + dot4_h(f3, as_type<vec<bfloat, 4>>(q1.zw))) * sv;
                }
            }
        }
        #pragma unroll
        for (int ti = 0; ti < MT; ++ti) {
            if (ti < tn) {
                const bfloat yk = bfloat(reduce8(acc[ti]));
                if (slane == 0) ys[size_t(tg_slot[ti]) * D + d] = yk;
            }
        }
    }
"""


def routed_experts_fast(x: mx.array, inds: mx.array, moe) -> mx.array:
    """Unified dispatch for M=1..112 with 100% bit-identical batch invariance (inv_diff = 0.0)."""
    m = x.shape[0]
    mode = getattr(moe.w13_proj, "mode", None)
    if (
        mode in ("nvfp4", "mxfp8")
        and x.dtype == mx.bfloat16
        and x.shape[-1] % 256 == 0
        and moe.down_proj.input_dims % 128 == 0
        and moe.num_experts == 128
    ):
        if m <= 2:
            return moe_m1_experts(x, inds, moe)
        if m <= 8:
            return moe_grouped_tiles(x, inds, moe, mt=4, sg_w13=4, sg_w2=4)
        if m <= 112:
            return moe_grouped_tiles(x, inds, moe, mt=8, sg_w13=2, sg_w2=2)
    return moe.routed_experts(x, inds)


def moe_grouped_tiles(
    x: mx.array,
    inds: mx.array,
    moe,
    *,
    mt: int = 8,
    sg_w13: int = 2,
    sg_w2: int = 2,
) -> mx.array:
    """Zero-spill compact tile-table grouped NVFP4/MXFP8 MoE (0.000000 bit-exact vs M=1)."""
    m = x.shape[0]
    k_dim = x.shape[-1]
    inter = moe.down_proj.input_dims
    d_out = moe.down_proj.output_dims
    top_k = moe.top_k
    num_e = moe.num_experts
    mode = moe.w13_proj.mode
    caps = _get_caps(moe.gate_cap, moe.up_cap)
    t_slots = m * top_k
    flat_inds = inds.reshape(t_slots).astype(mx.uint32)
    max_tiles = min(t_slots, num_e + (t_slots - 1) // mt)

    offsets, sorted_slots, tile_expert, tile_t0 = _mk(
        "bucket_tiles",
        ["inds"],
        ["offsets", "sorted_slots", "tile_expert", "tile_t0"],
        _BUCKET_TILES_SRC,
    )(
        inputs=[flat_inds],
        template=[("MT", mt), ("MAX_TILES", max_tiles)],
        grid=(128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(num_e + 1,), (t_slots,), (max_tiles,), (max_tiles,)],
        output_dtypes=[mx.int32, mx.uint32, mx.int32, mx.int32],
    )

    nx13 = (inter + sg_w13 * 4 - 1) // (sg_w13 * 4)
    nx2 = (d_out + sg_w2 * 4 - 1) // (sg_w2 * 4)

    if mode == "nvfp4":
        h_sorted = _mk(
            "grp_w13_nvfp4_t",
            ["x", "w", "scales", "gscale", "offsets", "sorted_slots", "tile_expert", "tile_t0", "caps"],
            ["h_sorted"],
            _GRP_W13_NVFP4_TILES_SRC,
        )(
            inputs=[x.reshape(m, k_dim), moe.w13_proj.weight, moe.w13_proj.scales, moe.w13_proj.global_scale, offsets, sorted_slots, tile_expert, tile_t0, caps],
            template=[("K", k_dim), ("INTER", inter), ("TOP_K", top_k), ("SG", sg_w13), ("MT", mt)],
            grid=(nx13 * 32 * sg_w13, max_tiles, 1),
            threadgroup=(32 * sg_w13, 1, 1),
            output_shapes=[(t_slots, inter)],
            output_dtypes=[mx.bfloat16],
        )[0]
        ys = _mk(
            "grp_w2_nvfp4_t",
            ["h_sorted", "w", "scales", "gscale", "offsets", "sorted_slots", "tile_expert", "tile_t0"],
            ["ys"],
            _GRP_W2_NVFP4_TILES_SRC,
        )(
            inputs=[h_sorted, moe.down_proj.weight, moe.down_proj.scales, moe.down_proj.global_scale, offsets, sorted_slots, tile_expert, tile_t0],
            template=[("D", d_out), ("INTER", inter), ("SG", sg_w2), ("MT", mt)],
            grid=(nx2 * 32 * sg_w2, max_tiles, 1),
            threadgroup=(32 * sg_w2, 1, 1),
            output_shapes=[(m, top_k, d_out)],
            output_dtypes=[mx.bfloat16],
        )[0]
        return ys
    else:
        h_sorted = _mk(
            "grp_w13_mxfp8_t",
            ["x", "w", "scales", "offsets", "sorted_slots", "tile_expert", "tile_t0", "caps"],
            ["h_sorted"],
            _GRP_W13_MXFP8_TILES_SRC,
        )(
            inputs=[x.reshape(m, k_dim), moe.w13_proj.weight, moe.w13_proj.scales, offsets, sorted_slots, tile_expert, tile_t0, caps],
            template=[("K", k_dim), ("INTER", inter), ("TOP_K", top_k), ("SG", sg_w13), ("MT", mt)],
            grid=(nx13 * 32 * sg_w13, max_tiles, 1),
            threadgroup=(32 * sg_w13, 1, 1),
            output_shapes=[(t_slots, inter)],
            output_dtypes=[mx.bfloat16],
        )[0]
        ys = _mk(
            "grp_w2_mxfp8_t",
            ["h_sorted", "w", "scales", "offsets", "sorted_slots", "tile_expert", "tile_t0"],
            ["ys"],
            _GRP_W2_MXFP8_TILES_SRC,
        )(
            inputs=[h_sorted, moe.down_proj.weight, moe.down_proj.scales, offsets, sorted_slots, tile_expert, tile_t0],
            template=[("D", d_out), ("INTER", inter), ("SG", sg_w2), ("MT", mt)],
            grid=(nx2 * 32 * sg_w2, max_tiles, 1),
            threadgroup=(32 * sg_w2, 1, 1),
            output_shapes=[(m, top_k, d_out)],
            output_dtypes=[mx.bfloat16],
        )[0]
        return ys


def moe_grouped_experts(
    x: mx.array,
    inds: mx.array,
    moe,
    *,
    mt: int = 8,
    sg_w13: int | None = None,
    sg_w2: int | None = None,
) -> mx.array:
    """Unique-expert grouped NVFP4/MXFP8 MoE returning unsorted ys (M, TOP_K, D)."""
    m = x.shape[0]
    k_dim = x.shape[-1]
    inter = moe.down_proj.input_dims
    d_out = moe.down_proj.output_dims
    top_k = moe.top_k
    num_e = moe.num_experts
    mode = moe.w13_proj.mode
    if sg_w13 is None:
        sg_w13 = 8 if (mode == "nvfp4" and m <= 6) else 4
    if sg_w2 is None:
        sg_w2 = 8 if (mode == "nvfp4" and m <= 6) else 4
    caps = _get_caps(moe.gate_cap, moe.up_cap)
    flat_inds = inds.reshape(m * top_k).astype(mx.uint32)
    offsets, sorted_slots, active_list = bucket_experts(flat_inds, num_experts=num_e)

    nx13 = (inter + sg_w13 * 4 - 1) // (sg_w13 * 4)
    nx2 = (d_out + sg_w2 * 4 - 1) // (sg_w2 * 4)
    t_slots = m * top_k
    max_act = min(num_e, t_slots)
    nz = 1 if m <= 8 else (2 if m <= 16 else 4)
    zstep = nz * mt

    if mode == "nvfp4":
        h_sorted = _mk(
            "grp_w13_nvfp4_c",
            ["x", "w", "scales", "gscale", "offsets", "sorted_slots", "active_list", "caps"],
            ["h_sorted"],
            _GRP_W13_NVFP4_SRC,
        )(
            inputs=[x.reshape(m, k_dim), moe.w13_proj.weight, moe.w13_proj.scales, moe.w13_proj.global_scale, offsets, sorted_slots, active_list, caps],
            template=[("K", k_dim), ("INTER", inter), ("TOP_K", top_k), ("SG", sg_w13), ("MT", mt), ("ZSTEP", zstep)],
            grid=(nx13 * 32 * sg_w13, max_act, nz),
            threadgroup=(32 * sg_w13, 1, 1),
            output_shapes=[(t_slots, inter)],
            output_dtypes=[mx.bfloat16],
        )[0]
        ys = _mk(
            "grp_w2_nvfp4_c",
            ["h_sorted", "w", "scales", "gscale", "offsets", "sorted_slots", "active_list"],
            ["ys"],
            _GRP_W2_NVFP4_SRC,
        )(
            inputs=[h_sorted, moe.down_proj.weight, moe.down_proj.scales, moe.down_proj.global_scale, offsets, sorted_slots, active_list],
            template=[("D", d_out), ("INTER", inter), ("SG", sg_w2), ("MT", mt), ("ZSTEP", zstep)],
            grid=(nx2 * 32 * sg_w2, max_act, nz),
            threadgroup=(32 * sg_w2, 1, 1),
            output_shapes=[(m, top_k, d_out)],
            output_dtypes=[mx.bfloat16],
        )[0]
        return ys
    else:
        h_sorted = _mk(
            "grp_w13_mxfp8_c",
            ["x", "w", "scales", "offsets", "sorted_slots", "active_list", "caps"],
            ["h_sorted"],
            _GRP_W13_MXFP8_SRC,
        )(
            inputs=[x.reshape(m, k_dim), moe.w13_proj.weight, moe.w13_proj.scales, offsets, sorted_slots, active_list, caps],
            template=[("K", k_dim), ("INTER", inter), ("TOP_K", top_k), ("SG", sg_w13), ("MT", mt), ("ZSTEP", zstep)],
            grid=(nx13 * 32 * sg_w13, max_act, nz),
            threadgroup=(32 * sg_w13, 1, 1),
            output_shapes=[(t_slots, inter)],
            output_dtypes=[mx.bfloat16],
        )[0]
        ys = _mk(
            "grp_w2_mxfp8_c",
            ["h_sorted", "w", "scales", "offsets", "sorted_slots", "active_list"],
            ["ys"],
            _GRP_W2_MXFP8_SRC,
        )(
            inputs=[h_sorted, moe.down_proj.weight, moe.down_proj.scales, offsets, sorted_slots, active_list],
            template=[("D", d_out), ("INTER", inter), ("SG", sg_w2), ("MT", mt), ("ZSTEP", zstep)],
            grid=(nx2 * 32 * sg_w2, max_act, nz),
            threadgroup=(32 * sg_w2, 1, 1),
            output_shapes=[(m, top_k, d_out)],
            output_dtypes=[mx.bfloat16],
        )[0]
        return ys


class _MoeForwardArgs(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int32),
        ("M", ctypes.c_int32),
        ("K", ctypes.c_int32),
        ("INTER", ctypes.c_int32),
        ("D", ctypes.c_int32),
        ("E", ctypes.c_int32),
        ("TOP_K", ctypes.c_int32),
        ("_pad", ctypes.c_int32),
        ("gate_cap", ctypes.c_float),
        ("up_cap", ctypes.c_float),
        ("x_ptr", ctypes.c_uint64),
        ("inds_ptr", ctypes.c_uint64),
        ("w13_ptr", ctypes.c_uint64),
        ("s13_ptr", ctypes.c_uint64),
        ("gs13_ptr", ctypes.c_uint64),
        ("w2_ptr", ctypes.c_uint64),
        ("s2_ptr", ctypes.c_uint64),
        ("gs2_ptr", ctypes.c_uint64),
        ("y_sorted_ptr", ctypes.c_uint64),
        ("inv_order_ptr", ctypes.c_uint64),
        ("offsets_ptr", ctypes.c_uint64),
        ("sorted_slots_ptr", ctypes.c_uint64),
    ]


class _MoeFusedCombineArgs(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int32),
        ("B", ctypes.c_int32),
        ("L", ctypes.c_int32),
        ("K", ctypes.c_int32),
        ("INTER", ctypes.c_int32),
        ("D", ctypes.c_int32),
        ("E", ctypes.c_int32),
        ("TOP_K", ctypes.c_int32),
        ("gate_cap", ctypes.c_float),
        ("up_cap", ctypes.c_float),
        ("has_mask", ctypes.c_int32),
        ("_pad", ctypes.c_int32),
        ("x_ptr", ctypes.c_uint64),
        ("inds_ptr", ctypes.c_uint64),
        ("w13_ptr", ctypes.c_uint64),
        ("s13_ptr", ctypes.c_uint64),
        ("gs13_ptr", ctypes.c_uint64),
        ("w2_ptr", ctypes.c_uint64),
        ("s2_ptr", ctypes.c_uint64),
        ("gs2_ptr", ctypes.c_uint64),
        ("rweights_ptr", ctypes.c_uint64),
        ("state_ptr", ctypes.c_uint64),
        ("taps_ptr", ctypes.c_uint64),
        ("resid_ptr", ctypes.c_uint64),
        ("mask_ptr", ctypes.c_uint64),
        ("out_x_ptr", ctypes.c_uint64),
        ("new_state_ptr", ctypes.c_uint64),
        ("y_sorted_ptr", ctypes.c_uint64),
        ("inv_order_ptr", ctypes.c_uint64),
    ]


_PREFILL_MOE_LIB = None
_PREFILL_MOE_TRIED = False


def _get_prefill_moe_lib():
    global _PREFILL_MOE_LIB, _PREFILL_MOE_TRIED
    if not _PREFILL_MOE_TRIED:
        _PREFILL_MOE_TRIED = True
        lib_path = Path(__file__).resolve().parent / "native" / "libinkling_prefill_moe.dylib"
        if lib_path.is_file():
            try:
                lib = ctypes.CDLL(str(lib_path))
                lib.prefill_moe_init.restype = ctypes.c_int
                lib.prefill_moe_register_buffer.argtypes = [ctypes.c_uint64, ctypes.c_size_t]
                lib.prefill_moe_register_buffer.restype = ctypes.c_int
                lib.prefill_moe_forward.argtypes = [ctypes.POINTER(_MoeForwardArgs), ctypes.POINTER(ctypes.c_double)]
                lib.prefill_moe_forward.restype = ctypes.c_int
                if hasattr(lib, "inkling_prefill_moe_nax_fused_combine"):
                    lib.inkling_prefill_moe_nax_fused_combine.argtypes = [
                        ctypes.POINTER(_MoeFusedCombineArgs),
                        ctypes.POINTER(ctypes.c_double),
                    ]
                    lib.inkling_prefill_moe_nax_fused_combine.restype = ctypes.c_int
                if lib.prefill_moe_init() == 0:
                    _PREFILL_MOE_LIB = lib
            except Exception:
                _PREFILL_MOE_LIB = None
    return _PREFILL_MOE_LIB


def _raw_ptr_bytes(arr: mx.array) -> tuple[np.ndarray, int, int]:
    view = arr.view(mx.uint16) if arr.dtype == mx.bfloat16 else arr
    nv = np.asarray(view)
    return nv, int(nv.ctypes.data), int(nv.nbytes)


def _ensure_moe_weight_ptrs(lib, moe):
    cached = getattr(moe, "_native_prefill_ptrs", None)
    if cached is not None:
        return cached
    mode_id = 0 if moe.w13_proj.mode == "nvfp4" else 1
    w13 = mx.contiguous(moe.w13_proj.weight)
    s13 = mx.contiguous(moe.w13_proj.scales)
    w2 = mx.contiguous(moe.down_proj.weight)
    s2 = mx.contiguous(moe.down_proj.scales)
    arrs = [w13, s13, w2, s2]
    gs13 = gs2 = None
    if mode_id == 0:
        gs13 = mx.contiguous(moe.w13_proj.global_scale.astype(mx.float32))
        gs2 = mx.contiguous(moe.down_proj.global_scale.astype(mx.float32))
        arrs.extend((gs13, gs2))
    mx.eval(*arrs)
    keep = []
    ptrs = []
    for a in (w13, s13, gs13, w2, s2, gs2):
        if a is None:
            ptrs.append(0)
        else:
            nv, p, nb = _raw_ptr_bytes(a)
            keep.append((a, nv))
            lib.prefill_moe_register_buffer(p, nb)
            ptrs.append(p)
    info = (mode_id, tuple(ptrs), keep)
    object.__setattr__(moe, "_native_prefill_ptrs", info)
    return info


_PREFILL_MOE_SCRATCH: dict[tuple[int, int], tuple[mx.array, np.ndarray, int, mx.array, np.ndarray, int]] = {}


def _get_prefill_moe_scratch(lib, t_slots: int, d_out: int):
    """Reuse pre-registered scratch (ys_sorted, inv_order) across all 48 visits (eliminates 7GB alloc churn!)."""
    key = (t_slots, d_out)
    entry = _PREFILL_MOE_SCRATCH.get(key)
    if entry is None:
        if t_slots > 32768 or len(_PREFILL_MOE_SCRATCH) >= 2:
            _PREFILL_MOE_SCRATCH.clear()
        ys = mx.zeros((t_slots, d_out), dtype=mx.bfloat16)
        inv = mx.zeros((t_slots,), dtype=mx.uint32)
        mx.eval(ys, inv)
        vy, yp, ynb = _raw_ptr_bytes(ys)
        vo, op, onb = _raw_ptr_bytes(inv)
        entry = (ys, vy, yp, inv, vo, op)
        _PREFILL_MOE_SCRATCH[key] = entry
    return entry


def prefill_moe_nax_sorted(
    x: mx.array,
    indices: mx.array,
    moe,
) -> tuple[mx.array, mx.array] | None:
    """Metal 4.0 NAX grouped prefill MoE (bucket + w13+SwiGLU + w2 in 1 CB, 22-27 TF/s).

    Returns `(ys_sorted [M*TOP_K, D] bf16, inv_order [M*TOP_K] uint32)` or `None` if unavailable.
    """
    lib = _get_prefill_moe_lib()
    if lib is None or x.dtype != mx.bfloat16 or mx.default_device() != mx.gpu:
        return None
    mode = getattr(moe.w13_proj, "mode", None)
    if mode not in ("nvfp4", "mxfp8"):
        return None
    k_dim = x.shape[-1]
    inter = moe.down_proj.input_dims
    d_out = moe.down_proj.output_dims
    if k_dim % 64 != 0 or inter % 64 != 0 or d_out % 64 != 0 or moe.num_experts > 256:
        return None
    m = x.size // k_dim
    top_k = moe.top_k
    t_slots = m * top_k

    mode_id, (w13_p, s13_p, gs13_p, w2_p, s2_p, gs2_p), _keep = _ensure_moe_weight_ptrs(lib, moe)
    x_c = mx.contiguous(x.reshape(m, k_dim))
    inds_c = mx.contiguous(indices.reshape(t_slots).astype(mx.uint32))
    mx.eval(x_c, inds_c)
    ys_sorted, vy, yp, inv_order, vo, op = _get_prefill_moe_scratch(lib, t_slots, d_out)

    vx, xp, _ = _raw_ptr_bytes(x_c)
    vi, ip, _ = _raw_ptr_bytes(inds_c)

    args = _MoeForwardArgs(
        mode=mode_id,
        M=m,
        K=k_dim,
        INTER=inter,
        D=d_out,
        E=moe.num_experts,
        TOP_K=top_k,
        _pad=0,
        gate_cap=float(moe.gate_cap),
        up_cap=float(moe.up_cap),
        x_ptr=xp,
        inds_ptr=ip,
        w13_ptr=w13_p,
        s13_ptr=s13_p,
        gs13_ptr=gs13_p,
        w2_ptr=w2_p,
        s2_ptr=s2_p,
        gs2_ptr=gs2_p,
        y_sorted_ptr=yp,
        inv_order_ptr=op,
        offsets_ptr=0,
        sorted_slots_ptr=0,
    )
    rc = lib.prefill_moe_forward(ctypes.byref(args), None)
    _ = (vx, vi, vy, vo)
    if rc != 0:
        return None
    return ys_sorted, inv_order


def prefill_moe_nax_fused_combine(
    x: mx.array,
    indices: mx.array,
    rweights: mx.array,
    state: mx.array,
    taps: mx.array,
    resid: mx.array,
    moe,
    *,
    b: int,
    length: int,
    mask: mx.array | None = None,
) -> tuple[mx.array, mx.array] | None:
    """4-dispatch 1-CB Metal 4.0 NAX MoE + 64-ch 128B-aligned unsort+combine+sconv+resid.

    Uses GPU-private internal ys_sorted/inv_order scratch (zero 302 MB MLX alloc) and returns
    `(out_x [B*L, D] bf16, new_state [B, 3, D] f32)` or `None` if unavailable.
    """
    lib = _get_prefill_moe_lib()
    if (
        lib is None
        or not hasattr(lib, "inkling_prefill_moe_nax_fused_combine")
        or x.dtype != mx.bfloat16
        or mx.default_device() != mx.gpu
    ):
        return None
    mode = getattr(moe.w13_proj, "mode", None)
    if mode not in ("nvfp4", "mxfp8"):
        return None
    k_dim = x.shape[-1]
    inter = moe.down_proj.input_dims
    d_out = moe.down_proj.output_dims
    if k_dim % 64 != 0 or inter % 64 != 0 or d_out % 64 != 0 or moe.num_experts > 256:
        return None
    m = b * length
    top_k = moe.top_k
    t_slots = m * top_k

    mode_id, (w13_p, s13_p, gs13_p, w2_p, s2_p, gs2_p), _keep = _ensure_moe_weight_ptrs(lib, moe)
    taps_info = getattr(moe, "_native_taps_ptr", None)
    if taps_info is None or taps_info[0] is not taps:
        taps_c = mx.contiguous(taps.astype(mx.float32))
        mx.eval(taps_c)
        vt, tp, tnb = _raw_ptr_bytes(taps_c)
        lib.prefill_moe_register_buffer(tp, tnb)
        taps_info = (taps, taps_c, vt, tp)
        object.__setattr__(moe, "_native_taps_ptr", taps_info)
    taps_p = taps_info[3]

    x_c = mx.contiguous(x.reshape(m, k_dim))
    inds_c = mx.contiguous(indices.reshape(t_slots).astype(mx.uint32))
    rw_c = mx.contiguous(rweights.reshape(t_slots).astype(mx.float32))
    st_c = mx.contiguous(state.astype(mx.float32))
    res_c = mx.contiguous(resid.reshape(m, d_out).astype(mx.float32))
    out_x = mx.zeros((m, d_out), dtype=mx.bfloat16)
    new_st = mx.zeros((b, 3, d_out), dtype=mx.float32)

    to_eval = [x_c, inds_c, rw_c, st_c, res_c, out_x, new_st]
    has_mask = mask is not None
    mask_c = None
    if has_mask:
        mask_c = mx.contiguous(mask.reshape(m).astype(mx.uint8))
        to_eval.append(mask_c)
    mx.eval(*to_eval)

    vx, xp, _ = _raw_ptr_bytes(x_c)
    vi, ip, _ = _raw_ptr_bytes(inds_c)
    vrw, rwp, _ = _raw_ptr_bytes(rw_c)
    vst, stp, _ = _raw_ptr_bytes(st_c)
    vres, resp, _ = _raw_ptr_bytes(res_c)
    vox, oxp, _ = _raw_ptr_bytes(out_x)
    vnst, nstp, _ = _raw_ptr_bytes(new_st)
    vm, mp = (None, 0)
    if mask_c is not None:
        vm, mp, _ = _raw_ptr_bytes(mask_c)

    args = _MoeFusedCombineArgs(
        mode=mode_id,
        B=b,
        L=length,
        K=k_dim,
        INTER=inter,
        D=d_out,
        E=moe.num_experts,
        TOP_K=top_k,
        gate_cap=float(moe.gate_cap),
        up_cap=float(moe.up_cap),
        has_mask=1 if has_mask else 0,
        _pad=0,
        x_ptr=xp,
        inds_ptr=ip,
        w13_ptr=w13_p,
        s13_ptr=s13_p,
        gs13_ptr=gs13_p,
        w2_ptr=w2_p,
        s2_ptr=s2_p,
        gs2_ptr=gs2_p,
        rweights_ptr=rwp,
        state_ptr=stp,
        taps_ptr=taps_p,
        resid_ptr=resp,
        mask_ptr=mp,
        out_x_ptr=oxp,
        new_state_ptr=nstp,
        y_sorted_ptr=0,
        inv_order_ptr=0,
    )
    rc = lib.inkling_prefill_moe_nax_fused_combine(ctypes.byref(args), None)
    _ = (vx, vi, vrw, vst, vres, vox, vnst, vm)
    if rc != 0:
        return None
    return out_x, new_st
