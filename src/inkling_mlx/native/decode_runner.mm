#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "decode_runner.h"

static const char* kMetalSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

constant bool FC_GATE [[function_constant(0)]];
constant bool FC_CONV [[function_constant(1)]];
constant int  FC_K    [[function_constant(2)]];
constant bool FC_DYN  [[function_constant(3)]];
constant int  FC_M    [[function_constant(4)]];
constant int  FC_R    [[function_constant(5)]];
constant int  FC_SG   [[function_constant(6)]];
constant int  FC_G    [[function_constant(7)]];
constant int  FC_HD   [[function_constant(8)]];
constant bool FC_SEQ  [[function_constant(9)]];
constant int  FC_KDIM [[function_constant(10)]];
constant int  FC_MT   [[function_constant(11)]];

inline float sigmoid_(float x) {
    return 1.0f / (1.0f + metal::precise::exp(-x));
}

inline float round_bf16(float x) {
    return float(bfloat(x));
}

struct CopyPrm {
    uint count;
};

kernel void inkling_copy_bf16(
    const device bfloat* src [[buffer(0)]],
    device bfloat* dst       [[buffer(1)]],
    constant CopyPrm& prm    [[buffer(2)]],
    uint gid                 [[thread_position_in_grid]])
{
    if (gid < prm.count) dst[gid] = src[gid];
}

kernel void inkling_copy_f32(
    const device float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    constant CopyPrm& prm   [[buffer(2)]],
    uint gid                [[thread_position_in_grid]])
{
    if (gid < prm.count) dst[gid] = src[gid];
}

struct CopyConv4Prm {
    uint n_c, n_d;
};

kernel void inkling_copy_conv4_last(
    const device float* ks_src [[buffer(0)]],
    const device float* vs_src [[buffer(1)]],
    const device float* cs_src [[buffer(2)]],
    const device float* ms_src [[buffer(3)]],
    device float* ks_dst       [[buffer(4)]],
    device float* vs_dst       [[buffer(5)]],
    device float* cs_dst       [[buffer(6)]],
    device float* ms_dst       [[buffer(7)]],
    constant CopyConv4Prm& p   [[buffer(8)]],
    uint gid                   [[thread_position_in_grid]])
{
    if (gid < p.n_c) {
        ks_dst[gid] = ks_src[gid];
        vs_dst[gid] = vs_src[gid];
    }
    if (gid < p.n_d) {
        cs_dst[gid] = cs_src[gid];
        ms_dst[gid] = ms_src[gid];
    }
}

struct CopyConv4DynPrm {
    uint n_c, n_d;
    int M;
};

kernel void inkling_copy_conv4_dyn(
    const device float* ks_base [[buffer(0)]],
    const device float* vs_base [[buffer(1)]],
    const device float* cs_base [[buffer(2)]],
    const device float* ms_base [[buffer(3)]],
    device float* ks_dst        [[buffer(4)]],
    device float* vs_dst        [[buffer(5)]],
    device float* cs_dst        [[buffer(6)]],
    device float* ms_dst        [[buffer(7)]],
    const device int* d_n_acc   [[buffer(8)]],
    constant CopyConv4DynPrm& p [[buffer(9)]],
    uint gid                    [[thread_position_in_grid]])
{
    const int n_toks = min(d_n_acc[0], p.M);
    if (n_toks <= 0) return;
    const size_t r = size_t(n_toks - 1);
    if (gid < p.n_c) {
        const size_t off_c = r * size_t(p.n_c) + size_t(gid);
        ks_dst[gid] = ks_base[off_c];
        vs_dst[gid] = vs_base[off_c];
    }
    if (gid < p.n_d) {
        const size_t off_d = r * size_t(p.n_d) + size_t(gid);
        cs_dst[gid] = cs_base[off_d];
        ms_dst[gid] = ms_base[off_d];
    }
}

struct CopyCapPrm {
    int M, cap_slot, n_cap;
};

kernel void inkling_copy_capture_bf16(
    const device bfloat* x  [[buffer(0)]],
    device bfloat* out_cap  [[buffer(1)]],
    constant CopyCapPrm& p  [[buffer(2)]],
    uint2 gid               [[thread_position_in_grid]])
{
    const int d = int(gid.x);
    const int r = int(gid.y);
    if (d < MODEL_D && r < p.M) {
        out_cap[(size_t(r) * p.n_cap + p.cap_slot) * MODEL_D + d] = x[size_t(r) * MODEL_D + d];
    }
}

struct CopyCapDynPrm {
    int M, row_elems;
};

kernel void inkling_copy_capture_dyn_bf16(
    const device bfloat* vcap   [[buffer(0)]],
    device bfloat* out_cap      [[buffer(1)]],
    const device int* d_n_acc   [[buffer(2)]],
    constant CopyCapDynPrm& p   [[buffer(3)]],
    uint2 gid                   [[thread_position_in_grid]])
{
    const int col = int(gid.x);
    const int r = int(gid.y);
    const int n_toks = min(d_n_acc[0], p.M);
    if (r < n_toks && col < p.row_elems) {
        const size_t idx = size_t(r) * size_t(p.row_elems) + size_t(col);
        out_cap[idx] = vcap[idx];
    }
}

struct KvCommitPrm {
    int n_toks, HKV, HD, old_idx, max_size, stride_h, stride_s;
};

kernel void inkling_kv_commit_rotating(
    const device bfloat* ko_ext [[buffer(0)]],
    const device bfloat* vo_ext [[buffer(1)]],
    device bfloat* kv_k         [[buffer(2)]],
    device bfloat* kv_v         [[buffer(3)]],
    constant KvCommitPrm& p     [[buffer(4)]],
    uint gid                    [[thread_position_in_grid]])
{
    const uint total = uint(p.n_toks * p.HKV * p.HD);
    if (gid >= total) return;
    const int d = int(gid % uint(p.HD));
    const int tmp = int(gid / uint(p.HD));
    const int h = tmp % p.HKV;
    const int m = tmp / p.HKV;
    const int slot = (p.old_idx + m) % p.max_size;
    const size_t src = (size_t(m) * p.HKV + h) * p.HD + d;
    const size_t dst = size_t(h) * p.stride_h + size_t(slot) * p.stride_s + d;
    kv_k[dst] = ko_ext[src];
    kv_v[dst] = vo_ext[src];
}

struct KvCommitDynPrm {
    int M, HKV, HD, old_idx, max_size, stride_h, stride_s;
};

kernel void inkling_kv_commit_rotating_dyn(
    const device bfloat* ko_ext [[buffer(0)]],
    const device bfloat* vo_ext [[buffer(1)]],
    device bfloat* kv_k         [[buffer(2)]],
    device bfloat* kv_v         [[buffer(3)]],
    const device int* d_n_acc   [[buffer(4)]],
    constant KvCommitDynPrm& p  [[buffer(5)]],
    uint gid                    [[thread_position_in_grid]])
{
    const int n_toks = min(d_n_acc[0], p.M);
    const uint total = uint(max(0, n_toks) * p.HKV * p.HD);
    if (gid >= total) return;
    const int d = int(gid % uint(p.HD));
    const int tmp = int(gid / uint(p.HD));
    const int h = tmp % p.HKV;
    const int m = tmp / p.HKV;
    const int slot = (p.old_idx + m) % p.max_size;
    const size_t src = (size_t(m) * p.HKV + h) * p.HD + d;
    const size_t dst = size_t(h) * p.stride_h + size_t(slot) * p.stride_s + d;
    kv_k[dst] = ko_ext[src];
    kv_v[dst] = vo_ext[src];
}

inline void embed_rms_norm_row(
    const device bfloat* embed_w,
    const device bfloat* norm_w,
    device bfloat* out,
    int row,
    int tok,
    float eps,
    uint tid,
    uint lane,
    uint sg,
    threadgroup float* buf,
    threadgroup float& red)
{
    constexpr int TG = MODEL_MIX_TG;
    constexpr int NS = TG / 32;
    constexpr int DPT = MODEL_MIX_DPT;
    float v[DPT];
    float s2 = 0.0f;
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) {
        const int d = int(tid) + c * TG;
        float val = d < MODEL_D ? float(embed_w[size_t(tok) * MODEL_D + d]) : 0.0f;
        v[c] = val;
        s2 += val * val;
    }
    const float ss = simd_sum(s2);
    if (lane == 0) buf[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float tot = simd_sum(int(lane) < NS ? buf[lane] : 0.0f);
        if (lane == 0) red = tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = metal::precise::rsqrt(red / float(MODEL_D) + eps);
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) {
        const int d = int(tid) + c * TG;
        if (d < MODEL_D) {
            out[size_t(row) * MODEL_D + d] = bfloat(v[c] * inv) * norm_w[d];
        }
    }
}

struct EmbedPrm {
    float eps;
    int toks[8];
};

kernel void inkling_embed_rms_norm_2d(
    const device bfloat* embed_w [[buffer(0)]],
    const device bfloat* norm_w  [[buffer(1)]],
    device bfloat* out           [[buffer(2)]],
    constant EmbedPrm& p         [[buffer(3)]],
    uint2 tg_pos                 [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]],
    uint lane                    [[thread_index_in_simdgroup]],
    uint sg                      [[simdgroup_index_in_threadgroup]])
{
    constexpr int NS = MODEL_MIX_TG / 32;
    const int row = int(tg_pos.y);
    threadgroup float buf[NS];
    threadgroup float red;
    embed_rms_norm_row(embed_w, norm_w, out, row, p.toks[row], p.eps, tid, lane, sg, buf, red);
}

kernel void inkling_embed_rms_norm_from_buf(
    const device bfloat* embed_w [[buffer(0)]],
    const device bfloat* norm_w  [[buffer(1)]],
    device bfloat* out           [[buffer(2)]],
    const device int* toks_buf   [[buffer(3)]],
    constant float& eps          [[buffer(4)]],
    uint2 tg_pos                 [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]],
    uint lane                    [[thread_index_in_simdgroup]],
    uint sg                      [[simdgroup_index_in_threadgroup]])
{
    constexpr int NS = MODEL_MIX_TG / 32;
    const int row = int(tg_pos.y);
    threadgroup float buf[NS];
    threadgroup float red;
    embed_rms_norm_row(embed_w, norm_w, out, row, toks_buf[row], eps, tid, lane, sg, buf, red);
}

struct EmbedBatchedPrm {
    float eps;
    int toks[64];
};

kernel void inkling_embed_rms_norm_batched(
    const device bfloat* embed_w [[buffer(0)]],
    const device bfloat* norm_w  [[buffer(1)]],
    device bfloat* out           [[buffer(2)]],
    constant EmbedBatchedPrm& p  [[buffer(3)]],
    uint2 tg_pos                 [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]],
    uint lane                    [[thread_index_in_simdgroup]],
    uint sg                      [[simdgroup_index_in_threadgroup]])
{
    constexpr int TG = MODEL_MIX_TG;
    constexpr int NS = TG / 32;
    constexpr int DPT = MODEL_MIX_DPT;
    const int row = int(tg_pos.y);
    const int tok = p.toks[row];
    threadgroup float buf[NS];
    threadgroup float red;

    float v[DPT];
    float s2 = 0.0f;
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) {
        const int d = int(tid) + c * TG;
        float val = d < MODEL_D ? float(embed_w[size_t(tok) * MODEL_D + d]) : 0.0f;
        v[c] = val;
        s2 += val * val;
    }
    const float ss = simd_sum(s2);
    if (lane == 0) buf[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float tot = simd_sum(int(lane) < NS ? buf[lane] : 0.0f);
        if (lane == 0) red = tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = metal::precise::rsqrt(red / float(MODEL_D) + p.eps);
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) {
        const int d = int(tid) + c * TG;
        if (d < MODEL_D) {
            out[size_t(row) * MODEL_D + d] = bfloat(v[c] * inv) * norm_w[d];
        }
    }
}

struct Layer0Prm {
    float eps;
};

kernel void inkling_layer0_rms(
    const device bfloat* x      [[buffer(0)]],
    const device bfloat* norm_w [[buffer(1)]],
    device float* normed_out    [[buffer(2)]],
    device bfloat* xq_out       [[buffer(3)]],
    constant Layer0Prm& p       [[buffer(4)]],
    uint2 tg_pos                [[threadgroup_position_in_grid]],
    uint tid                    [[thread_index_in_threadgroup]],
    uint lane                   [[thread_index_in_simdgroup]],
    uint sg                     [[simdgroup_index_in_threadgroup]])
{
    constexpr int TG = MODEL_MIX_TG;
    constexpr int NS = TG / 32;
    constexpr int DPT = MODEL_MIX_DPT;
    const int row = int(tg_pos.y);
    threadgroup float buf[NS];
    threadgroup float red;

    float v[DPT];
    float s2 = 0.0f;
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) {
        const int d = int(tid) + c * TG;
        float val = d < MODEL_D ? float(x[size_t(row) * MODEL_D + d]) : 0.0f;
        v[c] = val;
        s2 += val * val;
    }
    const float ss = simd_sum(s2);
    if (lane == 0) buf[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float tot = simd_sum(int(lane) < NS ? buf[lane] : 0.0f);
        if (lane == 0) red = tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = metal::precise::rsqrt(red / float(MODEL_D) + 1e-6f);
    const float inv_ln  = metal::precise::rsqrt(red / float(MODEL_D) + p.eps);
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) {
        const int d = int(tid) + c * TG;
        if (d < MODEL_D) {
            normed_out[size_t(row) * MODEL_D + d] = v[c] * inv_rms;
            xq_out[size_t(row) * MODEL_D + d] = bfloat(v[c] * inv_ln) * norm_w[d];
        }
    }
}

struct FinalNormPrm {
    int cap_slot, n_cap;
    float eps;
};

kernel void inkling_final_rms_norm(
    const device float* pre_norm [[buffer(0)]],
    const device bfloat* norm_w  [[buffer(1)]],
    device float* out_f32        [[buffer(2)]],
    device bfloat* out_cap_bf16  [[buffer(3)]],
    constant FinalNormPrm& p     [[buffer(4)]],
    uint2 tg_pos                 [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]],
    uint lane                    [[thread_index_in_simdgroup]],
    uint sg                      [[simdgroup_index_in_threadgroup]])
{
    constexpr int TG = MODEL_MIX_TG;
    constexpr int NS = TG / 32;
    constexpr int DPT = MODEL_MIX_DPT;
    const int row = int(tg_pos.y);
    threadgroup float buf[NS];
    threadgroup float red;

    float v[DPT];
    float s2 = 0.0f;
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) {
        const int d = int(tid) + c * TG;
        float val = d < MODEL_D ? pre_norm[size_t(row) * MODEL_D + d] : 0.0f;
        v[c] = val;
        s2 += val * val;
    }
    const float ss = simd_sum(s2);
    if (lane == 0) buf[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float tot = simd_sum(int(lane) < NS ? buf[lane] : 0.0f);
        if (lane == 0) red = tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = metal::precise::rsqrt(red / float(MODEL_D) + p.eps);
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) {
        const int d = int(tid) + c * TG;
        if (d < MODEL_D) {
            const float val = v[c] * inv * float(norm_w[d]);
            out_f32[size_t(row) * MODEL_D + d] = val;
            if (p.cap_slot >= 0) {
                out_cap_bf16[(size_t(row) * p.n_cap + p.cap_slot) * MODEL_D + d] = bfloat(val);
            }
        }
    }
}

kernel void inkling_final_rms_norm_bf16(
    const device float* pre_norm [[buffer(0)]],
    const device bfloat* norm_w  [[buffer(1)]],
    device bfloat* out_bf16      [[buffer(2)]],
    constant FinalNormPrm& p     [[buffer(3)]],
    uint2 tg_pos                 [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]],
    uint lane                    [[thread_index_in_simdgroup]],
    uint sg                      [[simdgroup_index_in_threadgroup]])
{
    constexpr int TG = MODEL_MIX_TG;
    constexpr int NS = TG / 32;
    constexpr int DPT = MODEL_MIX_DPT;
    const int row = int(tg_pos.y);
    threadgroup float buf[NS];
    threadgroup float red;

    float v[DPT];
    float s2 = 0.0f;
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) {
        const int d = int(tid) + c * TG;
        float val = d < MODEL_D ? pre_norm[size_t(row) * MODEL_D + d] : 0.0f;
        v[c] = val;
        s2 += val * val;
    }
    const float ss = simd_sum(s2);
    if (lane == 0) buf[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float tot = simd_sum(int(lane) < NS ? buf[lane] : 0.0f);
        if (lane == 0) red = tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = metal::precise::rsqrt(red / float(MODEL_D) + p.eps);
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) {
        const int d = int(tid) + c * TG;
        if (d < MODEL_D) {
            out_bf16[size_t(row) * MODEL_D + d] = bfloat(v[c] * inv * float(norm_w[d]));
        }
    }
}

struct MixPrm {
    int M, Q, QO, QN, MOFF;
    int ADD, NORM, SEQ;
    float eps;
};

#define MIX_REDUCE2(n, vals, tots) \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (int i_ = 0; i_ < (n); i_++) { \
        const float s_ = simd_sum((vals)[i_]); \
        if (lane == 0) buf[i_][sg] = s_; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (sg == 0) { \
        for (int i_ = 0; i_ < (n); i_++) { \
            const float s_ = simd_sum(int(lane) < NS ? buf[i_][lane] : 0.0f); \
            if (lane == 0) red[i_] = s_; \
        } \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (int i_ = 0; i_ < (n); i_++) (tots)[i_] = red[i_];

kernel void inkling_mix(
    const device bfloat* x         [[buffer(0)]],
    const device float* blocks     [[buffer(1)]],
    const device float* queries    [[buffer(2)]],
    const device float* state      [[buffer(3)]],
    const device float* taps       [[buffer(4)]],
    const device bfloat* res       [[buffer(5)]],
    const device float* messages   [[buffer(6)]],
    const device bfloat* down      [[buffer(7)]],
    const device float* forget     [[buffer(8)]],
    const device float* gain       [[buffer(9)]],
    const device bfloat* norm_w    [[buffer(10)]],
    device void* out_raw           [[buffer(11)]],
    device float* partial          [[buffer(12)]],
    device float* resid            [[buffer(13)]],
    device float* new_state        [[buffer(14)]],
    constant MixPrm& p             [[buffer(15)]],
    threadgroup float* tg_on       [[threadgroup(0)]],
    threadgroup float* tg_mn       [[threadgroup(1)]],
    uint2 tg_pos                   [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint lane                      [[thread_index_in_simdgroup]],
    uint sg                        [[simdgroup_index_in_threadgroup]])
{
    constexpr int TG = MODEL_MIX_TG;
    constexpr int NS = TG / 32;
    constexpr int DPT = MODEL_MIX_DPT;
    const int NSRC = FC_K + 1;
    const uint j = tg_pos.x;
    const uint row = tg_pos.y;
    const int jq = (p.Q == 1) ? 0 : int(j);

    threadgroup float buf[MAX_NRED][NS];
    threadgroup float red[MAX_NRED];

    float v[DPT];
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) {
        const int d = int(tid) + c * TG;
        float val = 0.0f;
        if (d < MODEL_D) {
            val = float(x[size_t(row) * MODEL_D + d]);
            if (FC_CONV) {
                const size_t so = size_t(row) * 3 * MODEL_D + d;
                const size_t to = size_t(d) * 4;
                float s0, s1, s2;
                if (p.SEQ) {
                    const int r = int(row);
                    s0 = r >= 3 ? float(x[size_t(r - 3) * MODEL_D + d]) : state[size_t(r) * MODEL_D + d];
                    s1 = r >= 2 ? float(x[size_t(r - 2) * MODEL_D + d]) : state[size_t(r + 1) * MODEL_D + d];
                    s2 = r >= 1 ? float(x[size_t(r - 1) * MODEL_D + d]) : state[size_t(r + 2) * MODEL_D + d];
                } else {
                    s0 = state[so]; s1 = state[so + MODEL_D]; s2 = state[so + 2 * MODEL_D];
                }
                float h = (taps[to] * s0 + taps[to + 1] * s1 + taps[to + 2] * s2 + taps[to + 3] * val + val) * 0.4472135955f;
                h = round_bf16(h);
                new_state[so] = s1;
                new_state[so + MODEL_D] = s2;
                new_state[so + 2 * MODEL_D] = val;
                val = p.ADD ? float(res[size_t(row) * MODEL_D + d]) + h : h;
                resid[size_t(row) * MODEL_D + d] = val;
            }
        }
        v[c] = val;
    }

    float one[1], tot1[1];
    one[0] = 0.0f;
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) one[0] += v[c] * v[c];
    MIX_REDUCE2(1, one, tot1)
    const float inv = metal::precise::rsqrt(tot1[0] / float(MODEL_D) + 1e-6f);
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) {
        const int d = int(tid) + c * TG;
        v[c] *= inv;
        if (d < MODEL_D && j == 0) partial[size_t(row) * MODEL_D + d] = v[c];
    }

    float acc[MAX_NRED], w[MAX_NRED];
    #pragma clang loop unroll(full)
    for (int i = 0; i < NSRC; i++) acc[i] = 0.0f;
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) {
        const int d = int(tid) + c * TG;
        if (d < MODEL_D) {
            const float qv = queries[size_t(jq) * MODEL_D + d];
            #pragma clang loop unroll(full)
            for (int i = 0; i < NSRC; i++) {
                const float sv = (i < FC_K) ? blocks[(size_t(i) * p.M + row) * MODEL_D + d] : v[c];
                acc[i] += sv * qv;
            }
        }
    }
    MIX_REDUCE2(NSRC, acc, w)
    const float sc = metal::precise::rsqrt(float(MODEL_D));
    float mw = -INFINITY, zw = 0.0f;
    #pragma clang loop unroll(full)
    for (int i = 0; i < NSRC; i++) mw = max(mw, w[i] * sc);
    #pragma clang loop unroll(full)
    for (int i = 0; i < NSRC; i++) {
        w[i] = metal::precise::exp(w[i] * sc - mw);
        zw += w[i];
    }
    float o[DPT];
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) {
        const int d = int(tid) + c * TG;
        float s = 0.0f;
        if (d < MODEL_D) {
            #pragma clang loop unroll(full)
            for (int i = 0; i < NSRC; i++) {
                const float sv = (i < FC_K) ? blocks[(size_t(i) * p.M + row) * MODEL_D + d] : v[c];
                s += w[i] / zw * sv;
            }
        }
        o[c] = s;
    }

    if (FC_GATE) {
        constexpr int RANK = MODEL_RANK;
        constexpr int NCH = TG / RANK;
        constexpr int CD = (MODEL_D + NCH - 1) / NCH;
        threadgroup float pg[TG];
        threadgroup float low[RANK];
        float mm[DPT], two[2], tt[2];
        two[0] = 0.0f; two[1] = 0.0f;
        #pragma clang loop unroll(full)
        for (int c = 0; c < DPT; c++) {
            const int d = int(tid) + c * TG;
            mm[c] = d < MODEL_D ? messages[(size_t(row) * 4 + p.MOFF + j) * MODEL_D + d] : 0.0f;
            two[0] += o[c] * o[c];
            two[1] += mm[c] * mm[c];
        }
        MIX_REDUCE2(2, two, tt)
        const float io = metal::precise::rsqrt(tt[0] / float(MODEL_D) + 1e-6f);
        const float im = metal::precise::rsqrt(tt[1] / float(MODEL_D) + 1e-6f);
        #pragma clang loop unroll(full)
        for (int c = 0; c < DPT; c++) {
            const int d = int(tid) + c * TG;
            if (d < MODEL_D) {
                tg_on[d] = round_bf16(o[c] * io);
                tg_mn[d] = round_bf16(mm[c] * im);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int r = int(tid) % RANK;
        const int chunk = int(tid) / RANK;
        const size_t base = (size_t(j) * RANK + r) * 2 * MODEL_D;
        float gp = 0.0f;
        if (chunk < NCH) {
            const int d_end = min(MODEL_D, (chunk + 1) * CD);
            for (int d = chunk * CD; d < d_end; d++) {
                gp += float(down[base + d]) * tg_on[d] + float(down[base + MODEL_D + d]) * tg_mn[d];
            }
        }
        pg[tid] = gp;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (chunk == 0) {
            for (int c = 1; c < NCH; c++) gp += pg[c * RANK + r];
            low[r] = round_bf16(gp);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (int c = 0; c < DPT; c++) {
            const int d = int(tid) + c * TG;
            if (d < MODEL_D) {
                const size_t fb = (size_t(j) * MODEL_D + d) * RANK;
                float f = 0.0f, gn = 0.0f;
                for (int k = 0; k < RANK; k++) {
                    f += forget[fb + k] * low[k];
                    gn += gain[fb + k] * low[k];
                }
                o[c] = (1.0f - metal::precise::tanh(f)) * o[c] + metal::precise::tanh(gn) * mm[c];
            }
        }
    }

    if (p.NORM) {
        device bfloat* out_bf = (device bfloat*)out_raw;
        float n2[1], nt[1];
        n2[0] = 0.0f;
        #pragma clang loop unroll(full)
        for (int c = 0; c < DPT; c++) n2[0] += o[c] * o[c];
        MIX_REDUCE2(1, n2, nt)
        const float invj = metal::precise::rsqrt(nt[0] / float(MODEL_D) + p.eps);
        #pragma clang loop unroll(full)
        for (int c = 0; c < DPT; c++) {
            const int d = int(tid) + c * TG;
            if (d < MODEL_D) {
                out_bf[(size_t(row) * p.QO + j) * MODEL_D + d] =
                    bfloat(o[c] * invj * float(norm_w[size_t(j % uint(p.QN)) * MODEL_D + d]));
            }
        }
    } else {
        device float* out_f = (device float*)out_raw;
        #pragma clang loop unroll(full)
        for (int c = 0; c < DPT; c++) {
            const int d = int(tid) + c * TG;
            if (d < MODEL_D) {
                out_f[(size_t(row) * p.QO + j) * MODEL_D + d] = o[c];
            }
        }
    }
}

struct MixQ4Prm {
    int M;
};

kernel void inkling_mix_q4(
    const device bfloat* x       [[buffer(0)]],
    const device float* blocks   [[buffer(1)]],
    const device float* queries  [[buffer(2)]],
    device float* messages       [[buffer(3)]],
    constant MixQ4Prm& p         [[buffer(4)]],
    uint2 tg_pos                 [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]],
    uint lane                    [[thread_index_in_simdgroup]],
    uint sg                      [[simdgroup_index_in_threadgroup]])
{
    constexpr int TG = MODEL_MIX_TG;
    constexpr int NS = TG / 32;
    constexpr int DPT = MODEL_MIX_DPT;
    const int NSRC = FC_K + 1;
    const uint row = tg_pos.y;

    threadgroup float buf[MAX_NRED][NS];
    threadgroup float red[MAX_NRED];
    threadgroup float w_sm[4][MAX_NRED];

    float v[DPT];
    float s2 = 0.0f;
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) {
        const int d = int(tid) + c * TG;
        float val = d < MODEL_D ? float(x[size_t(row) * MODEL_D + d]) : 0.0f;
        v[c] = val;
        s2 += val * val;
    }
    float one[1] = {s2}, tot1[1];
    MIX_REDUCE2(1, one, tot1)
    const float inv = metal::precise::rsqrt(tot1[0] / float(MODEL_D) + 1e-6f);
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) v[c] *= inv;

    float acc0[MAX_NRED], acc1[MAX_NRED], acc2[MAX_NRED], acc3[MAX_NRED];
    #pragma clang loop unroll(full)
    for (int i = 0; i < NSRC; i++) {
        acc0[i] = 0.0f; acc1[i] = 0.0f; acc2[i] = 0.0f; acc3[i] = 0.0f;
    }
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) {
        const int d = int(tid) + c * TG;
        if (d < MODEL_D) {
            const float q0 = queries[d];
            const float q1 = queries[MODEL_D + d];
            const float q2 = queries[2 * MODEL_D + d];
            const float q3 = queries[3 * MODEL_D + d];
            #pragma clang loop unroll(full)
            for (int i = 0; i < NSRC; i++) {
                const float sv = (i < FC_K) ? blocks[(size_t(i) * p.M + row) * MODEL_D + d] : v[c];
                acc0[i] += sv * q0;
                acc1[i] += sv * q1;
                acc2[i] += sv * q2;
                acc3[i] += sv * q3;
            }
        }
    }

    const float sc = metal::precise::rsqrt(float(MODEL_D));
    for (int q = 0; q < 4; q++) {
        float tmp_acc[MAX_NRED], w[MAX_NRED];
        #pragma clang loop unroll(full)
        for (int i = 0; i < NSRC; i++) {
            tmp_acc[i] = (q == 0) ? acc0[i] : ((q == 1) ? acc1[i] : ((q == 2) ? acc2[i] : acc3[i]));
        }
        MIX_REDUCE2(NSRC, tmp_acc, w)
        float mw = -INFINITY, zw = 0.0f;
        #pragma clang loop unroll(full)
        for (int i = 0; i < NSRC; i++) mw = max(mw, w[i] * sc);
        #pragma clang loop unroll(full)
        for (int i = 0; i < NSRC; i++) {
            w[i] = metal::precise::exp(w[i] * sc - mw);
            zw += w[i];
        }
        if (tid == 0) {
            for (int i = 0; i < NSRC; i++) w_sm[q][i] = w[i] / zw;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) {
        const int d = int(tid) + c * TG;
        if (d < MODEL_D) {
            float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
            #pragma clang loop unroll(full)
            for (int i = 0; i < NSRC; i++) {
                const float sv = (i < FC_K) ? blocks[(size_t(i) * p.M + row) * MODEL_D + d] : v[c];
                o0 += w_sm[0][i] * sv;
                o1 += w_sm[1][i] * sv;
                o2 += w_sm[2][i] * sv;
                o3 += w_sm[3][i] * sv;
            }
            const size_t mb = size_t(row) * 4 * MODEL_D + d;
            messages[mb]               = o0;
            messages[mb + MODEL_D]     = o1;
            messages[mb + 2 * MODEL_D] = o2;
            messages[mb + 3 * MODEL_D] = o3;
        }
    }
}

struct MixGateP1Prm {
    int M, Q, QO, MOFF;
    int ADD, SEQ;
};

kernel void inkling_mix_gate_p1(
    const device bfloat* x         [[buffer(0)]],
    const device float* blocks     [[buffer(1)]],
    const device float* queries    [[buffer(2)]],
    const device float* state      [[buffer(3)]],
    const device float* taps       [[buffer(4)]],
    const device bfloat* res       [[buffer(5)]],
    const device float* messages   [[buffer(6)]],
    device float* o_out            [[buffer(7)]],
    device bfloat* gate_in         [[buffer(8)]],
    device float* partial          [[buffer(9)]],
    device float* resid            [[buffer(10)]],
    device float* new_state        [[buffer(11)]],
    constant MixGateP1Prm& p       [[buffer(12)]],
    uint2 tg_pos                   [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint lane                      [[thread_index_in_simdgroup]],
    uint sg                        [[simdgroup_index_in_threadgroup]])
{
    constexpr int TG = MODEL_MIX_TG;
    constexpr int NS = TG / 32;
    constexpr int DPT = MODEL_MIX_DPT;
    const int NSRC = FC_K + 1;
    const uint j = tg_pos.x;
    const uint row = tg_pos.y;
    const int jq = (p.Q == 1) ? 0 : int(j);

    threadgroup float buf[MAX_NRED][NS];
    threadgroup float red[MAX_NRED];

    float v[DPT];
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) {
        const int d = int(tid) + c * TG;
        float val = 0.0f;
        if (d < MODEL_D) {
            val = float(x[size_t(row) * MODEL_D + d]);
            if (FC_CONV) {
                const size_t so = size_t(row) * 3 * MODEL_D + d;
                const size_t to = size_t(d) * 4;
                float s0, s1, s2;
                if (p.SEQ) {
                    const int r = int(row);
                    s0 = r >= 3 ? float(x[size_t(r - 3) * MODEL_D + d]) : state[size_t(r) * MODEL_D + d];
                    s1 = r >= 2 ? float(x[size_t(r - 2) * MODEL_D + d]) : state[size_t(r + 1) * MODEL_D + d];
                    s2 = r >= 1 ? float(x[size_t(r - 1) * MODEL_D + d]) : state[size_t(r + 2) * MODEL_D + d];
                } else {
                    s0 = state[so]; s1 = state[so + MODEL_D]; s2 = state[so + 2 * MODEL_D];
                }
                float h = (taps[to] * s0 + taps[to + 1] * s1 + taps[to + 2] * s2 + taps[to + 3] * val + val) * 0.4472135955f;
                h = round_bf16(h);
                new_state[so] = s1;
                new_state[so + MODEL_D] = s2;
                new_state[so + 2 * MODEL_D] = val;
                val = p.ADD ? float(res[size_t(row) * MODEL_D + d]) + h : h;
                resid[size_t(row) * MODEL_D + d] = val;
            }
        }
        v[c] = val;
    }

    float one[1], tot1[1];
    one[0] = 0.0f;
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) one[0] += v[c] * v[c];
    MIX_REDUCE2(1, one, tot1)
    const float inv = metal::precise::rsqrt(tot1[0] / float(MODEL_D) + 1e-6f);
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) {
        const int d = int(tid) + c * TG;
        v[c] *= inv;
        if (d < MODEL_D && j == 0) partial[size_t(row) * MODEL_D + d] = v[c];
    }

    float acc[MAX_NRED], w[MAX_NRED];
    #pragma clang loop unroll(full)
    for (int i = 0; i < NSRC; i++) acc[i] = 0.0f;
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) {
        const int d = int(tid) + c * TG;
        if (d < MODEL_D) {
            const float qv = queries[size_t(jq) * MODEL_D + d];
            #pragma clang loop unroll(full)
            for (int i = 0; i < NSRC; i++) {
                const float sv = (i < FC_K) ? blocks[(size_t(i) * p.M + row) * MODEL_D + d] : v[c];
                acc[i] += sv * qv;
            }
        }
    }
    MIX_REDUCE2(NSRC, acc, w)
    const float sc = metal::precise::rsqrt(float(MODEL_D));
    float mw = -INFINITY, zw = 0.0f;
    #pragma clang loop unroll(full)
    for (int i = 0; i < NSRC; i++) mw = max(mw, w[i] * sc);
    #pragma clang loop unroll(full)
    for (int i = 0; i < NSRC; i++) {
        w[i] = metal::precise::exp(w[i] * sc - mw);
        zw += w[i];
    }

    float o[DPT], mm[DPT], two[2], tt[2];
    two[0] = 0.0f; two[1] = 0.0f;
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) {
        const int d = int(tid) + c * TG;
        float s = 0.0f;
        float mv = 0.0f;
        if (d < MODEL_D) {
            #pragma clang loop unroll(full)
            for (int i = 0; i < NSRC; i++) {
                const float sv = (i < FC_K) ? blocks[(size_t(i) * p.M + row) * MODEL_D + d] : v[c];
                s += w[i] / zw * sv;
            }
            mv = messages[(size_t(row) * 4 + p.MOFF + j) * MODEL_D + d];
        }
        o[c] = s;
        mm[c] = mv;
        two[0] += s * s;
        two[1] += mv * mv;
    }
    MIX_REDUCE2(2, two, tt)
    const float io = metal::precise::rsqrt(tt[0] / float(MODEL_D) + 1e-6f);
    const float im = metal::precise::rsqrt(tt[1] / float(MODEL_D) + 1e-6f);
    const size_t goff = (size_t(j) * p.M + row) * (2 * MODEL_D);
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) {
        const int d = int(tid) + c * TG;
        if (d < MODEL_D) {
            o_out[(size_t(row) * p.QO + j) * MODEL_D + d] = o[c];
            gate_in[goff + d]           = bfloat(o[c] * io);
            gate_in[goff + MODEL_D + d] = bfloat(mm[c] * im);
        }
    }
}

struct MsgDownPrm {
    int M;
};

inline void store_msg_down(device float* dst, size_t idx, float v) { dst[idx] = round_bf16(v); }
inline void store_msg_down(device bfloat* dst, size_t idx, float v) { dst[idx] = bfloat(v); }

template <typename OutT>
inline void msg_down_batched_impl(
    const device bfloat* gate_in,
    const device bfloat* down,
    device OutT* low_out,
    constant MsgDownPrm& p,
    uint3 tg_pos,
    uint lane,
    uint sg)
{
    const int r = int(tg_pos.x) * 8 + int(sg);
    const int j = int(tg_pos.y);
    const int m0 = int(tg_pos.z) * 16;
    if (r >= MODEL_RANK || m0 >= p.M) return;
    const int mn = min(16, p.M - m0);

    constexpr int K = 2 * MODEL_D;
    constexpr int K4 = K / 4;
    const size_t w_base = (size_t(j) * MODEL_RANK + r) * K;
    const device vec<bfloat, 4>* w4 = (const device vec<bfloat, 4>*)(down + w_base);
    const device bfloat* g_base = gate_in + size_t(j) * p.M * K;

    float acc[16];
    #pragma clang loop unroll(full)
    for (int ti = 0; ti < 16; ++ti) acc[ti] = 0.0f;

    for (int idx = int(lane); idx < K4; idx += 32) {
        const vec<float, 4> wf = vec<float, 4>(w4[idx]);
        if (mn == 16) {
            #pragma clang loop unroll(full)
            for (int ti = 0; ti < 16; ++ti) {
                const device vec<bfloat, 4>* g4 = (const device vec<bfloat, 4>*)(g_base + size_t(m0 + ti) * K);
                acc[ti] += dot(wf, vec<float, 4>(g4[idx]));
            }
        } else {
            #pragma clang loop unroll(full)
            for (int ti = 0; ti < 16; ++ti) {
                if (ti < mn) {
                    const device vec<bfloat, 4>* g4 = (const device vec<bfloat, 4>*)(g_base + size_t(m0 + ti) * K);
                    acc[ti] += dot(wf, vec<float, 4>(g4[idx]));
                }
            }
        }
    }
    for (int k = K4 * 4 + int(lane); k < K; k += 32) {
        const float wv = float(down[w_base + k]);
        #pragma clang loop unroll(full)
        for (int ti = 0; ti < 16; ++ti) {
            if (ti < mn) {
                acc[ti] += wv * float(g_base[size_t(m0 + ti) * K + k]);
            }
        }
    }
    #pragma clang loop unroll(full)
    for (int ti = 0; ti < 16; ++ti) {
        if (ti < mn) {
            const float tot = simd_sum(acc[ti]);
            if (lane == 0) {
                store_msg_down(low_out, (size_t(j) * p.M + (m0 + ti)) * MODEL_RANK + r, tot);
            }
        }
    }
}

kernel void inkling_msg_down_batched(
    const device bfloat* gate_in [[buffer(0)]],
    const device bfloat* down    [[buffer(1)]],
    device float* low_out        [[buffer(2)]],
    constant MsgDownPrm& p       [[buffer(3)]],
    uint3 tg_pos                 [[threadgroup_position_in_grid]],
    uint lane                    [[thread_index_in_simdgroup]],
    uint sg                      [[simdgroup_index_in_threadgroup]])
{
    msg_down_batched_impl<float>(gate_in, down, low_out, p, tg_pos, lane, sg);
}

kernel void inkling_msg_down_batched_bf16(
    const device bfloat* gate_in [[buffer(0)]],
    const device bfloat* down    [[buffer(1)]],
    device bfloat* low_out       [[buffer(2)]],
    constant MsgDownPrm& p       [[buffer(3)]],
    uint3 tg_pos                 [[threadgroup_position_in_grid]],
    uint lane                    [[thread_index_in_simdgroup]],
    uint sg                      [[simdgroup_index_in_threadgroup]])
{
    msg_down_batched_impl<bfloat>(gate_in, down, low_out, p, tg_pos, lane, sg);
}

struct MsgFgPrm {
    int M, QO, MOFF;
};

kernel void inkling_msg_fg_blend(
    const device float* low_in   [[buffer(0)]],
    const device float* forget   [[buffer(1)]],
    const device float* gain     [[buffer(2)]],
    const device float* messages [[buffer(3)]],
    device float* o_out          [[buffer(4)]],
    constant MsgFgPrm& p         [[buffer(5)]],
    uint2 tg_pos                 [[threadgroup_position_in_grid]],
    uint2 ltid                   [[thread_position_in_threadgroup]])
{
    const int d0 = int(tg_pos.x) * 32;
    const int j = int(tg_pos.y);
    const int d_l = int(ltid.x);
    const int row_w = int(ltid.y);
    const int d = d0 + d_l;
    const int tid = row_w * 32 + d_l;

    threadgroup float tg_low[64][17];
    threadgroup float tg_f[32][17];
    threadgroup float tg_g[32][17];

    float acc_f[8], acc_g[8];
    #pragma clang loop unroll(full)
    for (int w = 0; w < 8; ++w) { acc_f[w] = 0.0f; acc_g[w] = 0.0f; }

    for (int r0 = 0; r0 < MODEL_RANK; r0 += 16) {
        const int rn = min(16, MODEL_RANK - r0);
        for (int idx = tid; idx < p.M * rn; idx += 256) {
            const int mr = idx / rn;
            const int rk = idx % rn;
            tg_low[mr][rk] = low_in[(size_t(j) * p.M + mr) * MODEL_RANK + r0 + rk];
        }
        const int dc = tid >> 4;
        const int rk = tid & 15;
        if (rk < rn) {
            if (d0 + dc < MODEL_D) {
                const size_t fb = (size_t(j) * MODEL_D + (d0 + dc)) * MODEL_RANK + r0 + rk;
                tg_f[dc][rk] = forget[fb];
                tg_g[dc][rk] = gain[fb];
            }
            if (d0 + dc + 16 < MODEL_D) {
                const size_t fb = (size_t(j) * MODEL_D + (d0 + dc + 16)) * MODEL_RANK + r0 + rk;
                tg_f[dc + 16][rk] = forget[fb];
                tg_g[dc + 16][rk] = gain[fb];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (d < MODEL_D) {
            for (int w = 0; w < 8; ++w) {
                const int mr = row_w + w * 8;
                if (mr < p.M) {
                    float af = 0.0f, ag = 0.0f;
                    for (int k = 0; k < rn; ++k) {
                        const float lv = tg_low[mr][k];
                        af += tg_f[d_l][k] * lv;
                        ag += tg_g[d_l][k] * lv;
                    }
                    acc_f[w] += af;
                    acc_g[w] += ag;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (d < MODEL_D) {
        for (int w = 0; w < 8; ++w) {
            const int mr = row_w + w * 8;
            if (mr < p.M) {
                const size_t oo = (size_t(mr) * p.QO + j) * MODEL_D + d;
                const size_t mo = (size_t(mr) * 4 + p.MOFF + j) * MODEL_D + d;
                o_out[oo] = (1.0f - metal::precise::tanh(acc_f[w])) * o_out[oo]
                          + metal::precise::tanh(acc_g[w]) * messages[mo];
            }
        }
    }
}

struct RmsMultiPrm {
    int QO, QN;
    float eps;
};

kernel void inkling_rms_norm_multi(
    const device float* x      [[buffer(0)]],
    const device bfloat* norm_w[[buffer(1)]],
    device bfloat* out         [[buffer(2)]],
    constant RmsMultiPrm& p    [[buffer(3)]],
    uint2 tg_pos               [[threadgroup_position_in_grid]],
    uint tid                   [[thread_index_in_threadgroup]],
    uint lane                  [[thread_index_in_simdgroup]],
    uint sg                    [[simdgroup_index_in_threadgroup]])
{
    constexpr int TG = MODEL_MIX_TG;
    constexpr int NS = TG / 32;
    constexpr int DPT = MODEL_MIX_DPT;
    const uint j = tg_pos.x;
    const uint row = tg_pos.y;
    threadgroup float buf[NS];
    threadgroup float red;

    float v[DPT];
    float s2 = 0.0f;
    const size_t base = (size_t(row) * p.QO + j) * MODEL_D;
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) {
        const int d = int(tid) + c * TG;
        float val = d < MODEL_D ? x[base + d] : 0.0f;
        v[c] = val;
        s2 += val * val;
    }
    const float ss = simd_sum(s2);
    if (lane == 0) buf[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float tot = simd_sum(int(lane) < NS ? buf[lane] : 0.0f);
        if (lane == 0) red = tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = metal::precise::rsqrt(red / float(MODEL_D) + p.eps);
    const size_t nb = size_t(j % uint(p.QN)) * MODEL_D;
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) {
        const int d = int(tid) + c * TG;
        if (d < MODEL_D) {
            out[base + d] = bfloat(v[c] * inv * float(norm_w[nb + d]));
        }
    }
}

struct MsgBlendNormPrm {
    int M, QO, QN, MOFF;
    float eps;
};

kernel void inkling_msg_blend_norm_bf16(
    const device float* o_in     [[buffer(0)]],
    const device float* messages [[buffer(1)]],
    const device bfloat* fg      [[buffer(2)]],
    const device bfloat* norm_w  [[buffer(3)]],
    device bfloat* out           [[buffer(4)]],
    constant MsgBlendNormPrm& p  [[buffer(5)]],
    uint2 tg_pos                 [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]],
    uint lane                    [[thread_index_in_simdgroup]],
    uint sg                      [[simdgroup_index_in_threadgroup]])
{
    constexpr int TG = MODEL_MIX_TG;
    constexpr int NS = TG / 32;
    constexpr int DPT = MODEL_MIX_DPT;
    const int j = int(tg_pos.x);
    const int row = int(tg_pos.y);

    threadgroup float buf[NS];
    threadgroup float red;

    const size_t ob = (size_t(row) * p.QO + j) * MODEL_D;
    const size_t mb = (size_t(row) * 4 + p.MOFF + j) * MODEL_D;
    const size_t fb = (size_t(j) * p.M + row) * (2 * MODEL_D);

    float o[DPT];
    float s2 = 0.0f;
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) {
        const int d = int(tid) + c * TG;
        float val = 0.0f;
        if (d < MODEL_D) {
            const float f = float(fg[fb + d]);
            const float g = float(fg[fb + MODEL_D + d]);
            val = (1.0f - metal::precise::tanh(f)) * o_in[ob + d]
                + metal::precise::tanh(g) * messages[mb + d];
            s2 += val * val;
        }
        o[c] = val;
    }
    const float ss = simd_sum(s2);
    if (lane == 0) buf[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float tot = simd_sum(int(lane) < NS ? buf[lane] : 0.0f);
        if (lane == 0) red = tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = metal::precise::rsqrt(red / float(MODEL_D) + p.eps);
    const size_t nb = size_t(j % p.QN) * MODEL_D;
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) {
        const int d = int(tid) + c * TG;
        if (d < MODEL_D) {
            out[ob + d] = bfloat(o[c] * inv * float(norm_w[nb + d]));
        }
    }
}

struct QkvPrm {
    int S, N, AT3, AT4;
};

kernel void inkling_qkv_wide(
    const device bfloat* x [[buffer(0)]],
    const device bfloat* w [[buffer(1)]],
    device bfloat* out     [[buffer(2)]],
    constant QkvPrm& p     [[buffer(3)]],
    uint2 tg_pos           [[threadgroup_position_in_grid]],
    uint lane              [[thread_index_in_simdgroup]],
    uint sg                [[simdgroup_index_in_threadgroup]])
{
    constexpr int UNROLL = 4;
    const int row = int(tg_pos.y) * FC_SG + int(sg);
    if (row >= p.N) return;

    const int src = (p.S == 1) ? 0 : (row < p.AT3 ? 0 : (row < p.AT4 ? 1 : 2));
    const device vec<bfloat, 4>* w4 = (const device vec<bfloat, 4>*)(w + size_t(row) * FC_KDIM);
    const device vec<bfloat, 4>* x4 = (const device vec<bfloat, 4>*)(x);
    const int K4 = FC_KDIM / 4;
    const int K_MAIN = K4 - (K4 % (32 * UNROLL));
    const int x_row_stride4 = (p.S * FC_KDIM) / 4;
    const int x_src_off4 = (src * FC_KDIM) / 4;

    float acc[8];
    #pragma clang loop unroll(full)
    for (int m = 0; m < FC_M; ++m) acc[m] = 0.0f;

    for (int base = 0; base < K_MAIN; base += 32 * UNROLL) {
        vec<float, 4> wf[UNROLL];
        #pragma clang loop unroll(full)
        for (int u = 0; u < UNROLL; ++u) {
            wf[u] = vec<float, 4>(w4[base + u * 32 + lane]);
        }
        #pragma clang loop unroll(full)
        for (int m = 0; m < FC_M; ++m) {
            const int xoff = m * x_row_stride4 + x_src_off4 + base + int(lane);
            float a = 0.0f;
            #pragma clang loop unroll(full)
            for (int u = 0; u < UNROLL; ++u) {
                a += dot(wf[u], vec<float, 4>(x4[xoff + u * 32]));
            }
            acc[m] += a;
        }
    }
    for (int idx = K_MAIN + int(lane); idx < K4; idx += 32) {
        const vec<float, 4> wf = vec<float, 4>(w4[idx]);
        #pragma clang loop unroll(full)
        for (int m = 0; m < FC_M; ++m) {
            acc[m] += dot(wf, vec<float, 4>(x4[m * x_row_stride4 + x_src_off4 + idx]));
        }
    }
    #pragma clang loop unroll(full)
    for (int m = 0; m < FC_M; ++m) {
        const float v = simd_sum(acc[m]);
        if (lane == 0) out[size_t(m) * p.N + row] = bfloat(v);
    }
}

struct KvPrepPrm {
    int M, HQ, HKV, R;
    int QKV_W, QOFF, ROFF, KOFF, VOFF;
    int X_STRIDE, XK_OFF, XV_OFF;
    int TAU, SEQ, WRITE_DIRECT, IS_ROTATING, WRITE_BASE, MAX_SIZE;
    int KV_OFFSET_BEFORE, STRIDE_H, STRIDE_S;
    float qk_thr, rel_thr, eps, log_n_floor, log_alpha;
};

#define KV_REDUCE(n, vals, tots) \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (int i_ = 0; i_ < (n); i_++) { \
        const float s_ = simd_sum((vals)[i_]); \
        if (lane == 0) buf[i_][sg] = s_; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (int i_ = 0; i_ < (n); i_++) { \
        float t_ = 0.0f; \
        for (int g_ = 0; g_ < NS; g_++) t_ += buf[i_][g_]; \
        (tots)[i_] = t_; \
    }

kernel void inkling_kv_prep(
    const device bfloat* qkv    [[buffer(0)]],
    const device bfloat* qkv_in [[buffer(1)]],
    const device bfloat* sink_w [[buffer(2)]],
    const device bfloat* kdyn_w [[buffer(3)]],
    const device bfloat* vdyn_w [[buffer(4)]],
    const device float* kstate  [[buffer(5)]],
    const device float* vstate  [[buffer(6)]],
    const device void* ktaps_v  [[buffer(7)]],
    const device void* vtaps_v  [[buffer(8)]],
    const device bfloat* qscale [[buffer(9)]],
    const device bfloat* rscale [[buffer(10)]],
    const device bfloat* proj   [[buffer(11)]],
    device bfloat* qo           [[buffer(12)]],
    device bfloat* bias         [[buffer(13)]],
    device bfloat* ko_dst       [[buffer(14)]],
    device bfloat* vo_dst       [[buffer(15)]],
    device bfloat* sinks        [[buffer(16)]],
    device float* kstate_new    [[buffer(17)]],
    device float* vstate_new    [[buffer(18)]],
    constant KvPrepPrm& p       [[buffer(19)]],
    uint2 tg_pos                [[threadgroup_position_in_grid]],
    uint t                      [[thread_index_in_threadgroup]],
    uint lane                   [[thread_index_in_simdgroup]],
    uint sg                     [[simdgroup_index_in_threadgroup]])
{
    const int TG = max(32, ((FC_HD + 31) / 32) * 32);
    const int NS = TG / 32;
    const int C = p.HKV * FC_HD;
    const int h = int(tg_pos.x);
    const int row = int(tg_pos.y);

    float tau_ = 1.0f;
    if (p.TAU) {
        const float pos = float(p.KV_OFFSET_BEFORE + 1 + (p.SEQ ? row : 0));
        tau_ = 1.0f + p.log_alpha * metal::precise::log(max(pos / p.log_n_floor, 1.0f));
    }

    threadgroup float buf[9][16];
    threadgroup float rr[MODEL_DR];

    const device bfloat* xq_row = qkv_in + size_t(row) * p.X_STRIDE;
    const device bfloat* xk_row = qkv_in + size_t(row) * p.X_STRIDE + p.XK_OFF;
    const device bfloat* xv_row = qkv_in + size_t(row) * p.X_STRIDE + p.XV_OFF;

    float nine[9], dots[9];
    #pragma clang loop unroll(full)
    for (int i = 0; i < 9; i++) nine[i] = 0.0f;
    for (int d = int(t); d < MODEL_D; d += TG) {
        nine[0] += float(xq_row[d]) * float(sink_w[size_t(h) * MODEL_D + d]);
        if (FC_DYN && h < p.HKV) {
            const float kx = float(xk_row[d]);
            const float vx = float(xv_row[d]);
            #pragma clang loop unroll(full)
            for (int lag = 0; lag < 4; lag++) {
                nine[1 + lag] += kx * float(kdyn_w[size_t(h * 4 + lag) * MODEL_D + d]);
                nine[5 + lag] += vx * float(vdyn_w[size_t(h * 4 + lag) * MODEL_D + d]);
            }
        }
    }
    KV_REDUCE(FC_DYN ? 9 : 1, nine, dots)
    if (t == 0) sinks[size_t(row) * p.HQ + h] = bfloat(dots[0]);

    float four[4], tot[4];
    const float qy = int(t) < FC_HD ? float(qkv[size_t(row) * p.QKV_W + p.QOFF + h * FC_HD + int(t)]) * float(qscale[h * FC_HD + int(t)]) : 0.0f;
    const float ry = int(t) < MODEL_DR ? float(qkv[size_t(row) * p.QKV_W + p.ROFF + h * MODEL_DR + int(t)]) * float(rscale[h * MODEL_DR + int(t)]) : 0.0f;
    float ky = 0.0f, vy = 0.0f;
    if (h < p.HKV && int(t) < FC_HD) {
        const int c = h * FC_HD + int(t);
        const size_t so = size_t(row) * 3 * C + c;
        const float kx = float(qkv[size_t(row) * p.QKV_W + p.KOFF + c]);
        const float vx = float(qkv[size_t(row) * p.QKV_W + p.VOFF + c]);
        float k0, k1, k2, v0, v1, v2;
        if (p.SEQ) {
            k0 = row >= 3 ? float(qkv[size_t(row - 3) * p.QKV_W + p.KOFF + c]) : kstate[size_t(row) * C + c];
            k1 = row >= 2 ? float(qkv[size_t(row - 2) * p.QKV_W + p.KOFF + c]) : kstate[size_t(row + 1) * C + c];
            k2 = row >= 1 ? float(qkv[size_t(row - 1) * p.QKV_W + p.KOFF + c]) : kstate[size_t(row + 2) * C + c];
            v0 = row >= 3 ? float(qkv[size_t(row - 3) * p.QKV_W + p.VOFF + c]) : vstate[size_t(row) * C + c];
            v1 = row >= 2 ? float(qkv[size_t(row - 2) * p.QKV_W + p.VOFF + c]) : vstate[size_t(row + 1) * C + c];
            v2 = row >= 1 ? float(qkv[size_t(row - 1) * p.QKV_W + p.VOFF + c]) : vstate[size_t(row + 2) * C + c];
        } else {
            k0 = kstate[so]; k1 = kstate[so + C]; k2 = kstate[so + 2 * C];
            v0 = vstate[so]; v1 = vstate[so + C]; v2 = vstate[so + 2 * C];
        }
        float kc, vc;
        if (FC_DYN) {
            const device bfloat* ktaps = (const device bfloat*)ktaps_v;
            const device bfloat* vtaps = (const device bfloat*)vtaps_v;
            kc = ((float(ktaps[c]) + round_bf16(dots[1])) * kx
                + (float(ktaps[C + c]) + round_bf16(dots[2])) * k2
                + (float(ktaps[2 * C + c]) + round_bf16(dots[3])) * k1
                + (float(ktaps[3 * C + c]) + round_bf16(dots[4])) * k0
                + kx) * 0.3333333333f;
            vc = ((float(vtaps[c]) + round_bf16(dots[5])) * vx
                + (float(vtaps[C + c]) + round_bf16(dots[6])) * v2
                + (float(vtaps[2 * C + c]) + round_bf16(dots[7])) * v1
                + (float(vtaps[3 * C + c]) + round_bf16(dots[8])) * v0
                + vx) * 0.3333333333f;
        } else {
            const device float* ktaps = (const device float*)ktaps_v;
            const device float* vtaps = (const device float*)vtaps_v;
            const size_t to = size_t(c) * 4;
            kc = (ktaps[to] * k0 + ktaps[to + 1] * k1 + ktaps[to + 2] * k2 + ktaps[to + 3] * kx + kx) * 0.4472135955f;
            vc = (vtaps[to] * v0 + vtaps[to + 1] * v1 + vtaps[to + 2] * v2 + vtaps[to + 3] * vx + vx) * 0.4472135955f;
        }
        ky = round_bf16(kc);
        vy = round_bf16(vc);
        kstate_new[so] = k1; kstate_new[so + C] = k2; kstate_new[so + 2 * C] = kx;
        vstate_new[so] = v1; vstate_new[so + C] = v2; vstate_new[so + 2 * C] = vx;
    }
    four[0] = qy * qy; four[1] = ry * ry; four[2] = ky * ky; four[3] = vy * vy;
    KV_REDUCE(4, four, tot)

    if (int(t) < FC_HD) {
        float qv = round_bf16(qy * min(p.qk_thr / (metal::precise::sqrt(tot[0]) + 1e-6f), 1.0f));
        if (p.TAU) qv = round_bf16(qv * tau_);
        qo[(size_t(row) * p.HQ + h) * FC_HD + int(t)] = bfloat(qv);
        if (h < p.HKV) {
            const bfloat kv_out = bfloat(ky * min(p.qk_thr / (metal::precise::sqrt(tot[2]) + 1e-6f), 1.0f));
            const bfloat vv_out = bfloat(vy * metal::precise::rsqrt(tot[3] / float(FC_HD) + p.eps));
            if (p.WRITE_DIRECT) {
                const int slot = p.IS_ROTATING ? ((p.WRITE_BASE + row) % p.MAX_SIZE) : (p.WRITE_BASE + row);
                const size_t xo = size_t(h) * p.STRIDE_H + size_t(slot) * p.STRIDE_S + int(t);
                ko_dst[xo] = kv_out;
                vo_dst[xo] = vv_out;
            } else {
                const size_t xo = (size_t(row) * p.HKV + h) * FC_HD + int(t);
                ko_dst[xo] = kv_out;
                vo_dst[xo] = vv_out;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(t) < MODEL_DR) {
        float rv = round_bf16(ry * min(p.rel_thr / (metal::precise::sqrt(tot[1]) + 1e-6f), 1.0f));
        if (p.TAU) rv = round_bf16(rv * tau_);
        rr[t] = rv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int d = int(t); d < p.R; d += TG) {
        float b = 0.0f;
        #pragma clang loop unroll(full)
        for (int i = 0; i < MODEL_DR; i++) b += rr[i] * float(proj[size_t(i) * p.R + d]);
        bias[(size_t(row) * p.HQ + h) * p.R + d] = bfloat(b);
    }
}

struct KvPrepBatchedPrm {
    int M, HQ, HKV, R;
    int QKV_W, QOFF, ROFF, KOFF, VOFF;
    int X_STRIDE, XK_OFF, XV_OFF;
    int TAU, WRITE_SLOT, STRIDE_B, STRIDE_H, STRIDE_S;
    float qk_thr, rel_thr, eps;
};

kernel void inkling_kv_prep_batched(
    const device bfloat* qkv    [[buffer(0)]],
    const device bfloat* qkv_in [[buffer(1)]],
    const device bfloat* sink_w [[buffer(2)]],
    const device bfloat* kdyn_w [[buffer(3)]],
    const device bfloat* vdyn_w [[buffer(4)]],
    device float* kstate        [[buffer(5)]],
    device float* vstate        [[buffer(6)]],
    const device void* ktaps_v  [[buffer(7)]],
    const device void* vtaps_v  [[buffer(8)]],
    const device bfloat* qscale [[buffer(9)]],
    const device bfloat* rscale [[buffer(10)]],
    const device bfloat* proj   [[buffer(11)]],
    const device float* tau_buf [[buffer(12)]],
    device bfloat* qo           [[buffer(13)]],
    device bfloat* bias         [[buffer(14)]],
    device bfloat* ko_dst       [[buffer(15)]],
    device bfloat* vo_dst       [[buffer(16)]],
    device bfloat* sinks        [[buffer(17)]],
    constant KvPrepBatchedPrm& p[[buffer(18)]],
    uint2 tg_pos                [[threadgroup_position_in_grid]],
    uint t                      [[thread_index_in_threadgroup]],
    uint lane                   [[thread_index_in_simdgroup]],
    uint sg                     [[simdgroup_index_in_threadgroup]])
{
    const int TG = max(32, ((FC_HD + 31) / 32) * 32);
    const int NS = TG / 32;
    const int C = p.HKV * FC_HD;
    const int h = int(tg_pos.x);
    const int row = int(tg_pos.y);
    const float tau_ = p.TAU ? tau_buf[row] : 1.0f;

    threadgroup float buf[9][16];
    threadgroup float rr[MODEL_DR];

    const device bfloat* xq_row = qkv_in + size_t(row) * p.X_STRIDE;
    const device bfloat* xk_row = qkv_in + size_t(row) * p.X_STRIDE + p.XK_OFF;
    const device bfloat* xv_row = qkv_in + size_t(row) * p.X_STRIDE + p.XV_OFF;

    float nine[9], dots[9];
    #pragma clang loop unroll(full)
    for (int i = 0; i < 9; i++) nine[i] = 0.0f;
    for (int d = int(t); d < MODEL_D; d += TG) {
        nine[0] += float(xq_row[d]) * float(sink_w[size_t(h) * MODEL_D + d]);
        if (FC_DYN && h < p.HKV) {
            const float kx = float(xk_row[d]);
            const float vx = float(xv_row[d]);
            #pragma clang loop unroll(full)
            for (int lag = 0; lag < 4; lag++) {
                nine[1 + lag] += kx * float(kdyn_w[size_t(h * 4 + lag) * MODEL_D + d]);
                nine[5 + lag] += vx * float(vdyn_w[size_t(h * 4 + lag) * MODEL_D + d]);
            }
        }
    }
    KV_REDUCE(FC_DYN ? 9 : 1, nine, dots)
    if (t == 0) sinks[size_t(row) * p.HQ + h] = bfloat(dots[0]);

    float four[4], tot[4];
    const float qy = int(t) < FC_HD ? float(qkv[size_t(row) * p.QKV_W + p.QOFF + h * FC_HD + int(t)]) * float(qscale[h * FC_HD + int(t)]) : 0.0f;
    const float ry = int(t) < MODEL_DR ? float(qkv[size_t(row) * p.QKV_W + p.ROFF + h * MODEL_DR + int(t)]) * float(rscale[h * MODEL_DR + int(t)]) : 0.0f;
    float ky = 0.0f, vy = 0.0f;
    if (h < p.HKV && int(t) < FC_HD) {
        const int c = h * FC_HD + int(t);
        const size_t so = size_t(row) * 3 * C + c;
        const float kx = float(qkv[size_t(row) * p.QKV_W + p.KOFF + c]);
        const float vx = float(qkv[size_t(row) * p.QKV_W + p.VOFF + c]);
        const float k0 = kstate[so], k1 = kstate[so + C], k2 = kstate[so + 2 * C];
        const float v0 = vstate[so], v1 = vstate[so + C], v2 = vstate[so + 2 * C];
        float kc, vc;
        if (FC_DYN) {
            const device bfloat* ktaps = (const device bfloat*)ktaps_v;
            const device bfloat* vtaps = (const device bfloat*)vtaps_v;
            kc = ((float(ktaps[c]) + round_bf16(dots[1])) * kx
                + (float(ktaps[C + c]) + round_bf16(dots[2])) * k2
                + (float(ktaps[2 * C + c]) + round_bf16(dots[3])) * k1
                + (float(ktaps[3 * C + c]) + round_bf16(dots[4])) * k0
                + kx) * 0.3333333333f;
            vc = ((float(vtaps[c]) + round_bf16(dots[5])) * vx
                + (float(vtaps[C + c]) + round_bf16(dots[6])) * v2
                + (float(vtaps[2 * C + c]) + round_bf16(dots[7])) * v1
                + (float(vtaps[3 * C + c]) + round_bf16(dots[8])) * v0
                + vx) * 0.3333333333f;
        } else {
            const device float* ktaps = (const device float*)ktaps_v;
            const device float* vtaps = (const device float*)vtaps_v;
            const size_t to = size_t(c) * 4;
            kc = (ktaps[to] * k0 + ktaps[to + 1] * k1 + ktaps[to + 2] * k2 + ktaps[to + 3] * kx + kx) * 0.4472135955f;
            vc = (vtaps[to] * v0 + vtaps[to + 1] * v1 + vtaps[to + 2] * v2 + vtaps[to + 3] * vx + vx) * 0.4472135955f;
        }
        ky = round_bf16(kc);
        vy = round_bf16(vc);
        kstate[so] = k1; kstate[so + C] = k2; kstate[so + 2 * C] = kx;
        vstate[so] = v1; vstate[so + C] = v2; vstate[so + 2 * C] = vx;
    }
    four[0] = qy * qy; four[1] = ry * ry; four[2] = ky * ky; four[3] = vy * vy;
    KV_REDUCE(4, four, tot)

    if (int(t) < FC_HD) {
        float qv = round_bf16(qy * min(p.qk_thr / (metal::precise::sqrt(tot[0]) + 1e-6f), 1.0f));
        if (p.TAU) qv = round_bf16(qv * tau_);
        qo[(size_t(row) * p.HQ + h) * FC_HD + int(t)] = bfloat(qv);
        if (h < p.HKV) {
            const bfloat kv_out = bfloat(ky * min(p.qk_thr / (metal::precise::sqrt(tot[2]) + 1e-6f), 1.0f));
            const bfloat vv_out = bfloat(vy * metal::precise::rsqrt(tot[3] / float(FC_HD) + p.eps));
            const size_t xo = size_t(row) * size_t(p.STRIDE_B)
                            + size_t(h) * size_t(p.STRIDE_H)
                            + size_t(p.WRITE_SLOT) * size_t(p.STRIDE_S)
                            + int(t);
            ko_dst[xo] = kv_out;
            vo_dst[xo] = vv_out;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(t) < MODEL_DR) {
        float rv = round_bf16(ry * min(p.rel_thr / (metal::precise::sqrt(tot[1]) + 1e-6f), 1.0f));
        if (p.TAU) rv = round_bf16(rv * tau_);
        rr[t] = rv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int d = int(t); d < p.R; d += TG) {
        float b = 0.0f;
        #pragma clang loop unroll(full)
        for (int i = 0; i < MODEL_DR; i++) b += rr[i] * float(proj[size_t(i) * p.R + d]);
        bias[(size_t(row) * p.HQ + h) * p.R + d] = bfloat(b);
    }
}

struct AttendPrm {
    int M, HQ, HKV, R, GW, GOFF, WINDOW;
    int S_kv, ext_m, ring_idx, stride_h, stride_s;
    float scale;
};

kernel void inkling_attend(
    const device bfloat* q      [[buffer(0)]],
    const device bfloat* keys   [[buffer(1)]],
    const device bfloat* values [[buffer(2)]],
    const device bfloat* ko_ext [[buffer(3)]],
    const device bfloat* vo_ext [[buffer(4)]],
    const device bfloat* bias   [[buffer(5)]],
    const device bfloat* sinks  [[buffer(6)]],
    const device bfloat* gate   [[buffer(7)]],
    device bfloat* out          [[buffer(8)]],
    constant AttendPrm& p       [[buffer(9)]],
    threadgroup float* p_buf    [[threadgroup(0)]],
    uint2 tg_pos                [[threadgroup_position_in_grid]],
    uint t                      [[thread_index_in_threadgroup]],
    uint lane                   [[thread_index_in_simdgroup]],
    uint sg                     [[simdgroup_index_in_threadgroup]])
{
    const int TG = min(1024, ((FC_HD * 8 + 31) / 32) * 32);
    const int NS = TG / 32;
    const int CH = TG / FC_HD;
    const int CK = (TG + CH - 1) / CH;
    const int kvh = int(tg_pos.x);
    const int row = int(tg_pos.y);

    threadgroup float buf[MAX_G][32];
    threadgroup float red[MAX_G];

    const device bfloat* kp = keys + size_t(kvh) * p.stride_h;
    const device bfloat* vp = values + size_t(kvh) * p.stride_h;
    const size_t qb = (size_t(row) * p.HQ + kvh * FC_G) * FC_HD;

    const int S_total = p.S_kv + p.ext_m;
    const int last = FC_SEQ ? (S_total - p.M + row) : (S_total - 1);
    const int first_valid = p.WINDOW > 0 ? max(0, last - p.WINDOW + 1) : 0;
    const int n_valid = max(0, last - first_valid + 1);
    const int place = first_valid + int(t);
    const bool seen = int(t) < n_valid;

    float sc[MAX_G], m[MAX_G], z[MAX_G];
    #pragma clang loop unroll(full)
    for (int g = 0; g < FC_G; g++) sc[g] = -INFINITY;
    if (seen) {
        const device bfloat* ks;
        if (place < p.S_kv) {
            const int phys = (p.ring_idx == 0) ? place : ((place + p.ring_idx) % p.S_kv);
            ks = kp + size_t(phys) * p.stride_s;
        } else {
            ks = ko_ext + (size_t(place - p.S_kv) * p.HKV + kvh) * FC_HD;
        }
        #pragma clang loop unroll(full)
        for (int g = 0; g < FC_G; g++) sc[g] = 0.0f;
        for (int d = 0; d < FC_HD; d++) {
            const float kd = float(ks[d]);
            #pragma clang loop unroll(full)
            for (int g = 0; g < FC_G; g++) sc[g] += float(q[qb + g * FC_HD + d]) * kd;
        }
        const int dist = last - place;
        #pragma clang loop unroll(full)
        for (int g = 0; g < FC_G; g++) {
            sc[g] = sc[g] * p.scale
                  + (dist < p.R ? float(bias[(size_t(row) * p.HQ + kvh * FC_G + g) * p.R + dist]) : 0.0f);
        }
    }
    #pragma clang loop unroll(full)
    for (int g = 0; g < FC_G; g++) {
        const float v = simd_max(sc[g]);
        if (lane == 0) buf[g][sg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        #pragma clang loop unroll(full)
        for (int g = 0; g < FC_G; g++) {
            const float v = simd_max(int(lane) < NS ? buf[g][lane] : -INFINITY);
            if (lane == 0) red[g] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (int g = 0; g < FC_G; g++) {
        m[g] = max(red[g], float(sinks[size_t(row) * p.HQ + kvh * FC_G + g]));
        const float e = seen ? metal::precise::exp(sc[g] - m[g]) : 0.0f;
        p_buf[g * TG + int(t)] = e;
        z[g] = simd_sum(e);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (int g = 0; g < FC_G; g++) {
        if (lane == 0) buf[g][sg] = z[g];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        #pragma clang loop unroll(full)
        for (int g = 0; g < FC_G; g++) {
            const float v = simd_sum(int(lane) < NS ? buf[g][lane] : 0.0f);
            if (lane == 0) red[g] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (int g = 0; g < FC_G; g++) {
        z[g] = red[g] + metal::precise::exp(float(sinks[size_t(row) * p.HQ + kvh * FC_G + g]) - m[g]);
    }

    const int d = int(t) % FC_HD;
    const int ch = int(t) / FC_HD;
    float o[MAX_G];
    #pragma clang loop unroll(full)
    for (int g = 0; g < FC_G; g++) o[g] = 0.0f;
    const int i_end = min(n_valid, (ch + 1) * CK);
    for (int i = ch * CK; i < i_end; i++) {
        const int p_idx = first_valid + i;
        float vd;
        if (p_idx < p.S_kv) {
            const int phys = (p.ring_idx == 0) ? p_idx : ((p_idx + p.ring_idx) % p.S_kv);
            vd = float(vp[size_t(phys) * p.stride_s + d]);
        } else {
            vd = float(vo_ext[(size_t(p_idx - p.S_kv) * p.HKV + kvh) * FC_HD + d]);
        }
        #pragma clang loop unroll(full)
        for (int g = 0; g < FC_G; g++) o[g] += p_buf[g * TG + i] * vd;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (int g = 0; g < FC_G; g++) p_buf[g * TG + int(t)] = o[g];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ch == 0) {
        #pragma clang loop unroll(full)
        for (int g = 0; g < FC_G; g++) {
            float v = o[g];
            for (int c = 1; c < CH; c++) v += p_buf[g * TG + c * FC_HD + d];
            v = round_bf16(v / z[g]);
            const size_t at = size_t(kvh * FC_G + g) * FC_HD + d;
            out[size_t(row) * p.HQ * FC_HD + at] =
                bfloat(v * sigmoid_(float(gate[size_t(row) * p.GW + p.GOFF + at])));
        }
    }
}

struct AttendBatchedPrm {
    int M, HQ, HKV, R, GW, GOFF, WINDOW;
    int S_kv, ring_idx, stride_b, stride_h, stride_s;
    float scale;
};

kernel void inkling_attend_batched(
    const device bfloat* q      [[buffer(0)]],
    const device bfloat* keys   [[buffer(1)]],
    const device bfloat* values [[buffer(2)]],
    const device bfloat* bias   [[buffer(3)]],
    const device bfloat* sinks  [[buffer(4)]],
    const device bfloat* gate   [[buffer(5)]],
    const device int* counts    [[buffer(6)]],
    device bfloat* out          [[buffer(7)]],
    constant AttendBatchedPrm& p[[buffer(8)]],
    threadgroup float* p_buf    [[threadgroup(0)]],
    uint2 tg_pos                [[threadgroup_position_in_grid]],
    uint t                      [[thread_index_in_threadgroup]],
    uint lane                   [[thread_index_in_simdgroup]],
    uint sg                     [[simdgroup_index_in_threadgroup]])
{
    const int TG = min(1024, ((FC_HD * 8 + 31) / 32) * 32);
    const int NS = TG / 32;
    const int CH = TG / FC_HD;
    const int CK = (TG + CH - 1) / CH;
    const int kvh = int(tg_pos.x);
    const int row = int(tg_pos.y);

    threadgroup float buf[MAX_G][32];
    threadgroup float red[MAX_G];

    const device bfloat* kp = keys + size_t(row) * p.stride_b + size_t(kvh) * p.stride_h;
    const device bfloat* vp = values + size_t(row) * p.stride_b + size_t(kvh) * p.stride_h;
    const size_t qb = (size_t(row) * p.HQ + kvh * FC_G) * FC_HD;

    const int count = counts[row];
    const int last = p.S_kv - 1;
    const int first_valid = max(p.S_kv - count, p.WINDOW > 0 ? max(0, last - p.WINDOW + 1) : 0);
    const int n_valid = max(0, last - first_valid + 1);
    const int place = first_valid + int(t);
    const bool seen = int(t) < n_valid;

    float sc[MAX_G], m[MAX_G], z[MAX_G];
    #pragma clang loop unroll(full)
    for (int g = 0; g < FC_G; g++) sc[g] = -INFINITY;
    if (seen) {
        const int phys = (p.ring_idx == 0) ? place : ((place + p.ring_idx) % p.S_kv);
        const device bfloat* ks = kp + size_t(phys) * p.stride_s;
        #pragma clang loop unroll(full)
        for (int g = 0; g < FC_G; g++) sc[g] = 0.0f;
        for (int d = 0; d < FC_HD; d++) {
            const float kd = float(ks[d]);
            #pragma clang loop unroll(full)
            for (int g = 0; g < FC_G; g++) sc[g] += float(q[qb + g * FC_HD + d]) * kd;
        }
        const int dist = last - place;
        #pragma clang loop unroll(full)
        for (int g = 0; g < FC_G; g++) {
            sc[g] = sc[g] * p.scale
                  + (dist < p.R ? float(bias[(size_t(row) * p.HQ + kvh * FC_G + g) * p.R + dist]) : 0.0f);
        }
    }
    #pragma clang loop unroll(full)
    for (int g = 0; g < FC_G; g++) {
        const float v = simd_max(sc[g]);
        if (lane == 0) buf[g][sg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        #pragma clang loop unroll(full)
        for (int g = 0; g < FC_G; g++) {
            const float v = simd_max(int(lane) < NS ? buf[g][lane] : -INFINITY);
            if (lane == 0) red[g] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (int g = 0; g < FC_G; g++) {
        m[g] = max(red[g], float(sinks[size_t(row) * p.HQ + kvh * FC_G + g]));
        const float e = seen ? metal::precise::exp(sc[g] - m[g]) : 0.0f;
        p_buf[g * TG + int(t)] = e;
        z[g] = simd_sum(e);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (int g = 0; g < FC_G; g++) {
        if (lane == 0) buf[g][sg] = z[g];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        #pragma clang loop unroll(full)
        for (int g = 0; g < FC_G; g++) {
            const float v = simd_sum(int(lane) < NS ? buf[g][lane] : 0.0f);
            if (lane == 0) red[g] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (int g = 0; g < FC_G; g++) {
        z[g] = red[g] + metal::precise::exp(float(sinks[size_t(row) * p.HQ + kvh * FC_G + g]) - m[g]);
    }

    const int d = int(t) % FC_HD;
    const int ch = int(t) / FC_HD;
    float o[MAX_G];
    #pragma clang loop unroll(full)
    for (int g = 0; g < FC_G; g++) o[g] = 0.0f;
    const int i_end = min(n_valid, (ch + 1) * CK);
    for (int i = ch * CK; i < i_end; i++) {
        const int p_idx = first_valid + i;
        const int phys = (p.ring_idx == 0) ? p_idx : ((p_idx + p.ring_idx) % p.S_kv);
        const float vd = float(vp[size_t(phys) * p.stride_s + d]);
        #pragma clang loop unroll(full)
        for (int g = 0; g < FC_G; g++) o[g] += p_buf[g * TG + i] * vd;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (int g = 0; g < FC_G; g++) p_buf[g * TG + int(t)] = o[g];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ch == 0) {
        #pragma clang loop unroll(full)
        for (int g = 0; g < FC_G; g++) {
            float v = o[g];
            for (int c = 1; c < CH; c++) v += p_buf[g * TG + c * FC_HD + d];
            v = round_bf16(v / z[g]);
            const size_t at = size_t(kvh * FC_G + g) * FC_HD + d;
            out[size_t(row) * p.HQ * FC_HD + at] =
                bfloat(v * sigmoid_(float(gate[size_t(row) * p.GW + p.GOFF + at])));
        }
    }
}

struct DecodePartPrm {
    int M, HQ, HKV, REL, BS, nblocks, WINDOW, SEQ;
    int S_kv, ext_m, ring_idx, stride_h, stride_s;
    float scale;
};

kernel void inkling_decode_partials(
    const device bfloat* q      [[buffer(0)]],
    const device bfloat* keys   [[buffer(1)]],
    const device bfloat* values [[buffer(2)]],
    const device bfloat* ko_ext [[buffer(3)]],
    const device bfloat* vo_ext [[buffer(4)]],
    const device bfloat* rel    [[buffer(5)]],
    device float* part_max      [[buffer(6)]],
    device float* part_sum      [[buffer(7)]],
    device float* part_acc      [[buffer(8)]],
    constant DecodePartPrm& p   [[buffer(9)]],
    threadgroup float* ta       [[threadgroup(0)]],
    uint3 tg_pos                [[threadgroup_position_in_grid]],
    uint tid                    [[thread_index_in_threadgroup]],
    uint lane                   [[thread_index_in_simdgroup]],
    uint sg                     [[simdgroup_index_in_threadgroup]])
{
    constexpr int SG = 8;
    const int E = FC_HD / 32;
    const int block = int(tg_pos.x);
    const int g = int(tg_pos.y);
    const int batch = int(tg_pos.z);

    const int S_total = p.S_kv + p.ext_m;
    const int last = p.SEQ ? (S_total - p.M + batch) : (S_total - 1);
    const int ks = p.WINDOW > 0 ? max(0, last - p.WINDOW + 1) : 0;
    const int start = ks + block * p.BS;
    const int end = min(start + p.BS, last + 1);

    float qv[MAX_G][4];
    const size_t q_base = (size_t(batch) * p.HQ + g * FC_G) * FC_HD + int(lane) * E;
    #pragma clang loop unroll(full)
    for (int h = 0; h < FC_G; ++h)
        #pragma clang loop unroll(full)
        for (int e = 0; e < E; ++e)
            qv[h][e] = float(q[q_base + size_t(h) * FC_HD + e]) * p.scale;

    float m[MAX_G], l[MAX_G], acc[MAX_G][4];
    #pragma clang loop unroll(full)
    for (int h = 0; h < FC_G; ++h) {
        m[h] = -INFINITY;
        l[h] = 0.0f;
        #pragma clang loop unroll(full)
        for (int e = 0; e < E; ++e) acc[h][e] = 0.0f;
    }

    const device bfloat* k_head = keys + size_t(g) * p.stride_h;
    const device bfloat* v_head = values + size_t(g) * p.stride_h;
    const size_t rel_base = (size_t(batch) * p.HQ + g * FC_G) * p.REL;

    for (int j = start + int(sg); j < end; j += SG) {
        const device bfloat* kp;
        const device bfloat* vp;
        if (j < p.S_kv) {
            const int phys = (p.ring_idx == 0) ? j : ((j + p.ring_idx) % p.S_kv);
            kp = k_head + size_t(phys) * p.stride_s + int(lane) * E;
            vp = v_head + size_t(phys) * p.stride_s + int(lane) * E;
        } else {
            const size_t eo = (size_t(j - p.S_kv) * p.HKV + g) * FC_HD + int(lane) * E;
            kp = ko_ext + eo;
            vp = vo_ext + eo;
        }
        float kv[4], vv[4];
        #pragma clang loop unroll(full)
        for (int e = 0; e < E; ++e) {
            kv[e] = float(kp[e]);
            vv[e] = float(vp[e]);
        }
        const int dist = last - j;
        #pragma clang loop unroll(full)
        for (int h = 0; h < FC_G; ++h) {
            float s = 0.0f;
            #pragma clang loop unroll(full)
            for (int e = 0; e < E; ++e) s += qv[h][e] * kv[e];
            s = simd_sum(s);
            if (dist < p.REL) s += float(rel[rel_base + size_t(h) * p.REL + dist]);
            const float mn = max(m[h], s);
            const float f = metal::exp(m[h] - mn);
            const float pv = metal::exp(s - mn);
            l[h] = l[h] * f + pv;
            #pragma clang loop unroll(full)
            for (int e = 0; e < E; ++e) acc[h][e] = acc[h][e] * f + pv * vv[e];
            m[h] = mn;
        }
    }

    threadgroup float tm[8][MAX_G];
    threadgroup float tl[8][MAX_G];
    #pragma clang loop unroll(full)
    for (int h = 0; h < FC_G; ++h) {
        if (lane == 0) {
            tm[sg][h] = m[h];
            tl[sg][h] = l[h];
        }
        #pragma clang loop unroll(full)
        for (int e = 0; e < E; ++e) {
            ta[(int(sg) * FC_G + h) * FC_HD + int(lane) * E + e] = acc[h][e];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = tid; t < uint(FC_G * FC_HD); t += SG * 32) {
        const int h = int(t) / FC_HD;
        const int d = int(t) % FC_HD;
        float Mv = -INFINITY;
        #pragma clang loop unroll(full)
        for (int s = 0; s < SG; ++s) Mv = max(Mv, tm[s][h]);
        float Lv = 0.0f, Av = 0.0f;
        if (Mv > -INFINITY) {
            #pragma clang loop unroll(full)
            for (int s = 0; s < SG; ++s) {
                const float w = metal::exp(tm[s][h] - Mv);
                Lv += tl[s][h] * w;
                Av += ta[(s * FC_G + h) * FC_HD + d] * w;
            }
        }
        const size_t row_out = (size_t(batch) * p.nblocks + block) * p.HQ + g * FC_G + h;
        part_acc[row_out * FC_HD + d] = Av;
        if (d == 0) {
            part_max[row_out] = Mv;
            part_sum[row_out] = Lv;
        }
    }
}

struct DecodePartBatchedPrm {
    int M, HQ, HKV, REL, BS, nblocks, WINDOW;
    int S_kv, ring_idx, stride_b, stride_h, stride_s;
    float scale;
};

kernel void inkling_decode_partials_batched(
    const device bfloat* q      [[buffer(0)]],
    const device bfloat* keys   [[buffer(1)]],
    const device bfloat* values [[buffer(2)]],
    const device bfloat* rel    [[buffer(3)]],
    const device int* counts    [[buffer(4)]],
    device float* part_max      [[buffer(5)]],
    device float* part_sum      [[buffer(6)]],
    device float* part_acc      [[buffer(7)]],
    constant DecodePartBatchedPrm& p[[buffer(8)]],
    threadgroup float* ta       [[threadgroup(0)]],
    uint3 tg_pos                [[threadgroup_position_in_grid]],
    uint tid                    [[thread_index_in_threadgroup]],
    uint lane                   [[thread_index_in_simdgroup]],
    uint sg                     [[simdgroup_index_in_threadgroup]])
{
    constexpr int SG = 8;
    const int E = FC_HD / 32;
    const int block = int(tg_pos.x);
    const int g = int(tg_pos.y);
    const int batch = int(tg_pos.z);

    const int count = counts[batch];
    const int last = p.S_kv - 1;
    const int ks_raw = max(0, p.S_kv - count);
    const int ks = p.WINDOW > 0 ? max(ks_raw, max(0, last - p.WINDOW + 1)) : ks_raw;
    const int start = ks + block * p.BS;
    const int end = min(start + p.BS, last + 1);

    float qv[MAX_G][4];
    const size_t q_base = (size_t(batch) * p.HQ + g * FC_G) * FC_HD + int(lane) * E;
    #pragma clang loop unroll(full)
    for (int h = 0; h < FC_G; ++h)
        #pragma clang loop unroll(full)
        for (int e = 0; e < E; ++e)
            qv[h][e] = float(q[q_base + size_t(h) * FC_HD + e]) * p.scale;

    float m[MAX_G], l[MAX_G], acc[MAX_G][4];
    #pragma clang loop unroll(full)
    for (int h = 0; h < FC_G; ++h) {
        m[h] = -INFINITY;
        l[h] = 0.0f;
        #pragma clang loop unroll(full)
        for (int e = 0; e < E; ++e) acc[h][e] = 0.0f;
    }

    const device bfloat* k_head = keys + size_t(batch) * p.stride_b + size_t(g) * p.stride_h;
    const device bfloat* v_head = values + size_t(batch) * p.stride_b + size_t(g) * p.stride_h;
    const size_t rel_base = (size_t(batch) * p.HQ + g * FC_G) * p.REL;

    for (int j = start + int(sg); j < end; j += SG) {
        const int phys = (p.ring_idx == 0) ? j : ((j + p.ring_idx) % p.S_kv);
        const device bfloat* kp = k_head + size_t(phys) * p.stride_s + int(lane) * E;
        const device bfloat* vp = v_head + size_t(phys) * p.stride_s + int(lane) * E;
        float kv[4], vv[4];
        #pragma clang loop unroll(full)
        for (int e = 0; e < E; ++e) {
            kv[e] = float(kp[e]);
            vv[e] = float(vp[e]);
        }
        const int dist = last - j;
        #pragma clang loop unroll(full)
        for (int h = 0; h < FC_G; ++h) {
            float s = 0.0f;
            #pragma clang loop unroll(full)
            for (int e = 0; e < E; ++e) s += qv[h][e] * kv[e];
            s = simd_sum(s);
            if (dist < p.REL) s += float(rel[rel_base + size_t(h) * p.REL + dist]);
            const float mn = max(m[h], s);
            const float f = metal::exp(m[h] - mn);
            const float pv = metal::exp(s - mn);
            l[h] = l[h] * f + pv;
            #pragma clang loop unroll(full)
            for (int e = 0; e < E; ++e) acc[h][e] = acc[h][e] * f + pv * vv[e];
            m[h] = mn;
        }
    }

    threadgroup float tm[8][MAX_G];
    threadgroup float tl[8][MAX_G];
    #pragma clang loop unroll(full)
    for (int h = 0; h < FC_G; ++h) {
        if (lane == 0) {
            tm[sg][h] = m[h];
            tl[sg][h] = l[h];
        }
        #pragma clang loop unroll(full)
        for (int e = 0; e < E; ++e) {
            ta[(int(sg) * FC_G + h) * FC_HD + int(lane) * E + e] = acc[h][e];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = tid; t < uint(FC_G * FC_HD); t += SG * 32) {
        const int h = int(t) / FC_HD;
        const int d = int(t) % FC_HD;
        float Mv = -INFINITY;
        #pragma clang loop unroll(full)
        for (int s = 0; s < SG; ++s) Mv = max(Mv, tm[s][h]);
        float Lv = 0.0f, Av = 0.0f;
        if (Mv > -INFINITY) {
            #pragma clang loop unroll(full)
            for (int s = 0; s < SG; ++s) {
                const float w = metal::exp(tm[s][h] - Mv);
                Lv += tl[s][h] * w;
                Av += ta[(s * FC_G + h) * FC_HD + d] * w;
            }
        }
        const size_t row_out = (size_t(batch) * p.nblocks + block) * p.HQ + g * FC_G + h;
        part_acc[row_out * FC_HD + d] = Av;
        if (d == 0) {
            part_max[row_out] = Mv;
            part_sum[row_out] = Lv;
        }
    }
}

struct DecodeCombPrm {
    int HQ, HD, nblocks, GW, GOFF;
};

kernel void inkling_decode_combine(
    const device float* part_max [[buffer(0)]],
    const device float* part_sum [[buffer(1)]],
    const device float* part_acc [[buffer(2)]],
    const device bfloat* sinks   [[buffer(3)]],
    const device bfloat* gate    [[buffer(4)]],
    device bfloat* out           [[buffer(5)]],
    constant DecodeCombPrm& p    [[buffer(6)]],
    uint2 tg_pos                 [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]])
{
    const int h = int(tg_pos.x);
    const int batch = int(tg_pos.y);
    const int d = int(tid);
    const float sk = float(sinks[batch * p.HQ + h]);
    float Mv = sk;
    const size_t base = size_t(batch) * p.nblocks * p.HQ + h;
    for (int b = 0; b < p.nblocks; ++b) {
        Mv = max(Mv, part_max[base + size_t(b) * p.HQ]);
    }
    float Lv = metal::exp(sk - Mv), Av = 0.0f;
    for (int b = 0; b < p.nblocks; ++b) {
        const size_t idx = base + size_t(b) * p.HQ;
        const float pm = part_max[idx];
        if (pm > -INFINITY) {
            const float w = metal::exp(pm - Mv);
            Lv += part_sum[idx] * w;
            Av += part_acc[idx * p.HD + d] * w;
        }
    }
    float v = round_bf16(Av / Lv);
    const float gv = float(gate[size_t(batch) * p.GW + p.GOFF + h * p.HD + d]);
    v = v * sigmoid_(gv);
    out[(size_t(batch) * p.HQ + h) * p.HD + d] = bfloat(v);
}

struct RoutePrm {
    float route_scale;
};

kernel void inkling_route(
    const device bfloat* x     [[buffer(0)]],
    const device float* w      [[buffer(1)]],
    const device float* bias   [[buffer(2)]],
    device uint* inds          [[buffer(3)]],
    device float* weights      [[buffer(4)]],
    constant RoutePrm& p       [[buffer(5)]],
    uint tg_x                  [[threadgroup_position_in_grid]],
    uint tid                   [[thread_position_in_threadgroup]])
{
    constexpr int PARTS = MODEL_ROUTE_PARTS;
    constexpr int CHUNK = (MODEL_D + PARTS - 1) / PARTS;
    const int row = int(tg_x);
    const int e = int(tid) / PARTS;
    const int part = int(tid) % PARTS;

    threadgroup float partial[MODEL_E * PARTS];
    threadgroup float score[MODEL_E];
    threadgroup float chosen[MODEL_TOP_K];

    float dot = 0.0f;
    const int d_end = min(MODEL_D, (part + 1) * CHUNK);
    for (int d = part * CHUNK; d < d_end; d++) {
        dot += float(x[size_t(row) * MODEL_D + d]) * w[size_t(e) * MODEL_D + d];
    }
    partial[tid] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (part == 0) {
        #pragma clang loop unroll(full)
        for (int i = 1; i < PARTS; i++) dot += partial[e * PARTS + i];
        score[e] = sigmoid_(dot);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (part == 0) {
        const float mine = score[e] + bias[e];
        int rank = 0;
        for (int i = 0; i < MODEL_E; i++) {
            const float other = score[i] + bias[i];
            rank += (other > mine || (other == mine && i < e)) ? 1 : 0;
        }
        if (rank < MODEL_TOP_K) {
            inds[size_t(row) * MODEL_TOP_K + rank] = uint(e);
            chosen[rank] = score[e];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(tid) < MODEL_TOP_K) {
        float sum = 0.0f;
        #pragma clang loop unroll(full)
        for (int k = 0; k < MODEL_TOP_K; k++) sum += chosen[k];
        weights[size_t(row) * MODEL_TOP_K + int(tid)] = chosen[tid] / (sum + 1e-20f) * p.route_scale;
    }
}

struct RouteBatchedPrm {
    int M;
    float route_scale;
};

kernel void inkling_route_scores(
    const device bfloat* x      [[buffer(0)]],
    const device float* w       [[buffer(1)]],
    device float* scores        [[buffer(2)]],
    constant RouteBatchedPrm& p [[buffer(3)]],
    uint tg_x                   [[threadgroup_position_in_grid]],
    uint tid                    [[thread_position_in_threadgroup]],
    uint lane                   [[thread_index_in_simdgroup]],
    uint sg                     [[simdgroup_index_in_threadgroup]])
{
    constexpr int TG = 256;
    constexpr int NS = TG / 32;
    constexpr int DPT = (MODEL_D + TG - 1) / TG;
    const int e = int(tg_x);
    threadgroup float sbuf[4][NS];
    threadgroup float red[4];

    float wreg[DPT];
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; ++c) {
        const int d = int(tid) + c * TG;
        wreg[c] = (d < MODEL_D) ? w[size_t(e) * MODEL_D + d] : 0.0f;
    }

    for (int m0 = 0; m0 < p.M; m0 += 4) {
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const int mn = min(4, p.M - m0);
        #pragma clang loop unroll(full)
        for (int c = 0; c < DPT; ++c) {
            const int d = int(tid) + c * TG;
            if (d < MODEL_D) {
                const float wv = wreg[c];
                #pragma clang loop unroll(full)
                for (int mi = 0; mi < 4; ++mi) {
                    if (mi < mn) {
                        acc[mi] += float(x[size_t(m0 + mi) * MODEL_D + d]) * wv;
                    }
                }
            }
        }
        #pragma clang loop unroll(full)
        for (int mi = 0; mi < 4; ++mi) {
            if (mi < mn) {
                const float ss = simd_sum(acc[mi]);
                if (lane == 0) sbuf[mi][sg] = ss;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0 && int(lane) < mn) {
            float tot = 0.0f;
            #pragma clang loop unroll(full)
            for (int s = 0; s < NS; ++s) tot += sbuf[lane][s];
            red[lane] = tot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (int(tid) < mn) {
            scores[size_t(m0 + int(tid)) * MODEL_E + e] = sigmoid_(red[tid]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void inkling_route_topk(
    const device float* scores  [[buffer(0)]],
    const device float* bias    [[buffer(1)]],
    device uint* inds           [[buffer(2)]],
    device float* weights       [[buffer(3)]],
    constant RouteBatchedPrm& p [[buffer(4)]],
    uint tg_x                   [[threadgroup_position_in_grid]],
    uint tid                    [[thread_position_in_threadgroup]])
{
    const int row = int(tg_x);
    threadgroup float sc[MODEL_E];
    threadgroup float bs[MODEL_E];
    threadgroup float chosen[MODEL_TOP_K];
    if (int(tid) < MODEL_E) {
        const float s = scores[size_t(row) * MODEL_E + tid];
        sc[tid] = s;
        bs[tid] = s + bias[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(tid) < MODEL_E) {
        const float mine = bs[tid];
        int rank = 0;
        for (int i = 0; i < MODEL_E; ++i) {
            const float other = bs[i];
            rank += (other > mine || (other == mine && i < int(tid))) ? 1 : 0;
        }
        if (rank < MODEL_TOP_K) {
            inds[size_t(row) * MODEL_TOP_K + rank] = uint(tid);
            chosen[rank] = sc[tid];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(tid) < MODEL_TOP_K) {
        float sum = 0.0f;
        #pragma clang loop unroll(full)
        for (int k = 0; k < MODEL_TOP_K; ++k) sum += chosen[k];
        weights[size_t(row) * MODEL_TOP_K + int(tid)] = chosen[tid] / (sum + 1e-20f) * p.route_scale;
    }
}

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

constant half2 E2M1_PAIR_H[256] = {
  half2(0.0h,0.0h), half2(0.5h,0.0h), half2(1.0h,0.0h), half2(1.5h,0.0h), half2(2.0h,0.0h), half2(3.0h,0.0h), half2(4.0h,0.0h), half2(6.0h,0.0h),
  half2(-0.0h,0.0h), half2(-0.5h,0.0h), half2(-1.0h,0.0h), half2(-1.5h,0.0h), half2(-2.0h,0.0h), half2(-3.0h,0.0h), half2(-4.0h,0.0h), half2(-6.0h,0.0h),
  half2(0.0h,0.5h), half2(0.5h,0.5h), half2(1.0h,0.5h), half2(1.5h,0.5h), half2(2.0h,0.5h), half2(3.0h,0.5h), half2(4.0h,0.5h), half2(6.0h,0.5h),
  half2(-0.0h,0.5h), half2(-0.5h,0.5h), half2(-1.0h,0.5h), half2(-1.5h,0.5h), half2(-2.0h,0.5h), half2(-3.0h,0.5h), half2(-4.0h,0.5h), half2(-6.0h,0.5h),
  half2(0.0h,1.0h), half2(0.5h,1.0h), half2(1.0h,1.0h), half2(1.5h,1.0h), half2(2.0h,1.0h), half2(3.0h,1.0h), half2(4.0h,1.0h), half2(6.0h,1.0h),
  half2(-0.0h,1.0h), half2(-0.5h,1.0h), half2(-1.0h,1.0h), half2(-1.5h,1.0h), half2(-2.0h,1.0h), half2(-3.0h,1.0h), half2(-4.0h,1.0h), half2(-6.0h,1.0h),
  half2(0.0h,1.5h), half2(0.5h,1.5h), half2(1.0h,1.5h), half2(1.5h,1.5h), half2(2.0h,1.5h), half2(3.0h,1.5h), half2(4.0h,1.5h), half2(6.0h,1.5h),
  half2(-0.0h,1.5h), half2(-0.5h,1.5h), half2(-1.0h,1.5h), half2(-1.5h,1.5h), half2(-2.0h,1.5h), half2(-3.0h,1.5h), half2(-4.0h,1.5h), half2(-6.0h,1.5h),
  half2(0.0h,2.0h), half2(0.5h,2.0h), half2(1.0h,2.0h), half2(1.5h,2.0h), half2(2.0h,2.0h), half2(3.0h,2.0h), half2(4.0h,2.0h), half2(6.0h,2.0h),
  half2(-0.0h,2.0h), half2(-0.5h,2.0h), half2(-1.0h,2.0h), half2(-1.5h,2.0h), half2(-2.0h,2.0h), half2(-3.0h,2.0h), half2(-4.0h,2.0h), half2(-6.0h,2.0h),
  half2(0.0h,3.0h), half2(0.5h,3.0h), half2(1.0h,3.0h), half2(1.5h,3.0h), half2(2.0h,3.0h), half2(3.0h,3.0h), half2(4.0h,3.0h), half2(6.0h,3.0h),
  half2(-0.0h,3.0h), half2(-0.5h,3.0h), half2(-1.0h,3.0h), half2(-1.5h,3.0h), half2(-2.0h,3.0h), half2(-3.0h,3.0h), half2(-4.0h,3.0h), half2(-6.0h,3.0h),
  half2(0.0h,4.0h), half2(0.5h,4.0h), half2(1.0h,4.0h), half2(1.5h,4.0h), half2(2.0h,4.0h), half2(3.0h,4.0h), half2(4.0h,4.0h), half2(6.0h,4.0h),
  half2(-0.0h,4.0h), half2(-0.5h,4.0h), half2(-1.0h,4.0h), half2(-1.5h,4.0h), half2(-2.0h,4.0h), half2(-3.0h,4.0h), half2(-4.0h,4.0h), half2(-6.0h,4.0h),
  half2(0.0h,6.0h), half2(0.5h,6.0h), half2(1.0h,6.0h), half2(1.5h,6.0h), half2(2.0h,6.0h), half2(3.0h,6.0h), half2(4.0h,6.0h), half2(6.0h,6.0h),
  half2(-0.0h,6.0h), half2(-0.5h,6.0h), half2(-1.0h,6.0h), half2(-1.5h,6.0h), half2(-2.0h,6.0h), half2(-3.0h,6.0h), half2(-4.0h,6.0h), half2(-6.0h,6.0h),
  half2(0.0h,-0.0h), half2(0.5h,-0.0h), half2(1.0h,-0.0h), half2(1.5h,-0.0h), half2(2.0h,-0.0h), half2(3.0h,-0.0h), half2(4.0h,-0.0h), half2(6.0h,-0.0h),
  half2(-0.0h,-0.0h), half2(-0.5h,-0.0h), half2(-1.0h,-0.0h), half2(-1.5h,-0.0h), half2(-2.0h,-0.0h), half2(-3.0h,-0.0h), half2(-4.0h,-0.0h), half2(-6.0h,-0.0h),
  half2(0.0h,-0.5h), half2(0.5h,-0.5h), half2(1.0h,-0.5h), half2(1.5h,-0.5h), half2(2.0h,-0.5h), half2(3.0h,-0.5h), half2(4.0h,-0.5h), half2(6.0h,-0.5h),
  half2(-0.0h,-0.5h), half2(-0.5h,-0.5h), half2(-1.0h,-0.5h), half2(-1.5h,-0.5h), half2(-2.0h,-0.5h), half2(-3.0h,-0.5h), half2(-4.0h,-0.5h), half2(-6.0h,-0.5h),
  half2(0.0h,-1.0h), half2(0.5h,-1.0h), half2(1.0h,-1.0h), half2(1.5h,-1.0h), half2(2.0h,-1.0h), half2(3.0h,-1.0h), half2(4.0h,-1.0h), half2(6.0h,-1.0h),
  half2(-0.0h,-1.0h), half2(-0.5h,-1.0h), half2(-1.0h,-1.0h), half2(-1.5h,-1.0h), half2(-2.0h,-1.0h), half2(-3.0h,-1.0h), half2(-4.0h,-1.0h), half2(-6.0h,-1.0h),
  half2(0.0h,-1.5h), half2(0.5h,-1.5h), half2(1.0h,-1.5h), half2(1.5h,-1.5h), half2(2.0h,-1.5h), half2(3.0h,-1.5h), half2(4.0h,-1.5h), half2(6.0h,-1.5h),
  half2(-0.0h,-1.5h), half2(-0.5h,-1.5h), half2(-1.0h,-1.5h), half2(-1.5h,-1.5h), half2(-2.0h,-1.5h), half2(-3.0h,-1.5h), half2(-4.0h,-1.5h), half2(-6.0h,-1.5h),
  half2(0.0h,-2.0h), half2(0.5h,-2.0h), half2(1.0h,-2.0h), half2(1.5h,-2.0h), half2(2.0h,-2.0h), half2(3.0h,-2.0h), half2(4.0h,-2.0h), half2(6.0h,-2.0h),
  half2(-0.0h,-2.0h), half2(-0.5h,-2.0h), half2(-1.0h,-2.0h), half2(-1.5h,-2.0h), half2(-2.0h,-2.0h), half2(-3.0h,-2.0h), half2(-4.0h,-2.0h), half2(-6.0h,-2.0h),
  half2(0.0h,-3.0h), half2(0.5h,-3.0h), half2(1.0h,-3.0h), half2(1.5h,-3.0h), half2(2.0h,-3.0h), half2(3.0h,-3.0h), half2(4.0h,-3.0h), half2(6.0h,-3.0h),
  half2(-0.0h,-3.0h), half2(-0.5h,-3.0h), half2(-1.0h,-3.0h), half2(-1.5h,-3.0h), half2(-2.0h,-3.0h), half2(-3.0h,-3.0h), half2(-4.0h,-3.0h), half2(-6.0h,-3.0h),
  half2(0.0h,-4.0h), half2(0.5h,-4.0h), half2(1.0h,-4.0h), half2(1.5h,-4.0h), half2(2.0h,-4.0h), half2(3.0h,-4.0h), half2(4.0h,-4.0h), half2(6.0h,-4.0h),
  half2(-0.0h,-4.0h), half2(-0.5h,-4.0h), half2(-1.0h,-4.0h), half2(-1.5h,-4.0h), half2(-2.0h,-4.0h), half2(-3.0h,-4.0h), half2(-4.0h,-4.0h), half2(-6.0h,-4.0h),
  half2(0.0h,-6.0h), half2(0.5h,-6.0h), half2(1.0h,-6.0h), half2(1.5h,-6.0h), half2(2.0h,-6.0h), half2(3.0h,-6.0h), half2(4.0h,-6.0h), half2(6.0h,-6.0h),
  half2(-0.0h,-6.0h), half2(-0.5h,-6.0h), half2(-1.0h,-6.0h), half2(-1.5h,-6.0h), half2(-2.0h,-6.0h), half2(-3.0h,-6.0h), half2(-4.0h,-6.0h), half2(-6.0h,-6.0h)
};

inline void unpack8_nvfp4_h(uint w, thread half4& lo, thread half4& hi) {
    half2 p0 = E2M1_PAIR_H[w & 0xFFu];
    half2 p1 = E2M1_PAIR_H[(w >> 8) & 0xFFu];
    half2 p2 = E2M1_PAIR_H[(w >> 16) & 0xFFu];
    half2 p3 = E2M1_PAIR_H[w >> 24];
    lo = half4(p0, p1);
    hi = half4(p2, p3);
}

struct BucketPrm {
    int T, MAX_ACT;
};

kernel void inkling_bucket(
    const device uint* inds  [[buffer(0)]],
    device int* offsets      [[buffer(1)]],
    device uint* sorted_slots[[buffer(2)]],
    device int* active_list  [[buffer(3)]],
    constant BucketPrm& p    [[buffer(4)]],
    uint tid                 [[thread_position_in_threadgroup]],
    uint lane                [[thread_index_in_simdgroup]],
    uint sg                  [[simdgroup_index_in_threadgroup]])
{
    constexpr int TG = MODEL_BUCKET_TG;
    constexpr int NS = TG / 32;
    threadgroup atomic_uint curs[MODEL_E];
    if (int(tid) < MODEL_E) atomic_store_explicit(&curs[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = int(tid); i < p.T; i += TG) {
        uint e = inds[i];
        atomic_fetch_add_explicit(&curs[e], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint c = (int(tid) < MODEL_E) ? atomic_load_explicit(&curs[tid], memory_order_relaxed) : 0u;
    const uint sg_inc = simd_prefix_inclusive_sum(c);
    const uint sg_exc = sg_inc - c;
    const uint act = c > 0u ? 1u : 0u;
    const uint act_inc = simd_prefix_inclusive_sum(act);
    const uint act_exc = act_inc - act;

    threadgroup uint sg_tots[NS];
    threadgroup uint act_tots[NS];
    if (lane == 31) { sg_tots[sg] = sg_inc; act_tots[sg] = act_inc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint base = 0, abase = 0;
    for (uint s = 0; s < sg; ++s) { base += sg_tots[s]; abase += act_tots[s]; }
    const uint off = base + sg_exc;
    const uint aoff = abase + act_exc;

    if (int(tid) < MODEL_E) offsets[tid] = int(off);
    if (int(tid) == MODEL_E - 1) offsets[MODEL_E] = int(off + c);
    for (int i = int(tid); i < p.MAX_ACT; i += TG) active_list[i] = -1;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (int(tid) < MODEL_E && c > 0u) active_list[aoff] = int(tid);
    if (int(tid) < MODEL_E) atomic_store_explicit(&curs[tid], off, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = int(tid); i < p.T; i += TG) {
        uint e = inds[i];
        uint pos = atomic_fetch_add_explicit(&curs[e], 1u, memory_order_relaxed);
        sorted_slots[pos] = uint(i);
    }
}

struct BucketTilesPrm {
    int T, MT, NX13, NX2;
};

kernel void inkling_bucket_tiles(
    const device uint* inds    [[buffer(0)]],
    device int* offsets        [[buffer(1)]],
    device uint* sorted_slots  [[buffer(2)]],
    device int* tile_expert    [[buffer(3)]],
    device int* tile_t0        [[buffer(4)]],
    device uint* indirect_args [[buffer(5)]],
    constant BucketTilesPrm& p [[buffer(6)]],
    uint tid                   [[thread_position_in_threadgroup]],
    uint lane                  [[thread_index_in_simdgroup]],
    uint sg                    [[simdgroup_index_in_threadgroup]])
{
    constexpr int TG = MODEL_BUCKET_TG;
    constexpr int NS = TG / 32;
    threadgroup atomic_uint curs[MODEL_E];
    if (int(tid) < MODEL_E) atomic_store_explicit(&curs[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = int(tid); i < p.T; i += TG) {
        uint e = inds[i];
        atomic_fetch_add_explicit(&curs[e], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint c = (int(tid) < MODEL_E) ? atomic_load_explicit(&curs[tid], memory_order_relaxed) : 0u;
    const uint n_tiles = (c + uint(p.MT) - 1u) / uint(p.MT);
    const uint sg_inc = simd_prefix_inclusive_sum(c);
    const uint sg_exc = sg_inc - c;
    const uint t_inc = simd_prefix_inclusive_sum(n_tiles);
    const uint t_exc = t_inc - n_tiles;

    threadgroup uint sg_tots[NS];
    threadgroup uint t_tots[NS];
    if (lane == 31) { sg_tots[sg] = sg_inc; t_tots[sg] = t_inc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint base = 0u, tbase = 0u;
    for (uint s = 0u; s < sg; ++s) { base += sg_tots[s]; tbase += t_tots[s]; }
    const uint off = base + sg_exc;
    const uint toff = tbase + t_exc;

    if (int(tid) < MODEL_E) offsets[tid] = int(off);
    if (int(tid) == MODEL_E - 1) {
        offsets[MODEL_E] = int(off + c);
        const uint total_tiles = toff + n_tiles;
        indirect_args[0] = uint(p.NX13);
        indirect_args[1] = total_tiles;
        indirect_args[2] = 1u;
        indirect_args[4] = uint(p.NX2);
        indirect_args[5] = total_tiles;
        indirect_args[6] = 1u;
    }
    if (int(tid) < MODEL_E) {
        for (uint k = 0u; k < n_tiles; ++k) {
            tile_expert[toff + k] = int(tid);
            tile_t0[toff + k] = int(k * uint(p.MT));
        }
        atomic_store_explicit(&curs[tid], off, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = int(tid); i < p.T; i += TG) {
        uint e = inds[i];
        uint pos = atomic_fetch_add_explicit(&curs[e], 1u, memory_order_relaxed);
        sorted_slots[pos] = uint(i);
    }
}

struct MoePrm {
    float gate_cap, up_cap;
};

kernel void inkling_m1_w13_nvfp4(
    const device bfloat* x       [[buffer(0)]],
    const device uint* w         [[buffer(1)]],
    const device uchar* scales   [[buffer(2)]],
    const device float* gscale   [[buffer(3)]],
    const device uint* inds      [[buffer(4)]],
    device bfloat* out           [[buffer(5)]],
    constant MoePrm& p           [[buffer(6)]],
    uint3 tg_pos                 [[threadgroup_position_in_grid]],
    uint lane                    [[thread_index_in_simdgroup]],
    uint sg                      [[simdgroup_index_in_threadgroup]])
{
    const uint sub = lane >> 3;
    const uint slane = lane & 7u;
    const int r = (int(tg_pos.x) * FC_SG + int(sg)) * 4 + int(sub);
    const int slot = int(tg_pos.y);
    const int m = int(tg_pos.z);
    if (r >= MODEL_INTER) return;

    const uint e = inds[m * MODEL_TOP_K + slot];
    const float oscale = gscale[e] * (1.0f / 2688.0f);
    constexpr int PAIRS = MODEL_D / 32;
    constexpr int GROUPS = MODEL_D / 16;
    constexpr int W_STRIDE = MODEL_D / 8;
    const device bfloat* xm = x + size_t(m) * MODEL_D;
    const size_t w_base_g = (size_t(e) * (2 * MODEL_INTER) + size_t(2 * r)) * W_STRIDE;
    const size_t w_base_u = w_base_g + W_STRIDE;
    const size_t s_base_g = (size_t(e) * (2 * MODEL_INTER) + size_t(2 * r)) * GROUPS;
    const size_t s_base_u = s_base_g + GROUPS;

    float acc_g = 0.0f, acc_u = 0.0f;
    for (int pi = int(slane); pi < PAIRS; pi += 8) {
        const uint4 wg = *(const device uint4*)(w + w_base_g + pi * 4);
        const uchar2 sg2 = *(const device uchar2*)(scales + s_base_g + pi * 2);
        const uint4 wu = *(const device uint4*)(w + w_base_u + pi * 4);
        const uchar2 su2 = *(const device uchar2*)(scales + s_base_u + pi * 2);
        auto xp = (const device vec<bfloat, 4>*)(xm + pi * 32);
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
    const float gate = round_bf16(reduce8(acc_g) * oscale);
    const float up   = round_bf16(reduce8(acc_u) * oscale);
    if (slane == 0) {
        out[(size_t(m) * MODEL_TOP_K + slot) * MODEL_INTER + r] = bfloat(softcapped_swiglu(gate, up, p.gate_cap, p.up_cap));
    }
}

kernel void inkling_m1_w13_mxfp8(
    const device bfloat* x       [[buffer(0)]],
    const device uint* w         [[buffer(1)]],
    const device uchar* scales   [[buffer(2)]],
    const device uint* inds      [[buffer(3)]],
    device bfloat* out           [[buffer(4)]],
    constant MoePrm& p           [[buffer(5)]],
    uint3 tg_pos                 [[threadgroup_position_in_grid]],
    uint lane                    [[thread_index_in_simdgroup]],
    uint sg                      [[simdgroup_index_in_threadgroup]])
{
    const uint sub = lane >> 3;
    const uint slane = lane & 7u;
    const int r = (int(tg_pos.x) * FC_SG + int(sg)) * 4 + int(sub);
    const int slot = int(tg_pos.y);
    const int m = int(tg_pos.z);
    if (r >= MODEL_INTER) return;

    const uint e = inds[m * MODEL_TOP_K + slot];
    constexpr int HGROUPS = MODEL_D / 16;
    constexpr int SGROUPS = MODEL_D / 32;
    constexpr int W_STRIDE = MODEL_D / 4;
    const device bfloat* xm = x + size_t(m) * MODEL_D;
    const size_t w_base_g = (size_t(e) * (2 * MODEL_INTER) + size_t(2 * r)) * W_STRIDE;
    const size_t w_base_u = w_base_g + W_STRIDE;
    const size_t s_base_g = (size_t(e) * (2 * MODEL_INTER) + size_t(2 * r)) * SGROUPS;
    const size_t s_base_u = s_base_g + SGROUPS;

    float acc_g = 0.0f, acc_u = 0.0f;
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
    const float gate = round_bf16(reduce8(acc_g));
    const float up   = round_bf16(reduce8(acc_u));
    if (slane == 0) {
        out[(size_t(m) * MODEL_TOP_K + slot) * MODEL_INTER + r] = bfloat(softcapped_swiglu(gate, up, p.gate_cap, p.up_cap));
    }
}

kernel void inkling_m1_w2_nvfp4(
    const device bfloat* h       [[buffer(0)]],
    const device uint* w         [[buffer(1)]],
    const device uchar* scales   [[buffer(2)]],
    const device float* gscale   [[buffer(3)]],
    const device uint* inds      [[buffer(4)]],
    device bfloat* ys            [[buffer(5)]],
    uint3 tg_pos                 [[threadgroup_position_in_grid]],
    uint lane                    [[thread_index_in_simdgroup]],
    uint sg                      [[simdgroup_index_in_threadgroup]])
{
    const uint sub = lane >> 3;
    const uint slane = lane & 7u;
    const int d = (int(tg_pos.x) * FC_SG + int(sg)) * 4 + int(sub);
    const int slot = int(tg_pos.y);
    const int m = int(tg_pos.z);
    if (d >= MODEL_D) return;

    const uint e = inds[m * MODEL_TOP_K + slot];
    const float oscale = gscale[e] * (1.0f / 2688.0f);
    const device bfloat* hm = h + (size_t(m) * MODEL_TOP_K + slot) * MODEL_INTER;
    constexpr int GROUPS = MODEL_INTER / 16;
    constexpr int W_STRIDE = MODEL_INTER / 8;
    const size_t w_base = (size_t(e) * MODEL_D + size_t(d)) * W_STRIDE;
    const size_t s_base = (size_t(e) * MODEL_D + size_t(d)) * GROUPS;

    float acc = 0.0f;
    for (int g = int(slane); g < GROUPS; g += 8) {
        const uint2 ww = *(const device uint2*)(w + w_base + g * 2);
        const float sv = decode_fp8_e4m3(scales[s_base + g]);
        auto hp = (const device vec<bfloat, 4>*)(hm + g * 16);
        acc += (dot8_nvfp4(ww.x, hp[0], hp[1]) + dot8_nvfp4(ww.y, hp[2], hp[3])) * sv;
    }
    const bfloat yk = bfloat(reduce8(acc) * oscale);
    if (slane == 0) ys[(size_t(m) * MODEL_TOP_K + slot) * MODEL_D + d] = yk;
}

kernel void inkling_m1_w2_mxfp8(
    const device bfloat* h       [[buffer(0)]],
    const device uint* w         [[buffer(1)]],
    const device uchar* scales   [[buffer(2)]],
    const device uint* inds      [[buffer(3)]],
    device bfloat* ys            [[buffer(4)]],
    uint3 tg_pos                 [[threadgroup_position_in_grid]],
    uint lane                    [[thread_index_in_simdgroup]],
    uint sg                      [[simdgroup_index_in_threadgroup]])
{
    const uint sub = lane >> 3;
    const uint slane = lane & 7u;
    const int d = (int(tg_pos.x) * FC_SG + int(sg)) * 4 + int(sub);
    const int slot = int(tg_pos.y);
    const int m = int(tg_pos.z);
    if (d >= MODEL_D) return;

    const uint e = inds[m * MODEL_TOP_K + slot];
    const device bfloat* hm = h + (size_t(m) * MODEL_TOP_K + slot) * MODEL_INTER;
    constexpr int HGROUPS = MODEL_INTER / 16;
    constexpr int SGROUPS = MODEL_INTER / 32;
    constexpr int W_STRIDE = MODEL_INTER / 4;
    const size_t w_base = (size_t(e) * MODEL_D + size_t(d)) * W_STRIDE;
    const size_t s_base = (size_t(e) * MODEL_D + size_t(d)) * SGROUPS;

    float acc = 0.0f;
    for (int hg = int(slane); hg < HGROUPS; hg += 8) {
        const uint4 ww = *(const device uint4*)(w + w_base + hg * 4);
        const float sv = decode_fp8_e8m0(scales[s_base + (hg >> 1)]);
        auto hp = (const device vec<bfloat, 4>*)(hm + hg * 16);
        acc += (dot4_fp8(ww.x, hp[0]) + dot4_fp8(ww.y, hp[1]) + dot4_fp8(ww.z, hp[2]) + dot4_fp8(ww.w, hp[3])) * sv;
    }
    const bfloat yk = bfloat(reduce8(acc));
    if (slane == 0) ys[(size_t(m) * MODEL_TOP_K + slot) * MODEL_D + d] = yk;
}

kernel void inkling_grp_w13_nvfp4(
    const device bfloat* x         [[buffer(0)]],
    const device uint* w           [[buffer(1)]],
    const device uchar* scales     [[buffer(2)]],
    const device float* gscale     [[buffer(3)]],
    const device int* offsets      [[buffer(4)]],
    const device uint* sorted_slots[[buffer(5)]],
    const device int* active_list  [[buffer(6)]],
    device bfloat* h_sorted        [[buffer(7)]],
    constant MoePrm& p             [[buffer(8)]],
    uint3 tg_pos                   [[threadgroup_position_in_grid]],
    uint tid_tg                    [[thread_index_in_threadgroup]],
    uint lane                      [[thread_index_in_simdgroup]],
    uint sg                        [[simdgroup_index_in_threadgroup]])
{
    const int e = active_list[int(tg_pos.y)];
    if (e < 0) return;
    const int beg = offsets[e];
    const int cnt = offsets[e + 1] - beg;
    if (cnt == 0) return;

    const uint sub = lane >> 3;
    const uint slane = lane & 7u;
    const int r = (int(tg_pos.x) * FC_SG + int(sg)) * 4 + int(sub);
    const bool valid_row = (r < MODEL_INTER);

    const float oscale = gscale[e] * (1.0f / 2688.0f);
    constexpr int PAIRS = MODEL_D / 32;
    constexpr int GROUPS = MODEL_D / 16;
    constexpr int W_STRIDE = MODEL_D / 8;
    const size_t w_base_g = (size_t(e) * (2 * MODEL_INTER) + size_t(2 * r)) * W_STRIDE;
    const size_t w_base_u = w_base_g + W_STRIDE;
    const size_t s_base_g = (size_t(e) * (2 * MODEL_INTER) + size_t(2 * r)) * GROUPS;
    const size_t s_base_u = s_base_g + GROUPS;

    threadgroup int tg_tok[8];
    for (int t0 = 0; t0 < cnt; t0 += FC_MT) {
        const int tn = min(FC_MT, cnt - t0);
        if (t0 > 0) threadgroup_barrier(mem_flags::mem_threadgroup);
        if (int(tid_tg) < tn) tg_tok[tid_tg] = int(sorted_slots[beg + t0 + tid_tg] / uint(MODEL_TOP_K));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!valid_row) continue;

        if (tn == 1) {
            const device bfloat* xm = x + size_t(tg_tok[0]) * MODEL_D;
            float ag = 0.0f, au = 0.0f;
            for (int pi = int(slane); pi < PAIRS; pi += 8) {
                const uint4 wg = *(const device uint4*)(w + w_base_g + pi * 4);
                const uchar2 sg2 = *(const device uchar2*)(scales + s_base_g + pi * 2);
                const uint4 wu = *(const device uint4*)(w + w_base_u + pi * 4);
                const uchar2 su2 = *(const device uchar2*)(scales + s_base_u + pi * 2);
                const device uint4* xq = (const device uint4*)(xm + pi * 32);
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
            const float gate = round_bf16(reduce8(ag) * oscale);
            const float up   = round_bf16(reduce8(au) * oscale);
            if (slane == 0) {
                h_sorted[size_t(beg + t0) * MODEL_INTER + r] = bfloat(softcapped_swiglu(gate, up, p.gate_cap, p.up_cap));
            }
        } else if (tn == FC_MT) {
            float acc_g[8];
            float acc_u[8];
            #pragma clang loop unroll(full)
            for (int ti = 0; ti < FC_MT; ++ti) { acc_g[ti] = 0.0f; acc_u[ti] = 0.0f; }
            for (int pi = int(slane); pi < PAIRS; pi += 8) {
                const uint4 wg = *(const device uint4*)(w + w_base_g + pi * 4);
                const uchar2 sg2 = *(const device uchar2*)(scales + s_base_g + pi * 2);
                const float sg0 = decode_fp8_e4m3(sg2.x), sg1 = decode_fp8_e4m3(sg2.y);
                const uint4 wu = *(const device uint4*)(w + w_base_u + pi * 4);
                const uchar2 su2 = *(const device uchar2*)(scales + s_base_u + pi * 2);
                const float su0 = decode_fp8_e4m3(su2.x), su1 = decode_fp8_e4m3(su2.y);
                {
                    half4 g0, g1, g2, g3, u0, u1, u2, u3;
                    unpack8_nvfp4_h(wg.x, g0, g1); unpack8_nvfp4_h(wg.y, g2, g3);
                    unpack8_nvfp4_h(wu.x, u0, u1); unpack8_nvfp4_h(wu.y, u2, u3);
                    #pragma clang loop unroll(full)
                    for (int ti = 0; ti < FC_MT; ++ti) {
                        const device uint4* xq = (const device uint4*)(x + size_t(tg_tok[ti]) * MODEL_D + pi * 32);
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
                    #pragma clang loop unroll(full)
                    for (int ti = 0; ti < FC_MT; ++ti) {
                        const device uint4* xq = (const device uint4*)(x + size_t(tg_tok[ti]) * MODEL_D + pi * 32 + 16);
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
            #pragma clang loop unroll(full)
            for (int ti = 0; ti < FC_MT; ++ti) {
                const float gate = round_bf16(reduce8(acc_g[ti]) * oscale);
                const float up   = round_bf16(reduce8(acc_u[ti]) * oscale);
                if (slane == 0) {
                    h_sorted[size_t(beg + t0 + ti) * MODEL_INTER + r] = bfloat(softcapped_swiglu(gate, up, p.gate_cap, p.up_cap));
                }
            }
        } else {
            float acc_g[8];
            float acc_u[8];
            #pragma clang loop unroll(full)
            for (int ti = 0; ti < FC_MT; ++ti) { acc_g[ti] = 0.0f; acc_u[ti] = 0.0f; }
            for (int pi = int(slane); pi < PAIRS; pi += 8) {
                const uint4 wg = *(const device uint4*)(w + w_base_g + pi * 4);
                const uchar2 sg2 = *(const device uchar2*)(scales + s_base_g + pi * 2);
                const float sg0 = decode_fp8_e4m3(sg2.x), sg1 = decode_fp8_e4m3(sg2.y);
                const uint4 wu = *(const device uint4*)(w + w_base_u + pi * 4);
                const uchar2 su2 = *(const device uchar2*)(scales + s_base_u + pi * 2);
                const float su0 = decode_fp8_e4m3(su2.x), su1 = decode_fp8_e4m3(su2.y);
                {
                    half4 g0, g1, g2, g3, u0, u1, u2, u3;
                    unpack8_nvfp4_h(wg.x, g0, g1); unpack8_nvfp4_h(wg.y, g2, g3);
                    unpack8_nvfp4_h(wu.x, u0, u1); unpack8_nvfp4_h(wu.y, u2, u3);
                    #pragma clang loop unroll(full)
                    for (int ti = 0; ti < FC_MT; ++ti) {
                        if (ti < tn) {
                            const device uint4* xq = (const device uint4*)(x + size_t(tg_tok[ti]) * MODEL_D + pi * 32);
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
                    #pragma clang loop unroll(full)
                    for (int ti = 0; ti < FC_MT; ++ti) {
                        if (ti < tn) {
                            const device uint4* xq = (const device uint4*)(x + size_t(tg_tok[ti]) * MODEL_D + pi * 32 + 16);
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
            #pragma clang loop unroll(full)
            for (int ti = 0; ti < FC_MT; ++ti) {
                if (ti < tn) {
                    const float gate = round_bf16(reduce8(acc_g[ti]) * oscale);
                    const float up   = round_bf16(reduce8(acc_u[ti]) * oscale);
                    if (slane == 0) {
                        h_sorted[size_t(beg + t0 + ti) * MODEL_INTER + r] = bfloat(softcapped_swiglu(gate, up, p.gate_cap, p.up_cap));
                    }
                }
            }
        }
    }
}

kernel void inkling_grp_w13_mxfp8(
    const device bfloat* x         [[buffer(0)]],
    const device uint* w           [[buffer(1)]],
    const device uchar* scales     [[buffer(2)]],
    const device int* offsets      [[buffer(3)]],
    const device uint* sorted_slots[[buffer(4)]],
    const device int* active_list  [[buffer(5)]],
    device bfloat* h_sorted        [[buffer(6)]],
    constant MoePrm& p             [[buffer(7)]],
    uint3 tg_pos                   [[threadgroup_position_in_grid]],
    uint tid_tg                    [[thread_index_in_threadgroup]],
    uint lane                      [[thread_index_in_simdgroup]],
    uint sg                        [[simdgroup_index_in_threadgroup]])
{
    const int e = active_list[int(tg_pos.y)];
    if (e < 0) return;
    const int beg = offsets[e];
    const int cnt = offsets[e + 1] - beg;
    if (cnt == 0) return;

    const uint sub = lane >> 3;
    const uint slane = lane & 7u;
    const int r = (int(tg_pos.x) * FC_SG + int(sg)) * 4 + int(sub);
    const bool valid_row = (r < MODEL_INTER);

    constexpr int HGROUPS = MODEL_D / 16;
    constexpr int SGROUPS = MODEL_D / 32;
    constexpr int W_STRIDE = MODEL_D / 4;
    const size_t w_base_g = (size_t(e) * (2 * MODEL_INTER) + size_t(2 * r)) * W_STRIDE;
    const size_t w_base_u = w_base_g + W_STRIDE;
    const size_t s_base_g = (size_t(e) * (2 * MODEL_INTER) + size_t(2 * r)) * SGROUPS;
    const size_t s_base_u = s_base_g + SGROUPS;

    threadgroup int tg_tok[8];
    for (int t0 = 0; t0 < cnt; t0 += FC_MT) {
        const int tn = min(FC_MT, cnt - t0);
        if (t0 > 0) threadgroup_barrier(mem_flags::mem_threadgroup);
        if (int(tid_tg) < tn) tg_tok[tid_tg] = int(sorted_slots[beg + t0 + tid_tg] / uint(MODEL_TOP_K));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!valid_row) continue;

        if (tn == 1) {
            const device bfloat* xm = x + size_t(tg_tok[0]) * MODEL_D;
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
            const float gate = round_bf16(reduce8(ag));
            const float up   = round_bf16(reduce8(au));
            if (slane == 0) {
                h_sorted[size_t(beg + t0) * MODEL_INTER + r] = bfloat(softcapped_swiglu(gate, up, p.gate_cap, p.up_cap));
            }
        } else if (tn == FC_MT) {
            float acc_g[8];
            float acc_u[8];
            #pragma clang loop unroll(full)
            for (int ti = 0; ti < FC_MT; ++ti) { acc_g[ti] = 0.0f; acc_u[ti] = 0.0f; }
            for (int hg = int(slane); hg < HGROUPS; hg += 8) {
                const uint4 wg = *(const device uint4*)(w + w_base_g + hg * 4);
                const float sg_val = decode_fp8_e8m0_x256(scales[s_base_g + (hg >> 1)]);
                const uint4 wu = *(const device uint4*)(w + w_base_u + hg * 4);
                const float su_val = decode_fp8_e8m0_x256(scales[s_base_u + (hg >> 1)]);
                const half4 g0 = unpack4_fp8_h_div256(wg.x), g1 = unpack4_fp8_h_div256(wg.y);
                const half4 g2 = unpack4_fp8_h_div256(wg.z), g3 = unpack4_fp8_h_div256(wg.w);
                const half4 u0 = unpack4_fp8_h_div256(wu.x), u1 = unpack4_fp8_h_div256(wu.y);
                const half4 u2 = unpack4_fp8_h_div256(wu.z), u3 = unpack4_fp8_h_div256(wu.w);
                #pragma clang loop unroll(full)
                for (int ti = 0; ti < FC_MT; ++ti) {
                    const device uint4* xq = (const device uint4*)(x + size_t(tg_tok[ti]) * MODEL_D + hg * 16);
                    const uint4 q0 = xq[0], q1 = xq[1];
                    const float4 fx0 = float4(as_type<vec<bfloat, 4>>(q0.xy));
                    const float4 fx1 = float4(as_type<vec<bfloat, 4>>(q0.zw));
                    const float4 fx2 = float4(as_type<vec<bfloat, 4>>(q1.xy));
                    const float4 fx3 = float4(as_type<vec<bfloat, 4>>(q1.zw));
                    acc_g[ti] += (dot4_hf(g0, fx0) + dot4_hf(g1, fx1) + dot4_hf(g2, fx2) + dot4_hf(g3, fx3)) * sg_val;
                    acc_u[ti] += (dot4_hf(u0, fx0) + dot4_hf(u1, fx1) + dot4_hf(u2, fx2) + dot4_hf(u3, fx3)) * su_val;
                }
            }
            #pragma clang loop unroll(full)
            for (int ti = 0; ti < FC_MT; ++ti) {
                const float gate = round_bf16(reduce8(acc_g[ti]));
                const float up   = round_bf16(reduce8(acc_u[ti]));
                if (slane == 0) {
                    h_sorted[size_t(beg + t0 + ti) * MODEL_INTER + r] = bfloat(softcapped_swiglu(gate, up, p.gate_cap, p.up_cap));
                }
            }
        } else {
            float acc_g[8];
            float acc_u[8];
            #pragma clang loop unroll(full)
            for (int ti = 0; ti < FC_MT; ++ti) { acc_g[ti] = 0.0f; acc_u[ti] = 0.0f; }
            for (int hg = int(slane); hg < HGROUPS; hg += 8) {
                const uint4 wg = *(const device uint4*)(w + w_base_g + hg * 4);
                const float sg_val = decode_fp8_e8m0_x256(scales[s_base_g + (hg >> 1)]);
                const uint4 wu = *(const device uint4*)(w + w_base_u + hg * 4);
                const float su_val = decode_fp8_e8m0_x256(scales[s_base_u + (hg >> 1)]);
                const half4 g0 = unpack4_fp8_h_div256(wg.x), g1 = unpack4_fp8_h_div256(wg.y);
                const half4 g2 = unpack4_fp8_h_div256(wg.z), g3 = unpack4_fp8_h_div256(wg.w);
                const half4 u0 = unpack4_fp8_h_div256(wu.x), u1 = unpack4_fp8_h_div256(wu.y);
                const half4 u2 = unpack4_fp8_h_div256(wu.z), u3 = unpack4_fp8_h_div256(wu.w);
                #pragma clang loop unroll(full)
                for (int ti = 0; ti < FC_MT; ++ti) {
                    if (ti < tn) {
                        const device uint4* xq = (const device uint4*)(x + size_t(tg_tok[ti]) * MODEL_D + hg * 16);
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
            #pragma clang loop unroll(full)
            for (int ti = 0; ti < FC_MT; ++ti) {
                if (ti < tn) {
                    const float gate = round_bf16(reduce8(acc_g[ti]));
                    const float up   = round_bf16(reduce8(acc_u[ti]));
                    if (slane == 0) {
                        h_sorted[size_t(beg + t0 + ti) * MODEL_INTER + r] = bfloat(softcapped_swiglu(gate, up, p.gate_cap, p.up_cap));
                    }
                }
            }
        }
    }
}

kernel void inkling_grp_w2_nvfp4(
    const device bfloat* h_sorted  [[buffer(0)]],
    const device uint* w           [[buffer(1)]],
    const device uchar* scales     [[buffer(2)]],
    const device float* gscale     [[buffer(3)]],
    const device int* offsets      [[buffer(4)]],
    const device uint* sorted_slots[[buffer(5)]],
    const device int* active_list  [[buffer(6)]],
    device bfloat* ys              [[buffer(7)]],
    uint3 tg_pos                   [[threadgroup_position_in_grid]],
    uint tid_tg                    [[thread_index_in_threadgroup]],
    uint lane                      [[thread_index_in_simdgroup]],
    uint sg                        [[simdgroup_index_in_threadgroup]])
{
    const int e = active_list[int(tg_pos.y)];
    if (e < 0) return;
    const int beg = offsets[e];
    const int cnt = offsets[e + 1] - beg;
    if (cnt == 0) return;

    const uint sub = lane >> 3;
    const uint slane = lane & 7u;
    const int d = (int(tg_pos.x) * FC_SG + int(sg)) * 4 + int(sub);
    const bool valid_row = (d < MODEL_D);

    const float oscale = gscale[e] * (1.0f / 2688.0f);
    constexpr int GROUPS = MODEL_INTER / 16;
    constexpr int W_STRIDE = MODEL_INTER / 8;
    const size_t w_base = (size_t(e) * MODEL_D + size_t(d)) * W_STRIDE;
    const size_t s_base = (size_t(e) * MODEL_D + size_t(d)) * GROUPS;

    threadgroup uint tg_slot[8];
    for (int t0 = 0; t0 < cnt; t0 += FC_MT) {
        const int tn = min(FC_MT, cnt - t0);
        if (t0 > 0) threadgroup_barrier(mem_flags::mem_threadgroup);
        if (int(tid_tg) < tn) tg_slot[tid_tg] = sorted_slots[beg + t0 + tid_tg];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!valid_row) continue;

        if (tn == 1) {
            const device bfloat* hm = h_sorted + size_t(beg + t0) * MODEL_INTER;
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
            if (slane == 0) ys[size_t(tg_slot[0]) * MODEL_D + d] = yk;
        } else if (tn == FC_MT) {
            float acc[8];
            #pragma clang loop unroll(full)
            for (int ti = 0; ti < FC_MT; ++ti) acc[ti] = 0.0f;
            for (int g = int(slane); g < GROUPS; g += 8) {
                const uint2 ww = *(const device uint2*)(w + w_base + g * 2);
                const float sv = decode_fp8_e4m3(scales[s_base + g]);
                half4 f0, f1, f2, f3;
                unpack8_nvfp4_h(ww.x, f0, f1); unpack8_nvfp4_h(ww.y, f2, f3);
                #pragma clang loop unroll(full)
                for (int ti = 0; ti < FC_MT; ++ti) {
                    const device uint4* hq = (const device uint4*)(h_sorted + size_t(beg + t0 + ti) * MODEL_INTER + g * 16);
                    const uint4 q0 = hq[0], q1 = hq[1];
                    acc[ti] += (dot8_h(f0, f1, as_type<vec<bfloat, 4>>(q0.xy), as_type<vec<bfloat, 4>>(q0.zw))
                              + dot8_h(f2, f3, as_type<vec<bfloat, 4>>(q1.xy), as_type<vec<bfloat, 4>>(q1.zw))) * sv;
                }
            }
            #pragma clang loop unroll(full)
            for (int ti = 0; ti < FC_MT; ++ti) {
                const bfloat yk = bfloat(reduce8(acc[ti]) * oscale);
                if (slane == 0) ys[size_t(tg_slot[ti]) * MODEL_D + d] = yk;
            }
        } else {
            float acc[8];
            #pragma clang loop unroll(full)
            for (int ti = 0; ti < FC_MT; ++ti) acc[ti] = 0.0f;
            for (int g = int(slane); g < GROUPS; g += 8) {
                const uint2 ww = *(const device uint2*)(w + w_base + g * 2);
                const float sv = decode_fp8_e4m3(scales[s_base + g]);
                half4 f0, f1, f2, f3;
                unpack8_nvfp4_h(ww.x, f0, f1); unpack8_nvfp4_h(ww.y, f2, f3);
                #pragma clang loop unroll(full)
                for (int ti = 0; ti < FC_MT; ++ti) {
                    if (ti < tn) {
                        const device uint4* hq = (const device uint4*)(h_sorted + size_t(beg + t0 + ti) * MODEL_INTER + g * 16);
                        const uint4 q0 = hq[0], q1 = hq[1];
                        acc[ti] += (dot8_h(f0, f1, as_type<vec<bfloat, 4>>(q0.xy), as_type<vec<bfloat, 4>>(q0.zw))
                                  + dot8_h(f2, f3, as_type<vec<bfloat, 4>>(q1.xy), as_type<vec<bfloat, 4>>(q1.zw))) * sv;
                    }
                }
            }
            #pragma clang loop unroll(full)
            for (int ti = 0; ti < FC_MT; ++ti) {
                if (ti < tn) {
                    const bfloat yk = bfloat(reduce8(acc[ti]) * oscale);
                    if (slane == 0) ys[size_t(tg_slot[ti]) * MODEL_D + d] = yk;
                }
            }
        }
    }
}

kernel void inkling_grp_w2_mxfp8(
    const device bfloat* h_sorted  [[buffer(0)]],
    const device uint* w           [[buffer(1)]],
    const device uchar* scales     [[buffer(2)]],
    const device int* offsets      [[buffer(3)]],
    const device uint* sorted_slots[[buffer(4)]],
    const device int* active_list  [[buffer(5)]],
    device bfloat* ys              [[buffer(6)]],
    uint3 tg_pos                   [[threadgroup_position_in_grid]],
    uint tid_tg                    [[thread_index_in_threadgroup]],
    uint lane                      [[thread_index_in_simdgroup]],
    uint sg                        [[simdgroup_index_in_threadgroup]])
{
    const int e = active_list[int(tg_pos.y)];
    if (e < 0) return;
    const int beg = offsets[e];
    const int cnt = offsets[e + 1] - beg;
    if (cnt == 0) return;

    const uint sub = lane >> 3;
    const uint slane = lane & 7u;
    const int d = (int(tg_pos.x) * FC_SG + int(sg)) * 4 + int(sub);
    const bool valid_row = (d < MODEL_D);

    constexpr int HGROUPS = MODEL_INTER / 16;
    constexpr int SGROUPS = MODEL_INTER / 32;
    constexpr int W_STRIDE = MODEL_INTER / 4;
    const size_t w_base = (size_t(e) * MODEL_D + size_t(d)) * W_STRIDE;
    const size_t s_base = (size_t(e) * MODEL_D + size_t(d)) * SGROUPS;

    threadgroup uint tg_slot[8];
    for (int t0 = 0; t0 < cnt; t0 += FC_MT) {
        const int tn = min(FC_MT, cnt - t0);
        if (t0 > 0) threadgroup_barrier(mem_flags::mem_threadgroup);
        if (int(tid_tg) < tn) tg_slot[tid_tg] = sorted_slots[beg + t0 + tid_tg];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!valid_row) continue;

        if (tn == 1) {
            const device bfloat* hm = h_sorted + size_t(beg + t0) * MODEL_INTER;
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
            if (slane == 0) ys[size_t(tg_slot[0]) * MODEL_D + d] = yk;
        } else if (tn == FC_MT) {
            float acc[8];
            #pragma clang loop unroll(full)
            for (int ti = 0; ti < FC_MT; ++ti) acc[ti] = 0.0f;
            for (int hg = int(slane); hg < HGROUPS; hg += 8) {
                const uint4 ww = *(const device uint4*)(w + w_base + hg * 4);
                const float sv = decode_fp8_e8m0_x256(scales[s_base + (hg >> 1)]);
                const half4 f0 = unpack4_fp8_h_div256(ww.x), f1 = unpack4_fp8_h_div256(ww.y);
                const half4 f2 = unpack4_fp8_h_div256(ww.z), f3 = unpack4_fp8_h_div256(ww.w);
                #pragma clang loop unroll(full)
                for (int ti = 0; ti < FC_MT; ++ti) {
                    const device uint4* hq = (const device uint4*)(h_sorted + size_t(beg + t0 + ti) * MODEL_INTER + hg * 16);
                    const uint4 q0 = hq[0], q1 = hq[1];
                    acc[ti] += (dot4_h(f0, as_type<vec<bfloat, 4>>(q0.xy))
                              + dot4_h(f1, as_type<vec<bfloat, 4>>(q0.zw))
                              + dot4_h(f2, as_type<vec<bfloat, 4>>(q1.xy))
                              + dot4_h(f3, as_type<vec<bfloat, 4>>(q1.zw))) * sv;
                }
            }
            #pragma clang loop unroll(full)
            for (int ti = 0; ti < FC_MT; ++ti) {
                const bfloat yk = bfloat(reduce8(acc[ti]));
                if (slane == 0) ys[size_t(tg_slot[ti]) * MODEL_D + d] = yk;
            }
        } else {
            float acc[8];
            #pragma clang loop unroll(full)
            for (int ti = 0; ti < FC_MT; ++ti) acc[ti] = 0.0f;
            for (int hg = int(slane); hg < HGROUPS; hg += 8) {
                const uint4 ww = *(const device uint4*)(w + w_base + hg * 4);
                const float sv = decode_fp8_e8m0_x256(scales[s_base + (hg >> 1)]);
                const half4 f0 = unpack4_fp8_h_div256(ww.x), f1 = unpack4_fp8_h_div256(ww.y);
                const half4 f2 = unpack4_fp8_h_div256(ww.z), f3 = unpack4_fp8_h_div256(ww.w);
                #pragma clang loop unroll(full)
                for (int ti = 0; ti < FC_MT; ++ti) {
                    if (ti < tn) {
                        const device uint4* hq = (const device uint4*)(h_sorted + size_t(beg + t0 + ti) * MODEL_INTER + hg * 16);
                        const uint4 q0 = hq[0], q1 = hq[1];
                        acc[ti] += (dot4_h(f0, as_type<vec<bfloat, 4>>(q0.xy))
                                  + dot4_h(f1, as_type<vec<bfloat, 4>>(q0.zw))
                                  + dot4_h(f2, as_type<vec<bfloat, 4>>(q1.xy))
                                  + dot4_h(f3, as_type<vec<bfloat, 4>>(q1.zw))) * sv;
                    }
                }
            }
            #pragma clang loop unroll(full)
            for (int ti = 0; ti < FC_MT; ++ti) {
                if (ti < tn) {
                    const bfloat yk = bfloat(reduce8(acc[ti]));
                    if (slane == 0) ys[size_t(tg_slot[ti]) * MODEL_D + d] = yk;
                }
            }
        }
    }
}

kernel void inkling_grp_w13_nvfp4_tiles(
    const device bfloat* x         [[buffer(0)]],
    const device uint* w           [[buffer(1)]],
    const device uchar* scales     [[buffer(2)]],
    const device float* gscale     [[buffer(3)]],
    const device int* offsets      [[buffer(4)]],
    const device uint* sorted_slots[[buffer(5)]],
    const device int* tile_expert  [[buffer(6)]],
    const device int* tile_t0      [[buffer(7)]],
    device bfloat* h_sorted        [[buffer(8)]],
    constant MoePrm& p             [[buffer(9)]],
    uint3 tg_pos                   [[threadgroup_position_in_grid]],
    uint tid_tg                    [[thread_index_in_threadgroup]],
    uint lane                      [[thread_index_in_simdgroup]],
    uint sg                        [[simdgroup_index_in_threadgroup]])
{
    const int tile_idx = int(tg_pos.y);
    const int e = tile_expert[tile_idx];
    if (e < 0) return;
    const int beg = offsets[e];
    const int cnt = offsets[e + 1] - beg;
    const int t0 = tile_t0[tile_idx];
    const int tn = min(FC_MT, cnt - t0);
    if (tn <= 0) return;

    const uint sub = lane >> 3;
    const uint slane = lane & 7u;
    const int r = (int(tg_pos.x) * FC_SG + int(sg)) * 4 + int(sub);
    const bool valid_row = (r < MODEL_INTER);

    const float oscale = gscale[e] * (1.0f / 2688.0f);
    constexpr int PAIRS = MODEL_D / 32;
    constexpr int GROUPS = MODEL_D / 16;
    constexpr int W_STRIDE = MODEL_D / 8;
    const size_t w_base_g = (size_t(e) * (2 * MODEL_INTER) + size_t(2 * r)) * W_STRIDE;
    const size_t w_base_u = w_base_g + W_STRIDE;
    const size_t s_base_g = (size_t(e) * (2 * MODEL_INTER) + size_t(2 * r)) * GROUPS;
    const size_t s_base_u = s_base_g + GROUPS;

    threadgroup int tg_tok[8];
    if (int(tid_tg) < tn) tg_tok[tid_tg] = int(sorted_slots[beg + t0 + tid_tg] / uint(MODEL_TOP_K));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!valid_row) return;

    if (tn == 1) {
        const device bfloat* xm = x + size_t(tg_tok[0]) * MODEL_D;
        float ag = 0.0f, au = 0.0f;
        for (int pi = int(slane); pi < PAIRS; pi += 8) {
            const uint4 wg = *(const device uint4*)(w + w_base_g + pi * 4);
            const uchar2 sg2 = *(const device uchar2*)(scales + s_base_g + pi * 2);
            const uint4 wu = *(const device uint4*)(w + w_base_u + pi * 4);
            const uchar2 su2 = *(const device uchar2*)(scales + s_base_u + pi * 2);
            const device uint4* xq = (const device uint4*)(xm + pi * 32);
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
        const float gate = round_bf16(reduce8(ag) * oscale);
        const float up   = round_bf16(reduce8(au) * oscale);
        if (slane == 0) {
            h_sorted[size_t(beg + t0) * MODEL_INTER + r] = bfloat(softcapped_swiglu(gate, up, p.gate_cap, p.up_cap));
        }
    } else if (tn == FC_MT) {
        float acc_g[8], acc_u[8];
        #pragma clang loop unroll(full)
        for (int ti = 0; ti < FC_MT; ++ti) { acc_g[ti] = 0.0f; acc_u[ti] = 0.0f; }
        for (int pi = int(slane); pi < PAIRS; pi += 8) {
            const uint4 wg = *(const device uint4*)(w + w_base_g + pi * 4);
            const uchar2 sg2 = *(const device uchar2*)(scales + s_base_g + pi * 2);
            const float sg0 = decode_fp8_e4m3(sg2.x), sg1 = decode_fp8_e4m3(sg2.y);
            const uint4 wu = *(const device uint4*)(w + w_base_u + pi * 4);
            const uchar2 su2 = *(const device uchar2*)(scales + s_base_u + pi * 2);
            const float su0 = decode_fp8_e4m3(su2.x), su1 = decode_fp8_e4m3(su2.y);
            {
                half4 g0, g1, g2, g3, u0, u1, u2, u3;
                unpack8_nvfp4_h(wg.x, g0, g1); unpack8_nvfp4_h(wg.y, g2, g3);
                unpack8_nvfp4_h(wu.x, u0, u1); unpack8_nvfp4_h(wu.y, u2, u3);
                #pragma clang loop unroll(full)
                for (int ti = 0; ti < FC_MT; ++ti) {
                    const device uint4* xq = (const device uint4*)(x + size_t(tg_tok[ti]) * MODEL_D + pi * 32);
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
                #pragma clang loop unroll(full)
                for (int ti = 0; ti < FC_MT; ++ti) {
                    const device uint4* xq = (const device uint4*)(x + size_t(tg_tok[ti]) * MODEL_D + pi * 32 + 16);
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
        #pragma clang loop unroll(full)
        for (int ti = 0; ti < FC_MT; ++ti) {
            const float gate = round_bf16(reduce8(acc_g[ti]) * oscale);
            const float up   = round_bf16(reduce8(acc_u[ti]) * oscale);
            if (slane == 0) {
                h_sorted[size_t(beg + t0 + ti) * MODEL_INTER + r] = bfloat(softcapped_swiglu(gate, up, p.gate_cap, p.up_cap));
            }
        }
    } else {
        float acc_g[8], acc_u[8];
        #pragma clang loop unroll(full)
        for (int ti = 0; ti < FC_MT; ++ti) { acc_g[ti] = 0.0f; acc_u[ti] = 0.0f; }
        for (int pi = int(slane); pi < PAIRS; pi += 8) {
            const uint4 wg = *(const device uint4*)(w + w_base_g + pi * 4);
            const uchar2 sg2 = *(const device uchar2*)(scales + s_base_g + pi * 2);
            const float sg0 = decode_fp8_e4m3(sg2.x), sg1 = decode_fp8_e4m3(sg2.y);
            const uint4 wu = *(const device uint4*)(w + w_base_u + pi * 4);
            const uchar2 su2 = *(const device uchar2*)(scales + s_base_u + pi * 2);
            const float su0 = decode_fp8_e4m3(su2.x), su1 = decode_fp8_e4m3(su2.y);
            {
                half4 g0, g1, g2, g3, u0, u1, u2, u3;
                unpack8_nvfp4_h(wg.x, g0, g1); unpack8_nvfp4_h(wg.y, g2, g3);
                unpack8_nvfp4_h(wu.x, u0, u1); unpack8_nvfp4_h(wu.y, u2, u3);
                #pragma clang loop unroll(full)
                for (int ti = 0; ti < FC_MT; ++ti) {
                    if (ti < tn) {
                        const device uint4* xq = (const device uint4*)(x + size_t(tg_tok[ti]) * MODEL_D + pi * 32);
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
                #pragma clang loop unroll(full)
                for (int ti = 0; ti < FC_MT; ++ti) {
                    if (ti < tn) {
                        const device uint4* xq = (const device uint4*)(x + size_t(tg_tok[ti]) * MODEL_D + pi * 32 + 16);
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
        #pragma clang loop unroll(full)
        for (int ti = 0; ti < FC_MT; ++ti) {
            if (ti < tn) {
                const float gate = round_bf16(reduce8(acc_g[ti]) * oscale);
                const float up   = round_bf16(reduce8(acc_u[ti]) * oscale);
                if (slane == 0) {
                    h_sorted[size_t(beg + t0 + ti) * MODEL_INTER + r] = bfloat(softcapped_swiglu(gate, up, p.gate_cap, p.up_cap));
                }
            }
        }
    }
}

kernel void inkling_grp_w13_mxfp8_tiles(
    const device bfloat* x         [[buffer(0)]],
    const device uint* w           [[buffer(1)]],
    const device uchar* scales     [[buffer(2)]],
    const device int* offsets      [[buffer(3)]],
    const device uint* sorted_slots[[buffer(4)]],
    const device int* tile_expert  [[buffer(5)]],
    const device int* tile_t0      [[buffer(6)]],
    device bfloat* h_sorted        [[buffer(7)]],
    constant MoePrm& p             [[buffer(8)]],
    uint3 tg_pos                   [[threadgroup_position_in_grid]],
    uint tid_tg                    [[thread_index_in_threadgroup]],
    uint lane                      [[thread_index_in_simdgroup]],
    uint sg                        [[simdgroup_index_in_threadgroup]])
{
    const int tile_idx = int(tg_pos.y);
    const int e = tile_expert[tile_idx];
    if (e < 0) return;
    const int beg = offsets[e];
    const int cnt = offsets[e + 1] - beg;
    const int t0 = tile_t0[tile_idx];
    const int tn = min(FC_MT, cnt - t0);
    if (tn <= 0) return;

    const uint sub = lane >> 3;
    const uint slane = lane & 7u;
    const int r = (int(tg_pos.x) * FC_SG + int(sg)) * 4 + int(sub);
    const bool valid_row = (r < MODEL_INTER);

    constexpr int HGROUPS = MODEL_D / 16;
    constexpr int SGROUPS = MODEL_D / 32;
    constexpr int W_STRIDE = MODEL_D / 4;
    const size_t w_base_g = (size_t(e) * (2 * MODEL_INTER) + size_t(2 * r)) * W_STRIDE;
    const size_t w_base_u = w_base_g + W_STRIDE;
    const size_t s_base_g = (size_t(e) * (2 * MODEL_INTER) + size_t(2 * r)) * SGROUPS;
    const size_t s_base_u = s_base_g + SGROUPS;

    threadgroup int tg_tok[8];
    if (int(tid_tg) < tn) tg_tok[tid_tg] = int(sorted_slots[beg + t0 + tid_tg] / uint(MODEL_TOP_K));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!valid_row) return;

    if (tn == 1) {
        const device bfloat* xm = x + size_t(tg_tok[0]) * MODEL_D;
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
        const float gate = round_bf16(reduce8(ag));
        const float up   = round_bf16(reduce8(au));
        if (slane == 0) {
            h_sorted[size_t(beg + t0) * MODEL_INTER + r] = bfloat(softcapped_swiglu(gate, up, p.gate_cap, p.up_cap));
        }
    } else if (tn == FC_MT) {
        float acc_g[8], acc_u[8];
        #pragma clang loop unroll(full)
        for (int ti = 0; ti < FC_MT; ++ti) { acc_g[ti] = 0.0f; acc_u[ti] = 0.0f; }
        for (int hg = int(slane); hg < HGROUPS; hg += 8) {
            const uint4 wg = *(const device uint4*)(w + w_base_g + hg * 4);
            const float sg_val = decode_fp8_e8m0_x256(scales[s_base_g + (hg >> 1)]);
            const uint4 wu = *(const device uint4*)(w + w_base_u + hg * 4);
            const float su_val = decode_fp8_e8m0_x256(scales[s_base_u + (hg >> 1)]);
            const half4 g0 = unpack4_fp8_h_div256(wg.x), g1 = unpack4_fp8_h_div256(wg.y);
            const half4 g2 = unpack4_fp8_h_div256(wg.z), g3 = unpack4_fp8_h_div256(wg.w);
            const half4 u0 = unpack4_fp8_h_div256(wu.x), u1 = unpack4_fp8_h_div256(wu.y);
            const half4 u2 = unpack4_fp8_h_div256(wu.z), u3 = unpack4_fp8_h_div256(wu.w);
            #pragma clang loop unroll(full)
            for (int ti = 0; ti < FC_MT; ++ti) {
                const device uint4* xq = (const device uint4*)(x + size_t(tg_tok[ti]) * MODEL_D + hg * 16);
                const uint4 q0 = xq[0], q1 = xq[1];
                const float4 fx0 = float4(as_type<vec<bfloat, 4>>(q0.xy));
                const float4 fx1 = float4(as_type<vec<bfloat, 4>>(q0.zw));
                const float4 fx2 = float4(as_type<vec<bfloat, 4>>(q1.xy));
                const float4 fx3 = float4(as_type<vec<bfloat, 4>>(q1.zw));
                acc_g[ti] += (dot4_hf(g0, fx0) + dot4_hf(g1, fx1) + dot4_hf(g2, fx2) + dot4_hf(g3, fx3)) * sg_val;
                acc_u[ti] += (dot4_hf(u0, fx0) + dot4_hf(u1, fx1) + dot4_hf(u2, fx2) + dot4_hf(u3, fx3)) * su_val;
            }
        }
        #pragma clang loop unroll(full)
        for (int ti = 0; ti < FC_MT; ++ti) {
            const float gate = round_bf16(reduce8(acc_g[ti]));
            const float up   = round_bf16(reduce8(acc_u[ti]));
            if (slane == 0) {
                h_sorted[size_t(beg + t0 + ti) * MODEL_INTER + r] = bfloat(softcapped_swiglu(gate, up, p.gate_cap, p.up_cap));
            }
        }
    } else {
        float acc_g[8], acc_u[8];
        #pragma clang loop unroll(full)
        for (int ti = 0; ti < FC_MT; ++ti) { acc_g[ti] = 0.0f; acc_u[ti] = 0.0f; }
        for (int hg = int(slane); hg < HGROUPS; hg += 8) {
            const uint4 wg = *(const device uint4*)(w + w_base_g + hg * 4);
            const float sg_val = decode_fp8_e8m0_x256(scales[s_base_g + (hg >> 1)]);
            const uint4 wu = *(const device uint4*)(w + w_base_u + hg * 4);
            const float su_val = decode_fp8_e8m0_x256(scales[s_base_u + (hg >> 1)]);
            const half4 g0 = unpack4_fp8_h_div256(wg.x), g1 = unpack4_fp8_h_div256(wg.y);
            const half4 g2 = unpack4_fp8_h_div256(wg.z), g3 = unpack4_fp8_h_div256(wg.w);
            const half4 u0 = unpack4_fp8_h_div256(wu.x), u1 = unpack4_fp8_h_div256(wu.y);
            const half4 u2 = unpack4_fp8_h_div256(wu.z), u3 = unpack4_fp8_h_div256(wu.w);
            #pragma clang loop unroll(full)
            for (int ti = 0; ti < FC_MT; ++ti) {
                if (ti < tn) {
                    const device uint4* xq = (const device uint4*)(x + size_t(tg_tok[ti]) * MODEL_D + hg * 16);
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
        #pragma clang loop unroll(full)
        for (int ti = 0; ti < FC_MT; ++ti) {
            if (ti < tn) {
                const float gate = round_bf16(reduce8(acc_g[ti]));
                const float up   = round_bf16(reduce8(acc_u[ti]));
                if (slane == 0) {
                    h_sorted[size_t(beg + t0 + ti) * MODEL_INTER + r] = bfloat(softcapped_swiglu(gate, up, p.gate_cap, p.up_cap));
                }
            }
        }
    }
}

kernel void inkling_grp_w2_nvfp4_tiles(
    const device bfloat* h_sorted  [[buffer(0)]],
    const device uint* w           [[buffer(1)]],
    const device uchar* scales     [[buffer(2)]],
    const device float* gscale     [[buffer(3)]],
    const device int* offsets      [[buffer(4)]],
    const device uint* sorted_slots[[buffer(5)]],
    const device int* tile_expert  [[buffer(6)]],
    const device int* tile_t0      [[buffer(7)]],
    device bfloat* ys              [[buffer(8)]],
    uint3 tg_pos                   [[threadgroup_position_in_grid]],
    uint tid_tg                    [[thread_index_in_threadgroup]],
    uint lane                      [[thread_index_in_simdgroup]],
    uint sg                        [[simdgroup_index_in_threadgroup]])
{
    const int tile_idx = int(tg_pos.y);
    const int e = tile_expert[tile_idx];
    if (e < 0) return;
    const int beg = offsets[e];
    const int cnt = offsets[e + 1] - beg;
    const int t0 = tile_t0[tile_idx];
    const int tn = min(FC_MT, cnt - t0);
    if (tn <= 0) return;

    const uint sub = lane >> 3;
    const uint slane = lane & 7u;
    const int d = (int(tg_pos.x) * FC_SG + int(sg)) * 4 + int(sub);
    const bool valid_row = (d < MODEL_D);

    const float oscale = gscale[e] * (1.0f / 2688.0f);
    constexpr int GROUPS = MODEL_INTER / 16;
    constexpr int W_STRIDE = MODEL_INTER / 8;
    const size_t w_base = (size_t(e) * MODEL_D + size_t(d)) * W_STRIDE;
    const size_t s_base = (size_t(e) * MODEL_D + size_t(d)) * GROUPS;

    threadgroup uint tg_slot[8];
    if (int(tid_tg) < tn) tg_slot[tid_tg] = sorted_slots[beg + t0 + tid_tg];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!valid_row) return;

    if (tn == 1) {
        const device bfloat* hm = h_sorted + size_t(beg + t0) * MODEL_INTER;
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
        if (slane == 0) ys[size_t(tg_slot[0]) * MODEL_D + d] = yk;
    } else if (tn == FC_MT) {
        float acc[8];
        #pragma clang loop unroll(full)
        for (int ti = 0; ti < FC_MT; ++ti) acc[ti] = 0.0f;
        for (int g = int(slane); g < GROUPS; g += 8) {
            const uint2 ww = *(const device uint2*)(w + w_base + g * 2);
            const float sv = decode_fp8_e4m3(scales[s_base + g]);
            half4 f0, f1, f2, f3;
            unpack8_nvfp4_h(ww.x, f0, f1); unpack8_nvfp4_h(ww.y, f2, f3);
            #pragma clang loop unroll(full)
            for (int ti = 0; ti < FC_MT; ++ti) {
                const device uint4* hq = (const device uint4*)(h_sorted + size_t(beg + t0 + ti) * MODEL_INTER + g * 16);
                const uint4 q0 = hq[0], q1 = hq[1];
                acc[ti] += (dot8_h(f0, f1, as_type<vec<bfloat, 4>>(q0.xy), as_type<vec<bfloat, 4>>(q0.zw))
                          + dot8_h(f2, f3, as_type<vec<bfloat, 4>>(q1.xy), as_type<vec<bfloat, 4>>(q1.zw))) * sv;
            }
        }
        #pragma clang loop unroll(full)
        for (int ti = 0; ti < FC_MT; ++ti) {
            const bfloat yk = bfloat(reduce8(acc[ti]) * oscale);
            if (slane == 0) ys[size_t(tg_slot[ti]) * MODEL_D + d] = yk;
        }
    } else {
        float acc[8];
        #pragma clang loop unroll(full)
        for (int ti = 0; ti < FC_MT; ++ti) acc[ti] = 0.0f;
        for (int g = int(slane); g < GROUPS; g += 8) {
            const uint2 ww = *(const device uint2*)(w + w_base + g * 2);
            const float sv = decode_fp8_e4m3(scales[s_base + g]);
            half4 f0, f1, f2, f3;
            unpack8_nvfp4_h(ww.x, f0, f1); unpack8_nvfp4_h(ww.y, f2, f3);
            #pragma clang loop unroll(full)
            for (int ti = 0; ti < FC_MT; ++ti) {
                if (ti < tn) {
                    const device uint4* hq = (const device uint4*)(h_sorted + size_t(beg + t0 + ti) * MODEL_INTER + g * 16);
                    const uint4 q0 = hq[0], q1 = hq[1];
                    acc[ti] += (dot8_h(f0, f1, as_type<vec<bfloat, 4>>(q0.xy), as_type<vec<bfloat, 4>>(q0.zw))
                              + dot8_h(f2, f3, as_type<vec<bfloat, 4>>(q1.xy), as_type<vec<bfloat, 4>>(q1.zw))) * sv;
                }
            }
        }
        #pragma clang loop unroll(full)
        for (int ti = 0; ti < FC_MT; ++ti) {
            if (ti < tn) {
                const bfloat yk = bfloat(reduce8(acc[ti]) * oscale);
                if (slane == 0) ys[size_t(tg_slot[ti]) * MODEL_D + d] = yk;
            }
        }
    }
}

kernel void inkling_grp_w2_mxfp8_tiles(
    const device bfloat* h_sorted  [[buffer(0)]],
    const device uint* w           [[buffer(1)]],
    const device uchar* scales     [[buffer(2)]],
    const device int* offsets      [[buffer(3)]],
    const device uint* sorted_slots[[buffer(4)]],
    const device int* tile_expert  [[buffer(5)]],
    const device int* tile_t0      [[buffer(6)]],
    device bfloat* ys              [[buffer(7)]],
    uint3 tg_pos                   [[threadgroup_position_in_grid]],
    uint tid_tg                    [[thread_index_in_threadgroup]],
    uint lane                      [[thread_index_in_simdgroup]],
    uint sg                        [[simdgroup_index_in_threadgroup]])
{
    const int tile_idx = int(tg_pos.y);
    const int e = tile_expert[tile_idx];
    if (e < 0) return;
    const int beg = offsets[e];
    const int cnt = offsets[e + 1] - beg;
    const int t0 = tile_t0[tile_idx];
    const int tn = min(FC_MT, cnt - t0);
    if (tn <= 0) return;

    const uint sub = lane >> 3;
    const uint slane = lane & 7u;
    const int d = (int(tg_pos.x) * FC_SG + int(sg)) * 4 + int(sub);
    const bool valid_row = (d < MODEL_D);

    constexpr int HGROUPS = MODEL_INTER / 16;
    constexpr int SGROUPS = MODEL_INTER / 32;
    constexpr int W_STRIDE = MODEL_INTER / 4;
    const size_t w_base = (size_t(e) * MODEL_D + size_t(d)) * W_STRIDE;
    const size_t s_base = (size_t(e) * MODEL_D + size_t(d)) * SGROUPS;

    threadgroup uint tg_slot[8];
    if (int(tid_tg) < tn) tg_slot[tid_tg] = sorted_slots[beg + t0 + tid_tg];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!valid_row) return;

    if (tn == 1) {
        const device bfloat* hm = h_sorted + size_t(beg + t0) * MODEL_INTER;
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
        if (slane == 0) ys[size_t(tg_slot[0]) * MODEL_D + d] = yk;
    } else if (tn == FC_MT) {
        float acc[8];
        #pragma clang loop unroll(full)
        for (int ti = 0; ti < FC_MT; ++ti) acc[ti] = 0.0f;
        for (int hg = int(slane); hg < HGROUPS; hg += 8) {
            const uint4 ww = *(const device uint4*)(w + w_base + hg * 4);
            const float sv = decode_fp8_e8m0_x256(scales[s_base + (hg >> 1)]);
            const half4 f0 = unpack4_fp8_h_div256(ww.x), f1 = unpack4_fp8_h_div256(ww.y);
            const half4 f2 = unpack4_fp8_h_div256(ww.z), f3 = unpack4_fp8_h_div256(ww.w);
            #pragma clang loop unroll(full)
            for (int ti = 0; ti < FC_MT; ++ti) {
                const device uint4* hq = (const device uint4*)(h_sorted + size_t(beg + t0 + ti) * MODEL_INTER + hg * 16);
                const uint4 q0 = hq[0], q1 = hq[1];
                acc[ti] += (dot4_h(f0, as_type<vec<bfloat, 4>>(q0.xy))
                          + dot4_h(f1, as_type<vec<bfloat, 4>>(q0.zw))
                          + dot4_h(f2, as_type<vec<bfloat, 4>>(q1.xy))
                          + dot4_h(f3, as_type<vec<bfloat, 4>>(q1.zw))) * sv;
            }
        }
        #pragma clang loop unroll(full)
        for (int ti = 0; ti < FC_MT; ++ti) {
            const bfloat yk = bfloat(reduce8(acc[ti]));
            if (slane == 0) ys[size_t(tg_slot[ti]) * MODEL_D + d] = yk;
        }
    } else {
        float acc[8];
        #pragma clang loop unroll(full)
        for (int ti = 0; ti < FC_MT; ++ti) acc[ti] = 0.0f;
        for (int hg = int(slane); hg < HGROUPS; hg += 8) {
            const uint4 ww = *(const device uint4*)(w + w_base + hg * 4);
            const float sv = decode_fp8_e8m0_x256(scales[s_base + (hg >> 1)]);
            const half4 f0 = unpack4_fp8_h_div256(ww.x), f1 = unpack4_fp8_h_div256(ww.y);
            const half4 f2 = unpack4_fp8_h_div256(ww.z), f3 = unpack4_fp8_h_div256(ww.w);
            #pragma clang loop unroll(full)
            for (int ti = 0; ti < FC_MT; ++ti) {
                if (ti < tn) {
                    const device uint4* hq = (const device uint4*)(h_sorted + size_t(beg + t0 + ti) * MODEL_INTER + hg * 16);
                    const uint4 q0 = hq[0], q1 = hq[1];
                    acc[ti] += (dot4_h(f0, as_type<vec<bfloat, 4>>(q0.xy))
                              + dot4_h(f1, as_type<vec<bfloat, 4>>(q0.zw))
                              + dot4_h(f2, as_type<vec<bfloat, 4>>(q1.xy))
                              + dot4_h(f3, as_type<vec<bfloat, 4>>(q1.zw))) * sv;
                }
            }
        }
        #pragma clang loop unroll(full)
        for (int ti = 0; ti < FC_MT; ++ti) {
            if (ti < tn) {
                const bfloat yk = bfloat(reduce8(acc[ti]));
                if (slane == 0) ys[size_t(tg_slot[ti]) * MODEL_D + d] = yk;
            }
        }
    }
}

struct CombinePrm {
    int M;
};

kernel void inkling_combine(
    const device bfloat* ys    [[buffer(0)]],
    const device float* weights[[buffer(1)]],
    const device float* state  [[buffer(2)]],
    const device float* taps   [[buffer(3)]],
    const device float* resid  [[buffer(4)]],
    device bfloat* out         [[buffer(5)]],
    device float* new_state    [[buffer(6)]],
    constant CombinePrm& p     [[buffer(7)]],
    uint2 gid                  [[thread_position_in_grid]])
{
    const int d = int(gid.x);
    const int row = int(gid.y);
    if (d >= MODEL_D || row >= p.M) return;

    float ysum[4];
    if (FC_SEQ) {
        #pragma clang loop unroll(full)
        for (int j = 0; j < 4; j++) {
            const int r = row - 3 + j;
            ysum[j] = 0.0f;
            if (r >= 0) {
                float y = 0.0f;
                #pragma clang loop unroll(full)
                for (int k = 0; k < MODEL_TOP_K; k++) {
                    y += float(ys[(size_t(r) * MODEL_TOP_K + k) * MODEL_D + d]) * weights[size_t(r) * MODEL_TOP_K + k];
                }
                ysum[j] = round_bf16(y);
            }
        }
    } else {
        float y = 0.0f;
        #pragma clang loop unroll(full)
        for (int k = 0; k < MODEL_TOP_K; k++) {
            y += float(ys[(size_t(row) * MODEL_TOP_K + k) * MODEL_D + d]) * weights[size_t(row) * MODEL_TOP_K + k];
        }
        ysum[3] = round_bf16(y);
    }
    const float y = ysum[3];
    const size_t so = size_t(row) * 3 * MODEL_D + d;
    const size_t to = size_t(d) * 4;
    float s0, s1, s2;
    if (FC_SEQ) {
        s0 = row >= 3 ? ysum[0] : state[size_t(row) * MODEL_D + d];
        s1 = row >= 2 ? ysum[1] : state[size_t(row + 1) * MODEL_D + d];
        s2 = row >= 1 ? ysum[2] : state[size_t(row + 2) * MODEL_D + d];
    } else {
        s0 = state[so]; s1 = state[so + MODEL_D]; s2 = state[so + 2 * MODEL_D];
    }
    float h = (taps[to] * s0 + taps[to + 1] * s1 + taps[to + 2] * s2 + taps[to + 3] * y + y) * 0.4472135955f;
    h = round_bf16(h);
    new_state[so] = s1;
    new_state[so + MODEL_D] = s2;
    new_state[so + 2 * MODEL_D] = y;
    out[size_t(row) * MODEL_D + d] = bfloat(resid[size_t(row) * MODEL_D + d] + h);
}

struct ArgmaxP1Prm {
    int NTG;
};

kernel void inkling_head_argmax_p1(
    const device float* h    [[buffer(0)]],
    const device bfloat* w   [[buffer(1)]],
    device float* part_val   [[buffer(2)]],
    device uint* part_idx    [[buffer(3)]],
    constant ArgmaxP1Prm& p  [[buffer(4)]],
    uint tg_x                [[threadgroup_position_in_grid]],
    uint lane                [[thread_index_in_simdgroup]],
    uint sg                  [[simdgroup_index_in_threadgroup]])
{
    constexpr int SG = 8;
    constexpr int K8 = MODEL_D / 8;
    constexpr int K4 = MODEL_D / 4;
    const int tg_id = int(tg_x);
    const int v_base = tg_id * (SG * FC_R);
    const device float4* h4 = (const device float4*)(h);

    threadgroup float sg_v[SG][8];
    threadgroup uint  sg_i[SG][8];

    const int row0 = v_base + int(sg) * FC_R;
    if (row0 >= MODEL_V_LIMIT) {
        if (lane == 0) {
            #pragma clang loop unroll(full)
            for (int m = 0; m < FC_M; ++m) {
                sg_v[sg][m] = -INFINITY;
                sg_i[sg][m] = 0u;
            }
        }
    } else if (FC_R == 4 && FC_M == 1) {
        const int row1 = min(row0 + 1, MODEL_V_LIMIT - 1);
        const int row2 = min(row0 + 2, MODEL_V_LIMIT - 1);
        const int row3 = min(row0 + 3, MODEL_V_LIMIT - 1);
        const device uint4* w0 = (const device uint4*)(w + size_t(row0) * MODEL_D);
        const device uint4* w1 = (const device uint4*)(w + size_t(row1) * MODEL_D);
        const device uint4* w2 = (const device uint4*)(w + size_t(row2) * MODEL_D);
        const device uint4* w3 = (const device uint4*)(w + size_t(row3) * MODEL_D);
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int idx = int(lane); idx < K8; idx += 32) {
            const float4 h_lo = h4[idx * 2];
            const float4 h_hi = h4[idx * 2 + 1];
            const uint4 q0 = w0[idx], q1 = w1[idx], q2 = w2[idx], q3 = w3[idx];
            a0 += dot(float4(as_type<vec<bfloat, 4>>(q0.xy)), h_lo)
                + dot(float4(as_type<vec<bfloat, 4>>(q0.zw)), h_hi);
            a1 += dot(float4(as_type<vec<bfloat, 4>>(q1.xy)), h_lo)
                + dot(float4(as_type<vec<bfloat, 4>>(q1.zw)), h_hi);
            a2 += dot(float4(as_type<vec<bfloat, 4>>(q2.xy)), h_lo)
                + dot(float4(as_type<vec<bfloat, 4>>(q2.zw)), h_hi);
            a3 += dot(float4(as_type<vec<bfloat, 4>>(q3.xy)), h_lo)
                + dot(float4(as_type<vec<bfloat, 4>>(q3.zw)), h_hi);
        }
        const float t0 = simd_sum(a0);
        const float t1 = simd_sum(a1);
        const float t2 = simd_sum(a2);
        const float t3 = simd_sum(a3);
        if (lane == 0) {
            float bv0 = t0; uint bi0 = uint(row0);
            if (row0 + 1 < MODEL_V_LIMIT && t1 > bv0) { bv0 = t1; bi0 = uint(row1); }
            if (row0 + 2 < MODEL_V_LIMIT && t2 > bv0) { bv0 = t2; bi0 = uint(row2); }
            if (row0 + 3 < MODEL_V_LIMIT && t3 > bv0) { bv0 = t3; bi0 = uint(row3); }
            sg_v[sg][0] = bv0;
            sg_i[sg][0] = bi0;
        }
    } else if (FC_R == 2) {
        const int row1 = min(row0 + 1, MODEL_V_LIMIT - 1);
        const bool has_r1 = (row0 + 1) < MODEL_V_LIMIT;
        const device uint4* wa8 = (const device uint4*)(w + size_t(row0) * MODEL_D);
        const device uint4* wb8 = (const device uint4*)(w + size_t(row1) * MODEL_D);
        float acc0[8], acc1[8];
        #pragma clang loop unroll(full)
        for (int m = 0; m < FC_M; ++m) { acc0[m] = 0.0f; acc1[m] = 0.0f; }

        if (K8 % 96 == 0 && FC_M <= 4) {
            for (int base = 0; base < K8; base += 96) {
                const int i0 = base + int(lane);
                const int i1 = i0 + 32;
                const int i2 = i0 + 64;
                const uint4 qa0 = wa8[i0], qa1 = wa8[i1], qa2 = wa8[i2];
                const uint4 qb0 = wb8[i0], qb1 = wb8[i1], qb2 = wb8[i2];
                {
                    const float4 wa_lo = float4(as_type<vec<bfloat, 4>>(qa0.xy));
                    const float4 wa_hi = float4(as_type<vec<bfloat, 4>>(qa0.zw));
                    const float4 wb_lo = float4(as_type<vec<bfloat, 4>>(qb0.xy));
                    const float4 wb_hi = float4(as_type<vec<bfloat, 4>>(qb0.zw));
                    #pragma clang loop unroll(full)
                    for (int m = 0; m < FC_M; ++m) {
                        const device float4* hm = h4 + size_t(m) * K4 + i0 * 2;
                        const float4 h_lo = hm[0], h_hi = hm[1];
                        acc0[m] += dot(wa_lo, h_lo) + dot(wa_hi, h_hi);
                        acc1[m] += dot(wb_lo, h_lo) + dot(wb_hi, h_hi);
                    }
                }
                {
                    const float4 wa_lo = float4(as_type<vec<bfloat, 4>>(qa1.xy));
                    const float4 wa_hi = float4(as_type<vec<bfloat, 4>>(qa1.zw));
                    const float4 wb_lo = float4(as_type<vec<bfloat, 4>>(qb1.xy));
                    const float4 wb_hi = float4(as_type<vec<bfloat, 4>>(qb1.zw));
                    #pragma clang loop unroll(full)
                    for (int m = 0; m < FC_M; ++m) {
                        const device float4* hm = h4 + size_t(m) * K4 + i1 * 2;
                        const float4 h_lo = hm[0], h_hi = hm[1];
                        acc0[m] += dot(wa_lo, h_lo) + dot(wa_hi, h_hi);
                        acc1[m] += dot(wb_lo, h_lo) + dot(wb_hi, h_hi);
                    }
                }
                {
                    const float4 wa_lo = float4(as_type<vec<bfloat, 4>>(qa2.xy));
                    const float4 wa_hi = float4(as_type<vec<bfloat, 4>>(qa2.zw));
                    const float4 wb_lo = float4(as_type<vec<bfloat, 4>>(qb2.xy));
                    const float4 wb_hi = float4(as_type<vec<bfloat, 4>>(qb2.zw));
                    #pragma clang loop unroll(full)
                    for (int m = 0; m < FC_M; ++m) {
                        const device float4* hm = h4 + size_t(m) * K4 + i2 * 2;
                        const float4 h_lo = hm[0], h_hi = hm[1];
                        acc0[m] += dot(wa_lo, h_lo) + dot(wa_hi, h_hi);
                        acc1[m] += dot(wb_lo, h_lo) + dot(wb_hi, h_hi);
                    }
                }
            }
        } else {
            for (int idx = int(lane); idx < K8; idx += 32) {
                const uint4 qa = wa8[idx];
                const uint4 qb = wb8[idx];
                const float4 wa_lo = float4(as_type<vec<bfloat, 4>>(qa.xy));
                const float4 wa_hi = float4(as_type<vec<bfloat, 4>>(qa.zw));
                const float4 wb_lo = float4(as_type<vec<bfloat, 4>>(qb.xy));
                const float4 wb_hi = float4(as_type<vec<bfloat, 4>>(qb.zw));
                #pragma clang loop unroll(full)
                for (int m = 0; m < FC_M; ++m) {
                    const device float4* hm = h4 + size_t(m) * K4 + idx * 2;
                    const float4 h_lo = hm[0], h_hi = hm[1];
                    acc0[m] += dot(wa_lo, h_lo) + dot(wa_hi, h_hi);
                    acc1[m] += dot(wb_lo, h_lo) + dot(wb_hi, h_hi);
                }
            }
        }
        #pragma clang loop unroll(full)
        for (int m = 0; m < FC_M; ++m) {
            const float t0 = simd_sum(acc0[m]);
            const float t1 = simd_sum(acc1[m]);
            if (lane == 0) {
                float bvm = t0;
                uint  bim = uint(row0);
                if (has_r1 && t1 > bvm) {
                    bvm = t1;
                    bim = uint(row1);
                }
                sg_v[sg][m] = bvm;
                sg_i[sg][m] = bim;
            }
        }
    } else {
        float bv[8];
        uint  bi[8];
        #pragma clang loop unroll(full)
        for (int m = 0; m < FC_M; ++m) { bv[m] = -INFINITY; bi[m] = 0u; }
        #pragma clang loop unroll(full)
        for (int ri = 0; ri < FC_R; ++ri) {
            const int v = row0 + ri;
            if (v < MODEL_V_LIMIT) {
                const device uint4* w8 = (const device uint4*)(w + size_t(v) * MODEL_D);
                float acc[8];
                #pragma clang loop unroll(full)
                for (int m = 0; m < FC_M; ++m) acc[m] = 0.0f;
                for (int idx = int(lane); idx < K8; idx += 32) {
                    const uint4 q = w8[idx];
                    const float4 w_lo = float4(as_type<vec<bfloat, 4>>(q.xy));
                    const float4 w_hi = float4(as_type<vec<bfloat, 4>>(q.zw));
                    #pragma clang loop unroll(full)
                    for (int m = 0; m < FC_M; ++m) {
                        const device float4* hm = h4 + size_t(m) * K4 + idx * 2;
                        acc[m] += dot(w_lo, hm[0]) + dot(w_hi, hm[1]);
                    }
                }
                #pragma clang loop unroll(full)
                for (int m = 0; m < FC_M; ++m) {
                    const float tot = simd_sum(acc[m]);
                    if (lane == 0 && tot > bv[m]) {
                        bv[m] = tot;
                        bi[m] = uint(v);
                    }
                }
            }
        }
        if (lane == 0) {
            #pragma clang loop unroll(full)
            for (int m = 0; m < FC_M; ++m) {
                sg_v[sg][m] = bv[m];
                sg_i[sg][m] = bi[m];
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && int(lane) < FC_M) {
        const int m = int(lane);
        float fv = sg_v[0][m];
        uint  fi = sg_i[0][m];
        #pragma clang loop unroll(full)
        for (int s = 1; s < SG; ++s) {
            const float sv = sg_v[s][m];
            const uint  si = sg_i[s][m];
            if (sv > fv || (sv == fv && si < fi)) {
                fv = sv;
                fi = si;
            }
        }
        part_val[size_t(m) * p.NTG + tg_id] = fv;
        part_idx[size_t(m) * p.NTG + tg_id] = fi;
    }
}

struct ArgmaxP2Prm {
    int NTG;
};

kernel void inkling_head_argmax_p2(
    const device float* part_val [[buffer(0)]],
    const device uint* part_idx  [[buffer(1)]],
    device int* out_tok          [[buffer(2)]],
    constant ArgmaxP2Prm& p      [[buffer(3)]],
    uint2 tg_pos                 [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]],
    uint lane                    [[thread_index_in_simdgroup]],
    uint sg                      [[simdgroup_index_in_threadgroup]])
{
    const int m = int(tg_pos.y);
    const size_t base = size_t(m) * p.NTG;
    float bv = -INFINITY;
    uint bi = 0u;
    for (int i = int(tid); i < p.NTG; i += 256) {
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
        float fv = sg_v[0];
        uint fi = sg_i[0];
        for (int s = 1; s < 8; ++s) {
            if (sg_v[s] > fv || (sg_v[s] == fv && sg_i[s] < fi)) { fv = sg_v[s]; fi = sg_i[s]; }
        }
        out_tok[m] = int(fi);
    }
}

struct LogitsMaxP1Prm {
    int NTG;
    float temperature;
    float softcap;
};

inline float4 round_bf16_4(float4 x) {
    return float4(float(bfloat(x.x)), float(bfloat(x.y)), float(bfloat(x.z)), float(bfloat(x.w)));
}

kernel void inkling_head_logits_max_p1(
    const device float* h     [[buffer(0)]],
    const device bfloat* w    [[buffer(1)]],
    device float* out_logits  [[buffer(2)]],
    device float* part_max    [[buffer(3)]],
    constant LogitsMaxP1Prm& p[[buffer(4)]],
    uint tg_x                 [[threadgroup_position_in_grid]],
    uint lane                 [[thread_index_in_simdgroup]],
    uint sg                   [[simdgroup_index_in_threadgroup]])
{
    constexpr int SG = 8;
    constexpr int K8 = MODEL_D / 8;
    constexpr int K4 = MODEL_D / 4;
    const int tg_id = int(tg_x);
    const int v_base = tg_id * (SG * 2);
    const device float4* h4 = (const device float4*)(h);

    threadgroup float sg_v[SG][8];

    const int row0 = v_base + int(sg) * 2;
    if (row0 >= MODEL_V_LIMIT) {
        if (lane == 0) {
            #pragma clang loop unroll(full)
            for (int m = 0; m < FC_M; ++m) {
                sg_v[sg][m] = -INFINITY;
            }
        }
    } else {
        const int row1 = min(row0 + 1, MODEL_V_LIMIT - 1);
        const bool has_r1 = (row0 + 1) < MODEL_V_LIMIT;
        const device uint4* wa8 = (const device uint4*)(w + size_t(row0) * MODEL_D);
        const device uint4* wb8 = (const device uint4*)(w + size_t(row1) * MODEL_D);
        float acc0[8], acc1[8];
        #pragma clang loop unroll(full)
        for (int m = 0; m < FC_M; ++m) { acc0[m] = 0.0f; acc1[m] = 0.0f; }

        if (K8 % 96 == 0 && FC_M <= 4) {
            for (int base = 0; base < K8; base += 96) {
                const int i0 = base + int(lane);
                const int i1 = i0 + 32;
                const int i2 = i0 + 64;
                const uint4 qa0 = wa8[i0], qa1 = wa8[i1], qa2 = wa8[i2];
                const uint4 qb0 = wb8[i0], qb1 = wb8[i1], qb2 = wb8[i2];
                {
                    const float4 wa_lo = float4(as_type<vec<bfloat, 4>>(qa0.xy));
                    const float4 wa_hi = float4(as_type<vec<bfloat, 4>>(qa0.zw));
                    const float4 wb_lo = float4(as_type<vec<bfloat, 4>>(qb0.xy));
                    const float4 wb_hi = float4(as_type<vec<bfloat, 4>>(qb0.zw));
                    #pragma clang loop unroll(full)
                    for (int m = 0; m < FC_M; ++m) {
                        const device float4* hm = h4 + size_t(m) * K4 + i0 * 2;
                        const float4 h_lo = round_bf16_4(hm[0]);
                        const float4 h_hi = round_bf16_4(hm[1]);
                        acc0[m] += dot(wa_lo, h_lo) + dot(wa_hi, h_hi);
                        acc1[m] += dot(wb_lo, h_lo) + dot(wb_hi, h_hi);
                    }
                }
                {
                    const float4 wa_lo = float4(as_type<vec<bfloat, 4>>(qa1.xy));
                    const float4 wa_hi = float4(as_type<vec<bfloat, 4>>(qa1.zw));
                    const float4 wb_lo = float4(as_type<vec<bfloat, 4>>(qb1.xy));
                    const float4 wb_hi = float4(as_type<vec<bfloat, 4>>(qb1.zw));
                    #pragma clang loop unroll(full)
                    for (int m = 0; m < FC_M; ++m) {
                        const device float4* hm = h4 + size_t(m) * K4 + i1 * 2;
                        const float4 h_lo = round_bf16_4(hm[0]);
                        const float4 h_hi = round_bf16_4(hm[1]);
                        acc0[m] += dot(wa_lo, h_lo) + dot(wa_hi, h_hi);
                        acc1[m] += dot(wb_lo, h_lo) + dot(wb_hi, h_hi);
                    }
                }
                {
                    const float4 wa_lo = float4(as_type<vec<bfloat, 4>>(qa2.xy));
                    const float4 wa_hi = float4(as_type<vec<bfloat, 4>>(qa2.zw));
                    const float4 wb_lo = float4(as_type<vec<bfloat, 4>>(qb2.xy));
                    const float4 wb_hi = float4(as_type<vec<bfloat, 4>>(qb2.zw));
                    #pragma clang loop unroll(full)
                    for (int m = 0; m < FC_M; ++m) {
                        const device float4* hm = h4 + size_t(m) * K4 + i2 * 2;
                        const float4 h_lo = round_bf16_4(hm[0]);
                        const float4 h_hi = round_bf16_4(hm[1]);
                        acc0[m] += dot(wa_lo, h_lo) + dot(wa_hi, h_hi);
                        acc1[m] += dot(wb_lo, h_lo) + dot(wb_hi, h_hi);
                    }
                }
            }
        } else {
            for (int idx = int(lane); idx < K8; idx += 32) {
                const uint4 qa = wa8[idx];
                const uint4 qb = wb8[idx];
                const float4 wa_lo = float4(as_type<vec<bfloat, 4>>(qa.xy));
                const float4 wa_hi = float4(as_type<vec<bfloat, 4>>(qa.zw));
                const float4 wb_lo = float4(as_type<vec<bfloat, 4>>(qb.xy));
                const float4 wb_hi = float4(as_type<vec<bfloat, 4>>(qb.zw));
                #pragma clang loop unroll(full)
                for (int m = 0; m < FC_M; ++m) {
                    const device float4* hm = h4 + size_t(m) * K4 + idx * 2;
                    const float4 h_lo = round_bf16_4(hm[0]);
                    const float4 h_hi = round_bf16_4(hm[1]);
                    acc0[m] += dot(wa_lo, h_lo) + dot(wa_hi, h_hi);
                    acc1[m] += dot(wb_lo, h_lo) + dot(wb_hi, h_hi);
                }
            }
        }
        #pragma clang loop unroll(full)
        for (int m = 0; m < FC_M; ++m) {
            const float t0 = simd_sum(acc0[m]);
            const float t1 = simd_sum(acc1[m]);
            if (lane == 0) {
                float v0 = t0;
                if (p.softcap > 0.0f) v0 = p.softcap * metal::precise::tanh(v0 / p.softcap);
                v0 /= p.temperature;
                out_logits[size_t(m) * size_t(MODEL_V_LIMIT) + size_t(row0)] = v0;
                float bvm = v0;
                if (has_r1) {
                    float v1 = t1;
                    if (p.softcap > 0.0f) v1 = p.softcap * metal::precise::tanh(v1 / p.softcap);
                    v1 /= p.temperature;
                    out_logits[size_t(m) * size_t(MODEL_V_LIMIT) + size_t(row1)] = v1;
                    bvm = max(bvm, v1);
                }
                sg_v[sg][m] = bvm;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && int(lane) < FC_M) {
        const int m = int(lane);
        float fv = sg_v[0][m];
        #pragma clang loop unroll(full)
        for (int s = 1; s < SG; ++s) {
            fv = max(fv, sg_v[s][m]);
        }
        part_max[size_t(m) * p.NTG + tg_id] = fv;
    }
}

struct BatchMTPStreamMetaGpu {
    int32_t anchor_token;
    int32_t n_ctx;
    int32_t ctx_start;
    int32_t num_draft;
    int32_t d_start;
    int32_t v_start;
    int32_t is_sampled;
    int32_t do_top_p;
    float   temperature;
    float   top_p;
    float   softcap;
    float   _pad0;
    ulong   rng_seed;
    ulong   rng_step;
    ulong   out_hidden_addr;
};

struct BatchMTPTargetVisitMetaGpu {
    ulong   kv_k_addr;
    ulong   kv_v_addr;
    ulong   conv_addr[4];
    ulong   hist_ks_addr;
    ulong   hist_vs_addr;
    ulong   hist_cs_addr;
    ulong   hist_ms_addr;
    ulong   stage_k_addr;
    ulong   stage_v_addr;
    int32_t prev_offset;
    int32_t old_idx;
    int32_t max_size;
    int32_t is_rotating;
    int32_t need_staging;
    int32_t write_base;
    int32_t s_kv;
    int32_t ext_m;
    int32_t ring_idx;
    int32_t stride_h;
    int32_t stride_s;
    int32_t v_start;
    int32_t M_b;
    int32_t n_kv;
    int32_t head_dim;
    int32_t _pad;
};

kernel void inkling_final_rms_norm_bf16_cap(
    const device float* pre_norm [[buffer(0)]],
    const device bfloat* norm_w  [[buffer(1)]],
    device bfloat* out_bf16      [[buffer(2)]],
    device bfloat* out_cap_bf16  [[buffer(3)]],
    constant FinalNormPrm& p     [[buffer(4)]],
    uint2 tg_pos                 [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]],
    uint lane                    [[thread_index_in_simdgroup]],
    uint sg                      [[simdgroup_index_in_threadgroup]])
{
    constexpr int TG = MODEL_MIX_TG;
    constexpr int NS = TG / 32;
    constexpr int DPT = MODEL_MIX_DPT;
    const int row = int(tg_pos.y);
    threadgroup float buf[NS];
    threadgroup float red;

    float v[DPT];
    float s2 = 0.0f;
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) {
        const int d = int(tid) + c * TG;
        float val = d < MODEL_D ? pre_norm[size_t(row) * MODEL_D + d] : 0.0f;
        v[c] = val;
        s2 += val * val;
    }
    const float ss = simd_sum(s2);
    if (lane == 0) buf[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float tot = simd_sum(int(lane) < NS ? buf[lane] : 0.0f);
        if (lane == 0) red = tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = metal::precise::rsqrt(red / float(MODEL_D) + p.eps);
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) {
        const int d = int(tid) + c * TG;
        if (d < MODEL_D) {
            const bfloat bv = bfloat(v[c] * inv * float(norm_w[d]));
            out_bf16[size_t(row) * MODEL_D + d] = bv;
            if (p.cap_slot >= 0) {
                out_cap_bf16[(size_t(row) * p.n_cap + p.cap_slot) * MODEL_D + d] = bv;
            }
        }
    }
}

kernel void inkling_head_logits_max_mixed_p1(
    const device float* h                    [[buffer(0)]],
    const device bfloat* w                   [[buffer(1)]],
    device float* out_logits                 [[buffer(2)]],
    device float* part_max                   [[buffer(3)]],
    device float* part_val                   [[buffer(4)]],
    device uint* part_idx                    [[buffer(5)]],
    const device int* tok_to_stream          [[buffer(6)]],
    const device BatchMTPStreamMetaGpu* sm   [[buffer(7)]],
    constant ArgmaxP1Prm& p                  [[buffer(8)]],
    uint tg_x                                [[threadgroup_position_in_grid]],
    uint lane                                [[thread_index_in_simdgroup]],
    uint sg                                  [[simdgroup_index_in_threadgroup]])
{
    constexpr int SG = 8;
    constexpr int K8 = MODEL_D / 8;
    constexpr int K4 = MODEL_D / 4;
    const int tg_id = int(tg_x);
    const int v_base = tg_id * (SG * 2);
    const device float4* h4 = (const device float4*)(h);

    threadgroup float sg_v[SG][8];
    threadgroup uint  sg_i[SG][8];

    bool is_s[8];
    float temps[8], caps[8];
    #pragma clang loop unroll(full)
    for (int m = 0; m < FC_M; ++m) {
        const BatchMTPStreamMetaGpu meta = sm[tok_to_stream[m]];
        is_s[m]  = (meta.is_sampled != 0);
        temps[m] = meta.temperature;
        caps[m]  = meta.softcap;
    }

    const int row0 = v_base + int(sg) * 2;
    if (row0 >= MODEL_V_LIMIT) {
        if (lane == 0) {
            #pragma clang loop unroll(full)
            for (int m = 0; m < FC_M; ++m) {
                sg_v[sg][m] = -INFINITY;
                sg_i[sg][m] = 0u;
            }
        }
    } else {
        const int row1 = min(row0 + 1, MODEL_V_LIMIT - 1);
        const bool has_r1 = (row0 + 1) < MODEL_V_LIMIT;
        const device uint4* wa8 = (const device uint4*)(w + size_t(row0) * MODEL_D);
        const device uint4* wb8 = (const device uint4*)(w + size_t(row1) * MODEL_D);
        float acc0[8], acc1[8];
        #pragma clang loop unroll(full)
        for (int m = 0; m < FC_M; ++m) { acc0[m] = 0.0f; acc1[m] = 0.0f; }

        for (int idx = int(lane); idx < K8; idx += 32) {
            const uint4 qa = wa8[idx];
            const uint4 qb = wb8[idx];
            const float4 wa_lo = float4(as_type<vec<bfloat, 4>>(qa.xy));
            const float4 wa_hi = float4(as_type<vec<bfloat, 4>>(qa.zw));
            const float4 wb_lo = float4(as_type<vec<bfloat, 4>>(qb.xy));
            const float4 wb_hi = float4(as_type<vec<bfloat, 4>>(qb.zw));
            #pragma clang loop unroll(full)
            for (int m = 0; m < FC_M; ++m) {
                const device float4* hm = h4 + size_t(m) * K4 + idx * 2;
                const float4 h_lo = is_s[m] ? round_bf16_4(hm[0]) : hm[0];
                const float4 h_hi = is_s[m] ? round_bf16_4(hm[1]) : hm[1];
                acc0[m] += dot(wa_lo, h_lo) + dot(wa_hi, h_hi);
                acc1[m] += dot(wb_lo, h_lo) + dot(wb_hi, h_hi);
            }
        }
        #pragma clang loop unroll(full)
        for (int m = 0; m < FC_M; ++m) {
            const float t0 = simd_sum(acc0[m]);
            const float t1 = simd_sum(acc1[m]);
            if (lane == 0) {
                if (is_s[m]) {
                    float v0 = t0;
                    if (caps[m] > 0.0f) v0 = caps[m] * metal::precise::tanh(v0 / caps[m]);
                    v0 /= temps[m];
                    out_logits[size_t(m) * size_t(MODEL_V_LIMIT) + size_t(row0)] = v0;
                    float bvm = v0;
                    if (has_r1) {
                        float v1 = t1;
                        if (caps[m] > 0.0f) v1 = caps[m] * metal::precise::tanh(v1 / caps[m]);
                        v1 /= temps[m];
                        out_logits[size_t(m) * size_t(MODEL_V_LIMIT) + size_t(row1)] = v1;
                        bvm = max(bvm, v1);
                    }
                    sg_v[sg][m] = bvm;
                } else {
                    float bvm = t0;
                    uint  bim = uint(row0);
                    if (has_r1 && t1 > bvm) {
                        bvm = t1;
                        bim = uint(row1);
                    }
                    sg_v[sg][m] = bvm;
                    sg_i[sg][m] = bim;
                }
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && int(lane) < FC_M) {
        const int m = int(lane);
        if (is_s[m]) {
            float fv = sg_v[0][m];
            #pragma clang loop unroll(full)
            for (int s = 1; s < SG; ++s) fv = max(fv, sg_v[s][m]);
            part_max[size_t(m) * p.NTG + tg_id] = fv;
        } else {
            float fv = sg_v[0][m];
            uint  fi = sg_i[0][m];
            #pragma clang loop unroll(full)
            for (int s = 1; s < SG; ++s) {
                const float sv = sg_v[s][m];
                const uint  si = sg_i[s][m];
                if (sv > fv || (sv == fv && si < fi)) { fv = sv; fi = si; }
            }
            part_val[size_t(m) * p.NTG + tg_id] = fv;
            part_idx[size_t(m) * p.NTG + tg_id] = fi;
        }
    }
}

kernel void inkling_mix_mtp_batched(
    const device bfloat* x                          [[buffer(0)]],
    const device float* blocks                      [[buffer(1)]],
    const device float* queries                     [[buffer(2)]],
    const device float* taps                        [[buffer(3)]],
    const device bfloat* res                        [[buffer(4)]],
    const device float* messages                    [[buffer(5)]],
    const device bfloat* down                       [[buffer(6)]],
    const device float* forget                      [[buffer(7)]],
    const device float* gain                        [[buffer(8)]],
    const device bfloat* norm_w                     [[buffer(9)]],
    device void* out_raw                            [[buffer(10)]],
    device float* partial                           [[buffer(11)]],
    device float* resid                             [[buffer(12)]],
    const device int* tok_to_stream                 [[buffer(13)]],
    const device int* tok_local_pos                 [[buffer(14)]],
    const device BatchMTPTargetVisitMetaGpu* vmeta  [[buffer(15)]],
    constant MixPrm& p                              [[buffer(16)]],
    threadgroup float* tg_on                        [[threadgroup(0)]],
    threadgroup float* tg_mn                        [[threadgroup(1)]],
    uint2 tg_pos                                    [[threadgroup_position_in_grid]],
    uint tid                                        [[thread_index_in_threadgroup]],
    uint lane                                       [[thread_index_in_simdgroup]],
    uint sg                                         [[simdgroup_index_in_threadgroup]])
{
    constexpr int TG = MODEL_MIX_TG;
    constexpr int NS = TG / 32;
    constexpr int DPT = MODEL_MIX_DPT;
    const int NSRC = FC_K + 1;
    const uint j = tg_pos.x;
    const uint row = tg_pos.y;
    const int jq = (p.Q == 1) ? 0 : int(j);

    const BatchMTPTargetVisitMetaGpu vm = vmeta[tok_to_stream[row]];
    const int r_loc = tok_local_pos[row];
    const device float* state = (const device float*)(vm.conv_addr[2]);
    device float* new_state = (device float*)(vm.hist_cs_addr);

    threadgroup float buf[MAX_NRED][NS];
    threadgroup float red[MAX_NRED];

    float v[DPT];
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) {
        const int d = int(tid) + c * TG;
        float val = 0.0f;
        if (d < MODEL_D) {
            val = float(x[size_t(row) * MODEL_D + d]);
            const size_t so = size_t(row) * 3 * MODEL_D + d;
            const size_t to = size_t(d) * 4;
            const float s0 = r_loc >= 3 ? float(x[size_t(int(row) - 3) * MODEL_D + d]) : state[size_t(r_loc) * MODEL_D + d];
            const float s1 = r_loc >= 2 ? float(x[size_t(int(row) - 2) * MODEL_D + d]) : state[size_t(r_loc + 1) * MODEL_D + d];
            const float s2 = r_loc >= 1 ? float(x[size_t(int(row) - 1) * MODEL_D + d]) : state[size_t(r_loc + 2) * MODEL_D + d];
            float h = (taps[to] * s0 + taps[to + 1] * s1 + taps[to + 2] * s2 + taps[to + 3] * val + val) * 0.4472135955f;
            h = round_bf16(h);
            new_state[so] = s1;
            new_state[so + MODEL_D] = s2;
            new_state[so + 2 * MODEL_D] = val;
            val = p.ADD ? float(res[size_t(row) * MODEL_D + d]) + h : h;
            resid[size_t(row) * MODEL_D + d] = val;
        }
        v[c] = val;
    }

    float one[1], tot1[1];
    one[0] = 0.0f;
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) one[0] += v[c] * v[c];
    MIX_REDUCE2(1, one, tot1)
    const float inv = metal::precise::rsqrt(tot1[0] / float(MODEL_D) + 1e-6f);
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) {
        const int d = int(tid) + c * TG;
        v[c] *= inv;
        if (d < MODEL_D && j == 0) partial[size_t(row) * MODEL_D + d] = v[c];
    }

    float acc[MAX_NRED], w[MAX_NRED];
    #pragma clang loop unroll(full)
    for (int i = 0; i < NSRC; i++) acc[i] = 0.0f;
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) {
        const int d = int(tid) + c * TG;
        if (d < MODEL_D) {
            const float qv = queries[size_t(jq) * MODEL_D + d];
            #pragma clang loop unroll(full)
            for (int i = 0; i < NSRC; i++) {
                const float sv = (i < FC_K) ? blocks[(size_t(i) * p.M + row) * MODEL_D + d] : v[c];
                acc[i] += sv * qv;
            }
        }
    }
    MIX_REDUCE2(NSRC, acc, w)
    const float sc = metal::precise::rsqrt(float(MODEL_D));
    float mw = -INFINITY, zw = 0.0f;
    #pragma clang loop unroll(full)
    for (int i = 0; i < NSRC; i++) mw = max(mw, w[i] * sc);
    #pragma clang loop unroll(full)
    for (int i = 0; i < NSRC; i++) {
        w[i] = metal::precise::exp(w[i] * sc - mw);
        zw += w[i];
    }
    float o[DPT];
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) {
        const int d = int(tid) + c * TG;
        float s = 0.0f;
        if (d < MODEL_D) {
            #pragma clang loop unroll(full)
            for (int i = 0; i < NSRC; i++) {
                const float sv = (i < FC_K) ? blocks[(size_t(i) * p.M + row) * MODEL_D + d] : v[c];
                s += w[i] / zw * sv;
            }
        }
        o[c] = s;
    }

    if (FC_GATE) {
        constexpr int RANK = MODEL_RANK;
        constexpr int NCH = TG / RANK;
        constexpr int CD = (MODEL_D + NCH - 1) / NCH;
        threadgroup float pg[TG];
        threadgroup float low[RANK];
        float mm[DPT], two[2], tt[2];
        two[0] = 0.0f; two[1] = 0.0f;
        #pragma clang loop unroll(full)
        for (int c = 0; c < DPT; c++) {
            const int d = int(tid) + c * TG;
            mm[c] = d < MODEL_D ? messages[(size_t(row) * 4 + p.MOFF + j) * MODEL_D + d] : 0.0f;
            two[0] += o[c] * o[c];
            two[1] += mm[c] * mm[c];
        }
        MIX_REDUCE2(2, two, tt)
        const float io = metal::precise::rsqrt(tt[0] / float(MODEL_D) + 1e-6f);
        const float im = metal::precise::rsqrt(tt[1] / float(MODEL_D) + 1e-6f);
        #pragma clang loop unroll(full)
        for (int c = 0; c < DPT; c++) {
            const int d = int(tid) + c * TG;
            if (d < MODEL_D) {
                tg_on[d] = round_bf16(o[c] * io);
                tg_mn[d] = round_bf16(mm[c] * im);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int r = int(tid) % RANK;
        const int chunk = int(tid) / RANK;
        const size_t base = (size_t(j) * RANK + r) * 2 * MODEL_D;
        float gp = 0.0f;
        if (chunk < NCH) {
            const int d_end = min(MODEL_D, (chunk + 1) * CD);
            for (int d = chunk * CD; d < d_end; d++) {
                gp += float(down[base + d]) * tg_on[d] + float(down[base + MODEL_D + d]) * tg_mn[d];
            }
        }
        pg[tid] = gp;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (chunk == 0) {
            for (int c = 1; c < NCH; c++) gp += pg[c * RANK + r];
            low[r] = round_bf16(gp);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (int c = 0; c < DPT; c++) {
            const int d = int(tid) + c * TG;
            if (d < MODEL_D) {
                const size_t fb = (size_t(j) * MODEL_D + d) * RANK;
                float f = 0.0f, gn = 0.0f;
                for (int k = 0; k < RANK; k++) {
                    f += forget[fb + k] * low[k];
                    gn += gain[fb + k] * low[k];
                }
                o[c] = (1.0f - metal::precise::tanh(f)) * o[c] + metal::precise::tanh(gn) * mm[c];
            }
        }
    }

    device bfloat* out_bf = (device bfloat*)out_raw;
    float n2[1], nt[1];
    n2[0] = 0.0f;
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) n2[0] += o[c] * o[c];
    MIX_REDUCE2(1, n2, nt)
    const float invj = metal::precise::rsqrt(nt[0] / float(MODEL_D) + p.eps);
    #pragma clang loop unroll(full)
    for (int c = 0; c < DPT; c++) {
        const int d = int(tid) + c * TG;
        if (d < MODEL_D) {
            out_bf[(size_t(row) * p.QO + j) * MODEL_D + d] =
                bfloat(o[c] * invj * float(norm_w[size_t(j % uint(p.QN)) * MODEL_D + d]));
        }
    }
}

struct KvPrepMtpBatchedPrm {
    int M, HQ, HKV, R;
    int QKV_W, QOFF, ROFF, KOFF, VOFF;
    int X_STRIDE, XK_OFF, XV_OFF;
    int TAU;
    float qk_thr, rel_thr, eps, log_n_floor, log_alpha;
};

kernel void inkling_kv_prep_mtp_batched(
    const device bfloat* qkv                        [[buffer(0)]],
    const device bfloat* qkv_in                     [[buffer(1)]],
    const device bfloat* sink_w                     [[buffer(2)]],
    const device bfloat* kdyn_w                     [[buffer(3)]],
    const device bfloat* vdyn_w                     [[buffer(4)]],
    const device void* ktaps_v                      [[buffer(5)]],
    const device void* vtaps_v                      [[buffer(6)]],
    const device bfloat* qscale                     [[buffer(7)]],
    const device bfloat* rscale                     [[buffer(8)]],
    const device bfloat* proj                       [[buffer(9)]],
    device bfloat* qo                               [[buffer(10)]],
    device bfloat* bias                             [[buffer(11)]],
    device bfloat* sinks                            [[buffer(12)]],
    const device int* tok_to_stream                 [[buffer(13)]],
    const device int* tok_local_pos                 [[buffer(14)]],
    const device BatchMTPTargetVisitMetaGpu* vmeta  [[buffer(15)]],
    constant KvPrepMtpBatchedPrm& p                 [[buffer(16)]],
    uint2 tg_pos                                    [[threadgroup_position_in_grid]],
    uint t                                          [[thread_index_in_threadgroup]],
    uint lane                                       [[thread_index_in_simdgroup]],
    uint sg                                         [[simdgroup_index_in_threadgroup]])
{
    const int TG = max(32, ((FC_HD + 31) / 32) * 32);
    const int NS = TG / 32;
    const int C = p.HKV * FC_HD;
    const int h = int(tg_pos.x);
    const int row = int(tg_pos.y);

    const BatchMTPTargetVisitMetaGpu vm = vmeta[tok_to_stream[row]];
    const int r_loc = tok_local_pos[row];

    float tau_ = 1.0f;
    if (p.TAU) {
        const float pos = float(vm.prev_offset + 1 + r_loc);
        tau_ = 1.0f + p.log_alpha * metal::precise::log(max(pos / p.log_n_floor, 1.0f));
    }

    threadgroup float buf[9][16];
    threadgroup float rr[MODEL_DR];

    const device bfloat* xq_row = qkv_in + size_t(row) * p.X_STRIDE;
    const device bfloat* xk_row = qkv_in + size_t(row) * p.X_STRIDE + p.XK_OFF;
    const device bfloat* xv_row = qkv_in + size_t(row) * p.X_STRIDE + p.XV_OFF;

    float nine[9], dots[9];
    #pragma clang loop unroll(full)
    for (int i = 0; i < 9; i++) nine[i] = 0.0f;
    for (int d = int(t); d < MODEL_D; d += TG) {
        nine[0] += float(xq_row[d]) * float(sink_w[size_t(h) * MODEL_D + d]);
        if (FC_DYN && h < p.HKV) {
            const float kx = float(xk_row[d]);
            const float vx = float(xv_row[d]);
            #pragma clang loop unroll(full)
            for (int lag = 0; lag < 4; lag++) {
                nine[1 + lag] += kx * float(kdyn_w[size_t(h * 4 + lag) * MODEL_D + d]);
                nine[5 + lag] += vx * float(vdyn_w[size_t(h * 4 + lag) * MODEL_D + d]);
            }
        }
    }
    KV_REDUCE(FC_DYN ? 9 : 1, nine, dots)
    if (t == 0) sinks[size_t(row) * p.HQ + h] = bfloat(dots[0]);

    float four[4], tot[4];
    const float qy = int(t) < FC_HD ? float(qkv[size_t(row) * p.QKV_W + p.QOFF + h * FC_HD + int(t)]) * float(qscale[h * FC_HD + int(t)]) : 0.0f;
    const float ry = int(t) < MODEL_DR ? float(qkv[size_t(row) * p.QKV_W + p.ROFF + h * MODEL_DR + int(t)]) * float(rscale[h * MODEL_DR + int(t)]) : 0.0f;
    float ky = 0.0f, vy = 0.0f;
    if (h < p.HKV && int(t) < FC_HD) {
        const int c = h * FC_HD + int(t);
        const size_t so = size_t(row) * 3 * C + c;
        const device float* kstate = (const device float*)(vm.conv_addr[0]);
        const device float* vstate = (const device float*)(vm.conv_addr[1]);
        device float* kstate_new = (device float*)(vm.hist_ks_addr);
        device float* vstate_new = (device float*)(vm.hist_vs_addr);

        const float kx = float(qkv[size_t(row) * p.QKV_W + p.KOFF + c]);
        const float vx = float(qkv[size_t(row) * p.QKV_W + p.VOFF + c]);
        const float k0 = r_loc >= 3 ? float(qkv[size_t(row - 3) * p.QKV_W + p.KOFF + c]) : kstate[size_t(r_loc) * C + c];
        const float k1 = r_loc >= 2 ? float(qkv[size_t(row - 2) * p.QKV_W + p.KOFF + c]) : kstate[size_t(r_loc + 1) * C + c];
        const float k2 = r_loc >= 1 ? float(qkv[size_t(row - 1) * p.QKV_W + p.KOFF + c]) : kstate[size_t(r_loc + 2) * C + c];
        const float v0 = r_loc >= 3 ? float(qkv[size_t(row - 3) * p.QKV_W + p.VOFF + c]) : vstate[size_t(r_loc) * C + c];
        const float v1 = r_loc >= 2 ? float(qkv[size_t(row - 2) * p.QKV_W + p.VOFF + c]) : vstate[size_t(r_loc + 1) * C + c];
        const float v2 = r_loc >= 1 ? float(qkv[size_t(row - 1) * p.QKV_W + p.VOFF + c]) : vstate[size_t(r_loc + 2) * C + c];
        float kc, vc;
        if (FC_DYN) {
            const device bfloat* ktaps = (const device bfloat*)ktaps_v;
            const device bfloat* vtaps = (const device bfloat*)vtaps_v;
            kc = ((float(ktaps[c]) + round_bf16(dots[1])) * kx
                + (float(ktaps[C + c]) + round_bf16(dots[2])) * k2
                + (float(ktaps[2 * C + c]) + round_bf16(dots[3])) * k1
                + (float(ktaps[3 * C + c]) + round_bf16(dots[4])) * k0
                + kx) * 0.3333333333f;
            vc = ((float(vtaps[c]) + round_bf16(dots[5])) * vx
                + (float(vtaps[C + c]) + round_bf16(dots[6])) * v2
                + (float(vtaps[2 * C + c]) + round_bf16(dots[7])) * v1
                + (float(vtaps[3 * C + c]) + round_bf16(dots[8])) * v0
                + vx) * 0.3333333333f;
        } else {
            const device float* ktaps = (const device float*)ktaps_v;
            const device float* vtaps = (const device float*)vtaps_v;
            const size_t to = size_t(c) * 4;
            kc = (ktaps[to] * k0 + ktaps[to + 1] * k1 + ktaps[to + 2] * k2 + ktaps[to + 3] * kx + kx) * 0.4472135955f;
            vc = (vtaps[to] * v0 + vtaps[to + 1] * v1 + vtaps[to + 2] * v2 + vtaps[to + 3] * vx + vx) * 0.4472135955f;
        }
        ky = round_bf16(kc);
        vy = round_bf16(vc);
        kstate_new[so] = k1; kstate_new[so + C] = k2; kstate_new[so + 2 * C] = kx;
        vstate_new[so] = v1; vstate_new[so + C] = v2; vstate_new[so + 2 * C] = vx;
    }
    four[0] = qy * qy; four[1] = ry * ry; four[2] = ky * ky; four[3] = vy * vy;
    KV_REDUCE(4, four, tot)

    if (int(t) < FC_HD) {
        float qv = round_bf16(qy * min(p.qk_thr / (metal::precise::sqrt(tot[0]) + 1e-6f), 1.0f));
        if (p.TAU) qv = round_bf16(qv * tau_);
        qo[(size_t(row) * p.HQ + h) * FC_HD + int(t)] = bfloat(qv);
        if (h < p.HKV) {
            const bfloat kv_out = bfloat(ky * min(p.qk_thr / (metal::precise::sqrt(tot[2]) + 1e-6f), 1.0f));
            const bfloat vv_out = bfloat(vy * metal::precise::rsqrt(tot[3] / float(FC_HD) + p.eps));
            if (vm.need_staging == 0) {
                const int slot = (vm.is_rotating != 0) ? ((vm.write_base + r_loc) % vm.max_size) : (vm.write_base + r_loc);
                const size_t xo = size_t(h) * size_t(vm.stride_h) + size_t(slot) * size_t(vm.stride_s) + int(t);
                ((device bfloat*)(vm.kv_k_addr))[xo] = kv_out;
                ((device bfloat*)(vm.kv_v_addr))[xo] = vv_out;
            } else {
                const size_t xo = (size_t(row) * p.HKV + h) * FC_HD + int(t);
                ((device bfloat*)(vm.stage_k_addr))[xo] = kv_out;
                ((device bfloat*)(vm.stage_v_addr))[xo] = vv_out;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(t) < MODEL_DR) {
        float rv = round_bf16(ry * min(p.rel_thr / (metal::precise::sqrt(tot[1]) + 1e-6f), 1.0f));
        if (p.TAU) rv = round_bf16(rv * tau_);
        rr[t] = rv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int d = int(t); d < p.R; d += TG) {
        float b = 0.0f;
        #pragma clang loop unroll(full)
        for (int i = 0; i < MODEL_DR; i++) b += rr[i] * float(proj[size_t(i) * p.R + d]);
        bias[(size_t(row) * p.HQ + h) * p.R + d] = bfloat(b);
    }
}

struct AttendMtpBatchedPrm {
    int M, HQ, HKV, R, GW, GOFF, WINDOW;
    float scale;
};

kernel void inkling_attend_mtp_batched(
    const device bfloat* q                          [[buffer(0)]],
    const device bfloat* bias                       [[buffer(1)]],
    const device bfloat* sinks                      [[buffer(2)]],
    const device bfloat* gate                       [[buffer(3)]],
    device bfloat* out                              [[buffer(4)]],
    const device int* tok_to_stream                 [[buffer(5)]],
    const device int* tok_local_pos                 [[buffer(6)]],
    const device BatchMTPTargetVisitMetaGpu* vmeta  [[buffer(7)]],
    constant AttendMtpBatchedPrm& p                 [[buffer(8)]],
    threadgroup float* p_buf                        [[threadgroup(0)]],
    uint2 tg_pos                                    [[threadgroup_position_in_grid]],
    uint t                                          [[thread_index_in_threadgroup]],
    uint lane                                       [[thread_index_in_simdgroup]],
    uint sg                                         [[simdgroup_index_in_threadgroup]])
{
    const int TG = min(1024, ((FC_HD * 8 + 31) / 32) * 32);
    const int NS = TG / 32;
    const int CH = TG / FC_HD;
    const int CK = (TG + CH - 1) / CH;
    const int kvh = int(tg_pos.x);
    const int row = int(tg_pos.y);

    const BatchMTPTargetVisitMetaGpu vm = vmeta[tok_to_stream[row]];
    const int r_loc = tok_local_pos[row];

    threadgroup float buf[MAX_G][32];
    threadgroup float red[MAX_G];

    const device bfloat* kp = (const device bfloat*)(vm.kv_k_addr) + size_t(kvh) * size_t(vm.stride_h);
    const device bfloat* vp = (const device bfloat*)(vm.kv_v_addr) + size_t(kvh) * size_t(vm.stride_h);
    const device bfloat* ko_ext = (const device bfloat*)(vm.stage_k_addr);
    const device bfloat* vo_ext = (const device bfloat*)(vm.stage_v_addr);
    const size_t qb = (size_t(row) * p.HQ + kvh * FC_G) * FC_HD;

    const int S_total = vm.s_kv + vm.ext_m;
    const int last = S_total - vm.M_b + r_loc;
    const int first_valid = p.WINDOW > 0 ? max(0, last - p.WINDOW + 1) : 0;
    const int n_valid = max(0, last - first_valid + 1);
    const int place = first_valid + int(t);
    const bool seen = int(t) < n_valid;

    float sc[MAX_G], m[MAX_G], z[MAX_G];
    #pragma clang loop unroll(full)
    for (int g = 0; g < FC_G; g++) sc[g] = -INFINITY;
    if (seen) {
        const device bfloat* ks;
        if (place < vm.s_kv) {
            const int phys = (vm.ring_idx == 0) ? place : ((place + vm.ring_idx) % vm.s_kv);
            ks = kp + size_t(phys) * size_t(vm.stride_s);
        } else {
            ks = ko_ext + (size_t(vm.v_start + (place - vm.s_kv)) * p.HKV + kvh) * FC_HD;
        }
        #pragma clang loop unroll(full)
        for (int g = 0; g < FC_G; g++) sc[g] = 0.0f;
        for (int d = 0; d < FC_HD; d++) {
            const float kd = float(ks[d]);
            #pragma clang loop unroll(full)
            for (int g = 0; g < FC_G; g++) sc[g] += float(q[qb + g * FC_HD + d]) * kd;
        }
        const int dist = last - place;
        #pragma clang loop unroll(full)
        for (int g = 0; g < FC_G; g++) {
            sc[g] = sc[g] * p.scale
                  + (dist < p.R ? float(bias[(size_t(row) * p.HQ + kvh * FC_G + g) * p.R + dist]) : 0.0f);
        }
    }
    #pragma clang loop unroll(full)
    for (int g = 0; g < FC_G; g++) {
        const float v = simd_max(sc[g]);
        if (lane == 0) buf[g][sg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        #pragma clang loop unroll(full)
        for (int g = 0; g < FC_G; g++) {
            const float v = simd_max(int(lane) < NS ? buf[g][lane] : -INFINITY);
            if (lane == 0) red[g] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (int g = 0; g < FC_G; g++) {
        m[g] = max(red[g], float(sinks[size_t(row) * p.HQ + kvh * FC_G + g]));
        const float e = seen ? metal::precise::exp(sc[g] - m[g]) : 0.0f;
        p_buf[g * TG + int(t)] = e;
        z[g] = simd_sum(e);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (int g = 0; g < FC_G; g++) {
        if (lane == 0) buf[g][sg] = z[g];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        #pragma clang loop unroll(full)
        for (int g = 0; g < FC_G; g++) {
            const float v = simd_sum(int(lane) < NS ? buf[g][lane] : 0.0f);
            if (lane == 0) red[g] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (int g = 0; g < FC_G; g++) {
        z[g] = red[g] + metal::precise::exp(float(sinks[size_t(row) * p.HQ + kvh * FC_G + g]) - m[g]);
    }

    const int d = int(t) % FC_HD;
    const int ch = int(t) / FC_HD;
    float o[MAX_G];
    #pragma clang loop unroll(full)
    for (int g = 0; g < FC_G; g++) o[g] = 0.0f;
    const int i_end = min(n_valid, (ch + 1) * CK);
    for (int i = ch * CK; i < i_end; i++) {
        const int p_idx = first_valid + i;
        float vd;
        if (p_idx < vm.s_kv) {
            const int phys = (vm.ring_idx == 0) ? p_idx : ((p_idx + vm.ring_idx) % vm.s_kv);
            vd = float(vp[size_t(phys) * size_t(vm.stride_s) + d]);
        } else {
            vd = float(vo_ext[(size_t(vm.v_start + (p_idx - vm.s_kv)) * p.HKV + kvh) * FC_HD + d]);
        }
        #pragma clang loop unroll(full)
        for (int g = 0; g < FC_G; g++) o[g] += p_buf[g * TG + i] * vd;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (int g = 0; g < FC_G; g++) p_buf[g * TG + int(t)] = o[g];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ch == 0) {
        #pragma clang loop unroll(full)
        for (int g = 0; g < FC_G; g++) {
            float v = o[g];
            for (int c = 1; c < CH; c++) v += p_buf[g * TG + c * FC_HD + d];
            v = round_bf16(v / z[g]);
            const size_t at = size_t(kvh * FC_G + g) * FC_HD + d;
            out[size_t(row) * p.HQ * FC_HD + at] =
                bfloat(v * sigmoid_(float(gate[size_t(row) * p.GW + p.GOFF + at])));
        }
    }
}

struct DecodePartMtpBatchedPrm {
    int M, HQ, HKV, REL, BS, nblocks, WINDOW;
    float scale;
};

kernel void inkling_decode_partials_mtp_batched(
    const device bfloat* q                          [[buffer(0)]],
    const device bfloat* rel                        [[buffer(1)]],
    device float* part_max                          [[buffer(2)]],
    device float* part_sum                          [[buffer(3)]],
    device float* part_acc                          [[buffer(4)]],
    const device int* tok_to_stream                 [[buffer(5)]],
    const device int* tok_local_pos                 [[buffer(6)]],
    const device BatchMTPTargetVisitMetaGpu* vmeta  [[buffer(7)]],
    constant DecodePartMtpBatchedPrm& p             [[buffer(8)]],
    threadgroup float* ta                           [[threadgroup(0)]],
    uint3 tg_pos                                    [[threadgroup_position_in_grid]],
    uint tid                                        [[thread_index_in_threadgroup]],
    uint lane                                       [[thread_index_in_simdgroup]],
    uint sg                                         [[simdgroup_index_in_threadgroup]])
{
    constexpr int SG = 8;
    const int E = FC_HD / 32;
    const int block = int(tg_pos.x);
    const int g = int(tg_pos.y);
    const int batch = int(tg_pos.z);

    const BatchMTPTargetVisitMetaGpu vm = vmeta[tok_to_stream[batch]];
    const int r_loc = tok_local_pos[batch];

    const int S_total = vm.s_kv + vm.ext_m;
    const int last = S_total - vm.M_b + r_loc;
    const int ks = p.WINDOW > 0 ? max(0, last - p.WINDOW + 1) : 0;
    const int start = ks + block * p.BS;
    const int end = min(start + p.BS, last + 1);

    float qv[MAX_G][4];
    const size_t q_base = (size_t(batch) * p.HQ + g * FC_G) * FC_HD + int(lane) * E;
    #pragma clang loop unroll(full)
    for (int h = 0; h < FC_G; ++h)
        #pragma clang loop unroll(full)
        for (int e = 0; e < E; ++e)
            qv[h][e] = float(q[q_base + size_t(h) * FC_HD + e]) * p.scale;

    float m[MAX_G], l[MAX_G], acc[MAX_G][4];
    #pragma clang loop unroll(full)
    for (int h = 0; h < FC_G; ++h) {
        m[h] = -INFINITY;
        l[h] = 0.0f;
        #pragma clang loop unroll(full)
        for (int e = 0; e < E; ++e) acc[h][e] = 0.0f;
    }

    const device bfloat* k_head = (const device bfloat*)(vm.kv_k_addr) + size_t(g) * size_t(vm.stride_h);
    const device bfloat* v_head = (const device bfloat*)(vm.kv_v_addr) + size_t(g) * size_t(vm.stride_h);
    const device bfloat* ko_ext = (const device bfloat*)(vm.stage_k_addr);
    const device bfloat* vo_ext = (const device bfloat*)(vm.stage_v_addr);
    const size_t rel_base = (size_t(batch) * p.HQ + g * FC_G) * p.REL;

    for (int j = start + int(sg); j < end; j += SG) {
        const device bfloat* kp;
        const device bfloat* vp;
        if (j < vm.s_kv) {
            const int phys = (vm.ring_idx == 0) ? j : ((j + vm.ring_idx) % vm.s_kv);
            kp = k_head + size_t(phys) * size_t(vm.stride_s) + int(lane) * E;
            vp = v_head + size_t(phys) * size_t(vm.stride_s) + int(lane) * E;
        } else {
            const size_t eo = (size_t(vm.v_start + (j - vm.s_kv)) * p.HKV + g) * FC_HD + int(lane) * E;
            kp = ko_ext + eo;
            vp = vo_ext + eo;
        }
        float kv[4], vv[4];
        #pragma clang loop unroll(full)
        for (int e = 0; e < E; ++e) {
            kv[e] = float(kp[e]);
            vv[e] = float(vp[e]);
        }
        const int dist = last - j;
        #pragma clang loop unroll(full)
        for (int h = 0; h < FC_G; ++h) {
            float s = 0.0f;
            #pragma clang loop unroll(full)
            for (int e = 0; e < E; ++e) s += qv[h][e] * kv[e];
            s = simd_sum(s);
            if (dist < p.REL) s += float(rel[rel_base + size_t(h) * p.REL + dist]);
            const float mn = max(m[h], s);
            const float f = metal::exp(m[h] - mn);
            const float pv = metal::exp(s - mn);
            l[h] = l[h] * f + pv;
            #pragma clang loop unroll(full)
            for (int e = 0; e < E; ++e) acc[h][e] = acc[h][e] * f + pv * vv[e];
            m[h] = mn;
        }
    }

    threadgroup float tm[8][MAX_G];
    threadgroup float tl[8][MAX_G];
    #pragma clang loop unroll(full)
    for (int h = 0; h < FC_G; ++h) {
        if (lane == 0) {
            tm[sg][h] = m[h];
            tl[sg][h] = l[h];
        }
        #pragma clang loop unroll(full)
        for (int e = 0; e < E; ++e) {
            ta[(int(sg) * FC_G + h) * FC_HD + int(lane) * E + e] = acc[h][e];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = tid; t < uint(FC_G * FC_HD); t += SG * 32) {
        const int h = int(t) / FC_HD;
        const int d = int(t) % FC_HD;
        float Mv = -INFINITY;
        #pragma clang loop unroll(full)
        for (int s = 0; s < SG; ++s) Mv = max(Mv, tm[s][h]);
        float Lv = 0.0f, Av = 0.0f;
        if (Mv > -INFINITY) {
            #pragma clang loop unroll(full)
            for (int s = 0; s < SG; ++s) {
                const float w = metal::exp(tm[s][h] - Mv);
                Lv += tl[s][h] * w;
                Av += ta[(s * FC_G + h) * FC_HD + d] * w;
            }
        }
        const size_t row_out = (size_t(batch) * p.nblocks + block) * p.HQ + g * FC_G + h;
        part_acc[row_out * FC_HD + d] = Av;
        if (d == 0) {
            part_max[row_out] = Mv;
            part_sum[row_out] = Lv;
        }
    }
}

kernel void inkling_combine_mtp_batched(
    const device bfloat* ys                         [[buffer(0)]],
    const device float* weights                     [[buffer(1)]],
    const device float* taps                        [[buffer(2)]],
    const device float* resid                       [[buffer(3)]],
    device bfloat* out                              [[buffer(4)]],
    const device int* tok_to_stream                 [[buffer(5)]],
    const device int* tok_local_pos                 [[buffer(6)]],
    const device BatchMTPTargetVisitMetaGpu* vmeta  [[buffer(7)]],
    constant CombinePrm& p                          [[buffer(8)]],
    uint2 gid                                       [[thread_position_in_grid]])
{
    const int d = int(gid.x);
    const int row = int(gid.y);
    if (d >= MODEL_D || row >= p.M) return;

    const BatchMTPTargetVisitMetaGpu vm = vmeta[tok_to_stream[row]];
    const int r_loc = tok_local_pos[row];
    const device float* state = (const device float*)(vm.conv_addr[3]);
    device float* new_state = (device float*)(vm.hist_ms_addr);

    float ysum[4];
    #pragma clang loop unroll(full)
    for (int j = 0; j < 4; j++) {
        const int r = r_loc - 3 + j;
        ysum[j] = 0.0f;
        if (r >= 0) {
            const int g_r = int(row) - 3 + j;
            float y = 0.0f;
            #pragma clang loop unroll(full)
            for (int k = 0; k < MODEL_TOP_K; k++) {
                y += float(ys[(size_t(g_r) * MODEL_TOP_K + k) * MODEL_D + d]) * weights[size_t(g_r) * MODEL_TOP_K + k];
            }
            ysum[j] = round_bf16(y);
        }
    }
    const float y = ysum[3];
    const size_t so = size_t(row) * 3 * MODEL_D + d;
    const size_t to = size_t(d) * 4;
    const float s0 = r_loc >= 3 ? ysum[0] : state[size_t(r_loc) * MODEL_D + d];
    const float s1 = r_loc >= 2 ? ysum[1] : state[size_t(r_loc + 1) * MODEL_D + d];
    const float s2 = r_loc >= 1 ? ysum[2] : state[size_t(r_loc + 2) * MODEL_D + d];
    float h = (taps[to] * s0 + taps[to + 1] * s1 + taps[to + 2] * s2 + taps[to + 3] * y + y) * 0.4472135955f;
    h = round_bf16(h);
    new_state[so] = s1;
    new_state[so + MODEL_D] = s2;
    new_state[so + 2 * MODEL_D] = y;
    out[size_t(row) * MODEL_D + d] = bfloat(resid[size_t(row) * MODEL_D + d] + h);
}

struct MtpBatchCommitPrm {
    int B, NV, row_elems;
};

kernel void inkling_kv_commit_rotating_mtp_batched(
    const device BatchMTPTargetVisitMetaGpu* all_vmeta  [[buffer(0)]],
    const device int* d_n_acc                           [[buffer(1)]],
    constant MtpBatchCommitPrm& p                       [[buffer(2)]],
    uint3 tg_pos                                        [[threadgroup_position_in_grid]],
    uint tid                                            [[thread_index_in_threadgroup]])
{
    const int gid = int(tg_pos.x) * 256 + int(tid);
    const int v   = int(tg_pos.y);
    const int b   = int(tg_pos.z);
    if (v >= p.NV || b >= p.B) return;
    const BatchMTPTargetVisitMetaGpu vm = all_vmeta[size_t(v) * p.B + b];
    if (vm.need_staging == 0) return;

    const int n_toks = min(d_n_acc[b], vm.M_b);
    const int total = max(0, n_toks) * vm.n_kv * vm.head_dim;
    if (gid >= total) return;

    const int d = gid % vm.head_dim;
    const int tmp = gid / vm.head_dim;
    const int h = tmp % vm.n_kv;
    const int m = tmp / vm.n_kv;
    const int slot = (vm.old_idx + m) % vm.max_size;
    const size_t src = (size_t(vm.v_start + m) * vm.n_kv + h) * vm.head_dim + d;
    const size_t dst = size_t(h) * size_t(vm.stride_h) + size_t(slot) * size_t(vm.stride_s) + d;
    ((device bfloat*)(vm.kv_k_addr))[dst] = ((const device bfloat*)(vm.stage_k_addr))[src];
    ((device bfloat*)(vm.kv_v_addr))[dst] = ((const device bfloat*)(vm.stage_v_addr))[src];
}

kernel void inkling_copy_conv4_mtp_batched(
    const device BatchMTPTargetVisitMetaGpu* all_vmeta  [[buffer(0)]],
    const device int* d_n_acc                           [[buffer(1)]],
    constant MtpBatchCommitPrm& p                       [[buffer(2)]],
    uint3 tg_pos                                        [[threadgroup_position_in_grid]],
    uint tid                                            [[thread_index_in_threadgroup]])
{
    const uint gid = tg_pos.x * 256u + tid;
    const int v    = int(tg_pos.y);
    const int b    = int(tg_pos.z);
    if (v >= p.NV || b >= p.B) return;
    const BatchMTPTargetVisitMetaGpu vm = all_vmeta[size_t(v) * p.B + b];
    const int n_toks = min(d_n_acc[b], vm.M_b);
    if (n_toks <= 0) return;

    const size_t r = size_t(vm.v_start + n_toks - 1);
    const uint nc = uint(3 * vm.n_kv * vm.head_dim);
    const uint nd = uint(3 * MODEL_D);
    if (gid < nc) {
        const size_t off_c = r * size_t(nc) + size_t(gid);
        ((device float*)(vm.conv_addr[0]))[gid] = ((const device float*)(vm.hist_ks_addr))[off_c];
        ((device float*)(vm.conv_addr[1]))[gid] = ((const device float*)(vm.hist_vs_addr))[off_c];
    }
    if (gid < nd) {
        const size_t off_d = r * size_t(nd) + size_t(gid);
        ((device float*)(vm.conv_addr[2]))[gid] = ((const device float*)(vm.hist_cs_addr))[off_d];
        ((device float*)(vm.conv_addr[3]))[gid] = ((const device float*)(vm.hist_ms_addr))[off_d];
    }
}

kernel void inkling_copy_capture_mtp_batched(
    const device bfloat* vcap                   [[buffer(0)]],
    const device BatchMTPStreamMetaGpu* sm_all  [[buffer(1)]],
    const device int* d_n_acc                   [[buffer(2)]],
    constant MtpBatchCommitPrm& p               [[buffer(3)]],
    uint3 tg_pos                                [[threadgroup_position_in_grid]],
    uint tid                                    [[thread_index_in_threadgroup]])
{
    const int col = int(tg_pos.x) * 256 + int(tid);
    const int r   = int(tg_pos.y);
    const int b   = int(tg_pos.z);
    if (b >= p.B || col >= p.row_elems) return;
    const BatchMTPStreamMetaGpu sm = sm_all[b];
    const int n_toks = min(d_n_acc[b], sm.num_draft + 1);
    if (r >= n_toks) return;
    const size_t src_idx = size_t(sm.v_start + r) * size_t(p.row_elems) + size_t(col);
    const size_t dst_idx = size_t(r) * size_t(p.row_elems) + size_t(col);
    ((device bfloat*)(sm.out_hidden_addr))[dst_idx] = vcap[src_idx];
}
)MSL";

static const char* kMetal4NaxSource = R"MSL(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

struct NaxQkvPrm {
    int M, S, K, N, AT3, AT4, NT0, NT1, NT2;
};

template <int BM, int BN, int SG>
inline void nax_qkv_impl(
    const device bfloat* A,
    const device bfloat* B,
    device bfloat* C,
    constant NaxQkvPrm& p,
    uint2 tgid)
{
    using ext_t = dextents<int32_t, 2>;
    using str_t = array<int32_t, 2>;
    using ten_t = tensor<device bfloat, ext_t, tensor_inline>;

    const int m_off = int(tgid.y) * BM;
    if (m_off >= p.M) return;

    int src = 0, seg_start = 0, seg_len = p.N, n_off = int(tgid.x) * BN;
    if (p.S > 1) {
        const int gx = int(tgid.x);
        if (gx < p.NT0) {
            src = 0; seg_start = 0; seg_len = p.AT3; n_off = gx * BN;
        } else if (gx < p.NT0 + p.NT1) {
            src = 1; seg_start = p.AT3; seg_len = p.AT4 - p.AT3; n_off = (gx - p.NT0) * BN;
        } else {
            src = 2; seg_start = p.AT4; seg_len = p.N - p.AT4; n_off = (gx - p.NT0 - p.NT1) * BN;
        }
    }
    if (n_off >= seg_len) return;

    ten_t tA((device bfloat*)(A + size_t(src) * size_t(p.K)), ext_t(p.K, p.M), str_t{1, p.S * p.K});
    ten_t tB((device bfloat*)(B + size_t(seg_start) * size_t(p.K)), ext_t(p.K, seg_len), str_t{1, p.K});
    ten_t tC(C + seg_start, ext_t(seg_len, p.M), str_t{1, p.N});

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, static_cast<int>(dynamic_extent),
        false, true, true, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<SG>> op;

    if (m_off + BM <= p.M && n_off + BN <= seg_len) {
        auto mA = tA.slice<dynamic_extent, BM>(0, m_off);
        auto mB = tB.slice<dynamic_extent, BN>(0, n_off);
        auto mC = tC.slice<BN, BM>(n_off, m_off);
        auto cT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
        op.run(mA, mB, cT);
        auto dOut = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), bfloat>();
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) dOut[i] = static_cast<bfloat>(cT[i]);
        dOut.store(mC);
    } else {
        auto mA = tA.slice(0, m_off);
        auto mB = tB.slice(0, n_off);
        auto mC = tC.slice(n_off, m_off);
        auto cT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
        op.run(mA, mB, cT);
        auto dOut = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), bfloat>();
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) dOut[i] = static_cast<bfloat>(cT[i]);
        dOut.store(mC);
    }
}

#define DEFINE_NAX_QKV(BM, BN, SG)                                             \
kernel void inkling_nax_qkv_bf16_##BM##x##BN(                                  \
    const device bfloat* A [[buffer(0)]],                                      \
    const device bfloat* B [[buffer(1)]],                                      \
    device bfloat* C       [[buffer(2)]],                                      \
    constant NaxQkvPrm& p  [[buffer(3)]],                                      \
    uint2 tgid             [[threadgroup_position_in_grid]])                   \
{                                                                              \
    nax_qkv_impl<BM, BN, SG>(A, B, C, p, tgid);                                \
}

DEFINE_NAX_QKV(16, 64, 4)
DEFINE_NAX_QKV(32, 64, 4)
DEFINE_NAX_QKV(64, 64, 4)
DEFINE_NAX_QKV(64, 32, 4)

struct NaxArgmaxPrm {
    int M, NTG;
};

template <int BM, int BN, int SG>
inline void nax_head_argmax_p1_impl(
    const device bfloat* h,
    const device bfloat* w,
    device float* part_val,
    device uint* part_idx,
    constant NaxArgmaxPrm& p,
    threadgroup float* C_tg,
    uint2 tgid,
    uint lane,
    uint sg)
{
    using ext_t = dextents<int32_t, 2>;
    using str_t = array<int32_t, 2>;
    using ten_dev = tensor<device bfloat, ext_t, tensor_inline>;
    using ten_tg  = tensor<threadgroup float, ext_t, tensor_inline>;

    const int n_off = int(tgid.x) * BN;
    const int m_off = int(tgid.y) * BM;
    if (n_off >= MODEL_V_LIMIT || m_off >= p.M) return;
    const int valid_m = min(BM, p.M - m_off);
    const int valid_n = min(BN, MODEL_V_LIMIT - n_off);

    ten_dev tA((device bfloat*)h, ext_t(MODEL_D, p.M), str_t{1, MODEL_D});
    ten_dev tB((device bfloat*)w, ext_t(MODEL_D, MODEL_V_LIMIT), str_t{1, MODEL_D});
    ten_tg  tC(C_tg, ext_t(valid_n, valid_m), str_t{1, BN});

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, static_cast<int>(dynamic_extent),
        false, true, true, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<SG>> op;

    if (valid_m == BM && valid_n == BN) {
        auto mA = tA.slice<dynamic_extent, BM>(0, m_off);
        auto mB = tB.slice<dynamic_extent, BN>(0, n_off);
        auto mC = tC.slice<BN, BM>(0, 0);
        auto cT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
        op.run(mA, mB, cT);
        cT.store(mC);
    } else {
        auto mA = tA.slice(0, m_off);
        auto mB = tB.slice(0, n_off);
        auto mC = tC.slice(0, 0);
        auto cT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
        op.run(mA, mB, cT);
        cT.store(mC);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int r = int(sg); r < valid_m; r += SG) {
        const int i0 = int(lane);
        const int i1 = i0 + 32;
        const float v0 = (i0 < valid_n) ? C_tg[r * BN + i0] : -INFINITY;
        const float v1 = (i1 < valid_n) ? C_tg[r * BN + i1] : -INFINITY;
        float bv = v0;
        uint bi = uint(n_off + i0);
        if (v1 > bv) {
            bv = v1;
            bi = uint(n_off + i1);
        }
        for (ushort off = 16; off > 0; off >>= 1) {
            const float ov = simd_shuffle_xor(bv, off);
            const uint  oi = simd_shuffle_xor(bi, off);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) {
            const int global_b = m_off + r;
            part_val[size_t(global_b) * p.NTG + int(tgid.x)] = bv;
            part_idx[size_t(global_b) * p.NTG + int(tgid.x)] = bi;
        }
    }
}

#define DEFINE_NAX_ARGMAX(BM, BN, SG)                                          \
kernel void inkling_nax_head_argmax_p1_##BM##x##BN(                            \
    const device bfloat* h  [[buffer(0)]],                                     \
    const device bfloat* w  [[buffer(1)]],                                     \
    device float* part_val  [[buffer(2)]],                                     \
    device uint* part_idx   [[buffer(3)]],                                     \
    constant NaxArgmaxPrm& p[[buffer(4)]],                                     \
    uint2 tgid              [[threadgroup_position_in_grid]],                  \
    uint lane               [[thread_index_in_simdgroup]],                     \
    uint sg                 [[simdgroup_index_in_threadgroup]])                \
{                                                                              \
    threadgroup float C_tg[BM * BN];                                           \
    nax_head_argmax_p1_impl<BM, BN, SG>(h, w, part_val, part_idx, p, C_tg, tgid, lane, sg); \
}

DEFINE_NAX_ARGMAX(16, 64, 4)

template <typename Op, typename MA, typename MB, typename MC>
inline void nax_run_store_bf16(thread Op& op, thread MA& mA, thread MB& mB, thread MC& mC) {
    auto cT = op.template get_destination_cooperative_tensor<MA, MB, float>();
    op.run(mA, mB, cT);
    auto dOut = op.template get_destination_cooperative_tensor<MA, MB, bfloat>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) dOut[i] = static_cast<bfloat>(cT[i]);
    }
    dOut.store(mC);
}

struct NaxMsgPrm {
    int M;
};

template <int BM, int BN, int SG>
inline void nax_msg_fg_impl(
    const device bfloat* A,
    const device bfloat* B,
    device bfloat* C,
    constant NaxMsgPrm& p,
    uint3 tgid)
{
    using ext_t = dextents<int32_t, 2>;
    using str_t = array<int32_t, 2>;
    using ten_t = tensor<device bfloat, ext_t, tensor_inline>;

    const int n_off = int(tgid.x) * BN;
    const int m_off = int(tgid.y) * BM;
    const int j     = int(tgid.z);
    constexpr int K2D = 2 * MODEL_D;
    if (n_off >= K2D || m_off >= p.M) return;

    const device bfloat* Aj = A + size_t(j) * size_t(p.M) * size_t(MODEL_RANK);
    const device bfloat* Bj = B + size_t(j) * size_t(MODEL_RANK) * size_t(K2D);
    device bfloat*       Cj = C + size_t(j) * size_t(p.M) * size_t(K2D);

    ten_t tA((device bfloat*)Aj, ext_t(MODEL_RANK, p.M), str_t{1, MODEL_RANK});
    ten_t tB((device bfloat*)Bj, ext_t(K2D, MODEL_RANK), str_t{1, K2D});
    ten_t tC(Cj, ext_t(K2D, p.M), str_t{1, K2D});

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, static_cast<int>(dynamic_extent),
        false, false, true, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<SG>> op;

    if (m_off + BM <= p.M && n_off + BN <= K2D) {
        auto mA = tA.slice<dynamic_extent, BM>(0, m_off);
        auto mB = tB.slice<BN, dynamic_extent>(n_off, 0);
        auto mC = tC.slice<BN, BM>(n_off, m_off);
        nax_run_store_bf16(op, mA, mB, mC);
    } else {
        auto mA = tA.slice(0, m_off);
        auto mB = tB.slice(n_off, 0);
        auto mC = tC.slice(n_off, m_off);
        nax_run_store_bf16(op, mA, mB, mC);
    }
}

#define DEFINE_NAX_MSG_FG(BM, BN, SG)                                          \
kernel void inkling_nax_msg_fg_##BM##x##BN(                                    \
    const device bfloat* A [[buffer(0)]],                                      \
    const device bfloat* B [[buffer(1)]],                                      \
    device bfloat* C       [[buffer(2)]],                                      \
    constant NaxMsgPrm& p  [[buffer(3)]],                                      \
    uint3 tgid             [[threadgroup_position_in_grid]])                   \
{                                                                              \
    nax_msg_fg_impl<BM, BN, SG>(A, B, C, p, tgid);                             \
}

DEFINE_NAX_MSG_FG(16, 64, 4)
DEFINE_NAX_MSG_FG(32, 64, 4)
DEFINE_NAX_MSG_FG(64, 64, 4)

template <int BM, int BN, int SG>
inline void nax_router_scores_impl(
    const device bfloat* A,
    const device bfloat* B,
    device float* C,
    constant NaxMsgPrm& p,
    uint2 tgid)
{
    using ext_t = dextents<int32_t, 2>;
    using str_t = array<int32_t, 2>;
    using ten_bf = tensor<device bfloat, ext_t, tensor_inline>;
    using ten_f  = tensor<device float, ext_t, tensor_inline>;

    const int n_off = int(tgid.x) * BN;
    const int m_off = int(tgid.y) * BM;
    if (n_off >= MODEL_E || m_off >= p.M) return;

    ten_bf tA((device bfloat*)A, ext_t(MODEL_D, p.M), str_t{1, MODEL_D});
    ten_bf tB((device bfloat*)B, ext_t(MODEL_D, MODEL_E), str_t{1, MODEL_D});
    ten_f  tC(C, ext_t(MODEL_E, p.M), str_t{1, MODEL_E});

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, static_cast<int>(dynamic_extent),
        false, true, true, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<SG>> op;

    if (m_off + BM <= p.M && n_off + BN <= MODEL_E) {
        auto mA = tA.slice<dynamic_extent, BM>(0, m_off);
        auto mB = tB.slice<dynamic_extent, BN>(0, n_off);
        auto mC = tC.slice<BN, BM>(n_off, m_off);
        auto cT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
        op.run(mA, mB, cT);
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
            if (cT.is_valid_element(i)) cT[i] = 1.0f / (1.0f + metal::precise::exp(-cT[i]));
        }
        cT.store(mC);
    } else {
        auto mA = tA.slice(0, m_off);
        auto mB = tB.slice(0, n_off);
        auto mC = tC.slice(n_off, m_off);
        auto cT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
        op.run(mA, mB, cT);
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
            if (cT.is_valid_element(i)) cT[i] = 1.0f / (1.0f + metal::precise::exp(-cT[i]));
        }
        cT.store(mC);
    }
}

kernel void inkling_nax_router_16x32(
    const device bfloat* A [[buffer(0)]],
    const device bfloat* B [[buffer(1)]],
    device float* C        [[buffer(2)]],
    constant NaxMsgPrm& p  [[buffer(3)]],
    uint2 tgid             [[threadgroup_position_in_grid]])
{
    nax_router_scores_impl<16, 32, 4>(A, B, C, p, tgid);
}

struct NaxBatchMTPStreamMetaGpu {
    int32_t anchor_token;
    int32_t n_ctx;
    int32_t ctx_start;
    int32_t num_draft;
    int32_t d_start;
    int32_t v_start;
    int32_t is_sampled;
    int32_t do_top_p;
    float   temperature;
    float   top_p;
    float   softcap;
    float   _pad0;
    ulong   rng_seed;
    ulong   rng_step;
    ulong   out_hidden_addr;
};

kernel void inkling_nax_head_logits_max_p1_16x64(
    const device bfloat* h                         [[buffer(0)]],
    const device bfloat* w                         [[buffer(1)]],
    device float* out_logits                       [[buffer(2)]],
    device float* part_max                         [[buffer(3)]],
    device float* part_val                         [[buffer(4)]],
    device uint* part_idx                          [[buffer(5)]],
    const device int* tok_to_stream                [[buffer(6)]],
    const device NaxBatchMTPStreamMetaGpu* sm_all  [[buffer(7)]],
    constant NaxArgmaxPrm& p                       [[buffer(8)]],
    uint2 tgid                                     [[threadgroup_position_in_grid]],
    uint lane                                      [[thread_index_in_simdgroup]],
    uint sg                                        [[simdgroup_index_in_threadgroup]])
{
    constexpr int BM = 16;
    constexpr int BN = 64;
    constexpr int SG = 4;
    using ext_t = dextents<int32_t, 2>;
    using str_t = array<int32_t, 2>;
    using ten_dev = tensor<device bfloat, ext_t, tensor_inline>;
    using ten_tg  = tensor<threadgroup float, ext_t, tensor_inline>;

    const int n_off = int(tgid.x) * BN;
    const int m_off = int(tgid.y) * BM;
    if (n_off >= MODEL_V_LIMIT || m_off >= p.M) return;
    const int valid_m = min(BM, p.M - m_off);
    const int valid_n = min(BN, MODEL_V_LIMIT - n_off);

    threadgroup float C_tg[BM * BN];
    ten_dev tA((device bfloat*)h, ext_t(MODEL_D, p.M), str_t{1, MODEL_D});
    ten_dev tB((device bfloat*)w, ext_t(MODEL_D, MODEL_V_LIMIT), str_t{1, MODEL_D});
    ten_tg  tC(C_tg, ext_t(valid_n, valid_m), str_t{1, BN});

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, static_cast<int>(dynamic_extent),
        false, true, true, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<SG>> op;

    if (valid_m == BM && valid_n == BN) {
        auto mA = tA.slice<dynamic_extent, BM>(0, m_off);
        auto mB = tB.slice<dynamic_extent, BN>(0, n_off);
        auto mC = tC.slice<BN, BM>(0, 0);
        auto cT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
        op.run(mA, mB, cT);
        cT.store(mC);
    } else {
        auto mA = tA.slice(0, m_off);
        auto mB = tB.slice(0, n_off);
        auto mC = tC.slice(0, 0);
        auto cT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
        op.run(mA, mB, cT);
        cT.store(mC);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int r = int(sg); r < valid_m; r += SG) {
        const int global_m = m_off + r;
        const NaxBatchMTPStreamMetaGpu sm = sm_all[tok_to_stream[global_m]];
        const int i0 = int(lane);
        const int i1 = i0 + 32;
        if (sm.is_sampled != 0) {
            float v0 = -INFINITY, v1 = -INFINITY;
            if (i0 < valid_n) {
                v0 = C_tg[r * BN + i0];
                if (sm.softcap > 0.0f) v0 = sm.softcap * metal::precise::tanh(v0 / sm.softcap);
                v0 /= sm.temperature;
                out_logits[size_t(global_m) * MODEL_V_LIMIT + size_t(n_off + i0)] = v0;
            }
            if (i1 < valid_n) {
                v1 = C_tg[r * BN + i1];
                if (sm.softcap > 0.0f) v1 = sm.softcap * metal::precise::tanh(v1 / sm.softcap);
                v1 /= sm.temperature;
                out_logits[size_t(global_m) * MODEL_V_LIMIT + size_t(n_off + i1)] = v1;
            }
            const float bv = simd_max(max(v0, v1));
            if (lane == 0) part_max[size_t(global_m) * p.NTG + int(tgid.x)] = bv;
        } else {
            const float v0 = (i0 < valid_n) ? C_tg[r * BN + i0] : -INFINITY;
            const float v1 = (i1 < valid_n) ? C_tg[r * BN + i1] : -INFINITY;
            float bv = v0;
            uint bi = uint(n_off + i0);
            if (v1 > bv) { bv = v1; bi = uint(n_off + i1); }
            for (ushort off = 16; off > 0; off >>= 1) {
                const float ov = simd_shuffle_xor(bv, off);
                const uint  oi = simd_shuffle_xor(bi, off);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            if (lane == 0) {
                part_val[size_t(global_m) * p.NTG + int(tgid.x)] = bv;
                part_idx[size_t(global_m) * p.NTG + int(tgid.x)] = bi;
            }
        }
    }
}

struct NaxGuPrm {
    int M, K, INTER;
};

template <int BM, int BN, int SG>
inline void mtp_nax_gu_silu_impl(
    const device bfloat* A,
    const device bfloat* gu_w,
    device bfloat* C,
    constant NaxGuPrm& p,
    uint2 tgid)
{
    using ext_t = dextents<int32_t, 2>;
    using str_t = array<int32_t, 2>;
    using ten_t = tensor<device bfloat, ext_t, tensor_inline>;

    const int n_off = int(tgid.x) * BN;
    const int m_off = int(tgid.y) * BM;
    if (n_off >= p.INTER || m_off >= p.M) return;

    ten_t tA((device bfloat*)A, ext_t(p.K, p.M), str_t{1, p.K});
    ten_t tBg((device bfloat*)gu_w, ext_t(p.K, p.INTER), str_t{1, p.K});
    ten_t tBu((device bfloat*)(gu_w + size_t(p.INTER) * size_t(p.K)), ext_t(p.K, p.INTER), str_t{1, p.K});
    ten_t tC(C, ext_t(p.INTER, p.M), str_t{1, p.INTER});

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, static_cast<int>(dynamic_extent),
        false, true, true, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<SG>> op;

    if (m_off + BM <= p.M && n_off + BN <= p.INTER) {
        auto mA  = tA.slice<dynamic_extent, BM>(0, m_off);
        auto mBg = tBg.slice<dynamic_extent, BN>(0, n_off);
        auto mBu = tBu.slice<dynamic_extent, BN>(0, n_off);
        auto mC  = tC.slice<BN, BM>(n_off, m_off);
        auto cTg = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mBg), float>();
        auto cTu = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mBu), float>();
        op.run(mA, mBg, cTg);
        op.run(mA, mBu, cTu);
        auto dOut = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mBg), bfloat>();
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < cTg.get_capacity(); ++i) {
            if (cTg.is_valid_element(i)) {
                const float g = float(static_cast<bfloat>(cTg[i]));
                const float u = float(static_cast<bfloat>(cTu[i]));
                dOut[i] = static_cast<bfloat>((g / (1.0f + metal::precise::exp(-g))) * u);
            }
        }
        dOut.store(mC);
    } else {
        auto mA  = tA.slice(0, m_off);
        auto mBg = tBg.slice(0, n_off);
        auto mBu = tBu.slice(0, n_off);
        auto mC  = tC.slice(n_off, m_off);
        auto cTg = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mBg), float>();
        auto cTu = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mBu), float>();
        op.run(mA, mBg, cTg);
        op.run(mA, mBu, cTu);
        auto dOut = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mBg), bfloat>();
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < cTg.get_capacity(); ++i) {
            if (cTg.is_valid_element(i)) {
                const float g = float(static_cast<bfloat>(cTg[i]));
                const float u = float(static_cast<bfloat>(cTu[i]));
                dOut[i] = static_cast<bfloat>((g / (1.0f + metal::precise::exp(-g))) * u);
            }
        }
        dOut.store(mC);
    }
}

#define DEFINE_MTP_NAX_GU_SILU(BM, BN, SG)                                     \
kernel void mtp_nax_gu_silu_##BM##x##BN(                                       \
    const device bfloat* A    [[buffer(0)]],                                   \
    const device bfloat* gu_w [[buffer(1)]],                                   \
    device bfloat* C          [[buffer(2)]],                                   \
    constant NaxGuPrm& p      [[buffer(3)]],                                   \
    uint2 tgid                [[threadgroup_position_in_grid]])                \
{                                                                              \
    mtp_nax_gu_silu_impl<BM, BN, SG>(A, gu_w, C, p, tgid);                     \
}

DEFINE_MTP_NAX_GU_SILU(16, 64, 4)
DEFINE_MTP_NAX_GU_SILU(32, 64, 4)
)MSL";

static const char* kMTPUnifiedSource = R"MSL(
#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

constant int FC_M              [[function_constant(0)]];
constant int FC_K              [[function_constant(1)]];
constant int FC_SG             [[function_constant(2)]];
constant int FC_HAS_RES        [[function_constant(3)]];
constant int FC_HAS_EMBED_NORM [[function_constant(4)]];

inline float philox_u01(ulong seed, ulong step, uint stream_id, uint elem_idx) {
    uint4 ctr = uint4(elem_idx, stream_id, uint(step), uint(step >> 32));
    uint2 key = uint2(uint(seed), uint(seed >> 32));
    #pragma clang loop unroll(full)
    for (int r = 0; r < 10; ++r) {
        const uint hi0 = mulhi(0xD2511F53u, ctr.x);
        const uint lo0 = 0xD2511F53u * ctr.x;
        const uint hi1 = mulhi(0xCD9E8D57u, ctr.z);
        const uint lo1 = 0xCD9E8D57u * ctr.z;
        ctr = uint4(hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0);
        key.x += 0x9E3779B9u;
        key.y += 0xBB67AE85u;
    }
    return (float(ctr.x >> 9) + 0.5f) * (1.0f / 8388608.0f);
}

struct InitXPrm {
    int anchor_token;
    int D;
    int block_size;
    int markov_rank;
    float embed_norm_eps;
};

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mtp_init_x(
    const device bfloat* embed_tokens [[buffer(0)]],
    const device bfloat* embed_norm_w [[buffer(1)]],
    const device bfloat* mask_embed   [[buffer(2)]],
    device bfloat* x_out              [[buffer(3)]],
    device bfloat* rnn_state          [[buffer(4)]],
    device int* prev_tok              [[buffer(5)]],
    device int* verify_toks           [[buffer(6)]],
    constant InitXPrm& p              [[buffer(7)]],
    uint tid                          [[thread_position_in_threadgroup]],
    uint lane                         [[thread_index_in_simdgroup]],
    uint sg                           [[simdgroup_index_in_threadgroup]])
{
    const device bfloat* emb_row = embed_tokens + size_t(p.anchor_token) * size_t(p.D);
    if (FC_HAS_EMBED_NORM != 0) {
        float ss = 0.0f;
        for (int d = int(tid); d < p.D; d += 256) {
            const float v = float(emb_row[d]);
            ss += v * v;
        }
        ss = simd_sum(ss);
        threadgroup float sg_ss[8];
        if (lane == 0) sg_ss[sg] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = 0.0f;
        if (sg == 0 && lane < 8) tot = sg_ss[lane];
        tot = simd_sum(tot);
        if (tid == 0) sg_ss[0] = metal::precise::rsqrt(tot / float(p.D) + p.embed_norm_eps);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inv_rms = sg_ss[0];
        for (int d = int(tid); d < p.D; d += 256) {
            x_out[d] = bfloat(float(bfloat(float(emb_row[d]) * inv_rms)) * float(embed_norm_w[d]));
        }
    } else {
        for (int d = int(tid); d < p.D; d += 256) {
            x_out[d] = emb_row[d];
        }
    }

    for (int d = int(tid); d < p.D; d += 256) {
        const bfloat mval = mask_embed[d];
        for (int r = 1; r < p.block_size; ++r) {
            x_out[size_t(r) * size_t(p.D) + d] = mval;
        }
    }

    for (int i = int(tid); i < p.markov_rank; i += 256) {
        rnn_state[i] = bfloat(0.0f);
    }
    if (tid == 0) {
        prev_tok[0] = p.anchor_token;
        verify_toks[0] = p.anchor_token;
    }
}

struct MixPrm {
    int D;
    int n_target_layers;
    int row_start;
    int alpha_row;
    float eps;
};

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mtp_dflare_mix(
    const device bfloat* hidden     [[buffer(0)]],
    const device float* fuse_alpha  [[buffer(1)]],
    const device bfloat* norm_w     [[buffer(2)]],
    device bfloat* out              [[buffer(3)]],
    constant MixPrm& p              [[buffer(4)]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tid                        [[thread_position_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint sg                         [[simdgroup_index_in_threadgroup]])
{
    constexpr int DPT = (MTP_D + 255) / 256;
    const int r = int(tgid);
    const int src_row = p.row_start + r;
    const device bfloat* h_base = hidden + size_t(src_row) * size_t(MTP_NCAP) * size_t(MTP_D);
    const device float* a_row = fuse_alpha + size_t(p.alpha_row) * size_t(MTP_NCAP);

    float alpha[MTP_NCAP];
    #pragma clang loop unroll(full)
    for (int tl = 0; tl < MTP_NCAP; ++tl) {
        alpha[tl] = float(bfloat(a_row[tl]));
    }

    float vals[DPT];
    float ss = 0.0f;
    int cnt = 0;
    for (int d = int(tid); d < MTP_D; d += 256) {
        float v = 0.0f;
        #pragma clang loop unroll(full)
        for (int tl = 0; tl < MTP_NCAP; ++tl) {
            v += float(h_base[tl * MTP_D + d]) * alpha[tl];
        }
        vals[cnt++] = v;
        ss += v * v;
    }

    ss = simd_sum(ss);
    threadgroup float sg_ss[8];
    if (lane == 0) sg_ss[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    if (sg == 0 && lane < 8) tot = sg_ss[lane];
    tot = simd_sum(tot);
    if (tid == 0) sg_ss[0] = metal::precise::rsqrt(tot / float(MTP_D) + p.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = sg_ss[0];

    device bfloat* out_row = out + size_t(r) * size_t(MTP_D);
    int idx = 0;
    for (int d = int(tid); d < MTP_D; d += 256) {
        out_row[d] = bfloat(vals[idx++] * inv_rms * float(norm_w[d]));
    }
}

struct AddNormPrm {
    int D;
    int src_row_off;
    int write_res;
    float eps;
};

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mtp_add_rmsnorm(
    const device bfloat* x_in       [[buffer(0)]],
    const device bfloat* res_in     [[buffer(1)]],
    const device bfloat* norm_w     [[buffer(2)]],
    device bfloat* res_out          [[buffer(3)]],
    device bfloat* norm_out         [[buffer(4)]],
    constant AddNormPrm& p          [[buffer(5)]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tid                        [[thread_position_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint sg                         [[simdgroup_index_in_threadgroup]])
{
    constexpr int DPT = (MTP_D + 255) / 256;
    const int r = int(tgid);
    const int sr = p.src_row_off + r;
    const device bfloat* xr = x_in + size_t(sr) * size_t(MTP_D);
    const device bfloat* rr = res_in + size_t(sr) * size_t(MTP_D);
    device bfloat* ro = res_out + size_t(r) * size_t(MTP_D);
    device bfloat* no = norm_out + size_t(r) * size_t(MTP_D);

    float vals[DPT];
    float ss = 0.0f;
    int cnt = 0;
    for (int d = int(tid); d < MTP_D; d += 256) {
        float v = float(xr[d]);
        if (FC_HAS_RES != 0) {
            v += float(rr[d]);
        }
        if (p.write_res != 0) {
            ro[d] = bfloat(v);
        }
        vals[cnt++] = v;
        ss += v * v;
    }

    ss = simd_sum(ss);
    threadgroup float sg_ss[8];
    if (lane == 0) sg_ss[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    if (sg == 0 && lane < 8) tot = sg_ss[lane];
    tot = simd_sum(tot);
    if (tid == 0) sg_ss[0] = metal::precise::rsqrt(tot / float(MTP_D) + p.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = sg_ss[0];

    int idx = 0;
    for (int d = int(tid); d < MTP_D; d += 256) {
        if (FC_HAS_RES == 0) {
            no[d] = bfloat(float(bfloat(vals[idx++] * inv_rms)) * float(norm_w[d]));
        } else {
            no[d] = bfloat(vals[idx++] * inv_rms * float(norm_w[d]));
        }
    }
}

struct GemvPrm {
    int N;
};

kernel void mtp_bf16_gemv(
    const device bfloat* x   [[buffer(0)]],
    const device bfloat* w   [[buffer(1)]],
    device bfloat* out       [[buffer(2)]],
    constant GemvPrm& p      [[buffer(3)]],
    uint tgid                [[threadgroup_position_in_grid]],
    uint lane                [[thread_index_in_simdgroup]],
    uint sg                  [[simdgroup_index_in_threadgroup]])
{
    const int row = int(tgid) * FC_SG + int(sg);
    if (row >= p.N) return;

    const int K4 = FC_K / 4;
    const int K_MAIN = K4 - (K4 % 96);
    const device vec<bfloat, 4>* w4 = (const device vec<bfloat, 4>*)(w + size_t(row) * size_t(FC_K));
    const device vec<bfloat, 4>* x4 = (const device vec<bfloat, 4>*)(x);

    float acc[8];
    #pragma clang loop unroll(full)
    for (int m = 0; m < FC_M; ++m) acc[m] = 0.0f;

    for (int base = 0; base < K_MAIN; base += 96) {
        const int off = base + int(lane);
        const vec<float, 4> w0 = vec<float, 4>(w4[off]);
        const vec<float, 4> w1 = vec<float, 4>(w4[off + 32]);
        const vec<float, 4> w2 = vec<float, 4>(w4[off + 64]);

        #pragma clang loop unroll(full)
        for (int m = 0; m < FC_M; ++m) {
            const device vec<bfloat, 4>* xm = x4 + m * K4 + off;
            acc[m] += dot(w0, vec<float, 4>(xm[0]))
                    + dot(w1, vec<float, 4>(xm[32]))
                    + dot(w2, vec<float, 4>(xm[64]));
        }
    }
    for (int idx = K_MAIN + int(lane); idx < K4; idx += 32) {
        const vec<float, 4> wv = vec<float, 4>(w4[idx]);
        #pragma clang loop unroll(full)
        for (int m = 0; m < FC_M; ++m) {
            acc[m] += dot(wv, vec<float, 4>(x4[m * K4 + idx]));
        }
    }

    #pragma clang loop unroll(full)
    for (int m = 0; m < FC_M; ++m) {
        const float v = simd_sum(acc[m]);
        if (lane == 0) {
            out[size_t(m) * size_t(p.N) + size_t(row)] = bfloat(v);
        }
    }
}

kernel void mtp_bf16_gemv_r2(
    const device bfloat* x   [[buffer(0)]],
    const device bfloat* w   [[buffer(1)]],
    device bfloat* out       [[buffer(2)]],
    constant GemvPrm& p      [[buffer(3)]],
    uint tgid                [[threadgroup_position_in_grid]],
    uint lane                [[thread_index_in_simdgroup]],
    uint sg                  [[simdgroup_index_in_threadgroup]])
{
    const int row0 = (int(tgid) * FC_SG + int(sg)) * 2;
    if (row0 >= p.N) return;
    const int row1 = min(row0 + 1, p.N - 1);

    const int K4 = FC_K / 4;
    const int K_MAIN = K4 - (K4 % 64);
    const device vec<bfloat, 4>* wa4 = (const device vec<bfloat, 4>*)(w + size_t(row0) * size_t(FC_K));
    const device vec<bfloat, 4>* wb4 = (const device vec<bfloat, 4>*)(w + size_t(row1) * size_t(FC_K));
    const device vec<bfloat, 4>* x4  = (const device vec<bfloat, 4>*)(x);

    float acc0[8];
    float acc1[8];
    #pragma clang loop unroll(full)
    for (int m = 0; m < FC_M; ++m) {
        acc0[m] = 0.0f;
        acc1[m] = 0.0f;
    }

    for (int base = 0; base < K_MAIN; base += 64) {
        const int off = base + int(lane);
        const vec<float, 4> wa0 = vec<float, 4>(wa4[off]);
        const vec<float, 4> wa1 = vec<float, 4>(wa4[off + 32]);
        const vec<float, 4> wb0 = vec<float, 4>(wb4[off]);
        const vec<float, 4> wb1 = vec<float, 4>(wb4[off + 32]);

        #pragma clang loop unroll(full)
        for (int m = 0; m < FC_M; ++m) {
            const device vec<bfloat, 4>* xm = x4 + m * K4 + off;
            const vec<float, 4> xv0 = vec<float, 4>(xm[0]);
            const vec<float, 4> xv1 = vec<float, 4>(xm[32]);
            acc0[m] += dot(wa0, xv0) + dot(wa1, xv1);
            acc1[m] += dot(wb0, xv0) + dot(wb1, xv1);
        }
    }
    for (int idx = K_MAIN + int(lane); idx < K4; idx += 32) {
        const vec<float, 4> wav = vec<float, 4>(wa4[idx]);
        const vec<float, 4> wbv = vec<float, 4>(wb4[idx]);
        #pragma clang loop unroll(full)
        for (int m = 0; m < FC_M; ++m) {
            const vec<float, 4> xv = vec<float, 4>(x4[m * K4 + idx]);
            acc0[m] += dot(wav, xv);
            acc1[m] += dot(wbv, xv);
        }
    }

    const bool has_r1 = (row0 + 1) < p.N;
    #pragma clang loop unroll(full)
    for (int m = 0; m < FC_M; ++m) {
        const float v0 = simd_sum(acc0[m]);
        const float v1 = simd_sum(acc1[m]);
        if (lane == 0) {
            out[size_t(m) * size_t(p.N) + size_t(row0)] = bfloat(v0);
            if (has_r1) {
                out[size_t(m) * size_t(p.N) + size_t(row0 + 1)] = bfloat(v1);
            }
        }
    }
}

kernel void mtp_bf16_gu_silu(
    const device bfloat* x     [[buffer(0)]],
    const device bfloat* gu_w  [[buffer(1)]],
    device bfloat* out         [[buffer(2)]],
    constant GemvPrm& p        [[buffer(3)]],
    uint tgid                  [[threadgroup_position_in_grid]],
    uint lane                  [[thread_index_in_simdgroup]],
    uint sg                    [[simdgroup_index_in_threadgroup]])
{
    const int row = int(tgid) * FC_SG + int(sg);
    if (row >= p.N) return;

    const int K4 = FC_K / 4;
    const int K_MAIN = K4 - (K4 % 64);
    const device vec<bfloat, 4>* wg4 = (const device vec<bfloat, 4>*)(gu_w + size_t(row) * size_t(FC_K));
    const device vec<bfloat, 4>* wu4 = (const device vec<bfloat, 4>*)(gu_w + size_t(p.N + row) * size_t(FC_K));
    const device vec<bfloat, 4>* x4 = (const device vec<bfloat, 4>*)(x);

    float acc_g[8];
    float acc_u[8];
    #pragma clang loop unroll(full)
    for (int m = 0; m < FC_M; ++m) {
        acc_g[m] = 0.0f;
        acc_u[m] = 0.0f;
    }

    for (int base = 0; base < K_MAIN; base += 64) {
        const int off = base + int(lane);
        const vec<float, 4> wg0 = vec<float, 4>(wg4[off]);
        const vec<float, 4> wg1 = vec<float, 4>(wg4[off + 32]);
        const vec<float, 4> wu0 = vec<float, 4>(wu4[off]);
        const vec<float, 4> wu1 = vec<float, 4>(wu4[off + 32]);

        #pragma clang loop unroll(full)
        for (int m = 0; m < FC_M; ++m) {
            const device vec<bfloat, 4>* xm = x4 + m * K4 + off;
            const vec<float, 4> x0 = vec<float, 4>(xm[0]);
            const vec<float, 4> x1 = vec<float, 4>(xm[32]);
            acc_g[m] += dot(wg0, x0) + dot(wg1, x1);
            acc_u[m] += dot(wu0, x0) + dot(wu1, x1);
        }
    }
    for (int idx = K_MAIN + int(lane); idx < K4; idx += 32) {
        const vec<float, 4> wgv = vec<float, 4>(wg4[idx]);
        const vec<float, 4> wuv = vec<float, 4>(wu4[idx]);
        #pragma clang loop unroll(full)
        for (int m = 0; m < FC_M; ++m) {
            const vec<float, 4> xv = vec<float, 4>(x4[m * K4 + idx]);
            acc_g[m] += dot(wgv, xv);
            acc_u[m] += dot(wuv, xv);
        }
    }

    #pragma clang loop unroll(full)
    for (int m = 0; m < FC_M; ++m) {
        const float g_sum = simd_sum(acc_g[m]);
        const float u_sum = simd_sum(acc_u[m]);
        if (lane == 0) {
            const float g = float(bfloat(g_sum));
            const float u = float(bfloat(u_sum));
            const float act = (g / (1.0f + metal::precise::exp(-g))) * u;
            out[size_t(m) * size_t(p.N) + size_t(row)] = bfloat(act);
        }
    }
}

struct KvCtxWritePrm {
    int kv_dim;
    int offset_before;
    int is_rotating;
    int window;
    int ring_start;
    int k_stride_h;
    int k_stride_s;
    int v_stride_h;
    int v_stride_s;
    float eps;
    float neg_log2_theta_over_half;
};

[[kernel, max_total_threads_per_threadgroup(64)]]
kernel void mtp_kv_ctx_norm_rope_write(
    const device bfloat* kv_ctx      [[buffer(0)]],
    const device bfloat* k_norm_w    [[buffer(1)]],
    const device float* inv_freq     [[buffer(2)]],
    device bfloat* k_cache           [[buffer(3)]],
    device bfloat* v_cache           [[buffer(4)]],
    constant KvCtxWritePrm& p        [[buffer(5)]],
    uint2 tgid                       [[threadgroup_position_in_grid]],
    uint2 tid2                       [[thread_position_in_threadgroup]],
    uint lane                        [[thread_index_in_simdgroup]],
    uint sg                          [[simdgroup_index_in_threadgroup]])
{
    constexpr int HALF_HD = MTP_HD / 2;
    constexpr int TG = (HALF_HD < 32) ? 32 : HALF_HD;
    constexpr int NS = TG / 32;
    const int t = int(tgid.x);
    const int h = int(tgid.y);
    const int d = int(tid2.x);

    const device bfloat* row = kv_ctx + size_t(t) * size_t(2 * p.kv_dim);
    const device bfloat* k_src = row + h * MTP_HD;
    const device bfloat* v_src = row + p.kv_dim + h * MTP_HD;

    const bool active = (d < HALF_HD);
    const float k0 = active ? float(k_src[d]) : 0.0f;
    const float k1 = active ? float(k_src[d + HALF_HD]) : 0.0f;

    float ss = simd_sum(k0 * k0 + k1 * k1);
    threadgroup float sg_ss[4];
    if (lane == 0) sg_ss[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot_ss = 0.0f;
    if (sg == 0 && int(lane) < NS) tot_ss = sg_ss[lane];
    tot_ss = simd_sum(tot_ss);
    if (d == 0) sg_ss[0] = metal::precise::rsqrt(tot_ss / float(MTP_HD) + p.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!active) return;

    const float inv_rms = sg_ss[0];
    const float kn0 = float(bfloat(float(bfloat(k0 * inv_rms)) * float(k_norm_w[d])));
    const float kn1 = float(bfloat(float(bfloat(k1 * inv_rms)) * float(k_norm_w[d + HALF_HD])));

    const float theta = float(p.offset_before + t) * metal::exp2(p.neg_log2_theta_over_half * float(d));
    const float c = metal::precise::cos(theta);
    const float s = metal::precise::sin(theta);

    const int dst_slot = (p.is_rotating != 0) ? ((p.ring_start + t) % p.window) : (p.offset_before + t);
    device bfloat* k_dst = k_cache + size_t(h) * size_t(p.k_stride_h) + size_t(dst_slot) * size_t(p.k_stride_s);
    device bfloat* v_dst = v_cache + size_t(h) * size_t(p.v_stride_h) + size_t(dst_slot) * size_t(p.v_stride_s);

    k_dst[d]            = bfloat(kn0 * c - kn1 * s);
    k_dst[d + HALF_HD]  = bfloat(kn1 * c + kn0 * s);
    v_dst[d]            = v_src[d];
    v_dst[d + HALF_HD]  = v_src[d + HALF_HD];
}

struct QkvPackPrm {
    int q_dim;
    int kv_dim;
    int total_dim;
    int rope_offset;
    float eps;
    float neg_log2_theta_over_half;
};

[[kernel, max_total_threads_per_threadgroup(64)]]
kernel void mtp_qkv_norm_rope_pack(
    const device bfloat* qkv         [[buffer(0)]],
    const device bfloat* q_norm_w    [[buffer(1)]],
    const device bfloat* k_norm_w    [[buffer(2)]],
    const device float* inv_freq     [[buffer(3)]],
    device bfloat* q_out             [[buffer(4)]],
    device bfloat* bk_out            [[buffer(5)]],
    device bfloat* bv_out            [[buffer(6)]],
    constant QkvPackPrm& p           [[buffer(7)]],
    uint2 tgid                       [[threadgroup_position_in_grid]],
    uint2 tid2                       [[thread_position_in_threadgroup]],
    uint lane                        [[thread_index_in_simdgroup]],
    uint sg                          [[simdgroup_index_in_threadgroup]])
{
    constexpr int HALF_HD = MTP_HD / 2;
    constexpr int TG = (HALF_HD < 32) ? 32 : HALF_HD;
    constexpr int NS = TG / 32;
    const int t = int(tgid.x);
    const int idx = int(tgid.y);
    const int d = int(tid2.x);
    const bool active = (d < HALF_HD);

    const device bfloat* row = qkv + size_t(t) * size_t(p.total_dim);

    if (idx >= MTP_H + MTP_KVH) {
        if (!active) return;
        const int h = idx - (MTP_H + MTP_KVH);
        const device bfloat* src = row + p.q_dim + p.kv_dim + h * MTP_HD;
        device bfloat* dst = bv_out + (size_t(h) * MTP_B + t) * MTP_HD;
        dst[d]           = src[d];
        dst[d + HALF_HD] = src[d + HALF_HD];
        return;
    }

    const bool is_q = (idx < MTP_H);
    const int h = is_q ? idx : (idx - MTP_H);
    const device bfloat* src = is_q ? (row + h * MTP_HD) : (row + p.q_dim + h * MTP_HD);
    const device bfloat* nw = is_q ? q_norm_w : k_norm_w;

    const float x0 = active ? float(src[d]) : 0.0f;
    const float x1 = active ? float(src[d + HALF_HD]) : 0.0f;

    float ss = simd_sum(x0 * x0 + x1 * x1);
    threadgroup float sg_ss[4];
    if (lane == 0) sg_ss[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot_ss = 0.0f;
    if (sg == 0 && int(lane) < NS) tot_ss = sg_ss[lane];
    tot_ss = simd_sum(tot_ss);
    if (d == 0) sg_ss[0] = metal::precise::rsqrt(tot_ss / float(MTP_HD) + p.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!active) return;

    const float inv_rms = sg_ss[0];
    const float n0 = float(bfloat(float(bfloat(x0 * inv_rms)) * float(nw[d])));
    const float n1 = float(bfloat(float(bfloat(x1 * inv_rms)) * float(nw[d + HALF_HD])));

    const float theta = float(p.rope_offset + t) * metal::exp2(p.neg_log2_theta_over_half * float(d));
    const float c = metal::precise::cos(theta);
    const float s = metal::precise::sin(theta);

    device bfloat* dst = is_q ? (q_out + (size_t(h) * MTP_B + t) * MTP_HD) : (bk_out + (size_t(h) * MTP_B + t) * MTP_HD);
    dst[d]           = bfloat(n0 * c - n1 * s);
    dst[d + HALF_HD] = bfloat(n1 * c + n0 * s);
}

struct SdpaPrm {
    int valid_ctx;
    int total_k;
    int k_stride_h;
    int k_stride_s;
    int v_stride_h;
    int v_stride_s;
    int n_splits;
    int split_chunk;
    float scale2;
};

static inline void sdpa_core(
    const device bfloat* q_buf,
    const device bfloat* k_cache,
    const device bfloat* v_cache,
    const device bfloat* bk_buf,
    const device bfloat* bv_buf,
    SdpaPrm p,
    int kv_h,
    int k_start,
    int k_end,
    uint tid,
    uint lane,
    uint sg,
    thread float o[MTP_G][4],
    thread float m[MTP_G],
    thread float l[MTP_G],
    threadgroup bfloat* K_tg,
    threadgroup bfloat* V_tg)
{
    constexpr int E = MTP_HD / 32;
    const int qt = int(sg);
    const int qh0 = kv_h * MTP_G;
    const int col = int(lane) * E;

    float q_reg[MTP_G][4];
    #pragma clang loop unroll(full)
    for (int g = 0; g < MTP_G; ++g) {
        const device bfloat* qp = q_buf + ((qh0 + g) * MTP_B + qt) * MTP_HD + col;
        #pragma clang loop unroll(full)
        for (int e = 0; e < E; ++e) {
            q_reg[g][e] = float(qp[e]);
            o[g][e] = 0.0f;
        }
        m[g] = -INFINITY;
        l[g] = 0.0f;
    }

    const device bfloat* kc_h = k_cache + size_t(kv_h) * size_t(p.k_stride_h);
    const device bfloat* vc_h = v_cache + size_t(kv_h) * size_t(p.v_stride_h);
    const device bfloat* bk_h = bk_buf + size_t(kv_h) * size_t(MTP_B * MTP_HD);
    const device bfloat* bv_h = bv_buf + size_t(kv_h) * size_t(MTP_B * MTP_HD);

    const int load_row = int(tid >> 5);
    const int load_col = int(tid & 31u) * E;

    for (int k_base = k_start; k_base < k_end; k_base += MTP_B) {
        const int valid_n = min(MTP_B, k_end - k_base);
        const int gk = k_base + load_row;
        if (gk < k_end) {
            const device bfloat* kp_src;
            const device bfloat* vp_src;
            if (gk < p.valid_ctx) {
                kp_src = kc_h + size_t(gk) * size_t(p.k_stride_s) + load_col;
                vp_src = vc_h + size_t(gk) * size_t(p.v_stride_s) + load_col;
            } else {
                const int bi = gk - p.valid_ctx;
                kp_src = bk_h + size_t(bi) * MTP_HD + load_col;
                vp_src = bv_h + size_t(bi) * MTP_HD + load_col;
            }
            threadgroup bfloat* kd = K_tg + load_row * MTP_HD + load_col;
            threadgroup bfloat* vd = V_tg + load_row * MTP_HD + load_col;
            #if MTP_HD == 128
                *(threadgroup vec<bfloat, 4>*)(kd) = *(const device vec<bfloat, 4>*)(kp_src);
                *(threadgroup vec<bfloat, 4>*)(vd) = *(const device vec<bfloat, 4>*)(vp_src);
            #else
                #pragma clang loop unroll(full)
                for (int e = 0; e < E; ++e) {
                    kd[e] = kp_src[e];
                    vd[e] = vp_src[e];
                }
            #endif
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float s_arr[MTP_G][MTP_B];
        float tm[MTP_G];
        #pragma clang loop unroll(full)
        for (int g = 0; g < MTP_G; ++g) tm[g] = -INFINITY;

        for (int j = 0; j < valid_n; ++j) {
            float kv_r[4];
            const threadgroup bfloat* ks = K_tg + j * MTP_HD + col;
            #pragma clang loop unroll(full)
            for (int e = 0; e < E; ++e) kv_r[e] = float(ks[e]);
            #pragma clang loop unroll(full)
            for (int g = 0; g < MTP_G; ++g) {
                float d_acc = 0.0f;
                #pragma clang loop unroll(full)
                for (int e = 0; e < E; ++e) d_acc += q_reg[g][e] * kv_r[e];
                const float dg = simd_sum(d_acc) * p.scale2;
                s_arr[g][j] = dg;
                tm[g] = max(tm[g], dg);
            }
        }

        if (k_base == k_start) {
            #pragma clang loop unroll(full)
            for (int g = 0; g < MTP_G; ++g) m[g] = tm[g];
        } else {
            #pragma clang loop unroll(full)
            for (int g = 0; g < MTP_G; ++g) {
                const float nm = max(m[g], tm[g]);
                const float f = metal::precise::exp2(m[g] - nm);
                l[g] *= f;
                #pragma clang loop unroll(full)
                for (int e = 0; e < E; ++e) o[g][e] *= f;
                m[g] = nm;
            }
        }

        for (int j = 0; j < valid_n; ++j) {
            float vv_r[4];
            const threadgroup bfloat* vs = V_tg + j * MTP_HD + col;
            #pragma clang loop unroll(full)
            for (int e = 0; e < E; ++e) vv_r[e] = float(vs[e]);
            #pragma clang loop unroll(full)
            for (int g = 0; g < MTP_G; ++g) {
                const float pg = metal::precise::exp2(s_arr[g][j] - m[g]);
                l[g] += pg;
                #pragma clang loop unroll(full)
                for (int e = 0; e < E; ++e) o[g][e] += pg * vv_r[e];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mtp_sdpa_single(
    const device bfloat* q_buf    [[buffer(0)]],
    const device bfloat* k_cache  [[buffer(1)]],
    const device bfloat* v_cache  [[buffer(2)]],
    const device bfloat* bk_buf   [[buffer(3)]],
    const device bfloat* bv_buf   [[buffer(4)]],
    device bfloat* attn_out       [[buffer(5)]],
    constant SdpaPrm& p           [[buffer(6)]],
    uint tgid                     [[threadgroup_position_in_grid]],
    uint tid                      [[thread_position_in_threadgroup]],
    uint lane                     [[thread_index_in_simdgroup]],
    uint sg                       [[simdgroup_index_in_threadgroup]])
{
    constexpr int E = MTP_HD / 32;
    threadgroup bfloat K_tg[MTP_B * MTP_HD];
    threadgroup bfloat V_tg[MTP_B * MTP_HD];

    const int kv_h = int(tgid);
    float o[MTP_G][4];
    float m[MTP_G], l[MTP_G];
    sdpa_core(q_buf, k_cache, v_cache, bk_buf, bv_buf, p, kv_h, 0, p.total_k,
              tid, lane, sg, o, m, l, K_tg, V_tg);

    const int qt = int(sg);
    const int qh0 = kv_h * MTP_G;
    const int col = int(lane) * E;
    device bfloat* row_out = attn_out + size_t(qt) * size_t(MTP_H * MTP_HD);
    #pragma clang loop unroll(full)
    for (int g = 0; g < MTP_G; ++g) {
        const float r = 1.0f / l[g];
        device bfloat* dst = row_out + size_t(qh0 + g) * MTP_HD + col;
        #pragma clang loop unroll(full)
        for (int e = 0; e < E; ++e) dst[e] = bfloat(o[g][e] * r);
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mtp_sdpa_split_p1(
    const device bfloat* q_buf    [[buffer(0)]],
    const device bfloat* k_cache  [[buffer(1)]],
    const device bfloat* v_cache  [[buffer(2)]],
    const device bfloat* bk_buf   [[buffer(3)]],
    const device bfloat* bv_buf   [[buffer(4)]],
    device float* part_o          [[buffer(5)]],
    device float* part_ml         [[buffer(6)]],
    constant SdpaPrm& p           [[buffer(7)]],
    uint2 tgid                    [[threadgroup_position_in_grid]],
    uint2 tid2                    [[thread_position_in_threadgroup]],
    uint lane                     [[thread_index_in_simdgroup]],
    uint sg                       [[simdgroup_index_in_threadgroup]])
{
    constexpr int E = MTP_HD / 32;
    const uint tid = tid2.x;
    threadgroup bfloat K_tg[MTP_B * MTP_HD];
    threadgroup bfloat V_tg[MTP_B * MTP_HD];

    const int kv_h = int(tgid.x);
    const int sp = int(tgid.y);
    const int k_start = sp * p.split_chunk;
    const int k_end = min(p.total_k, k_start + p.split_chunk);

    const int qt = int(sg);
    const int qh0 = kv_h * MTP_G;
    const int col = int(lane) * E;

    float o[MTP_G][4];
    float m[MTP_G], l[MTP_G];
    #pragma clang loop unroll(full)
    for (int g = 0; g < MTP_G; ++g) {
        #pragma clang loop unroll(full)
        for (int e = 0; e < E; ++e) o[g][e] = 0.0f;
        m[g] = -INFINITY;
        l[g] = 0.0f;
    }

    if (k_start < k_end) {
        sdpa_core(q_buf, k_cache, v_cache, bk_buf, bv_buf, p, kv_h, k_start, k_end,
                  tid, lane, sg, o, m, l, K_tg, V_tg);
    }

    #pragma clang loop unroll(full)
    for (int g = 0; g < MTP_G; ++g) {
        const size_t base_g = (size_t(sp) * MTP_H + size_t(qh0 + g)) * MTP_B + size_t(qt);
        device float* po = part_o + base_g * MTP_HD + col;
        #pragma clang loop unroll(full)
        for (int e = 0; e < E; ++e) po[e] = o[g][e];
        if (lane == 0) {
            part_ml[base_g * 2 + 0] = m[g];
            part_ml[base_g * 2 + 1] = l[g];
        }
    }
}

[[kernel, max_total_threads_per_threadgroup(32)]]
kernel void mtp_sdpa_split_p2(
    const device float* part_o    [[buffer(0)]],
    const device float* part_ml   [[buffer(1)]],
    device bfloat* attn_out       [[buffer(2)]],
    constant SdpaPrm& p           [[buffer(3)]],
    uint2 tgid                    [[threadgroup_position_in_grid]],
    uint lane                     [[thread_index_in_simdgroup]])
{
    constexpr int E = MTP_HD / 32;
    const int qt = int(tgid.x);
    const int qh = int(tgid.y);
    const int col = int(lane) * E;

    float m_max = -INFINITY;
    for (int sp = 0; sp < p.n_splits; ++sp) {
        const size_t idx = (size_t(sp) * MTP_H + size_t(qh)) * MTP_B + size_t(qt);
        const float ls = part_ml[idx * 2 + 1];
        if (ls > 0.0f) {
            m_max = max(m_max, part_ml[idx * 2 + 0]);
        }
    }

    float o_tot[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float l_tot = 0.0f;
    for (int sp = 0; sp < p.n_splits; ++sp) {
        const size_t idx = (size_t(sp) * MTP_H + size_t(qh)) * MTP_B + size_t(qt);
        const float ms = part_ml[idx * 2 + 0];
        const float ls = part_ml[idx * 2 + 1];
        if (ls > 0.0f) {
            const float w = metal::precise::exp2(ms - m_max);
            l_tot += w * ls;
            const device float* po = part_o + idx * MTP_HD + col;
            #pragma clang loop unroll(full)
            for (int e = 0; e < E; ++e) o_tot[e] += w * po[e];
        }
    }

    device bfloat* dst = attn_out + size_t(qt) * size_t(MTP_H * MTP_HD) + size_t(qh) * MTP_HD + col;
    const float inv_l = 1.0f / l_tot;
    #pragma clang loop unroll(full)
    for (int e = 0; e < E; ++e) dst[e] = bfloat(o_tot[e] * inv_l);
}

struct MarkovCtxPrm {
    int D;
    int out_dim;
};

[[kernel, max_total_threads_per_threadgroup(32)]]
kernel void mtp_markov_context_p1(
    const device bfloat* x_out      [[buffer(0)]],
    const device bfloat* f_rnn      [[buffer(1)]],
    const device float* w_ctx_t     [[buffer(2)]],
    device float* part_x            [[buffer(3)]],
    device float* part_f            [[buffer(4)]],
    constant MarkovCtxPrm& p        [[buffer(5)]],
    uint2 tgid                      [[threadgroup_position_in_grid]],
    uint lane                       [[thread_index_in_simdgroup]])
{
    const int col = int(tgid.x) * 32 + int(lane);
    const int sp = int(tgid.y);
    if (col >= p.out_dim) return;

    const int chunk = p.D / 8;
    if (sp < 8) {
        const int k0 = sp * chunk;
        const int k1 = k0 + chunk;
        float acc[7];
        #pragma clang loop unroll(full)
        for (int m = 0; m < FC_M; ++m) acc[m] = 0.0f;

        for (int k = k0; k < k1; k += 4) {
            const float w0 = w_ctx_t[size_t(k + 0) * size_t(p.out_dim) + size_t(col)];
            const float w1 = w_ctx_t[size_t(k + 1) * size_t(p.out_dim) + size_t(col)];
            const float w2 = w_ctx_t[size_t(k + 2) * size_t(p.out_dim) + size_t(col)];
            const float w3 = w_ctx_t[size_t(k + 3) * size_t(p.out_dim) + size_t(col)];
            #pragma clang loop unroll(full)
            for (int m = 0; m < FC_M; ++m) {
                const vec<bfloat, 4> xv = *(const device vec<bfloat, 4>*)(x_out + size_t(m) * size_t(p.D) + size_t(k));
                acc[m] += float(xv[0]) * w0 + float(xv[1]) * w1 + float(xv[2]) * w2 + float(xv[3]) * w3;
            }
        }
        #pragma clang loop unroll(full)
        for (int m = 0; m < FC_M; ++m) {
            part_x[(size_t(sp) * size_t(FC_M) + size_t(m)) * size_t(p.out_dim) + size_t(col)] = acc[m];
        }
    } else {
        const int fsp = sp - 8;
        const int k0 = fsp * chunk;
        const int k1 = k0 + chunk;
        const device float* wf_base = w_ctx_t + size_t(p.D) * size_t(p.out_dim);
        float acc_f = 0.0f;
        for (int k = k0; k < k1; k += 4) {
            const vec<bfloat, 4> fv = *(const device vec<bfloat, 4>*)(f_rnn + size_t(k));
            acc_f += float(fv[0]) * wf_base[size_t(k + 0) * size_t(p.out_dim) + size_t(col)]
                   + float(fv[1]) * wf_base[size_t(k + 1) * size_t(p.out_dim) + size_t(col)]
                   + float(fv[2]) * wf_base[size_t(k + 2) * size_t(p.out_dim) + size_t(col)]
                   + float(fv[3]) * wf_base[size_t(k + 3) * size_t(p.out_dim) + size_t(col)];
        }
        part_f[size_t(fsp) * size_t(p.out_dim) + size_t(col)] = acc_f;
    }
}

[[kernel, max_total_threads_per_threadgroup(32)]]
kernel void mtp_markov_context_p2(
    const device float* part_x      [[buffer(0)]],
    const device float* part_f      [[buffer(1)]],
    const device float* rnn_b       [[buffer(2)]],
    device float* ctx_out           [[buffer(3)]],
    constant MarkovCtxPrm& p        [[buffer(4)]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint lane                       [[thread_index_in_simdgroup]])
{
    const int col = int(tgid) * 32 + int(lane);
    if (col >= p.out_dim) return;

    float f_tot = rnn_b[col];
    #pragma clang loop unroll(full)
    for (int sp = 0; sp < 8; ++sp) {
        f_tot += part_f[size_t(sp) * size_t(p.out_dim) + size_t(col)];
    }

    #pragma clang loop unroll(full)
    for (int m = 0; m < FC_M; ++m) {
        float x_tot = f_tot;
        #pragma clang loop unroll(full)
        for (int sp = 0; sp < 8; ++sp) {
            x_tot += part_x[(size_t(sp) * size_t(FC_M) + size_t(m)) * size_t(p.out_dim) + size_t(col)];
        }
        ctx_out[size_t(m) * size_t(p.out_dim) + size_t(col)] = x_tot;
    }
}

struct RnnStepPrm {
    int step_idx;
    int rank;
};

[[kernel, max_total_threads_per_threadgroup(32)]]
kernel void mtp_rnn_step(
    const device bfloat* state_in   [[buffer(0)]],
    const device int* prev_tok      [[buffer(1)]],
    const device bfloat* markov_w1  [[buffer(2)]],
    const device float* w_step_t    [[buffer(3)]],
    const device float* rnn_ctx     [[buffer(4)]],
    device bfloat* state_out        [[buffer(5)]],
    device bfloat* latent_out       [[buffer(6)]],
    constant RnnStepPrm& p          [[buffer(7)]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint lane                       [[thread_index_in_simdgroup]])
{
    const int rank = MTP_RANK;
    const int out_dim = 3 * rank;
    const int j = int(tgid) * 32 + int(lane);

    threadgroup float tg_in[2 * MTP_RANK];
    const int tok = prev_tok[0];
    const device bfloat* emb = markov_w1 + size_t(tok) * size_t(rank);

    for (int i = int(lane); i < rank; i += 32) {
        tg_in[i] = float(state_in[i]);
        tg_in[rank + i] = float(emb[i]);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    for (int k = 0; k < 2 * rank; k += 4) {
        const float x0 = tg_in[k + 0], x1 = tg_in[k + 1], x2 = tg_in[k + 2], x3 = tg_in[k + 3];
        const device float* wk0 = w_step_t + size_t(k + 0) * size_t(out_dim);
        const device float* wk1 = wk0 + out_dim;
        const device float* wk2 = wk1 + out_dim;
        const device float* wk3 = wk2 + out_dim;
        acc0 += x0 * wk0[j] + x1 * wk1[j] + x2 * wk2[j] + x3 * wk3[j];
        acc1 += x0 * wk0[rank + j] + x1 * wk1[rank + j] + x2 * wk2[rank + j] + x3 * wk3[rank + j];
        acc2 += x0 * wk0[2 * rank + j] + x1 * wk1[2 * rank + j] + x2 * wk2[2 * rank + j] + x3 * wk3[2 * rank + j];
    }

    const device float* ctx_i = rnn_ctx + size_t(p.step_idx) * size_t(out_dim);
    const float r0 = float(bfloat(acc0 + ctx_i[j]));
    const float r1 = float(bfloat(acc1 + ctx_i[rank + j]));
    const float r2 = float(bfloat(acc2 + ctx_i[2 * rank + j]));

    const float gate = float(bfloat(1.0f / (1.0f + metal::precise::exp(-r0))));
    const float cand = float(bfloat(metal::precise::tanh(r1)));
    const float one_minus_g = float(bfloat(1.0f - gate));
    const float old_s = tg_in[j];
    const bfloat new_s = bfloat(float(bfloat(gate * old_s)) + float(bfloat(one_minus_g * cand)));
    const bfloat lat = bfloat(metal::precise::tanh(r2));

    state_out[j] = new_s;
    latent_out[j] = lat;
}

inline float markov_dot_row(const device bfloat* fold_w, int v, const threadgroup float* lat, uint lane) {
    constexpr int KPL = MTP_RANK / 32;
    const int k = int(lane) * KPL;
    const device bfloat* wp_b = fold_w + size_t(v) * size_t(MTP_RANK) + k;
    float dot_v = 0.0f;
#if MTP_RANK == 256
    const device vec<bfloat, 4>* wp = (const device vec<bfloat, 4>*)(wp_b);
    const vec<bfloat, 4> wa = wp[0];
    const vec<bfloat, 4> wb = wp[1];
    dot_v = float(wa[0]) * lat[k + 0] + float(wa[1]) * lat[k + 1]
          + float(wa[2]) * lat[k + 2] + float(wa[3]) * lat[k + 3]
          + float(wb[0]) * lat[k + 4] + float(wb[1]) * lat[k + 5]
          + float(wb[2]) * lat[k + 6] + float(wb[3]) * lat[k + 7];
#elif (MTP_RANK % 128) == 0
    const device vec<bfloat, 4>* wp = (const device vec<bfloat, 4>*)(wp_b);
    for (int i = 0; i < KPL / 4; ++i) {
        const vec<bfloat, 4> wv = wp[i];
        const int off = k + i * 4;
        dot_v += float(wv[0]) * lat[off + 0] + float(wv[1]) * lat[off + 1]
               + float(wv[2]) * lat[off + 2] + float(wv[3]) * lat[off + 3];
    }
#else
    #pragma clang loop unroll(full)
    for (int i = 0; i < KPL; ++i) {
        dot_v += float(wp_b[i]) * lat[k + i];
    }
#endif
    return simd_sum(dot_v);
}

struct MarkovArgmaxPrm {
    int v_limit;
    int step_idx;
    int n_part;
};

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mtp_markov_argmax_p1(
    const device bfloat* latent  [[buffer(0)]],
    const device bfloat* fold_w  [[buffer(1)]],
    const device bfloat* fold_b  [[buffer(2)]],
    const device bfloat* base_l  [[buffer(3)]],
    device float* part_val       [[buffer(4)]],
    device uint* part_idx        [[buffer(5)]],
    constant MarkovArgmaxPrm& p  [[buffer(6)]],
    uint tgid                    [[threadgroup_position_in_grid]],
    uint tid                     [[thread_position_in_threadgroup]],
    uint lane                    [[thread_index_in_simdgroup]],
    uint sg                      [[simdgroup_index_in_threadgroup]])
{
    constexpr int SG = 8;
    constexpr int ROWS_PER_SG = 8;
    const int v_base = int(tgid) * (SG * ROWS_PER_SG);

    threadgroup float lat[MTP_RANK];
    for (int i = int(tid); i < MTP_RANK; i += 256) {
        lat[i] = float(latent[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const device bfloat* base_row = base_l + size_t(p.step_idx) * size_t(p.v_limit);

    float best_val = -INFINITY;
    uint best_idx = 0u;

    #pragma clang loop unroll(full)
    for (int ri = 0; ri < ROWS_PER_SG; ++ri) {
        const int v = v_base + int(sg) * ROWS_PER_SG + ri;
        if (v < p.v_limit) {
            const float total = markov_dot_row(fold_w, v, lat, lane);
            if (lane == 0) {
                const float score = float(bfloat(float(base_row[v]) + float(bfloat(total + float(fold_b[v])))));
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
        part_val[tgid] = bv;
        part_idx[tgid] = bi;
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mtp_markov_argmax_p2(
    const device float* part_val [[buffer(0)]],
    const device uint* part_idx  [[buffer(1)]],
    device int* d_toks           [[buffer(2)]],
    device int* prev_tok         [[buffer(3)]],
    device int* verify_toks      [[buffer(4)]],
    constant MarkovArgmaxPrm& p  [[buffer(5)]],
    uint tid                     [[thread_position_in_threadgroup]],
    uint lane                    [[thread_index_in_simdgroup]],
    uint sg                      [[simdgroup_index_in_threadgroup]])
{
    float bv = -INFINITY;
    uint bi = 0u;
    for (int i = int(tid); i < p.n_part; i += 256) {
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
    if (lane == 0) {
        sg_v[sg] = bv;
        sg_i[sg] = bi;
    }
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
        d_toks[p.step_idx] = int(fi);
        prev_tok[0] = int(fi);
        verify_toks[1 + p.step_idx] = int(fi);
    }
}

struct MarkovLogitsPrm {
    int v_limit;
    int step_idx;
    int n_part;
    float temperature;
    float softcap;
};

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mtp_markov_logits_max_p1(
    const device bfloat* latent  [[buffer(0)]],
    const device bfloat* fold_w  [[buffer(1)]],
    const device bfloat* fold_b  [[buffer(2)]],
    const device bfloat* base_l  [[buffer(3)]],
    device float* q_logits_out   [[buffer(4)]],
    device float* part_max       [[buffer(5)]],
    constant MarkovLogitsPrm& p  [[buffer(6)]],
    uint tgid                    [[threadgroup_position_in_grid]],
    uint tid                     [[thread_position_in_threadgroup]],
    uint lane                    [[thread_index_in_simdgroup]],
    uint sg                      [[simdgroup_index_in_threadgroup]])
{
    constexpr int SG = 8;
    constexpr int ROWS_PER_SG = 8;
    const int v_base = int(tgid) * (SG * ROWS_PER_SG);

    threadgroup float lat[MTP_RANK];
    for (int i = int(tid); i < MTP_RANK; i += 256) {
        lat[i] = float(latent[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const device bfloat* base_row = base_l + size_t(p.step_idx) * size_t(p.v_limit);
    device float* out_row = q_logits_out + size_t(p.step_idx) * size_t(p.v_limit);

    float best_val = -INFINITY;
    #pragma clang loop unroll(full)
    for (int ri = 0; ri < ROWS_PER_SG; ++ri) {
        const int v = v_base + int(sg) * ROWS_PER_SG + ri;
        if (v < p.v_limit) {
            const float total = markov_dot_row(fold_w, v, lat, lane);
            if (lane == 0) {
                const float raw = float(bfloat(float(base_row[v]) + float(bfloat(total + float(fold_b[v])))));
                float val = (p.softcap > 0.0f) ? (p.softcap * metal::precise::tanh(raw / p.softcap)) : raw;
                val /= p.temperature;
                out_row[v] = val;
                if (val > best_val) best_val = val;
            }
        }
    }

    threadgroup float sg_val[SG];
    if (lane == 0) sg_val[sg] = best_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        float bv = sg_val[0];
        for (int s = 1; s < SG; ++s) bv = max(bv, sg_val[s]);
        part_max[tgid] = bv;
    }
}

struct SoftmaxRedPrm {
    int v_limit;
    int n_part;
    int row_off;
};

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mtp_softmax_max_p2(
    const device float* part_max [[buffer(0)]],
    device float* row_max        [[buffer(1)]],
    constant SoftmaxRedPrm& p    [[buffer(2)]],
    uint2 tgid                   [[threadgroup_position_in_grid]],
    uint2 tid2                   [[thread_position_in_threadgroup]],
    uint lane                    [[thread_index_in_simdgroup]],
    uint sg                      [[simdgroup_index_in_threadgroup]])
{
    const uint tid = tid2.x;
    const size_t base = size_t(tgid.y) * size_t(p.n_part);
    float bv = -INFINITY;
    for (int i = int(tid); i < p.n_part; i += 256) {
        bv = max(bv, part_max[base + size_t(i)]);
    }
    bv = simd_max(bv);
    threadgroup float sg_v[8];
    if (lane == 0) sg_v[sg] = bv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float tot = simd_max(lane < 8u ? sg_v[lane] : -INFINITY);
        if (lane == 0) row_max[p.row_off + int(tgid.y)] = tot;
    }
}

struct SoftmaxVecPrm {
    int v_limit;
    int n_tg;
    int row_off;
    int pass_idx;
    float top_p;
};

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mtp_softmax_exp_sum_p1(
    device float* probs          [[buffer(0)]],
    const device float* row_max  [[buffer(1)]],
    device float* part_sum       [[buffer(2)]],
    constant SoftmaxVecPrm& p    [[buffer(3)]],
    uint2 tgid                   [[threadgroup_position_in_grid]],
    uint2 tid2                   [[thread_position_in_threadgroup]],
    uint lane                    [[thread_index_in_simdgroup]],
    uint sg                      [[simdgroup_index_in_threadgroup]])
{
    const uint tid = tid2.x;
    const int m = p.row_off + int(tgid.y);
    const int tg = int(tgid.x);
    const int chunk = (p.v_limit + p.n_tg - 1) / p.n_tg;
    const int v_beg = tg * chunk;
    const int v_end = min(p.v_limit, v_beg + chunk);
    const float mx = row_max[m];
    device float* pr = probs + size_t(m) * size_t(p.v_limit);

    float s = 0.0f;
    for (int base = v_beg; base < v_end; base += 256) {
        const int v = base + int(tid);
        if (v < v_end) {
            const float e = metal::precise::exp(pr[v] - mx);
            pr[v] = e;
            s += e;
        }
    }
    s = simd_sum(s);
    threadgroup float sg_s[8];
    if (lane == 0) sg_s[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float tot = simd_sum(lane < 8u ? sg_s[lane] : 0.0f);
        if (lane == 0) part_sum[size_t(tgid.y) * size_t(p.n_tg) + size_t(tg)] = tot;
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mtp_softmax_sum_p2(
    const device float* part_sum [[buffer(0)]],
    device float* row_inv_sum    [[buffer(1)]],
    constant SoftmaxRedPrm& p    [[buffer(2)]],
    uint2 tgid                   [[threadgroup_position_in_grid]],
    uint2 tid2                   [[thread_position_in_threadgroup]],
    uint lane                    [[thread_index_in_simdgroup]],
    uint sg                      [[simdgroup_index_in_threadgroup]])
{
    const uint tid = tid2.x;
    const size_t base = size_t(tgid.y) * size_t(p.n_part);
    float s = 0.0f;
    for (int i = int(tid); i < p.n_part; i += 256) {
        s += part_sum[base + size_t(i)];
    }
    s = simd_sum(s);
    threadgroup float sg_s[8];
    if (lane == 0) sg_s[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float tot = simd_sum(lane < 8u ? sg_s[lane] : 0.0f);
        if (lane == 0) row_inv_sum[p.row_off + int(tgid.y)] = (tot > 0.0f) ? (1.0f / tot) : 0.0f;
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mtp_probs_scale(
    device float* probs            [[buffer(0)]],
    const device float* row_inv_sum[[buffer(1)]],
    constant SoftmaxVecPrm& p      [[buffer(2)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint2 tid2                     [[thread_position_in_threadgroup]])
{
    const uint tid = tid2.x;
    const int m = p.row_off + int(tgid.y);
    const int tg = int(tgid.x);
    const int chunk = (p.v_limit + p.n_tg - 1) / p.n_tg;
    const int v_beg = tg * chunk;
    const int v_end = min(p.v_limit, v_beg + chunk);
    const float inv = row_inv_sum[m];
    device float* pr = probs + size_t(m) * size_t(p.v_limit);
    for (int base = v_beg; base < v_end; base += 256) {
        const int v = base + int(tid);
        if (v < v_end) pr[v] *= inv;
    }
}

struct TopPState {
    uint prefix_bits;
    float sum_gt;
    uint thresh_bits;
    uint tie_keep_count;
    uint found_all;
    uint _pad[3];
    uint tie_base[64];
};

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mtp_topp_hist(
    device float* probs            [[buffer(0)]],
    const device float* row_inv_sum[[buffer(1)]],
    const device TopPState* states [[buffer(2)]],
    device float* part_sum         [[buffer(3)]],
    device uint* part_cnt          [[buffer(4)]],
    constant SoftmaxVecPrm& p      [[buffer(5)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint2 tid2                     [[thread_position_in_threadgroup]],
    uint lane                      [[thread_index_in_simdgroup]],
    uint sg                        [[simdgroup_index_in_threadgroup]])
{
    const uint tid = tid2.x;
    const int m = p.row_off + int(tgid.y);
    const int tg = int(tgid.x);
    const TopPState st = states[m];
    if (p.pass_idx > 0 && st.found_all != 0u) return;

    const float inv = (p.pass_idx == 0) ? row_inv_sum[m] : 1.0f;
    const int shift = 25 - p.pass_idx * 5;
    const uint mask_shift = uint(shift + 5);
    const uint prefix_shifted = st.prefix_bits >> mask_shift;

    threadgroup float sg_sum[8][32];
    threadgroup uint  sg_cnt[8][32];
    sg_sum[tid >> 5][tid & 31u] = 0.0f;
    sg_cnt[tid >> 5][tid & 31u] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int chunk = (p.v_limit + p.n_tg - 1) / p.n_tg;
    const int v_beg = tg * chunk;
    const int v_end = min(p.v_limit, v_beg + chunk);
    device float* pr = probs + size_t(m) * size_t(p.v_limit);

    for (int base = v_beg; base < v_end; base += 256) {
        const int v = base + int(tid);
        const bool in_range = (v < v_end);
        float pv = 0.0f;
        bool match = false;
        uint bin = 0u;
        if (in_range) {
            pv = pr[v];
            if (p.pass_idx == 0) {
                pv *= inv;
                pr[v] = pv;
            }
            const uint bits = (pv <= 0.0f) ? 0u : as_type<uint>(pv);
            match = (p.pass_idx == 0) || ((bits >> mask_shift) == prefix_shifted);
            bin = (p.pass_idx == 0) ? min(31u, bits >> 25) : ((bits >> uint(shift)) & 31u);
        }
        if (simd_any(match)) {
            for (int b = 0; b < 32; ++b) {
                const bool hit = match && (bin == uint(b));
                if (simd_any(hit)) {
                    const float bs = simd_sum(hit ? pv : 0.0f);
                    const uint  bc = simd_sum(hit ? 1u : 0u);
                    if (lane == 0) {
                        sg_sum[sg][b] += bs;
                        sg_cnt[sg][b] += bc;
                    }
                }
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float fs = 0.0f;
        uint fc = 0u;
        #pragma clang loop unroll(full)
        for (int s = 0; s < 8; ++s) {
            fs += sg_sum[s][lane];
            fc += sg_cnt[s][lane];
        }
        const size_t out_idx = (size_t(tgid.y) * size_t(p.n_tg) + size_t(tg)) * 32u + size_t(lane);
        part_sum[out_idx] = fs;
        part_cnt[out_idx] = fc;
    }
}

[[kernel, max_total_threads_per_threadgroup(32)]]
kernel void mtp_topp_scan(
    const device float* part_sum [[buffer(0)]],
    const device uint* part_cnt  [[buffer(1)]],
    device TopPState* states     [[buffer(2)]],
    constant SoftmaxVecPrm& p    [[buffer(3)]],
    uint2 tgid                   [[threadgroup_position_in_grid]],
    uint lane                    [[thread_index_in_simdgroup]])
{
    const int m = p.row_off + int(tgid.y);
    if (p.pass_idx > 0 && states[m].found_all != 0u) return;

    float b_sum = 0.0f;
    uint b_cnt = 0u;
    const size_t base = size_t(tgid.y) * size_t(p.n_tg);
    for (int tg = 0; tg < p.n_tg; ++tg) {
        const size_t idx = (base + size_t(tg)) * 32u + size_t(lane);
        b_sum += part_sum[idx];
        b_cnt += part_cnt[idx];
    }

    const float base_gt = (p.pass_idx == 0) ? 0.0f : states[m].sum_gt;
    const uint rev = 31u - lane;
    const float rev_sum = simd_shuffle(b_sum, rev);
    const float rev_excl = simd_prefix_exclusive_sum(rev_sum);
    const float sum_above = base_gt + simd_shuffle(rev_excl, rev);
    const float sum_incl = sum_above + b_sum;

    if (p.pass_idx == 0) {
        const float total_sum = simd_shuffle(sum_incl, 0u);
        if (total_sum < p.top_p) {
            if (lane == 0) {
                states[m].found_all = 1u;
                states[m].thresh_bits = 0u;
                states[m].tie_keep_count = 0u;
            }
            return;
        }
    }

    const bool winner = (b_cnt > 0u) && (sum_above < p.top_p) && (sum_incl >= p.top_p);
    int win_lane = simd_max(winner ? int(lane) : -1);
    if (win_lane < 0) {
        if (base_gt >= p.top_p) {
            win_lane = simd_max((b_cnt > 0u) ? int(lane) : -1);
        } else {
            win_lane = simd_min((b_cnt > 0u) ? int(lane) : 32);
        }
        if (win_lane < 0 || win_lane >= 32) win_lane = 0;
    }

    const uint b_star = uint(win_lane);
    const int shift = 25 - p.pass_idx * 5;
    const uint prev_pref = (p.pass_idx == 0) ? 0u : states[m].prefix_bits;
    const uint new_prefix = prev_pref | (b_star << uint(shift));
    const float new_sum_gt = simd_shuffle(sum_above, b_star);
    const uint tot_ties = simd_shuffle(b_cnt, b_star);

    if (p.pass_idx == 5) {
        const uint c0 = (int(lane) < p.n_tg) ? part_cnt[(base + size_t(lane)) * 32u + size_t(b_star)] : 0u;
        const uint p0 = simd_prefix_exclusive_sum(c0);
        const uint t0 = simd_sum(c0);
        const uint c1 = (int(32u + lane) < p.n_tg) ? part_cnt[(base + size_t(32u + lane)) * 32u + size_t(b_star)] : 0u;
        const uint p1 = t0 + simd_prefix_exclusive_sum(c1);
        if (int(lane) < p.n_tg) states[m].tie_base[lane] = p0;
        if (int(32u + lane) < p.n_tg) states[m].tie_base[32u + lane] = p1;

        if (lane == 0) {
            states[m].prefix_bits = new_prefix;
            states[m].sum_gt = new_sum_gt;
            states[m].thresh_bits = new_prefix;
            states[m].found_all = 0u;
            const float t_val = as_type<float>(new_prefix);
            if (t_val <= 0.0f || new_sum_gt + float(tot_ties) * t_val < p.top_p) {
                states[m].tie_keep_count = tot_ties;
            } else {
                uint k = uint(max(0.0f, floor((p.top_p - new_sum_gt) / t_val)));
                k = min(k, tot_ties);
                while (k < tot_ties && (new_sum_gt + float(k) * t_val) < p.top_p) ++k;
                while (k > 0u && !((new_sum_gt + float(k - 1u) * t_val) < p.top_p)) --k;
                states[m].tie_keep_count = k;
            }
        }
    } else {
        if (lane == 0) {
            states[m].prefix_bits = new_prefix;
            states[m].sum_gt = new_sum_gt;
            states[m].found_all = 0u;
        }
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mtp_topp_mask_sum(
    device float* probs            [[buffer(0)]],
    const device TopPState* states [[buffer(1)]],
    device float* part_sum         [[buffer(2)]],
    constant SoftmaxVecPrm& p      [[buffer(3)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint2 tid2                     [[thread_position_in_threadgroup]],
    uint lane                      [[thread_index_in_simdgroup]],
    uint sg                        [[simdgroup_index_in_threadgroup]])
{
    const uint tid = tid2.x;
    const int m = p.row_off + int(tgid.y);
    const int tg = int(tgid.x);
    const TopPState st = states[m];
    const uint thresh_bits = st.thresh_bits;
    uint local_tie_off = st.tie_base[tg];

    const int chunk = (p.v_limit + p.n_tg - 1) / p.n_tg;
    const int v_beg = tg * chunk;
    const int v_end = min(p.v_limit, v_beg + chunk);
    device float* pr = probs + size_t(m) * size_t(p.v_limit);

    float sum_kept = 0.0f;
    threadgroup uint sg_cnts[8];

    for (int base = v_beg; base < v_end; base += 256) {
        const int v = base + int(tid);
        const bool in_range = (v < v_end);
        float pv = in_range ? pr[v] : 0.0f;
        const uint bits = (pv <= 0.0f) ? 0u : as_type<uint>(pv);
        const bool is_tie = (st.found_all == 0u) && in_range && (bits == thresh_bits);

        const uint t_flag = is_tie ? 1u : 0u;
        const uint sg_inc = simd_prefix_inclusive_sum(t_flag);
        const uint sg_exc = sg_inc - t_flag;
        if (lane == 31) sg_cnts[sg] = sg_inc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint tg_base_rank = 0u;
        for (uint s = 0u; s < sg; ++s) tg_base_rank += sg_cnts[s];
        uint total_chunk = 0u;
        #pragma clang loop unroll(full)
        for (uint s = 0u; s < 8u; ++s) total_chunk += sg_cnts[s];

        const uint my_tie_rank = local_tie_off + tg_base_rank + sg_exc;
        local_tie_off += total_chunk;

        const bool keep = (st.found_all != 0u) || (bits > thresh_bits) || (is_tie && my_tie_rank < st.tie_keep_count);
        if (in_range && !keep) {
            pv = 0.0f;
            pr[v] = 0.0f;
        }
        sum_kept += pv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    sum_kept = simd_sum(sum_kept);
    threadgroup float sg_s[8];
    if (lane == 0) sg_s[sg] = sum_kept;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float tot = simd_sum(lane < 8u ? sg_s[lane] : 0.0f);
        if (lane == 0) part_sum[size_t(tgid.y) * size_t(p.n_tg) + size_t(tg)] = tot;
    }
}

struct ScaleGumbelPrm {
    ulong rng_seed;
    ulong rng_step;
    int v_limit;
    int n_tg;
    int step_idx;
    int _pad;
};

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mtp_probs_scale_gumbel_p1(
    device float* probs            [[buffer(0)]],
    const device float* row_inv_sum[[buffer(1)]],
    device float* part_val         [[buffer(2)]],
    device uint* part_idx          [[buffer(3)]],
    constant ScaleGumbelPrm& p     [[buffer(4)]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]],
    uint lane                      [[thread_index_in_simdgroup]],
    uint sg                        [[simdgroup_index_in_threadgroup]])
{
    const int s = p.step_idx;
    const int tg = int(tgid);
    const int chunk = (p.v_limit + p.n_tg - 1) / p.n_tg;
    const int v_beg = tg * chunk;
    const int v_end = min(p.v_limit, v_beg + chunk);
    const float inv = row_inv_sum[s];
    device float* pr = probs + size_t(s) * size_t(p.v_limit);

    float bv = -INFINITY;
    uint bi = 0u;
    for (int base = v_beg; base < v_end; base += 256) {
        const int v = base + int(tid);
        if (v < v_end) {
            const float pv = pr[v] * inv;
            pr[v] = pv;
            if (pv > 0.0f) {
                const float u = philox_u01(p.rng_seed, p.rng_step, uint(s), uint(v));
                const float sc = metal::precise::log(pv) - metal::precise::log(-metal::precise::log(u));
                if (sc > bv || (sc == bv && uint(v) < bi)) {
                    bv = sc;
                    bi = uint(v);
                }
            }
        }
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
        float fv = sg_v[0];
        uint fi = sg_i[0];
        for (int k = 1; k < 8; ++k) {
            if (sg_v[k] > fv || (sg_v[k] == fv && sg_i[k] < fi)) { fv = sg_v[k]; fi = sg_i[k]; }
        }
        part_val[tg] = fv;
        part_idx[tg] = fi;
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mtp_gumbel_p2(
    const device float* part_val [[buffer(0)]],
    const device uint* part_idx  [[buffer(1)]],
    device int* d_toks           [[buffer(2)]],
    device int* prev_tok         [[buffer(3)]],
    device int* verify_toks      [[buffer(4)]],
    constant ScaleGumbelPrm& p   [[buffer(5)]],
    uint tid                     [[thread_position_in_threadgroup]],
    uint lane                    [[thread_index_in_simdgroup]],
    uint sg                      [[simdgroup_index_in_threadgroup]])
{
    float bv = -INFINITY;
    uint bi = 0u;
    for (int i = int(tid); i < p.n_tg; i += 256) {
        const float v = part_val[i];
        const uint idx = part_idx[i];
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
        float fv = sg_v[0];
        uint fi = sg_i[0];
        for (int k = 1; k < 8; ++k) {
            if (sg_v[k] > fv || (sg_v[k] == fv && sg_i[k] < fi)) { fv = sg_v[k]; fi = sg_i[k]; }
        }
        d_toks[p.step_idx] = int(fi);
        prev_tok[0] = int(fi);
        verify_toks[1 + p.step_idx] = int(fi);
    }
}

struct GreedyAcceptPrm {
    int num_draft;
};

[[kernel, max_total_threads_per_threadgroup(32)]]
kernel void mtp_greedy_accept(
    const device int* d_toks   [[buffer(0)]],
    const device int* t_toks   [[buffer(1)]],
    device int* out_toks       [[buffer(2)]],
    device int* d_n_acc        [[buffer(3)]],
    constant GreedyAcceptPrm& p[[buffer(4)]],
    uint tid                   [[thread_position_in_threadgroup]])
{
    if (tid != 0u) return;
    int match = 0;
    while (match < p.num_draft && d_toks[match] == t_toks[match]) {
        out_toks[match] = d_toks[match];
        ++match;
    }
    out_toks[match] = t_toks[match];
    d_n_acc[0] = match + 1;
}

struct RejectPrm {
    ulong rng_seed;
    ulong rng_step;
    int v_limit;
    int num_draft;
};

[[kernel, max_total_threads_per_threadgroup(32)]]
kernel void mtp_reject_prefix(
    const device int* d_toks     [[buffer(0)]],
    const device float* p_probs  [[buffer(1)]],
    const device float* q_probs  [[buffer(2)]],
    device int* out_toks         [[buffer(3)]],
    device int* d_reject_j       [[buffer(4)]],
    device int* d_n_acc          [[buffer(5)]],
    constant RejectPrm& p        [[buffer(6)]],
    uint tid                     [[thread_position_in_threadgroup]])
{
    if (tid != 0u) return;
    int rej = p.num_draft;
    for (int j = 0; j < p.num_draft; ++j) {
        const int dj = d_toks[j];
        const size_t idx = size_t(j) * size_t(p.v_limit) + size_t(dj);
        const float pd = p_probs[idx];
        const float qd = q_probs[idx];
        const float ratio = (qd <= 0.0f) ? 1.0f : min(1.0f, pd / qd);
        const float u = philox_u01(p.rng_seed, p.rng_step, 1000u + uint(j), 0u);
        if (u <= ratio) {
            out_toks[j] = dj;
        } else {
            rej = j;
            break;
        }
    }
    d_reject_j[0] = rej;
    d_n_acc[0] = rej + 1;
}

struct RejectSamplePrm {
    ulong rng_seed;
    ulong rng_step;
    int v_limit;
    int n_tg;
    int num_draft;
    int _pad;
};

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mtp_reject_sample_p1(
    const device float* p_probs  [[buffer(0)]],
    const device float* q_probs  [[buffer(1)]],
    const device int* d_reject_j [[buffer(2)]],
    device float* part_res_val   [[buffer(3)]],
    device uint* part_res_idx    [[buffer(4)]],
    device float* part_p_val     [[buffer(5)]],
    device uint* part_p_idx      [[buffer(6)]],
    device float* part_res_sum   [[buffer(7)]],
    constant RejectSamplePrm& p  [[buffer(8)]],
    uint tgid                    [[threadgroup_position_in_grid]],
    uint tid                     [[thread_position_in_threadgroup]],
    uint lane                    [[thread_index_in_simdgroup]],
    uint sg                      [[simdgroup_index_in_threadgroup]])
{
    const int rej = d_reject_j[0];
    const int tg = int(tgid);
    const int chunk = (p.v_limit + p.n_tg - 1) / p.n_tg;
    const int v_beg = tg * chunk;
    const int v_end = min(p.v_limit, v_beg + chunk);

    const device float* pr = p_probs + size_t(rej) * size_t(p.v_limit);
    const device float* qr = (rej < p.num_draft) ? (q_probs + size_t(rej) * size_t(p.v_limit)) : nullptr;

    float brv = -INFINITY, bpv = -INFINITY;
    uint bri = 0u, bpi = 0u;
    float s_res = 0.0f;

    for (int base = v_beg; base < v_end; base += 256) {
        const int v = base + int(tid);
        if (v < v_end) {
            const float pv = pr[v];
            const float rv = (rej < p.num_draft) ? max(0.0f, pv - qr[v]) : 0.0f;
            s_res += rv;
            if (pv > 0.0f || rv > 0.0f) {
                const float u = philox_u01(p.rng_seed, p.rng_step, 2000u, uint(v));
                const float g = -metal::precise::log(-metal::precise::log(u));
                if (pv > 0.0f) {
                    const float sp = metal::precise::log(pv) + g;
                    if (sp > bpv || (sp == bpv && uint(v) < bpi)) { bpv = sp; bpi = uint(v); }
                }
                if (rv > 0.0f) {
                    const float sr = metal::precise::log(rv) + g;
                    if (sr > brv || (sr == brv && uint(v) < bri)) { brv = sr; bri = uint(v); }
                }
            }
        }
    }

    const float ss_res = simd_sum(s_res);
    for (ushort off = 16; off > 0; off >>= 1) {
        const float orv = simd_shuffle_xor(brv, off);
        const uint ori = simd_shuffle_xor(bri, off);
        if (orv > brv || (orv == brv && ori < bri)) { brv = orv; bri = ori; }
        const float opv = simd_shuffle_xor(bpv, off);
        const uint opi = simd_shuffle_xor(bpi, off);
        if (opv > bpv || (opv == bpv && opi < bpi)) { bpv = opv; bpi = opi; }
    }

    threadgroup float sg_rv[8], sg_pv[8], sg_rs[8];
    threadgroup uint  sg_ri[8], sg_pi[8];
    if (lane == 0) {
        sg_rv[sg] = brv; sg_ri[sg] = bri;
        sg_pv[sg] = bpv; sg_pi[sg] = bpi;
        sg_rs[sg] = ss_res;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        float frv = sg_rv[0], fpv = sg_pv[0], tot_s = sg_rs[0];
        uint fri = sg_ri[0], fpi = sg_pi[0];
        for (int k = 1; k < 8; ++k) {
            tot_s += sg_rs[k];
            if (sg_rv[k] > frv || (sg_rv[k] == frv && sg_ri[k] < fri)) { frv = sg_rv[k]; fri = sg_ri[k]; }
            if (sg_pv[k] > fpv || (sg_pv[k] == fpv && sg_pi[k] < fpi)) { fpv = sg_pv[k]; fpi = sg_pi[k]; }
        }
        part_res_val[tg] = frv;
        part_res_idx[tg] = fri;
        part_p_val[tg]   = fpv;
        part_p_idx[tg]   = fpi;
        part_res_sum[tg] = tot_s;
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mtp_reject_sample_p2(
    const device float* part_res_val [[buffer(0)]],
    const device uint* part_res_idx  [[buffer(1)]],
    const device float* part_p_val   [[buffer(2)]],
    const device uint* part_p_idx    [[buffer(3)]],
    const device float* part_res_sum [[buffer(4)]],
    const device int* d_reject_j     [[buffer(5)]],
    device int* out_toks             [[buffer(6)]],
    constant RejectSamplePrm& p      [[buffer(7)]],
    uint tid                         [[thread_position_in_threadgroup]],
    uint lane                        [[thread_index_in_simdgroup]],
    uint sg                          [[simdgroup_index_in_threadgroup]])
{
    const int rej = d_reject_j[0];
    float s_res = 0.0f;
    float brv = -INFINITY, bpv = -INFINITY;
    uint bri = 0u, bpi = 0u;
    for (int i = int(tid); i < p.n_tg; i += 256) {
        s_res += part_res_sum[i];
        const float rv = part_res_val[i];
        const uint ri = part_res_idx[i];
        if (rv > brv || (rv == brv && ri < bri)) { brv = rv; bri = ri; }
        const float pv = part_p_val[i];
        const uint pi = part_p_idx[i];
        if (pv > bpv || (pv == bpv && pi < bpi)) { bpv = pv; bpi = pi; }
    }
    const float ss_res = simd_sum(s_res);
    for (ushort off = 16; off > 0; off >>= 1) {
        const float orv = simd_shuffle_xor(brv, off);
        const uint ori = simd_shuffle_xor(bri, off);
        if (orv > brv || (orv == brv && ori < bri)) { brv = orv; bri = ori; }
        const float opv = simd_shuffle_xor(bpv, off);
        const uint opi = simd_shuffle_xor(bpi, off);
        if (opv > bpv || (opv == bpv && opi < bpi)) { bpv = opv; bpi = opi; }
    }
    threadgroup float sg_rv[8], sg_pv[8], sg_rs[8];
    threadgroup uint  sg_ri[8], sg_pi[8];
    if (lane == 0) {
        sg_rv[sg] = brv; sg_ri[sg] = bri;
        sg_pv[sg] = bpv; sg_pi[sg] = bpi;
        sg_rs[sg] = ss_res;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        float frv = sg_rv[0], fpv = sg_pv[0], tot_s = sg_rs[0];
        uint fri = sg_ri[0], fpi = sg_pi[0];
        for (int k = 1; k < 8; ++k) {
            tot_s += sg_rs[k];
            if (sg_rv[k] > frv || (sg_rv[k] == frv && sg_ri[k] < fri)) { frv = sg_rv[k]; fri = sg_ri[k]; }
            if (sg_pv[k] > fpv || (sg_pv[k] == fpv && sg_pi[k] < fpi)) { fpv = sg_pv[k]; fpi = sg_pi[k]; }
        }
        const bool use_res = (rej < p.num_draft) && (tot_s > 1e-12f);
        out_toks[rej] = int(use_res ? fri : fpi);
    }
}

struct BatchMTPStreamMetaGpu {
    int32_t anchor_token;
    int32_t n_ctx;
    int32_t ctx_start;
    int32_t num_draft;
    int32_t d_start;
    int32_t v_start;
    int32_t is_sampled;
    int32_t do_top_p;
    float   temperature;
    float   top_p;
    float   softcap;
    float   _pad0;
    ulong   rng_seed;
    ulong   rng_step;
    ulong   out_hidden_addr;
};

struct BatchMTPDraftKVMetaGpu {
    ulong   k_addr;
    ulong   v_addr;
    int32_t offset_before;
    int32_t is_rotating;
    int32_t window;
    int32_t ring_start;
    int32_t valid_ctx;
    int32_t k_stride_h;
    int32_t k_stride_s;
    int32_t v_stride_h;
    int32_t v_stride_s;
    int32_t offset_after;
};

struct BatchInitXPrm {
    int B, D, block_size, markov_rank;
    float embed_norm_eps;
};

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mtp_batch_init_x(
    const device bfloat* embed_tokens        [[buffer(0)]],
    const device bfloat* embed_norm_w        [[buffer(1)]],
    const device bfloat* mask_embed          [[buffer(2)]],
    const device BatchMTPStreamMetaGpu* sm   [[buffer(3)]],
    device bfloat* x_out                     [[buffer(4)]],
    device bfloat* rnn_state                 [[buffer(5)]],
    device int* prev_tok                     [[buffer(6)]],
    device int* verify_toks                  [[buffer(7)]],
    constant BatchInitXPrm& p                [[buffer(8)]],
    uint tgid                                [[threadgroup_position_in_grid]],
    uint tid                                 [[thread_position_in_threadgroup]],
    uint lane                                [[thread_index_in_simdgroup]],
    uint sg                                  [[simdgroup_index_in_threadgroup]])
{
    const int b = int(tgid);
    if (b >= p.B) return;
    const BatchMTPStreamMetaGpu meta = sm[b];
    const device bfloat* emb_row = embed_tokens + size_t(meta.anchor_token) * size_t(p.D);
    device bfloat* xb = x_out + size_t(b * p.block_size) * size_t(p.D);

    if (FC_HAS_EMBED_NORM != 0) {
        float ss = 0.0f;
        for (int d = int(tid); d < p.D; d += 256) {
            const float v = float(emb_row[d]);
            ss += v * v;
        }
        ss = simd_sum(ss);
        threadgroup float sg_ss[8];
        if (lane == 0) sg_ss[sg] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = 0.0f;
        if (sg == 0 && lane < 8) tot = sg_ss[lane];
        tot = simd_sum(tot);
        if (tid == 0) sg_ss[0] = metal::precise::rsqrt(tot / float(p.D) + p.embed_norm_eps);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inv_rms = sg_ss[0];
        for (int d = int(tid); d < p.D; d += 256) {
            xb[d] = bfloat(float(bfloat(float(emb_row[d]) * inv_rms)) * float(embed_norm_w[d]));
        }
    } else {
        for (int d = int(tid); d < p.D; d += 256) {
            xb[d] = emb_row[d];
        }
    }

    for (int d = int(tid); d < p.D; d += 256) {
        const bfloat mval = mask_embed[d];
        for (int r = 1; r < p.block_size; ++r) {
            xb[size_t(r) * size_t(p.D) + d] = mval;
        }
    }

    device bfloat* rs = rnn_state + size_t(b) * size_t(p.markov_rank);
    for (int i = int(tid); i < p.markov_rank; i += 256) {
        rs[i] = bfloat(0.0f);
    }
    if (tid == 0) {
        prev_tok[b] = meta.anchor_token;
        verify_toks[meta.v_start] = meta.anchor_token;
    }
}

struct BatchGemvPrm {
    int M, K, N;
};

kernel void mtp_bf16_gemv_batched(
    const device bfloat* x  [[buffer(0)]],
    const device bfloat* w  [[buffer(1)]],
    device bfloat* out      [[buffer(2)]],
    constant BatchGemvPrm& p[[buffer(3)]],
    uint tgid               [[threadgroup_position_in_grid]],
    uint lane               [[thread_index_in_simdgroup]],
    uint sg                 [[simdgroup_index_in_threadgroup]])
{
    const int row = int(tgid) * 4 + int(sg);
    if (row >= p.N) return;
    const int K4 = p.K / 4;
    const device vec<bfloat, 4>* w4 = (const device vec<bfloat, 4>*)(w + size_t(row) * size_t(p.K));
    const device vec<bfloat, 4>* x4 = (const device vec<bfloat, 4>*)(x);

    for (int mc = 0; mc < p.M; mc += 8) {
        const int mn = min(8, p.M - mc);
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (int idx = int(lane); idx < K4; idx += 32) {
            const vec<float, 4> wv = vec<float, 4>(w4[idx]);
            for (int m = 0; m < mn; ++m) {
                acc[m] += dot(wv, vec<float, 4>(x4[(mc + m) * K4 + idx]));
            }
        }
        for (int m = 0; m < mn; ++m) {
            const float v = simd_sum(acc[m]);
            if (lane == 0) out[size_t(mc + m) * size_t(p.N) + size_t(row)] = bfloat(v);
        }
    }
}

kernel void mtp_bf16_gu_silu_batched(
    const device bfloat* x     [[buffer(0)]],
    const device bfloat* gu_w  [[buffer(1)]],
    device bfloat* out         [[buffer(2)]],
    constant BatchGemvPrm& p   [[buffer(3)]],
    uint tgid                  [[threadgroup_position_in_grid]],
    uint lane                  [[thread_index_in_simdgroup]],
    uint sg                    [[simdgroup_index_in_threadgroup]])
{
    const int row = int(tgid) * 4 + int(sg);
    if (row >= p.N) return;
    const int K4 = p.K / 4;
    const device vec<bfloat, 4>* wg4 = (const device vec<bfloat, 4>*)(gu_w + size_t(row) * size_t(p.K));
    const device vec<bfloat, 4>* wu4 = (const device vec<bfloat, 4>*)(gu_w + size_t(p.N + row) * size_t(p.K));
    const device vec<bfloat, 4>* x4  = (const device vec<bfloat, 4>*)(x);

    for (int mc = 0; mc < p.M; mc += 8) {
        const int mn = min(8, p.M - mc);
        float acc_g[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float acc_u[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (int idx = int(lane); idx < K4; idx += 32) {
            const vec<float, 4> wgv = vec<float, 4>(wg4[idx]);
            const vec<float, 4> wuv = vec<float, 4>(wu4[idx]);
            for (int m = 0; m < mn; ++m) {
                const vec<float, 4> xv = vec<float, 4>(x4[(mc + m) * K4 + idx]);
                acc_g[m] += dot(wgv, xv);
                acc_u[m] += dot(wuv, xv);
            }
        }
        for (int m = 0; m < mn; ++m) {
            const float g = float(bfloat(simd_sum(acc_g[m])));
            const float u = float(bfloat(simd_sum(acc_u[m])));
            if (lane == 0) {
                out[size_t(mc + m) * size_t(p.N) + size_t(row)] =
                    bfloat((g / (1.0f + metal::precise::exp(-g))) * u);
            }
        }
    }
}

struct BatchKvCtxWritePrm {
    int kv_dim, num_layers, layer_idx, _pad;
    float eps, neg_log2_theta_over_half;
};

[[kernel, max_total_threads_per_threadgroup(64)]]
kernel void mtp_batch_kv_ctx_norm_rope_write(
    const device bfloat* kv_ctx                  [[buffer(0)]],
    const device bfloat* k_norm_w                [[buffer(1)]],
    const device int* ctx_to_stream              [[buffer(2)]],
    const device int* ctx_local_pos              [[buffer(3)]],
    const device BatchMTPDraftKVMetaGpu* dkmeta  [[buffer(4)]],
    constant BatchKvCtxWritePrm& p               [[buffer(5)]],
    uint2 tgid                                   [[threadgroup_position_in_grid]],
    uint2 tid2                                   [[thread_position_in_threadgroup]],
    uint lane                                    [[thread_index_in_simdgroup]],
    uint sg                                      [[simdgroup_index_in_threadgroup]])
{
    constexpr int HALF_HD = MTP_HD / 2;
    constexpr int TG = (HALF_HD < 32) ? 32 : HALF_HD;
    constexpr int NS = TG / 32;
    const int c_idx = int(tgid.x);
    const int h     = int(tgid.y);
    const int d     = int(tid2.x);

    const int b = ctx_to_stream[c_idx];
    const int t = ctx_local_pos[c_idx];
    const BatchMTPDraftKVMetaGpu dm = dkmeta[b * p.num_layers + p.layer_idx];

    const device bfloat* row = kv_ctx + size_t(c_idx) * size_t(2 * p.kv_dim);
    const device bfloat* k_src = row + h * MTP_HD;
    const device bfloat* v_src = row + p.kv_dim + h * MTP_HD;

    const bool active = (d < HALF_HD);
    const float k0 = active ? float(k_src[d]) : 0.0f;
    const float k1 = active ? float(k_src[d + HALF_HD]) : 0.0f;

    float ss = simd_sum(k0 * k0 + k1 * k1);
    threadgroup float sg_ss[4];
    if (lane == 0) sg_ss[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot_ss = 0.0f;
    if (sg == 0 && int(lane) < NS) tot_ss = sg_ss[lane];
    tot_ss = simd_sum(tot_ss);
    if (d == 0) sg_ss[0] = metal::precise::rsqrt(tot_ss / float(MTP_HD) + p.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!active) return;

    const float inv_rms = sg_ss[0];
    const float kn0 = float(bfloat(float(bfloat(k0 * inv_rms)) * float(k_norm_w[d])));
    const float kn1 = float(bfloat(float(bfloat(k1 * inv_rms)) * float(k_norm_w[d + HALF_HD])));

    const float theta = float(dm.offset_before + t) * metal::exp2(p.neg_log2_theta_over_half * float(d));
    const float c = metal::precise::cos(theta);
    const float s = metal::precise::sin(theta);

    const int dst_slot = (dm.is_rotating != 0) ? ((dm.ring_start + t) % dm.window) : (dm.offset_before + t);
    device bfloat* k_dst = (device bfloat*)(dm.k_addr) + size_t(h) * size_t(dm.k_stride_h) + size_t(dst_slot) * size_t(dm.k_stride_s);
    device bfloat* v_dst = (device bfloat*)(dm.v_addr) + size_t(h) * size_t(dm.v_stride_h) + size_t(dst_slot) * size_t(dm.v_stride_s);

    k_dst[d]           = bfloat(kn0 * c - kn1 * s);
    k_dst[d + HALF_HD] = bfloat(kn1 * c + kn0 * s);
    v_dst[d]           = v_src[d];
    v_dst[d + HALF_HD] = v_src[d + HALF_HD];
}

struct BatchQkvPackPrm {
    int q_dim, kv_dim, total_dim, num_layers, layer_idx, _pad;
    float eps, neg_log2_theta_over_half;
};

[[kernel, max_total_threads_per_threadgroup(64)]]
kernel void mtp_batch_qkv_norm_rope_pack(
    const device bfloat* qkv                     [[buffer(0)]],
    const device bfloat* q_norm_w                [[buffer(1)]],
    const device bfloat* k_norm_w                [[buffer(2)]],
    const device BatchMTPDraftKVMetaGpu* dkmeta  [[buffer(3)]],
    device bfloat* q_out                         [[buffer(4)]],
    device bfloat* bk_out                        [[buffer(5)]],
    device bfloat* bv_out                        [[buffer(6)]],
    constant BatchQkvPackPrm& p                  [[buffer(7)]],
    uint2 tgid                                   [[threadgroup_position_in_grid]],
    uint2 tid2                                   [[thread_position_in_threadgroup]],
    uint lane                                    [[thread_index_in_simdgroup]],
    uint sg                                      [[simdgroup_index_in_threadgroup]])
{
    constexpr int HALF_HD = MTP_HD / 2;
    constexpr int TG = (HALF_HD < 32) ? 32 : HALF_HD;
    constexpr int NS = TG / 32;
    const int t_glob = int(tgid.x);
    const int b = t_glob / MTP_B;
    const int t = t_glob % MTP_B;
    const int idx = int(tgid.y);
    const int d = int(tid2.x);
    const bool active = (d < HALF_HD);

    const device bfloat* row = qkv + size_t(t_glob) * size_t(p.total_dim);

    if (idx >= MTP_H + MTP_KVH) {
        if (!active) return;
        const int h = idx - (MTP_H + MTP_KVH);
        const device bfloat* src = row + p.q_dim + p.kv_dim + h * MTP_HD;
        device bfloat* dst = bv_out + ((size_t(b) * MTP_KVH + size_t(h)) * MTP_B + size_t(t)) * MTP_HD;
        dst[d]           = src[d];
        dst[d + HALF_HD] = src[d + HALF_HD];
        return;
    }

    const BatchMTPDraftKVMetaGpu dm = dkmeta[b * p.num_layers + p.layer_idx];
    const bool is_q = (idx < MTP_H);
    const int h = is_q ? idx : (idx - MTP_H);
    const device bfloat* src = is_q ? (row + h * MTP_HD) : (row + p.q_dim + h * MTP_HD);
    const device bfloat* nw = is_q ? q_norm_w : k_norm_w;

    const float x0 = active ? float(src[d]) : 0.0f;
    const float x1 = active ? float(src[d + HALF_HD]) : 0.0f;

    float ss = simd_sum(x0 * x0 + x1 * x1);
    threadgroup float sg_ss[4];
    if (lane == 0) sg_ss[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot_ss = 0.0f;
    if (sg == 0 && int(lane) < NS) tot_ss = sg_ss[lane];
    tot_ss = simd_sum(tot_ss);
    if (d == 0) sg_ss[0] = metal::precise::rsqrt(tot_ss / float(MTP_HD) + p.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!active) return;

    const float inv_rms = sg_ss[0];
    const float n0 = float(bfloat(float(bfloat(x0 * inv_rms)) * float(nw[d])));
    const float n1 = float(bfloat(float(bfloat(x1 * inv_rms)) * float(nw[d + HALF_HD])));

    const float theta = float(dm.offset_after + t) * metal::exp2(p.neg_log2_theta_over_half * float(d));
    const float c = metal::precise::cos(theta);
    const float s = metal::precise::sin(theta);

    device bfloat* dst = is_q
        ? (q_out  + ((size_t(b) * MTP_H   + size_t(h)) * MTP_B + size_t(t)) * MTP_HD)
        : (bk_out + ((size_t(b) * MTP_KVH + size_t(h)) * MTP_B + size_t(t)) * MTP_HD);
    dst[d]           = bfloat(n0 * c - n1 * s);
    dst[d + HALF_HD] = bfloat(n1 * c + n0 * s);
}

struct BatchSdpaPrm {
    int num_layers, layer_idx, n_splits, _pad;
    float scale2;
};

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mtp_batch_sdpa_single(
    const device bfloat* q_buf                   [[buffer(0)]],
    const device bfloat* bk_buf                  [[buffer(1)]],
    const device bfloat* bv_buf                  [[buffer(2)]],
    const device BatchMTPDraftKVMetaGpu* dkmeta  [[buffer(3)]],
    device bfloat* attn_out                      [[buffer(4)]],
    constant BatchSdpaPrm& p                     [[buffer(5)]],
    uint2 tgid                                   [[threadgroup_position_in_grid]],
    uint tid                                     [[thread_index_in_threadgroup]],
    uint lane                                    [[thread_index_in_simdgroup]],
    uint sg                                      [[simdgroup_index_in_threadgroup]])
{
    constexpr int E = MTP_HD / 32;
    threadgroup bfloat K_tg[MTP_B * MTP_HD];
    threadgroup bfloat V_tg[MTP_B * MTP_HD];

    const int kv_h = int(tgid.x);
    const int b    = int(tgid.y);
    const BatchMTPDraftKVMetaGpu dm = dkmeta[b * p.num_layers + p.layer_idx];
    const int total_k = dm.valid_ctx + MTP_B;

    SdpaPrm sprm{
        dm.valid_ctx, total_k, dm.k_stride_h, dm.k_stride_s,
        dm.v_stride_h, dm.v_stride_s, 1, total_k, p.scale2
    };
    const device bfloat* qb  = q_buf  + size_t(b) * size_t(MTP_H * MTP_B * MTP_HD);
    const device bfloat* bkb = bk_buf + size_t(b) * size_t(MTP_KVH * MTP_B * MTP_HD);
    const device bfloat* bvb = bv_buf + size_t(b) * size_t(MTP_KVH * MTP_B * MTP_HD);

    float o[MTP_G][4];
    float m[MTP_G], l[MTP_G];
    sdpa_core(qb, (const device bfloat*)(dm.k_addr), (const device bfloat*)(dm.v_addr),
              bkb, bvb, sprm, kv_h, 0, total_k, tid, lane, sg, o, m, l, K_tg, V_tg);

    const int qt = int(sg);
    const int qh0 = kv_h * MTP_G;
    const int col = int(lane) * E;
    device bfloat* row_out = attn_out + size_t(b * MTP_B + qt) * size_t(MTP_H * MTP_HD);
    #pragma clang loop unroll(full)
    for (int g = 0; g < MTP_G; ++g) {
        const float r = 1.0f / l[g];
        device bfloat* dst = row_out + size_t(qh0 + g) * MTP_HD + col;
        #pragma clang loop unroll(full)
        for (int e = 0; e < E; ++e) dst[e] = bfloat(o[g][e] * r);
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mtp_batch_sdpa_split_p1(
    const device bfloat* q_buf                   [[buffer(0)]],
    const device bfloat* bk_buf                  [[buffer(1)]],
    const device bfloat* bv_buf                  [[buffer(2)]],
    const device BatchMTPDraftKVMetaGpu* dkmeta  [[buffer(3)]],
    device float* part_o                         [[buffer(4)]],
    device float* part_ml                        [[buffer(5)]],
    constant BatchSdpaPrm& p                     [[buffer(6)]],
    uint3 tgid                                   [[threadgroup_position_in_grid]],
    uint tid                                     [[thread_index_in_threadgroup]],
    uint lane                                    [[thread_index_in_simdgroup]],
    uint sg                                      [[simdgroup_index_in_threadgroup]])
{
    constexpr int E = MTP_HD / 32;
    threadgroup bfloat K_tg[MTP_B * MTP_HD];
    threadgroup bfloat V_tg[MTP_B * MTP_HD];

    const int kv_h = int(tgid.x);
    const int sp   = int(tgid.y);
    const int b    = int(tgid.z);
    const BatchMTPDraftKVMetaGpu dm = dkmeta[b * p.num_layers + p.layer_idx];
    const int total_k = dm.valid_ctx + MTP_B;
    int chunk = (total_k + p.n_splits - 1) / p.n_splits;
    chunk = max(8, ((chunk + 7) / 8) * 8);
    const int k_start = sp * chunk;
    const int k_end = min(total_k, k_start + chunk);

    const int qt = int(sg);
    const int qh0 = kv_h * MTP_G;
    const int col = int(lane) * E;

    float o[MTP_G][4];
    float m[MTP_G], l[MTP_G];
    #pragma clang loop unroll(full)
    for (int g = 0; g < MTP_G; ++g) {
        #pragma clang loop unroll(full)
        for (int e = 0; e < E; ++e) o[g][e] = 0.0f;
        m[g] = -INFINITY;
        l[g] = 0.0f;
    }

    if (k_start < k_end) {
        SdpaPrm sprm{
            dm.valid_ctx, total_k, dm.k_stride_h, dm.k_stride_s,
            dm.v_stride_h, dm.v_stride_s, p.n_splits, chunk, p.scale2
        };
        const device bfloat* qb  = q_buf  + size_t(b) * size_t(MTP_H * MTP_B * MTP_HD);
        const device bfloat* bkb = bk_buf + size_t(b) * size_t(MTP_KVH * MTP_B * MTP_HD);
        const device bfloat* bvb = bv_buf + size_t(b) * size_t(MTP_KVH * MTP_B * MTP_HD);
        sdpa_core(qb, (const device bfloat*)(dm.k_addr), (const device bfloat*)(dm.v_addr),
                  bkb, bvb, sprm, kv_h, k_start, k_end, tid, lane, sg, o, m, l, K_tg, V_tg);
    }

    #pragma clang loop unroll(full)
    for (int g = 0; g < MTP_G; ++g) {
        const size_t base_g = ((size_t(b) * size_t(p.n_splits) + size_t(sp)) * MTP_H + size_t(qh0 + g)) * MTP_B + size_t(qt);
        device float* po = part_o + base_g * MTP_HD + col;
        #pragma clang loop unroll(full)
        for (int e = 0; e < E; ++e) po[e] = o[g][e];
        if (lane == 0) {
            part_ml[base_g * 2 + 0] = m[g];
            part_ml[base_g * 2 + 1] = l[g];
        }
    }
}

[[kernel, max_total_threads_per_threadgroup(32)]]
kernel void mtp_batch_sdpa_split_p2(
    const device float* part_o  [[buffer(0)]],
    const device float* part_ml [[buffer(1)]],
    device bfloat* attn_out     [[buffer(2)]],
    constant BatchSdpaPrm& p    [[buffer(3)]],
    uint3 tgid                  [[threadgroup_position_in_grid]],
    uint lane                   [[thread_index_in_simdgroup]])
{
    constexpr int E = MTP_HD / 32;
    const int qt = int(tgid.x);
    const int qh = int(tgid.y);
    const int b  = int(tgid.z);
    const int col = int(lane) * E;

    float m_max = -INFINITY;
    for (int sp = 0; sp < p.n_splits; ++sp) {
        const size_t idx = ((size_t(b) * size_t(p.n_splits) + size_t(sp)) * MTP_H + size_t(qh)) * MTP_B + size_t(qt);
        const float ls = part_ml[idx * 2 + 1];
        if (ls > 0.0f) m_max = max(m_max, part_ml[idx * 2 + 0]);
    }

    float o_tot[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float l_tot = 0.0f;
    for (int sp = 0; sp < p.n_splits; ++sp) {
        const size_t idx = ((size_t(b) * size_t(p.n_splits) + size_t(sp)) * MTP_H + size_t(qh)) * MTP_B + size_t(qt);
        const float ms = part_ml[idx * 2 + 0];
        const float ls = part_ml[idx * 2 + 1];
        if (ls > 0.0f) {
            const float w = metal::precise::exp2(ms - m_max);
            l_tot += w * ls;
            const device float* po = part_o + idx * MTP_HD + col;
            #pragma clang loop unroll(full)
            for (int e = 0; e < E; ++e) o_tot[e] += w * po[e];
        }
    }

    device bfloat* dst = attn_out + size_t(b * MTP_B + qt) * size_t(MTP_H * MTP_HD) + size_t(qh) * MTP_HD + col;
    const float inv_l = 1.0f / l_tot;
    #pragma clang loop unroll(full)
    for (int e = 0; e < E; ++e) dst[e] = bfloat(o_tot[e] * inv_l);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mtp_batch_final_norm(
    const device bfloat* x_in        [[buffer(0)]],
    const device bfloat* res_in      [[buffer(1)]],
    const device bfloat* norm_w      [[buffer(2)]],
    const device int* d_to_stream    [[buffer(3)]],
    const device int* d_local_step   [[buffer(4)]],
    device bfloat* norm_out          [[buffer(5)]],
    constant AddNormPrm& p           [[buffer(6)]],
    uint tgid                        [[threadgroup_position_in_grid]],
    uint tid                         [[thread_position_in_threadgroup]],
    uint lane                        [[thread_index_in_simdgroup]],
    uint sg                          [[simdgroup_index_in_threadgroup]])
{
    constexpr int DPT = (MTP_D + 255) / 256;
    const int u = int(tgid);
    const int b = d_to_stream[u];
    const int s = d_local_step[u];
    const int sr = b * MTP_B + 1 + s;
    const device bfloat* xr = x_in + size_t(sr) * size_t(MTP_D);
    const device bfloat* rr = res_in + size_t(sr) * size_t(MTP_D);
    device bfloat* no = norm_out + size_t(u) * size_t(MTP_D);

    float vals[DPT];
    float ss = 0.0f;
    int cnt = 0;
    for (int d = int(tid); d < MTP_D; d += 256) {
        const float v = float(xr[d]) + float(rr[d]);
        vals[cnt++] = v;
        ss += v * v;
    }

    ss = simd_sum(ss);
    threadgroup float sg_ss[8];
    if (lane == 0) sg_ss[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    if (sg == 0 && lane < 8) tot = sg_ss[lane];
    tot = simd_sum(tot);
    if (tid == 0) sg_ss[0] = metal::precise::rsqrt(tot / float(MTP_D) + p.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = sg_ss[0];

    int idx = 0;
    for (int d = int(tid); d < MTP_D; d += 256) {
        no[d] = bfloat(vals[idx++] * inv_rms * float(norm_w[d]));
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mtp_batch_f_rnn(
    const device bfloat* hidden              [[buffer(0)]],
    const device float* fuse_alpha           [[buffer(1)]],
    const device bfloat* norm_w              [[buffer(2)]],
    const device BatchMTPStreamMetaGpu* sm   [[buffer(3)]],
    device bfloat* out                       [[buffer(4)]],
    constant MixPrm& p                       [[buffer(5)]],
    uint tgid                                [[threadgroup_position_in_grid]],
    uint tid                                 [[thread_position_in_threadgroup]],
    uint lane                                [[thread_index_in_simdgroup]],
    uint sg                                  [[simdgroup_index_in_threadgroup]])
{
    constexpr int DPT = (MTP_D + 255) / 256;
    const int b = int(tgid);
    const int src_row = sm[b].ctx_start + sm[b].n_ctx - 1;
    const device bfloat* h_base = hidden + size_t(src_row) * size_t(MTP_NCAP) * size_t(MTP_D);
    const device float* a_row = fuse_alpha + size_t(p.alpha_row) * size_t(MTP_NCAP);

    float alpha[MTP_NCAP];
    #pragma clang loop unroll(full)
    for (int tl = 0; tl < MTP_NCAP; ++tl) alpha[tl] = float(bfloat(a_row[tl]));

    float vals[DPT];
    float ss = 0.0f;
    int cnt = 0;
    for (int d = int(tid); d < MTP_D; d += 256) {
        float v = 0.0f;
        #pragma clang loop unroll(full)
        for (int tl = 0; tl < MTP_NCAP; ++tl) v += float(h_base[tl * MTP_D + d]) * alpha[tl];
        vals[cnt++] = v;
        ss += v * v;
    }

    ss = simd_sum(ss);
    threadgroup float sg_ss[8];
    if (lane == 0) sg_ss[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    if (sg == 0 && lane < 8) tot = sg_ss[lane];
    tot = simd_sum(tot);
    if (tid == 0) sg_ss[0] = metal::precise::rsqrt(tot / float(MTP_D) + p.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = sg_ss[0];

    device bfloat* out_row = out + size_t(b) * size_t(MTP_D);
    int idx = 0;
    for (int d = int(tid); d < MTP_D; d += 256) {
        out_row[d] = bfloat(vals[idx++] * inv_rms * float(norm_w[d]));
    }
}

struct BatchMarkovCtxPrm {
    int B, m_d_tot, D, out_dim;
};

[[kernel, max_total_threads_per_threadgroup(32)]]
kernel void mtp_batch_markov_ctx_p1(
    const device bfloat* x_out      [[buffer(0)]],
    const device bfloat* f_rnn      [[buffer(1)]],
    const device float* w_ctx_t     [[buffer(2)]],
    device float* part_x            [[buffer(3)]],
    device float* part_f            [[buffer(4)]],
    constant BatchMarkovCtxPrm& p   [[buffer(5)]],
    uint2 tgid                      [[threadgroup_position_in_grid]],
    uint lane                       [[thread_index_in_simdgroup]])
{
    const int col = int(tgid.x) * 32 + int(lane);
    const int sp = int(tgid.y);
    if (col >= p.out_dim) return;

    const int chunk = p.D / 8;
    if (sp < 8) {
        const int k0 = sp * chunk;
        const int k1 = k0 + chunk;
        float acc[28];
        for (int u = 0; u < p.m_d_tot; ++u) acc[u] = 0.0f;

        for (int k = k0; k < k1; k += 4) {
            const float w0 = w_ctx_t[size_t(k + 0) * size_t(p.out_dim) + size_t(col)];
            const float w1 = w_ctx_t[size_t(k + 1) * size_t(p.out_dim) + size_t(col)];
            const float w2 = w_ctx_t[size_t(k + 2) * size_t(p.out_dim) + size_t(col)];
            const float w3 = w_ctx_t[size_t(k + 3) * size_t(p.out_dim) + size_t(col)];
            for (int u = 0; u < p.m_d_tot; ++u) {
                const vec<bfloat, 4> xv = *(const device vec<bfloat, 4>*)(x_out + size_t(u) * size_t(p.D) + size_t(k));
                acc[u] += float(xv[0]) * w0 + float(xv[1]) * w1 + float(xv[2]) * w2 + float(xv[3]) * w3;
            }
        }
        for (int u = 0; u < p.m_d_tot; ++u) {
            part_x[(size_t(sp) * size_t(p.m_d_tot) + size_t(u)) * size_t(p.out_dim) + size_t(col)] = acc[u];
        }
    } else {
        const int fsp = sp - 8;
        const int k0 = fsp * chunk;
        const int k1 = k0 + chunk;
        const device float* wf_base = w_ctx_t + size_t(p.D) * size_t(p.out_dim);
        float acc_f[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int k = k0; k < k1; k += 4) {
            const float w0 = wf_base[size_t(k + 0) * size_t(p.out_dim) + size_t(col)];
            const float w1 = wf_base[size_t(k + 1) * size_t(p.out_dim) + size_t(col)];
            const float w2 = wf_base[size_t(k + 2) * size_t(p.out_dim) + size_t(col)];
            const float w3 = wf_base[size_t(k + 3) * size_t(p.out_dim) + size_t(col)];
            for (int b = 0; b < p.B; ++b) {
                const vec<bfloat, 4> fv = *(const device vec<bfloat, 4>*)(f_rnn + size_t(b) * size_t(p.D) + size_t(k));
                acc_f[b] += float(fv[0]) * w0 + float(fv[1]) * w1 + float(fv[2]) * w2 + float(fv[3]) * w3;
            }
        }
        for (int b = 0; b < p.B; ++b) {
            part_f[(size_t(fsp) * size_t(p.B) + size_t(b)) * size_t(p.out_dim) + size_t(col)] = acc_f[b];
        }
    }
}

[[kernel, max_total_threads_per_threadgroup(32)]]
kernel void mtp_batch_markov_ctx_p2(
    const device float* part_x      [[buffer(0)]],
    const device float* part_f      [[buffer(1)]],
    const device float* rnn_b       [[buffer(2)]],
    const device int* d_to_stream   [[buffer(3)]],
    device float* ctx_out           [[buffer(4)]],
    constant BatchMarkovCtxPrm& p   [[buffer(5)]],
    uint2 tgid                      [[threadgroup_position_in_grid]],
    uint lane                       [[thread_index_in_simdgroup]])
{
    const int col = int(tgid.x) * 32 + int(lane);
    const int u   = int(tgid.y);
    if (col >= p.out_dim || u >= p.m_d_tot) return;
    const int b = d_to_stream[u];

    float tot = rnn_b[col];
    #pragma clang loop unroll(full)
    for (int sp = 0; sp < 8; ++sp) {
        tot += part_f[(size_t(sp) * size_t(p.B) + size_t(b)) * size_t(p.out_dim) + size_t(col)];
        tot += part_x[(size_t(sp) * size_t(p.m_d_tot) + size_t(u)) * size_t(p.out_dim) + size_t(col)];
    }
    ctx_out[size_t(u) * size_t(p.out_dim) + size_t(col)] = tot;
}

struct BatchRnnStepPrm {
    int B, step_idx, rank, _pad;
};

[[kernel, max_total_threads_per_threadgroup(32)]]
kernel void mtp_batch_rnn_step(
    const device bfloat* state_in            [[buffer(0)]],
    const device int* prev_tok               [[buffer(1)]],
    const device bfloat* markov_w1           [[buffer(2)]],
    const device float* w_step_t             [[buffer(3)]],
    const device float* rnn_ctx              [[buffer(4)]],
    const device BatchMTPStreamMetaGpu* sm   [[buffer(5)]],
    device bfloat* state_out                 [[buffer(6)]],
    device bfloat* latent_out                [[buffer(7)]],
    constant BatchRnnStepPrm& p              [[buffer(8)]],
    uint2 tgid                               [[threadgroup_position_in_grid]],
    uint lane                                [[thread_index_in_simdgroup]])
{
    const int b = int(tgid.y);
    if (b >= p.B) return;
    const BatchMTPStreamMetaGpu meta = sm[b];
    if (p.step_idx >= meta.num_draft) return;

    const int rank = MTP_RANK;
    const int out_dim = 3 * rank;
    const int j = int(tgid.x) * 32 + int(lane);

    threadgroup float tg_in[2 * MTP_RANK];
    const int tok = prev_tok[b];
    const device bfloat* st_in_b = state_in + size_t(b) * size_t(rank);
    const device bfloat* emb = markov_w1 + size_t(tok) * size_t(rank);

    for (int i = int(lane); i < rank; i += 32) {
        tg_in[i] = float(st_in_b[i]);
        tg_in[rank + i] = float(emb[i]);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
    for (int k = 0; k < 2 * rank; k += 4) {
        const float x0 = tg_in[k + 0], x1 = tg_in[k + 1], x2 = tg_in[k + 2], x3 = tg_in[k + 3];
        const device float* wk0 = w_step_t + size_t(k + 0) * size_t(out_dim);
        const device float* wk1 = wk0 + out_dim;
        const device float* wk2 = wk1 + out_dim;
        const device float* wk3 = wk2 + out_dim;
        acc0 += x0 * wk0[j] + x1 * wk1[j] + x2 * wk2[j] + x3 * wk3[j];
        acc1 += x0 * wk0[rank + j] + x1 * wk1[rank + j] + x2 * wk2[rank + j] + x3 * wk3[rank + j];
        acc2 += x0 * wk0[2 * rank + j] + x1 * wk1[2 * rank + j] + x2 * wk2[2 * rank + j] + x3 * wk3[2 * rank + j];
    }

    const int u = meta.d_start + p.step_idx;
    const device float* ctx_i = rnn_ctx + size_t(u) * size_t(out_dim);
    const float r0 = float(bfloat(acc0 + ctx_i[j]));
    const float r1 = float(bfloat(acc1 + ctx_i[rank + j]));
    const float r2 = float(bfloat(acc2 + ctx_i[2 * rank + j]));

    const float gate = float(bfloat(1.0f / (1.0f + metal::precise::exp(-r0))));
    const float cand = float(bfloat(metal::precise::tanh(r1)));
    const float one_minus_g = float(bfloat(1.0f - gate));
    const float old_s = tg_in[j];
    const bfloat new_s = bfloat(float(bfloat(gate * old_s)) + float(bfloat(one_minus_g * cand)));
    const bfloat lat = bfloat(metal::precise::tanh(r2));

    state_out[size_t(b) * size_t(rank) + j] = new_s;
    latent_out[size_t(b) * size_t(rank) + j] = lat;
}

struct BatchMarkovPrm {
    int B, v_limit, step_idx, n_part, max_k, _pad;
};

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mtp_batch_markov_mixed_p1(
    const device bfloat* latent              [[buffer(0)]],
    const device bfloat* fold_w              [[buffer(1)]],
    const device bfloat* fold_b              [[buffer(2)]],
    const device bfloat* base_l              [[buffer(3)]],
    const device BatchMTPStreamMetaGpu* sm   [[buffer(4)]],
    device float* part_val                   [[buffer(5)]],
    device uint* part_idx                    [[buffer(6)]],
    device float* q_logits_out               [[buffer(7)]],
    device float* part_max                   [[buffer(8)]],
    constant BatchMarkovPrm& p               [[buffer(9)]],
    uint tgid                                [[threadgroup_position_in_grid]],
    uint tid                                 [[thread_position_in_threadgroup]],
    uint lane                                [[thread_index_in_simdgroup]],
    uint sg                                  [[simdgroup_index_in_threadgroup]])
{
    constexpr int SG = 8;
    constexpr int ROWS_PER_SG = 8;
    constexpr int KPL = MTP_RANK / 32;
    const int v_base = int(tgid) * (SG * ROWS_PER_SG);

    threadgroup float lat[4][MTP_RANK];
    for (int b = 0; b < p.B; ++b) {
        if (p.step_idx < sm[b].num_draft) {
            const device bfloat* lb = latent + size_t(b) * size_t(MTP_RANK);
            for (int i = int(tid); i < MTP_RANK; i += 256) {
                lat[b][i] = float(lb[i]);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float best_val[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    uint  best_idx[4] = {0u, 0u, 0u, 0u};

    const int k = int(lane) * KPL;
    #pragma clang loop unroll(full)
    for (int ri = 0; ri < ROWS_PER_SG; ++ri) {
        const int v = v_base + int(sg) * ROWS_PER_SG + ri;
        if (v < p.v_limit) {
            const device bfloat* wp_b = fold_w + size_t(v) * size_t(MTP_RANK) + k;
            float dot_v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #if MTP_RANK == 256
            const device vec<bfloat, 4>* wp = (const device vec<bfloat, 4>*)(wp_b);
            const vec<bfloat, 4> wa = wp[0];
            const vec<bfloat, 4> wb = wp[1];
            for (int b = 0; b < p.B; ++b) {
                if (p.step_idx < sm[b].num_draft) {
                    dot_v[b] = float(wa[0]) * lat[b][k + 0] + float(wa[1]) * lat[b][k + 1]
                             + float(wa[2]) * lat[b][k + 2] + float(wa[3]) * lat[b][k + 3]
                             + float(wb[0]) * lat[b][k + 4] + float(wb[1]) * lat[b][k + 5]
                             + float(wb[2]) * lat[b][k + 6] + float(wb[3]) * lat[b][k + 7];
                }
            }
        #elif (MTP_RANK % 128) == 0
            const device vec<bfloat, 4>* wp = (const device vec<bfloat, 4>*)(wp_b);
            for (int i = 0; i < KPL / 4; ++i) {
                const vec<bfloat, 4> wv = wp[i];
                const int off = k + i * 4;
                for (int b = 0; b < p.B; ++b) {
                    if (p.step_idx < sm[b].num_draft) {
                        dot_v[b] += float(wv[0]) * lat[b][off + 0] + float(wv[1]) * lat[b][off + 1]
                                  + float(wv[2]) * lat[b][off + 2] + float(wv[3]) * lat[b][off + 3];
                    }
                }
            }
        #else
            #pragma clang loop unroll(full)
            for (int i = 0; i < KPL; ++i) {
                const float wfv = float(wp_b[i]);
                for (int b = 0; b < p.B; ++b) {
                    if (p.step_idx < sm[b].num_draft) {
                        dot_v[b] += wfv * lat[b][k + i];
                    }
                }
            }
        #endif
            const float fbv = float(fold_b[v]);
            for (int b = 0; b < p.B; ++b) {
                if (p.step_idx < sm[b].num_draft) {
                    const float tot = simd_sum(dot_v[b]);
                    if (lane == 0) {
                        const BatchMTPStreamMetaGpu meta = sm[b];
                        const int u = meta.d_start + p.step_idx;
                        const float base_v = float(base_l[size_t(u) * size_t(p.v_limit) + size_t(v)]);
                        const float raw = float(bfloat(base_v + float(bfloat(tot + fbv))));
                        if (meta.is_sampled != 0) {
                            float val = (meta.softcap > 0.0f) ? (meta.softcap * metal::precise::tanh(raw / meta.softcap)) : raw;
                            val /= meta.temperature;
                            q_logits_out[size_t(u) * size_t(p.v_limit) + size_t(v)] = val;
                            if (val > best_val[b]) best_val[b] = val;
                        } else {
                            if (raw > best_val[b]) {
                                best_val[b] = raw;
                                best_idx[b] = uint(v);
                            }
                        }
                    }
                }
            }
        }
    }

    threadgroup float sg_val[4][SG];
    threadgroup uint  sg_idx[4][SG];
    if (lane == 0) {
        for (int b = 0; b < p.B; ++b) {
            sg_val[b][sg] = best_val[b];
            sg_idx[b][sg] = best_idx[b];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && int(lane) < p.B) {
        const int b = int(lane);
        const BatchMTPStreamMetaGpu meta = sm[b];
        if (p.step_idx < meta.num_draft) {
            float bv = sg_val[b][0];
            uint  bi = sg_idx[b][0];
            if (meta.is_sampled != 0) {
                for (int s = 1; s < SG; ++s) bv = max(bv, sg_val[b][s]);
                part_max[size_t(b) * size_t(p.n_part) + tgid] = bv;
            } else {
                for (int s = 1; s < SG; ++s) {
                    if (sg_val[b][s] > bv) { bv = sg_val[b][s]; bi = sg_idx[b][s]; }
                }
                part_val[size_t(b) * size_t(p.n_part) + tgid] = bv;
                part_idx[size_t(b) * size_t(p.n_part) + tgid] = bi;
            }
        }
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mtp_batch_markov_argmax_p2(
    const device float* part_val             [[buffer(0)]],
    const device uint* part_idx              [[buffer(1)]],
    const device BatchMTPStreamMetaGpu* sm   [[buffer(2)]],
    device int* d_toks                       [[buffer(3)]],
    device int* prev_tok                     [[buffer(4)]],
    device int* verify_toks                  [[buffer(5)]],
    constant BatchMarkovPrm& p               [[buffer(6)]],
    uint2 tgid                               [[threadgroup_position_in_grid]],
    uint2 tid2                               [[thread_position_in_threadgroup]],
    uint lane                                [[thread_index_in_simdgroup]],
    uint sg                                  [[simdgroup_index_in_threadgroup]])
{
    const int b = int(tgid.y);
    if (b >= p.B) return;
    const BatchMTPStreamMetaGpu meta = sm[b];
    if (p.step_idx >= meta.num_draft || meta.is_sampled != 0) return;

    const uint tid = tid2.x;
    const size_t base = size_t(b) * size_t(p.n_part);
    float bv = -INFINITY;
    uint bi = 0u;
    for (int i = int(tid); i < p.n_part; i += 256) {
        const float v = part_val[base + size_t(i)];
        const uint idx = part_idx[base + size_t(i)];
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
        float fv = sg_v[0];
        uint fi = sg_i[0];
        for (int s = 1; s < 8; ++s) {
            if (sg_v[s] > fv || (sg_v[s] == fv && sg_i[s] < fi)) { fv = sg_v[s]; fi = sg_i[s]; }
        }
        d_toks[b * p.max_k + p.step_idx] = int(fi);
        prev_tok[b] = int(fi);
        verify_toks[meta.v_start + 1 + p.step_idx] = int(fi);
    }
}

struct BatchSoftmaxPrm {
    int v_limit, n_part, n_tg, step_idx, pass_idx, mode, max_k, only_topp;
};

inline int resolve_batch_softmax_stream(
    int y,
    int mode,
    int step_idx,
    const device int* tok_to_stream,
    const device BatchMTPStreamMetaGpu* sm,
    thread int& out_prob_row)
{
    if (mode == 1) {
        const BatchMTPStreamMetaGpu meta = sm[y];
        if (step_idx >= meta.num_draft || meta.is_sampled == 0) return -1;
        out_prob_row = meta.d_start + step_idx;
        return y;
    } else {
        const int b = tok_to_stream[y];
        if (sm[b].is_sampled == 0) return -1;
        out_prob_row = y;
        return b;
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mtp_batch_softmax_max_p2(
    const device float* part_max             [[buffer(0)]],
    device float* row_max                    [[buffer(1)]],
    const device int* tok_to_stream          [[buffer(2)]],
    const device BatchMTPStreamMetaGpu* sm   [[buffer(3)]],
    constant BatchSoftmaxPrm& p              [[buffer(4)]],
    uint2 tgid                               [[threadgroup_position_in_grid]],
    uint2 tid2                               [[thread_position_in_threadgroup]],
    uint lane                                [[thread_index_in_simdgroup]],
    uint sg                                  [[simdgroup_index_in_threadgroup]])
{
    int prob_row = 0;
    const int b = resolve_batch_softmax_stream(int(tgid.y), p.mode, p.step_idx, tok_to_stream, sm, prob_row);
    if (b < 0) return;

    const uint tid = tid2.x;
    const size_t base = size_t(tgid.y) * size_t(p.n_part);
    float bv = -INFINITY;
    for (int i = int(tid); i < p.n_part; i += 256) {
        bv = max(bv, part_max[base + size_t(i)]);
    }
    bv = simd_max(bv);
    threadgroup float sg_v[8];
    if (lane == 0) sg_v[sg] = bv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float tot = simd_max(lane < 8u ? sg_v[lane] : -INFINITY);
        if (lane == 0) row_max[int(tgid.y)] = tot;
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mtp_batch_softmax_exp_sum_p1(
    device float* probs                      [[buffer(0)]],
    const device float* row_max              [[buffer(1)]],
    device float* part_sum                   [[buffer(2)]],
    const device int* tok_to_stream          [[buffer(3)]],
    const device BatchMTPStreamMetaGpu* sm   [[buffer(4)]],
    constant BatchSoftmaxPrm& p              [[buffer(5)]],
    uint2 tgid                               [[threadgroup_position_in_grid]],
    uint2 tid2                               [[thread_position_in_threadgroup]],
    uint lane                                [[thread_index_in_simdgroup]],
    uint sg                                  [[simdgroup_index_in_threadgroup]])
{
    int prob_row = 0;
    const int b = resolve_batch_softmax_stream(int(tgid.y), p.mode, p.step_idx, tok_to_stream, sm, prob_row);
    if (b < 0) return;

    const uint tid = tid2.x;
    const int tg = int(tgid.x);
    const int chunk = (p.v_limit + p.n_tg - 1) / p.n_tg;
    const int v_beg = tg * chunk;
    const int v_end = min(p.v_limit, v_beg + chunk);
    const float mx = row_max[int(tgid.y)];
    device float* pr = probs + size_t(prob_row) * size_t(p.v_limit);

    float s = 0.0f;
    for (int base = v_beg; base < v_end; base += 256) {
        const int v = base + int(tid);
        if (v < v_end) {
            const float e = metal::precise::exp(pr[v] - mx);
            pr[v] = e;
            s += e;
        }
    }
    s = simd_sum(s);
    threadgroup float sg_s[8];
    if (lane == 0) sg_s[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float tot = simd_sum(lane < 8u ? sg_s[lane] : 0.0f);
        if (lane == 0) part_sum[size_t(tgid.y) * size_t(p.n_tg) + size_t(tg)] = tot;
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mtp_batch_softmax_sum_p2(
    const device float* part_sum             [[buffer(0)]],
    device float* row_inv_sum                [[buffer(1)]],
    const device int* tok_to_stream          [[buffer(2)]],
    const device BatchMTPStreamMetaGpu* sm   [[buffer(3)]],
    constant BatchSoftmaxPrm& p              [[buffer(4)]],
    uint2 tgid                               [[threadgroup_position_in_grid]],
    uint2 tid2                               [[thread_position_in_threadgroup]],
    uint lane                                [[thread_index_in_simdgroup]],
    uint sg                                  [[simdgroup_index_in_threadgroup]])
{
    int prob_row = 0;
    const int b = resolve_batch_softmax_stream(int(tgid.y), p.mode, p.step_idx, tok_to_stream, sm, prob_row);
    if (b < 0) return;
    if (p.only_topp != 0 && sm[b].do_top_p == 0) return;

    const uint tid = tid2.x;
    const size_t base = size_t(tgid.y) * size_t(p.n_part);
    float s = 0.0f;
    for (int i = int(tid); i < p.n_part; i += 256) {
        s += part_sum[base + size_t(i)];
    }
    s = simd_sum(s);
    threadgroup float sg_s[8];
    if (lane == 0) sg_s[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float tot = simd_sum(lane < 8u ? sg_s[lane] : 0.0f);
        if (lane == 0) row_inv_sum[int(tgid.y)] = (tot > 0.0f) ? (1.0f / tot) : 0.0f;
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mtp_batch_topp_hist(
    device float* probs                      [[buffer(0)]],
    const device float* row_inv_sum          [[buffer(1)]],
    const device TopPState* states           [[buffer(2)]],
    device float* part_sum                   [[buffer(3)]],
    device uint* part_cnt                    [[buffer(4)]],
    const device int* tok_to_stream          [[buffer(5)]],
    const device BatchMTPStreamMetaGpu* sm   [[buffer(6)]],
    constant BatchSoftmaxPrm& p              [[buffer(7)]],
    uint2 tgid                               [[threadgroup_position_in_grid]],
    uint2 tid2                               [[thread_position_in_threadgroup]],
    uint lane                                [[thread_index_in_simdgroup]],
    uint sg                                  [[simdgroup_index_in_threadgroup]])
{
    int prob_row = 0;
    const int bs = resolve_batch_softmax_stream(int(tgid.y), p.mode, p.step_idx, tok_to_stream, sm, prob_row);
    if (bs < 0 || sm[bs].do_top_p == 0) return;

    const uint tid = tid2.x;
    const int m = int(tgid.y);
    const int tg = int(tgid.x);
    const TopPState st = states[m];
    if (p.pass_idx > 0 && st.found_all != 0u) return;

    const float inv = (p.pass_idx == 0) ? row_inv_sum[m] : 1.0f;
    const int shift = 25 - p.pass_idx * 5;
    const uint mask_shift = uint(shift + 5);
    const uint prefix_shifted = st.prefix_bits >> mask_shift;

    threadgroup float sg_sum[8][32];
    threadgroup uint  sg_cnt[8][32];
    sg_sum[tid >> 5][tid & 31u] = 0.0f;
    sg_cnt[tid >> 5][tid & 31u] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int chunk = (p.v_limit + p.n_tg - 1) / p.n_tg;
    const int v_beg = tg * chunk;
    const int v_end = min(p.v_limit, v_beg + chunk);
    device float* pr = probs + size_t(prob_row) * size_t(p.v_limit);

    for (int base = v_beg; base < v_end; base += 256) {
        const int v = base + int(tid);
        const bool in_range = (v < v_end);
        float pv = 0.0f;
        bool match = false;
        uint bin = 0u;
        if (in_range) {
            pv = pr[v];
            if (p.pass_idx == 0) {
                pv *= inv;
                pr[v] = pv;
            }
            const uint bits = (pv <= 0.0f) ? 0u : as_type<uint>(pv);
            match = (p.pass_idx == 0) || ((bits >> mask_shift) == prefix_shifted);
            bin = (p.pass_idx == 0) ? min(31u, bits >> 25) : ((bits >> uint(shift)) & 31u);
        }
        if (simd_any(match)) {
            for (int b = 0; b < 32; ++b) {
                const bool hit = match && (bin == uint(b));
                if (simd_any(hit)) {
                    const float bs_sum = simd_sum(hit ? pv : 0.0f);
                    const uint  bc_cnt = simd_sum(hit ? 1u : 0u);
                    if (lane == 0) {
                        sg_sum[sg][b] += bs_sum;
                        sg_cnt[sg][b] += bc_cnt;
                    }
                }
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float fs = 0.0f;
        uint fc = 0u;
        #pragma clang loop unroll(full)
        for (int s = 0; s < 8; ++s) {
            fs += sg_sum[s][lane];
            fc += sg_cnt[s][lane];
        }
        const size_t out_idx = (size_t(m) * size_t(p.n_tg) + size_t(tg)) * 32u + size_t(lane);
        part_sum[out_idx] = fs;
        part_cnt[out_idx] = fc;
    }
}

[[kernel, max_total_threads_per_threadgroup(32)]]
kernel void mtp_batch_topp_scan(
    const device float* part_sum             [[buffer(0)]],
    const device uint* part_cnt              [[buffer(1)]],
    device TopPState* states                 [[buffer(2)]],
    const device int* tok_to_stream          [[buffer(3)]],
    const device BatchMTPStreamMetaGpu* sm   [[buffer(4)]],
    constant BatchSoftmaxPrm& p              [[buffer(5)]],
    uint2 tgid                               [[threadgroup_position_in_grid]],
    uint lane                                [[thread_index_in_simdgroup]])
{
    int prob_row = 0;
    const int bs = resolve_batch_softmax_stream(int(tgid.y), p.mode, p.step_idx, tok_to_stream, sm, prob_row);
    if (bs < 0 || sm[bs].do_top_p == 0) return;

    const int m = int(tgid.y);
    if (p.pass_idx > 0 && states[m].found_all != 0u) return;
    const float top_p = sm[bs].top_p;

    float b_sum = 0.0f;
    uint b_cnt = 0u;
    const size_t base = size_t(m) * size_t(p.n_tg);
    for (int tg = 0; tg < p.n_tg; ++tg) {
        const size_t idx = (base + size_t(tg)) * 32u + size_t(lane);
        b_sum += part_sum[idx];
        b_cnt += part_cnt[idx];
    }

    const float base_gt = (p.pass_idx == 0) ? 0.0f : states[m].sum_gt;
    const uint rev = 31u - lane;
    const float rev_sum = simd_shuffle(b_sum, rev);
    const float rev_excl = simd_prefix_exclusive_sum(rev_sum);
    const float sum_above = base_gt + simd_shuffle(rev_excl, rev);
    const float sum_incl = sum_above + b_sum;

    if (p.pass_idx == 0) {
        const float total_sum = simd_shuffle(sum_incl, 0u);
        if (total_sum < top_p) {
            if (lane == 0) {
                states[m].found_all = 1u;
                states[m].thresh_bits = 0u;
                states[m].tie_keep_count = 0u;
            }
            return;
        }
    }

    const bool winner = (b_cnt > 0u) && (sum_above < top_p) && (sum_incl >= top_p);
    int win_lane = simd_max(winner ? int(lane) : -1);
    if (win_lane < 0) {
        if (base_gt >= top_p) {
            win_lane = simd_max((b_cnt > 0u) ? int(lane) : -1);
        } else {
            win_lane = simd_min((b_cnt > 0u) ? int(lane) : 32);
        }
        if (win_lane < 0 || win_lane >= 32) win_lane = 0;
    }

    const uint b_star = uint(win_lane);
    const int shift = 25 - p.pass_idx * 5;
    const uint prev_pref = (p.pass_idx == 0) ? 0u : states[m].prefix_bits;
    const uint new_prefix = prev_pref | (b_star << uint(shift));
    const float new_sum_gt = simd_shuffle(sum_above, b_star);
    const uint tot_ties = simd_shuffle(b_cnt, b_star);

    if (p.pass_idx == 5) {
        const uint c0 = (int(lane) < p.n_tg) ? part_cnt[(base + size_t(lane)) * 32u + size_t(b_star)] : 0u;
        const uint p0 = simd_prefix_exclusive_sum(c0);
        const uint t0 = simd_sum(c0);
        const uint c1 = (int(32u + lane) < p.n_tg) ? part_cnt[(base + size_t(32u + lane)) * 32u + size_t(b_star)] : 0u;
        const uint p1 = t0 + simd_prefix_exclusive_sum(c1);
        if (int(lane) < p.n_tg) states[m].tie_base[lane] = p0;
        if (int(32u + lane) < p.n_tg) states[m].tie_base[32u + lane] = p1;

        if (lane == 0) {
            states[m].prefix_bits = new_prefix;
            states[m].sum_gt = new_sum_gt;
            states[m].thresh_bits = new_prefix;
            states[m].found_all = 0u;
            const float t_val = as_type<float>(new_prefix);
            if (t_val <= 0.0f || new_sum_gt + float(tot_ties) * t_val < top_p) {
                states[m].tie_keep_count = tot_ties;
            } else {
                uint k = uint(max(0.0f, floor((top_p - new_sum_gt) / t_val)));
                k = min(k, tot_ties);
                while (k < tot_ties && (new_sum_gt + float(k) * t_val) < top_p) ++k;
                while (k > 0u && !((new_sum_gt + float(k - 1u) * t_val) < top_p)) --k;
                states[m].tie_keep_count = k;
            }
        }
    } else {
        if (lane == 0) {
            states[m].prefix_bits = new_prefix;
            states[m].sum_gt = new_sum_gt;
            states[m].found_all = 0u;
        }
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mtp_batch_topp_mask_sum(
    device float* probs                      [[buffer(0)]],
    const device TopPState* states           [[buffer(1)]],
    device float* part_sum                   [[buffer(2)]],
    const device int* tok_to_stream          [[buffer(3)]],
    const device BatchMTPStreamMetaGpu* sm   [[buffer(4)]],
    constant BatchSoftmaxPrm& p              [[buffer(5)]],
    uint2 tgid                               [[threadgroup_position_in_grid]],
    uint2 tid2                               [[thread_position_in_threadgroup]],
    uint lane                                [[thread_index_in_simdgroup]],
    uint sg                                  [[simdgroup_index_in_threadgroup]])
{
    int prob_row = 0;
    const int bs = resolve_batch_softmax_stream(int(tgid.y), p.mode, p.step_idx, tok_to_stream, sm, prob_row);
    if (bs < 0 || sm[bs].do_top_p == 0) return;

    const uint tid = tid2.x;
    const int m = int(tgid.y);
    const int tg = int(tgid.x);
    const TopPState st = states[m];
    const uint thresh_bits = st.thresh_bits;
    uint local_tie_off = st.tie_base[tg];

    const int chunk = (p.v_limit + p.n_tg - 1) / p.n_tg;
    const int v_beg = tg * chunk;
    const int v_end = min(p.v_limit, v_beg + chunk);
    device float* pr = probs + size_t(prob_row) * size_t(p.v_limit);

    float sum_kept = 0.0f;
    threadgroup uint sg_cnts[8];

    for (int base = v_beg; base < v_end; base += 256) {
        const int v = base + int(tid);
        const bool in_range = (v < v_end);
        float pv = in_range ? pr[v] : 0.0f;
        const uint bits = (pv <= 0.0f) ? 0u : as_type<uint>(pv);
        const bool is_tie = (st.found_all == 0u) && in_range && (bits == thresh_bits);

        const uint t_flag = is_tie ? 1u : 0u;
        const uint sg_inc = simd_prefix_inclusive_sum(t_flag);
        const uint sg_exc = sg_inc - t_flag;
        if (lane == 31) sg_cnts[sg] = sg_inc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint tg_base_rank = 0u;
        for (uint s = 0u; s < sg; ++s) tg_base_rank += sg_cnts[s];
        uint total_chunk = 0u;
        #pragma clang loop unroll(full)
        for (uint s = 0u; s < 8u; ++s) total_chunk += sg_cnts[s];

        const uint my_tie_rank = local_tie_off + tg_base_rank + sg_exc;
        local_tie_off += total_chunk;

        const bool keep = (st.found_all != 0u) || (bits > thresh_bits) || (is_tie && my_tie_rank < st.tie_keep_count);
        if (in_range && !keep) {
            pv = 0.0f;
            pr[v] = 0.0f;
        }
        sum_kept += pv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    sum_kept = simd_sum(sum_kept);
    threadgroup float sg_s[8];
    if (lane == 0) sg_s[sg] = sum_kept;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float tot = simd_sum(lane < 8u ? sg_s[lane] : 0.0f);
        if (lane == 0) part_sum[size_t(m) * size_t(p.n_tg) + size_t(tg)] = tot;
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mtp_batch_probs_scale(
    device float* probs                      [[buffer(0)]],
    const device float* row_inv_sum          [[buffer(1)]],
    const device int* tok_to_stream          [[buffer(2)]],
    const device BatchMTPStreamMetaGpu* sm   [[buffer(3)]],
    constant BatchSoftmaxPrm& p              [[buffer(4)]],
    uint2 tgid                               [[threadgroup_position_in_grid]],
    uint2 tid2                               [[thread_position_in_threadgroup]])
{
    const int m = int(tgid.y);
    if (sm[tok_to_stream[m]].is_sampled == 0) return;

    const uint tid = tid2.x;
    const int tg = int(tgid.x);
    const int chunk = (p.v_limit + p.n_tg - 1) / p.n_tg;
    const int v_beg = tg * chunk;
    const int v_end = min(p.v_limit, v_beg + chunk);
    const float inv = row_inv_sum[m];
    device float* pr = probs + size_t(m) * size_t(p.v_limit);
    for (int base = v_beg; base < v_end; base += 256) {
        const int v = base + int(tid);
        if (v < v_end) pr[v] *= inv;
    }
}

struct BatchGumbelPrm {
    int v_limit, n_tg, step_idx, max_k;
};

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mtp_batch_probs_scale_gumbel_p1(
    device float* probs                      [[buffer(0)]],
    const device float* row_inv_sum          [[buffer(1)]],
    const device BatchMTPStreamMetaGpu* sm   [[buffer(2)]],
    device float* part_val                   [[buffer(3)]],
    device uint* part_idx                    [[buffer(4)]],
    constant BatchGumbelPrm& p               [[buffer(5)]],
    uint2 tgid                               [[threadgroup_position_in_grid]],
    uint2 tid2                               [[thread_position_in_threadgroup]],
    uint lane                                [[thread_index_in_simdgroup]],
    uint sg                                  [[simdgroup_index_in_threadgroup]])
{
    const int b = int(tgid.y);
    const BatchMTPStreamMetaGpu meta = sm[b];
    if (p.step_idx >= meta.num_draft || meta.is_sampled == 0) return;

    const uint tid = tid2.x;
    const int tg = int(tgid.x);
    const int chunk = (p.v_limit + p.n_tg - 1) / p.n_tg;
    const int v_beg = tg * chunk;
    const int v_end = min(p.v_limit, v_beg + chunk);
    const float inv = row_inv_sum[b];
    const int prob_row = meta.d_start + p.step_idx;
    device float* pr = probs + size_t(prob_row) * size_t(p.v_limit);

    float bv = -INFINITY;
    uint bi = 0u;
    for (int base = v_beg; base < v_end; base += 256) {
        const int v = base + int(tid);
        if (v < v_end) {
            const float pv = pr[v] * inv;
            pr[v] = pv;
            if (pv > 0.0f) {
                const float u = philox_u01(meta.rng_seed, meta.rng_step, uint(p.step_idx), uint(v));
                const float sc = metal::precise::log(pv) - metal::precise::log(-metal::precise::log(u));
                if (sc > bv || (sc == bv && uint(v) < bi)) { bv = sc; bi = uint(v); }
            }
        }
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
        float fv = sg_v[0];
        uint fi = sg_i[0];
        for (int k = 1; k < 8; ++k) {
            if (sg_v[k] > fv || (sg_v[k] == fv && sg_i[k] < fi)) { fv = sg_v[k]; fi = sg_i[k]; }
        }
        part_val[size_t(b) * size_t(p.n_tg) + size_t(tg)] = fv;
        part_idx[size_t(b) * size_t(p.n_tg) + size_t(tg)] = fi;
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mtp_batch_gumbel_p2(
    const device float* part_val             [[buffer(0)]],
    const device uint* part_idx              [[buffer(1)]],
    const device BatchMTPStreamMetaGpu* sm   [[buffer(2)]],
    device int* d_toks                       [[buffer(3)]],
    device int* prev_tok                     [[buffer(4)]],
    device int* verify_toks                  [[buffer(5)]],
    constant BatchGumbelPrm& p               [[buffer(6)]],
    uint2 tgid                               [[threadgroup_position_in_grid]],
    uint2 tid2                               [[thread_position_in_threadgroup]],
    uint lane                                [[thread_index_in_simdgroup]],
    uint sg                                  [[simdgroup_index_in_threadgroup]])
{
    const int b = int(tgid.y);
    const BatchMTPStreamMetaGpu meta = sm[b];
    if (p.step_idx >= meta.num_draft || meta.is_sampled == 0) return;

    const uint tid = tid2.x;
    const size_t base = size_t(b) * size_t(p.n_tg);
    float bv = -INFINITY;
    uint bi = 0u;
    for (int i = int(tid); i < p.n_tg; i += 256) {
        const float v = part_val[base + size_t(i)];
        const uint idx = part_idx[base + size_t(i)];
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
        float fv = sg_v[0];
        uint fi = sg_i[0];
        for (int k = 1; k < 8; ++k) {
            if (sg_v[k] > fv || (sg_v[k] == fv && sg_i[k] < fi)) { fv = sg_v[k]; fi = sg_i[k]; }
        }
        d_toks[b * p.max_k + p.step_idx] = int(fi);
        prev_tok[b] = int(fi);
        verify_toks[meta.v_start + 1 + p.step_idx] = int(fi);
    }
}

struct BatchRejectPrm {
    int v_limit, n_tg, max_k, _pad;
};

[[kernel, max_total_threads_per_threadgroup(32)]]
kernel void mtp_batch_greedy_accept(
    const device int* d_toks                 [[buffer(0)]],
    const device int* t_toks                 [[buffer(1)]],
    const device BatchMTPStreamMetaGpu* sm   [[buffer(2)]],
    device int* out_toks                     [[buffer(3)]],
    device int* d_n_acc                      [[buffer(4)]],
    constant BatchRejectPrm& p               [[buffer(5)]],
    uint tgid                                [[threadgroup_position_in_grid]],
    uint tid                                 [[thread_position_in_threadgroup]])
{
    if (tid != 0u) return;
    const int b = int(tgid);
    const BatchMTPStreamMetaGpu meta = sm[b];
    if (meta.is_sampled != 0) return;

    const device int* dt = d_toks + b * p.max_k;
    const device int* tt = t_toks + meta.v_start;
    device int* ot = out_toks + b * MTP_B;

    int match = 0;
    while (match < meta.num_draft && dt[match] == tt[match]) {
        ot[match] = dt[match];
        ++match;
    }
    ot[match] = tt[match];
    d_n_acc[b] = match + 1;
}

[[kernel, max_total_threads_per_threadgroup(32)]]
kernel void mtp_batch_reject_prefix(
    const device int* d_toks                 [[buffer(0)]],
    const device float* p_probs              [[buffer(1)]],
    const device float* q_probs              [[buffer(2)]],
    const device BatchMTPStreamMetaGpu* sm   [[buffer(3)]],
    device int* out_toks                     [[buffer(4)]],
    device int* d_reject_j                   [[buffer(5)]],
    device int* d_n_acc                      [[buffer(6)]],
    constant BatchRejectPrm& p               [[buffer(7)]],
    uint tgid                                [[threadgroup_position_in_grid]],
    uint tid                                 [[thread_position_in_threadgroup]])
{
    if (tid != 0u) return;
    const int b = int(tgid);
    const BatchMTPStreamMetaGpu meta = sm[b];
    if (meta.is_sampled == 0) return;

    const device int* dt = d_toks + b * p.max_k;
    device int* ot = out_toks + b * MTP_B;
    int rej = meta.num_draft;
    for (int j = 0; j < meta.num_draft; ++j) {
        const int dj = dt[j];
        const float pd = p_probs[size_t(meta.v_start + j) * size_t(p.v_limit) + size_t(dj)];
        const float qd = q_probs[size_t(meta.d_start + j) * size_t(p.v_limit) + size_t(dj)];
        const float ratio = (qd <= 0.0f) ? 1.0f : min(1.0f, pd / qd);
        const float u = philox_u01(meta.rng_seed, meta.rng_step, 1000u + uint(j), 0u);
        if (u <= ratio) {
            ot[j] = dj;
        } else {
            rej = j;
            break;
        }
    }
    d_reject_j[b] = rej;
    d_n_acc[b] = rej + 1;
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mtp_batch_reject_sample_p1(
    const device float* p_probs              [[buffer(0)]],
    const device float* q_probs              [[buffer(1)]],
    const device int* d_reject_j             [[buffer(2)]],
    const device BatchMTPStreamMetaGpu* sm   [[buffer(3)]],
    device float* part_res_val               [[buffer(4)]],
    device uint* part_res_idx                [[buffer(5)]],
    device float* part_p_val                 [[buffer(6)]],
    device uint* part_p_idx                  [[buffer(7)]],
    device float* part_res_sum               [[buffer(8)]],
    constant BatchRejectPrm& p               [[buffer(9)]],
    uint2 tgid                               [[threadgroup_position_in_grid]],
    uint2 tid2                               [[thread_position_in_threadgroup]],
    uint lane                                [[thread_index_in_simdgroup]],
    uint sg                                  [[simdgroup_index_in_threadgroup]])
{
    const int b = int(tgid.y);
    const BatchMTPStreamMetaGpu meta = sm[b];
    if (meta.is_sampled == 0) return;

    const uint tid = tid2.x;
    const int rej = d_reject_j[b];
    const int tg = int(tgid.x);
    const int chunk = (p.v_limit + p.n_tg - 1) / p.n_tg;
    const int v_beg = tg * chunk;
    const int v_end = min(p.v_limit, v_beg + chunk);

    const device float* pr = p_probs + size_t(meta.v_start + rej) * size_t(p.v_limit);
    const device float* qr = (rej < meta.num_draft) ? (q_probs + size_t(meta.d_start + rej) * size_t(p.v_limit)) : nullptr;

    float brv = -INFINITY, bpv = -INFINITY;
    uint bri = 0u, bpi = 0u;
    float s_res = 0.0f;

    for (int base = v_beg; base < v_end; base += 256) {
        const int v = base + int(tid);
        if (v < v_end) {
            const float pv = pr[v];
            const float rv = (rej < meta.num_draft) ? max(0.0f, pv - qr[v]) : 0.0f;
            s_res += rv;
            if (pv > 0.0f || rv > 0.0f) {
                const float u = philox_u01(meta.rng_seed, meta.rng_step, 2000u, uint(v));
                const float g = -metal::precise::log(-metal::precise::log(u));
                if (pv > 0.0f) {
                    const float sp = metal::precise::log(pv) + g;
                    if (sp > bpv || (sp == bpv && uint(v) < bpi)) { bpv = sp; bpi = uint(v); }
                }
                if (rv > 0.0f) {
                    const float sr = metal::precise::log(rv) + g;
                    if (sr > brv || (sr == brv && uint(v) < bri)) { brv = sr; bri = uint(v); }
                }
            }
        }
    }

    const float ss_res = simd_sum(s_res);
    for (ushort off = 16; off > 0; off >>= 1) {
        const float orv = simd_shuffle_xor(brv, off);
        const uint ori = simd_shuffle_xor(bri, off);
        if (orv > brv || (orv == brv && ori < bri)) { brv = orv; bri = ori; }
        const float opv = simd_shuffle_xor(bpv, off);
        const uint opi = simd_shuffle_xor(bpi, off);
        if (opv > bpv || (opv == bpv && opi < bpi)) { bpv = opv; bpi = opi; }
    }

    threadgroup float sg_rv[8], sg_pv[8], sg_rs[8];
    threadgroup uint  sg_ri[8], sg_pi[8];
    if (lane == 0) {
        sg_rv[sg] = brv; sg_ri[sg] = bri;
        sg_pv[sg] = bpv; sg_pi[sg] = bpi;
        sg_rs[sg] = ss_res;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        float frv = sg_rv[0], fpv = sg_pv[0], tot_s = sg_rs[0];
        uint fri = sg_ri[0], fpi = sg_pi[0];
        for (int k = 1; k < 8; ++k) {
            tot_s += sg_rs[k];
            if (sg_rv[k] > frv || (sg_rv[k] == frv && sg_ri[k] < fri)) { frv = sg_rv[k]; fri = sg_ri[k]; }
            if (sg_pv[k] > fpv || (sg_pv[k] == fpv && sg_pi[k] < fpi)) { fpv = sg_pv[k]; fpi = sg_pi[k]; }
        }
        const size_t out_i = size_t(b) * size_t(p.n_tg) + size_t(tg);
        part_res_val[out_i] = frv;
        part_res_idx[out_i] = fri;
        part_p_val[out_i]   = fpv;
        part_p_idx[out_i]   = fpi;
        part_res_sum[out_i] = tot_s;
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mtp_batch_reject_sample_p2(
    const device float* part_res_val         [[buffer(0)]],
    const device uint* part_res_idx          [[buffer(1)]],
    const device float* part_p_val           [[buffer(2)]],
    const device uint* part_p_idx            [[buffer(3)]],
    const device float* part_res_sum         [[buffer(4)]],
    const device int* d_reject_j             [[buffer(5)]],
    const device BatchMTPStreamMetaGpu* sm   [[buffer(6)]],
    device int* out_toks                     [[buffer(7)]],
    constant BatchRejectPrm& p               [[buffer(8)]],
    uint2 tgid                               [[threadgroup_position_in_grid]],
    uint2 tid2                               [[thread_position_in_threadgroup]],
    uint lane                                [[thread_index_in_simdgroup]],
    uint sg                                  [[simdgroup_index_in_threadgroup]])
{
    const int b = int(tgid.y);
    const BatchMTPStreamMetaGpu meta = sm[b];
    if (meta.is_sampled == 0) return;

    const uint tid = tid2.x;
    const int rej = d_reject_j[b];
    const size_t base = size_t(b) * size_t(p.n_tg);
    float s_res = 0.0f;
    float brv = -INFINITY, bpv = -INFINITY;
    uint bri = 0u, bpi = 0u;
    for (int i = int(tid); i < p.n_tg; i += 256) {
        const size_t idx = base + size_t(i);
        s_res += part_res_sum[idx];
        const float rv = part_res_val[idx];
        const uint ri = part_res_idx[idx];
        if (rv > brv || (rv == brv && ri < bri)) { brv = rv; bri = ri; }
        const float pv = part_p_val[idx];
        const uint pi = part_p_idx[idx];
        if (pv > bpv || (pv == bpv && pi < bpi)) { bpv = pv; bpi = pi; }
    }
    const float ss_res = simd_sum(s_res);
    for (ushort off = 16; off > 0; off >>= 1) {
        const float orv = simd_shuffle_xor(brv, off);
        const uint ori = simd_shuffle_xor(bri, off);
        if (orv > brv || (orv == brv && ori < bri)) { brv = orv; bri = ori; }
        const float opv = simd_shuffle_xor(bpv, off);
        const uint opi = simd_shuffle_xor(bpi, off);
        if (opv > bpv || (opv == bpv && opi < bpi)) { bpv = opv; bpi = opi; }
    }
    threadgroup float sg_rv[8], sg_pv[8], sg_rs[8];
    threadgroup uint  sg_ri[8], sg_pi[8];
    if (lane == 0) {
        sg_rv[sg] = brv; sg_ri[sg] = bri;
        sg_pv[sg] = bpv; sg_pi[sg] = bpi;
        sg_rs[sg] = ss_res;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        float frv = sg_rv[0], fpv = sg_pv[0], tot_s = sg_rs[0];
        uint fri = sg_ri[0], fpi = sg_pi[0];
        for (int k = 1; k < 8; ++k) {
            tot_s += sg_rs[k];
            if (sg_rv[k] > frv || (sg_rv[k] == frv && sg_ri[k] < fri)) { frv = sg_rv[k]; fri = sg_ri[k]; }
            if (sg_pv[k] > fpv || (sg_pv[k] == fpv && sg_pi[k] < fpi)) { fpv = sg_pv[k]; fpi = sg_pi[k]; }
        }
        const bool use_res = (rej < meta.num_draft) && (tot_s > 1e-12f);
        out_toks[b * MTP_B + rej] = int(use_res ? fri : fpi);
    }
}
)MSL";

struct CopyPrm         { uint32_t count; };
struct CopyConv4Prm    { uint32_t n_c, n_d; };
struct CopyConv4DynPrm { uint32_t n_c, n_d; int32_t M; };
struct CopyCapPrm      { int32_t M, cap_slot, n_cap; };
struct CopyCapDynPrm   { int32_t M, row_elems; };
struct KvCommitPrm     { int32_t n_toks, HKV, HD, old_idx, max_size, stride_h, stride_s; };
struct KvCommitDynPrm  { int32_t M, HKV, HD, old_idx, max_size, stride_h, stride_s; };
struct LogitsMaxP1Prm  { int32_t NTG; float temperature, softcap; };
struct EmbedPrm      { float eps; int32_t toks[8]; };
struct EmbedBatchedPrm { float eps; int32_t toks[64]; };
struct Layer0Prm     { float eps; };
struct FinalNormPrm  { int32_t cap_slot, n_cap; float eps; };
struct MixPrm        { int32_t M, Q, QO, QN, MOFF, ADD, NORM, SEQ; float eps; };
struct MixQ4Prm      { int32_t M; };
struct MixGateP1Prm  { int32_t M, Q, QO, MOFF, ADD, SEQ; };
struct MsgDownPrm    { int32_t M; };
struct MsgFgPrm      { int32_t M, QO, MOFF; };
struct RmsMultiPrm   { int32_t QO, QN; float eps; };
struct MsgBlendNormPrm { int32_t M, QO, QN, MOFF; float eps; };
struct QkvPrm        { int32_t S, N, AT3, AT4; };
struct KvPrepPrm {
    int32_t M, HQ, HKV, R;
    int32_t QKV_W, QOFF, ROFF, KOFF, VOFF;
    int32_t X_STRIDE, XK_OFF, XV_OFF;
    int32_t TAU, SEQ, WRITE_DIRECT, IS_ROTATING, WRITE_BASE, MAX_SIZE;
    int32_t KV_OFFSET_BEFORE, STRIDE_H, STRIDE_S;
    float qk_thr, rel_thr, eps, log_n_floor, log_alpha;
};
struct KvPrepBatchedPrm {
    int32_t M, HQ, HKV, R;
    int32_t QKV_W, QOFF, ROFF, KOFF, VOFF;
    int32_t X_STRIDE, XK_OFF, XV_OFF;
    int32_t TAU, WRITE_SLOT, STRIDE_B, STRIDE_H, STRIDE_S;
    float qk_thr, rel_thr, eps;
};
struct AttendPrm {
    int32_t M, HQ, HKV, R, GW, GOFF, WINDOW;
    int32_t S_kv, ext_m, ring_idx, stride_h, stride_s;
    float scale;
};
struct AttendBatchedPrm {
    int32_t M, HQ, HKV, R, GW, GOFF, WINDOW;
    int32_t S_kv, ring_idx, stride_b, stride_h, stride_s;
    float scale;
};
struct DecodePartPrm {
    int32_t M, HQ, HKV, REL, BS, nblocks, WINDOW, SEQ;
    int32_t S_kv, ext_m, ring_idx, stride_h, stride_s;
    float scale;
};
struct DecodePartBatchedPrm {
    int32_t M, HQ, HKV, REL, BS, nblocks, WINDOW;
    int32_t S_kv, ring_idx, stride_b, stride_h, stride_s;
    float scale;
};
struct DecodeCombPrm   { int32_t HQ, HD, nblocks, GW, GOFF; };
struct RoutePrm        { float route_scale; };
struct RouteBatchedPrm { int32_t M; float route_scale; };
struct BucketPrm       { int32_t T, MAX_ACT; };
struct BucketTilesPrm  { int32_t T, MT, NX13, NX2; };
struct MoePrm          { float gate_cap, up_cap; };
struct CombinePrm      { int32_t M; };
struct ArgmaxP1Prm     { int32_t NTG; };
struct ArgmaxP2Prm     { int32_t NTG; };
struct NaxQkvPrm       { int32_t M, S, K, N, AT3, AT4, NT0, NT1, NT2; };
struct NaxArgmaxPrm    { int32_t M, NTG; };
struct NaxMsgPrm       { int32_t M; };

struct MTPInitXPrm {
    int32_t anchor_token, D, block_size, markov_rank;
    float   embed_norm_eps;
};
struct MTPMixPrm {
    int32_t D, n_target_layers, row_start, alpha_row;
    float   eps;
};
struct MTPAddNormPrm {
    int32_t D, src_row_off, write_res;
    float   eps;
};
struct MTPGemvPrm { int32_t N; };
struct MTPKvCtxWritePrm {
    int32_t kv_dim, offset_before, is_rotating, window, ring_start;
    int32_t k_stride_h, k_stride_s, v_stride_h, v_stride_s;
    float   eps, neg_log2_theta_over_half;
};
struct MTPQkvPackPrm {
    int32_t q_dim, kv_dim, total_dim, rope_offset;
    float   eps, neg_log2_theta_over_half;
};
struct MTPSdpaPrm {
    int32_t valid_ctx, total_k, k_stride_h, k_stride_s, v_stride_h, v_stride_s;
    int32_t n_splits, split_chunk;
    float   scale2;
};
struct MTPMarkovCtxPrm   { int32_t D, out_dim; };
struct MTPRnnStepPrm     { int32_t step_idx, rank; };
struct MTPMarkovArgmaxPrm { int32_t v_limit, step_idx, n_part; };
struct MTPMarkovLogitsPrm {
    int32_t v_limit, step_idx, n_part;
    float   temperature, softcap;
};
struct MTPSoftmaxRedPrm  { int32_t v_limit, n_part, row_off; };
struct MTPSoftmaxVecPrm  { int32_t v_limit, n_tg, row_off, pass_idx; float top_p; };
struct MTPTopPState {
    uint32_t prefix_bits;
    float    sum_gt;
    uint32_t thresh_bits, tie_keep_count, found_all;
    uint32_t _pad[3];
    uint32_t tie_base[64];
};
struct MTPScaleGumbelPrm {
    uint64_t rng_seed, rng_step;
    int32_t  v_limit, n_tg, step_idx, _pad;
};
struct MTPGreedyAcceptPrm { int32_t num_draft; };
struct MTPRejectPrm {
    uint64_t rng_seed, rng_step;
    int32_t  v_limit, num_draft;
};
struct MTPRejectSamplePrm {
    uint64_t rng_seed, rng_step;
    int32_t  v_limit, n_tg, num_draft, _pad;
};

struct BatchMTPStreamMetaGpu {
    int32_t  anchor_token;
    int32_t  n_ctx;
    int32_t  ctx_start;
    int32_t  num_draft;
    int32_t  d_start;
    int32_t  v_start;
    int32_t  is_sampled;
    int32_t  do_top_p;
    float    temperature;
    float    top_p;
    float    softcap;
    float    _pad0;
    uint64_t rng_seed;
    uint64_t rng_step;
    uint64_t out_hidden_addr;
};

struct BatchMTPDraftKVMetaGpu {
    uint64_t k_addr;
    uint64_t v_addr;
    int32_t  offset_before;
    int32_t  is_rotating;
    int32_t  window;
    int32_t  ring_start;
    int32_t  valid_ctx;
    int32_t  k_stride_h;
    int32_t  k_stride_s;
    int32_t  v_stride_h;
    int32_t  v_stride_s;
    int32_t  offset_after;
};

struct BatchMTPTargetVisitMetaGpu {
    uint64_t kv_k_addr;
    uint64_t kv_v_addr;
    uint64_t conv_addr[4];
    uint64_t hist_ks_addr;
    uint64_t hist_vs_addr;
    uint64_t hist_cs_addr;
    uint64_t hist_ms_addr;
    uint64_t stage_k_addr;
    uint64_t stage_v_addr;
    int32_t  prev_offset;
    int32_t  old_idx;
    int32_t  max_size;
    int32_t  is_rotating;
    int32_t  need_staging;
    int32_t  write_base;
    int32_t  s_kv;
    int32_t  ext_m;
    int32_t  ring_idx;
    int32_t  stride_h;
    int32_t  stride_s;
    int32_t  v_start;
    int32_t  M_b;
    int32_t  n_kv;
    int32_t  head_dim;
    int32_t  _pad;
};

struct KvPrepMtpBatchedPrm {
    int32_t M, HQ, HKV, R;
    int32_t QKV_W, QOFF, ROFF, KOFF, VOFF;
    int32_t X_STRIDE, XK_OFF, XV_OFF;
    int32_t TAU;
    float qk_thr, rel_thr, eps, log_n_floor, log_alpha;
};

struct AttendMtpBatchedPrm {
    int32_t M, HQ, HKV, R, GW, GOFF, WINDOW;
    float scale;
};

struct DecodePartMtpBatchedPrm {
    int32_t M, HQ, HKV, REL, BS, nblocks, WINDOW;
    float scale;
};

struct MtpBatchCommitPrm {
    int32_t B, NV, row_elems;
};

struct NaxGuPrm {
    int32_t M, K, INTER;
};

struct BatchInitXPrm {
    int32_t B, D, block_size, markov_rank;
    float embed_norm_eps;
};

struct BatchGemvPrm {
    int32_t M, K, N;
};

struct BatchKvCtxWritePrm {
    int32_t kv_dim, num_layers, layer_idx, _pad;
    float eps, neg_log2_theta_over_half;
};

struct BatchQkvPackPrm {
    int32_t q_dim, kv_dim, total_dim, num_layers, layer_idx, _pad;
    float eps, neg_log2_theta_over_half;
};

struct BatchSdpaPrm {
    int32_t num_layers, layer_idx, n_splits, _pad;
    float scale2;
};

struct BatchMarkovCtxPrm {
    int32_t B, m_d_tot, D, out_dim;
};

struct BatchRnnStepPrm {
    int32_t B, step_idx, rank, _pad;
};

struct BatchMarkovPrm {
    int32_t B, v_limit, step_idx, n_part, max_k, _pad;
};

struct BatchSoftmaxPrm {
    int32_t v_limit, n_part, n_tg, step_idx, pass_idx, mode, max_k, only_topp;
};

struct BatchGumbelPrm {
    int32_t v_limit, n_tg, step_idx, max_k;
};

struct BatchRejectPrm {
    int32_t v_limit, n_tg, max_k, _pad;
};

struct BufSlice {
    id<MTLBuffer> buf = nil;
    size_t offset = 0;
};

struct CachedPtrBuf {
    id<MTLBuffer> buf = nil;
    size_t span = 0;
};

struct VisitEntry {
    int32_t layer_idx;
    int32_t cache_idx;
    int32_t pass_idx;
};

struct LayerDeviceWeights {
    bool registered = false;
    int32_t is_sliding = 0;
    int32_t is_block_boundary = 0;
    int32_t has_attn_input = 0;
    int32_t has_message_gates = 0;
    int32_t n_heads = 0;
    int32_t n_kv = 0;
    int32_t head_dim = 0;
    int32_t sliding_window = 0;
    int32_t rel_extent = 0;
    int32_t qkv_at[5] = {0, 0, 0, 0, 0};
    int32_t qkv_total_rows = 0;
    int32_t moe_mode = 0;
    float attn_eps = 1e-6f;
    float mlp_eps = 1e-6f;

    BufSlice stack, o_proj;
    BufSlice attn_query, attn_norm;
    BufSlice mlp_query, mlp_norm;
    BufSlice attn_taps, mlp_taps;
    BufSlice k_taps, v_taps, kdyn_w, vdyn_w;
    BufSlice attn_gate_down, attn_gate_forget, attn_gate_gain;
    BufSlice mlp_gate_down, mlp_gate_forget, mlp_gate_gain;
    BufSlice router_w, router_b;
    BufSlice sink_w, q_scale, rel_scale, rel_proj;
    BufSlice w13_weight, w13_scales, w13_gscale;
    BufSlice w2_weight, w2_scales, w2_gscale;
    BufSlice attn_gate_fg_bf16, mlp_gate_fg_bf16, router_w_bf16;
    bool has_attn_gate_fg_bf16 = false;
    bool has_mlp_gate_fg_bf16 = false;
    bool has_router_w_bf16 = false;
};

struct GlobalDeviceWeights {
    bool registered = false;
    BufSlice embed_tokens, embed_norm;
    BufSlice message_queries, final_query, final_norm, lm_head;
    const void* embed_tokens_ptr = nullptr;
    const void* embed_norm_ptr = nullptr;
    const void* lm_head_ptr = nullptr;
    float embed_norm_eps = 1e-6f;
    float final_norm_eps = 1e-6f;
};

struct MTPLayerMetalWeights {
    int32_t is_local = 0;
    int32_t window = 0;
    BufSlice dflare_norm;
    BufSlice input_layernorm;
    BufSlice post_attention_layernorm;
    BufSlice kv_ctx_w;
    BufSlice k_norm_ctx;
    BufSlice qkv_w;
    BufSlice q_norm;
    BufSlice k_norm;
    BufSlice o_proj_w;
    BufSlice gu_w;
    BufSlice down_proj_w;
};

struct MTPAttachedState {
    bool attached = false;
    MTPDraftConfig cfg{};
    std::vector<int32_t> capture_layer_ids;
    bool has_embed_norm = false;

    id<MTLLibrary> lib = nil;
    BufSlice embed_tokens, embed_norm, mask_embed, final_norm, lm_head;
    BufSlice fuse_alpha, markov_dflare_norm, markov_w1;
    BufSlice rnn_w_ctx_t, rnn_w_step_t, rnn_b_f32, fold_weight, fold_bias;
    std::vector<MTPLayerMetalWeights> layers;

    id<MTLComputePipelineState> pso_init_x[2] = {nil, nil};
    id<MTLComputePipelineState> pso_dflare_mix = nil;
    id<MTLComputePipelineState> pso_add_rmsnorm[2] = {nil, nil};
    id<MTLComputePipelineState> pso_gemv_ctx_kv[9] = {};
    id<MTLComputePipelineState> pso_gemv_qkv = nil;
    id<MTLComputePipelineState> pso_gemv_o_proj = nil;
    id<MTLComputePipelineState> pso_gu_silu = nil;
    id<MTLComputePipelineState> pso_gemv_down = nil;
    id<MTLComputePipelineState> pso_gemv_head_r2[8] = {};
    id<MTLComputePipelineState> pso_kv_ctx_write = nil;
    id<MTLComputePipelineState> pso_qkv_pack = nil;
    id<MTLComputePipelineState> pso_sdpa_single = nil;
    id<MTLComputePipelineState> pso_sdpa_split_p1 = nil;
    id<MTLComputePipelineState> pso_sdpa_split_p2 = nil;
    id<MTLComputePipelineState> pso_markov_ctx_p1[8] = {};
    id<MTLComputePipelineState> pso_markov_ctx_p2[8] = {};
    id<MTLComputePipelineState> pso_rnn_step = nil;
    id<MTLComputePipelineState> pso_markov_argmax_p1 = nil;
    id<MTLComputePipelineState> pso_markov_argmax_p2 = nil;
    id<MTLComputePipelineState> pso_markov_logits_max_p1 = nil;
    id<MTLComputePipelineState> pso_softmax_max_p2 = nil;
    id<MTLComputePipelineState> pso_softmax_exp_sum_p1 = nil;
    id<MTLComputePipelineState> pso_softmax_sum_p2 = nil;
    id<MTLComputePipelineState> pso_probs_scale = nil;
    id<MTLComputePipelineState> pso_topp_hist = nil;
    id<MTLComputePipelineState> pso_topp_scan = nil;
    id<MTLComputePipelineState> pso_topp_mask_sum = nil;
    id<MTLComputePipelineState> pso_probs_scale_gumbel_p1 = nil;
    id<MTLComputePipelineState> pso_gumbel_p2 = nil;
    id<MTLComputePipelineState> pso_greedy_accept = nil;
    id<MTLComputePipelineState> pso_reject_prefix = nil;
    id<MTLComputePipelineState> pso_reject_sample_p1 = nil;
    id<MTLComputePipelineState> pso_reject_sample_p2 = nil;

    id<MTLBuffer> hidden_buf = nil;
    id<MTLBuffer> inv_freq_buf = nil;
    id<MTLBuffer> x_buf = nil;
    id<MTLBuffer> residual_buf = nil;
    id<MTLBuffer> norm_buf = nil;
    id<MTLBuffer> ctx_buf = nil;
    id<MTLBuffer> kv_ctx_buf = nil;
    id<MTLBuffer> qkv_buf = nil;
    id<MTLBuffer> q_buf = nil;
    id<MTLBuffer> bk_buf = nil;
    id<MTLBuffer> bv_buf = nil;
    id<MTLBuffer> attn_out_buf = nil;
    id<MTLBuffer> mlp_mid_buf = nil;
    id<MTLBuffer> sdpa_part_o = nil;
    id<MTLBuffer> sdpa_part_ml = nil;
    id<MTLBuffer> x_out_buf = nil;
    id<MTLBuffer> base_logits_buf = nil;
    id<MTLBuffer> f_rnn_buf = nil;
    id<MTLBuffer> rnn_ctx_buf = nil;
    id<MTLBuffer> rnn_state_buf[2] = {nil, nil};
    id<MTLBuffer> rnn_latent_buf = nil;
    id<MTLBuffer> markov_ctx_part_x = nil;
    id<MTLBuffer> markov_ctx_part_f = nil;
    id<MTLBuffer> q_probs_buf = nil;
    id<MTLBuffer> p_probs_buf = nil;
    id<MTLBuffer> row_max_buf = nil;
    id<MTLBuffer> row_inv_sum_buf = nil;
    id<MTLBuffer> topp_states_buf = nil;
    id<MTLBuffer> softmax_part_buf = nil;
    id<MTLBuffer> topp_part_cnt_buf = nil;
    id<MTLBuffer> markov_part_val = nil;
    id<MTLBuffer> markov_part_idx = nil;
    id<MTLBuffer> reject_res_val = nil;
    id<MTLBuffer> reject_res_idx = nil;
    id<MTLBuffer> reject_p_val = nil;
    id<MTLBuffer> reject_p_idx = nil;
    id<MTLBuffer> reject_res_sum = nil;
    id<MTLBuffer> prev_tok_buf = nil;
    id<MTLBuffer> d_toks_buf = nil;
    id<MTLBuffer> verify_toks_buf = nil;
    id<MTLBuffer> out_toks_buf = nil;
    id<MTLBuffer> d_reject_j_buf = nil;
    id<MTLBuffer> d_n_acc_buf = nil;

    int32_t last_num_draft = 0;
    std::vector<BufSlice> tmp_kv_k;
    std::vector<BufSlice> tmp_kv_v;

    bool batched_init = false;
    bool batched_sampling_init = false;
    int32_t batched_max_mv = 0;
    size_t batched_verify_kv_visit_stride_bytes = 0;
    std::vector<size_t> batched_conv_hist_visit_off;

    id<MTLComputePipelineState> pso_batch_init_x[2] = {nil, nil};
    id<MTLComputePipelineState> pso_batch_gemv = nil;
    id<MTLComputePipelineState> pso_batch_gu_silu = nil;
    id<MTLComputePipelineState> pso_batch_kv_ctx_write = nil;
    id<MTLComputePipelineState> pso_batch_qkv_pack = nil;
    id<MTLComputePipelineState> pso_batch_sdpa_single = nil;
    id<MTLComputePipelineState> pso_batch_sdpa_split_p1 = nil;
    id<MTLComputePipelineState> pso_batch_sdpa_split_p2 = nil;
    id<MTLComputePipelineState> pso_batch_final_norm = nil;
    id<MTLComputePipelineState> pso_batch_f_rnn = nil;
    id<MTLComputePipelineState> pso_batch_markov_ctx_p1 = nil;
    id<MTLComputePipelineState> pso_batch_markov_ctx_p2 = nil;
    id<MTLComputePipelineState> pso_batch_rnn_step = nil;
    id<MTLComputePipelineState> pso_batch_markov_mixed_p1 = nil;
    id<MTLComputePipelineState> pso_batch_markov_argmax_p2 = nil;
    id<MTLComputePipelineState> pso_batch_softmax_max_p2 = nil;
    id<MTLComputePipelineState> pso_batch_softmax_exp_sum_p1 = nil;
    id<MTLComputePipelineState> pso_batch_softmax_sum_p2 = nil;
    id<MTLComputePipelineState> pso_batch_topp_hist = nil;
    id<MTLComputePipelineState> pso_batch_topp_scan = nil;
    id<MTLComputePipelineState> pso_batch_topp_mask_sum = nil;
    id<MTLComputePipelineState> pso_batch_probs_scale = nil;
    id<MTLComputePipelineState> pso_batch_probs_scale_gumbel_p1 = nil;
    id<MTLComputePipelineState> pso_batch_gumbel_p2 = nil;
    id<MTLComputePipelineState> pso_batch_greedy_accept = nil;
    id<MTLComputePipelineState> pso_batch_reject_prefix = nil;
    id<MTLComputePipelineState> pso_batch_reject_sample_p1 = nil;
    id<MTLComputePipelineState> pso_batch_reject_sample_p2 = nil;

    id<MTLBuffer> batch_stream_meta_buf = nil;
    id<MTLBuffer> batch_draft_kv_meta_buf = nil;
    id<MTLBuffer> batch_target_vmeta_buf = nil;
    id<MTLBuffer> batch_ctx_to_stream_buf = nil;
    id<MTLBuffer> batch_ctx_local_pos_buf = nil;
    id<MTLBuffer> batch_d_to_stream_buf = nil;
    id<MTLBuffer> batch_d_local_step_buf = nil;
    id<MTLBuffer> batch_tok_to_stream_buf = nil;
    id<MTLBuffer> batch_tok_local_pos_buf = nil;
    id<MTLBuffer> batch_hidden_buf = nil;
    id<MTLBuffer> batch_x_buf = nil;
    id<MTLBuffer> batch_residual_buf = nil;
    id<MTLBuffer> batch_norm_buf = nil;
    id<MTLBuffer> batch_ctx_buf = nil;
    id<MTLBuffer> batch_kv_ctx_buf = nil;
    id<MTLBuffer> batch_qkv_buf = nil;
    id<MTLBuffer> batch_q_buf = nil;
    id<MTLBuffer> batch_bk_buf = nil;
    id<MTLBuffer> batch_bv_buf = nil;
    id<MTLBuffer> batch_attn_out_buf = nil;
    id<MTLBuffer> batch_mlp_mid_buf = nil;
    id<MTLBuffer> batch_sdpa_part_o = nil;
    id<MTLBuffer> batch_sdpa_part_ml = nil;
    id<MTLBuffer> batch_x_out_buf = nil;
    id<MTLBuffer> batch_base_logits_buf = nil;
    id<MTLBuffer> batch_f_rnn_buf = nil;
    id<MTLBuffer> batch_rnn_ctx_buf = nil;
    id<MTLBuffer> batch_rnn_state_buf[2] = {nil, nil};
    id<MTLBuffer> batch_rnn_latent_buf = nil;
    id<MTLBuffer> batch_markov_ctx_part_x = nil;
    id<MTLBuffer> batch_markov_ctx_part_f = nil;
    id<MTLBuffer> batch_markov_part_val = nil;
    id<MTLBuffer> batch_markov_part_idx = nil;
    id<MTLBuffer> batch_prev_tok_buf = nil;
    id<MTLBuffer> batch_d_toks_buf = nil;
    id<MTLBuffer> batch_verify_toks_buf = nil;
    id<MTLBuffer> batch_out_toks_buf = nil;
    id<MTLBuffer> batch_d_reject_j_buf = nil;
    id<MTLBuffer> batch_d_n_acc_buf = nil;
    id<MTLBuffer> batch_vcap_buf = nil;
    id<MTLBuffer> batch_verify_k_buf = nil;
    id<MTLBuffer> batch_verify_v_buf = nil;
    id<MTLBuffer> batch_conv_hist_buf = nil;
    id<MTLBuffer> batch_argmax_pv_buf = nil;
    id<MTLBuffer> batch_argmax_pi_buf = nil;

    id<MTLBuffer> batch_q_probs_buf = nil;
    id<MTLBuffer> batch_p_probs_buf = nil;
    id<MTLBuffer> batch_row_max_buf = nil;
    id<MTLBuffer> batch_row_inv_sum_buf = nil;
    id<MTLBuffer> batch_topp_states_buf = nil;
    id<MTLBuffer> batch_softmax_part_buf = nil;
    id<MTLBuffer> batch_topp_part_cnt_buf = nil;
    id<MTLBuffer> batch_reject_res_val = nil;
    id<MTLBuffer> batch_reject_res_idx = nil;
    id<MTLBuffer> batch_reject_p_val = nil;
    id<MTLBuffer> batch_reject_p_idx = nil;
    id<MTLBuffer> batch_reject_res_sum = nil;
};

struct PendingRotating {
    bool active = false;
    int32_t old_idx = 0;
    int32_t max_size = 0;
    int32_t prev_offset = 0;
    int32_t m_tokens = 0;
};

struct PendingVerify {
    int32_t prev_offset = 0;
    int32_t old_idx = 0;
    int32_t max_size = 0;
    int32_t m_tokens = 0;
    bool    is_rotating = false;
    bool    kv_staged = false;
};

struct InklingCtx {
    InklingModelConfig cfg{};
    size_t page_size = 16384;
    int32_t max_g = 4;
    int32_t max_nred = 8;
    int32_t max_blocks = 7;
    int32_t max_nblocks = 128;
    int32_t max_captures = 8;
    int32_t mix_tg = 256;
    int32_t route_parts = 8;
    int32_t bucket_tg = 128;

    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> lib = nil;
    id<MTLLibrary> lib_nax = nil;

    id<MTLComputePipelineState> pso_copy_bf16 = nil;
    id<MTLComputePipelineState> pso_copy_f32 = nil;
    id<MTLComputePipelineState> pso_copy_conv4 = nil;
    id<MTLComputePipelineState> pso_copy_cap = nil;
    id<MTLComputePipelineState> pso_kv_commit = nil;
    id<MTLComputePipelineState> pso_embed = nil;
    id<MTLComputePipelineState> pso_embed_from_buf = nil;
    id<MTLComputePipelineState> pso_embed_batched = nil;
    id<MTLComputePipelineState> pso_copy_conv4_dyn = nil;
    id<MTLComputePipelineState> pso_copy_cap_dyn = nil;
    id<MTLComputePipelineState> pso_kv_commit_dyn = nil;
    id<MTLComputePipelineState> pso_head_logits_max_p1[9] = {};
    id<MTLComputePipelineState> pso_layer0_rms = nil;
    id<MTLComputePipelineState> pso_final_norm = nil;
    id<MTLComputePipelineState> pso_final_norm_bf16 = nil;
    id<MTLComputePipelineState> pso_msg_down_batched = nil;
    id<MTLComputePipelineState> pso_msg_down_batched_bf16 = nil;
    id<MTLComputePipelineState> pso_msg_fg_blend = nil;
    id<MTLComputePipelineState> pso_rms_multi = nil;
    id<MTLComputePipelineState> pso_msg_blend_norm = nil;
    id<MTLComputePipelineState> pso_decode_comb = nil;
    id<MTLComputePipelineState> pso_route = nil;
    id<MTLComputePipelineState> pso_route_scores = nil;
    id<MTLComputePipelineState> pso_route_topk = nil;
    id<MTLComputePipelineState> pso_bucket = nil;
    id<MTLComputePipelineState> pso_bucket_tiles = nil;
    id<MTLComputePipelineState> pso_m1_w13_nvfp4 = nil;
    id<MTLComputePipelineState> pso_m1_w13_mxfp8 = nil;
    id<MTLComputePipelineState> pso_m1_w2_nvfp4 = nil;
    id<MTLComputePipelineState> pso_m1_w2_mxfp8 = nil;
    id<MTLComputePipelineState> pso_combine_noseq = nil;
    id<MTLComputePipelineState> pso_combine_seq = nil;
    id<MTLComputePipelineState> pso_argmax_p2 = nil;
    id<MTLComputePipelineState> pso_nax_qkv_16x64 = nil;
    id<MTLComputePipelineState> pso_nax_qkv_32x64 = nil;
    id<MTLComputePipelineState> pso_nax_qkv_64x64 = nil;
    id<MTLComputePipelineState> pso_nax_qkv_64x32 = nil;
    id<MTLComputePipelineState> pso_nax_argmax_16x64 = nil;
    id<MTLComputePipelineState> pso_nax_msg_fg_16x64 = nil;
    id<MTLComputePipelineState> pso_nax_msg_fg_32x64 = nil;
    id<MTLComputePipelineState> pso_nax_msg_fg_64x64 = nil;
    id<MTLComputePipelineState> pso_nax_router_16x32 = nil;
    id<MTLComputePipelineState> pso_final_norm_bf16_cap = nil;
    id<MTLComputePipelineState> pso_combine_mtp_batched = nil;
    id<MTLComputePipelineState> pso_kv_commit_rot_mtp_batched = nil;
    id<MTLComputePipelineState> pso_copy_conv4_mtp_batched = nil;
    id<MTLComputePipelineState> pso_copy_cap_mtp_batched = nil;
    id<MTLComputePipelineState> pso_nax_head_logits_max_16x64 = nil;
    id<MTLComputePipelineState> pso_mtp_nax_gu_silu_16x64 = nil;
    id<MTLComputePipelineState> pso_mtp_nax_gu_silu_32x64 = nil;
    id<MTLComputePipelineState> pso_head_logits_max_mixed_p1[9] = {};

    std::unordered_map<uint64_t, id<MTLComputePipelineState>> pso_mix_cache;
    std::unordered_map<uint64_t, id<MTLComputePipelineState>> pso_mix_q4_cache;
    std::unordered_map<uint64_t, id<MTLComputePipelineState>> pso_mix_gate_p1_cache;
    std::unordered_map<uint64_t, id<MTLComputePipelineState>> pso_qkv_cache;
    std::unordered_map<uint64_t, id<MTLComputePipelineState>> pso_kv_prep_cache;
    std::unordered_map<uint64_t, id<MTLComputePipelineState>> pso_kv_prep_batched_cache;
    std::unordered_map<uint64_t, id<MTLComputePipelineState>> pso_attend_cache;
    std::unordered_map<uint64_t, id<MTLComputePipelineState>> pso_attend_batched_cache;
    std::unordered_map<uint64_t, id<MTLComputePipelineState>> pso_dpart_cache;
    std::unordered_map<uint64_t, id<MTLComputePipelineState>> pso_dpart_batched_cache;
    std::unordered_map<uint64_t, id<MTLComputePipelineState>> pso_grp_moe_cache;
    std::unordered_map<uint64_t, id<MTLComputePipelineState>> pso_grp_moe_tiles_cache;
    std::unordered_map<uint64_t, id<MTLComputePipelineState>> pso_argmax_p1_cache;
    std::unordered_map<uint64_t, id<MTLComputePipelineState>> pso_mix_mtp_batched_cache;
    std::unordered_map<uint64_t, id<MTLComputePipelineState>> pso_kv_prep_mtp_batched_cache;
    std::unordered_map<uint64_t, id<MTLComputePipelineState>> pso_attend_mtp_batched_cache;
    std::unordered_map<uint64_t, id<MTLComputePipelineState>> pso_dpart_mtp_batched_cache;

    id<MTLBuffer> zero_buf = nil;
    id<MTLBuffer> scratch = nil;
    size_t scratch_bytes = 0;

    id<MTLBuffer> part_buf = nil;
    size_t part_buf_bytes = 0;
    size_t part_off_max = 0;
    size_t part_off_sum = 0;
    size_t part_off_acc = 0;

    id<MTLBuffer> conv_hist_buf = nil;
    size_t conv_hist_bytes = 0;
    std::vector<size_t> conv_hist_visit_off;
    bool conv_hist_active = false;
    int32_t conv_hist_m = 0;
    int32_t last_verify_m = 0;

    size_t off_x = 0;
    size_t off_pre_loop_x = 0;
    size_t off_bank_x = 0;
    size_t off_blocks = 0;
    size_t off_bank_blocks = 0;
    size_t off_messages = 0;
    size_t off_qkv_in = 0;
    size_t off_qkv = 0;
    size_t off_qo = 0;
    size_t off_bias = 0;
    size_t off_sinks = 0;
    size_t off_att = 0;
    size_t off_att_out = 0;
    size_t off_mlp_in = 0;
    size_t off_resid = 0;
    size_t off_inds = 0;
    size_t off_weights = 0;
    size_t off_offsets = 0;
    size_t off_sorted_slots = 0;
    size_t off_active_list = 0;
    size_t off_moe_h = 0;
    size_t off_ys = 0;
    size_t off_scratch_conv_a = 0;
    size_t off_scratch_conv_b = 0;
    size_t off_scratch_conv_c = 0;
    size_t off_scratch_conv_d = 0;
    size_t off_scratch_partial = 0;
    size_t off_verify_k = 0;
    size_t off_verify_v = 0;
    size_t off_pre_norm = 0;
    size_t off_normed_f32 = 0;
    size_t off_normed_bf16 = 0;
    size_t off_captures = 0;
    size_t off_argmax_pv = 0;
    size_t off_argmax_pi = 0;
    size_t off_out_toks = 0;
    size_t off_msg_o = 0;
    size_t off_msg_gate_in = 0;
    size_t off_msg_low = 0;
    size_t off_route_scores = 0;
    size_t off_tile_expert = 0;
    size_t off_tile_t0 = 0;
    size_t off_indirect_args = 0;
    size_t off_batch_counts = 0;
    size_t off_batch_tau = 0;
    size_t verify_kv_visit_stride_bytes = 0;
    size_t batch_visit_meta_stride_bytes = 0;

    struct BatchVisitPlan {
        int32_t write_slot = 0;
        int32_t new_s_len = 0;
        int32_t ring_idx = 0;
        int32_t new_idx = 0;
        int32_t new_off_scalar = 0;
        int32_t nblocks = 0;
        int32_t bs = 0;
        bool use_splitk = false;
        bool use_tau = false;
    };
    std::vector<BatchVisitPlan> tmp_batch_plans;

    std::vector<VisitEntry> schedule;
    GlobalDeviceWeights globals;
    std::vector<LayerDeviceWeights> layers;
    std::vector<PendingRotating> pending_rot;
    std::vector<PendingVerify> pending_verify;
    std::vector<BufSlice> tmp_kv_k;
    std::vector<BufSlice> tmp_kv_v;
    std::vector<BufSlice> tmp_conv;

    MTPAttachedState mtp;

    std::unordered_map<uintptr_t, CachedPtrBuf> ptr_cache;

    mach_timebase_info_data_t tb{};
};

static inline size_t align_up(size_t v, size_t a) {
    return (v + a - 1) & ~(a - 1);
}

static inline double mach_to_ms(uint64_t ticks, const mach_timebase_info_data_t& tb) {
    return (double)ticks * (double)tb.numer / (double)tb.denom / 1e6;
}

static BufSlice wrap_static_ptr(InklingCtx* ctx, const void* ptr, size_t nbytes) {
    if (!ptr || nbytes == 0) {
        return {ctx->zero_buf, 0};
    }
    uintptr_t addr = (uintptr_t)ptr;
    uintptr_t base = addr & ~(ctx->page_size - 1);
    size_t off = (size_t)(addr - base);
    size_t span = align_up(off + nbytes, ctx->page_size);
    id<MTLBuffer> buf = [ctx->device newBufferWithBytesNoCopy:(void*)base
                                                       length:span
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
    if (!buf) {
        fprintf(stderr, "[inkling_decode] ERROR: newBufferWithBytesNoCopy failed (ptr=%p, bytes=%zu)\n", ptr, nbytes);
        return {nil, 0};
    }
    return {buf, off};
}

static BufSlice wrap_cached_ptr(InklingCtx* ctx, const void* ptr, size_t nbytes) {
    if (!ptr || nbytes == 0) {
        return {ctx->zero_buf, 0};
    }
    uintptr_t addr = (uintptr_t)ptr;
    uintptr_t base = addr & ~(ctx->page_size - 1);
    size_t off = (size_t)(addr - base);
    size_t span = align_up(off + nbytes, ctx->page_size);

    auto it = ctx->ptr_cache.find(base);
    if (it != ctx->ptr_cache.end() && it->second.span >= span) {
        return {it->second.buf, off};
    }
    id<MTLBuffer> buf = [ctx->device newBufferWithBytesNoCopy:(void*)base
                                                       length:span
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
    if (!buf) {
        fprintf(stderr, "[inkling_decode] ERROR: cached newBufferWithBytesNoCopy failed (ptr=%p, bytes=%zu)\n", ptr, nbytes);
        return {nil, 0};
    }
    ctx->ptr_cache[base] = {buf, span};
    return {buf, off};
}

static BufSlice wrap_ptr_nocache(InklingCtx* ctx, uintptr_t ptr, size_t nbytes) {
    if (!ptr || nbytes == 0) {
        return {ctx->zero_buf, 0};
    }
    uintptr_t base = ptr & ~(ctx->page_size - 1);
    size_t off = (size_t)(ptr - base);
    size_t span = align_up(off + nbytes, ctx->page_size);
    id<MTLBuffer> buf = [ctx->device newBufferWithBytesNoCopy:(void*)base
                                                       length:span
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
    if (!buf) {
        fprintf(stderr, "[inkling_decode] ERROR: nocache newBufferWithBytesNoCopy failed (ptr=%p, bytes=%zu)\n",
                (void*)ptr, nbytes);
        return {nil, 0};
    }
    return {buf, off};
}

static id<MTLComputePipelineState> make_pso(
    id<MTLDevice> dev,
    id<MTLLibrary> lib,
    NSString* name,
    MTLFunctionConstantValues* fcv = nil)
{
    NSError* err = nil;
    id<MTLFunction> fn = nil;
    if (fcv) {
        fn = [lib newFunctionWithName:name constantValues:fcv error:&err];
    } else {
        fn = [lib newFunctionWithName:name];
    }
    if (!fn) {
        fprintf(stderr, "[inkling_decode] ERROR: function '%s' not found: %s\n",
                name.UTF8String, err ? err.localizedDescription.UTF8String : "unknown");
        return nil;
    }
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        fprintf(stderr, "[inkling_decode] ERROR: PSO '%s' failed: %s\n",
                name.UTF8String, err ? err.localizedDescription.UTF8String : "unknown");
        return nil;
    }
    return pso;
}

static int window_tg(int hd) {
    int v = ((hd * 8 + 31) / 32) * 32;
    return v < 1024 ? v : 1024;
}

static id<MTLComputePipelineState> get_mix_pso(InklingCtx* ctx, bool gate, bool conv, int k) {
    uint64_t key = ((uint64_t)(gate ? 1 : 0) << 16) | ((uint64_t)(conv ? 1 : 0) << 8) | (uint64_t)(k & 0xFF);
    auto it = ctx->pso_mix_cache.find(key);
    if (it != ctx->pso_mix_cache.end()) return it->second;
    MTLFunctionConstantValues* fcv = [MTLFunctionConstantValues new];
    [fcv setConstantValue:&gate type:MTLDataTypeBool atIndex:0];
    [fcv setConstantValue:&conv type:MTLDataTypeBool atIndex:1];
    [fcv setConstantValue:&k    type:MTLDataTypeInt  atIndex:2];
    id<MTLComputePipelineState> pso = make_pso(ctx->device, ctx->lib, @"inkling_mix", fcv);
    ctx->pso_mix_cache[key] = pso;
    return pso;
}

static id<MTLComputePipelineState> get_qkv_pso(InklingCtx* ctx, int m, int sg, int kdim) {
    uint64_t key = ((uint64_t)kdim << 16) | ((uint64_t)m << 8) | (uint64_t)sg;
    auto it = ctx->pso_qkv_cache.find(key);
    if (it != ctx->pso_qkv_cache.end()) return it->second;
    MTLFunctionConstantValues* fcv = [MTLFunctionConstantValues new];
    [fcv setConstantValue:&m    type:MTLDataTypeInt atIndex:4];
    [fcv setConstantValue:&sg   type:MTLDataTypeInt atIndex:6];
    [fcv setConstantValue:&kdim type:MTLDataTypeInt atIndex:10];
    id<MTLComputePipelineState> pso = make_pso(ctx->device, ctx->lib, @"inkling_qkv_wide", fcv);
    ctx->pso_qkv_cache[key] = pso;
    return pso;
}

static id<MTLComputePipelineState> get_kv_prep_pso(InklingCtx* ctx, bool dyn, int hd) {
    uint64_t key = ((uint64_t)(dyn ? 1 : 0) << 16) | (uint64_t)hd;
    auto it = ctx->pso_kv_prep_cache.find(key);
    if (it != ctx->pso_kv_prep_cache.end()) return it->second;
    MTLFunctionConstantValues* fcv = [MTLFunctionConstantValues new];
    [fcv setConstantValue:&dyn type:MTLDataTypeBool atIndex:3];
    [fcv setConstantValue:&hd  type:MTLDataTypeInt  atIndex:8];
    id<MTLComputePipelineState> pso = make_pso(ctx->device, ctx->lib, @"inkling_kv_prep", fcv);
    ctx->pso_kv_prep_cache[key] = pso;
    return pso;
}

static id<MTLComputePipelineState> get_attend_pso(InklingCtx* ctx, int g, int hd, bool seq) {
    uint64_t key = ((uint64_t)g << 16) | ((uint64_t)hd << 1) | (uint64_t)(seq ? 1 : 0);
    auto it = ctx->pso_attend_cache.find(key);
    if (it != ctx->pso_attend_cache.end()) return it->second;
    MTLFunctionConstantValues* fcv = [MTLFunctionConstantValues new];
    [fcv setConstantValue:&g   type:MTLDataTypeInt  atIndex:7];
    [fcv setConstantValue:&hd  type:MTLDataTypeInt  atIndex:8];
    [fcv setConstantValue:&seq type:MTLDataTypeBool atIndex:9];
    id<MTLComputePipelineState> pso = make_pso(ctx->device, ctx->lib, @"inkling_attend", fcv);
    ctx->pso_attend_cache[key] = pso;
    return pso;
}

static id<MTLComputePipelineState> get_dpart_pso(InklingCtx* ctx, int g, int hd) {
    uint64_t key = ((uint64_t)g << 16) | (uint64_t)hd;
    auto it = ctx->pso_dpart_cache.find(key);
    if (it != ctx->pso_dpart_cache.end()) return it->second;
    MTLFunctionConstantValues* fcv = [MTLFunctionConstantValues new];
    [fcv setConstantValue:&g  type:MTLDataTypeInt atIndex:7];
    [fcv setConstantValue:&hd type:MTLDataTypeInt atIndex:8];
    id<MTLComputePipelineState> pso = make_pso(ctx->device, ctx->lib, @"inkling_decode_partials", fcv);
    ctx->pso_dpart_cache[key] = pso;
    return pso;
}

static id<MTLComputePipelineState> get_grp_moe_pso(InklingCtx* ctx, bool is_nvfp4, bool is_w2, int sg, int mt) {
    uint64_t key = ((uint64_t)(is_nvfp4 ? 0 : 1) << 24)
                 | ((uint64_t)(is_w2 ? 1 : 0) << 16)
                 | ((uint64_t)mt << 8)
                 | (uint64_t)sg;
    auto it = ctx->pso_grp_moe_cache.find(key);
    if (it != ctx->pso_grp_moe_cache.end()) return it->second;
    MTLFunctionConstantValues* fcv = [MTLFunctionConstantValues new];
    [fcv setConstantValue:&sg type:MTLDataTypeInt atIndex:6];
    [fcv setConstantValue:&mt type:MTLDataTypeInt atIndex:11];
    NSString* name = nil;
    if (is_nvfp4) {
        name = is_w2 ? @"inkling_grp_w2_nvfp4" : @"inkling_grp_w13_nvfp4";
    } else {
        name = is_w2 ? @"inkling_grp_w2_mxfp8" : @"inkling_grp_w13_mxfp8";
    }
    id<MTLComputePipelineState> pso = make_pso(ctx->device, ctx->lib, name, fcv);
    ctx->pso_grp_moe_cache[key] = pso;
    return pso;
}

static id<MTLComputePipelineState> get_argmax_p1_pso(InklingCtx* ctx, int m, int r) {
    uint64_t key = ((uint64_t)m << 8) | (uint64_t)r;
    auto it = ctx->pso_argmax_p1_cache.find(key);
    if (it != ctx->pso_argmax_p1_cache.end()) return it->second;
    MTLFunctionConstantValues* fcv = [MTLFunctionConstantValues new];
    [fcv setConstantValue:&m type:MTLDataTypeInt atIndex:4];
    [fcv setConstantValue:&r type:MTLDataTypeInt atIndex:5];
    id<MTLComputePipelineState> pso = make_pso(ctx->device, ctx->lib, @"inkling_head_argmax_p1", fcv);
    ctx->pso_argmax_p1_cache[key] = pso;
    return pso;
}

static id<MTLComputePipelineState> get_mix_q4_pso(InklingCtx* ctx, int k) {
    uint64_t key = (uint64_t)(k & 0xFF);
    auto it = ctx->pso_mix_q4_cache.find(key);
    if (it != ctx->pso_mix_q4_cache.end()) return it->second;
    MTLFunctionConstantValues* fcv = [MTLFunctionConstantValues new];
    [fcv setConstantValue:&k type:MTLDataTypeInt atIndex:2];
    id<MTLComputePipelineState> pso = make_pso(ctx->device, ctx->lib, @"inkling_mix_q4", fcv);
    ctx->pso_mix_q4_cache[key] = pso;
    return pso;
}

static id<MTLComputePipelineState> get_mix_gate_p1_pso(InklingCtx* ctx, bool conv, int k) {
    uint64_t key = ((uint64_t)(conv ? 1 : 0) << 8) | (uint64_t)(k & 0xFF);
    auto it = ctx->pso_mix_gate_p1_cache.find(key);
    if (it != ctx->pso_mix_gate_p1_cache.end()) return it->second;
    MTLFunctionConstantValues* fcv = [MTLFunctionConstantValues new];
    [fcv setConstantValue:&conv type:MTLDataTypeBool atIndex:1];
    [fcv setConstantValue:&k    type:MTLDataTypeInt  atIndex:2];
    id<MTLComputePipelineState> pso = make_pso(ctx->device, ctx->lib, @"inkling_mix_gate_p1", fcv);
    ctx->pso_mix_gate_p1_cache[key] = pso;
    return pso;
}

static id<MTLComputePipelineState> get_kv_prep_batched_pso(InklingCtx* ctx, bool dyn, int hd) {
    uint64_t key = ((uint64_t)(dyn ? 1 : 0) << 16) | (uint64_t)hd;
    auto it = ctx->pso_kv_prep_batched_cache.find(key);
    if (it != ctx->pso_kv_prep_batched_cache.end()) return it->second;
    MTLFunctionConstantValues* fcv = [MTLFunctionConstantValues new];
    [fcv setConstantValue:&dyn type:MTLDataTypeBool atIndex:3];
    [fcv setConstantValue:&hd  type:MTLDataTypeInt  atIndex:8];
    id<MTLComputePipelineState> pso = make_pso(ctx->device, ctx->lib, @"inkling_kv_prep_batched", fcv);
    ctx->pso_kv_prep_batched_cache[key] = pso;
    return pso;
}

static id<MTLComputePipelineState> get_attend_batched_pso(InklingCtx* ctx, int g, int hd) {
    uint64_t key = ((uint64_t)g << 16) | (uint64_t)hd;
    auto it = ctx->pso_attend_batched_cache.find(key);
    if (it != ctx->pso_attend_batched_cache.end()) return it->second;
    MTLFunctionConstantValues* fcv = [MTLFunctionConstantValues new];
    [fcv setConstantValue:&g  type:MTLDataTypeInt atIndex:7];
    [fcv setConstantValue:&hd type:MTLDataTypeInt atIndex:8];
    id<MTLComputePipelineState> pso = make_pso(ctx->device, ctx->lib, @"inkling_attend_batched", fcv);
    ctx->pso_attend_batched_cache[key] = pso;
    return pso;
}

static id<MTLComputePipelineState> get_dpart_batched_pso(InklingCtx* ctx, int g, int hd) {
    uint64_t key = ((uint64_t)g << 16) | (uint64_t)hd;
    auto it = ctx->pso_dpart_batched_cache.find(key);
    if (it != ctx->pso_dpart_batched_cache.end()) return it->second;
    MTLFunctionConstantValues* fcv = [MTLFunctionConstantValues new];
    [fcv setConstantValue:&g  type:MTLDataTypeInt atIndex:7];
    [fcv setConstantValue:&hd type:MTLDataTypeInt atIndex:8];
    id<MTLComputePipelineState> pso = make_pso(ctx->device, ctx->lib, @"inkling_decode_partials_batched", fcv);
    ctx->pso_dpart_batched_cache[key] = pso;
    return pso;
}

static id<MTLComputePipelineState> get_mix_mtp_batched_pso(InklingCtx* ctx, bool gate, int k) {
    uint64_t key = ((uint64_t)(gate ? 1 : 0) << 16) | (uint64_t)(k & 0xFF);
    auto it = ctx->pso_mix_mtp_batched_cache.find(key);
    if (it != ctx->pso_mix_mtp_batched_cache.end()) return it->second;
    MTLFunctionConstantValues* fcv = [MTLFunctionConstantValues new];
    [fcv setConstantValue:&gate type:MTLDataTypeBool atIndex:0];
    [fcv setConstantValue:&k    type:MTLDataTypeInt  atIndex:2];
    id<MTLComputePipelineState> pso = make_pso(ctx->device, ctx->lib, @"inkling_mix_mtp_batched", fcv);
    ctx->pso_mix_mtp_batched_cache[key] = pso;
    return pso;
}

static id<MTLComputePipelineState> get_kv_prep_mtp_batched_pso(InklingCtx* ctx, bool dyn, int hd) {
    uint64_t key = ((uint64_t)(dyn ? 1 : 0) << 16) | (uint64_t)hd;
    auto it = ctx->pso_kv_prep_mtp_batched_cache.find(key);
    if (it != ctx->pso_kv_prep_mtp_batched_cache.end()) return it->second;
    MTLFunctionConstantValues* fcv = [MTLFunctionConstantValues new];
    [fcv setConstantValue:&dyn type:MTLDataTypeBool atIndex:3];
    [fcv setConstantValue:&hd  type:MTLDataTypeInt  atIndex:8];
    id<MTLComputePipelineState> pso = make_pso(ctx->device, ctx->lib, @"inkling_kv_prep_mtp_batched", fcv);
    ctx->pso_kv_prep_mtp_batched_cache[key] = pso;
    return pso;
}

static id<MTLComputePipelineState> get_attend_mtp_batched_pso(InklingCtx* ctx, int g, int hd) {
    uint64_t key = ((uint64_t)g << 16) | (uint64_t)hd;
    auto it = ctx->pso_attend_mtp_batched_cache.find(key);
    if (it != ctx->pso_attend_mtp_batched_cache.end()) return it->second;
    MTLFunctionConstantValues* fcv = [MTLFunctionConstantValues new];
    [fcv setConstantValue:&g  type:MTLDataTypeInt atIndex:7];
    [fcv setConstantValue:&hd type:MTLDataTypeInt atIndex:8];
    id<MTLComputePipelineState> pso = make_pso(ctx->device, ctx->lib, @"inkling_attend_mtp_batched", fcv);
    ctx->pso_attend_mtp_batched_cache[key] = pso;
    return pso;
}

static id<MTLComputePipelineState> get_dpart_mtp_batched_pso(InklingCtx* ctx, int g, int hd) {
    uint64_t key = ((uint64_t)g << 16) | (uint64_t)hd;
    auto it = ctx->pso_dpart_mtp_batched_cache.find(key);
    if (it != ctx->pso_dpart_mtp_batched_cache.end()) return it->second;
    MTLFunctionConstantValues* fcv = [MTLFunctionConstantValues new];
    [fcv setConstantValue:&g  type:MTLDataTypeInt atIndex:7];
    [fcv setConstantValue:&hd type:MTLDataTypeInt atIndex:8];
    id<MTLComputePipelineState> pso = make_pso(ctx->device, ctx->lib, @"inkling_decode_partials_mtp_batched", fcv);
    ctx->pso_dpart_mtp_batched_cache[key] = pso;
    return pso;
}

static id<MTLComputePipelineState> get_head_logits_max_mixed_p1_pso(InklingCtx* ctx, int m) {
    if (m < 1 || m > 8) return nil;
    if (ctx->pso_head_logits_max_mixed_p1[m]) return ctx->pso_head_logits_max_mixed_p1[m];
    MTLFunctionConstantValues* fcv = [MTLFunctionConstantValues new];
    [fcv setConstantValue:&m type:MTLDataTypeInt atIndex:4];
    id<MTLComputePipelineState> pso = make_pso(ctx->device, ctx->lib, @"inkling_head_logits_max_mixed_p1", fcv);
    ctx->pso_head_logits_max_mixed_p1[m] = pso;
    return pso;
}

static id<MTLComputePipelineState> get_grp_moe_tiles_pso(InklingCtx* ctx, bool is_nvfp4, bool is_w2, int sg, int mt) {
    uint64_t key = ((uint64_t)(is_nvfp4 ? 0 : 1) << 24)
                 | ((uint64_t)(is_w2 ? 1 : 0) << 16)
                 | ((uint64_t)mt << 8)
                 | (uint64_t)sg;
    auto it = ctx->pso_grp_moe_tiles_cache.find(key);
    if (it != ctx->pso_grp_moe_tiles_cache.end()) return it->second;
    MTLFunctionConstantValues* fcv = [MTLFunctionConstantValues new];
    [fcv setConstantValue:&sg type:MTLDataTypeInt atIndex:6];
    [fcv setConstantValue:&mt type:MTLDataTypeInt atIndex:11];
    NSString* name = nil;
    if (is_nvfp4) {
        name = is_w2 ? @"inkling_grp_w2_nvfp4_tiles" : @"inkling_grp_w13_nvfp4_tiles";
    } else {
        name = is_w2 ? @"inkling_grp_w2_mxfp8_tiles" : @"inkling_grp_w13_mxfp8_tiles";
    }
    id<MTLComputePipelineState> pso = make_pso(ctx->device, ctx->lib, name, fcv);
    ctx->pso_grp_moe_tiles_cache[key] = pso;
    return pso;
}

static int ensure_part_buf(InklingCtx* ctx, size_t sz_max, size_t sz_sum, size_t sz_acc) {
    size_t off_max = 0;
    size_t off_sum = align_up(sz_max, 256);
    size_t off_acc = align_up(off_sum + sz_sum, 256);
    size_t need = align_up(off_acc + sz_acc, ctx->page_size);
    if (need > ctx->part_buf_bytes) {
        id<MTLBuffer> nb = [ctx->device newBufferWithLength:need
                                                     options:MTLResourceStorageModeShared];
        if (!nb) {
            fprintf(stderr, "[inkling_decode] ERROR: failed to grow split-K part_buf to %zu B\n", need);
            return -1;
        }
        ctx->part_buf = nb;
        ctx->part_buf_bytes = need;
    }
    ctx->part_off_max = off_max;
    ctx->part_off_sum = off_sum;
    ctx->part_off_acc = off_acc;
    return 0;
}

extern "C" {

InklingCtx* inkling_ctx_create(const InklingModelConfig* cfg) {
    @autoreleasepool {
        if (!cfg || cfg->n_layers <= 0 || cfg->hidden_size <= 0 || cfg->max_m < 1 || cfg->max_m > 64) {
            fprintf(stderr, "[inkling_decode] ERROR: invalid InklingModelConfig\n");
            return nullptr;
        }
        auto* ctx = new InklingCtx();
        ctx->cfg = *cfg;
        ctx->page_size = (size_t)getpagesize();
        mach_timebase_info(&ctx->tb);

        ctx->device = MTLCreateSystemDefaultDevice();
        if (!ctx->device) {
            fprintf(stderr, "[inkling_decode] ERROR: no Metal device\n");
            delete ctx;
            return nullptr;
        }
        ctx->queue = [ctx->device newCommandQueue];
        if (!ctx->queue) {
            delete ctx;
            return nullptr;
        }

        int start = cfg->loop_start;
        int end = cfg->loop_end;
        int passes = cfg->loop_passes;
        for (int i = 0; i < start; ++i) {
            ctx->schedule.push_back({i, i, 0});
        }
        int visit = start;
        for (int p = 0; p < passes; ++p) {
            for (int i = start; i < end; ++i) {
                ctx->schedule.push_back({i, visit++, p});
            }
        }
        for (int i = end; i < cfg->n_layers; ++i) {
            ctx->schedule.push_back({i, visit++, 0});
        }
        ctx->layers.resize(cfg->n_layers);
        ctx->pending_rot.resize(ctx->schedule.size());
        ctx->pending_verify.resize(ctx->schedule.size());
        ctx->conv_hist_visit_off.resize(ctx->schedule.size(), 0);
        ctx->tmp_kv_k.resize(ctx->schedule.size());
        ctx->tmp_kv_v.resize(ctx->schedule.size());
        ctx->tmp_conv.resize(ctx->schedule.size() * 4);
        ctx->tmp_batch_plans.resize(ctx->schedule.size());

        const int D = cfg->hidden_size;
        const int RANK = std::max(1, cfg->loop_gate_rank);
        int mix_tg = (D >= 1024) ? 1024 : (D >= 256 ? 256 : 32);
        if (mix_tg % RANK != 0) {
            mix_tg = std::min(mix_tg, std::max(RANK, 32));
        }
        ctx->mix_tg = mix_tg;
        const int mix_dpt = (D + mix_tg - 1) / mix_tg;
        ctx->route_parts = std::max(1, std::min(8, 1024 / cfg->n_experts));
        ctx->bucket_tg = std::min(1024, std::max(128, ((cfg->n_experts + 31) / 32) * 32));

        int est_g = std::max(3, (cfg->max_n_heads + std::max(1, cfg->max_n_kv) - 1) / std::max(1, cfg->max_n_kv));
        if (est_g > 8) est_g = 8;
        ctx->max_g = est_g;
        ctx->max_nred = 8;
        ctx->max_blocks = ctx->max_nred - 1;

        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.preprocessorMacros = @{
            @"MODEL_D":           @(cfg->hidden_size),
            @"MODEL_INTER":       @(cfg->inter_size),
            @"MODEL_E":           @(cfg->n_experts),
            @"MODEL_TOP_K":       @(cfg->top_k),
            @"MODEL_DR":          @(cfg->d_rel),
            @"MODEL_RANK":        @(RANK),
            @"MODEL_V_LIMIT":     @(cfg->unpadded_vocab_size),
            @"MODEL_MIX_TG":      @(mix_tg),
            @"MODEL_MIX_DPT":     @(mix_dpt),
            @"MODEL_ROUTE_PARTS": @(ctx->route_parts),
            @"MODEL_BUCKET_TG":   @(ctx->bucket_tg),
            @"MAX_G":             @(ctx->max_g),
            @"MAX_NRED":          @(ctx->max_nred),
        };
        if (@available(macOS 15.0, *)) {
            opts.mathMode = MTLMathModeSafe;
        }

        NSError* err = nil;
        NSString* src = [NSString stringWithUTF8String:kMetalSource];
        ctx->lib = [ctx->device newLibraryWithSource:src options:opts error:&err];
        if (!ctx->lib) {
            fprintf(stderr, "[inkling_decode] ERROR: Metal library compile failed:\n%s\n",
                    err ? err.localizedDescription.UTF8String : "unknown");
            delete ctx;
            return nullptr;
        }

        auto make_int_fcv = [](NSUInteger idx, int val) {
            MTLFunctionConstantValues* fcv = [MTLFunctionConstantValues new];
            [fcv setConstantValue:&val type:MTLDataTypeInt atIndex:idx];
            return fcv;
        };
        auto make_bool_fcv = [](NSUInteger idx, bool val) {
            MTLFunctionConstantValues* fcv = [MTLFunctionConstantValues new];
            [fcv setConstantValue:&val type:MTLDataTypeBool atIndex:idx];
            return fcv;
        };

        ctx->pso_copy_bf16    = make_pso(ctx->device, ctx->lib, @"inkling_copy_bf16");
        ctx->pso_copy_f32     = make_pso(ctx->device, ctx->lib, @"inkling_copy_f32");
        ctx->pso_copy_conv4   = make_pso(ctx->device, ctx->lib, @"inkling_copy_conv4_last");
        ctx->pso_copy_cap     = make_pso(ctx->device, ctx->lib, @"inkling_copy_capture_bf16");
        ctx->pso_kv_commit    = make_pso(ctx->device, ctx->lib, @"inkling_kv_commit_rotating");
        ctx->pso_embed        = make_pso(ctx->device, ctx->lib, @"inkling_embed_rms_norm_2d");
        ctx->pso_embed_batched= make_pso(ctx->device, ctx->lib, @"inkling_embed_rms_norm_batched");
        ctx->pso_layer0_rms   = make_pso(ctx->device, ctx->lib, @"inkling_layer0_rms");
        ctx->pso_final_norm   = make_pso(ctx->device, ctx->lib, @"inkling_final_rms_norm");
        ctx->pso_final_norm_bf16 = make_pso(ctx->device, ctx->lib, @"inkling_final_rms_norm_bf16");
        ctx->pso_msg_down_batched = make_pso(ctx->device, ctx->lib, @"inkling_msg_down_batched");
        ctx->pso_msg_down_batched_bf16 = make_pso(ctx->device, ctx->lib, @"inkling_msg_down_batched_bf16");
        ctx->pso_msg_fg_blend = make_pso(ctx->device, ctx->lib, @"inkling_msg_fg_blend");
        ctx->pso_rms_multi    = make_pso(ctx->device, ctx->lib, @"inkling_rms_norm_multi");
        ctx->pso_msg_blend_norm = make_pso(ctx->device, ctx->lib, @"inkling_msg_blend_norm_bf16");
        ctx->pso_decode_comb  = make_pso(ctx->device, ctx->lib, @"inkling_decode_combine");
        ctx->pso_route        = make_pso(ctx->device, ctx->lib, @"inkling_route");
        ctx->pso_route_scores = make_pso(ctx->device, ctx->lib, @"inkling_route_scores");
        ctx->pso_route_topk   = make_pso(ctx->device, ctx->lib, @"inkling_route_topk");
        ctx->pso_bucket       = make_pso(ctx->device, ctx->lib, @"inkling_bucket");
        ctx->pso_bucket_tiles = make_pso(ctx->device, ctx->lib, @"inkling_bucket_tiles");
        ctx->pso_m1_w13_nvfp4 = make_pso(ctx->device, ctx->lib, @"inkling_m1_w13_nvfp4", make_int_fcv(6, 8));
        ctx->pso_m1_w13_mxfp8 = make_pso(ctx->device, ctx->lib, @"inkling_m1_w13_mxfp8", make_int_fcv(6, 4));
        ctx->pso_m1_w2_nvfp4  = make_pso(ctx->device, ctx->lib, @"inkling_m1_w2_nvfp4",  make_int_fcv(6, 8));
        ctx->pso_m1_w2_mxfp8  = make_pso(ctx->device, ctx->lib, @"inkling_m1_w2_mxfp8",  make_int_fcv(6, 8));
        ctx->pso_combine_noseq = make_pso(ctx->device, ctx->lib, @"inkling_combine", make_bool_fcv(9, false));
        ctx->pso_combine_seq   = make_pso(ctx->device, ctx->lib, @"inkling_combine", make_bool_fcv(9, true));
        ctx->pso_argmax_p2    = make_pso(ctx->device, ctx->lib, @"inkling_head_argmax_p2");

        if (!ctx->pso_copy_bf16 || !ctx->pso_copy_f32 || !ctx->pso_copy_conv4 || !ctx->pso_copy_cap ||
            !ctx->pso_kv_commit || !ctx->pso_embed || !ctx->pso_embed_batched || !ctx->pso_layer0_rms ||
            !ctx->pso_final_norm || !ctx->pso_final_norm_bf16 || !ctx->pso_msg_down_batched || !ctx->pso_msg_down_batched_bf16 ||
            !ctx->pso_msg_fg_blend || !ctx->pso_rms_multi || !ctx->pso_msg_blend_norm || !ctx->pso_decode_comb ||
            !ctx->pso_route || !ctx->pso_route_scores || !ctx->pso_route_topk ||
            !ctx->pso_bucket || !ctx->pso_bucket_tiles ||
            !ctx->pso_m1_w13_nvfp4 || !ctx->pso_m1_w13_mxfp8 || !ctx->pso_m1_w2_nvfp4 || !ctx->pso_m1_w2_mxfp8 ||
            !ctx->pso_combine_noseq || !ctx->pso_combine_seq || !ctx->pso_argmax_p2) {
            delete ctx;
            return nullptr;
        }

        if (cfg->max_m >= 8) {
            MTLCompileOptions* nax_opts = [MTLCompileOptions new];
            nax_opts.preprocessorMacros = @{
                @"MODEL_D":       @(cfg->hidden_size),
                @"MODEL_V_LIMIT": @(cfg->unpadded_vocab_size),
                @"MODEL_E":       @(cfg->n_experts),
                @"MODEL_RANK":    @(RANK),
            };
            nax_opts.languageVersion = MTLLanguageVersion4_0;
            if (@available(macOS 15.0, *)) {
                nax_opts.mathMode = MTLMathModeSafe;
            }
            NSError* nax_err = nil;
            NSString* nax_src = [NSString stringWithUTF8String:kMetal4NaxSource];
            ctx->lib_nax = [ctx->device newLibraryWithSource:nax_src options:nax_opts error:&nax_err];
            if (!ctx->lib_nax) {
                fprintf(stderr, "[inkling_decode] ERROR: Metal 4.0 NAX library compile failed:\n%s\n",
                        nax_err ? nax_err.localizedDescription.UTF8String : "unknown");
                delete ctx;
                return nullptr;
            }
            ctx->pso_nax_qkv_16x64 = make_pso(ctx->device, ctx->lib_nax, @"inkling_nax_qkv_bf16_16x64");
            ctx->pso_nax_qkv_32x64 = make_pso(ctx->device, ctx->lib_nax, @"inkling_nax_qkv_bf16_32x64");
            ctx->pso_nax_qkv_64x64 = make_pso(ctx->device, ctx->lib_nax, @"inkling_nax_qkv_bf16_64x64");
            ctx->pso_nax_qkv_64x32 = make_pso(ctx->device, ctx->lib_nax, @"inkling_nax_qkv_bf16_64x32");
            ctx->pso_nax_argmax_16x64 = make_pso(ctx->device, ctx->lib_nax, @"inkling_nax_head_argmax_p1_16x64");
            ctx->pso_nax_msg_fg_16x64 = make_pso(ctx->device, ctx->lib_nax, @"inkling_nax_msg_fg_16x64");
            ctx->pso_nax_msg_fg_32x64 = make_pso(ctx->device, ctx->lib_nax, @"inkling_nax_msg_fg_32x64");
            ctx->pso_nax_msg_fg_64x64 = make_pso(ctx->device, ctx->lib_nax, @"inkling_nax_msg_fg_64x64");
            ctx->pso_nax_router_16x32 = make_pso(ctx->device, ctx->lib_nax, @"inkling_nax_router_16x32");
            if (!ctx->pso_nax_qkv_16x64 || !ctx->pso_nax_qkv_32x64 || !ctx->pso_nax_qkv_64x64 ||
                !ctx->pso_nax_qkv_64x32 || !ctx->pso_nax_argmax_16x64 ||
                !ctx->pso_nax_msg_fg_16x64 || !ctx->pso_nax_msg_fg_32x64 || !ctx->pso_nax_msg_fg_64x64 ||
                !ctx->pso_nax_router_16x32) {
                delete ctx;
                return nullptr;
            }
        }

        get_qkv_pso(ctx, 1, 4, cfg->hidden_size);
        get_qkv_pso(ctx, 1, 8, cfg->hidden_size);
        get_argmax_p1_pso(ctx, 1, 4);
        for (int k = 0; k <= 4; ++k) {
            get_mix_pso(ctx, false, false, k);
            get_mix_pso(ctx, true,  false, k);
            get_mix_pso(ctx, false, true,  k);
            get_mix_pso(ctx, true,  true,  k);
            get_mix_q4_pso(ctx, k);
            if (cfg->max_m >= 8) {
                get_mix_gate_p1_pso(ctx, false, k);
                get_mix_gate_p1_pso(ctx, true,  k);
            }
        }
        if (cfg->max_m >= 3) {
            get_grp_moe_tiles_pso(ctx, true,  false, 4, 4);
            get_grp_moe_tiles_pso(ctx, true,  true,  4, 4);
            get_grp_moe_tiles_pso(ctx, false, false, 4, 4);
            get_grp_moe_tiles_pso(ctx, false, true,  4, 4);
        }
        if (cfg->max_m >= 9) {
            get_grp_moe_tiles_pso(ctx, true,  false, 2, 8);
            get_grp_moe_tiles_pso(ctx, true,  true,  2, 8);
            get_grp_moe_tiles_pso(ctx, false, false, 2, 8);
            get_grp_moe_tiles_pso(ctx, false, true,  2, 8);
        }

        if (getenv("INKLING_DEBUG_PSO")) {
            auto p_m0 = get_mix_pso(ctx, false, false, 4);
            auto p_m1 = get_mix_pso(ctx, true, true, 4);
            auto p_att = get_attend_pso(ctx, est_g, cfg->max_head_dim, false);
            fprintf(stderr, "[inkling_decode] PSO maxTG: embed=%lu l0=%lu fn=%lu mix0=%lu mix1=%lu attend=%lu route=%lu\n",
                    (unsigned long)ctx->pso_embed.maxTotalThreadsPerThreadgroup,
                    (unsigned long)ctx->pso_layer0_rms.maxTotalThreadsPerThreadgroup,
                    (unsigned long)ctx->pso_final_norm.maxTotalThreadsPerThreadgroup,
                    (unsigned long)p_m0.maxTotalThreadsPerThreadgroup,
                    (unsigned long)p_m1.maxTotalThreadsPerThreadgroup,
                    (unsigned long)p_att.maxTotalThreadsPerThreadgroup,
                    (unsigned long)ctx->pso_route.maxTotalThreadsPerThreadgroup);
        }

        const size_t Du = (size_t)cfg->hidden_size;
        const size_t C_max = (size_t)cfg->max_n_kv * (size_t)cfg->max_head_dim;
        const size_t zero_len = align_up(std::max<size_t>(65536, 3 * std::max(Du, C_max) * sizeof(float)), ctx->page_size);
        ctx->zero_buf = [ctx->device newBufferWithLength:zero_len options:MTLResourceStorageModeShared];
        memset([ctx->zero_buf contents], 0, zero_len);

        const size_t M = (size_t)cfg->max_m;
        const size_t verify_M = (size_t)std::min<int32_t>(cfg->max_m, 8);
        const size_t INTER = (size_t)cfg->inter_size;
        const size_t HQ = (size_t)cfg->max_n_heads;
        const size_t HKV = (size_t)cfg->max_n_kv;
        const size_t HD = (size_t)cfg->max_head_dim;
        const size_t R = (size_t)cfg->max_rel_extent;
        const size_t DR = (size_t)cfg->d_rel;
        const size_t E = (size_t)cfg->n_experts;
        const size_t K_TOP = (size_t)cfg->top_k;
        const size_t NV = ctx->schedule.size();
        const size_t MAX_QKV_ROWS = 2 * HQ * HD + HQ * DR + 2 * HKV * HD;
        const size_t MAX_NTG_SIMT = ((size_t)cfg->unpadded_vocab_size + 15) / 16;
        const size_t MAX_NTG_NAX  = ((size_t)cfg->unpadded_vocab_size + 63) / 64;
        const size_t MB = (size_t)std::min<int32_t>(ctx->max_blocks, cfg->n_layers + 1);

        size_t off = 0;
        auto alloc = [&](size_t bytes) -> size_t {
            size_t res = align_up(off, 256);
            off = res + std::max<size_t>(bytes, 256);
            return res;
        };

        ctx->off_x               = alloc(M * Du * 2);
        ctx->off_pre_loop_x      = alloc(M * Du * 2);
        ctx->off_bank_x          = alloc(M * Du * 2);
        ctx->off_blocks          = alloc(MB * M * Du * 4);
        ctx->off_bank_blocks     = alloc(MB * M * Du * 4);
        ctx->off_messages        = alloc(M * 4 * Du * 4);
        ctx->off_qkv_in          = alloc(M * 3 * Du * 2);
        ctx->off_qkv             = alloc(M * MAX_QKV_ROWS * 2);
        ctx->off_qo              = alloc(M * HQ * HD * 2);
        ctx->off_bias            = alloc(M * HQ * R * 2);
        ctx->off_sinks           = alloc(M * HQ * 2);
        ctx->off_att             = alloc(M * HQ * HD * 2);
        ctx->off_att_out         = alloc(M * Du * 2);
        ctx->off_mlp_in          = alloc(M * Du * 2);
        ctx->off_resid           = alloc(M * Du * 4);
        ctx->off_inds            = alloc(M * K_TOP * 4);
        ctx->off_weights         = alloc(M * K_TOP * 4);
        ctx->off_offsets         = alloc((E + 1) * 4);
        ctx->off_sorted_slots    = alloc(M * K_TOP * 4);
        ctx->off_active_list     = alloc(std::max<size_t>(E, M * K_TOP) * 4);
        ctx->off_moe_h           = alloc(M * K_TOP * INTER * 2);
        ctx->off_ys              = alloc(M * K_TOP * Du * 2);
        ctx->off_scratch_conv_a  = alloc(verify_M * 3 * std::max(Du, C_max) * 4);
        ctx->off_scratch_conv_b  = alloc(verify_M * 3 * std::max(Du, C_max) * 4);
        ctx->off_scratch_conv_c  = alloc(verify_M * 3 * Du * 4);
        ctx->off_scratch_conv_d  = alloc(verify_M * 3 * Du * 4);
        ctx->off_scratch_partial = alloc(M * Du * 4);

        ctx->verify_kv_visit_stride_bytes = align_up(verify_M * HKV * HD * 2, 256);
        ctx->off_verify_k        = alloc(NV * ctx->verify_kv_visit_stride_bytes);
        ctx->off_verify_v        = alloc(NV * ctx->verify_kv_visit_stride_bytes);

        ctx->off_pre_norm        = alloc(M * Du * 4);
        ctx->off_normed_f32      = alloc(M * Du * 4);
        ctx->off_normed_bf16     = alloc(M * Du * 2);
        ctx->off_captures        = alloc(verify_M * (size_t)ctx->max_captures * Du * 2);
        ctx->off_argmax_pv       = alloc(std::max(verify_M * MAX_NTG_SIMT, M * MAX_NTG_NAX) * 4);
        ctx->off_argmax_pi       = alloc(std::max(verify_M * MAX_NTG_SIMT, M * MAX_NTG_NAX) * 4);
        ctx->off_out_toks        = alloc(M * 4);
        ctx->off_msg_o           = alloc(M * 3 * Du * 4);
        ctx->off_msg_gate_in     = alloc(M * 3 * 2 * Du * 2);
        ctx->off_msg_low         = alloc(M * 3 * (size_t)RANK * 4);
        ctx->off_route_scores    = alloc(M * E * 4);
        ctx->off_tile_expert     = alloc(M * K_TOP * 4);
        ctx->off_tile_t0         = alloc(M * K_TOP * 4);
        ctx->off_indirect_args   = alloc(256);
        ctx->batch_visit_meta_stride_bytes = align_up(M * 4, 256);
        ctx->off_batch_counts    = alloc(NV * ctx->batch_visit_meta_stride_bytes);
        ctx->off_batch_tau       = alloc(NV * ctx->batch_visit_meta_stride_bytes);

        ctx->scratch_bytes = align_up(off, ctx->page_size);
        ctx->scratch = [ctx->device newBufferWithLength:ctx->scratch_bytes
                                                options:MTLResourceStorageModeShared];
        if (!ctx->scratch) {
            fprintf(stderr, "[inkling_decode] ERROR: failed to allocate %zu B scratch arena\n", ctx->scratch_bytes);
            delete ctx;
            return nullptr;
        }
        return ctx;
    }
}

void inkling_ctx_destroy(InklingCtx* ctx) {
    @autoreleasepool {
        delete ctx;
    }
}

int inkling_ctx_register_globals(InklingCtx* ctx, const InklingGlobalWeights* g) {
    @autoreleasepool {
        if (!ctx || !g) return -1;
        ctx->globals.embed_tokens     = wrap_static_ptr(ctx, g->embed_tokens, g->embed_tokens_bytes);
        ctx->globals.embed_norm       = wrap_static_ptr(ctx, g->embed_norm, g->embed_norm_bytes);
        ctx->globals.message_queries  = wrap_static_ptr(ctx, g->message_queries, g->message_queries_bytes);
        ctx->globals.final_query      = wrap_static_ptr(ctx, g->final_query, g->final_query_bytes);
        ctx->globals.final_norm       = wrap_static_ptr(ctx, g->final_norm, g->final_norm_bytes);
        ctx->globals.lm_head          = wrap_static_ptr(ctx, g->lm_head, g->lm_head_bytes);
        ctx->globals.embed_tokens_ptr = g->embed_tokens;
        ctx->globals.embed_norm_ptr   = g->embed_norm;
        ctx->globals.lm_head_ptr      = g->lm_head;
        ctx->globals.embed_norm_eps   = g->embed_norm_eps;
        ctx->globals.final_norm_eps   = g->final_norm_eps;
        if (!ctx->globals.embed_tokens.buf || !ctx->globals.embed_norm.buf ||
            !ctx->globals.message_queries.buf || !ctx->globals.final_query.buf ||
            !ctx->globals.final_norm.buf) {
            return -2;
        }
        ctx->globals.registered = true;
        return 0;
    }
}

int inkling_ctx_register_layer(InklingCtx* ctx, const InklingLayerWeights* lw) {
    @autoreleasepool {
        if (!ctx || !lw || lw->layer_idx < 0 || lw->layer_idx >= ctx->cfg.n_layers) return -1;
        if (lw->n_heads / lw->n_kv > ctx->max_g) {
            fprintf(stderr, "[inkling_decode] ERROR: layer %d G=%d exceeds MAX_G=%d\n",
                    lw->layer_idx, lw->n_heads / lw->n_kv, ctx->max_g);
            return -2;
        }
        LayerDeviceWeights& dst = ctx->layers[lw->layer_idx];
        dst.is_sliding        = lw->is_sliding;
        dst.is_block_boundary = lw->is_block_boundary;
        dst.has_attn_input    = lw->has_attn_input;
        dst.has_message_gates = lw->has_message_gates;
        dst.n_heads           = lw->n_heads;
        dst.n_kv              = lw->n_kv;
        dst.head_dim          = lw->head_dim;
        dst.sliding_window    = lw->sliding_window;
        dst.rel_extent        = lw->rel_extent;
        for (int i = 0; i < 5; ++i) dst.qkv_at[i] = lw->qkv_at[i];
        dst.qkv_total_rows    = lw->qkv_total_rows;
        dst.moe_mode          = lw->moe_mode;
        dst.attn_eps          = lw->attn_eps;
        dst.mlp_eps           = lw->mlp_eps;

        dst.stack            = wrap_static_ptr(ctx, lw->stack, lw->stack_bytes);
        dst.o_proj           = wrap_static_ptr(ctx, lw->o_proj, lw->o_proj_bytes);
        dst.attn_query       = wrap_static_ptr(ctx, lw->attn_query, lw->attn_query_bytes);
        dst.attn_norm        = wrap_static_ptr(ctx, lw->attn_norm, lw->attn_norm_bytes);
        dst.mlp_query        = wrap_static_ptr(ctx, lw->mlp_query, lw->mlp_query_bytes);
        dst.mlp_norm         = wrap_static_ptr(ctx, lw->mlp_norm, lw->mlp_norm_bytes);
        dst.attn_taps        = wrap_static_ptr(ctx, lw->attn_taps, lw->attn_taps_bytes);
        dst.mlp_taps         = wrap_static_ptr(ctx, lw->mlp_taps, lw->mlp_taps_bytes);
        dst.k_taps           = wrap_static_ptr(ctx, lw->k_taps, lw->k_taps_bytes);
        dst.v_taps           = wrap_static_ptr(ctx, lw->v_taps, lw->v_taps_bytes);
        dst.kdyn_w           = wrap_static_ptr(ctx, lw->kdyn_w, lw->kdyn_w_bytes);
        dst.vdyn_w           = wrap_static_ptr(ctx, lw->vdyn_w, lw->vdyn_w_bytes);
        dst.attn_gate_down   = wrap_static_ptr(ctx, lw->attn_gate_down, lw->attn_gate_down_bytes);
        dst.attn_gate_forget = wrap_static_ptr(ctx, lw->attn_gate_forget, lw->attn_gate_forget_bytes);
        dst.attn_gate_gain   = wrap_static_ptr(ctx, lw->attn_gate_gain, lw->attn_gate_gain_bytes);
        dst.mlp_gate_down    = wrap_static_ptr(ctx, lw->mlp_gate_down, lw->mlp_gate_down_bytes);
        dst.mlp_gate_forget  = wrap_static_ptr(ctx, lw->mlp_gate_forget, lw->mlp_gate_forget_bytes);
        dst.mlp_gate_gain    = wrap_static_ptr(ctx, lw->mlp_gate_gain, lw->mlp_gate_gain_bytes);
        dst.router_w         = wrap_static_ptr(ctx, lw->router_w, lw->router_w_bytes);
        dst.router_b         = wrap_static_ptr(ctx, lw->router_b, lw->router_b_bytes);
        dst.sink_w           = wrap_static_ptr(ctx, lw->sink_w, lw->sink_w_bytes);
        dst.q_scale          = wrap_static_ptr(ctx, lw->q_scale, lw->q_scale_bytes);
        dst.rel_scale        = wrap_static_ptr(ctx, lw->rel_scale, lw->rel_scale_bytes);
        dst.rel_proj         = wrap_static_ptr(ctx, lw->rel_proj, lw->rel_proj_bytes);
        dst.w13_weight       = wrap_static_ptr(ctx, lw->w13_weight, lw->w13_weight_bytes);
        dst.w13_scales       = wrap_static_ptr(ctx, lw->w13_scales, lw->w13_scales_bytes);
        dst.w13_gscale       = wrap_static_ptr(ctx, lw->w13_gscale, lw->w13_gscale_bytes);
        dst.w2_weight        = wrap_static_ptr(ctx, lw->w2_weight, lw->w2_weight_bytes);
        dst.w2_scales        = wrap_static_ptr(ctx, lw->w2_scales, lw->w2_scales_bytes);
        dst.w2_gscale        = wrap_static_ptr(ctx, lw->w2_gscale, lw->w2_gscale_bytes);
        dst.attn_gate_fg_bf16 = wrap_static_ptr(ctx, lw->attn_gate_fg_bf16, lw->attn_gate_fg_bf16_bytes);
        dst.mlp_gate_fg_bf16  = wrap_static_ptr(ctx, lw->mlp_gate_fg_bf16,  lw->mlp_gate_fg_bf16_bytes);
        dst.router_w_bf16     = wrap_static_ptr(ctx, lw->router_w_bf16,     lw->router_w_bf16_bytes);

        if (!dst.stack.buf || !dst.o_proj.buf || !dst.router_w.buf || !dst.w13_weight.buf || !dst.w2_weight.buf ||
            !dst.attn_gate_fg_bf16.buf || !dst.mlp_gate_fg_bf16.buf || !dst.router_w_bf16.buf) {
            return -3;
        }
        const size_t rank = (size_t)std::max(1, ctx->cfg.loop_gate_rank);
        const size_t Du = (size_t)ctx->cfg.hidden_size;
        const size_t Eu = (size_t)ctx->cfg.n_experts;
        dst.has_attn_gate_fg_bf16 = (lw->attn_gate_fg_bf16 != nullptr &&
                                     lw->attn_gate_fg_bf16_bytes >= 3 * rank * 2 * Du * sizeof(uint16_t));
        dst.has_mlp_gate_fg_bf16  = (lw->mlp_gate_fg_bf16 != nullptr &&
                                     lw->mlp_gate_fg_bf16_bytes >= rank * 2 * Du * sizeof(uint16_t));
        dst.has_router_w_bf16     = (lw->router_w_bf16 != nullptr &&
                                     lw->router_w_bf16_bytes >= Eu * Du * sizeof(uint16_t));
        int g = lw->n_heads / lw->n_kv;
        get_qkv_pso(ctx, 1, (ctx->cfg.hidden_size >= 4096) ? 8 : 4, lw->n_heads * lw->head_dim);
        get_kv_prep_pso(ctx, !lw->is_sliding, lw->head_dim);
        get_attend_pso(ctx, g, lw->head_dim, false);
        get_attend_pso(ctx, g, lw->head_dim, true);
        get_dpart_pso(ctx, g, lw->head_dim);
        get_kv_prep_batched_pso(ctx, !lw->is_sliding, lw->head_dim);
        get_attend_batched_pso(ctx, g, lw->head_dim);
        get_dpart_batched_pso(ctx, g, lw->head_dim);

        dst.registered = true;
        return 0;
    }
}

int inkling_ctx_num_visits(const InklingCtx* ctx) {
    return ctx ? (int)ctx->schedule.size() : 0;
}

size_t inkling_ctx_visit_conv_offset(const InklingCtx* ctx, int32_t visit_idx, int32_t m_tokens) {
    if (!ctx || visit_idx < 0 || visit_idx > (int32_t)ctx->schedule.size()) return 0;
    size_t total = 0;
    const size_t D = (size_t)ctx->cfg.hidden_size;
    for (int32_t v = 0; v < visit_idx; ++v) {
        const auto& L = ctx->layers[ctx->schedule[v].layer_idx];
        const size_t C = (size_t)L.n_kv * (size_t)L.head_dim;
        total += (size_t)m_tokens * 3 * (2 * C + 2 * D);
    }
    return total;
}

size_t inkling_ctx_conv_states_floats(const InklingCtx* ctx, int32_t m_tokens) {
    if (!ctx) return 0;
    return inkling_ctx_visit_conv_offset(ctx, (int32_t)ctx->schedule.size(), m_tokens);
}

size_t inkling_ctx_scratch_bytes(const InklingCtx* ctx) {
    return ctx ? ctx->scratch_bytes : 0;
}

void inkling_ctx_clear_ptr_cache(InklingCtx* ctx) {
    @autoreleasepool {
        if (ctx) ctx->ptr_cache.clear();
    }
}

static inline int ensure_conv_hist_buf(InklingCtx* ctx, int32_t verify_max_m) {
    if (ctx->conv_hist_buf) return 0;
    const int32_t NV = (int32_t)ctx->schedule.size();
    const size_t D = (size_t)ctx->cfg.hidden_size;
    size_t cur = 0;
    const size_t max_m = (size_t)verify_max_m;
    for (int v = 0; v < NV; ++v) {
        const auto& L = ctx->layers[ctx->schedule[v].layer_idx];
        const size_t ch = (size_t)L.n_kv * (size_t)L.head_dim;
        ctx->conv_hist_visit_off[v] = cur;
        size_t bytes = max_m * 3 * (2 * ch + 2 * D) * sizeof(float);
        cur = align_up(cur + bytes, 256);
    }
    ctx->conv_hist_bytes = align_up(std::max<size_t>(cur, ctx->page_size), ctx->page_size);
    ctx->conv_hist_buf = [ctx->device newBufferWithLength:ctx->conv_hist_bytes
                                                  options:MTLResourceStorageModeShared];
    if (!ctx->conv_hist_buf) {
        fprintf(stderr, "[inkling_decode] ERROR: failed to allocate %zu B conv_hist_buf\n",
                ctx->conv_hist_bytes);
        return -8;
    }
    return 0;
}

static inline int prepare_target_kv_conv_part(
    InklingCtx*          ctx,
    int32_t              M,
    void* const*         kv_ptrs,
    void* const*         conv_ptrs,
    const InklingKVMeta* kv_meta)
{
    const int32_t NV = (int32_t)ctx->schedule.size();
    const int32_t D = ctx->cfg.hidden_size;
    auto& kv_k_bufs = ctx->tmp_kv_k;
    auto& kv_v_bufs = ctx->tmp_kv_v;
    auto& conv_bufs = ctx->tmp_conv;
    size_t max_part_meta = 0, max_part_acc = 0;
    for (int v = 0; v < NV; ++v) {
        const auto& L = ctx->layers[ctx->schedule[v].layer_idx];
        const InklingKVMeta& km = kv_meta[v];
        size_t kv_elems = (size_t)L.n_kv * (size_t)km.stride_h;
        size_t kv_bytes = kv_elems * sizeof(uint16_t);
        kv_k_bufs[v] = wrap_cached_ptr(ctx, kv_ptrs[v * 2 + 0], kv_bytes);
        kv_v_bufs[v] = wrap_cached_ptr(ctx, kv_ptrs[v * 2 + 1], kv_bytes);
        if (!kv_k_bufs[v].buf || !kv_v_bufs[v].buf) return -6;

        size_t ch = (size_t)L.n_kv * (size_t)L.head_dim;
        size_t c_bytes[4] = {
            3 * ch * sizeof(float),
            3 * ch * sizeof(float),
            3 * (size_t)D * sizeof(float),
            3 * (size_t)D * sizeof(float),
        };
        for (int c = 0; c < 4; ++c) {
            conv_bufs[v * 4 + c] = wrap_cached_ptr(ctx, conv_ptrs[v * 4 + c], c_bytes[c]);
            if (!conv_bufs[v * 4 + c].buf) return -7;
        }

        const int prev = km.offset;
        const bool is_rot = (km.is_rotating != 0);
        const bool need_staging = (is_rot && M > 1 && (prev + M > km.max_size));
        const int s_total = need_staging
                          ? (std::min(prev, km.max_size) + M)
                          : (is_rot ? std::min(prev + M, km.max_size) : (prev + M));
        const int max_valid = (L.sliding_window > 0) ? std::min(s_total, L.sliding_window) : s_total;
        if (max_valid > window_tg(L.head_dim)) {
            int bs = (s_total < 16384) ? 256 : 512;
            int nblocks = (s_total + bs - 1) / bs;
            while (nblocks > ctx->max_nblocks) {
                bs *= 2;
                nblocks = (s_total + bs - 1) / bs;
            }
            max_part_meta = std::max(max_part_meta, (size_t)nblocks * (size_t)L.n_heads);
            max_part_acc  = std::max(max_part_acc,  (size_t)nblocks * (size_t)L.n_heads * (size_t)L.head_dim);
        }
    }
    if (max_part_meta > 0) {
        size_t sz_meta = align_up((size_t)M * max_part_meta * sizeof(float), 256);
        size_t sz_acc  = align_up((size_t)M * max_part_acc  * sizeof(float), 256);
        if (ensure_part_buf(ctx, sz_meta, sz_meta, sz_acc) != 0) return -9;
    }
    return 0;
}

static inline void encode_target_visits_and_norm(
    InklingCtx*                  ctx,
    id<MTLComputeCommandEncoder> enc,
    int32_t                      M,
    int32_t                      SEQ,
    InklingKVMeta*               kv_meta,
    const int32_t*               capture_layer_ids,
    int32_t                      n_captures,
    bool                         enable_captures,
    bool                         force_final_norm,
    bool                         has_out_conv,
    BufSlice                     out_conv_slice,
    bool                         use_internal_conv_hist,
    int32_t                      commit_rotating,
    id<MTLComputePipelineState>  pso_grp_w13_nv,
    id<MTLComputePipelineState>  pso_grp_w2_nv,
    id<MTLComputePipelineState>  pso_grp_w13_mx,
    id<MTLComputePipelineState>  pso_grp_w2_mx,
    int32_t&                     n_dispatches)
{
    const int32_t NV = (int32_t)ctx->schedule.size();
    const int32_t D = ctx->cfg.hidden_size;
    const int32_t INTER = ctx->cfg.inter_size;
    const int32_t E = ctx->cfg.n_experts;
    const int32_t K_TOP = ctx->cfg.top_k;
    auto& kv_k_bufs = ctx->tmp_kv_k;
    auto& kv_v_bufs = ctx->tmp_kv_v;
    auto& conv_bufs = ctx->tmp_conv;

    auto dispatch_copy_bf16 = [&](id<MTLBuffer> sb, size_t so, id<MTLBuffer> db, size_t doff, uint32_t count) {
        if (count == 0) return;
        CopyPrm prm{count};
        [enc setComputePipelineState:ctx->pso_copy_bf16];
        [enc setBuffer:sb offset:so atIndex:0];
        [enc setBuffer:db offset:doff atIndex:1];
        [enc setBytes:&prm length:sizeof(prm) atIndex:2];
        uint32_t tg = 256;
        [enc dispatchThreadgroups:MTLSizeMake((count + tg - 1) / tg, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        ++n_dispatches;
    };

    auto dispatch_copy_f32 = [&](id<MTLBuffer> sb, size_t so, id<MTLBuffer> db, size_t doff, uint32_t count) {
        if (count == 0) return;
        CopyPrm prm{count};
        [enc setComputePipelineState:ctx->pso_copy_f32];
        [enc setBuffer:sb offset:so atIndex:0];
        [enc setBuffer:db offset:doff atIndex:1];
        [enc setBytes:&prm length:sizeof(prm) atIndex:2];
        uint32_t tg = 256;
        [enc dispatchThreadgroups:MTLSizeMake((count + tg - 1) / tg, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        ++n_dispatches;
    };

    auto encode_mix = [&](
        id<MTLBuffer> x_b, size_t x_o,
        id<MTLBuffer> blk_b, size_t blk_o,
        id<MTLBuffer> qry_b, size_t qry_o,
        id<MTLBuffer> st_b, size_t st_o,
        id<MTLBuffer> tap_b, size_t tap_o,
        id<MTLBuffer> res_b, size_t res_o,
        id<MTLBuffer> msg_b, size_t msg_o,
        id<MTLBuffer> dn_b, size_t dn_o,
        id<MTLBuffer> fg_b, size_t fg_o,
        id<MTLBuffer> gn_b, size_t gn_o,
        id<MTLBuffer> nw_b, size_t nw_o,
        id<MTLBuffer> out_b, size_t out_o,
        id<MTLBuffer> part_b, size_t part_o,
        id<MTLBuffer> resid_b, size_t resid_o,
        id<MTLBuffer> nst_b, size_t nst_o,
        int K, int Q, int QO, int QN, int MOFF,
        bool conv, bool add, bool gate, bool norm, bool seq, float eps)
    {
        MixPrm prm{M, Q, QO, QN, MOFF, add ? 1 : 0, norm ? 1 : 0, seq ? 1 : 0, eps};
        [enc setComputePipelineState:get_mix_pso(ctx, gate, conv, K)];
        [enc setBuffer:x_b     offset:x_o     atIndex:0];
        [enc setBuffer:blk_b   offset:blk_o   atIndex:1];
        [enc setBuffer:qry_b   offset:qry_o   atIndex:2];
        [enc setBuffer:st_b    offset:st_o    atIndex:3];
        [enc setBuffer:tap_b   offset:tap_o   atIndex:4];
        [enc setBuffer:res_b   offset:res_o   atIndex:5];
        [enc setBuffer:msg_b   offset:msg_o   atIndex:6];
        [enc setBuffer:dn_b    offset:dn_o    atIndex:7];
        [enc setBuffer:fg_b    offset:fg_o    atIndex:8];
        [enc setBuffer:gn_b    offset:gn_o    atIndex:9];
        [enc setBuffer:nw_b    offset:nw_o    atIndex:10];
        [enc setBuffer:out_b   offset:out_o   atIndex:11];
        [enc setBuffer:part_b  offset:part_o  atIndex:12];
        [enc setBuffer:resid_b offset:resid_o atIndex:13];
        [enc setBuffer:nst_b   offset:nst_o   atIndex:14];
        [enc setBytes:&prm length:sizeof(prm) atIndex:15];
        if (gate) {
            size_t tg_len = align_up((size_t)D * sizeof(float), 16);
            [enc setThreadgroupMemoryLength:tg_len atIndex:0];
            [enc setThreadgroupMemoryLength:tg_len atIndex:1];
        } else {
            [enc setThreadgroupMemoryLength:16 atIndex:0];
            [enc setThreadgroupMemoryLength:16 atIndex:1];
        }
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)QO, (NSUInteger)M, 1)
            threadsPerThreadgroup:MTLSizeMake((NSUInteger)ctx->mix_tg, 1, 1)];
        ++n_dispatches;
    };

    auto encode_qkv_wide = [&](
        id<MTLBuffer> x_b, size_t x_o,
        id<MTLBuffer> w_b, size_t w_o,
        id<MTLBuffer> out_b, size_t out_o,
        int S, int K, int N, int AT3, int AT4)
    {
        const int sg = (N >= 4096) ? 8 : 4;
        const int ny = (N + sg - 1) / sg;
        QkvPrm prm{S, N, AT3, AT4};
        [enc setComputePipelineState:get_qkv_pso(ctx, M, sg, K)];
        [enc setBuffer:x_b   offset:x_o   atIndex:0];
        [enc setBuffer:w_b   offset:w_o   atIndex:1];
        [enc setBuffer:out_b offset:out_o atIndex:2];
        [enc setBytes:&prm length:sizeof(prm) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)ny, 1)
            threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)sg, 1)];
        ++n_dispatches;
    };

    int cur_k = 0;
    int pre_loop_k = 0;
    int bank_k = 0;
    const size_t block_stride_bytes = (size_t)M * (size_t)D * sizeof(float);
    size_t conv_out_float_cursor = 0;

    for (int step = 0; step < NV; ++step) {
        const VisitEntry& ve = ctx->schedule[step];
        const int layer_idx = ve.layer_idx;
        const int pass_idx = ve.pass_idx;
        const LayerDeviceWeights& L = ctx->layers[layer_idx];
        const int ch = L.n_kv * L.head_dim;

        ctx->pending_rot[step].active = false;

        if (layer_idx == ctx->cfg.loop_start) {
            if (pass_idx == 0) {
                dispatch_copy_bf16(ctx->scratch, ctx->off_x, ctx->scratch, ctx->off_pre_loop_x, (uint32_t)(M * D));
                pre_loop_k = cur_k;
            } else {
                int done = pre_loop_k;
                bank_k = std::max(0, cur_k - (done + 1));
                if (bank_k > 0) {
                    dispatch_copy_f32(
                        ctx->scratch, ctx->off_blocks + (size_t)(done + 1) * block_stride_bytes,
                        ctx->scratch, ctx->off_bank_blocks,
                        (uint32_t)((size_t)bank_k * (size_t)M * (size_t)D));
                }
                dispatch_copy_bf16(ctx->scratch, ctx->off_x, ctx->scratch, ctx->off_bank_x, (uint32_t)(M * D));
                dispatch_copy_bf16(ctx->scratch, ctx->off_pre_loop_x, ctx->scratch, ctx->off_x, (uint32_t)(M * D));
                cur_k = pre_loop_k;
            }
        }

        if (enable_captures && n_captures > 0 && capture_layer_ids) {
            for (int ci = 0; ci < n_captures; ++ci) {
                if (capture_layer_ids[ci] == step) {
                    CopyCapPrm cprm{M, ci, n_captures};
                    [enc setComputePipelineState:ctx->pso_copy_cap];
                    [enc setBuffer:ctx->scratch offset:ctx->off_x        atIndex:0];
                    [enc setBuffer:ctx->scratch offset:ctx->off_captures atIndex:1];
                    [enc setBytes:&cprm length:sizeof(cprm) atIndex:2];
                    uint32_t tg = 256;
                    [enc dispatchThreadgroups:MTLSizeMake(((uint32_t)D + tg - 1) / tg, (NSUInteger)M, 1)
                        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                    ++n_dispatches;
                }
            }
        }

        const bool gated = (pass_idx > 0);
        if (gated) {
            size_t mq_off = ctx->globals.message_queries.offset
                          + (size_t)(layer_idx - ctx->cfg.loop_start) * 4 * (size_t)D * sizeof(float);
            encode_mix(
                ctx->scratch, ctx->off_bank_x,
                ctx->scratch, ctx->off_bank_blocks,
                ctx->globals.message_queries.buf, mq_off,
                ctx->zero_buf, 0,
                ctx->zero_buf, 0,
                ctx->zero_buf, 0,
                ctx->zero_buf, 0,
                ctx->zero_buf, 0,
                ctx->zero_buf, 0,
                ctx->zero_buf, 0,
                ctx->zero_buf, 0,
                ctx->scratch, ctx->off_messages,
                ctx->scratch, ctx->off_scratch_partial,
                ctx->scratch, ctx->off_resid,
                ctx->scratch, ctx->off_scratch_conv_a,
                bank_k, 4, 4, 1, 0,
                false, false, false, false, false, 1e-6f);
        }

        size_t normed_dst_off = L.is_block_boundary
                              ? (ctx->off_blocks + (size_t)cur_k * block_stride_bytes)
                              : ctx->off_scratch_partial;
        int qkv_s = 1;
        if (!L.has_attn_input) {
            Layer0Prm l0{L.attn_eps};
            [enc setComputePipelineState:ctx->pso_layer0_rms];
            [enc setBuffer:ctx->scratch      offset:ctx->off_x          atIndex:0];
            [enc setBuffer:L.attn_norm.buf   offset:L.attn_norm.offset  atIndex:1];
            [enc setBuffer:ctx->scratch      offset:normed_dst_off      atIndex:2];
            [enc setBuffer:ctx->scratch      offset:ctx->off_qkv_in     atIndex:3];
            [enc setBytes:&l0 length:sizeof(l0) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)M, 1)
                threadsPerThreadgroup:MTLSizeMake((NSUInteger)ctx->mix_tg, 1, 1)];
            ++n_dispatches;
            qkv_s = 1;
        } else {
            int Q = L.is_sliding ? 1 : 3;
            int QO = gated ? 3 : Q;
            int QN = L.is_sliding ? 1 : 3;
            encode_mix(
                ctx->scratch, ctx->off_x,
                ctx->scratch, ctx->off_blocks,
                L.attn_query.buf, L.attn_query.offset,
                ctx->zero_buf, 0,
                ctx->zero_buf, 0,
                ctx->zero_buf, 0,
                ctx->scratch, ctx->off_messages,
                L.attn_gate_down.buf, L.attn_gate_down.offset,
                L.attn_gate_forget.buf, L.attn_gate_forget.offset,
                L.attn_gate_gain.buf, L.attn_gate_gain.offset,
                L.attn_norm.buf, L.attn_norm.offset,
                ctx->scratch, ctx->off_qkv_in,
                ctx->scratch, normed_dst_off,
                ctx->scratch, ctx->off_resid,
                ctx->scratch, ctx->off_scratch_conv_a,
                cur_k, Q, QO, QN, 0,
                false, false, gated, true, false, L.attn_eps);
            qkv_s = QO;
        }

        encode_qkv_wide(
            ctx->scratch, ctx->off_qkv_in,
            L.stack.buf, L.stack.offset,
            ctx->scratch, ctx->off_qkv,
            qkv_s, D, L.qkv_total_rows, L.qkv_at[3], L.qkv_at[4]);

        InklingKVMeta& km = kv_meta[step];
        const int prev = km.offset;
        const int old_idx = km.idx;
        const bool is_rot = (km.is_rotating != 0);
        const bool need_staging = (is_rot && M > 1 && (prev + M > km.max_size));

        int write_base = 0;
        int s_kv = 0;
        int ext_m = 0;
        int ring_idx = 0;
        int new_offset = prev + M;
        int new_idx = old_idx;

        if (!is_rot) {
            write_base = prev;
            s_kv = prev + M;
            ext_m = 0;
            ring_idx = 0;
            new_idx = prev + M;
        } else if (!need_staging) {
            if (prev + M <= km.max_size) {
                write_base = prev;
                s_kv = prev + M;
                ext_m = 0;
                ring_idx = 0;
                new_idx = prev + M;
            } else {
                write_base = old_idx % km.max_size;
                s_kv = km.max_size;
                ext_m = 0;
                ring_idx = (write_base + 1) % km.max_size;
                new_idx = write_base + 1;
            }
        } else {
            s_kv = std::min(prev, km.max_size);
            ext_m = M;
            ring_idx = (prev >= km.max_size) ? (old_idx % km.max_size) : 0;
            if (commit_rotating) {
                new_idx = ((old_idx + M - 1) % km.max_size) + 1;
            } else {
                new_offset = prev;
                new_idx = old_idx;
                ctx->pending_rot[step] = {true, old_idx, km.max_size, prev, M};
            }
        }
        ctx->pending_verify[step] = {prev, old_idx, km.max_size, M, is_rot, need_staging && !commit_rotating};

        const size_t visit_c_floats = (size_t)M * 3 * (size_t)ch;
        const size_t visit_d_floats = (size_t)M * 3 * (size_t)D;
        BufSlice ks_out_dst, vs_out_dst, cs_out_dst, ms_out_dst;
        bool need_copy_conv_back = (M > 1) || has_out_conv;
        if (has_out_conv) {
            size_t b_off = out_conv_slice.offset;
            ks_out_dst = {out_conv_slice.buf, b_off + conv_out_float_cursor * sizeof(float)};
            conv_out_float_cursor += visit_c_floats;
            vs_out_dst = {out_conv_slice.buf, b_off + conv_out_float_cursor * sizeof(float)};
            conv_out_float_cursor += visit_c_floats;
            cs_out_dst = {out_conv_slice.buf, b_off + conv_out_float_cursor * sizeof(float)};
            conv_out_float_cursor += visit_d_floats;
            ms_out_dst = {out_conv_slice.buf, b_off + conv_out_float_cursor * sizeof(float)};
            conv_out_float_cursor += visit_d_floats;
        } else if (use_internal_conv_hist) {
            size_t cur = ctx->conv_hist_visit_off[step];
            ks_out_dst = {ctx->conv_hist_buf, cur};
            cur += visit_c_floats * sizeof(float);
            vs_out_dst = {ctx->conv_hist_buf, cur};
            cur += visit_c_floats * sizeof(float);
            cs_out_dst = {ctx->conv_hist_buf, cur};
            cur += visit_d_floats * sizeof(float);
            ms_out_dst = {ctx->conv_hist_buf, cur};
            need_copy_conv_back = false;
        } else if (M == 1) {
            ks_out_dst = conv_bufs[step * 4 + 0];
            vs_out_dst = conv_bufs[step * 4 + 1];
            cs_out_dst = conv_bufs[step * 4 + 2];
            ms_out_dst = conv_bufs[step * 4 + 3];
        } else {
            ks_out_dst = {ctx->scratch, ctx->off_scratch_conv_a};
            vs_out_dst = {ctx->scratch, ctx->off_scratch_conv_b};
            cs_out_dst = {ctx->scratch, ctx->off_scratch_conv_c};
            ms_out_dst = {ctx->scratch, ctx->off_scratch_conv_d};
        }

        const size_t vkv_off = (size_t)step * ctx->verify_kv_visit_stride_bytes;
        {
            int tg = std::max(32, ((L.head_dim + 31) / 32) * 32);
            const bool use_tau = (!L.is_sliding && ctx->cfg.log_scaling_n_floor > 0.0f);
            KvPrepPrm kprm{};
            kprm.M = M;
            kprm.HQ = L.n_heads;
            kprm.HKV = L.n_kv;
            kprm.R = L.rel_extent;
            kprm.QKV_W = L.qkv_total_rows;
            kprm.QOFF = L.qkv_at[0];
            kprm.ROFF = L.qkv_at[1];
            kprm.KOFF = L.qkv_at[3];
            kprm.VOFF = L.qkv_at[4];
            kprm.X_STRIDE = qkv_s * D;
            kprm.XK_OFF = (qkv_s == 3) ? D : 0;
            kprm.XV_OFF = (qkv_s == 3) ? (2 * D) : 0;
            kprm.TAU = use_tau ? 1 : 0;
            kprm.SEQ = SEQ;
            kprm.WRITE_DIRECT = need_staging ? 0 : 1;
            kprm.IS_ROTATING = is_rot ? 1 : 0;
            kprm.WRITE_BASE = write_base;
            kprm.MAX_SIZE = km.max_size;
            kprm.KV_OFFSET_BEFORE = prev;
            kprm.STRIDE_H = km.stride_h;
            kprm.STRIDE_S = km.stride_s;
            kprm.qk_thr = ctx->cfg.qk_l2_clip_threshold;
            kprm.rel_thr = ctx->cfg.rel_l2_clip_threshold;
            kprm.eps = ctx->cfg.rms_norm_eps;
            kprm.log_n_floor = ctx->cfg.log_scaling_n_floor;
            kprm.log_alpha = ctx->cfg.log_scaling_alpha;

            [enc setComputePipelineState:get_kv_prep_pso(ctx, !L.is_sliding, L.head_dim)];
            [enc setBuffer:ctx->scratch           offset:ctx->off_qkv            atIndex:0];
            [enc setBuffer:ctx->scratch           offset:ctx->off_qkv_in         atIndex:1];
            [enc setBuffer:L.sink_w.buf           offset:L.sink_w.offset         atIndex:2];
            [enc setBuffer:L.kdyn_w.buf           offset:L.kdyn_w.offset         atIndex:3];
            [enc setBuffer:L.vdyn_w.buf           offset:L.vdyn_w.offset         atIndex:4];
            [enc setBuffer:conv_bufs[step*4+0].buf offset:conv_bufs[step*4+0].offset atIndex:5];
            [enc setBuffer:conv_bufs[step*4+1].buf offset:conv_bufs[step*4+1].offset atIndex:6];
            [enc setBuffer:L.k_taps.buf           offset:L.k_taps.offset         atIndex:7];
            [enc setBuffer:L.v_taps.buf           offset:L.v_taps.offset         atIndex:8];
            [enc setBuffer:L.q_scale.buf          offset:L.q_scale.offset        atIndex:9];
            [enc setBuffer:L.rel_scale.buf        offset:L.rel_scale.offset      atIndex:10];
            [enc setBuffer:L.rel_proj.buf         offset:L.rel_proj.offset       atIndex:11];
            [enc setBuffer:ctx->scratch           offset:ctx->off_qo             atIndex:12];
            [enc setBuffer:ctx->scratch           offset:ctx->off_bias           atIndex:13];
            if (need_staging) {
                [enc setBuffer:ctx->scratch       offset:ctx->off_verify_k + vkv_off atIndex:14];
                [enc setBuffer:ctx->scratch       offset:ctx->off_verify_v + vkv_off atIndex:15];
            } else {
                [enc setBuffer:kv_k_bufs[step].buf offset:kv_k_bufs[step].offset atIndex:14];
                [enc setBuffer:kv_v_bufs[step].buf offset:kv_v_bufs[step].offset atIndex:15];
            }
            [enc setBuffer:ctx->scratch           offset:ctx->off_sinks          atIndex:16];
            [enc setBuffer:ks_out_dst.buf         offset:ks_out_dst.offset       atIndex:17];
            [enc setBuffer:vs_out_dst.buf         offset:vs_out_dst.offset       atIndex:18];
            [enc setBytes:&kprm length:sizeof(kprm) atIndex:19];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)L.n_heads, (NSUInteger)M, 1)
                threadsPerThreadgroup:MTLSizeMake((NSUInteger)tg, 1, 1)];
            ++n_dispatches;
        }

        const int s_total = s_kv + ext_m;
        const int win_tg = window_tg(L.head_dim);
        const int max_valid = (L.sliding_window > 0) ? std::min(s_total, L.sliding_window) : s_total;
        const float attn_scale = 1.0f / std::sqrt((float)L.head_dim);
        const int G = L.n_heads / L.n_kv;

        if (max_valid <= win_tg) {
            AttendPrm aprm{M, L.n_heads, L.n_kv, L.rel_extent,
                           L.qkv_total_rows, L.qkv_at[2], L.sliding_window,
                           s_kv, ext_m, ring_idx, km.stride_h, km.stride_s, attn_scale};
            [enc setComputePipelineState:get_attend_pso(ctx, G, L.head_dim, SEQ != 0)];
            [enc setBuffer:ctx->scratch            offset:ctx->off_qo                 atIndex:0];
            [enc setBuffer:kv_k_bufs[step].buf     offset:kv_k_bufs[step].offset      atIndex:1];
            [enc setBuffer:kv_v_bufs[step].buf     offset:kv_v_bufs[step].offset      atIndex:2];
            [enc setBuffer:ctx->scratch            offset:ctx->off_verify_k + vkv_off  atIndex:3];
            [enc setBuffer:ctx->scratch            offset:ctx->off_verify_v + vkv_off  atIndex:4];
            [enc setBuffer:ctx->scratch            offset:ctx->off_bias               atIndex:5];
            [enc setBuffer:ctx->scratch            offset:ctx->off_sinks              atIndex:6];
            [enc setBuffer:ctx->scratch            offset:ctx->off_qkv                atIndex:7];
            [enc setBuffer:ctx->scratch            offset:ctx->off_att                atIndex:8];
            [enc setBytes:&aprm length:sizeof(aprm) atIndex:9];
            [enc setThreadgroupMemoryLength:align_up((size_t)G * (size_t)win_tg * sizeof(float), 16) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)L.n_kv, (NSUInteger)M, 1)
                threadsPerThreadgroup:MTLSizeMake((NSUInteger)win_tg, 1, 1)];
            ++n_dispatches;
        } else {
            int bs = (s_total < 16384) ? 256 : 512;
            int nblocks = (s_total + bs - 1) / bs;
            while (nblocks > ctx->max_nblocks) {
                bs *= 2;
                nblocks = (s_total + bs - 1) / bs;
            }
            DecodePartPrm dprm{M, L.n_heads, L.n_kv, L.rel_extent, bs, nblocks,
                               L.sliding_window, SEQ, s_kv, ext_m, ring_idx,
                               km.stride_h, km.stride_s, attn_scale};
            [enc setComputePipelineState:get_dpart_pso(ctx, G, L.head_dim)];
            [enc setBuffer:ctx->scratch            offset:ctx->off_qo                 atIndex:0];
            [enc setBuffer:kv_k_bufs[step].buf     offset:kv_k_bufs[step].offset      atIndex:1];
            [enc setBuffer:kv_v_bufs[step].buf     offset:kv_v_bufs[step].offset      atIndex:2];
            [enc setBuffer:ctx->scratch            offset:ctx->off_verify_k + vkv_off  atIndex:3];
            [enc setBuffer:ctx->scratch            offset:ctx->off_verify_v + vkv_off  atIndex:4];
            [enc setBuffer:ctx->scratch            offset:ctx->off_bias               atIndex:5];
            [enc setBuffer:ctx->part_buf            offset:ctx->part_off_max           atIndex:6];
            [enc setBuffer:ctx->part_buf            offset:ctx->part_off_sum           atIndex:7];
            [enc setBuffer:ctx->part_buf            offset:ctx->part_off_acc           atIndex:8];
            [enc setBytes:&dprm length:sizeof(dprm) atIndex:9];
            [enc setThreadgroupMemoryLength:align_up(8 * (size_t)G * (size_t)L.head_dim * sizeof(float), 16) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nblocks, (NSUInteger)L.n_kv, (NSUInteger)M)
                threadsPerThreadgroup:MTLSizeMake(8 * 32, 1, 1)];
            ++n_dispatches;

            DecodeCombPrm cprm{L.n_heads, L.head_dim, nblocks, L.qkv_total_rows, L.qkv_at[2]};
            [enc setComputePipelineState:ctx->pso_decode_comb];
            [enc setBuffer:ctx->part_buf offset:ctx->part_off_max atIndex:0];
            [enc setBuffer:ctx->part_buf offset:ctx->part_off_sum atIndex:1];
            [enc setBuffer:ctx->part_buf offset:ctx->part_off_acc atIndex:2];
            [enc setBuffer:ctx->scratch offset:ctx->off_sinks    atIndex:3];
            [enc setBuffer:ctx->scratch offset:ctx->off_qkv      atIndex:4];
            [enc setBuffer:ctx->scratch offset:ctx->off_att      atIndex:5];
            [enc setBytes:&cprm length:sizeof(cprm) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)L.n_heads, (NSUInteger)M, 1)
                threadsPerThreadgroup:MTLSizeMake((NSUInteger)L.head_dim, 1, 1)];
            ++n_dispatches;
        }

        if (need_staging && commit_rotating) {
            KvCommitPrm kcprm{M, L.n_kv, L.head_dim, old_idx, km.max_size, km.stride_h, km.stride_s};
            [enc setComputePipelineState:ctx->pso_kv_commit];
            [enc setBuffer:ctx->scratch        offset:ctx->off_verify_k + vkv_off atIndex:0];
            [enc setBuffer:ctx->scratch        offset:ctx->off_verify_v + vkv_off atIndex:1];
            [enc setBuffer:kv_k_bufs[step].buf offset:kv_k_bufs[step].offset      atIndex:2];
            [enc setBuffer:kv_v_bufs[step].buf offset:kv_v_bufs[step].offset      atIndex:3];
            [enc setBytes:&kcprm length:sizeof(kcprm) atIndex:4];
            uint32_t total = (uint32_t)(M * L.n_kv * L.head_dim);
            [enc dispatchThreadgroups:MTLSizeMake((total + 255) / 256, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            ++n_dispatches;
        }

        km.offset = new_offset;
        km.idx = new_idx;

        encode_qkv_wide(
            ctx->scratch, ctx->off_att,
            L.o_proj.buf, L.o_proj.offset,
            ctx->scratch, ctx->off_att_out,
            1, L.n_heads * L.head_dim, D, 0, 0);

        if (L.is_block_boundary) {
            cur_k += 1;
        }
        encode_mix(
            ctx->scratch, ctx->off_att_out,
            ctx->scratch, ctx->off_blocks,
            L.mlp_query.buf, L.mlp_query.offset,
            conv_bufs[step*4+2].buf, conv_bufs[step*4+2].offset,
            L.attn_taps.buf, L.attn_taps.offset,
            L.is_block_boundary ? ctx->zero_buf : ctx->scratch, L.is_block_boundary ? 0 : ctx->off_x,
            ctx->scratch, ctx->off_messages,
            L.mlp_gate_down.buf, L.mlp_gate_down.offset,
            L.mlp_gate_forget.buf, L.mlp_gate_forget.offset,
            L.mlp_gate_gain.buf, L.mlp_gate_gain.offset,
            L.mlp_norm.buf, L.mlp_norm.offset,
            ctx->scratch, ctx->off_mlp_in,
            ctx->scratch, ctx->off_scratch_partial,
            ctx->scratch, ctx->off_resid,
            cs_out_dst.buf, cs_out_dst.offset,
            cur_k, 1, 1, 1, 3,
            true, !L.is_block_boundary, gated, true, SEQ != 0, L.mlp_eps);

        {
            RoutePrm rprm{ctx->cfg.route_scale};
            [enc setComputePipelineState:ctx->pso_route];
            [enc setBuffer:ctx->scratch   offset:ctx->off_mlp_in    atIndex:0];
            [enc setBuffer:L.router_w.buf offset:L.router_w.offset  atIndex:1];
            [enc setBuffer:L.router_b.buf offset:L.router_b.offset  atIndex:2];
            [enc setBuffer:ctx->scratch   offset:ctx->off_inds      atIndex:3];
            [enc setBuffer:ctx->scratch   offset:ctx->off_weights   atIndex:4];
            [enc setBytes:&rprm length:sizeof(rprm) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)M, 1, 1)
                threadsPerThreadgroup:MTLSizeMake((NSUInteger)(E * ctx->route_parts), 1, 1)];
            ++n_dispatches;
        }

        const bool is_nvfp4 = (L.moe_mode == 0);
        MoePrm mprm{ctx->cfg.mlp_gate_softcap, ctx->cfg.mlp_up_softcap};
        if (M == 1) {
            const int sg13 = is_nvfp4 ? 8 : 4;
            const int sg2  = 8;
            const int nx13 = (INTER + sg13 * 4 - 1) / (sg13 * 4);
            const int nx2  = (D + sg2 * 4 - 1) / (sg2 * 4);

            if (is_nvfp4) {
                [enc setComputePipelineState:ctx->pso_m1_w13_nvfp4];
                [enc setBuffer:ctx->scratch     offset:ctx->off_mlp_in     atIndex:0];
                [enc setBuffer:L.w13_weight.buf offset:L.w13_weight.offset atIndex:1];
                [enc setBuffer:L.w13_scales.buf offset:L.w13_scales.offset atIndex:2];
                [enc setBuffer:L.w13_gscale.buf offset:L.w13_gscale.offset atIndex:3];
                [enc setBuffer:ctx->scratch     offset:ctx->off_inds       atIndex:4];
                [enc setBuffer:ctx->scratch     offset:ctx->off_moe_h      atIndex:5];
                [enc setBytes:&mprm length:sizeof(mprm) atIndex:6];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nx13, (NSUInteger)K_TOP, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)(32 * sg13), 1, 1)];
                ++n_dispatches;

                [enc setComputePipelineState:ctx->pso_m1_w2_nvfp4];
                [enc setBuffer:ctx->scratch    offset:ctx->off_moe_h     atIndex:0];
                [enc setBuffer:L.w2_weight.buf offset:L.w2_weight.offset atIndex:1];
                [enc setBuffer:L.w2_scales.buf offset:L.w2_scales.offset atIndex:2];
                [enc setBuffer:L.w2_gscale.buf offset:L.w2_gscale.offset atIndex:3];
                [enc setBuffer:ctx->scratch    offset:ctx->off_inds      atIndex:4];
                [enc setBuffer:ctx->scratch    offset:ctx->off_ys        atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nx2, (NSUInteger)K_TOP, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)(32 * sg2), 1, 1)];
                ++n_dispatches;
            } else {
                [enc setComputePipelineState:ctx->pso_m1_w13_mxfp8];
                [enc setBuffer:ctx->scratch     offset:ctx->off_mlp_in     atIndex:0];
                [enc setBuffer:L.w13_weight.buf offset:L.w13_weight.offset atIndex:1];
                [enc setBuffer:L.w13_scales.buf offset:L.w13_scales.offset atIndex:2];
                [enc setBuffer:ctx->scratch     offset:ctx->off_inds       atIndex:3];
                [enc setBuffer:ctx->scratch     offset:ctx->off_moe_h      atIndex:4];
                [enc setBytes:&mprm length:sizeof(mprm) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nx13, (NSUInteger)K_TOP, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)(32 * sg13), 1, 1)];
                ++n_dispatches;

                [enc setComputePipelineState:ctx->pso_m1_w2_mxfp8];
                [enc setBuffer:ctx->scratch    offset:ctx->off_moe_h     atIndex:0];
                [enc setBuffer:L.w2_weight.buf offset:L.w2_weight.offset atIndex:1];
                [enc setBuffer:L.w2_scales.buf offset:L.w2_scales.offset atIndex:2];
                [enc setBuffer:ctx->scratch    offset:ctx->off_inds      atIndex:3];
                [enc setBuffer:ctx->scratch    offset:ctx->off_ys        atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nx2, (NSUInteger)K_TOP, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)(32 * sg2), 1, 1)];
                ++n_dispatches;
            }
        } else {
            const int t_slots = M * K_TOP;
            const int max_act = std::min(E, t_slots);
            BucketPrm bprm{t_slots, max_act};
            [enc setComputePipelineState:ctx->pso_bucket];
            [enc setBuffer:ctx->scratch offset:ctx->off_inds         atIndex:0];
            [enc setBuffer:ctx->scratch offset:ctx->off_offsets      atIndex:1];
            [enc setBuffer:ctx->scratch offset:ctx->off_sorted_slots atIndex:2];
            [enc setBuffer:ctx->scratch offset:ctx->off_active_list  atIndex:3];
            [enc setBytes:&bprm length:sizeof(bprm) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake((NSUInteger)ctx->bucket_tg, 1, 1)];
            ++n_dispatches;

            const int grp_mt = (M <= 4) ? 4 : 8;
            const int sg13 = is_nvfp4 ? 2 : 2;
            const int sg2  = is_nvfp4 ? 2 : 2;
            (void)grp_mt;
            const int nx13 = (INTER + sg13 * 4 - 1) / (sg13 * 4);
            const int nx2  = (D + sg2 * 4 - 1) / (sg2 * 4);

            if (is_nvfp4) {
                [enc setComputePipelineState:pso_grp_w13_nv];
                [enc setBuffer:ctx->scratch     offset:ctx->off_mlp_in       atIndex:0];
                [enc setBuffer:L.w13_weight.buf offset:L.w13_weight.offset   atIndex:1];
                [enc setBuffer:L.w13_scales.buf offset:L.w13_scales.offset   atIndex:2];
                [enc setBuffer:L.w13_gscale.buf offset:L.w13_gscale.offset   atIndex:3];
                [enc setBuffer:ctx->scratch     offset:ctx->off_offsets      atIndex:4];
                [enc setBuffer:ctx->scratch     offset:ctx->off_sorted_slots atIndex:5];
                [enc setBuffer:ctx->scratch     offset:ctx->off_active_list  atIndex:6];
                [enc setBuffer:ctx->scratch     offset:ctx->off_moe_h        atIndex:7];
                [enc setBytes:&mprm length:sizeof(mprm) atIndex:8];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nx13, (NSUInteger)max_act, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)(32 * sg13), 1, 1)];
                ++n_dispatches;

                [enc setComputePipelineState:pso_grp_w2_nv];
                [enc setBuffer:ctx->scratch    offset:ctx->off_moe_h        atIndex:0];
                [enc setBuffer:L.w2_weight.buf offset:L.w2_weight.offset    atIndex:1];
                [enc setBuffer:L.w2_scales.buf offset:L.w2_scales.offset    atIndex:2];
                [enc setBuffer:L.w2_gscale.buf offset:L.w2_gscale.offset    atIndex:3];
                [enc setBuffer:ctx->scratch    offset:ctx->off_offsets      atIndex:4];
                [enc setBuffer:ctx->scratch    offset:ctx->off_sorted_slots atIndex:5];
                [enc setBuffer:ctx->scratch    offset:ctx->off_active_list  atIndex:6];
                [enc setBuffer:ctx->scratch    offset:ctx->off_ys           atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nx2, (NSUInteger)max_act, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)(32 * sg2), 1, 1)];
                ++n_dispatches;
            } else {
                [enc setComputePipelineState:pso_grp_w13_mx];
                [enc setBuffer:ctx->scratch     offset:ctx->off_mlp_in       atIndex:0];
                [enc setBuffer:L.w13_weight.buf offset:L.w13_weight.offset   atIndex:1];
                [enc setBuffer:L.w13_scales.buf offset:L.w13_scales.offset   atIndex:2];
                [enc setBuffer:ctx->scratch     offset:ctx->off_offsets      atIndex:3];
                [enc setBuffer:ctx->scratch     offset:ctx->off_sorted_slots atIndex:4];
                [enc setBuffer:ctx->scratch     offset:ctx->off_active_list  atIndex:5];
                [enc setBuffer:ctx->scratch     offset:ctx->off_moe_h        atIndex:6];
                [enc setBytes:&mprm length:sizeof(mprm) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nx13, (NSUInteger)max_act, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)(32 * sg13), 1, 1)];
                ++n_dispatches;

                [enc setComputePipelineState:pso_grp_w2_mx];
                [enc setBuffer:ctx->scratch    offset:ctx->off_moe_h        atIndex:0];
                [enc setBuffer:L.w2_weight.buf offset:L.w2_weight.offset    atIndex:1];
                [enc setBuffer:L.w2_scales.buf offset:L.w2_scales.offset    atIndex:2];
                [enc setBuffer:ctx->scratch    offset:ctx->off_offsets      atIndex:3];
                [enc setBuffer:ctx->scratch    offset:ctx->off_sorted_slots atIndex:4];
                [enc setBuffer:ctx->scratch    offset:ctx->off_active_list  atIndex:5];
                [enc setBuffer:ctx->scratch    offset:ctx->off_ys           atIndex:6];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nx2, (NSUInteger)max_act, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)(32 * sg2), 1, 1)];
                ++n_dispatches;
            }
        }

        {
            CombinePrm cbprm{M};
            int tg = std::min(256, D);
            int nx = (D + tg - 1) / tg;
            [enc setComputePipelineState:(SEQ ? ctx->pso_combine_seq : ctx->pso_combine_noseq)];
            [enc setBuffer:ctx->scratch            offset:ctx->off_ys                atIndex:0];
            [enc setBuffer:ctx->scratch            offset:ctx->off_weights           atIndex:1];
            [enc setBuffer:conv_bufs[step*4+3].buf offset:conv_bufs[step*4+3].offset atIndex:2];
            [enc setBuffer:L.mlp_taps.buf          offset:L.mlp_taps.offset          atIndex:3];
            [enc setBuffer:ctx->scratch            offset:ctx->off_resid             atIndex:4];
            [enc setBuffer:ctx->scratch            offset:ctx->off_x                 atIndex:5];
            [enc setBuffer:ms_out_dst.buf          offset:ms_out_dst.offset          atIndex:6];
            [enc setBytes:&cbprm length:sizeof(cbprm) atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nx, (NSUInteger)M, 1)
                threadsPerThreadgroup:MTLSizeMake((NSUInteger)tg, 1, 1)];
            ++n_dispatches;
        }

        if (need_copy_conv_back) {
            const size_t last_c_off = (size_t)(M - 1) * 3 * (size_t)ch * sizeof(float);
            const size_t last_d_off = (size_t)(M - 1) * 3 * (size_t)D * sizeof(float);
            const uint32_t nc = (uint32_t)(3 * ch);
            const uint32_t nd = (uint32_t)(3 * D);
            const uint32_t nmax = std::max(nc, nd);
            CopyConv4Prm c4prm{nc, nd};
            [enc setComputePipelineState:ctx->pso_copy_conv4];
            [enc setBuffer:ks_out_dst.buf          offset:ks_out_dst.offset + last_c_off atIndex:0];
            [enc setBuffer:vs_out_dst.buf          offset:vs_out_dst.offset + last_c_off atIndex:1];
            [enc setBuffer:cs_out_dst.buf          offset:cs_out_dst.offset + last_d_off atIndex:2];
            [enc setBuffer:ms_out_dst.buf          offset:ms_out_dst.offset + last_d_off atIndex:3];
            [enc setBuffer:conv_bufs[step*4+0].buf offset:conv_bufs[step*4+0].offset     atIndex:4];
            [enc setBuffer:conv_bufs[step*4+1].buf offset:conv_bufs[step*4+1].offset     atIndex:5];
            [enc setBuffer:conv_bufs[step*4+2].buf offset:conv_bufs[step*4+2].offset     atIndex:6];
            [enc setBuffer:conv_bufs[step*4+3].buf offset:conv_bufs[step*4+3].offset     atIndex:7];
            [enc setBytes:&c4prm length:sizeof(c4prm) atIndex:8];
            [enc dispatchThreadgroups:MTLSizeMake((nmax + 255) / 256, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            ++n_dispatches;
        }
    }

    encode_mix(
        ctx->scratch, ctx->off_x,
        ctx->scratch, ctx->off_blocks,
        ctx->globals.final_query.buf, ctx->globals.final_query.offset,
        ctx->zero_buf, 0,
        ctx->zero_buf, 0,
        ctx->zero_buf, 0,
        ctx->zero_buf, 0,
        ctx->zero_buf, 0,
        ctx->zero_buf, 0,
        ctx->zero_buf, 0,
        ctx->zero_buf, 0,
        ctx->scratch, ctx->off_pre_norm,
        ctx->scratch, ctx->off_scratch_partial,
        ctx->scratch, ctx->off_resid,
        ctx->scratch, ctx->off_scratch_conv_a,
        cur_k, 1, 1, 1, 0,
        false, false, false, false, false, 1e-6f);

    int final_cap_slot = -1;
    if (enable_captures && n_captures > 0 && capture_layer_ids) {
        for (int ci = 0; ci < n_captures; ++ci) {
            if (capture_layer_ids[ci] == NV) {
                final_cap_slot = ci;
                break;
            }
        }
    }
    const bool need_final_norm = force_final_norm || (final_cap_slot >= 0);
    if (need_final_norm) {
        FinalNormPrm fprm{final_cap_slot, n_captures, ctx->globals.final_norm_eps};
        [enc setComputePipelineState:ctx->pso_final_norm];
        [enc setBuffer:ctx->scratch                offset:ctx->off_pre_norm           atIndex:0];
        [enc setBuffer:ctx->globals.final_norm.buf offset:ctx->globals.final_norm.offset atIndex:1];
        [enc setBuffer:ctx->scratch                offset:ctx->off_normed_f32         atIndex:2];
        [enc setBuffer:ctx->scratch                offset:ctx->off_captures           atIndex:3];
        [enc setBytes:&fprm length:sizeof(fprm) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)M, 1)
            threadsPerThreadgroup:MTLSizeMake((NSUInteger)ctx->mix_tg, 1, 1)];
        ++n_dispatches;
    }
}

int inkling_ctx_step(
    InklingCtx*          ctx,
    const int32_t*       token_ids_ptr,
    int32_t              m_tokens,
    int32_t              seq_mode,
    void* const*         kv_ptrs,
    void* const*         conv_ptrs,
    InklingKVMeta*       kv_meta,
    const int32_t*       capture_layer_ids,
    int32_t              n_captures,
    void*                out_pre_norm_ptr,
    int32_t*             out_toks_ptr,
    void*                out_captures_ptr,
    void*                out_conv_states_ptr,
    int32_t              commit_rotating,
    InklingStepStats*    stats_out)
{
    @autoreleasepool {
        if (!ctx || !ctx->globals.registered || !token_ids_ptr || !kv_ptrs || !conv_ptrs || !kv_meta) {
            return -1;
        }
        const int32_t verify_max_m = std::min<int32_t>(ctx->cfg.max_m, 8);
        if (m_tokens < 1 || m_tokens > verify_max_m || (seq_mode == 0 && m_tokens > verify_max_m)) {
            fprintf(stderr, "[inkling_decode] ERROR: m_tokens=%d out of [1, %d] (use inkling_ctx_step_batched for B > 8)\n",
                    m_tokens, verify_max_m);
            return -2;
        }
        if (n_captures > ctx->max_captures) {
            fprintf(stderr, "[inkling_decode] ERROR: n_captures=%d > max %d\n", n_captures, ctx->max_captures);
            return -3;
        }
        const int32_t D = ctx->cfg.hidden_size;
        const int32_t M = m_tokens;
        const int32_t SEQ = (seq_mode && M > 1) ? 1 : 0;

        for (int i = 0; i < ctx->cfg.n_layers; ++i) {
            if (!ctx->layers[i].registered) {
                fprintf(stderr, "[inkling_decode] ERROR: layer %d not registered\n", i);
                return -4;
            }
        }

        BufSlice out_conv_slice = {nil, 0};
        if (out_conv_states_ptr) {
            size_t n_floats = inkling_ctx_conv_states_floats(ctx, M);
            out_conv_slice = wrap_cached_ptr(ctx, out_conv_states_ptr, n_floats * sizeof(float));
            if (!out_conv_slice.buf) return -5;
        }

        const bool use_internal_conv_hist = (SEQ != 0) && (commit_rotating == 0) && (out_conv_states_ptr == nullptr);
        if (use_internal_conv_hist) {
            int rc = ensure_conv_hist_buf(ctx, verify_max_m);
            if (rc != 0) return rc;
        }
        ctx->conv_hist_active = use_internal_conv_hist;
        ctx->conv_hist_m = use_internal_conv_hist ? M : 0;
        ctx->last_verify_m = (SEQ != 0 && commit_rotating == 0) ? M : 0;

        {
            int rc = prepare_target_kv_conv_part(ctx, M, kv_ptrs, conv_ptrs, kv_meta);
            if (rc != 0) return rc;
        }

        const int grp_mt    = (M <= 4) ? 4 : 8;
        const int sg_grp_nv = 2;
        const int sg_grp_mx = 2;
        id<MTLComputePipelineState> pso_grp_w13_nv = (M > 1) ? get_grp_moe_pso(ctx, true,  false, sg_grp_nv, grp_mt) : nil;
        id<MTLComputePipelineState> pso_grp_w2_nv  = (M > 1) ? get_grp_moe_pso(ctx, true,  true,  sg_grp_nv, grp_mt) : nil;
        id<MTLComputePipelineState> pso_grp_w13_mx = (M > 1) ? get_grp_moe_pso(ctx, false, false, sg_grp_mx, grp_mt) : nil;
        id<MTLComputePipelineState> pso_grp_w2_mx  = (M > 1) ? get_grp_moe_pso(ctx, false, true,  sg_grp_mx, grp_mt) : nil;
        const int argmax_r = (M == 1) ? 4 : 2;
        id<MTLComputePipelineState> pso_argmax_p1 = (out_toks_ptr && ctx->globals.lm_head.buf != ctx->zero_buf)
                                                  ? get_argmax_p1_pso(ctx, M, argmax_r) : nil;

        uint64_t t_wall0 = mach_absolute_time();
        uint64_t t_enc0 = t_wall0;

        id<MTLCommandBuffer> cb = [ctx->queue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
        int32_t n_dispatches = 0;

        {
            EmbedPrm eprm{};
            eprm.eps = ctx->globals.embed_norm_eps;
            for (int i = 0; i < M; ++i) eprm.toks[i] = token_ids_ptr[i];
            [enc setComputePipelineState:ctx->pso_embed];
            [enc setBuffer:ctx->globals.embed_tokens.buf offset:ctx->globals.embed_tokens.offset atIndex:0];
            [enc setBuffer:ctx->globals.embed_norm.buf   offset:ctx->globals.embed_norm.offset   atIndex:1];
            [enc setBuffer:ctx->scratch                  offset:ctx->off_x                       atIndex:2];
            [enc setBytes:&eprm length:sizeof(eprm) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)M, 1)
                threadsPerThreadgroup:MTLSizeMake((NSUInteger)ctx->mix_tg, 1, 1)];
            ++n_dispatches;
        }

        encode_target_visits_and_norm(
            ctx, enc, M, SEQ, kv_meta,
            capture_layer_ids, n_captures, (out_captures_ptr != nullptr),
            (out_toks_ptr != nullptr), (out_conv_states_ptr != nullptr), out_conv_slice,
            use_internal_conv_hist, commit_rotating,
            pso_grp_w13_nv, pso_grp_w2_nv, pso_grp_w13_mx, pso_grp_w2_mx,
            n_dispatches);

        if (pso_argmax_p1) {
            const int sg = 8;
            const int v_limit = ctx->cfg.unpadded_vocab_size;
            const int ntg = (v_limit + sg * argmax_r - 1) / (sg * argmax_r);
            ArgmaxP1Prm p1{ntg};
            [enc setComputePipelineState:pso_argmax_p1];
            [enc setBuffer:ctx->scratch             offset:ctx->off_normed_f32         atIndex:0];
            [enc setBuffer:ctx->globals.lm_head.buf offset:ctx->globals.lm_head.offset atIndex:1];
            [enc setBuffer:ctx->scratch             offset:ctx->off_argmax_pv          atIndex:2];
            [enc setBuffer:ctx->scratch             offset:ctx->off_argmax_pi          atIndex:3];
            [enc setBytes:&p1 length:sizeof(p1) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)ntg, 1, 1)
                threadsPerThreadgroup:MTLSizeMake((NSUInteger)(sg * 32), 1, 1)];
            ++n_dispatches;

            ArgmaxP2Prm p2{ntg};
            [enc setComputePipelineState:ctx->pso_argmax_p2];
            [enc setBuffer:ctx->scratch offset:ctx->off_argmax_pv atIndex:0];
            [enc setBuffer:ctx->scratch offset:ctx->off_argmax_pi atIndex:1];
            [enc setBuffer:ctx->scratch offset:ctx->off_out_toks  atIndex:2];
            [enc setBytes:&p2 length:sizeof(p2) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)M, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            ++n_dispatches;
        }

        [enc endEncoding];
        uint64_t t_enc1 = mach_absolute_time();

        [cb commit];
        [cb waitUntilCompleted];
        uint64_t t_wall1 = mach_absolute_time();

        if (cb.status != MTLCommandBufferStatusCompleted) {
            fprintf(stderr, "[inkling_decode] ERROR: command buffer failed: %s\n",
                    cb.error ? cb.error.localizedDescription.UTF8String : "unknown");
            return -10;
        }

        const uint8_t* sptr = (const uint8_t*)[ctx->scratch contents];
        if (out_pre_norm_ptr) {
            memcpy(out_pre_norm_ptr, sptr + ctx->off_pre_norm, (size_t)M * (size_t)D * sizeof(float));
        }
        if (out_toks_ptr) {
            memcpy(out_toks_ptr, sptr + ctx->off_out_toks, (size_t)M * sizeof(int32_t));
        }
        if (out_captures_ptr && n_captures > 0) {
            memcpy(out_captures_ptr, sptr + ctx->off_captures,
                   (size_t)M * (size_t)n_captures * (size_t)D * sizeof(uint16_t));
        }

        if (stats_out) {
            stats_out->encode_ms    = mach_to_ms(t_enc1 - t_enc0, ctx->tb);
            stats_out->gpu_ms       = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
            stats_out->wall_ms      = mach_to_ms(t_wall1 - t_wall0, ctx->tb);
            stats_out->n_dispatches = n_dispatches;
        }
        return 0;
    }
}

int inkling_ctx_commit_rotating(
    InklingCtx*    ctx,
    void* const*   kv_ptrs,
    InklingKVMeta* kv_meta,
    int32_t        n_accepted)
{
    @autoreleasepool {
        if (!ctx || !kv_ptrs || !kv_meta || n_accepted < 0) return -1;
        const int32_t NV = (int32_t)ctx->schedule.size();

        bool any_active = false;
        for (int v = 0; v < NV; ++v) {
            if (ctx->pending_rot[v].active) { any_active = true; break; }
        }
        if (!any_active || n_accepted == 0) {
            for (int v = 0; v < NV; ++v) ctx->pending_rot[v].active = false;
            return 0;
        }

        auto& kv_k_bufs = ctx->tmp_kv_k;
        auto& kv_v_bufs = ctx->tmp_kv_v;
        for (int v = 0; v < NV; ++v) {
            if (!ctx->pending_rot[v].active) continue;
            const auto& L = ctx->layers[ctx->schedule[v].layer_idx];
            size_t kv_bytes = (size_t)L.n_kv * (size_t)kv_meta[v].stride_h * sizeof(uint16_t);
            kv_k_bufs[v] = wrap_cached_ptr(ctx, kv_ptrs[v * 2 + 0], kv_bytes);
            kv_v_bufs[v] = wrap_cached_ptr(ctx, kv_ptrs[v * 2 + 1], kv_bytes);
            if (!kv_k_bufs[v].buf || !kv_v_bufs[v].buf) return -2;
        }

        id<MTLCommandBuffer> cb = [ctx->queue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];

        for (int v = 0; v < NV; ++v) {
            PendingRotating& pr = ctx->pending_rot[v];
            if (!pr.active) continue;
            int n_toks = std::min(n_accepted, pr.m_tokens);
            const auto& L = ctx->layers[ctx->schedule[v].layer_idx];
            InklingKVMeta& km = kv_meta[v];
            const size_t vkv_off = (size_t)v * ctx->verify_kv_visit_stride_bytes;

            KvCommitPrm kcprm{n_toks, L.n_kv, L.head_dim, pr.old_idx, pr.max_size, km.stride_h, km.stride_s};
            [enc setComputePipelineState:ctx->pso_kv_commit];
            [enc setBuffer:ctx->scratch     offset:ctx->off_verify_k + vkv_off atIndex:0];
            [enc setBuffer:ctx->scratch     offset:ctx->off_verify_v + vkv_off atIndex:1];
            [enc setBuffer:kv_k_bufs[v].buf offset:kv_k_bufs[v].offset         atIndex:2];
            [enc setBuffer:kv_v_bufs[v].buf offset:kv_v_bufs[v].offset         atIndex:3];
            [enc setBytes:&kcprm length:sizeof(kcprm) atIndex:4];
            uint32_t total = (uint32_t)(n_toks * L.n_kv * L.head_dim);
            [enc dispatchThreadgroups:MTLSizeMake((total + 255) / 256, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            km.offset = pr.prev_offset + n_toks;
            if (km.offset <= pr.max_size) {
                km.idx = km.offset;
            } else {
                km.idx = ((pr.old_idx + n_toks - 1) % pr.max_size) + 1;
            }
            pr.active = false;
        }

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return (cb.status == MTLCommandBufferStatusCompleted) ? 0 : -3;
    }
}

int inkling_ctx_commit_verify(
    InklingCtx*    ctx,
    void* const*   kv_ptrs,
    void* const*   conv_ptrs,
    InklingKVMeta* kv_meta,
    int32_t        n_accepted,
    int32_t        m_tokens)
{
    if (!ctx || !kv_meta || n_accepted < 0 || m_tokens < 1 || n_accepted > m_tokens) return -1;
    if (ctx->last_verify_m > 0 && m_tokens != ctx->last_verify_m) return -2;

    const int32_t NV = (int32_t)ctx->schedule.size();
    const int32_t D = ctx->cfg.hidden_size;
    const uint8_t* sptr = (const uint8_t*)[ctx->scratch contents];
    const uint8_t* chptr = ctx->conv_hist_buf ? (const uint8_t*)[ctx->conv_hist_buf contents] : nullptr;

    for (int v = 0; v < NV; ++v) {
        PendingVerify& pv = ctx->pending_verify[v];
        InklingKVMeta& km = kv_meta[v];
        const auto& L = ctx->layers[ctx->schedule[v].layer_idx];
        const int32_t n_toks = std::min(n_accepted, pv.m_tokens);

        if (pv.kv_staged) {
            if (n_toks > 0) {
                if (!kv_ptrs || !kv_ptrs[v * 2 + 0] || !kv_ptrs[v * 2 + 1]) return -3;
                const size_t vkv_off = (size_t)v * ctx->verify_kv_visit_stride_bytes;
                const uint16_t* src_k = (const uint16_t*)(sptr + ctx->off_verify_k + vkv_off);
                const uint16_t* src_v = (const uint16_t*)(sptr + ctx->off_verify_v + vkv_off);
                uint16_t* dst_k = (uint16_t*)kv_ptrs[v * 2 + 0];
                uint16_t* dst_v = (uint16_t*)kv_ptrs[v * 2 + 1];
                const size_t row_bytes = (size_t)L.head_dim * sizeof(uint16_t);
                for (int m = 0; m < n_toks; ++m) {
                    const int phys = (pv.old_idx + m) % pv.max_size;
                    for (int h = 0; h < L.n_kv; ++h) {
                        const size_t src_off = ((size_t)m * (size_t)L.n_kv + (size_t)h) * (size_t)L.head_dim;
                        const size_t dst_off = (size_t)h * (size_t)km.stride_h + (size_t)phys * (size_t)km.stride_s;
                        memcpy(dst_k + dst_off, src_k + src_off, row_bytes);
                        memcpy(dst_v + dst_off, src_v + src_off, row_bytes);
                    }
                }
                km.offset = pv.prev_offset + n_toks;
                km.idx = (km.offset <= pv.max_size)
                       ? km.offset
                       : (((pv.old_idx + n_toks - 1) % pv.max_size) + 1);
            } else {
                km.offset = pv.prev_offset;
                km.idx = pv.old_idx;
            }
        } else {
            km.offset = pv.prev_offset + n_toks;
            km.idx = pv.prev_offset + n_toks;
        }

        if (ctx->conv_hist_active && chptr && conv_ptrs && n_toks > 0) {
            const size_t ch = (size_t)L.n_kv * (size_t)L.head_dim;
            const size_t c_row_bytes = 3 * ch * sizeof(float);
            const size_t d_row_bytes = 3 * (size_t)D * sizeof(float);
            const size_t M_hist = (size_t)ctx->conv_hist_m;
            const size_t r = (size_t)(n_toks - 1);
            const uint8_t* vbase = chptr + ctx->conv_hist_visit_off[v];
            const uint8_t* ks_base = vbase;
            const uint8_t* vs_base = ks_base + M_hist * c_row_bytes;
            const uint8_t* cs_base = vs_base + M_hist * c_row_bytes;
            const uint8_t* ms_base = cs_base + M_hist * d_row_bytes;
            memcpy(conv_ptrs[v * 4 + 0], ks_base + r * c_row_bytes, c_row_bytes);
            memcpy(conv_ptrs[v * 4 + 1], vs_base + r * c_row_bytes, c_row_bytes);
            memcpy(conv_ptrs[v * 4 + 2], cs_base + r * d_row_bytes, d_row_bytes);
            memcpy(conv_ptrs[v * 4 + 3], ms_base + r * d_row_bytes, d_row_bytes);
        }

        pv.kv_staged = false;
        ctx->pending_rot[v].active = false;
    }

    ctx->conv_hist_active = false;
    ctx->conv_hist_m = 0;
    ctx->last_verify_m = 0;
    return 0;
}

int inkling_ctx_step_batched(
    InklingCtx*             ctx,
    const int32_t*          token_ids_ptr,
    int32_t                 batch_size,
    void* const*            kv_ptrs,
    void* const*            conv_ptrs,
    InklingBatchKVMeta*     kv_meta,
    int32_t*                kv_row_offsets_ptr,
    void*                   out_pre_norm_ptr,
    int32_t*                out_toks_ptr,
    void*                   out_logits_ptr,
    InklingStepStats*       stats_out)
{
    @autoreleasepool {
        if (!ctx || !ctx->globals.registered || !token_ids_ptr || !kv_ptrs || !conv_ptrs ||
            !kv_meta || !kv_row_offsets_ptr) {
            return -1;
        }
        if (batch_size < 1 || batch_size > ctx->cfg.max_m) {
            fprintf(stderr, "[inkling_decode] ERROR: batch_size=%d out of [1, %d]\n",
                    batch_size, ctx->cfg.max_m);
            return -2;
        }
        if (out_logits_ptr != nullptr) {
            fprintf(stderr, "[inkling_decode] ERROR: out_logits_ptr is reserved NULL\n");
            return -11;
        }
        for (int i = 0; i < ctx->cfg.n_layers; ++i) {
            if (!ctx->layers[i].registered) {
                fprintf(stderr, "[inkling_decode] ERROR: layer %d not registered\n", i);
                return -4;
            }
        }

        const int32_t NV = (int32_t)ctx->schedule.size();
        const int32_t D = ctx->cfg.hidden_size;
        const int32_t INTER = ctx->cfg.inter_size;
        const int32_t E = ctx->cfg.n_experts;
        const int32_t K_TOP = ctx->cfg.top_k;
        const int32_t RANK = std::max(1, ctx->cfg.loop_gate_rank);
        const int32_t B = batch_size;

        auto& kv_k_bufs = ctx->tmp_kv_k;
        auto& kv_v_bufs = ctx->tmp_kv_v;
        auto& conv_bufs = ctx->tmp_conv;
        auto& plans     = ctx->tmp_batch_plans;

        uint8_t* sptr_rw = (uint8_t*)[ctx->scratch contents];
        const float log_floor = ctx->cfg.log_scaling_n_floor;
        const float log_alpha = ctx->cfg.log_scaling_alpha;
        size_t max_part_meta = 0;
        size_t max_part_acc  = 0;

        for (int v = 0; v < NV; ++v) {
            const auto& L = ctx->layers[ctx->schedule[v].layer_idx];
            const InklingBatchKVMeta& km = kv_meta[v];
            if (km.stride_b <= 0 || km.stride_h <= 0 || km.stride_s <= 0 || km.capacity <= 0) {
                return -6;
            }
            const size_t kv_bytes = (size_t)B * (size_t)km.stride_b * sizeof(uint16_t);
            kv_k_bufs[v] = wrap_cached_ptr(ctx, kv_ptrs[v * 2 + 0], kv_bytes);
            kv_v_bufs[v] = wrap_cached_ptr(ctx, kv_ptrs[v * 2 + 1], kv_bytes);
            if (!kv_k_bufs[v].buf || !kv_v_bufs[v].buf) return -6;

            const size_t ch = (size_t)L.n_kv * (size_t)L.head_dim;
            const size_t c_bytes[4] = {
                (size_t)B * 3 * ch * sizeof(float),
                (size_t)B * 3 * ch * sizeof(float),
                (size_t)B * 3 * (size_t)D * sizeof(float),
                (size_t)B * 3 * (size_t)D * sizeof(float),
            };
            for (int c = 0; c < 4; ++c) {
                conv_bufs[v * 4 + c] = wrap_cached_ptr(ctx, conv_ptrs[v * 4 + c], c_bytes[c]);
                if (!conv_bufs[v * 4 + c].buf) return -7;
            }

            InklingCtx::BatchVisitPlan& P = plans[v];
            const bool is_rot = (km.is_rotating != 0);
            if (!is_rot) {
                P.write_slot = km.idx;
                if (P.write_slot < 0 || P.write_slot >= km.capacity) {
                    fprintf(stderr, "[inkling_decode] ERROR: BatchKVCache visit %d write_slot=%d >= cap=%d\n",
                            v, P.write_slot, km.capacity);
                    return -6;
                }
                P.new_s_len = P.write_slot + 1;
                P.ring_idx = 0;
                P.new_idx = P.write_slot + 1;
                P.new_off_scalar = km.offset;
            } else {
                if (km.max_size <= 0) return -6;
                P.write_slot = (km.idx >= km.max_size) ? 0 : km.idx;
                if (P.write_slot < 0 || P.write_slot >= km.capacity) {
                    fprintf(stderr, "[inkling_decode] ERROR: BatchRotatingKVCache visit %d write_slot=%d >= cap=%d\n",
                            v, P.write_slot, km.capacity);
                    return -6;
                }
                P.new_idx = P.write_slot + 1;
                P.new_off_scalar = km.offset + 1;
                if (P.new_off_scalar < km.max_size) {
                    P.new_s_len = P.new_off_scalar;
                    P.ring_idx = 0;
                } else {
                    P.new_s_len = km.max_size;
                    P.ring_idx = P.new_idx % km.max_size;
                }
            }

            int32_t* counts_dst = (int32_t*)(sptr_rw + ctx->off_batch_counts + (size_t)v * ctx->batch_visit_meta_stride_bytes);
            float* tau_dst      = (float*)(sptr_rw + ctx->off_batch_tau + (size_t)v * ctx->batch_visit_meta_stride_bytes);
            const int32_t* row_off = kv_row_offsets_ptr + (size_t)v * (size_t)B;
            P.use_tau = (!L.is_sliding && log_floor > 0.0f);
            for (int32_t b = 0; b < B; ++b) {
                const int32_t old_r = row_off[b];
                const int32_t new_r = old_r + 1;
                counts_dst[b] = std::max(0, std::min(new_r, P.new_s_len));
                if (P.use_tau) {
                    const float pos = float(old_r + 1);
                    tau_dst[b] = 1.0f + log_alpha * std::log(std::max(pos / log_floor, 1.0f));
                }
            }

            const int win_tg = window_tg(L.head_dim);
            P.use_splitk = (P.new_s_len > win_tg) || (B >= 8 && P.new_s_len >= 128);
            if (P.use_splitk) {
                int bs = (P.new_s_len < 16384) ? 256 : 512;
                int nblocks = std::max(1, (P.new_s_len + bs - 1) / bs);
                while (nblocks > ctx->max_nblocks) {
                    bs *= 2;
                    nblocks = (P.new_s_len + bs - 1) / bs;
                }
                P.bs = bs;
                P.nblocks = nblocks;
                max_part_meta = std::max(max_part_meta, (size_t)nblocks * (size_t)L.n_heads);
                max_part_acc  = std::max(max_part_acc,  (size_t)nblocks * (size_t)L.n_heads * (size_t)L.head_dim);
            } else {
                P.bs = 0;
                P.nblocks = 0;
            }
        }

        if (max_part_meta > 0) {
            const size_t sz_meta = align_up((size_t)B * max_part_meta * sizeof(float), 256);
            const size_t sz_acc  = align_up((size_t)B * max_part_acc  * sizeof(float), 256);
            if (ensure_part_buf(ctx, sz_meta, sz_meta, sz_acc) != 0) return -9;
        }

        const bool use_nax_qkv   = (B >= 9);
        const bool use_nax_head  = (B >= 9);
        const bool use_nax_msg_b = (B >= 16) && (ctx->lib_nax != nil);
        const bool use_nax_rt_b  = (B >= 9) && (ctx->lib_nax != nil);
        const bool use_batch_msg = (B >= 16);
        const bool use_batch_rt  = (B >= 16 && E >= 32);
        const bool use_moe_tiles = (B >= 3);
        const bool use_nax_req   = use_nax_qkv || use_nax_head;
        if (use_nax_req && !ctx->lib_nax) return -12;

        id<MTLComputePipelineState> pso_nax_qkv    = nil;
        id<MTLComputePipelineState> pso_nax_o      = nil;
        id<MTLComputePipelineState> pso_nax_msg_fg = nil;
        if (ctx->lib_nax) {
            if (B <= 16) {
                pso_nax_qkv    = ctx->pso_nax_qkv_16x64;
                pso_nax_o      = ctx->pso_nax_qkv_16x64;
                pso_nax_msg_fg = ctx->pso_nax_msg_fg_16x64;
            } else if (B <= 32) {
                pso_nax_qkv    = ctx->pso_nax_qkv_32x64;
                pso_nax_o      = ctx->pso_nax_qkv_32x64;
                pso_nax_msg_fg = ctx->pso_nax_msg_fg_32x64;
            } else {
                pso_nax_qkv    = ctx->pso_nax_qkv_64x64;
                pso_nax_o      = ctx->pso_nax_qkv_64x64;
                pso_nax_msg_fg = ctx->pso_nax_msg_fg_64x64;
            }
        }

        const int grp_mt    = (B <= 8) ? 4 : 8;
        const int sg_grp_nv = (B <= 8) ? 4 : 2;
        const int sg_grp_mx = (B <= 8) ? 4 : 2;
        id<MTLComputePipelineState> pso_grp_w13_nv = nil;
        id<MTLComputePipelineState> pso_grp_w2_nv  = nil;
        id<MTLComputePipelineState> pso_grp_w13_mx = nil;
        id<MTLComputePipelineState> pso_grp_w2_mx  = nil;
        if (use_moe_tiles) {
            pso_grp_w13_nv = get_grp_moe_tiles_pso(ctx, true,  false, sg_grp_nv, grp_mt);
            pso_grp_w2_nv  = get_grp_moe_tiles_pso(ctx, true,  true,  sg_grp_nv, grp_mt);
            pso_grp_w13_mx = get_grp_moe_tiles_pso(ctx, false, false, sg_grp_mx, grp_mt);
            pso_grp_w2_mx  = get_grp_moe_tiles_pso(ctx, false, true,  sg_grp_mx, grp_mt);
        }

        const int argmax_r = (B == 1) ? 4 : 2;
        const bool want_argmax = (out_toks_ptr != nullptr && ctx->globals.lm_head.buf != ctx->zero_buf);
        id<MTLComputePipelineState> pso_simt_argmax_p1 = (want_argmax && !use_nax_head)
                                                         ? get_argmax_p1_pso(ctx, B, argmax_r) : nil;

        uint64_t t_wall0 = mach_absolute_time();
        uint64_t t_enc0 = t_wall0;

        id<MTLCommandBuffer> cb = [ctx->queue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
        int32_t n_dispatches = 0;

        auto dispatch_copy_bf16 = [&](id<MTLBuffer> sb, size_t so, id<MTLBuffer> db, size_t doff, uint32_t count) {
            if (count == 0) return;
            CopyPrm prm{count};
            [enc setComputePipelineState:ctx->pso_copy_bf16];
            [enc setBuffer:sb offset:so atIndex:0];
            [enc setBuffer:db offset:doff atIndex:1];
            [enc setBytes:&prm length:sizeof(prm) atIndex:2];
            uint32_t tg = 256;
            [enc dispatchThreadgroups:MTLSizeMake((count + tg - 1) / tg, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            ++n_dispatches;
        };

        auto dispatch_copy_f32 = [&](id<MTLBuffer> sb, size_t so, id<MTLBuffer> db, size_t doff, uint32_t count) {
            if (count == 0) return;
            CopyPrm prm{count};
            [enc setComputePipelineState:ctx->pso_copy_f32];
            [enc setBuffer:sb offset:so atIndex:0];
            [enc setBuffer:db offset:doff atIndex:1];
            [enc setBytes:&prm length:sizeof(prm) atIndex:2];
            uint32_t tg = 256;
            [enc dispatchThreadgroups:MTLSizeMake((count + tg - 1) / tg, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            ++n_dispatches;
        };

        auto encode_mix = [&](
            id<MTLBuffer> x_b, size_t x_o,
            id<MTLBuffer> blk_b, size_t blk_o,
            id<MTLBuffer> qry_b, size_t qry_o,
            id<MTLBuffer> st_b, size_t st_o,
            id<MTLBuffer> tap_b, size_t tap_o,
            id<MTLBuffer> res_b, size_t res_o,
            id<MTLBuffer> msg_b, size_t msg_o,
            id<MTLBuffer> dn_b, size_t dn_o,
            id<MTLBuffer> fg_b, size_t fg_o,
            id<MTLBuffer> gn_b, size_t gn_o,
            id<MTLBuffer> fg_bf16_b, size_t fg_bf16_o, bool has_fg_bf16,
            id<MTLBuffer> nw_b, size_t nw_o,
            id<MTLBuffer> out_b, size_t out_o,
            id<MTLBuffer> part_b, size_t part_o,
            id<MTLBuffer> resid_b, size_t resid_o,
            id<MTLBuffer> nst_b, size_t nst_o,
            int K, int Q, int QO, int QN, int MOFF,
            bool conv, bool add, bool gate, bool norm, float eps)
        {
            if (gate && use_nax_msg_b && has_fg_bf16) {
                MixGateP1Prm p1{B, Q, QO, MOFF, add ? 1 : 0, 0};
                [enc setComputePipelineState:get_mix_gate_p1_pso(ctx, conv, K)];
                [enc setBuffer:x_b                  offset:x_o                  atIndex:0];
                [enc setBuffer:blk_b                offset:blk_o                atIndex:1];
                [enc setBuffer:qry_b                offset:qry_o                atIndex:2];
                [enc setBuffer:st_b                 offset:st_o                 atIndex:3];
                [enc setBuffer:tap_b                offset:tap_o                atIndex:4];
                [enc setBuffer:res_b                offset:res_o                atIndex:5];
                [enc setBuffer:msg_b                offset:msg_o                atIndex:6];
                [enc setBuffer:ctx->scratch         offset:ctx->off_msg_o       atIndex:7];
                [enc setBuffer:ctx->scratch         offset:ctx->off_msg_gate_in atIndex:8];
                [enc setBuffer:part_b               offset:part_o               atIndex:9];
                [enc setBuffer:resid_b              offset:resid_o              atIndex:10];
                [enc setBuffer:nst_b                offset:nst_o                atIndex:11];
                [enc setBytes:&p1 length:sizeof(p1) atIndex:12];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)QO, (NSUInteger)B, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)ctx->mix_tg, 1, 1)];
                ++n_dispatches;

                MsgDownPrm pd{B};
                [enc setComputePipelineState:ctx->pso_msg_down_batched_bf16];
                [enc setBuffer:ctx->scratch         offset:ctx->off_msg_gate_in atIndex:0];
                [enc setBuffer:dn_b                 offset:dn_o                 atIndex:1];
                [enc setBuffer:ctx->scratch         offset:ctx->off_msg_low     atIndex:2];
                [enc setBytes:&pd length:sizeof(pd) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((RANK + 7) / 8),
                                                      (NSUInteger)QO,
                                                      (NSUInteger)((B + 15) / 16))
                    threadsPerThreadgroup:MTLSizeMake(8 * 32, 1, 1)];
                ++n_dispatches;

                NaxMsgPrm pm_nax{B};
                [enc setComputePipelineState:pso_nax_msg_fg];
                [enc setBuffer:ctx->scratch         offset:ctx->off_msg_low     atIndex:0];
                [enc setBuffer:fg_bf16_b            offset:fg_bf16_o            atIndex:1];
                [enc setBuffer:ctx->scratch         offset:ctx->off_msg_gate_in atIndex:2];
                [enc setBytes:&pm_nax length:sizeof(pm_nax) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((2 * D + 63) / 64),
                                                      1,
                                                      (NSUInteger)QO)
                    threadsPerThreadgroup:MTLSizeMake(4 * 32, 1, 1)];
                ++n_dispatches;

                MsgBlendNormPrm pbn{B, QO, QN, MOFF, eps};
                [enc setComputePipelineState:ctx->pso_msg_blend_norm];
                [enc setBuffer:ctx->scratch         offset:ctx->off_msg_o       atIndex:0];
                [enc setBuffer:msg_b                offset:msg_o                atIndex:1];
                [enc setBuffer:ctx->scratch         offset:ctx->off_msg_gate_in atIndex:2];
                [enc setBuffer:nw_b                 offset:nw_o                 atIndex:3];
                [enc setBuffer:out_b                offset:out_o                atIndex:4];
                [enc setBytes:&pbn length:sizeof(pbn) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)QO, (NSUInteger)B, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)ctx->mix_tg, 1, 1)];
                ++n_dispatches;
                return;
            }

            if (gate && use_batch_msg) {
                MixGateP1Prm p1{B, Q, QO, MOFF, add ? 1 : 0, 0};
                [enc setComputePipelineState:get_mix_gate_p1_pso(ctx, conv, K)];
                [enc setBuffer:x_b                  offset:x_o                  atIndex:0];
                [enc setBuffer:blk_b                offset:blk_o                atIndex:1];
                [enc setBuffer:qry_b                offset:qry_o                atIndex:2];
                [enc setBuffer:st_b                 offset:st_o                 atIndex:3];
                [enc setBuffer:tap_b                offset:tap_o                atIndex:4];
                [enc setBuffer:res_b                offset:res_o                atIndex:5];
                [enc setBuffer:msg_b                offset:msg_o                atIndex:6];
                [enc setBuffer:ctx->scratch         offset:ctx->off_msg_o       atIndex:7];
                [enc setBuffer:ctx->scratch         offset:ctx->off_msg_gate_in atIndex:8];
                [enc setBuffer:part_b               offset:part_o               atIndex:9];
                [enc setBuffer:resid_b              offset:resid_o              atIndex:10];
                [enc setBuffer:nst_b                offset:nst_o                atIndex:11];
                [enc setBytes:&p1 length:sizeof(p1) atIndex:12];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)QO, (NSUInteger)B, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)ctx->mix_tg, 1, 1)];
                ++n_dispatches;

                MsgDownPrm pd{B};
                [enc setComputePipelineState:ctx->pso_msg_down_batched];
                [enc setBuffer:ctx->scratch         offset:ctx->off_msg_gate_in atIndex:0];
                [enc setBuffer:dn_b                 offset:dn_o                 atIndex:1];
                [enc setBuffer:ctx->scratch         offset:ctx->off_msg_low     atIndex:2];
                [enc setBytes:&pd length:sizeof(pd) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((RANK + 7) / 8),
                                                      (NSUInteger)QO,
                                                      (NSUInteger)((B + 15) / 16))
                    threadsPerThreadgroup:MTLSizeMake(8 * 32, 1, 1)];
                ++n_dispatches;

                MsgFgPrm pf{B, QO, MOFF};
                [enc setComputePipelineState:ctx->pso_msg_fg_blend];
                [enc setBuffer:ctx->scratch         offset:ctx->off_msg_low     atIndex:0];
                [enc setBuffer:fg_b                 offset:fg_o                 atIndex:1];
                [enc setBuffer:gn_b                 offset:gn_o                 atIndex:2];
                [enc setBuffer:msg_b                offset:msg_o                atIndex:3];
                [enc setBuffer:ctx->scratch         offset:ctx->off_msg_o       atIndex:4];
                [enc setBytes:&pf length:sizeof(pf) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((D + 31) / 32), (NSUInteger)QO, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
                ++n_dispatches;

                RmsMultiPrm pm{QO, QN, eps};
                [enc setComputePipelineState:ctx->pso_rms_multi];
                [enc setBuffer:ctx->scratch         offset:ctx->off_msg_o       atIndex:0];
                [enc setBuffer:nw_b                 offset:nw_o                 atIndex:1];
                [enc setBuffer:out_b                offset:out_o                atIndex:2];
                [enc setBytes:&pm length:sizeof(pm) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)QO, (NSUInteger)B, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)ctx->mix_tg, 1, 1)];
                ++n_dispatches;
                return;
            }

            MixPrm prm{B, Q, QO, QN, MOFF, add ? 1 : 0, norm ? 1 : 0, 0, eps};
            [enc setComputePipelineState:get_mix_pso(ctx, gate, conv, K)];
            [enc setBuffer:x_b     offset:x_o     atIndex:0];
            [enc setBuffer:blk_b   offset:blk_o   atIndex:1];
            [enc setBuffer:qry_b   offset:qry_o   atIndex:2];
            [enc setBuffer:st_b    offset:st_o    atIndex:3];
            [enc setBuffer:tap_b   offset:tap_o   atIndex:4];
            [enc setBuffer:res_b   offset:res_o   atIndex:5];
            [enc setBuffer:msg_b   offset:msg_o   atIndex:6];
            [enc setBuffer:dn_b    offset:dn_o    atIndex:7];
            [enc setBuffer:fg_b    offset:fg_o    atIndex:8];
            [enc setBuffer:gn_b    offset:gn_o    atIndex:9];
            [enc setBuffer:nw_b    offset:nw_o    atIndex:10];
            [enc setBuffer:out_b   offset:out_o   atIndex:11];
            [enc setBuffer:part_b  offset:part_o  atIndex:12];
            [enc setBuffer:resid_b offset:resid_o atIndex:13];
            [enc setBuffer:nst_b   offset:nst_o   atIndex:14];
            [enc setBytes:&prm length:sizeof(prm) atIndex:15];
            if (gate) {
                size_t tg_len = align_up((size_t)D * sizeof(float), 16);
                [enc setThreadgroupMemoryLength:tg_len atIndex:0];
                [enc setThreadgroupMemoryLength:tg_len atIndex:1];
            } else {
                [enc setThreadgroupMemoryLength:16 atIndex:0];
                [enc setThreadgroupMemoryLength:16 atIndex:1];
            }
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)QO, (NSUInteger)B, 1)
                threadsPerThreadgroup:MTLSizeMake((NSUInteger)ctx->mix_tg, 1, 1)];
            ++n_dispatches;
        };

        auto encode_linear = [&](
            id<MTLBuffer> x_b, size_t x_o,
            id<MTLBuffer> w_b, size_t w_o,
            id<MTLBuffer> out_b, size_t out_o,
            int S, int K, int N, int AT3, int AT4)
        {
            if (!use_nax_qkv) {
                const int sg = (N >= 4096) ? 8 : 4;
                const int ny = (N + sg - 1) / sg;
                QkvPrm prm{S, N, AT3, AT4};
                [enc setComputePipelineState:get_qkv_pso(ctx, B, sg, K)];
                [enc setBuffer:x_b   offset:x_o   atIndex:0];
                [enc setBuffer:w_b   offset:w_o   atIndex:1];
                [enc setBuffer:out_b offset:out_o atIndex:2];
                [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)ny, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)sg, 1)];
                ++n_dispatches;
            } else {
                constexpr int bn = 64;
                const int nt0 = (S > 1) ? ((AT3 + bn - 1) / bn) : ((N + bn - 1) / bn);
                const int nt1 = (S > 1) ? (((AT4 - AT3) + bn - 1) / bn) : 0;
                const int nt2 = (S > 1) ? (((N - AT4) + bn - 1) / bn) : 0;
                const int total_nt = nt0 + nt1 + nt2;
                NaxQkvPrm prm{B, S, K, N, AT3, AT4, nt0, nt1, nt2};
                [enc setComputePipelineState:(S == 1 ? pso_nax_o : pso_nax_qkv)];
                [enc setBuffer:x_b   offset:x_o   atIndex:0];
                [enc setBuffer:w_b   offset:w_o   atIndex:1];
                [enc setBuffer:out_b offset:out_o atIndex:2];
                [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)total_nt, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(4 * 32, 1, 1)];
                ++n_dispatches;
            }
        };

        {
            EmbedBatchedPrm eprm{};
            eprm.eps = ctx->globals.embed_norm_eps;
            for (int i = 0; i < B; ++i) eprm.toks[i] = token_ids_ptr[i];
            [enc setComputePipelineState:ctx->pso_embed_batched];
            [enc setBuffer:ctx->globals.embed_tokens.buf offset:ctx->globals.embed_tokens.offset atIndex:0];
            [enc setBuffer:ctx->globals.embed_norm.buf   offset:ctx->globals.embed_norm.offset   atIndex:1];
            [enc setBuffer:ctx->scratch                  offset:ctx->off_x                       atIndex:2];
            [enc setBytes:&eprm length:sizeof(eprm) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)B, 1)
                threadsPerThreadgroup:MTLSizeMake((NSUInteger)ctx->mix_tg, 1, 1)];
            ++n_dispatches;
        }

        int cur_k = 0;
        int pre_loop_k = 0;
        int bank_k = 0;
        const size_t block_stride_bytes = (size_t)B * (size_t)D * sizeof(float);

        for (int step = 0; step < NV; ++step) {
            const VisitEntry& ve = ctx->schedule[step];
            const int layer_idx = ve.layer_idx;
            const int pass_idx = ve.pass_idx;
            const LayerDeviceWeights& L = ctx->layers[layer_idx];
            const InklingBatchKVMeta& km = kv_meta[step];
            const InklingCtx::BatchVisitPlan& P = plans[step];

            if (layer_idx == ctx->cfg.loop_start) {
                if (pass_idx == 0) {
                    dispatch_copy_bf16(ctx->scratch, ctx->off_x, ctx->scratch, ctx->off_pre_loop_x, (uint32_t)(B * D));
                    pre_loop_k = cur_k;
                } else {
                    int done = pre_loop_k;
                    bank_k = std::max(0, cur_k - (done + 1));
                    if (bank_k > 0) {
                        dispatch_copy_f32(
                            ctx->scratch, ctx->off_blocks + (size_t)(done + 1) * block_stride_bytes,
                            ctx->scratch, ctx->off_bank_blocks,
                            (uint32_t)((size_t)bank_k * (size_t)B * (size_t)D));
                    }
                    dispatch_copy_bf16(ctx->scratch, ctx->off_x, ctx->scratch, ctx->off_bank_x, (uint32_t)(B * D));
                    dispatch_copy_bf16(ctx->scratch, ctx->off_pre_loop_x, ctx->scratch, ctx->off_x, (uint32_t)(B * D));
                    cur_k = pre_loop_k;
                }
            }

            const bool gated = (pass_idx > 0);
            if (gated) {
                size_t mq_off = ctx->globals.message_queries.offset
                              + (size_t)(layer_idx - ctx->cfg.loop_start) * 4 * (size_t)D * sizeof(float);
                MixQ4Prm m4prm{B};
                [enc setComputePipelineState:get_mix_q4_pso(ctx, bank_k)];
                [enc setBuffer:ctx->scratch                  offset:ctx->off_bank_x      atIndex:0];
                [enc setBuffer:ctx->scratch                  offset:ctx->off_bank_blocks atIndex:1];
                [enc setBuffer:ctx->globals.message_queries.buf offset:mq_off            atIndex:2];
                [enc setBuffer:ctx->scratch                  offset:ctx->off_messages    atIndex:3];
                [enc setBytes:&m4prm length:sizeof(m4prm) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)B, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)ctx->mix_tg, 1, 1)];
                ++n_dispatches;
            }

            size_t normed_dst_off = L.is_block_boundary
                                  ? (ctx->off_blocks + (size_t)cur_k * block_stride_bytes)
                                  : ctx->off_scratch_partial;
            int qkv_s = 1;
            if (!L.has_attn_input) {
                Layer0Prm l0{L.attn_eps};
                [enc setComputePipelineState:ctx->pso_layer0_rms];
                [enc setBuffer:ctx->scratch      offset:ctx->off_x          atIndex:0];
                [enc setBuffer:L.attn_norm.buf   offset:L.attn_norm.offset  atIndex:1];
                [enc setBuffer:ctx->scratch      offset:normed_dst_off      atIndex:2];
                [enc setBuffer:ctx->scratch      offset:ctx->off_qkv_in     atIndex:3];
                [enc setBytes:&l0 length:sizeof(l0) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)B, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)ctx->mix_tg, 1, 1)];
                ++n_dispatches;
                qkv_s = 1;
            } else {
                int Q = L.is_sliding ? 1 : 3;
                int QO = gated ? 3 : Q;
                int QN = L.is_sliding ? 1 : 3;
                encode_mix(
                    ctx->scratch, ctx->off_x,
                    ctx->scratch, ctx->off_blocks,
                    L.attn_query.buf, L.attn_query.offset,
                    ctx->zero_buf, 0,
                    ctx->zero_buf, 0,
                    ctx->zero_buf, 0,
                    ctx->scratch, ctx->off_messages,
                    L.attn_gate_down.buf, L.attn_gate_down.offset,
                    L.attn_gate_forget.buf, L.attn_gate_forget.offset,
                    L.attn_gate_gain.buf, L.attn_gate_gain.offset,
                    L.attn_gate_fg_bf16.buf, L.attn_gate_fg_bf16.offset, L.has_attn_gate_fg_bf16,
                    L.attn_norm.buf, L.attn_norm.offset,
                    ctx->scratch, ctx->off_qkv_in,
                    ctx->scratch, normed_dst_off,
                    ctx->scratch, ctx->off_resid,
                    ctx->zero_buf, 0,
                    cur_k, Q, QO, QN, 0,
                    false, false, gated, true, L.attn_eps);
                qkv_s = QO;
            }

            encode_linear(
                ctx->scratch, ctx->off_qkv_in,
                L.stack.buf, L.stack.offset,
                ctx->scratch, ctx->off_qkv,
                qkv_s, D, L.qkv_total_rows, L.qkv_at[3], L.qkv_at[4]);

            const size_t meta_off = (size_t)step * ctx->batch_visit_meta_stride_bytes;
            {
                int tg = std::max(32, ((L.head_dim + 31) / 32) * 32);
                KvPrepBatchedPrm kprm{};
                kprm.M = B;
                kprm.HQ = L.n_heads;
                kprm.HKV = L.n_kv;
                kprm.R = L.rel_extent;
                kprm.QKV_W = L.qkv_total_rows;
                kprm.QOFF = L.qkv_at[0];
                kprm.ROFF = L.qkv_at[1];
                kprm.KOFF = L.qkv_at[3];
                kprm.VOFF = L.qkv_at[4];
                kprm.X_STRIDE = qkv_s * D;
                kprm.XK_OFF = (qkv_s == 3) ? D : 0;
                kprm.XV_OFF = (qkv_s == 3) ? (2 * D) : 0;
                kprm.TAU = P.use_tau ? 1 : 0;
                kprm.WRITE_SLOT = P.write_slot;
                kprm.STRIDE_B = km.stride_b;
                kprm.STRIDE_H = km.stride_h;
                kprm.STRIDE_S = km.stride_s;
                kprm.qk_thr = ctx->cfg.qk_l2_clip_threshold;
                kprm.rel_thr = ctx->cfg.rel_l2_clip_threshold;
                kprm.eps = ctx->cfg.rms_norm_eps;

                [enc setComputePipelineState:get_kv_prep_batched_pso(ctx, !L.is_sliding, L.head_dim)];
                [enc setBuffer:ctx->scratch            offset:ctx->off_qkv               atIndex:0];
                [enc setBuffer:ctx->scratch            offset:ctx->off_qkv_in            atIndex:1];
                [enc setBuffer:L.sink_w.buf            offset:L.sink_w.offset            atIndex:2];
                [enc setBuffer:L.kdyn_w.buf            offset:L.kdyn_w.offset            atIndex:3];
                [enc setBuffer:L.vdyn_w.buf            offset:L.vdyn_w.offset            atIndex:4];
                [enc setBuffer:conv_bufs[step*4+0].buf offset:conv_bufs[step*4+0].offset atIndex:5];
                [enc setBuffer:conv_bufs[step*4+1].buf offset:conv_bufs[step*4+1].offset atIndex:6];
                [enc setBuffer:L.k_taps.buf            offset:L.k_taps.offset            atIndex:7];
                [enc setBuffer:L.v_taps.buf            offset:L.v_taps.offset            atIndex:8];
                [enc setBuffer:L.q_scale.buf           offset:L.q_scale.offset           atIndex:9];
                [enc setBuffer:L.rel_scale.buf         offset:L.rel_scale.offset         atIndex:10];
                [enc setBuffer:L.rel_proj.buf          offset:L.rel_proj.offset          atIndex:11];
                [enc setBuffer:ctx->scratch            offset:ctx->off_batch_tau + meta_off atIndex:12];
                [enc setBuffer:ctx->scratch            offset:ctx->off_qo                atIndex:13];
                [enc setBuffer:ctx->scratch            offset:ctx->off_bias              atIndex:14];
                [enc setBuffer:kv_k_bufs[step].buf     offset:kv_k_bufs[step].offset     atIndex:15];
                [enc setBuffer:kv_v_bufs[step].buf     offset:kv_v_bufs[step].offset     atIndex:16];
                [enc setBuffer:ctx->scratch            offset:ctx->off_sinks             atIndex:17];
                [enc setBytes:&kprm length:sizeof(kprm) atIndex:18];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)L.n_heads, (NSUInteger)B, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)tg, 1, 1)];
                ++n_dispatches;
            }

            const int win_tg = window_tg(L.head_dim);
            const float attn_scale = 1.0f / std::sqrt((float)L.head_dim);
            const int G = L.n_heads / L.n_kv;
            if (!P.use_splitk) {
                AttendBatchedPrm aprm{B, L.n_heads, L.n_kv, L.rel_extent,
                                      L.qkv_total_rows, L.qkv_at[2], L.sliding_window,
                                      P.new_s_len, P.ring_idx, km.stride_b, km.stride_h, km.stride_s,
                                      attn_scale};
                [enc setComputePipelineState:get_attend_batched_pso(ctx, G, L.head_dim)];
                [enc setBuffer:ctx->scratch        offset:ctx->off_qo                    atIndex:0];
                [enc setBuffer:kv_k_bufs[step].buf offset:kv_k_bufs[step].offset         atIndex:1];
                [enc setBuffer:kv_v_bufs[step].buf offset:kv_v_bufs[step].offset         atIndex:2];
                [enc setBuffer:ctx->scratch        offset:ctx->off_bias                  atIndex:3];
                [enc setBuffer:ctx->scratch        offset:ctx->off_sinks                 atIndex:4];
                [enc setBuffer:ctx->scratch        offset:ctx->off_qkv                   atIndex:5];
                [enc setBuffer:ctx->scratch        offset:ctx->off_batch_counts + meta_off atIndex:6];
                [enc setBuffer:ctx->scratch        offset:ctx->off_att                   atIndex:7];
                [enc setBytes:&aprm length:sizeof(aprm) atIndex:8];
                [enc setThreadgroupMemoryLength:align_up((size_t)G * (size_t)win_tg * sizeof(float), 16) atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)L.n_kv, (NSUInteger)B, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)win_tg, 1, 1)];
                ++n_dispatches;
            } else {
                DecodePartBatchedPrm dprm{B, L.n_heads, L.n_kv, L.rel_extent, P.bs, P.nblocks,
                                          L.sliding_window, P.new_s_len, P.ring_idx,
                                          km.stride_b, km.stride_h, km.stride_s, attn_scale};
                [enc setComputePipelineState:get_dpart_batched_pso(ctx, G, L.head_dim)];
                [enc setBuffer:ctx->scratch        offset:ctx->off_qo                    atIndex:0];
                [enc setBuffer:kv_k_bufs[step].buf offset:kv_k_bufs[step].offset         atIndex:1];
                [enc setBuffer:kv_v_bufs[step].buf offset:kv_v_bufs[step].offset         atIndex:2];
                [enc setBuffer:ctx->scratch        offset:ctx->off_bias                  atIndex:3];
                [enc setBuffer:ctx->scratch        offset:ctx->off_batch_counts + meta_off atIndex:4];
                [enc setBuffer:ctx->part_buf       offset:ctx->part_off_max              atIndex:5];
                [enc setBuffer:ctx->part_buf       offset:ctx->part_off_sum              atIndex:6];
                [enc setBuffer:ctx->part_buf       offset:ctx->part_off_acc              atIndex:7];
                [enc setBytes:&dprm length:sizeof(dprm) atIndex:8];
                [enc setThreadgroupMemoryLength:align_up(8 * (size_t)G * (size_t)L.head_dim * sizeof(float), 16) atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)P.nblocks, (NSUInteger)L.n_kv, (NSUInteger)B)
                    threadsPerThreadgroup:MTLSizeMake(8 * 32, 1, 1)];
                ++n_dispatches;

                DecodeCombPrm cprm{L.n_heads, L.head_dim, P.nblocks, L.qkv_total_rows, L.qkv_at[2]};
                [enc setComputePipelineState:ctx->pso_decode_comb];
                [enc setBuffer:ctx->part_buf offset:ctx->part_off_max atIndex:0];
                [enc setBuffer:ctx->part_buf offset:ctx->part_off_sum atIndex:1];
                [enc setBuffer:ctx->part_buf offset:ctx->part_off_acc atIndex:2];
                [enc setBuffer:ctx->scratch  offset:ctx->off_sinks    atIndex:3];
                [enc setBuffer:ctx->scratch  offset:ctx->off_qkv      atIndex:4];
                [enc setBuffer:ctx->scratch  offset:ctx->off_att      atIndex:5];
                [enc setBytes:&cprm length:sizeof(cprm) atIndex:6];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)L.n_heads, (NSUInteger)B, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)L.head_dim, 1, 1)];
                ++n_dispatches;
            }

            encode_linear(
                ctx->scratch, ctx->off_att,
                L.o_proj.buf, L.o_proj.offset,
                ctx->scratch, ctx->off_att_out,
                1, L.n_heads * L.head_dim, D, 0, 0);

            if (L.is_block_boundary) {
                cur_k += 1;
            }
            encode_mix(
                ctx->scratch, ctx->off_att_out,
                ctx->scratch, ctx->off_blocks,
                L.mlp_query.buf, L.mlp_query.offset,
                conv_bufs[step*4+2].buf, conv_bufs[step*4+2].offset,
                L.attn_taps.buf, L.attn_taps.offset,
                L.is_block_boundary ? ctx->zero_buf : ctx->scratch, L.is_block_boundary ? 0 : ctx->off_x,
                ctx->scratch, ctx->off_messages,
                L.mlp_gate_down.buf, L.mlp_gate_down.offset,
                L.mlp_gate_forget.buf, L.mlp_gate_forget.offset,
                L.mlp_gate_gain.buf, L.mlp_gate_gain.offset,
                L.mlp_gate_fg_bf16.buf, L.mlp_gate_fg_bf16.offset, L.has_mlp_gate_fg_bf16,
                L.mlp_norm.buf, L.mlp_norm.offset,
                ctx->scratch, ctx->off_mlp_in,
                ctx->scratch, ctx->off_scratch_partial,
                ctx->scratch, ctx->off_resid,
                conv_bufs[step*4+2].buf, conv_bufs[step*4+2].offset,
                cur_k, 1, 1, 1, 3,
                true, !L.is_block_boundary, gated, true, L.mlp_eps);

            if (use_nax_rt_b && L.has_router_w_bf16) {
                NaxMsgPrm rprm_nax{B};
                [enc setComputePipelineState:ctx->pso_nax_router_16x32];
                [enc setBuffer:ctx->scratch        offset:ctx->off_mlp_in       atIndex:0];
                [enc setBuffer:L.router_w_bf16.buf offset:L.router_w_bf16.offset atIndex:1];
                [enc setBuffer:ctx->scratch        offset:ctx->off_route_scores atIndex:2];
                [enc setBytes:&rprm_nax length:sizeof(rprm_nax) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((E + 31) / 32),
                                                      (NSUInteger)((B + 15) / 16),
                                                      1)
                    threadsPerThreadgroup:MTLSizeMake(4 * 32, 1, 1)];
                ++n_dispatches;

                RouteBatchedPrm rprm{B, ctx->cfg.route_scale};
                [enc setComputePipelineState:ctx->pso_route_topk];
                [enc setBuffer:ctx->scratch   offset:ctx->off_route_scores atIndex:0];
                [enc setBuffer:L.router_b.buf offset:L.router_b.offset     atIndex:1];
                [enc setBuffer:ctx->scratch   offset:ctx->off_inds         atIndex:2];
                [enc setBuffer:ctx->scratch   offset:ctx->off_weights      atIndex:3];
                [enc setBytes:&rprm length:sizeof(rprm) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)B, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)ctx->bucket_tg, 1, 1)];
                ++n_dispatches;
            } else if (!use_batch_rt) {
                RoutePrm rprm{ctx->cfg.route_scale};
                [enc setComputePipelineState:ctx->pso_route];
                [enc setBuffer:ctx->scratch   offset:ctx->off_mlp_in   atIndex:0];
                [enc setBuffer:L.router_w.buf offset:L.router_w.offset atIndex:1];
                [enc setBuffer:L.router_b.buf offset:L.router_b.offset atIndex:2];
                [enc setBuffer:ctx->scratch   offset:ctx->off_inds     atIndex:3];
                [enc setBuffer:ctx->scratch   offset:ctx->off_weights  atIndex:4];
                [enc setBytes:&rprm length:sizeof(rprm) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)B, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)(E * ctx->route_parts), 1, 1)];
                ++n_dispatches;
            } else {
                RouteBatchedPrm rprm{B, ctx->cfg.route_scale};
                [enc setComputePipelineState:ctx->pso_route_scores];
                [enc setBuffer:ctx->scratch   offset:ctx->off_mlp_in       atIndex:0];
                [enc setBuffer:L.router_w.buf offset:L.router_w.offset     atIndex:1];
                [enc setBuffer:ctx->scratch   offset:ctx->off_route_scores atIndex:2];
                [enc setBytes:&rprm length:sizeof(rprm) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)E, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_dispatches;

                [enc setComputePipelineState:ctx->pso_route_topk];
                [enc setBuffer:ctx->scratch   offset:ctx->off_route_scores atIndex:0];
                [enc setBuffer:L.router_b.buf offset:L.router_b.offset     atIndex:1];
                [enc setBuffer:ctx->scratch   offset:ctx->off_inds         atIndex:2];
                [enc setBuffer:ctx->scratch   offset:ctx->off_weights      atIndex:3];
                [enc setBytes:&rprm length:sizeof(rprm) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)B, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)ctx->bucket_tg, 1, 1)];
                ++n_dispatches;
            }

            const bool is_nvfp4 = (L.moe_mode == 0);
            MoePrm mprm{ctx->cfg.mlp_gate_softcap, ctx->cfg.mlp_up_softcap};
            if (B <= 2) {
                const int sg13 = is_nvfp4 ? 8 : 4;
                const int sg2  = 8;
                const int nx13 = (INTER + sg13 * 4 - 1) / (sg13 * 4);
                const int nx2  = (D + sg2 * 4 - 1) / (sg2 * 4);
                if (is_nvfp4) {
                    [enc setComputePipelineState:ctx->pso_m1_w13_nvfp4];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_mlp_in     atIndex:0];
                    [enc setBuffer:L.w13_weight.buf offset:L.w13_weight.offset atIndex:1];
                    [enc setBuffer:L.w13_scales.buf offset:L.w13_scales.offset atIndex:2];
                    [enc setBuffer:L.w13_gscale.buf offset:L.w13_gscale.offset atIndex:3];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_inds       atIndex:4];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_moe_h      atIndex:5];
                    [enc setBytes:&mprm length:sizeof(mprm) atIndex:6];
                    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nx13, (NSUInteger)K_TOP, (NSUInteger)B)
                        threadsPerThreadgroup:MTLSizeMake((NSUInteger)(32 * sg13), 1, 1)];
                    ++n_dispatches;

                    [enc setComputePipelineState:ctx->pso_m1_w2_nvfp4];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_moe_h     atIndex:0];
                    [enc setBuffer:L.w2_weight.buf offset:L.w2_weight.offset atIndex:1];
                    [enc setBuffer:L.w2_scales.buf offset:L.w2_scales.offset atIndex:2];
                    [enc setBuffer:L.w2_gscale.buf offset:L.w2_gscale.offset atIndex:3];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_inds      atIndex:4];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_ys        atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nx2, (NSUInteger)K_TOP, (NSUInteger)B)
                        threadsPerThreadgroup:MTLSizeMake((NSUInteger)(32 * sg2), 1, 1)];
                    ++n_dispatches;
                } else {
                    [enc setComputePipelineState:ctx->pso_m1_w13_mxfp8];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_mlp_in     atIndex:0];
                    [enc setBuffer:L.w13_weight.buf offset:L.w13_weight.offset atIndex:1];
                    [enc setBuffer:L.w13_scales.buf offset:L.w13_scales.offset atIndex:2];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_inds       atIndex:3];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_moe_h      atIndex:4];
                    [enc setBytes:&mprm length:sizeof(mprm) atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nx13, (NSUInteger)K_TOP, (NSUInteger)B)
                        threadsPerThreadgroup:MTLSizeMake((NSUInteger)(32 * sg13), 1, 1)];
                    ++n_dispatches;

                    [enc setComputePipelineState:ctx->pso_m1_w2_mxfp8];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_moe_h     atIndex:0];
                    [enc setBuffer:L.w2_weight.buf offset:L.w2_weight.offset atIndex:1];
                    [enc setBuffer:L.w2_scales.buf offset:L.w2_scales.offset atIndex:2];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_inds      atIndex:3];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_ys        atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nx2, (NSUInteger)K_TOP, (NSUInteger)B)
                        threadsPerThreadgroup:MTLSizeMake((NSUInteger)(32 * sg2), 1, 1)];
                    ++n_dispatches;
                }
            } else {
                const int t_slots = B * K_TOP;
                const int sg13 = is_nvfp4 ? sg_grp_nv : sg_grp_mx;
                const int sg2  = is_nvfp4 ? sg_grp_nv : sg_grp_mx;
                const int nx13 = (INTER + sg13 * 4 - 1) / (sg13 * 4);
                const int nx2  = (D + sg2 * 4 - 1) / (sg2 * 4);
                BucketTilesPrm bprm{t_slots, grp_mt, nx13, nx2};
                [enc setComputePipelineState:ctx->pso_bucket_tiles];
                [enc setBuffer:ctx->scratch offset:ctx->off_inds          atIndex:0];
                [enc setBuffer:ctx->scratch offset:ctx->off_offsets       atIndex:1];
                [enc setBuffer:ctx->scratch offset:ctx->off_sorted_slots  atIndex:2];
                [enc setBuffer:ctx->scratch offset:ctx->off_tile_expert   atIndex:3];
                [enc setBuffer:ctx->scratch offset:ctx->off_tile_t0       atIndex:4];
                [enc setBuffer:ctx->scratch offset:ctx->off_indirect_args atIndex:5];
                [enc setBytes:&bprm length:sizeof(bprm) atIndex:6];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)ctx->bucket_tg, 1, 1)];
                ++n_dispatches;

                if (is_nvfp4) {
                    [enc setComputePipelineState:pso_grp_w13_nv];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_mlp_in       atIndex:0];
                    [enc setBuffer:L.w13_weight.buf offset:L.w13_weight.offset   atIndex:1];
                    [enc setBuffer:L.w13_scales.buf offset:L.w13_scales.offset   atIndex:2];
                    [enc setBuffer:L.w13_gscale.buf offset:L.w13_gscale.offset   atIndex:3];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_offsets      atIndex:4];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_sorted_slots atIndex:5];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_tile_expert  atIndex:6];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_tile_t0      atIndex:7];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_moe_h        atIndex:8];
                    [enc setBytes:&mprm length:sizeof(mprm) atIndex:9];
                    [enc dispatchThreadgroupsWithIndirectBuffer:ctx->scratch
                                           indirectBufferOffset:ctx->off_indirect_args
                                          threadsPerThreadgroup:MTLSizeMake((NSUInteger)(32 * sg13), 1, 1)];
                    ++n_dispatches;

                    [enc setComputePipelineState:pso_grp_w2_nv];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_moe_h        atIndex:0];
                    [enc setBuffer:L.w2_weight.buf offset:L.w2_weight.offset    atIndex:1];
                    [enc setBuffer:L.w2_scales.buf offset:L.w2_scales.offset    atIndex:2];
                    [enc setBuffer:L.w2_gscale.buf offset:L.w2_gscale.offset    atIndex:3];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_offsets      atIndex:4];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_sorted_slots atIndex:5];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_tile_expert  atIndex:6];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_tile_t0      atIndex:7];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_ys           atIndex:8];
                    [enc dispatchThreadgroupsWithIndirectBuffer:ctx->scratch
                                           indirectBufferOffset:ctx->off_indirect_args + 16
                                          threadsPerThreadgroup:MTLSizeMake((NSUInteger)(32 * sg2), 1, 1)];
                    ++n_dispatches;
                } else {
                    [enc setComputePipelineState:pso_grp_w13_mx];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_mlp_in       atIndex:0];
                    [enc setBuffer:L.w13_weight.buf offset:L.w13_weight.offset   atIndex:1];
                    [enc setBuffer:L.w13_scales.buf offset:L.w13_scales.offset   atIndex:2];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_offsets      atIndex:3];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_sorted_slots atIndex:4];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_tile_expert  atIndex:5];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_tile_t0      atIndex:6];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_moe_h        atIndex:7];
                    [enc setBytes:&mprm length:sizeof(mprm) atIndex:8];
                    [enc dispatchThreadgroupsWithIndirectBuffer:ctx->scratch
                                           indirectBufferOffset:ctx->off_indirect_args
                                          threadsPerThreadgroup:MTLSizeMake((NSUInteger)(32 * sg13), 1, 1)];
                    ++n_dispatches;

                    [enc setComputePipelineState:pso_grp_w2_mx];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_moe_h        atIndex:0];
                    [enc setBuffer:L.w2_weight.buf offset:L.w2_weight.offset    atIndex:1];
                    [enc setBuffer:L.w2_scales.buf offset:L.w2_scales.offset    atIndex:2];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_offsets      atIndex:3];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_sorted_slots atIndex:4];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_tile_expert  atIndex:5];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_tile_t0      atIndex:6];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_ys           atIndex:7];
                    [enc dispatchThreadgroupsWithIndirectBuffer:ctx->scratch
                                           indirectBufferOffset:ctx->off_indirect_args + 16
                                          threadsPerThreadgroup:MTLSizeMake((NSUInteger)(32 * sg2), 1, 1)];
                    ++n_dispatches;
                }
            }

            {
                CombinePrm cbprm{B};
                int tg = std::min(256, D);
                int nx = (D + tg - 1) / tg;
                [enc setComputePipelineState:ctx->pso_combine_noseq];
                [enc setBuffer:ctx->scratch            offset:ctx->off_ys                atIndex:0];
                [enc setBuffer:ctx->scratch            offset:ctx->off_weights           atIndex:1];
                [enc setBuffer:conv_bufs[step*4+3].buf offset:conv_bufs[step*4+3].offset atIndex:2];
                [enc setBuffer:L.mlp_taps.buf          offset:L.mlp_taps.offset          atIndex:3];
                [enc setBuffer:ctx->scratch            offset:ctx->off_resid             atIndex:4];
                [enc setBuffer:ctx->scratch            offset:ctx->off_x                 atIndex:5];
                [enc setBuffer:conv_bufs[step*4+3].buf offset:conv_bufs[step*4+3].offset atIndex:6];
                [enc setBytes:&cbprm length:sizeof(cbprm) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nx, (NSUInteger)B, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)tg, 1, 1)];
                ++n_dispatches;
            }
        }

        encode_mix(
            ctx->scratch, ctx->off_x,
            ctx->scratch, ctx->off_blocks,
            ctx->globals.final_query.buf, ctx->globals.final_query.offset,
            ctx->zero_buf, 0,
            ctx->zero_buf, 0,
            ctx->zero_buf, 0,
            ctx->zero_buf, 0,
            ctx->zero_buf, 0,
            ctx->zero_buf, 0,
            ctx->zero_buf, 0,
            ctx->zero_buf, 0, false,
            ctx->zero_buf, 0,
            ctx->scratch, ctx->off_pre_norm,
            ctx->scratch, ctx->off_scratch_partial,
            ctx->scratch, ctx->off_resid,
            ctx->zero_buf, 0,
            cur_k, 1, 1, 1, 0,
            false, false, false, false, 1e-6f);

        if (want_argmax) {
            const int v_limit = ctx->cfg.unpadded_vocab_size;
            FinalNormPrm fprm{-1, 0, ctx->globals.final_norm_eps};
            int ntg = 0;

            if (!use_nax_head) {
                [enc setComputePipelineState:ctx->pso_final_norm];
                [enc setBuffer:ctx->scratch                offset:ctx->off_pre_norm           atIndex:0];
                [enc setBuffer:ctx->globals.final_norm.buf offset:ctx->globals.final_norm.offset atIndex:1];
                [enc setBuffer:ctx->scratch                offset:ctx->off_normed_f32         atIndex:2];
                [enc setBuffer:ctx->scratch                offset:ctx->off_captures           atIndex:3];
                [enc setBytes:&fprm length:sizeof(fprm) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)B, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)ctx->mix_tg, 1, 1)];
                ++n_dispatches;

                const int sg = 8;
                ntg = (v_limit + sg * argmax_r - 1) / (sg * argmax_r);
                ArgmaxP1Prm p1{ntg};
                [enc setComputePipelineState:pso_simt_argmax_p1];
                [enc setBuffer:ctx->scratch             offset:ctx->off_normed_f32         atIndex:0];
                [enc setBuffer:ctx->globals.lm_head.buf offset:ctx->globals.lm_head.offset atIndex:1];
                [enc setBuffer:ctx->scratch             offset:ctx->off_argmax_pv          atIndex:2];
                [enc setBuffer:ctx->scratch             offset:ctx->off_argmax_pi          atIndex:3];
                [enc setBytes:&p1 length:sizeof(p1) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)ntg, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)(sg * 32), 1, 1)];
                ++n_dispatches;
            } else {
                [enc setComputePipelineState:ctx->pso_final_norm_bf16];
                [enc setBuffer:ctx->scratch                offset:ctx->off_pre_norm           atIndex:0];
                [enc setBuffer:ctx->globals.final_norm.buf offset:ctx->globals.final_norm.offset atIndex:1];
                [enc setBuffer:ctx->scratch                offset:ctx->off_normed_bf16        atIndex:2];
                [enc setBytes:&fprm length:sizeof(fprm) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)B, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)ctx->mix_tg, 1, 1)];
                ++n_dispatches;

                const int ntg_m = (B + 15) / 16;
                ntg = (v_limit + 63) / 64;
                NaxArgmaxPrm p1{B, ntg};
                [enc setComputePipelineState:ctx->pso_nax_argmax_16x64];
                [enc setBuffer:ctx->scratch             offset:ctx->off_normed_bf16        atIndex:0];
                [enc setBuffer:ctx->globals.lm_head.buf offset:ctx->globals.lm_head.offset atIndex:1];
                [enc setBuffer:ctx->scratch             offset:ctx->off_argmax_pv          atIndex:2];
                [enc setBuffer:ctx->scratch             offset:ctx->off_argmax_pi          atIndex:3];
                [enc setBytes:&p1 length:sizeof(p1) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)ntg, (NSUInteger)ntg_m, 1)
                    threadsPerThreadgroup:MTLSizeMake(4 * 32, 1, 1)];
                ++n_dispatches;
            }

            ArgmaxP2Prm p2{ntg};
            [enc setComputePipelineState:ctx->pso_argmax_p2];
            [enc setBuffer:ctx->scratch offset:ctx->off_argmax_pv atIndex:0];
            [enc setBuffer:ctx->scratch offset:ctx->off_argmax_pi atIndex:1];
            [enc setBuffer:ctx->scratch offset:ctx->off_out_toks  atIndex:2];
            [enc setBytes:&p2 length:sizeof(p2) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)B, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            ++n_dispatches;
        }

        [enc endEncoding];
        uint64_t t_enc1 = mach_absolute_time();

        [cb commit];
        [cb waitUntilCompleted];
        uint64_t t_wall1 = mach_absolute_time();

        if (cb.status != MTLCommandBufferStatusCompleted) {
            fprintf(stderr, "[inkling_decode] ERROR: batched command buffer failed: %s\n",
                    cb.error ? cb.error.localizedDescription.UTF8String : "unknown");
            return -10;
        }

        for (int v = 0; v < NV; ++v) {
            kv_meta[v].idx = plans[v].new_idx;
            kv_meta[v].offset = plans[v].new_off_scalar;
            int32_t* row_off = kv_row_offsets_ptr + (size_t)v * (size_t)B;
            for (int32_t b = 0; b < B; ++b) {
                row_off[b] += 1;
            }
        }

        const uint8_t* sptr = (const uint8_t*)[ctx->scratch contents];
        if (out_pre_norm_ptr) {
            memcpy(out_pre_norm_ptr, sptr + ctx->off_pre_norm, (size_t)B * (size_t)D * sizeof(float));
        }
        if (out_toks_ptr) {
            memcpy(out_toks_ptr, sptr + ctx->off_out_toks, (size_t)B * sizeof(int32_t));
        }

        if (stats_out) {
            stats_out->encode_ms    = mach_to_ms(t_enc1 - t_enc0, ctx->tb);
            stats_out->gpu_ms       = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
            stats_out->wall_ms      = mach_to_ms(t_wall1 - t_wall0, ctx->tb);
            stats_out->n_dispatches = n_dispatches;
        }
        return 0;
    }
}

static id<MTLComputePipelineState> make_mtp_pso(
    id<MTLDevice> dev,
    id<MTLLibrary> lib,
    NSString* name,
    int fc_m = -1,
    int fc_k = -1,
    int fc_sg = -1,
    int fc_has_res = -1,
    int fc_has_embed_norm = -1)
{
    MTLFunctionConstantValues* fcv = nil;
    if (fc_m >= 0 || fc_k >= 0 || fc_sg >= 0 || fc_has_res >= 0 || fc_has_embed_norm >= 0) {
        fcv = [MTLFunctionConstantValues new];
        if (fc_m >= 0)              [fcv setConstantValue:&fc_m              type:MTLDataTypeInt atIndex:0];
        if (fc_k >= 0)              [fcv setConstantValue:&fc_k              type:MTLDataTypeInt atIndex:1];
        if (fc_sg >= 0)             [fcv setConstantValue:&fc_sg             type:MTLDataTypeInt atIndex:2];
        if (fc_has_res >= 0)        [fcv setConstantValue:&fc_has_res        type:MTLDataTypeInt atIndex:3];
        if (fc_has_embed_norm >= 0) [fcv setConstantValue:&fc_has_embed_norm type:MTLDataTypeInt atIndex:4];
    }
    return make_pso(dev, lib, name, fcv);
}

static id<MTLComputePipelineState> get_head_logits_max_p1_pso(InklingCtx* ctx, int m) {
    if (m < 1 || m > 8) return nil;
    if (ctx->pso_head_logits_max_p1[m]) return ctx->pso_head_logits_max_p1[m];
    MTLFunctionConstantValues* fcv = [MTLFunctionConstantValues new];
    [fcv setConstantValue:&m type:MTLDataTypeInt atIndex:4];
    id<MTLComputePipelineState> pso = make_pso(ctx->device, ctx->lib, @"inkling_head_logits_max_p1", fcv);
    ctx->pso_head_logits_max_p1[m] = pso;
    return pso;
}

int inkling_ctx_attach_mtp(
    InklingCtx*                  ctx,
    const MTPDraftConfig*        cfg,
    const MTPDraftGlobalWeights* gw,
    const MTPDraftLayerWeights*  layers,
    const int32_t*               capture_layer_ids,
    int32_t                      n_capture_layers)
{
    @autoreleasepool {
        if (!ctx || !ctx->globals.registered || !cfg || !gw || !layers || !capture_layer_ids) {
            return -1;
        }
        if (cfg->hidden_size != ctx->cfg.hidden_size ||
            cfg->unpadded_vocab_size != ctx->cfg.unpadded_vocab_size ||
            cfg->vocab_size < cfg->unpadded_vocab_size ||
            (cfg->hidden_size % 32) != 0 || cfg->hidden_size < 32 ||
            cfg->num_layers <= 0 || cfg->num_layers > 8 ||
            cfg->num_heads <= 0 || cfg->num_kv_heads <= 0 ||
            (cfg->num_heads % cfg->num_kv_heads) != 0 ||
            (cfg->num_heads / cfg->num_kv_heads) > 8 ||
            cfg->head_dim < 32 || cfg->head_dim > 128 || (cfg->head_dim % 32) != 0 ||
            cfg->inter_size <= 0 || (cfg->inter_size % 4) != 0 ||
            cfg->block_size < 2 || cfg->block_size > 8 || cfg->block_size > ctx->cfg.max_m ||
            cfg->markov_rank < 32 || (cfg->markov_rank % 32) != 0) {
            fprintf(stderr, "[inkling_decode] ERROR: invalid MTPDraftConfig for attach_mtp\n");
            return -2;
        }
        const int32_t NV = (int32_t)ctx->schedule.size();
        if (n_capture_layers != cfg->n_target_layers || n_capture_layers <= 0 ||
            n_capture_layers > 8 || n_capture_layers > ctx->max_captures) {
            fprintf(stderr, "[inkling_decode] ERROR: invalid n_capture_layers=%d\n", n_capture_layers);
            return -3;
        }
        for (int32_t i = 0; i < n_capture_layers; ++i) {
            if (capture_layer_ids[i] < 0 || capture_layer_ids[i] > NV) {
                fprintf(stderr, "[inkling_decode] ERROR: capture_layer_ids[%d]=%d out of [0,%d]\n",
                        i, capture_layer_ids[i], NV);
                return -3;
            }
        }

        if (!ctx->pso_embed_from_buf) {
            ctx->pso_embed_from_buf = make_pso(ctx->device, ctx->lib, @"inkling_embed_rms_norm_from_buf");
            ctx->pso_copy_conv4_dyn = make_pso(ctx->device, ctx->lib, @"inkling_copy_conv4_dyn");
            ctx->pso_copy_cap_dyn   = make_pso(ctx->device, ctx->lib, @"inkling_copy_capture_dyn_bf16");
            ctx->pso_kv_commit_dyn  = make_pso(ctx->device, ctx->lib, @"inkling_kv_commit_rotating_dyn");
            if (!ctx->pso_embed_from_buf || !ctx->pso_copy_conv4_dyn ||
                !ctx->pso_copy_cap_dyn || !ctx->pso_kv_commit_dyn) {
                return -4;
            }
        }
        for (int m = 2; m <= cfg->block_size; ++m) {
            if (!get_head_logits_max_p1_pso(ctx, m)) return -4;
        }

        ctx->mtp = MTPAttachedState{};
        MTPAttachedState& mp = ctx->mtp;
        mp.cfg = *cfg;
        mp.capture_layer_ids.assign(capture_layer_ids, capture_layer_ids + n_capture_layers);

        const size_t D = (size_t)cfg->hidden_size;
        const size_t H = (size_t)cfg->num_heads;
        const size_t KVH = (size_t)cfg->num_kv_heads;
        const size_t HD = (size_t)cfg->head_dim;
        const size_t INTER = (size_t)cfg->inter_size;
        const size_t B = (size_t)cfg->block_size;
        const size_t RANK = (size_t)cfg->markov_rank;
        const size_t NCAP = (size_t)cfg->n_target_layers;
        const size_t V_UNPAD = (size_t)cfg->unpadded_vocab_size;

        auto wrap_req = [&](const void* ptr, size_t nbytes, size_t min_bytes, const char* name) -> BufSlice {
            if (!ptr || nbytes < min_bytes) {
                fprintf(stderr, "[inkling_decode] ERROR: MTP weight '%s' missing or too small (%p, %zu < %zu)\n",
                        name, ptr, nbytes, min_bytes);
                return {nil, 0};
            }
            return wrap_static_ptr(ctx, ptr, nbytes);
        };

        if (!gw->embed_tokens || gw->embed_tokens == ctx->globals.embed_tokens_ptr) {
            mp.embed_tokens = ctx->globals.embed_tokens;
        } else {
            mp.embed_tokens = wrap_req(gw->embed_tokens, gw->embed_tokens_bytes, V_UNPAD * D * 2, "embed_tokens");
        }
        if (!gw->embed_norm) {
            mp.has_embed_norm = false;
            mp.embed_norm = {ctx->zero_buf, 0};
        } else if (gw->embed_norm == ctx->globals.embed_norm_ptr) {
            mp.has_embed_norm = true;
            mp.embed_norm = ctx->globals.embed_norm;
        } else {
            mp.has_embed_norm = true;
            mp.embed_norm = wrap_req(gw->embed_norm, gw->embed_norm_bytes, D * 2, "embed_norm");
        }
        if ((!gw->lm_head || gw->lm_head == ctx->globals.lm_head_ptr) &&
            ctx->globals.lm_head.buf != ctx->zero_buf) {
            mp.lm_head = ctx->globals.lm_head;
        } else {
            mp.lm_head = wrap_req(gw->lm_head, gw->lm_head_bytes, V_UNPAD * D * 2, "lm_head");
        }

        mp.mask_embed         = wrap_req(gw->mask_embed, gw->mask_embed_bytes, D * 2, "mask_embed");
        mp.final_norm         = wrap_req(gw->final_norm, gw->final_norm_bytes, D * 2, "final_norm");
        mp.fuse_alpha         = wrap_req(gw->fuse_alpha, gw->fuse_alpha_bytes,
                                         ((size_t)cfg->num_layers + 1) * NCAP * 4, "fuse_alpha");
        mp.markov_dflare_norm = wrap_req(gw->markov_dflare_norm, gw->markov_dflare_norm_bytes, D * 2, "markov_dflare_norm");
        mp.markov_w1          = wrap_req(gw->markov_w1, gw->markov_w1_bytes, V_UNPAD * RANK * 2, "markov_w1");
        mp.rnn_w_ctx_t        = wrap_req(gw->rnn_w_ctx_t, gw->rnn_w_ctx_t_bytes, 2 * D * 3 * RANK * 4, "rnn_w_ctx_t");
        mp.rnn_w_step_t       = wrap_req(gw->rnn_w_step_t, gw->rnn_w_step_t_bytes, 2 * RANK * 3 * RANK * 4, "rnn_w_step_t");
        mp.rnn_b_f32          = wrap_req(gw->rnn_b_f32, gw->rnn_b_f32_bytes, 3 * RANK * 4, "rnn_b_f32");
        mp.fold_weight        = wrap_req(gw->fold_weight, gw->fold_weight_bytes, V_UNPAD * RANK * 2, "fold_weight");
        mp.fold_bias          = wrap_req(gw->fold_bias, gw->fold_bias_bytes, V_UNPAD * 2, "fold_bias");

        if (!mp.embed_tokens.buf || !mp.embed_norm.buf || !mp.lm_head.buf ||
            !mp.mask_embed.buf || !mp.final_norm.buf || !mp.fuse_alpha.buf ||
            !mp.markov_dflare_norm.buf || !mp.markov_w1.buf || !mp.rnn_w_ctx_t.buf ||
            !mp.rnn_w_step_t.buf || !mp.rnn_b_f32.buf || !mp.fold_weight.buf || !mp.fold_bias.buf) {
            ctx->mtp = MTPAttachedState{};
            return -5;
        }

        const size_t KV_DIM = KVH * HD;
        const size_t Q_DIM = H * HD;
        const size_t QKV_DIM = Q_DIM + 2 * KV_DIM;
        mp.layers.resize((size_t)cfg->num_layers);
        for (int i = 0; i < cfg->num_layers; ++i) {
            const MTPDraftLayerWeights& src = layers[i];
            MTPLayerMetalWeights& dst = mp.layers[(size_t)i];
            dst.is_local = src.is_local;
            dst.window   = src.window;
            dst.dflare_norm              = wrap_req(src.dflare_norm, src.dflare_norm_bytes, D * 2, "dflare_norm");
            dst.input_layernorm          = wrap_req(src.input_layernorm, src.input_layernorm_bytes, D * 2, "input_layernorm");
            dst.post_attention_layernorm = wrap_req(src.post_attention_layernorm, src.post_attention_layernorm_bytes, D * 2, "post_attn_ln");
            dst.kv_ctx_w                 = wrap_req(src.kv_ctx_w, src.kv_ctx_w_bytes, 2 * KV_DIM * D * 2, "kv_ctx_w");
            dst.k_norm_ctx               = wrap_req(src.k_norm_ctx, src.k_norm_ctx_bytes, HD * 2, "k_norm_ctx");
            dst.qkv_w                    = wrap_req(src.qkv_w, src.qkv_w_bytes, QKV_DIM * D * 2, "qkv_w");
            dst.q_norm                   = wrap_req(src.q_norm, src.q_norm_bytes, HD * 2, "q_norm");
            dst.k_norm                   = wrap_req(src.k_norm, src.k_norm_bytes, HD * 2, "k_norm");
            dst.o_proj_w                 = wrap_req(src.o_proj_w, src.o_proj_w_bytes, D * Q_DIM * 2, "o_proj_w");
            dst.gu_w                     = wrap_req(src.gu_w, src.gu_w_bytes, 2 * INTER * D * 2, "gu_w");
            dst.down_proj_w              = wrap_req(src.down_proj_w, src.down_proj_w_bytes, D * INTER * 2, "down_proj_w");
            if (!dst.dflare_norm.buf || !dst.input_layernorm.buf || !dst.post_attention_layernorm.buf ||
                !dst.kv_ctx_w.buf || !dst.k_norm_ctx.buf || !dst.qkv_w.buf || !dst.q_norm.buf ||
                !dst.k_norm.buf || !dst.o_proj_w.buf || !dst.gu_w.buf || !dst.down_proj_w.buf) {
                ctx->mtp = MTPAttachedState{};
                return -6;
            }
        }

        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.preprocessorMacros = @{
            @"MTP_B":    @(cfg->block_size),
            @"MTP_D":    @(cfg->hidden_size),
            @"MTP_G":    @(cfg->num_heads / cfg->num_kv_heads),
            @"MTP_H":    @(cfg->num_heads),
            @"MTP_HD":   @(cfg->head_dim),
            @"MTP_KVH":  @(cfg->num_kv_heads),
            @"MTP_NCAP": @(cfg->n_target_layers),
            @"MTP_RANK": @(cfg->markov_rank),
        };
        if (@available(macOS 15.0, *)) {
            opts.mathMode = MTLMathModeSafe;
        }
        NSError* err = nil;
        NSString* msrc = [NSString stringWithUTF8String:kMTPUnifiedSource];
        mp.lib = [ctx->device newLibraryWithSource:msrc options:opts error:&err];
        if (!mp.lib) {
            fprintf(stderr, "[inkling_decode] ERROR: MTP Metal compile failed:\n%s\n",
                    err ? err.localizedDescription.UTF8String : "unknown");
            ctx->mtp = MTPAttachedState{};
            return -7;
        }

        mp.pso_init_x[0]     = make_mtp_pso(ctx->device, mp.lib, @"mtp_init_x", -1, -1, -1, -1, 0);
        mp.pso_init_x[1]     = make_mtp_pso(ctx->device, mp.lib, @"mtp_init_x", -1, -1, -1, -1, 1);
        mp.pso_dflare_mix    = make_mtp_pso(ctx->device, mp.lib, @"mtp_dflare_mix");
        mp.pso_add_rmsnorm[0]= make_mtp_pso(ctx->device, mp.lib, @"mtp_add_rmsnorm", -1, -1, -1, 0);
        mp.pso_add_rmsnorm[1]= make_mtp_pso(ctx->device, mp.lib, @"mtp_add_rmsnorm", -1, -1, -1, 1);
        for (int m = 1; m <= (int)B; ++m) {
            mp.pso_gemv_ctx_kv[m] = make_mtp_pso(ctx->device, mp.lib, @"mtp_bf16_gemv", m, (int)D, 4);
            if (!mp.pso_gemv_ctx_kv[m]) { ctx->mtp = MTPAttachedState{}; return -8; }
        }
        mp.pso_gemv_qkv    = make_mtp_pso(ctx->device, mp.lib, @"mtp_bf16_gemv", (int)B, (int)D, 4);
        mp.pso_gemv_o_proj = make_mtp_pso(ctx->device, mp.lib, @"mtp_bf16_gemv", (int)B, (int)Q_DIM, 4);
        mp.pso_gu_silu     = make_mtp_pso(ctx->device, mp.lib, @"mtp_bf16_gu_silu", (int)B, (int)D, 4);
        mp.pso_gemv_down   = make_mtp_pso(ctx->device, mp.lib, @"mtp_bf16_gemv", (int)B, (int)INTER, 4);
        for (int k = 1; k < (int)B; ++k) {
            mp.pso_gemv_head_r2[k]  = make_mtp_pso(ctx->device, mp.lib, @"mtp_bf16_gemv_r2", k, (int)D, 4);
            mp.pso_markov_ctx_p1[k] = make_mtp_pso(ctx->device, mp.lib, @"mtp_markov_context_p1", k);
            mp.pso_markov_ctx_p2[k] = make_mtp_pso(ctx->device, mp.lib, @"mtp_markov_context_p2", k);
            if (!mp.pso_gemv_head_r2[k] || !mp.pso_markov_ctx_p1[k] || !mp.pso_markov_ctx_p2[k]) {
                ctx->mtp = MTPAttachedState{};
                return -8;
            }
        }
        mp.pso_kv_ctx_write          = make_mtp_pso(ctx->device, mp.lib, @"mtp_kv_ctx_norm_rope_write");
        mp.pso_qkv_pack              = make_mtp_pso(ctx->device, mp.lib, @"mtp_qkv_norm_rope_pack");
        mp.pso_sdpa_single           = make_mtp_pso(ctx->device, mp.lib, @"mtp_sdpa_single");
        mp.pso_sdpa_split_p1         = make_mtp_pso(ctx->device, mp.lib, @"mtp_sdpa_split_p1");
        mp.pso_sdpa_split_p2         = make_mtp_pso(ctx->device, mp.lib, @"mtp_sdpa_split_p2");
        mp.pso_rnn_step              = make_mtp_pso(ctx->device, mp.lib, @"mtp_rnn_step");
        mp.pso_markov_argmax_p1      = make_mtp_pso(ctx->device, mp.lib, @"mtp_markov_argmax_p1");
        mp.pso_markov_argmax_p2      = make_mtp_pso(ctx->device, mp.lib, @"mtp_markov_argmax_p2");
        mp.pso_markov_logits_max_p1  = make_mtp_pso(ctx->device, mp.lib, @"mtp_markov_logits_max_p1");
        mp.pso_softmax_max_p2        = make_mtp_pso(ctx->device, mp.lib, @"mtp_softmax_max_p2");
        mp.pso_softmax_exp_sum_p1    = make_mtp_pso(ctx->device, mp.lib, @"mtp_softmax_exp_sum_p1");
        mp.pso_softmax_sum_p2        = make_mtp_pso(ctx->device, mp.lib, @"mtp_softmax_sum_p2");
        mp.pso_probs_scale           = make_mtp_pso(ctx->device, mp.lib, @"mtp_probs_scale");
        mp.pso_topp_hist             = make_mtp_pso(ctx->device, mp.lib, @"mtp_topp_hist");
        mp.pso_topp_scan             = make_mtp_pso(ctx->device, mp.lib, @"mtp_topp_scan");
        mp.pso_topp_mask_sum         = make_mtp_pso(ctx->device, mp.lib, @"mtp_topp_mask_sum");
        mp.pso_probs_scale_gumbel_p1 = make_mtp_pso(ctx->device, mp.lib, @"mtp_probs_scale_gumbel_p1");
        mp.pso_gumbel_p2             = make_mtp_pso(ctx->device, mp.lib, @"mtp_gumbel_p2");
        mp.pso_greedy_accept         = make_mtp_pso(ctx->device, mp.lib, @"mtp_greedy_accept");
        mp.pso_reject_prefix         = make_mtp_pso(ctx->device, mp.lib, @"mtp_reject_prefix");
        mp.pso_reject_sample_p1      = make_mtp_pso(ctx->device, mp.lib, @"mtp_reject_sample_p1");
        mp.pso_reject_sample_p2      = make_mtp_pso(ctx->device, mp.lib, @"mtp_reject_sample_p2");

        if (!mp.pso_init_x[0] || !mp.pso_init_x[1] || !mp.pso_dflare_mix ||
            !mp.pso_add_rmsnorm[0] || !mp.pso_add_rmsnorm[1] ||
            !mp.pso_gemv_qkv || !mp.pso_gemv_o_proj || !mp.pso_gu_silu || !mp.pso_gemv_down ||
            !mp.pso_kv_ctx_write || !mp.pso_qkv_pack ||
            !mp.pso_sdpa_single || !mp.pso_sdpa_split_p1 || !mp.pso_sdpa_split_p2 ||
            !mp.pso_rnn_step || !mp.pso_markov_argmax_p1 || !mp.pso_markov_argmax_p2 ||
            !mp.pso_markov_logits_max_p1 || !mp.pso_softmax_max_p2 || !mp.pso_softmax_exp_sum_p1 ||
            !mp.pso_softmax_sum_p2 || !mp.pso_probs_scale || !mp.pso_topp_hist ||
            !mp.pso_topp_scan || !mp.pso_topp_mask_sum || !mp.pso_probs_scale_gumbel_p1 ||
            !mp.pso_gumbel_p2 || !mp.pso_greedy_accept || !mp.pso_reject_prefix ||
            !mp.pso_reject_sample_p1 || !mp.pso_reject_sample_p2) {
            ctx->mtp = MTPAttachedState{};
            return -8;
        }

        auto alloc_s = [&](size_t bytes) -> id<MTLBuffer> {
            return [ctx->device newBufferWithLength:std::max<size_t>(bytes, 256)
                                            options:MTLResourceStorageModeShared];
        };

        const size_t MAX_K = B - 1;
        const size_t kMaxSdpaSplits = 32;
        const size_t n_markov_tg = (V_UNPAD + 63) / 64;
        const size_t n_tg_target = (V_UNPAD + 15) / 16;
        const size_t n_tg_vec    = std::min<size_t>(64, std::max<size_t>(1, (V_UNPAD + 255) / 256));

        mp.hidden_buf        = alloc_s(B * NCAP * D * 2);
        mp.inv_freq_buf      = alloc_s((HD / 2) * sizeof(float));
        mp.x_buf             = alloc_s(B * D * 2);
        mp.residual_buf      = alloc_s(B * D * 2);
        mp.norm_buf          = alloc_s(B * D * 2);
        mp.ctx_buf           = alloc_s(B * D * 2);
        mp.kv_ctx_buf        = alloc_s(B * 2 * KV_DIM * 2);
        mp.qkv_buf           = alloc_s(B * QKV_DIM * 2);
        mp.q_buf             = alloc_s(H * B * HD * 2);
        mp.bk_buf            = alloc_s(KVH * B * HD * 2);
        mp.bv_buf            = alloc_s(KVH * B * HD * 2);
        mp.attn_out_buf      = alloc_s(B * Q_DIM * 2);
        mp.mlp_mid_buf       = alloc_s(B * INTER * 2);
        mp.sdpa_part_o       = alloc_s(kMaxSdpaSplits * H * B * HD * sizeof(float));
        mp.sdpa_part_ml      = alloc_s(kMaxSdpaSplits * H * B * 2 * sizeof(float));
        mp.x_out_buf         = alloc_s(MAX_K * D * 2);
        mp.base_logits_buf   = alloc_s(MAX_K * V_UNPAD * 2);
        mp.f_rnn_buf         = alloc_s(D * 2);
        mp.rnn_ctx_buf       = alloc_s(MAX_K * 3 * RANK * sizeof(float));
        mp.rnn_state_buf[0]  = alloc_s(RANK * 2);
        mp.rnn_state_buf[1]  = alloc_s(RANK * 2);
        mp.rnn_latent_buf    = alloc_s(RANK * 2);
        mp.markov_ctx_part_x = alloc_s(8 * MAX_K * 3 * RANK * sizeof(float));
        mp.markov_ctx_part_f = alloc_s(8 * 3 * RANK * sizeof(float));
        mp.q_probs_buf       = alloc_s(MAX_K * V_UNPAD * sizeof(float));
        mp.p_probs_buf       = alloc_s(B * V_UNPAD * sizeof(float));
        mp.row_max_buf       = alloc_s(B * sizeof(float));
        mp.row_inv_sum_buf   = alloc_s(B * sizeof(float));
        mp.topp_states_buf   = alloc_s(B * sizeof(MTPTopPState));
        mp.softmax_part_buf  = alloc_s(std::max(B * n_tg_target, B * n_tg_vec * 32) * sizeof(float));
        mp.topp_part_cnt_buf = alloc_s(B * n_tg_vec * 32 * sizeof(uint32_t));
        mp.markov_part_val   = alloc_s(std::max(n_markov_tg, n_tg_vec) * sizeof(float));
        mp.markov_part_idx   = alloc_s(std::max(n_markov_tg, n_tg_vec) * sizeof(uint32_t));
        mp.reject_res_val    = alloc_s(n_tg_vec * sizeof(float));
        mp.reject_res_idx    = alloc_s(n_tg_vec * sizeof(uint32_t));
        mp.reject_p_val      = alloc_s(n_tg_vec * sizeof(float));
        mp.reject_p_idx      = alloc_s(n_tg_vec * sizeof(uint32_t));
        mp.reject_res_sum    = alloc_s(n_tg_vec * sizeof(float));
        mp.prev_tok_buf      = alloc_s(256);
        mp.d_toks_buf        = alloc_s(8 * sizeof(int32_t));
        mp.verify_toks_buf   = alloc_s(8 * sizeof(int32_t));
        mp.out_toks_buf      = alloc_s(8 * sizeof(int32_t));
        mp.d_reject_j_buf    = alloc_s(256);
        mp.d_n_acc_buf       = alloc_s(256);

        if (!mp.hidden_buf || !mp.inv_freq_buf || !mp.x_buf || !mp.residual_buf ||
            !mp.norm_buf || !mp.ctx_buf || !mp.kv_ctx_buf || !mp.qkv_buf ||
            !mp.q_buf || !mp.bk_buf || !mp.bv_buf || !mp.attn_out_buf || !mp.mlp_mid_buf ||
            !mp.sdpa_part_o || !mp.sdpa_part_ml || !mp.x_out_buf || !mp.base_logits_buf ||
            !mp.f_rnn_buf || !mp.rnn_ctx_buf || !mp.rnn_state_buf[0] || !mp.rnn_state_buf[1] ||
            !mp.rnn_latent_buf || !mp.markov_ctx_part_x || !mp.markov_ctx_part_f ||
            !mp.q_probs_buf || !mp.p_probs_buf || !mp.row_max_buf || !mp.row_inv_sum_buf ||
            !mp.topp_states_buf || !mp.softmax_part_buf || !mp.topp_part_cnt_buf ||
            !mp.markov_part_val || !mp.markov_part_idx || !mp.reject_res_val ||
            !mp.reject_res_idx || !mp.reject_p_val || !mp.reject_p_idx || !mp.reject_res_sum ||
            !mp.prev_tok_buf || !mp.d_toks_buf || !mp.verify_toks_buf || !mp.out_toks_buf ||
            !mp.d_reject_j_buf || !mp.d_n_acc_buf) {
            ctx->mtp = MTPAttachedState{};
            return -9;
        }

        float* invp = (float*)[mp.inv_freq_buf contents];
        for (size_t d = 0; d < HD / 2; ++d) {
            invp[d] = std::pow(cfg->rope_theta, -2.0f * (float)d / (float)HD);
        }
        mp.tmp_kv_k.resize((size_t)cfg->num_layers);
        mp.tmp_kv_v.resize((size_t)cfg->num_layers);
        mp.attached = true;
        return 0;
    }
}

int inkling_ctx_mtp_step(
    InklingCtx*          ctx,
    int32_t              anchor_token,
    const void*          in_hidden_ptr,
    size_t               in_hidden_bytes,
    void*                out_hidden_ptr,
    size_t               out_hidden_bytes,
    int32_t              n_ctx,
    int32_t              num_draft,
    float                temperature,
    float                top_p,
    float                logit_softcap,
    uint64_t             rng_seed,
    uint64_t             rng_step,
    void* const*         kv_ptrs,
    void* const*         conv_ptrs,
    InklingKVMeta*       kv_meta,
    MTPKVCacheDesc*      draft_kv_descs,
    int32_t*             out_tokens,
    int32_t*             out_n_acc,
    InklingStepStats*    stats_out)
{
    @autoreleasepool {
        if (!ctx || !ctx->mtp.attached || !ctx->globals.registered ||
            !in_hidden_ptr || !out_hidden_ptr || !kv_ptrs || !conv_ptrs ||
            !kv_meta || !draft_kv_descs || !out_tokens || !out_n_acc) {
            return -1;
        }
        MTPAttachedState& mp = ctx->mtp;
        const MTPDraftConfig& mc = mp.cfg;
        const int32_t D = mc.hidden_size;
        const int32_t B = mc.block_size;
        const int32_t NCAP = mc.n_target_layers;
        const int32_t verify_max_m = std::min<int32_t>(ctx->cfg.max_m, 8);
        const int32_t M = num_draft + 1;
        if (n_ctx < 1 || n_ctx > B || num_draft < 1 || num_draft >= B || M > verify_max_m) {
            fprintf(stderr, "[inkling_decode] ERROR: invalid n_ctx=%d or num_draft=%d (B=%d, verify_max_m=%d)\n",
                    n_ctx, num_draft, B, verify_max_m);
            return -2;
        }
        if (anchor_token < 0 || anchor_token >= mc.vocab_size) {
            fprintf(stderr, "[inkling_decode] ERROR: anchor_token=%d out of range [0,%d)\n",
                    anchor_token, mc.vocab_size);
            return -2;
        }
        if (temperature > 0.0f && (top_p <= 0.0f || top_p > 1.0f)) {
            fprintf(stderr, "[inkling_decode] ERROR: invalid top_p=%.6f for sampled mode\n", top_p);
            return -2;
        }
        const size_t req_in_bytes  = (size_t)n_ctx * (size_t)NCAP * (size_t)D * sizeof(uint16_t);
        const size_t req_out_bytes = (size_t)M     * (size_t)NCAP * (size_t)D * sizeof(uint16_t);
        if (in_hidden_bytes < req_in_bytes || out_hidden_bytes < req_out_bytes) {
            fprintf(stderr, "[inkling_decode] ERROR: hidden buffer too small (in=%zu<%zu, out=%zu<%zu)\n",
                    in_hidden_bytes, req_in_bytes, out_hidden_bytes, req_out_bytes);
            return -3;
        }
        for (int i = 0; i < ctx->cfg.n_layers; ++i) {
            if (!ctx->layers[i].registered) return -4;
        }

        std::memcpy([mp.hidden_buf contents], in_hidden_ptr, req_in_bytes);

        BufSlice out_hidden_slice = wrap_ptr_nocache(ctx, (uintptr_t)out_hidden_ptr, out_hidden_bytes);
        if (!out_hidden_slice.buf) return -5;

        int32_t d_off_before[8]{}, d_off_after[8]{}, d_ring_start[8]{}, d_valid_ctx[8]{};
        int32_t d_win[8]{}, d_next_ring[8]{}, d_k_sh[8]{}, d_k_ss[8]{};
        for (int i = 0; i < mc.num_layers; ++i) {
            const MTPKVCacheDesc& kd = draft_kv_descs[i];
            const MTPLayerMetalWeights& lw = mp.layers[(size_t)i];
            d_off_before[i] = kd.offset;
            d_off_after[i]  = d_off_before[i] + n_ctx;
            if (kd.is_rotating != 0) {
                const int32_t W = (kd.window > 0) ? kd.window : lw.window;
                d_win[i] = W;
                if (W <= 0 || n_ctx > W) return -20 - i;
                const int32_t cur_idx = (kd.ring_idx >= 0) ? (kd.ring_idx % W) : (d_off_before[i] % W);
                if (d_off_before[i] + n_ctx <= W) {
                    if (kd.cache_cap < d_off_after[i]) return -20 - i;
                    d_ring_start[i] = (kd.ring_idx >= 0) ? (kd.ring_idx % W) : d_off_before[i];
                    d_valid_ctx[i]  = d_off_after[i];
                } else {
                    if (kd.cache_cap < W) return -20 - i;
                    d_ring_start[i] = cur_idx;
                    d_valid_ctx[i]  = W;
                }
                const int32_t end_mod = (d_ring_start[i] + n_ctx) % W;
                d_next_ring[i] = (end_mod == 0 && d_off_after[i] >= W) ? W : end_mod;
            } else {
                d_win[i] = 0;
                if (d_off_after[i] > kd.cache_cap) return -30 - i;
                d_ring_start[i] = d_off_before[i];
                d_valid_ctx[i]  = d_off_after[i];
                d_next_ring[i]  = d_off_after[i];
            }
            d_k_ss[i] = (kd.stride_s > 0) ? kd.stride_s : mc.head_dim;
            d_k_sh[i] = (kd.stride_h > 0) ? kd.stride_h : (kd.cache_cap * d_k_ss[i]);
            const size_t min_bytes = (size_t)mc.num_kv_heads * (size_t)d_k_sh[i] * sizeof(uint16_t);
            if ((kd.k_nbytes > 0 && kd.k_nbytes < min_bytes) ||
                (kd.v_nbytes > 0 && kd.v_nbytes < min_bytes)) {
                return -35 - i;
            }
            mp.tmp_kv_k[(size_t)i] = wrap_ptr_nocache(ctx, kd.k_ptr, std::max(kd.k_nbytes, min_bytes));
            mp.tmp_kv_v[(size_t)i] = wrap_ptr_nocache(ctx, kd.v_ptr, std::max(kd.v_nbytes, min_bytes));
            if (!mp.tmp_kv_k[(size_t)i].buf || !mp.tmp_kv_v[(size_t)i].buf) return -40 - i;
        }

        {
            int rc = ensure_conv_hist_buf(ctx, verify_max_m);
            if (rc != 0) return rc;
            rc = prepare_target_kv_conv_part(ctx, M, kv_ptrs, conv_ptrs, kv_meta);
            if (rc != 0) return rc;
        }
        ctx->conv_hist_active = false;
        ctx->conv_hist_m = 0;
        ctx->last_verify_m = 0;

        const int grp_mt = (M <= 4) ? 4 : 8;
        id<MTLComputePipelineState> pso_grp_w13_nv = get_grp_moe_pso(ctx, true,  false, 2, grp_mt);
        id<MTLComputePipelineState> pso_grp_w2_nv  = get_grp_moe_pso(ctx, true,  true,  2, grp_mt);
        id<MTLComputePipelineState> pso_grp_w13_mx = get_grp_moe_pso(ctx, false, false, 2, grp_mt);
        id<MTLComputePipelineState> pso_grp_w2_mx  = get_grp_moe_pso(ctx, false, true,  2, grp_mt);
        const bool is_sampled = (temperature > 0.0f);
        const bool do_top_p = is_sampled && (top_p > 0.0f && top_p < 1.0f);
        const BufSlice& target_lm_head = (ctx->globals.lm_head.buf != ctx->zero_buf)
                                       ? ctx->globals.lm_head : mp.lm_head;
        id<MTLComputePipelineState> pso_argmax_p1 = !is_sampled ? get_argmax_p1_pso(ctx, M, 2) : nil;
        id<MTLComputePipelineState> pso_logits_max_p1 = is_sampled ? get_head_logits_max_p1_pso(ctx, M) : nil;
        if ((!is_sampled && !pso_argmax_p1) || (is_sampled && !pso_logits_max_p1)) return -45;

        const uint64_t t_wall0 = mach_absolute_time();
        const uint64_t t_enc0  = t_wall0;

        id<MTLCommandBuffer> cb = [ctx->queue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
        int32_t n_dispatches = 0;

        const int32_t NV      = (int32_t)ctx->schedule.size();
        const int32_t KV_DIM  = mc.num_kv_heads * mc.head_dim;
        const int32_t Q_DIM   = mc.num_heads * mc.head_dim;
        const int32_t QKV_DIM = Q_DIM + 2 * KV_DIM;
        const int32_t INTER   = mc.inter_size;
        const int32_t V_UNPAD = mc.unpadded_vocab_size;
        const float   m_eps   = mc.rms_norm_eps;
        const float   neg_log2_theta_over_half = -std::log2(mc.rope_theta) / (float)(mc.head_dim / 2);
        const float   sdpa_scale2 = (1.0f / std::sqrt((float)mc.head_dim)) * 1.4426950408889634f;
        const NSUInteger tg_rope = (NSUInteger)std::max(32, mc.head_dim / 2);
        const NSUInteger tg_sdpa = (NSUInteger)(B * 32);

        const int32_t n_markov_tg = (V_UNPAD + 63) / 64;
        const int32_t n_tg_target = (V_UNPAD + 15) / 16;
        const int32_t n_tg_vec    = std::min<int32_t>(64, std::max<int32_t>(1, (V_UNPAD + 255) / 256));

        {
            MTPInitXPrm iprm{anchor_token, D, B, mc.markov_rank, mc.embed_norm_eps};
            [enc setComputePipelineState:mp.pso_init_x[mp.has_embed_norm ? 1 : 0]];
            [enc setBuffer:mp.embed_tokens.buf offset:mp.embed_tokens.offset atIndex:0];
            [enc setBuffer:mp.embed_norm.buf   offset:mp.embed_norm.offset   atIndex:1];
            [enc setBuffer:mp.mask_embed.buf   offset:mp.mask_embed.offset   atIndex:2];
            [enc setBuffer:mp.x_buf            offset:0                       atIndex:3];
            [enc setBuffer:mp.rnn_state_buf[0] offset:0                       atIndex:4];
            [enc setBuffer:mp.prev_tok_buf     offset:0                       atIndex:5];
            [enc setBuffer:mp.verify_toks_buf  offset:0                       atIndex:6];
            [enc setBytes:&iprm length:sizeof(iprm) atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            ++n_dispatches;
        }

        for (int i = 0; i < mc.num_layers; ++i) {
            const MTPLayerMetalWeights& lw = mp.layers[(size_t)i];
            const MTPKVCacheDesc& kd = draft_kv_descs[i];

            {
                MTPMixPrm prm{D, mc.n_target_layers, 0, i, m_eps};
                [enc setComputePipelineState:mp.pso_dflare_mix];
                [enc setBuffer:mp.hidden_buf     offset:0                    atIndex:0];
                [enc setBuffer:mp.fuse_alpha.buf offset:mp.fuse_alpha.offset atIndex:1];
                [enc setBuffer:lw.dflare_norm.buf offset:lw.dflare_norm.offset atIndex:2];
                [enc setBuffer:mp.ctx_buf        offset:0                    atIndex:3];
                [enc setBytes:&prm length:sizeof(prm) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_ctx, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_dispatches;
            }
            {
                MTPGemvPrm prm{2 * KV_DIM};
                [enc setComputePipelineState:mp.pso_gemv_ctx_kv[n_ctx]];
                [enc setBuffer:mp.ctx_buf     offset:0                  atIndex:0];
                [enc setBuffer:lw.kv_ctx_w.buf offset:lw.kv_ctx_w.offset atIndex:1];
                [enc setBuffer:mp.kv_ctx_buf  offset:0                  atIndex:2];
                [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((2 * KV_DIM + 3) / 4), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                ++n_dispatches;
            }
            {
                MTPKvCtxWritePrm prm{
                    KV_DIM, d_off_before[i], kd.is_rotating, d_win[i], d_ring_start[i],
                    d_k_sh[i], d_k_ss[i], d_k_sh[i], d_k_ss[i], m_eps, neg_log2_theta_over_half
                };
                [enc setComputePipelineState:mp.pso_kv_ctx_write];
                [enc setBuffer:mp.kv_ctx_buf                 offset:0                         atIndex:0];
                [enc setBuffer:lw.k_norm_ctx.buf             offset:lw.k_norm_ctx.offset     atIndex:1];
                [enc setBuffer:mp.inv_freq_buf               offset:0                         atIndex:2];
                [enc setBuffer:mp.tmp_kv_k[(size_t)i].buf    offset:mp.tmp_kv_k[(size_t)i].offset atIndex:3];
                [enc setBuffer:mp.tmp_kv_v[(size_t)i].buf    offset:mp.tmp_kv_v[(size_t)i].offset atIndex:4];
                [enc setBytes:&prm length:sizeof(prm) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_ctx, (NSUInteger)mc.num_kv_heads, 1)
                    threadsPerThreadgroup:MTLSizeMake(tg_rope, 1, 1)];
                ++n_dispatches;
            }
            {
                MTPAddNormPrm prm{D, 0, 1, m_eps};
                [enc setComputePipelineState:mp.pso_add_rmsnorm[i == 0 ? 0 : 1]];
                [enc setBuffer:mp.x_buf               offset:0                         atIndex:0];
                [enc setBuffer:mp.residual_buf        offset:0                         atIndex:1];
                [enc setBuffer:lw.input_layernorm.buf offset:lw.input_layernorm.offset atIndex:2];
                [enc setBuffer:mp.residual_buf        offset:0                         atIndex:3];
                [enc setBuffer:mp.norm_buf            offset:0                         atIndex:4];
                [enc setBytes:&prm length:sizeof(prm) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)B, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_dispatches;
            }
            {
                MTPGemvPrm prm{QKV_DIM};
                [enc setComputePipelineState:mp.pso_gemv_qkv];
                [enc setBuffer:mp.norm_buf offset:0               atIndex:0];
                [enc setBuffer:lw.qkv_w.buf offset:lw.qkv_w.offset atIndex:1];
                [enc setBuffer:mp.qkv_buf  offset:0               atIndex:2];
                [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((QKV_DIM + 3) / 4), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                ++n_dispatches;
            }
            {
                MTPQkvPackPrm prm{Q_DIM, KV_DIM, QKV_DIM, d_off_after[i], m_eps, neg_log2_theta_over_half};
                [enc setComputePipelineState:mp.pso_qkv_pack];
                [enc setBuffer:mp.qkv_buf      offset:0                atIndex:0];
                [enc setBuffer:lw.q_norm.buf   offset:lw.q_norm.offset atIndex:1];
                [enc setBuffer:lw.k_norm.buf   offset:lw.k_norm.offset atIndex:2];
                [enc setBuffer:mp.inv_freq_buf offset:0                atIndex:3];
                [enc setBuffer:mp.q_buf        offset:0                atIndex:4];
                [enc setBuffer:mp.bk_buf       offset:0                atIndex:5];
                [enc setBuffer:mp.bv_buf       offset:0                atIndex:6];
                [enc setBytes:&prm length:sizeof(prm) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)B, (NSUInteger)(mc.num_heads + 2 * mc.num_kv_heads), 1)
                    threadsPerThreadgroup:MTLSizeMake(tg_rope, 1, 1)];
                ++n_dispatches;
            }
            {
                const int32_t total_k = d_valid_ctx[i] + B;
                if (d_valid_ctx[i] < 256) {
                    MTPSdpaPrm prm{d_valid_ctx[i], total_k, d_k_sh[i], d_k_ss[i], d_k_sh[i], d_k_ss[i], 1, total_k, sdpa_scale2};
                    [enc setComputePipelineState:mp.pso_sdpa_single];
                    [enc setBuffer:mp.q_buf                    offset:0                         atIndex:0];
                    [enc setBuffer:mp.tmp_kv_k[(size_t)i].buf  offset:mp.tmp_kv_k[(size_t)i].offset atIndex:1];
                    [enc setBuffer:mp.tmp_kv_v[(size_t)i].buf  offset:mp.tmp_kv_v[(size_t)i].offset atIndex:2];
                    [enc setBuffer:mp.bk_buf                   offset:0                         atIndex:3];
                    [enc setBuffer:mp.bv_buf                   offset:0                         atIndex:4];
                    [enc setBuffer:mp.attn_out_buf             offset:0                         atIndex:5];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:6];
                    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)mc.num_kv_heads, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(tg_sdpa, 1, 1)];
                    ++n_dispatches;
                } else {
                    int32_t n_splits = std::min<int32_t>(32, std::max<int32_t>(4, (total_k + 255) / 256));
                    int32_t chunk = (total_k + n_splits - 1) / n_splits;
                    chunk = ((chunk + 7) / 8) * 8;
                    n_splits = (total_k + chunk - 1) / chunk;
                    MTPSdpaPrm prm{d_valid_ctx[i], total_k, d_k_sh[i], d_k_ss[i], d_k_sh[i], d_k_ss[i], n_splits, chunk, sdpa_scale2};
                    [enc setComputePipelineState:mp.pso_sdpa_split_p1];
                    [enc setBuffer:mp.q_buf                    offset:0                         atIndex:0];
                    [enc setBuffer:mp.tmp_kv_k[(size_t)i].buf  offset:mp.tmp_kv_k[(size_t)i].offset atIndex:1];
                    [enc setBuffer:mp.tmp_kv_v[(size_t)i].buf  offset:mp.tmp_kv_v[(size_t)i].offset atIndex:2];
                    [enc setBuffer:mp.bk_buf                   offset:0                         atIndex:3];
                    [enc setBuffer:mp.bv_buf                   offset:0                         atIndex:4];
                    [enc setBuffer:mp.sdpa_part_o              offset:0                         atIndex:5];
                    [enc setBuffer:mp.sdpa_part_ml             offset:0                         atIndex:6];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:7];
                    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)mc.num_kv_heads, (NSUInteger)n_splits, 1)
                        threadsPerThreadgroup:MTLSizeMake(tg_sdpa, 1, 1)];
                    ++n_dispatches;

                    [enc setComputePipelineState:mp.pso_sdpa_split_p2];
                    [enc setBuffer:mp.sdpa_part_o  offset:0 atIndex:0];
                    [enc setBuffer:mp.sdpa_part_ml offset:0 atIndex:1];
                    [enc setBuffer:mp.attn_out_buf offset:0 atIndex:2];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)B, (NSUInteger)mc.num_heads, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    ++n_dispatches;
                }
            }
            {
                MTPGemvPrm prm{D};
                [enc setComputePipelineState:mp.pso_gemv_o_proj];
                [enc setBuffer:mp.attn_out_buf offset:0                  atIndex:0];
                [enc setBuffer:lw.o_proj_w.buf offset:lw.o_proj_w.offset atIndex:1];
                [enc setBuffer:mp.x_buf        offset:0                  atIndex:2];
                [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((D + 3) / 4), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                ++n_dispatches;
            }
            {
                MTPAddNormPrm prm{D, 0, 1, m_eps};
                [enc setComputePipelineState:mp.pso_add_rmsnorm[1]];
                [enc setBuffer:mp.x_buf                        offset:0                                  atIndex:0];
                [enc setBuffer:mp.residual_buf                 offset:0                                  atIndex:1];
                [enc setBuffer:lw.post_attention_layernorm.buf offset:lw.post_attention_layernorm.offset atIndex:2];
                [enc setBuffer:mp.residual_buf                 offset:0                                  atIndex:3];
                [enc setBuffer:mp.norm_buf                     offset:0                                  atIndex:4];
                [enc setBytes:&prm length:sizeof(prm) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)B, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_dispatches;
            }
            {
                MTPGemvPrm prm{INTER};
                [enc setComputePipelineState:mp.pso_gu_silu];
                [enc setBuffer:mp.norm_buf    offset:0            atIndex:0];
                [enc setBuffer:lw.gu_w.buf    offset:lw.gu_w.offset atIndex:1];
                [enc setBuffer:mp.mlp_mid_buf offset:0            atIndex:2];
                [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((INTER + 3) / 4), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                ++n_dispatches;
            }
            {
                MTPGemvPrm prm{D};
                [enc setComputePipelineState:mp.pso_gemv_down];
                [enc setBuffer:mp.mlp_mid_buf     offset:0                     atIndex:0];
                [enc setBuffer:lw.down_proj_w.buf offset:lw.down_proj_w.offset atIndex:1];
                [enc setBuffer:mp.x_buf           offset:0                     atIndex:2];
                [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((D + 3) / 4), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                ++n_dispatches;
            }
        }

        {
            MTPAddNormPrm prm{D, 1, 0, m_eps};
            [enc setComputePipelineState:mp.pso_add_rmsnorm[1]];
            [enc setBuffer:mp.x_buf          offset:0                    atIndex:0];
            [enc setBuffer:mp.residual_buf   offset:0                    atIndex:1];
            [enc setBuffer:mp.final_norm.buf offset:mp.final_norm.offset atIndex:2];
            [enc setBuffer:ctx->zero_buf     offset:0                    atIndex:3];
            [enc setBuffer:mp.x_out_buf      offset:0                    atIndex:4];
            [enc setBytes:&prm length:sizeof(prm) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)num_draft, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            ++n_dispatches;
        }

        {
            MTPGemvPrm prm{V_UNPAD};
            [enc setComputePipelineState:mp.pso_gemv_head_r2[num_draft]];
            [enc setBuffer:mp.x_out_buf       offset:0                 atIndex:0];
            [enc setBuffer:mp.lm_head.buf     offset:mp.lm_head.offset atIndex:1];
            [enc setBuffer:mp.base_logits_buf offset:0                 atIndex:2];
            [enc setBytes:&prm length:sizeof(prm) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((V_UNPAD + 7) / 8), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            ++n_dispatches;
        }

        {
            MTPMixPrm prm{D, mc.n_target_layers, n_ctx - 1, mc.num_layers, m_eps};
            [enc setComputePipelineState:mp.pso_dflare_mix];
            [enc setBuffer:mp.hidden_buf             offset:0                             atIndex:0];
            [enc setBuffer:mp.fuse_alpha.buf         offset:mp.fuse_alpha.offset         atIndex:1];
            [enc setBuffer:mp.markov_dflare_norm.buf offset:mp.markov_dflare_norm.offset atIndex:2];
            [enc setBuffer:mp.f_rnn_buf              offset:0                             atIndex:3];
            [enc setBytes:&prm length:sizeof(prm) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            ++n_dispatches;
        }
        {
            const int32_t out_dim = 3 * mc.markov_rank;
            MTPMarkovCtxPrm prm{D, out_dim};
            [enc setComputePipelineState:mp.pso_markov_ctx_p1[num_draft]];
            [enc setBuffer:mp.x_out_buf         offset:0                      atIndex:0];
            [enc setBuffer:mp.f_rnn_buf         offset:0                      atIndex:1];
            [enc setBuffer:mp.rnn_w_ctx_t.buf   offset:mp.rnn_w_ctx_t.offset  atIndex:2];
            [enc setBuffer:mp.markov_ctx_part_x offset:0                      atIndex:3];
            [enc setBuffer:mp.markov_ctx_part_f offset:0                      atIndex:4];
            [enc setBytes:&prm length:sizeof(prm) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((out_dim + 31) / 32), 16, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            ++n_dispatches;

            [enc setComputePipelineState:mp.pso_markov_ctx_p2[num_draft]];
            [enc setBuffer:mp.markov_ctx_part_x offset:0                    atIndex:0];
            [enc setBuffer:mp.markov_ctx_part_f offset:0                    atIndex:1];
            [enc setBuffer:mp.rnn_b_f32.buf     offset:mp.rnn_b_f32.offset  atIndex:2];
            [enc setBuffer:mp.rnn_ctx_buf       offset:0                    atIndex:3];
            [enc setBytes:&prm length:sizeof(prm) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((out_dim + 31) / 32), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            ++n_dispatches;
        }

        for (int s = 0; s < num_draft; ++s) {
            {
                MTPRnnStepPrm prm{s, mc.markov_rank};
                [enc setComputePipelineState:mp.pso_rnn_step];
                [enc setBuffer:mp.rnn_state_buf[s & 1]       offset:0                       atIndex:0];
                [enc setBuffer:mp.prev_tok_buf               offset:0                       atIndex:1];
                [enc setBuffer:mp.markov_w1.buf              offset:mp.markov_w1.offset     atIndex:2];
                [enc setBuffer:mp.rnn_w_step_t.buf           offset:mp.rnn_w_step_t.offset  atIndex:3];
                [enc setBuffer:mp.rnn_ctx_buf                offset:0                       atIndex:4];
                [enc setBuffer:mp.rnn_state_buf[(s & 1) ^ 1] offset:0                       atIndex:5];
                [enc setBuffer:mp.rnn_latent_buf             offset:0                       atIndex:6];
                [enc setBytes:&prm length:sizeof(prm) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(mc.markov_rank / 32), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                ++n_dispatches;
            }
            if (!is_sampled) {
                MTPMarkovArgmaxPrm prm{V_UNPAD, s, n_markov_tg};
                [enc setComputePipelineState:mp.pso_markov_argmax_p1];
                [enc setBuffer:mp.rnn_latent_buf  offset:0                    atIndex:0];
                [enc setBuffer:mp.fold_weight.buf offset:mp.fold_weight.offset atIndex:1];
                [enc setBuffer:mp.fold_bias.buf   offset:mp.fold_bias.offset   atIndex:2];
                [enc setBuffer:mp.base_logits_buf offset:0                    atIndex:3];
                [enc setBuffer:mp.markov_part_val offset:0                    atIndex:4];
                [enc setBuffer:mp.markov_part_idx offset:0                    atIndex:5];
                [enc setBytes:&prm length:sizeof(prm) atIndex:6];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_markov_tg, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_dispatches;

                [enc setComputePipelineState:mp.pso_markov_argmax_p2];
                [enc setBuffer:mp.markov_part_val offset:0 atIndex:0];
                [enc setBuffer:mp.markov_part_idx offset:0 atIndex:1];
                [enc setBuffer:mp.d_toks_buf      offset:0 atIndex:2];
                [enc setBuffer:mp.prev_tok_buf    offset:0 atIndex:3];
                [enc setBuffer:mp.verify_toks_buf offset:0 atIndex:4];
                [enc setBytes:&prm length:sizeof(prm) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_dispatches;
            } else {
                {
                    MTPMarkovLogitsPrm prm{V_UNPAD, s, n_markov_tg, temperature, logit_softcap};
                    [enc setComputePipelineState:mp.pso_markov_logits_max_p1];
                    [enc setBuffer:mp.rnn_latent_buf   offset:0                    atIndex:0];
                    [enc setBuffer:mp.fold_weight.buf  offset:mp.fold_weight.offset atIndex:1];
                    [enc setBuffer:mp.fold_bias.buf    offset:mp.fold_bias.offset   atIndex:2];
                    [enc setBuffer:mp.base_logits_buf  offset:0                    atIndex:3];
                    [enc setBuffer:mp.q_probs_buf      offset:0                    atIndex:4];
                    [enc setBuffer:mp.softmax_part_buf offset:0                    atIndex:5];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:6];
                    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_markov_tg, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    ++n_dispatches;
                }
                {
                    MTPSoftmaxRedPrm prm{V_UNPAD, n_markov_tg, s};
                    [enc setComputePipelineState:mp.pso_softmax_max_p2];
                    [enc setBuffer:mp.softmax_part_buf offset:0 atIndex:0];
                    [enc setBuffer:mp.row_max_buf      offset:0 atIndex:1];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:2];
                    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    ++n_dispatches;
                }
                {
                    MTPSoftmaxVecPrm prm{V_UNPAD, n_tg_vec, s, 0, top_p};
                    [enc setComputePipelineState:mp.pso_softmax_exp_sum_p1];
                    [enc setBuffer:mp.q_probs_buf      offset:0 atIndex:0];
                    [enc setBuffer:mp.row_max_buf      offset:0 atIndex:1];
                    [enc setBuffer:mp.softmax_part_buf offset:0 atIndex:2];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tg_vec, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    ++n_dispatches;
                }
                {
                    MTPSoftmaxRedPrm prm{V_UNPAD, n_tg_vec, s};
                    [enc setComputePipelineState:mp.pso_softmax_sum_p2];
                    [enc setBuffer:mp.softmax_part_buf offset:0 atIndex:0];
                    [enc setBuffer:mp.row_inv_sum_buf  offset:0 atIndex:1];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:2];
                    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    ++n_dispatches;
                }
                if (do_top_p) {
                    for (int pass = 0; pass < 6; ++pass) {
                        MTPSoftmaxVecPrm prm{V_UNPAD, n_tg_vec, s, pass, top_p};
                        [enc setComputePipelineState:mp.pso_topp_hist];
                        [enc setBuffer:mp.q_probs_buf       offset:0 atIndex:0];
                        [enc setBuffer:mp.row_inv_sum_buf   offset:0 atIndex:1];
                        [enc setBuffer:mp.topp_states_buf   offset:0 atIndex:2];
                        [enc setBuffer:mp.softmax_part_buf  offset:0 atIndex:3];
                        [enc setBuffer:mp.topp_part_cnt_buf offset:0 atIndex:4];
                        [enc setBytes:&prm length:sizeof(prm) atIndex:5];
                        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tg_vec, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        ++n_dispatches;

                        [enc setComputePipelineState:mp.pso_topp_scan];
                        [enc setBuffer:mp.softmax_part_buf  offset:0 atIndex:0];
                        [enc setBuffer:mp.topp_part_cnt_buf offset:0 atIndex:1];
                        [enc setBuffer:mp.topp_states_buf   offset:0 atIndex:2];
                        [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                        ++n_dispatches;
                    }
                    {
                        MTPSoftmaxVecPrm prm{V_UNPAD, n_tg_vec, s, 0, top_p};
                        [enc setComputePipelineState:mp.pso_topp_mask_sum];
                        [enc setBuffer:mp.q_probs_buf      offset:0 atIndex:0];
                        [enc setBuffer:mp.topp_states_buf  offset:0 atIndex:1];
                        [enc setBuffer:mp.softmax_part_buf offset:0 atIndex:2];
                        [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tg_vec, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        ++n_dispatches;
                    }
                    {
                        MTPSoftmaxRedPrm prm{V_UNPAD, n_tg_vec, s};
                        [enc setComputePipelineState:mp.pso_softmax_sum_p2];
                        [enc setBuffer:mp.softmax_part_buf offset:0 atIndex:0];
                        [enc setBuffer:mp.row_inv_sum_buf  offset:0 atIndex:1];
                        [enc setBytes:&prm length:sizeof(prm) atIndex:2];
                        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        ++n_dispatches;
                    }
                }
                {
                    MTPScaleGumbelPrm prm{rng_seed, rng_step, V_UNPAD, n_tg_vec, s, 0};
                    [enc setComputePipelineState:mp.pso_probs_scale_gumbel_p1];
                    [enc setBuffer:mp.q_probs_buf     offset:0 atIndex:0];
                    [enc setBuffer:mp.row_inv_sum_buf offset:0 atIndex:1];
                    [enc setBuffer:mp.markov_part_val offset:0 atIndex:2];
                    [enc setBuffer:mp.markov_part_idx offset:0 atIndex:3];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tg_vec, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    ++n_dispatches;

                    [enc setComputePipelineState:mp.pso_gumbel_p2];
                    [enc setBuffer:mp.markov_part_val offset:0 atIndex:0];
                    [enc setBuffer:mp.markov_part_idx offset:0 atIndex:1];
                    [enc setBuffer:mp.d_toks_buf      offset:0 atIndex:2];
                    [enc setBuffer:mp.prev_tok_buf    offset:0 atIndex:3];
                    [enc setBuffer:mp.verify_toks_buf offset:0 atIndex:4];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    ++n_dispatches;
                }
            }
        }

        {
            const float embed_eps = ctx->globals.embed_norm_eps;
            [enc setComputePipelineState:ctx->pso_embed_from_buf];
            [enc setBuffer:ctx->globals.embed_tokens.buf offset:ctx->globals.embed_tokens.offset atIndex:0];
            [enc setBuffer:ctx->globals.embed_norm.buf   offset:ctx->globals.embed_norm.offset   atIndex:1];
            [enc setBuffer:ctx->scratch                  offset:ctx->off_x                       atIndex:2];
            [enc setBuffer:mp.verify_toks_buf            offset:0                                atIndex:3];
            [enc setBytes:&embed_eps length:sizeof(embed_eps) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)M, 1)
                threadsPerThreadgroup:MTLSizeMake((NSUInteger)ctx->mix_tg, 1, 1)];
            ++n_dispatches;
        }

        encode_target_visits_and_norm(
            ctx, enc, M, 1, kv_meta,
            mp.capture_layer_ids.data(), (int32_t)mp.capture_layer_ids.size(),
            true, true,
            false, BufSlice{nil, 0},
            true, 0,
            pso_grp_w13_nv, pso_grp_w2_nv, pso_grp_w13_mx, pso_grp_w2_mx,
            n_dispatches);

        if (!is_sampled) {
            ArgmaxP1Prm p1{n_tg_target};
            [enc setComputePipelineState:pso_argmax_p1];
            [enc setBuffer:ctx->scratch         offset:ctx->off_normed_f32    atIndex:0];
            [enc setBuffer:target_lm_head.buf   offset:target_lm_head.offset  atIndex:1];
            [enc setBuffer:ctx->scratch         offset:ctx->off_argmax_pv     atIndex:2];
            [enc setBuffer:ctx->scratch         offset:ctx->off_argmax_pi     atIndex:3];
            [enc setBytes:&p1 length:sizeof(p1) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tg_target, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(8 * 32, 1, 1)];
            ++n_dispatches;

            ArgmaxP2Prm p2{n_tg_target};
            [enc setComputePipelineState:ctx->pso_argmax_p2];
            [enc setBuffer:ctx->scratch offset:ctx->off_argmax_pv atIndex:0];
            [enc setBuffer:ctx->scratch offset:ctx->off_argmax_pi atIndex:1];
            [enc setBuffer:ctx->scratch offset:ctx->off_out_toks  atIndex:2];
            [enc setBytes:&p2 length:sizeof(p2) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)M, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            ++n_dispatches;
        } else {
            {
                LogitsMaxP1Prm prm{n_tg_target, temperature, logit_softcap};
                [enc setComputePipelineState:pso_logits_max_p1];
                [enc setBuffer:ctx->scratch         offset:ctx->off_normed_f32   atIndex:0];
                [enc setBuffer:target_lm_head.buf   offset:target_lm_head.offset atIndex:1];
                [enc setBuffer:mp.p_probs_buf       offset:0                     atIndex:2];
                [enc setBuffer:mp.softmax_part_buf  offset:0                     atIndex:3];
                [enc setBytes:&prm length:sizeof(prm) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tg_target, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(8 * 32, 1, 1)];
                ++n_dispatches;
            }
            {
                MTPSoftmaxRedPrm prm{V_UNPAD, n_tg_target, 0};
                [enc setComputePipelineState:mp.pso_softmax_max_p2];
                [enc setBuffer:mp.softmax_part_buf offset:0 atIndex:0];
                [enc setBuffer:mp.row_max_buf      offset:0 atIndex:1];
                [enc setBytes:&prm length:sizeof(prm) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)M, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_dispatches;
            }
            {
                MTPSoftmaxVecPrm prm{V_UNPAD, n_tg_vec, 0, 0, top_p};
                [enc setComputePipelineState:mp.pso_softmax_exp_sum_p1];
                [enc setBuffer:mp.p_probs_buf      offset:0 atIndex:0];
                [enc setBuffer:mp.row_max_buf      offset:0 atIndex:1];
                [enc setBuffer:mp.softmax_part_buf offset:0 atIndex:2];
                [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tg_vec, (NSUInteger)M, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_dispatches;
            }
            {
                MTPSoftmaxRedPrm prm{V_UNPAD, n_tg_vec, 0};
                [enc setComputePipelineState:mp.pso_softmax_sum_p2];
                [enc setBuffer:mp.softmax_part_buf offset:0 atIndex:0];
                [enc setBuffer:mp.row_inv_sum_buf  offset:0 atIndex:1];
                [enc setBytes:&prm length:sizeof(prm) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)M, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_dispatches;
            }
            if (do_top_p) {
                for (int pass = 0; pass < 6; ++pass) {
                    MTPSoftmaxVecPrm prm{V_UNPAD, n_tg_vec, 0, pass, top_p};
                    [enc setComputePipelineState:mp.pso_topp_hist];
                    [enc setBuffer:mp.p_probs_buf       offset:0 atIndex:0];
                    [enc setBuffer:mp.row_inv_sum_buf   offset:0 atIndex:1];
                    [enc setBuffer:mp.topp_states_buf   offset:0 atIndex:2];
                    [enc setBuffer:mp.softmax_part_buf  offset:0 atIndex:3];
                    [enc setBuffer:mp.topp_part_cnt_buf offset:0 atIndex:4];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tg_vec, (NSUInteger)M, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    ++n_dispatches;

                    [enc setComputePipelineState:mp.pso_topp_scan];
                    [enc setBuffer:mp.softmax_part_buf  offset:0 atIndex:0];
                    [enc setBuffer:mp.topp_part_cnt_buf offset:0 atIndex:1];
                    [enc setBuffer:mp.topp_states_buf   offset:0 atIndex:2];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)M, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    ++n_dispatches;
                }
                {
                    MTPSoftmaxVecPrm prm{V_UNPAD, n_tg_vec, 0, 0, top_p};
                    [enc setComputePipelineState:mp.pso_topp_mask_sum];
                    [enc setBuffer:mp.p_probs_buf      offset:0 atIndex:0];
                    [enc setBuffer:mp.topp_states_buf  offset:0 atIndex:1];
                    [enc setBuffer:mp.softmax_part_buf offset:0 atIndex:2];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tg_vec, (NSUInteger)M, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    ++n_dispatches;
                }
                {
                    MTPSoftmaxRedPrm prm{V_UNPAD, n_tg_vec, 0};
                    [enc setComputePipelineState:mp.pso_softmax_sum_p2];
                    [enc setBuffer:mp.softmax_part_buf offset:0 atIndex:0];
                    [enc setBuffer:mp.row_inv_sum_buf  offset:0 atIndex:1];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:2];
                    [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)M, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    ++n_dispatches;
                }
            }
            {
                MTPSoftmaxVecPrm prm{V_UNPAD, n_tg_vec, 0, 0, top_p};
                [enc setComputePipelineState:mp.pso_probs_scale];
                [enc setBuffer:mp.p_probs_buf     offset:0 atIndex:0];
                [enc setBuffer:mp.row_inv_sum_buf offset:0 atIndex:1];
                [enc setBytes:&prm length:sizeof(prm) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tg_vec, (NSUInteger)M, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_dispatches;
            }
        }

        if (!is_sampled) {
            MTPGreedyAcceptPrm prm{num_draft};
            [enc setComputePipelineState:mp.pso_greedy_accept];
            [enc setBuffer:mp.d_toks_buf   offset:0                  atIndex:0];
            [enc setBuffer:ctx->scratch    offset:ctx->off_out_toks  atIndex:1];
            [enc setBuffer:mp.out_toks_buf offset:0                  atIndex:2];
            [enc setBuffer:mp.d_n_acc_buf  offset:0                  atIndex:3];
            [enc setBytes:&prm length:sizeof(prm) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            ++n_dispatches;
        } else {
            {
                MTPRejectPrm prm{rng_seed, rng_step, V_UNPAD, num_draft};
                [enc setComputePipelineState:mp.pso_reject_prefix];
                [enc setBuffer:mp.d_toks_buf     offset:0 atIndex:0];
                [enc setBuffer:mp.p_probs_buf    offset:0 atIndex:1];
                [enc setBuffer:mp.q_probs_buf    offset:0 atIndex:2];
                [enc setBuffer:mp.out_toks_buf   offset:0 atIndex:3];
                [enc setBuffer:mp.d_reject_j_buf offset:0 atIndex:4];
                [enc setBuffer:mp.d_n_acc_buf    offset:0 atIndex:5];
                [enc setBytes:&prm length:sizeof(prm) atIndex:6];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                ++n_dispatches;
            }
            {
                MTPRejectSamplePrm prm{rng_seed, rng_step, V_UNPAD, n_tg_vec, num_draft, 0};
                [enc setComputePipelineState:mp.pso_reject_sample_p1];
                [enc setBuffer:mp.p_probs_buf    offset:0 atIndex:0];
                [enc setBuffer:mp.q_probs_buf    offset:0 atIndex:1];
                [enc setBuffer:mp.d_reject_j_buf offset:0 atIndex:2];
                [enc setBuffer:mp.reject_res_val offset:0 atIndex:3];
                [enc setBuffer:mp.reject_res_idx offset:0 atIndex:4];
                [enc setBuffer:mp.reject_p_val   offset:0 atIndex:5];
                [enc setBuffer:mp.reject_p_idx   offset:0 atIndex:6];
                [enc setBuffer:mp.reject_res_sum offset:0 atIndex:7];
                [enc setBytes:&prm length:sizeof(prm) atIndex:8];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tg_vec, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_dispatches;

                [enc setComputePipelineState:mp.pso_reject_sample_p2];
                [enc setBuffer:mp.reject_res_val offset:0 atIndex:0];
                [enc setBuffer:mp.reject_res_idx offset:0 atIndex:1];
                [enc setBuffer:mp.reject_p_val   offset:0 atIndex:2];
                [enc setBuffer:mp.reject_p_idx   offset:0 atIndex:3];
                [enc setBuffer:mp.reject_res_sum offset:0 atIndex:4];
                [enc setBuffer:mp.d_reject_j_buf offset:0 atIndex:5];
                [enc setBuffer:mp.out_toks_buf   offset:0 atIndex:6];
                [enc setBytes:&prm length:sizeof(prm) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_dispatches;
            }
        }

        for (int v = 0; v < NV; ++v) {
            const PendingVerify& pv = ctx->pending_verify[v];
            const LayerDeviceWeights& L = ctx->layers[ctx->schedule[v].layer_idx];
            const InklingKVMeta& km = kv_meta[v];
            if (pv.kv_staged) {
                const size_t vkv_off = (size_t)v * ctx->verify_kv_visit_stride_bytes;
                KvCommitDynPrm kcprm{M, L.n_kv, L.head_dim, pv.old_idx, pv.max_size, km.stride_h, km.stride_s};
                [enc setComputePipelineState:ctx->pso_kv_commit_dyn];
                [enc setBuffer:ctx->scratch         offset:ctx->off_verify_k + vkv_off atIndex:0];
                [enc setBuffer:ctx->scratch         offset:ctx->off_verify_v + vkv_off atIndex:1];
                [enc setBuffer:ctx->tmp_kv_k[v].buf offset:ctx->tmp_kv_k[v].offset      atIndex:2];
                [enc setBuffer:ctx->tmp_kv_v[v].buf offset:ctx->tmp_kv_v[v].offset      atIndex:3];
                [enc setBuffer:mp.d_n_acc_buf       offset:0                             atIndex:4];
                [enc setBytes:&kcprm length:sizeof(kcprm) atIndex:5];
                const uint32_t total = (uint32_t)(M * L.n_kv * L.head_dim);
                [enc dispatchThreadgroups:MTLSizeMake((total + 255) / 256, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_dispatches;
            }
            {
                const size_t ch = (size_t)L.n_kv * (size_t)L.head_dim;
                const size_t visit_c_bytes = (size_t)M * 3 * ch * sizeof(float);
                const size_t visit_d_bytes = (size_t)M * 3 * (size_t)D * sizeof(float);
                const size_t cur = ctx->conv_hist_visit_off[v];
                const uint32_t nc = (uint32_t)(3 * ch);
                const uint32_t nd = (uint32_t)(3 * (size_t)D);
                const uint32_t nmax = std::max(nc, nd);
                CopyConv4DynPrm c4prm{nc, nd, M};
                [enc setComputePipelineState:ctx->pso_copy_conv4_dyn];
                [enc setBuffer:ctx->conv_hist_buf        offset:cur                                     atIndex:0];
                [enc setBuffer:ctx->conv_hist_buf        offset:cur + visit_c_bytes                     atIndex:1];
                [enc setBuffer:ctx->conv_hist_buf        offset:cur + 2 * visit_c_bytes                 atIndex:2];
                [enc setBuffer:ctx->conv_hist_buf        offset:cur + 2 * visit_c_bytes + visit_d_bytes atIndex:3];
                [enc setBuffer:ctx->tmp_conv[v * 4 + 0].buf offset:ctx->tmp_conv[v * 4 + 0].offset      atIndex:4];
                [enc setBuffer:ctx->tmp_conv[v * 4 + 1].buf offset:ctx->tmp_conv[v * 4 + 1].offset      atIndex:5];
                [enc setBuffer:ctx->tmp_conv[v * 4 + 2].buf offset:ctx->tmp_conv[v * 4 + 2].offset      atIndex:6];
                [enc setBuffer:ctx->tmp_conv[v * 4 + 3].buf offset:ctx->tmp_conv[v * 4 + 3].offset      atIndex:7];
                [enc setBuffer:mp.d_n_acc_buf            offset:0                                       atIndex:8];
                [enc setBytes:&c4prm length:sizeof(c4prm) atIndex:9];
                [enc dispatchThreadgroups:MTLSizeMake((nmax + 255) / 256, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_dispatches;
            }
        }
        {
            const int32_t row_elems = NCAP * D;
            CopyCapDynPrm cprm{M, row_elems};
            [enc setComputePipelineState:ctx->pso_copy_cap_dyn];
            [enc setBuffer:ctx->scratch         offset:ctx->off_captures    atIndex:0];
            [enc setBuffer:out_hidden_slice.buf offset:out_hidden_slice.offset atIndex:1];
            [enc setBuffer:mp.d_n_acc_buf       offset:0                    atIndex:2];
            [enc setBytes:&cprm length:sizeof(cprm) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((row_elems + 255) / 256), (NSUInteger)M, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            ++n_dispatches;
        }

        [enc endEncoding];
        const uint64_t t_enc1 = mach_absolute_time();

        [cb commit];
        [cb waitUntilCompleted];
        const uint64_t t_wall1 = mach_absolute_time();

        if (cb.status != MTLCommandBufferStatusCompleted) {
            fprintf(stderr, "[inkling_decode] ERROR: unified mtp command buffer failed: %s\n",
                    cb.error ? cb.error.localizedDescription.UTF8String : "unknown");
            for (int v = 0; v < NV; ++v) {
                kv_meta[v].offset = ctx->pending_verify[v].prev_offset;
                kv_meta[v].idx    = ctx->pending_verify[v].old_idx;
                ctx->pending_verify[v].kv_staged = false;
                ctx->pending_rot[v].active = false;
            }
            return -10;
        }

        const int32_t n_acc = std::max<int32_t>(1, std::min<int32_t>(M, *(const int32_t*)[mp.d_n_acc_buf contents]));
        const int32_t* gpu_out_toks = (const int32_t*)[mp.out_toks_buf contents];
        for (int32_t j = 0; j < n_acc; ++j) {
            out_tokens[j] = gpu_out_toks[j];
        }
        *out_n_acc = n_acc;
        mp.last_num_draft = num_draft;

        for (int v = 0; v < NV; ++v) {
            PendingVerify& pv = ctx->pending_verify[v];
            InklingKVMeta& km = kv_meta[v];
            if (pv.kv_staged) {
                km.offset = pv.prev_offset + n_acc;
                km.idx    = (km.offset <= pv.max_size)
                          ? km.offset
                          : (((pv.old_idx + n_acc - 1) % pv.max_size) + 1);
            } else {
                km.offset = pv.prev_offset + n_acc;
                km.idx    = pv.prev_offset + n_acc;
            }
            pv.kv_staged = false;
            ctx->pending_rot[v].active = false;
        }

        for (int i = 0; i < mc.num_layers; ++i) {
            draft_kv_descs[i].offset   = d_off_after[i];
            draft_kv_descs[i].ring_idx = d_next_ring[i];
        }

        if (stats_out) {
            stats_out->encode_ms    = mach_to_ms(t_enc1 - t_enc0, ctx->tb);
            stats_out->gpu_ms       = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
            stats_out->wall_ms      = mach_to_ms(t_wall1 - t_wall0, ctx->tb);
            stats_out->n_dispatches = n_dispatches;
        }
        return 0;
    }
}

static int ensure_mtp_batched_state(InklingCtx* ctx, bool need_sampling) {
    MTPAttachedState& mp = ctx->mtp;
    const MTPDraftConfig& mc = mp.cfg;
    const int32_t NV = (int32_t)ctx->schedule.size();
    const int32_t max_mv = std::min<int32_t>(ctx->cfg.max_m, 32);

    if (!ctx->pso_final_norm_bf16_cap) {
        ctx->pso_final_norm_bf16_cap    = make_pso(ctx->device, ctx->lib, @"inkling_final_rms_norm_bf16_cap");
        ctx->pso_combine_mtp_batched    = make_pso(ctx->device, ctx->lib, @"inkling_combine_mtp_batched");
        ctx->pso_kv_commit_rot_mtp_batched = make_pso(ctx->device, ctx->lib, @"inkling_kv_commit_rotating_mtp_batched");
        ctx->pso_copy_conv4_mtp_batched = make_pso(ctx->device, ctx->lib, @"inkling_copy_conv4_mtp_batched");
        ctx->pso_copy_cap_mtp_batched   = make_pso(ctx->device, ctx->lib, @"inkling_copy_capture_mtp_batched");
        if (!ctx->pso_final_norm_bf16_cap || !ctx->pso_combine_mtp_batched ||
            !ctx->pso_kv_commit_rot_mtp_batched || !ctx->pso_copy_conv4_mtp_batched ||
            !ctx->pso_copy_cap_mtp_batched) {
            return -8;
        }
        for (int k = 0; k <= 4; ++k) {
            get_mix_mtp_batched_pso(ctx, false, k);
            get_mix_mtp_batched_pso(ctx, true, k);
        }
        for (int v = 0; v < NV; ++v) {
            const LayerDeviceWeights& L = ctx->layers[ctx->schedule[v].layer_idx];
            const int g = L.n_heads / L.n_kv;
            get_kv_prep_mtp_batched_pso(ctx, !L.is_sliding, L.head_dim);
            get_attend_mtp_batched_pso(ctx, g, L.head_dim);
            get_dpart_mtp_batched_pso(ctx, g, L.head_dim);
        }
        for (int m = 1; m <= 8; ++m) {
            if (!get_head_logits_max_mixed_p1_pso(ctx, m)) return -8;
        }
    }
    if (ctx->lib_nax && !ctx->pso_nax_head_logits_max_16x64) {
        ctx->pso_nax_head_logits_max_16x64 = make_pso(ctx->device, ctx->lib_nax, @"inkling_nax_head_logits_max_p1_16x64");
        ctx->pso_mtp_nax_gu_silu_16x64     = make_pso(ctx->device, ctx->lib_nax, @"mtp_nax_gu_silu_16x64");
        ctx->pso_mtp_nax_gu_silu_32x64     = make_pso(ctx->device, ctx->lib_nax, @"mtp_nax_gu_silu_32x64");
        if (!ctx->pso_nax_head_logits_max_16x64 || !ctx->pso_mtp_nax_gu_silu_16x64 ||
            !ctx->pso_mtp_nax_gu_silu_32x64) {
            return -8;
        }
    }

    auto alloc_s = [&](size_t bytes) -> id<MTLBuffer> {
        return [ctx->device newBufferWithLength:align_up(std::max<size_t>(bytes, 256), 256)
                                        options:MTLResourceStorageModeShared];
    };

    const size_t V_UNPAD = (size_t)mc.unpadded_vocab_size;
    const size_t n_tg_target_simt = (V_UNPAD + 15) / 16;
    const size_t n_tg_vec = std::min<size_t>(64, std::max<size_t>(1, (V_UNPAD + 255) / 256));
    const size_t n_markov_tg = (V_UNPAD + 63) / 64;

    if (!mp.batched_init || mp.batched_max_mv < max_mv) {
        mp.pso_batch_init_x[0] = make_mtp_pso(ctx->device, mp.lib, @"mtp_batch_init_x", -1, -1, -1, -1, 0);
        mp.pso_batch_init_x[1] = make_mtp_pso(ctx->device, mp.lib, @"mtp_batch_init_x", -1, -1, -1, -1, 1);
        mp.pso_batch_gemv                  = make_mtp_pso(ctx->device, mp.lib, @"mtp_bf16_gemv_batched");
        mp.pso_batch_gu_silu               = make_mtp_pso(ctx->device, mp.lib, @"mtp_bf16_gu_silu_batched");
        mp.pso_batch_kv_ctx_write          = make_mtp_pso(ctx->device, mp.lib, @"mtp_batch_kv_ctx_norm_rope_write");
        mp.pso_batch_qkv_pack              = make_mtp_pso(ctx->device, mp.lib, @"mtp_batch_qkv_norm_rope_pack");
        mp.pso_batch_sdpa_single           = make_mtp_pso(ctx->device, mp.lib, @"mtp_batch_sdpa_single");
        mp.pso_batch_sdpa_split_p1         = make_mtp_pso(ctx->device, mp.lib, @"mtp_batch_sdpa_split_p1");
        mp.pso_batch_sdpa_split_p2         = make_mtp_pso(ctx->device, mp.lib, @"mtp_batch_sdpa_split_p2");
        mp.pso_batch_final_norm            = make_mtp_pso(ctx->device, mp.lib, @"mtp_batch_final_norm");
        mp.pso_batch_f_rnn                 = make_mtp_pso(ctx->device, mp.lib, @"mtp_batch_f_rnn");
        mp.pso_batch_markov_ctx_p1         = make_mtp_pso(ctx->device, mp.lib, @"mtp_batch_markov_ctx_p1");
        mp.pso_batch_markov_ctx_p2         = make_mtp_pso(ctx->device, mp.lib, @"mtp_batch_markov_ctx_p2");
        mp.pso_batch_rnn_step              = make_mtp_pso(ctx->device, mp.lib, @"mtp_batch_rnn_step");
        mp.pso_batch_markov_mixed_p1       = make_mtp_pso(ctx->device, mp.lib, @"mtp_batch_markov_mixed_p1");
        mp.pso_batch_markov_argmax_p2      = make_mtp_pso(ctx->device, mp.lib, @"mtp_batch_markov_argmax_p2");
        mp.pso_batch_softmax_max_p2        = make_mtp_pso(ctx->device, mp.lib, @"mtp_batch_softmax_max_p2");
        mp.pso_batch_softmax_exp_sum_p1    = make_mtp_pso(ctx->device, mp.lib, @"mtp_batch_softmax_exp_sum_p1");
        mp.pso_batch_softmax_sum_p2        = make_mtp_pso(ctx->device, mp.lib, @"mtp_batch_softmax_sum_p2");
        mp.pso_batch_topp_hist             = make_mtp_pso(ctx->device, mp.lib, @"mtp_batch_topp_hist");
        mp.pso_batch_topp_scan             = make_mtp_pso(ctx->device, mp.lib, @"mtp_batch_topp_scan");
        mp.pso_batch_topp_mask_sum         = make_mtp_pso(ctx->device, mp.lib, @"mtp_batch_topp_mask_sum");
        mp.pso_batch_probs_scale           = make_mtp_pso(ctx->device, mp.lib, @"mtp_batch_probs_scale");
        mp.pso_batch_probs_scale_gumbel_p1 = make_mtp_pso(ctx->device, mp.lib, @"mtp_batch_probs_scale_gumbel_p1");
        mp.pso_batch_gumbel_p2             = make_mtp_pso(ctx->device, mp.lib, @"mtp_batch_gumbel_p2");
        mp.pso_batch_greedy_accept         = make_mtp_pso(ctx->device, mp.lib, @"mtp_batch_greedy_accept");
        mp.pso_batch_reject_prefix         = make_mtp_pso(ctx->device, mp.lib, @"mtp_batch_reject_prefix");
        mp.pso_batch_reject_sample_p1      = make_mtp_pso(ctx->device, mp.lib, @"mtp_batch_reject_sample_p1");
        mp.pso_batch_reject_sample_p2      = make_mtp_pso(ctx->device, mp.lib, @"mtp_batch_reject_sample_p2");

        if (!mp.pso_batch_init_x[0] || !mp.pso_batch_init_x[1] || !mp.pso_batch_gemv || !mp.pso_batch_gu_silu ||
            !mp.pso_batch_kv_ctx_write || !mp.pso_batch_qkv_pack || !mp.pso_batch_sdpa_single ||
            !mp.pso_batch_sdpa_split_p1 || !mp.pso_batch_sdpa_split_p2 || !mp.pso_batch_final_norm ||
            !mp.pso_batch_f_rnn || !mp.pso_batch_markov_ctx_p1 || !mp.pso_batch_markov_ctx_p2 ||
            !mp.pso_batch_rnn_step || !mp.pso_batch_markov_mixed_p1 || !mp.pso_batch_markov_argmax_p2 ||
            !mp.pso_batch_softmax_max_p2 || !mp.pso_batch_softmax_exp_sum_p1 || !mp.pso_batch_softmax_sum_p2 ||
            !mp.pso_batch_topp_hist || !mp.pso_batch_topp_scan || !mp.pso_batch_topp_mask_sum ||
            !mp.pso_batch_probs_scale || !mp.pso_batch_probs_scale_gumbel_p1 || !mp.pso_batch_gumbel_p2 ||
            !mp.pso_batch_greedy_accept || !mp.pso_batch_reject_prefix || !mp.pso_batch_reject_sample_p1 ||
            !mp.pso_batch_reject_sample_p2) {
            return -8;
        }

        const size_t MAX_STREAMS = 4;
        const size_t D = (size_t)mc.hidden_size;
        const size_t H = (size_t)mc.num_heads;
        const size_t KVH = (size_t)mc.num_kv_heads;
        const size_t HD = (size_t)mc.head_dim;
        const size_t KV_DIM = KVH * HD;
        const size_t Q_DIM = H * HD;
        const size_t QKV_DIM = Q_DIM + 2 * KV_DIM;
        const size_t INTER = (size_t)mc.inter_size;
        const size_t B_BLK = (size_t)mc.block_size;
        const size_t MAX_MQ = MAX_STREAMS * B_BLK;
        const size_t MAX_MD = MAX_STREAMS * (B_BLK - 1);
        const size_t MAX_MV = (size_t)max_mv;
        const size_t RANK = (size_t)mc.markov_rank;
        const size_t NCAP = (size_t)mc.n_target_layers;
        const size_t kMaxSdpaSplits = 32;

        mp.batched_verify_kv_visit_stride_bytes = align_up(MAX_MV * (size_t)ctx->cfg.max_n_kv * (size_t)ctx->cfg.max_head_dim * sizeof(uint16_t), 256);
        mp.batched_conv_hist_visit_off.resize((size_t)NV);
        size_t ch_cur = 0;
        for (int v = 0; v < NV; ++v) {
            const auto& L = ctx->layers[ctx->schedule[v].layer_idx];
            const size_t ch = (size_t)L.n_kv * (size_t)L.head_dim;
            mp.batched_conv_hist_visit_off[(size_t)v] = ch_cur;
            const size_t bytes = MAX_MV * 3 * (2 * ch + 2 * D) * sizeof(float);
            ch_cur = align_up(ch_cur + bytes, 256);
        }

        mp.batch_stream_meta_buf   = alloc_s(MAX_STREAMS * sizeof(BatchMTPStreamMetaGpu));
        mp.batch_draft_kv_meta_buf = alloc_s(MAX_STREAMS * (size_t)mc.num_layers * sizeof(BatchMTPDraftKVMetaGpu));
        mp.batch_target_vmeta_buf  = alloc_s((size_t)NV * MAX_STREAMS * sizeof(BatchMTPTargetVisitMetaGpu));
        mp.batch_ctx_to_stream_buf = alloc_s(MAX_MQ * sizeof(int32_t));
        mp.batch_ctx_local_pos_buf = alloc_s(MAX_MQ * sizeof(int32_t));
        mp.batch_d_to_stream_buf   = alloc_s(MAX_MD * sizeof(int32_t));
        mp.batch_d_local_step_buf  = alloc_s(MAX_MD * sizeof(int32_t));
        mp.batch_tok_to_stream_buf = alloc_s(MAX_MV * sizeof(int32_t));
        mp.batch_tok_local_pos_buf = alloc_s(MAX_MV * sizeof(int32_t));
        mp.batch_hidden_buf        = alloc_s(MAX_MQ * NCAP * D * 2);
        mp.batch_x_buf             = alloc_s(MAX_MQ * D * 2);
        mp.batch_residual_buf      = alloc_s(MAX_MQ * D * 2);
        mp.batch_norm_buf          = alloc_s(MAX_MQ * D * 2);
        mp.batch_ctx_buf           = alloc_s(MAX_MQ * D * 2);
        mp.batch_kv_ctx_buf        = alloc_s(MAX_MQ * 2 * KV_DIM * 2);
        mp.batch_qkv_buf           = alloc_s(MAX_MQ * QKV_DIM * 2);
        mp.batch_q_buf             = alloc_s(MAX_STREAMS * H * B_BLK * HD * 2);
        mp.batch_bk_buf            = alloc_s(MAX_STREAMS * KVH * B_BLK * HD * 2);
        mp.batch_bv_buf            = alloc_s(MAX_STREAMS * KVH * B_BLK * HD * 2);
        mp.batch_attn_out_buf      = alloc_s(MAX_MQ * Q_DIM * 2);
        mp.batch_mlp_mid_buf       = alloc_s(MAX_MQ * INTER * 2);
        mp.batch_sdpa_part_o       = alloc_s(MAX_STREAMS * kMaxSdpaSplits * H * B_BLK * HD * sizeof(float));
        mp.batch_sdpa_part_ml      = alloc_s(MAX_STREAMS * kMaxSdpaSplits * H * B_BLK * 2 * sizeof(float));
        mp.batch_x_out_buf         = alloc_s(MAX_MD * D * 2);
        mp.batch_base_logits_buf   = alloc_s(MAX_MD * V_UNPAD * 2);
        mp.batch_f_rnn_buf         = alloc_s(MAX_STREAMS * D * 2);
        mp.batch_rnn_ctx_buf       = alloc_s(MAX_MD * 3 * RANK * sizeof(float));
        mp.batch_rnn_state_buf[0]  = alloc_s(MAX_STREAMS * RANK * 2);
        mp.batch_rnn_state_buf[1]  = alloc_s(MAX_STREAMS * RANK * 2);
        mp.batch_rnn_latent_buf    = alloc_s(MAX_STREAMS * RANK * 2);
        mp.batch_markov_ctx_part_x = alloc_s(8 * MAX_MD * 3 * RANK * sizeof(float));
        mp.batch_markov_ctx_part_f = alloc_s(8 * MAX_STREAMS * 3 * RANK * sizeof(float));
        mp.batch_markov_part_val   = alloc_s(MAX_STREAMS * std::max(n_markov_tg, n_tg_vec) * sizeof(float));
        mp.batch_markov_part_idx   = alloc_s(MAX_STREAMS * std::max(n_markov_tg, n_tg_vec) * sizeof(uint32_t));
        mp.batch_prev_tok_buf      = alloc_s(MAX_STREAMS * sizeof(int32_t));
        mp.batch_d_toks_buf        = alloc_s(MAX_STREAMS * (B_BLK - 1) * sizeof(int32_t));
        mp.batch_verify_toks_buf   = alloc_s(MAX_MV * sizeof(int32_t));
        mp.batch_out_toks_buf      = alloc_s(MAX_STREAMS * B_BLK * sizeof(int32_t));
        mp.batch_d_reject_j_buf    = alloc_s(MAX_STREAMS * sizeof(int32_t));
        mp.batch_d_n_acc_buf       = alloc_s(MAX_STREAMS * sizeof(int32_t));
        mp.batch_vcap_buf          = alloc_s(MAX_MV * NCAP * D * 2);
        mp.batch_verify_k_buf      = alloc_s((size_t)NV * mp.batched_verify_kv_visit_stride_bytes);
        mp.batch_verify_v_buf      = alloc_s((size_t)NV * mp.batched_verify_kv_visit_stride_bytes);
        mp.batch_conv_hist_buf     = alloc_s(align_up(std::max<size_t>(ch_cur, ctx->page_size), ctx->page_size));
        mp.batch_argmax_pv_buf     = alloc_s(MAX_MV * n_tg_target_simt * sizeof(float));
        mp.batch_argmax_pi_buf     = alloc_s(MAX_MV * n_tg_target_simt * sizeof(uint32_t));

        if (!mp.batch_stream_meta_buf || !mp.batch_draft_kv_meta_buf || !mp.batch_target_vmeta_buf ||
            !mp.batch_ctx_to_stream_buf || !mp.batch_ctx_local_pos_buf || !mp.batch_d_to_stream_buf ||
            !mp.batch_d_local_step_buf || !mp.batch_tok_to_stream_buf || !mp.batch_tok_local_pos_buf ||
            !mp.batch_hidden_buf || !mp.batch_x_buf || !mp.batch_residual_buf || !mp.batch_norm_buf ||
            !mp.batch_ctx_buf || !mp.batch_kv_ctx_buf || !mp.batch_qkv_buf || !mp.batch_q_buf ||
            !mp.batch_bk_buf || !mp.batch_bv_buf || !mp.batch_attn_out_buf || !mp.batch_mlp_mid_buf ||
            !mp.batch_sdpa_part_o || !mp.batch_sdpa_part_ml || !mp.batch_x_out_buf ||
            !mp.batch_base_logits_buf || !mp.batch_f_rnn_buf || !mp.batch_rnn_ctx_buf ||
            !mp.batch_rnn_state_buf[0] || !mp.batch_rnn_state_buf[1] || !mp.batch_rnn_latent_buf ||
            !mp.batch_markov_ctx_part_x || !mp.batch_markov_ctx_part_f || !mp.batch_markov_part_val ||
            !mp.batch_markov_part_idx || !mp.batch_prev_tok_buf || !mp.batch_d_toks_buf ||
            !mp.batch_verify_toks_buf || !mp.batch_out_toks_buf || !mp.batch_d_reject_j_buf ||
            !mp.batch_d_n_acc_buf || !mp.batch_vcap_buf || !mp.batch_verify_k_buf ||
            !mp.batch_verify_v_buf || !mp.batch_conv_hist_buf || !mp.batch_argmax_pv_buf ||
            !mp.batch_argmax_pi_buf) {
            return -9;
        }
        mp.batched_max_mv = max_mv;
        mp.batched_init = true;
    }

    if (need_sampling && (!mp.batched_sampling_init || !mp.batch_q_probs_buf)) {
        const size_t MAX_STREAMS = 4;
        const size_t B_BLK = (size_t)mc.block_size;
        const size_t MAX_MD = MAX_STREAMS * (B_BLK - 1);
        const size_t MAX_MV = (size_t)mp.batched_max_mv;
        mp.batch_q_probs_buf       = alloc_s(MAX_MD * V_UNPAD * sizeof(float));
        mp.batch_p_probs_buf       = alloc_s(MAX_MV * V_UNPAD * sizeof(float));
        mp.batch_row_max_buf       = alloc_s(MAX_MV * sizeof(float));
        mp.batch_row_inv_sum_buf   = alloc_s(MAX_MV * sizeof(float));
        mp.batch_topp_states_buf   = alloc_s(MAX_MV * sizeof(MTPTopPState));
        mp.batch_softmax_part_buf  = alloc_s(MAX_MV * std::max(n_tg_target_simt, n_tg_vec * 32) * sizeof(float));
        mp.batch_topp_part_cnt_buf = alloc_s(MAX_MV * n_tg_vec * 32 * sizeof(uint32_t));
        mp.batch_reject_res_val    = alloc_s(MAX_STREAMS * n_tg_vec * sizeof(float));
        mp.batch_reject_res_idx    = alloc_s(MAX_STREAMS * n_tg_vec * sizeof(uint32_t));
        mp.batch_reject_p_val      = alloc_s(MAX_STREAMS * n_tg_vec * sizeof(float));
        mp.batch_reject_p_idx      = alloc_s(MAX_STREAMS * n_tg_vec * sizeof(uint32_t));
        mp.batch_reject_res_sum    = alloc_s(MAX_STREAMS * n_tg_vec * sizeof(float));
        if (!mp.batch_q_probs_buf || !mp.batch_p_probs_buf || !mp.batch_row_max_buf ||
            !mp.batch_row_inv_sum_buf || !mp.batch_topp_states_buf || !mp.batch_softmax_part_buf ||
            !mp.batch_topp_part_cnt_buf || !mp.batch_reject_res_val || !mp.batch_reject_res_idx ||
            !mp.batch_reject_p_val || !mp.batch_reject_p_idx || !mp.batch_reject_res_sum) {
            return -9;
        }
        mp.batched_sampling_init = true;
    }
    return 0;
}

int inkling_ctx_mtp_step_batched(
    InklingCtx*             ctx,
    InklingMTPStreamDesc*   streams,
    int32_t                 batch_size,
    InklingStepStats*       stats_out)
{
    @autoreleasepool {
        if (!ctx || !ctx->mtp.attached || !ctx->globals.registered || !streams) return -1;
        if (batch_size < 1 || batch_size > 4) {
            fprintf(stderr, "[inkling_decode] ERROR: batched MTP batch_size=%d out of [1, 4]\n", batch_size);
            return -2;
        }
        for (int i = 0; i < ctx->cfg.n_layers; ++i) {
            if (!ctx->layers[i].registered) return -4;
        }

        MTPAttachedState& mp = ctx->mtp;
        const MTPDraftConfig& mc = mp.cfg;
        const int32_t B = batch_size;
        const int32_t D = mc.hidden_size;
        const int32_t B_BLK = mc.block_size;
        const int32_t NCAP = mc.n_target_layers;
        const int32_t NV = (int32_t)ctx->schedule.size();
        const int32_t INTER_D = mc.inter_size;
        const int32_t KV_DIM = mc.num_kv_heads * mc.head_dim;
        const int32_t Q_DIM = mc.num_heads * mc.head_dim;
        const int32_t QKV_DIM = Q_DIM + 2 * KV_DIM;
        const int32_t V_UNPAD = mc.unpadded_vocab_size;

        int32_t sum_n_ctx = 0;
        int32_t m_d_tot = 0;
        int32_t m_v = 0;
        int32_t max_num_draft = 0;
        int32_t max_mv_b = 0;
        bool any_sampled = false;
        bool all_sampled = true;
        bool any_top_p = false;

        for (int b = 0; b < B; ++b) {
            const InklingMTPStreamDesc& s = streams[b];
            if (!s.in_hidden_ptr || !s.out_hidden_ptr || !s.kv_ptrs || !s.conv_ptrs ||
                !s.kv_meta || !s.draft_kv_descs) {
                return -1;
            }
            if (s.n_ctx < 1 || s.n_ctx > B_BLK || s.num_draft < 1 || s.num_draft >= B_BLK) {
                return -2;
            }
            const int32_t M_b = s.num_draft + 1;
            const size_t req_in  = (size_t)s.n_ctx * (size_t)NCAP * (size_t)D * sizeof(uint16_t);
            const size_t req_out = (size_t)M_b     * (size_t)NCAP * (size_t)D * sizeof(uint16_t);
            if (s.in_hidden_bytes < req_in || s.out_hidden_bytes < req_out) {
                return -3;
            }
            const bool is_s = (s.temperature > 0.0f);
            any_sampled = any_sampled || is_s;
            all_sampled = all_sampled && is_s;
            if (is_s && s.top_p > 0.0f && s.top_p < 1.0f) any_top_p = true;
            sum_n_ctx += s.n_ctx;
            m_d_tot += s.num_draft;
            m_v += M_b;
            max_num_draft = std::max(max_num_draft, s.num_draft);
            max_mv_b = std::max(max_mv_b, M_b);
        }
        if (m_v > ctx->cfg.max_m || m_v > 32) {
            fprintf(stderr, "[inkling_decode] ERROR: total verify tokens M_v=%d exceeds max_m=%d / 32\n",
                    m_v, ctx->cfg.max_m);
            return -2;
        }

        int rc_init = ensure_mtp_batched_state(ctx, any_sampled);
        if (rc_init != 0) return rc_init;

        std::vector<id<MTLBuffer>> kept_bufs;
        std::vector<__unsafe_unretained id<MTLResource>> used_res;
        std::unordered_set<void*> seen_bufs;
        kept_bufs.reserve(256);
        used_res.reserve(256);
        auto track_buf = [&](id<MTLBuffer> buf) {
            if (!buf) return;
            void* key = (__bridge void*)buf;
            if (seen_bufs.insert(key).second) {
                kept_bufs.push_back(buf);
                used_res.push_back(buf);
            }
        };
        track_buf(mp.batch_conv_hist_buf);
        track_buf(mp.batch_verify_k_buf);
        track_buf(mp.batch_verify_v_buf);

        struct DraftStreamPlan {
            int32_t off_after[8];
            int32_t next_ring[8];
        } d_plans[4]{};

        BatchMTPStreamMetaGpu* sm_cpu = (BatchMTPStreamMetaGpu*)[mp.batch_stream_meta_buf contents];
        BatchMTPDraftKVMetaGpu* dk_cpu = (BatchMTPDraftKVMetaGpu*)[mp.batch_draft_kv_meta_buf contents];
        int32_t* ctx_to_stream_cpu = (int32_t*)[mp.batch_ctx_to_stream_buf contents];
        int32_t* ctx_local_pos_cpu = (int32_t*)[mp.batch_ctx_local_pos_buf contents];
        int32_t* d_to_stream_cpu   = (int32_t*)[mp.batch_d_to_stream_buf contents];
        int32_t* d_local_step_cpu  = (int32_t*)[mp.batch_d_local_step_buf contents];
        int32_t* tok_to_stream_cpu = (int32_t*)[mp.batch_tok_to_stream_buf contents];
        int32_t* tok_local_pos_cpu = (int32_t*)[mp.batch_tok_local_pos_buf contents];
        uint8_t* hidden_dst = (uint8_t*)[mp.batch_hidden_buf contents];

        int32_t ctx_cur = 0, d_cur = 0, v_cur = 0;
        int32_t max_d_valid_ctx[8]{};
        for (int b = 0; b < B; ++b) {
            const InklingMTPStreamDesc& s = streams[b];
            const int32_t M_b = s.num_draft + 1;
            const size_t req_in = (size_t)s.n_ctx * (size_t)NCAP * (size_t)D * sizeof(uint16_t);
            std::memcpy(hidden_dst + (size_t)ctx_cur * (size_t)NCAP * (size_t)D * sizeof(uint16_t),
                        s.in_hidden_ptr, req_in);

            BufSlice out_h = wrap_ptr_nocache(ctx, (uintptr_t)s.out_hidden_ptr, s.out_hidden_bytes);
            if (!out_h.buf) return -5;
            track_buf(out_h.buf);

            const bool is_s = (s.temperature > 0.0f);
            const bool do_tp = is_s && (s.top_p > 0.0f && s.top_p < 1.0f);
            BatchMTPStreamMetaGpu& sm = sm_cpu[b];
            sm.anchor_token    = s.anchor_token;
            sm.n_ctx           = s.n_ctx;
            sm.ctx_start       = ctx_cur;
            sm.num_draft       = s.num_draft;
            sm.d_start         = d_cur;
            sm.v_start         = v_cur;
            sm.is_sampled      = is_s ? 1 : 0;
            sm.do_top_p        = do_tp ? 1 : 0;
            sm.temperature     = s.temperature;
            sm.top_p           = s.top_p;
            sm.softcap         = s.logit_softcap;
            sm._pad0           = 0.0f;
            sm.rng_seed        = s.rng_seed;
            sm.rng_step        = s.rng_step;
            sm.out_hidden_addr = (uint64_t)[out_h.buf gpuAddress] + (uint64_t)out_h.offset;

            for (int t = 0; t < s.n_ctx; ++t) {
                ctx_to_stream_cpu[ctx_cur + t] = b;
                ctx_local_pos_cpu[ctx_cur + t] = t;
            }
            for (int j = 0; j < s.num_draft; ++j) {
                d_to_stream_cpu[d_cur + j]  = b;
                d_local_step_cpu[d_cur + j] = j;
            }
            for (int r = 0; r < M_b; ++r) {
                tok_to_stream_cpu[v_cur + r] = b;
                tok_local_pos_cpu[v_cur + r] = r;
            }

            for (int i = 0; i < mc.num_layers; ++i) {
                const MTPKVCacheDesc& kd = s.draft_kv_descs[i];
                const MTPLayerMetalWeights& lw = mp.layers[(size_t)i];
                const int32_t off_before = kd.offset;
                const int32_t off_after  = off_before + s.n_ctx;
                int32_t win = 0, ring_start = 0, valid_ctx = 0, next_ring = 0;
                if (kd.is_rotating != 0) {
                    const int32_t W = (kd.window > 0) ? kd.window : lw.window;
                    win = W;
                    if (W <= 0 || s.n_ctx > W) return -20 - i;
                    const int32_t cur_idx = (kd.ring_idx >= 0) ? (kd.ring_idx % W) : (off_before % W);
                    if (off_before + s.n_ctx <= W) {
                        if (kd.cache_cap < off_after) return -20 - i;
                        ring_start = (kd.ring_idx >= 0) ? (kd.ring_idx % W) : off_before;
                        valid_ctx  = off_after;
                    } else {
                        if (kd.cache_cap < W) return -20 - i;
                        ring_start = cur_idx;
                        valid_ctx  = W;
                    }
                    const int32_t end_mod = (ring_start + s.n_ctx) % W;
                    next_ring = (end_mod == 0 && off_after >= W) ? W : end_mod;
                } else {
                    if (off_after > kd.cache_cap) return -30 - i;
                    ring_start = off_before;
                    valid_ctx  = off_after;
                    next_ring  = off_after;
                }
                const int32_t k_ss = (kd.stride_s > 0) ? kd.stride_s : mc.head_dim;
                const int32_t k_sh = (kd.stride_h > 0) ? kd.stride_h : (kd.cache_cap * k_ss);
                const size_t min_bytes = (size_t)mc.num_kv_heads * (size_t)k_sh * sizeof(uint16_t);
                if ((kd.k_nbytes > 0 && kd.k_nbytes < min_bytes) ||
                    (kd.v_nbytes > 0 && kd.v_nbytes < min_bytes)) {
                    return -35 - i;
                }
                BufSlice dk_s = wrap_ptr_nocache(ctx, kd.k_ptr, std::max(kd.k_nbytes, min_bytes));
                BufSlice dv_s = wrap_ptr_nocache(ctx, kd.v_ptr, std::max(kd.v_nbytes, min_bytes));
                if (!dk_s.buf || !dv_s.buf) return -40 - i;
                track_buf(dk_s.buf);
                track_buf(dv_s.buf);

                BatchMTPDraftKVMetaGpu& dm = dk_cpu[(size_t)b * (size_t)mc.num_layers + (size_t)i];
                dm.k_addr        = (uint64_t)[dk_s.buf gpuAddress] + (uint64_t)dk_s.offset;
                dm.v_addr        = (uint64_t)[dv_s.buf gpuAddress] + (uint64_t)dv_s.offset;
                dm.offset_before = off_before;
                dm.is_rotating   = kd.is_rotating;
                dm.window        = win;
                dm.ring_start    = ring_start;
                dm.valid_ctx     = valid_ctx;
                dm.k_stride_h    = k_sh;
                dm.k_stride_s    = k_ss;
                dm.v_stride_h    = k_sh;
                dm.v_stride_s    = k_ss;
                dm.offset_after  = off_after;

                d_plans[b].off_after[i] = off_after;
                d_plans[b].next_ring[i] = next_ring;
                max_d_valid_ctx[i] = std::max(max_d_valid_ctx[i], valid_ctx);
            }

            ctx_cur += s.n_ctx;
            d_cur   += s.num_draft;
            v_cur   += M_b;
        }

        BatchMTPTargetVisitMetaGpu* vm_cpu = (BatchMTPTargetVisitMetaGpu*)[mp.batch_target_vmeta_buf contents];
        const uint64_t conv_hist_base = (uint64_t)[mp.batch_conv_hist_buf gpuAddress];
        const uint64_t verify_k_base  = (uint64_t)[mp.batch_verify_k_buf gpuAddress];
        const uint64_t verify_v_base  = (uint64_t)[mp.batch_verify_v_buf gpuAddress];

        struct TargetVisitPlan {
            int32_t max_valid = 0;
            bool use_splitk = false;
            int32_t bs = 0;
            int32_t nblocks = 0;
        };
        std::vector<TargetVisitPlan> vplans((size_t)NV);
        size_t max_part_meta = 0, max_part_acc = 0;
        bool any_need_staging = false;
        int32_t max_commit_elems = 0;
        uint32_t max_conv_elems = 0;
        size_t ch_cur = 0;

        for (int step = 0; step < NV; ++step) {
            const LayerDeviceWeights& L = ctx->layers[ctx->schedule[step].layer_idx];
            const size_t ch = (size_t)L.n_kv * (size_t)L.head_dim;
            const size_t visit_c_bytes = (size_t)m_v * 3 * ch * sizeof(float);
            const size_t visit_d_bytes = (size_t)m_v * 3 * (size_t)ctx->cfg.hidden_size * sizeof(float);
            const uint64_t hks = conv_hist_base + (uint64_t)ch_cur;
            const uint64_t hvs = hks + (uint64_t)visit_c_bytes;
            const uint64_t hcs = hvs + (uint64_t)visit_c_bytes;
            const uint64_t hms = hcs + (uint64_t)visit_d_bytes;
            ch_cur = align_up(ch_cur + 2 * visit_c_bytes + 2 * visit_d_bytes, 256);

            const uint64_t sk = verify_k_base + (uint64_t)step * (uint64_t)mp.batched_verify_kv_visit_stride_bytes;
            const uint64_t sv = verify_v_base + (uint64_t)step * (uint64_t)mp.batched_verify_kv_visit_stride_bytes;

            const uint32_t nc = (uint32_t)(3 * ch);
            const uint32_t nd = (uint32_t)(3 * (size_t)ctx->cfg.hidden_size);
            max_conv_elems = std::max(max_conv_elems, std::max(nc, nd));

            int32_t max_valid_v = 0;
            for (int b = 0; b < B; ++b) {
                const InklingMTPStreamDesc& s = streams[b];
                const int32_t M_b = s.num_draft + 1;
                const InklingKVMeta& km = s.kv_meta[step];

                const size_t kv_bytes = (size_t)L.n_kv * (size_t)km.stride_h * sizeof(uint16_t);
                BufSlice k_s = wrap_cached_ptr(ctx, s.kv_ptrs[step * 2 + 0], kv_bytes);
                BufSlice v_s = wrap_cached_ptr(ctx, s.kv_ptrs[step * 2 + 1], kv_bytes);
                if (!k_s.buf || !v_s.buf) return -6;
                track_buf(k_s.buf);
                track_buf(v_s.buf);

                const size_t c_bytes[4] = {
                    3 * ch * sizeof(float), 3 * ch * sizeof(float),
                    3 * (size_t)ctx->cfg.hidden_size * sizeof(float),
                    3 * (size_t)ctx->cfg.hidden_size * sizeof(float),
                };
                uint64_t conv_addrs[4]{};
                for (int c = 0; c < 4; ++c) {
                    BufSlice cs = wrap_cached_ptr(ctx, s.conv_ptrs[step * 4 + c], c_bytes[c]);
                    if (!cs.buf) return -7;
                    track_buf(cs.buf);
                    conv_addrs[c] = (uint64_t)[cs.buf gpuAddress] + (uint64_t)cs.offset;
                }

                const int32_t prev = km.offset;
                const int32_t old_idx = km.idx;
                const bool is_rot = (km.is_rotating != 0);
                const bool need_staging = (is_rot && M_b > 1 && (prev + M_b > km.max_size));
                int32_t write_base = 0, s_kv = 0, ext_m = 0, ring_idx = 0;
                if (!is_rot || (!need_staging && prev + M_b <= km.max_size)) {
                    write_base = prev;
                    s_kv = prev + M_b;
                    ext_m = 0;
                    ring_idx = 0;
                } else if (!need_staging) {
                    write_base = old_idx % km.max_size;
                    s_kv = km.max_size;
                    ext_m = 0;
                    ring_idx = (write_base + 1) % km.max_size;
                } else {
                    s_kv = std::min(prev, km.max_size);
                    ext_m = M_b;
                    ring_idx = (prev <= km.max_size) ? 0 : (old_idx % km.max_size);
                }

                const int32_t s_total_b = s_kv + ext_m;
                const int32_t max_valid_b = (L.sliding_window > 0) ? std::min(s_total_b, L.sliding_window) : s_total_b;
                max_valid_v = std::max(max_valid_v, max_valid_b);

                if (need_staging) {
                    any_need_staging = true;
                    max_commit_elems = std::max(max_commit_elems, M_b * L.n_kv * L.head_dim);
                }

                BatchMTPTargetVisitMetaGpu& vm = vm_cpu[(size_t)step * (size_t)B + (size_t)b];
                vm.kv_k_addr    = (uint64_t)[k_s.buf gpuAddress] + (uint64_t)k_s.offset;
                vm.kv_v_addr    = (uint64_t)[v_s.buf gpuAddress] + (uint64_t)v_s.offset;
                for (int c = 0; c < 4; ++c) vm.conv_addr[c] = conv_addrs[c];
                vm.hist_ks_addr = hks;
                vm.hist_vs_addr = hvs;
                vm.hist_cs_addr = hcs;
                vm.hist_ms_addr = hms;
                vm.stage_k_addr = sk;
                vm.stage_v_addr = sv;
                vm.prev_offset  = prev;
                vm.old_idx      = old_idx;
                vm.max_size     = km.max_size;
                vm.is_rotating  = is_rot ? 1 : 0;
                vm.need_staging = need_staging ? 1 : 0;
                vm.write_base   = write_base;
                vm.s_kv         = s_kv;
                vm.ext_m        = ext_m;
                vm.ring_idx     = ring_idx;
                vm.stride_h     = km.stride_h;
                vm.stride_s     = km.stride_s;
                vm.v_start      = sm_cpu[b].v_start;
                vm.M_b          = M_b;
                vm.n_kv         = L.n_kv;
                vm.head_dim     = L.head_dim;
                vm._pad         = 0;
            }

            TargetVisitPlan& vp = vplans[(size_t)step];
            vp.max_valid = max_valid_v;
            const int win_tg = window_tg(L.head_dim);
            vp.use_splitk = (max_valid_v > win_tg);
            if (vp.use_splitk) {
                int bs = (max_valid_v < 16384) ? 256 : 512;
                int nblocks = std::max(1, (max_valid_v + bs - 1) / bs);
                while (nblocks > ctx->max_nblocks) {
                    bs *= 2;
                    nblocks = (max_valid_v + bs - 1) / bs;
                }
                vp.bs = bs;
                vp.nblocks = nblocks;
                max_part_meta = std::max(max_part_meta, (size_t)nblocks * (size_t)L.n_heads);
                max_part_acc  = std::max(max_part_acc,  (size_t)nblocks * (size_t)L.n_heads * (size_t)L.head_dim);
            }
        }

        if (max_part_meta > 0) {
            const size_t sz_meta = align_up((size_t)m_v * max_part_meta * sizeof(float), 256);
            const size_t sz_acc  = align_up((size_t)m_v * max_part_acc  * sizeof(float), 256);
            if (ensure_part_buf(ctx, sz_meta, sz_meta, sz_acc) != 0) return -9;
        }

        const bool no_nax_env = (getenv("INKLING_MTP_NO_NAX") != nullptr && atoi(getenv("INKLING_MTP_NO_NAX")) != 0);
        const bool has_nax = (ctx->lib_nax != nil) && !no_nax_env && ((ctx->cfg.hidden_size % 16) == 0);
        const bool use_nax_v_qkv  = has_nax && (m_v >= 9);
        const bool use_nax_v_head = has_nax && (m_v >= 9);
        const bool use_nax_d_ctx  = has_nax && (sum_n_ctx >= 9);
        const bool use_nax_d_blk  = has_nax && ((B * B_BLK) >= 9);
        const bool use_nax_d_head = has_nax && (m_d_tot >= 9);

        const int grp_mt = (m_v <= 8) ? 4 : 8;
        const int sg_grp = (m_v <= 8) ? 4 : 2;
        id<MTLComputePipelineState> pso_grp_w13_nv = nil, pso_grp_w2_nv = nil;
        id<MTLComputePipelineState> pso_grp_w13_mx = nil, pso_grp_w2_mx = nil;
        if (m_v >= 3) {
            pso_grp_w13_nv = get_grp_moe_tiles_pso(ctx, true,  false, sg_grp, grp_mt);
            pso_grp_w2_nv  = get_grp_moe_tiles_pso(ctx, true,  true,  sg_grp, grp_mt);
            pso_grp_w13_mx = get_grp_moe_tiles_pso(ctx, false, false, sg_grp, grp_mt);
            pso_grp_w2_mx  = get_grp_moe_tiles_pso(ctx, false, true,  sg_grp, grp_mt);
        }

        const BufSlice& target_lm_head = (ctx->globals.lm_head.buf != ctx->zero_buf)
                                       ? ctx->globals.lm_head : mp.lm_head;
        id<MTLComputePipelineState> pso_nax_v = (m_v <= 16) ? ctx->pso_nax_qkv_16x64 : ctx->pso_nax_qkv_32x64;
        id<MTLComputePipelineState> pso_argmax_simt_m = (!any_sampled && !use_nax_v_head && m_v <= 8)
                                                      ? get_argmax_p1_pso(ctx, m_v, (m_v == 1) ? 4 : 2) : nil;

        const uint64_t t_wall0 = mach_absolute_time();
        const uint64_t t_enc0  = t_wall0;

        id<MTLCommandBuffer> cb = [ctx->queue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
        if (!used_res.empty()) {
            [enc useResources:used_res.data()
                        count:(NSUInteger)used_res.size()
                        usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
        }
        int32_t n_dispatches = 0;

        const float   m_eps = mc.rms_norm_eps;
        const float   neg_log2_theta_over_half = -std::log2(mc.rope_theta) / (float)(mc.head_dim / 2);
        const float   sdpa_scale2 = (1.0f / std::sqrt((float)mc.head_dim)) * 1.4426950408889634f;
        const NSUInteger tg_rope = (NSUInteger)std::max(32, mc.head_dim / 2);
        const NSUInteger tg_sdpa = (NSUInteger)(B_BLK * 32);

        const int32_t n_markov_tg      = (V_UNPAD + 63) / 64;
        const int32_t n_tg_target_simt = (V_UNPAD + 15) / 16;
        const int32_t n_tg_target_nax  = (V_UNPAD + 63) / 64;
        const int32_t n_tg_vec         = std::min<int32_t>(64, std::max<int32_t>(1, (V_UNPAD + 255) / 256));

        auto encode_draft_bf16_gemm = [&](
            id<MTLBuffer> x_b, id<MTLBuffer> w_b, size_t w_o, id<MTLBuffer> out_b,
            int M_rows, int K_dim, int N_dim, bool allow_nax, id<MTLComputePipelineState> exact_pso)
        {
            if (allow_nax && M_rows >= 9 && (K_dim % 16) == 0) {
                id<MTLComputePipelineState> pso_nax = (M_rows <= 16) ? ctx->pso_nax_qkv_16x64 : ctx->pso_nax_qkv_32x64;
                const int nt = (N_dim + 63) / 64;
                NaxQkvPrm prm{M_rows, 1, K_dim, N_dim, 0, 0, nt, 0, 0};
                [enc setComputePipelineState:pso_nax];
                [enc setBuffer:x_b   offset:0   atIndex:0];
                [enc setBuffer:w_b   offset:w_o atIndex:1];
                [enc setBuffer:out_b offset:0   atIndex:2];
                [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nt, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(4 * 32, 1, 1)];
                ++n_dispatches;
            } else if (exact_pso != nil) {
                MTPGemvPrm prm{N_dim};
                [enc setComputePipelineState:exact_pso];
                [enc setBuffer:x_b   offset:0   atIndex:0];
                [enc setBuffer:w_b   offset:w_o atIndex:1];
                [enc setBuffer:out_b offset:0   atIndex:2];
                [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((N_dim + 3) / 4), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                ++n_dispatches;
            } else {
                BatchGemvPrm prm{M_rows, K_dim, N_dim};
                [enc setComputePipelineState:mp.pso_batch_gemv];
                [enc setBuffer:x_b   offset:0   atIndex:0];
                [enc setBuffer:w_b   offset:w_o atIndex:1];
                [enc setBuffer:out_b offset:0   atIndex:2];
                [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((N_dim + 3) / 4), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                ++n_dispatches;
            }
        };

        const int32_t M_q = B * B_BLK;
        {
            BatchInitXPrm iprm{B, D, B_BLK, mc.markov_rank, mc.embed_norm_eps};
            [enc setComputePipelineState:mp.pso_batch_init_x[mp.has_embed_norm ? 1 : 0]];
            [enc setBuffer:mp.embed_tokens.buf       offset:mp.embed_tokens.offset atIndex:0];
            [enc setBuffer:mp.embed_norm.buf         offset:mp.embed_norm.offset   atIndex:1];
            [enc setBuffer:mp.mask_embed.buf         offset:mp.mask_embed.offset   atIndex:2];
            [enc setBuffer:mp.batch_stream_meta_buf  offset:0                       atIndex:3];
            [enc setBuffer:mp.batch_x_buf            offset:0                       atIndex:4];
            [enc setBuffer:mp.batch_rnn_state_buf[0] offset:0                       atIndex:5];
            [enc setBuffer:mp.batch_prev_tok_buf     offset:0                       atIndex:6];
            [enc setBuffer:mp.batch_verify_toks_buf  offset:0                       atIndex:7];
            [enc setBytes:&iprm length:sizeof(iprm) atIndex:8];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)B, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            ++n_dispatches;
        }

        for (int i = 0; i < mc.num_layers; ++i) {
            const MTPLayerMetalWeights& lw = mp.layers[(size_t)i];
            {
                MTPMixPrm prm{D, NCAP, 0, i, m_eps};
                [enc setComputePipelineState:mp.pso_dflare_mix];
                [enc setBuffer:mp.batch_hidden_buf offset:0                    atIndex:0];
                [enc setBuffer:mp.fuse_alpha.buf   offset:mp.fuse_alpha.offset atIndex:1];
                [enc setBuffer:lw.dflare_norm.buf  offset:lw.dflare_norm.offset atIndex:2];
                [enc setBuffer:mp.batch_ctx_buf    offset:0                    atIndex:3];
                [enc setBytes:&prm length:sizeof(prm) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)sum_n_ctx, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_dispatches;
            }
            encode_draft_bf16_gemm(
                mp.batch_ctx_buf, lw.kv_ctx_w.buf, lw.kv_ctx_w.offset, mp.batch_kv_ctx_buf,
                sum_n_ctx, D, 2 * KV_DIM, use_nax_d_ctx,
                (sum_n_ctx <= B_BLK) ? mp.pso_gemv_ctx_kv[sum_n_ctx] : nil);
            {
                BatchKvCtxWritePrm prm{KV_DIM, mc.num_layers, i, 0, m_eps, neg_log2_theta_over_half};
                [enc setComputePipelineState:mp.pso_batch_kv_ctx_write];
                [enc setBuffer:mp.batch_kv_ctx_buf       offset:0                  atIndex:0];
                [enc setBuffer:lw.k_norm_ctx.buf         offset:lw.k_norm_ctx.offset atIndex:1];
                [enc setBuffer:mp.batch_ctx_to_stream_buf offset:0                 atIndex:2];
                [enc setBuffer:mp.batch_ctx_local_pos_buf offset:0                 atIndex:3];
                [enc setBuffer:mp.batch_draft_kv_meta_buf offset:0                 atIndex:4];
                [enc setBytes:&prm length:sizeof(prm) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)sum_n_ctx, (NSUInteger)mc.num_kv_heads, 1)
                    threadsPerThreadgroup:MTLSizeMake(tg_rope, 1, 1)];
                ++n_dispatches;
            }
            {
                MTPAddNormPrm prm{D, 0, 1, m_eps};
                [enc setComputePipelineState:mp.pso_add_rmsnorm[i == 0 ? 0 : 1]];
                [enc setBuffer:mp.batch_x_buf        offset:0                         atIndex:0];
                [enc setBuffer:mp.batch_residual_buf offset:0                         atIndex:1];
                [enc setBuffer:lw.input_layernorm.buf offset:lw.input_layernorm.offset atIndex:2];
                [enc setBuffer:mp.batch_residual_buf offset:0                         atIndex:3];
                [enc setBuffer:mp.batch_norm_buf     offset:0                         atIndex:4];
                [enc setBytes:&prm length:sizeof(prm) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)M_q, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_dispatches;
            }
            encode_draft_bf16_gemm(
                mp.batch_norm_buf, lw.qkv_w.buf, lw.qkv_w.offset, mp.batch_qkv_buf,
                M_q, D, QKV_DIM, use_nax_d_blk, (B == 1) ? mp.pso_gemv_qkv : nil);
            {
                BatchQkvPackPrm prm{Q_DIM, KV_DIM, QKV_DIM, mc.num_layers, i, 0, m_eps, neg_log2_theta_over_half};
                [enc setComputePipelineState:mp.pso_batch_qkv_pack];
                [enc setBuffer:mp.batch_qkv_buf           offset:0             atIndex:0];
                [enc setBuffer:lw.q_norm.buf              offset:lw.q_norm.offset atIndex:1];
                [enc setBuffer:lw.k_norm.buf              offset:lw.k_norm.offset atIndex:2];
                [enc setBuffer:mp.batch_draft_kv_meta_buf offset:0             atIndex:3];
                [enc setBuffer:mp.batch_q_buf             offset:0             atIndex:4];
                [enc setBuffer:mp.batch_bk_buf            offset:0             atIndex:5];
                [enc setBuffer:mp.batch_bv_buf            offset:0             atIndex:6];
                [enc setBytes:&prm length:sizeof(prm) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)M_q, (NSUInteger)(mc.num_heads + 2 * mc.num_kv_heads), 1)
                    threadsPerThreadgroup:MTLSizeMake(tg_rope, 1, 1)];
                ++n_dispatches;
            }
            {
                const int32_t max_total_k = max_d_valid_ctx[i] + B_BLK;
                if (max_total_k <= 256) {
                    BatchSdpaPrm prm{mc.num_layers, i, 1, 0, sdpa_scale2};
                    [enc setComputePipelineState:mp.pso_batch_sdpa_single];
                    [enc setBuffer:mp.batch_q_buf             offset:0 atIndex:0];
                    [enc setBuffer:mp.batch_bk_buf            offset:0 atIndex:1];
                    [enc setBuffer:mp.batch_bv_buf            offset:0 atIndex:2];
                    [enc setBuffer:mp.batch_draft_kv_meta_buf offset:0 atIndex:3];
                    [enc setBuffer:mp.batch_attn_out_buf      offset:0 atIndex:4];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)mc.num_kv_heads, (NSUInteger)B, 1)
                        threadsPerThreadgroup:MTLSizeMake(tg_sdpa, 1, 1)];
                    ++n_dispatches;
                } else {
                    int32_t n_splits = std::min<int32_t>(32, std::max<int32_t>(4, (max_total_k + 255) / 256));
                    int32_t chunk = (max_total_k + n_splits - 1) / n_splits;
                    chunk = std::max<int32_t>(8, ((chunk + 7) / 8) * 8);
                    n_splits = (max_total_k + chunk - 1) / chunk;
                    BatchSdpaPrm prm{mc.num_layers, i, n_splits, 0, sdpa_scale2};
                    [enc setComputePipelineState:mp.pso_batch_sdpa_split_p1];
                    [enc setBuffer:mp.batch_q_buf             offset:0 atIndex:0];
                    [enc setBuffer:mp.batch_bk_buf            offset:0 atIndex:1];
                    [enc setBuffer:mp.batch_bv_buf            offset:0 atIndex:2];
                    [enc setBuffer:mp.batch_draft_kv_meta_buf offset:0 atIndex:3];
                    [enc setBuffer:mp.batch_sdpa_part_o       offset:0 atIndex:4];
                    [enc setBuffer:mp.batch_sdpa_part_ml      offset:0 atIndex:5];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:6];
                    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)mc.num_kv_heads, (NSUInteger)n_splits, (NSUInteger)B)
                        threadsPerThreadgroup:MTLSizeMake(tg_sdpa, 1, 1)];
                    ++n_dispatches;

                    [enc setComputePipelineState:mp.pso_batch_sdpa_split_p2];
                    [enc setBuffer:mp.batch_sdpa_part_o  offset:0 atIndex:0];
                    [enc setBuffer:mp.batch_sdpa_part_ml offset:0 atIndex:1];
                    [enc setBuffer:mp.batch_attn_out_buf offset:0 atIndex:2];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)B_BLK, (NSUInteger)mc.num_heads, (NSUInteger)B)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    ++n_dispatches;
                }
            }
            encode_draft_bf16_gemm(
                mp.batch_attn_out_buf, lw.o_proj_w.buf, lw.o_proj_w.offset, mp.batch_x_buf,
                M_q, Q_DIM, D, use_nax_d_blk, (B == 1) ? mp.pso_gemv_o_proj : nil);
            {
                MTPAddNormPrm prm{D, 0, 1, m_eps};
                [enc setComputePipelineState:mp.pso_add_rmsnorm[1]];
                [enc setBuffer:mp.batch_x_buf                 offset:0                                  atIndex:0];
                [enc setBuffer:mp.batch_residual_buf          offset:0                                  atIndex:1];
                [enc setBuffer:lw.post_attention_layernorm.buf offset:lw.post_attention_layernorm.offset atIndex:2];
                [enc setBuffer:mp.batch_residual_buf          offset:0                                  atIndex:3];
                [enc setBuffer:mp.batch_norm_buf              offset:0                                  atIndex:4];
                [enc setBytes:&prm length:sizeof(prm) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)M_q, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_dispatches;
            }
            if (use_nax_d_blk && M_q >= 9 && (D % 16) == 0) {
                id<MTLComputePipelineState> pso_gu_nax = (M_q <= 16) ? ctx->pso_mtp_nax_gu_silu_16x64 : ctx->pso_mtp_nax_gu_silu_32x64;
                NaxGuPrm prm{M_q, D, INTER_D};
                [enc setComputePipelineState:pso_gu_nax];
                [enc setBuffer:mp.batch_norm_buf    offset:0            atIndex:0];
                [enc setBuffer:lw.gu_w.buf          offset:lw.gu_w.offset atIndex:1];
                [enc setBuffer:mp.batch_mlp_mid_buf offset:0            atIndex:2];
                [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((INTER_D + 63) / 64), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(4 * 32, 1, 1)];
                ++n_dispatches;
            } else if (B == 1) {
                MTPGemvPrm prm{INTER_D};
                [enc setComputePipelineState:mp.pso_gu_silu];
                [enc setBuffer:mp.batch_norm_buf    offset:0            atIndex:0];
                [enc setBuffer:lw.gu_w.buf          offset:lw.gu_w.offset atIndex:1];
                [enc setBuffer:mp.batch_mlp_mid_buf offset:0            atIndex:2];
                [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((INTER_D + 3) / 4), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                ++n_dispatches;
            } else {
                BatchGemvPrm prm{M_q, D, INTER_D};
                [enc setComputePipelineState:mp.pso_batch_gu_silu];
                [enc setBuffer:mp.batch_norm_buf    offset:0            atIndex:0];
                [enc setBuffer:lw.gu_w.buf          offset:lw.gu_w.offset atIndex:1];
                [enc setBuffer:mp.batch_mlp_mid_buf offset:0            atIndex:2];
                [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((INTER_D + 3) / 4), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                ++n_dispatches;
            }
            encode_draft_bf16_gemm(
                mp.batch_mlp_mid_buf, lw.down_proj_w.buf, lw.down_proj_w.offset, mp.batch_x_buf,
                M_q, INTER_D, D, use_nax_d_blk, (B == 1) ? mp.pso_gemv_down : nil);
        }

        {
            MTPAddNormPrm prm{D, 0, 0, m_eps};
            [enc setComputePipelineState:mp.pso_batch_final_norm];
            [enc setBuffer:mp.batch_x_buf           offset:0                    atIndex:0];
            [enc setBuffer:mp.batch_residual_buf    offset:0                    atIndex:1];
            [enc setBuffer:mp.final_norm.buf        offset:mp.final_norm.offset atIndex:2];
            [enc setBuffer:mp.batch_d_to_stream_buf offset:0                    atIndex:3];
            [enc setBuffer:mp.batch_d_local_step_buf offset:0                   atIndex:4];
            [enc setBuffer:mp.batch_x_out_buf       offset:0                    atIndex:5];
            [enc setBytes:&prm length:sizeof(prm) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)m_d_tot, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            ++n_dispatches;
        }

        encode_draft_bf16_gemm(
            mp.batch_x_out_buf, mp.lm_head.buf, mp.lm_head.offset, mp.batch_base_logits_buf,
            m_d_tot, D, V_UNPAD, use_nax_d_head, nil);

        {
            MTPMixPrm prm{D, NCAP, 0, mc.num_layers, m_eps};
            [enc setComputePipelineState:mp.pso_batch_f_rnn];
            [enc setBuffer:mp.batch_hidden_buf       offset:0                            atIndex:0];
            [enc setBuffer:mp.fuse_alpha.buf         offset:mp.fuse_alpha.offset         atIndex:1];
            [enc setBuffer:mp.markov_dflare_norm.buf offset:mp.markov_dflare_norm.offset atIndex:2];
            [enc setBuffer:mp.batch_stream_meta_buf  offset:0                            atIndex:3];
            [enc setBuffer:mp.batch_f_rnn_buf        offset:0                            atIndex:4];
            [enc setBytes:&prm length:sizeof(prm) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)B, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            ++n_dispatches;
        }
        {
            const int32_t out_dim = 3 * mc.markov_rank;
            BatchMarkovCtxPrm prm{B, m_d_tot, D, out_dim};
            [enc setComputePipelineState:mp.pso_batch_markov_ctx_p1];
            [enc setBuffer:mp.batch_x_out_buf        offset:0                     atIndex:0];
            [enc setBuffer:mp.batch_f_rnn_buf        offset:0                     atIndex:1];
            [enc setBuffer:mp.rnn_w_ctx_t.buf        offset:mp.rnn_w_ctx_t.offset atIndex:2];
            [enc setBuffer:mp.batch_markov_ctx_part_x offset:0                    atIndex:3];
            [enc setBuffer:mp.batch_markov_ctx_part_f offset:0                    atIndex:4];
            [enc setBytes:&prm length:sizeof(prm) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((out_dim + 31) / 32), 16, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            ++n_dispatches;

            [enc setComputePipelineState:mp.pso_batch_markov_ctx_p2];
            [enc setBuffer:mp.batch_markov_ctx_part_x offset:0                    atIndex:0];
            [enc setBuffer:mp.batch_markov_ctx_part_f offset:0                    atIndex:1];
            [enc setBuffer:mp.rnn_b_f32.buf           offset:mp.rnn_b_f32.offset  atIndex:2];
            [enc setBuffer:mp.batch_d_to_stream_buf   offset:0                    atIndex:3];
            [enc setBuffer:mp.batch_rnn_ctx_buf       offset:0                    atIndex:4];
            [enc setBytes:&prm length:sizeof(prm) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((out_dim + 31) / 32), (NSUInteger)m_d_tot, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            ++n_dispatches;
        }

        const int32_t max_k = B_BLK - 1;
        for (int s = 0; s < max_num_draft; ++s) {
            {
                BatchRnnStepPrm prm{B, s, mc.markov_rank, 0};
                [enc setComputePipelineState:mp.pso_batch_rnn_step];
                [enc setBuffer:mp.batch_rnn_state_buf[s & 1]       offset:0                      atIndex:0];
                [enc setBuffer:mp.batch_prev_tok_buf               offset:0                      atIndex:1];
                [enc setBuffer:mp.markov_w1.buf                    offset:mp.markov_w1.offset    atIndex:2];
                [enc setBuffer:mp.rnn_w_step_t.buf                 offset:mp.rnn_w_step_t.offset atIndex:3];
                [enc setBuffer:mp.batch_rnn_ctx_buf                offset:0                      atIndex:4];
                [enc setBuffer:mp.batch_stream_meta_buf            offset:0                      atIndex:5];
                [enc setBuffer:mp.batch_rnn_state_buf[(s & 1) ^ 1] offset:0                      atIndex:6];
                [enc setBuffer:mp.batch_rnn_latent_buf             offset:0                      atIndex:7];
                [enc setBytes:&prm length:sizeof(prm) atIndex:8];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(mc.markov_rank / 32), (NSUInteger)B, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                ++n_dispatches;
            }
            {
                BatchMarkovPrm prm{B, V_UNPAD, s, n_markov_tg, max_k, 0};
                [enc setComputePipelineState:mp.pso_batch_markov_mixed_p1];
                [enc setBuffer:mp.batch_rnn_latent_buf                                    offset:0                      atIndex:0];
                [enc setBuffer:mp.fold_weight.buf                                         offset:mp.fold_weight.offset atIndex:1];
                [enc setBuffer:mp.fold_bias.buf                                           offset:mp.fold_bias.offset   atIndex:2];
                [enc setBuffer:mp.batch_base_logits_buf                                   offset:0                      atIndex:3];
                [enc setBuffer:mp.batch_stream_meta_buf                                   offset:0                      atIndex:4];
                [enc setBuffer:mp.batch_markov_part_val                                   offset:0                      atIndex:5];
                [enc setBuffer:mp.batch_markov_part_idx                                   offset:0                      atIndex:6];
                [enc setBuffer:(any_sampled ? mp.batch_q_probs_buf : ctx->zero_buf)       offset:0                      atIndex:7];
                [enc setBuffer:(any_sampled ? mp.batch_softmax_part_buf : ctx->zero_buf)  offset:0                      atIndex:8];
                [enc setBytes:&prm length:sizeof(prm) atIndex:9];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_markov_tg, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_dispatches;

                if (!all_sampled) {
                    [enc setComputePipelineState:mp.pso_batch_markov_argmax_p2];
                    [enc setBuffer:mp.batch_markov_part_val offset:0 atIndex:0];
                    [enc setBuffer:mp.batch_markov_part_idx offset:0 atIndex:1];
                    [enc setBuffer:mp.batch_stream_meta_buf offset:0 atIndex:2];
                    [enc setBuffer:mp.batch_d_toks_buf      offset:0 atIndex:3];
                    [enc setBuffer:mp.batch_prev_tok_buf    offset:0 atIndex:4];
                    [enc setBuffer:mp.batch_verify_toks_buf offset:0 atIndex:5];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:6];
                    [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)B, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    ++n_dispatches;
                }
            }
            if (any_sampled) {
                {
                    BatchSoftmaxPrm prm{V_UNPAD, n_markov_tg, n_tg_vec, s, 0, 1, max_k, 0};
                    [enc setComputePipelineState:mp.pso_batch_softmax_max_p2];
                    [enc setBuffer:mp.batch_softmax_part_buf  offset:0 atIndex:0];
                    [enc setBuffer:mp.batch_row_max_buf       offset:0 atIndex:1];
                    [enc setBuffer:mp.batch_tok_to_stream_buf offset:0 atIndex:2];
                    [enc setBuffer:mp.batch_stream_meta_buf   offset:0 atIndex:3];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)B, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    ++n_dispatches;
                }
                {
                    BatchSoftmaxPrm prm{V_UNPAD, n_tg_vec, n_tg_vec, s, 0, 1, max_k, 0};
                    [enc setComputePipelineState:mp.pso_batch_softmax_exp_sum_p1];
                    [enc setBuffer:mp.batch_q_probs_buf       offset:0 atIndex:0];
                    [enc setBuffer:mp.batch_row_max_buf       offset:0 atIndex:1];
                    [enc setBuffer:mp.batch_softmax_part_buf  offset:0 atIndex:2];
                    [enc setBuffer:mp.batch_tok_to_stream_buf offset:0 atIndex:3];
                    [enc setBuffer:mp.batch_stream_meta_buf   offset:0 atIndex:4];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tg_vec, (NSUInteger)B, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    ++n_dispatches;

                    [enc setComputePipelineState:mp.pso_batch_softmax_sum_p2];
                    [enc setBuffer:mp.batch_softmax_part_buf  offset:0 atIndex:0];
                    [enc setBuffer:mp.batch_row_inv_sum_buf   offset:0 atIndex:1];
                    [enc setBuffer:mp.batch_tok_to_stream_buf offset:0 atIndex:2];
                    [enc setBuffer:mp.batch_stream_meta_buf   offset:0 atIndex:3];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)B, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    ++n_dispatches;
                }
                if (any_top_p) {
                    for (int pass = 0; pass < 6; ++pass) {
                        BatchSoftmaxPrm prm{V_UNPAD, n_tg_vec, n_tg_vec, s, pass, 1, max_k, 0};
                        [enc setComputePipelineState:mp.pso_batch_topp_hist];
                        [enc setBuffer:mp.batch_q_probs_buf       offset:0 atIndex:0];
                        [enc setBuffer:mp.batch_row_inv_sum_buf   offset:0 atIndex:1];
                        [enc setBuffer:mp.batch_topp_states_buf   offset:0 atIndex:2];
                        [enc setBuffer:mp.batch_softmax_part_buf  offset:0 atIndex:3];
                        [enc setBuffer:mp.batch_topp_part_cnt_buf offset:0 atIndex:4];
                        [enc setBuffer:mp.batch_tok_to_stream_buf offset:0 atIndex:5];
                        [enc setBuffer:mp.batch_stream_meta_buf   offset:0 atIndex:6];
                        [enc setBytes:&prm length:sizeof(prm) atIndex:7];
                        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tg_vec, (NSUInteger)B, 1)
                            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        ++n_dispatches;

                        [enc setComputePipelineState:mp.pso_batch_topp_scan];
                        [enc setBuffer:mp.batch_softmax_part_buf  offset:0 atIndex:0];
                        [enc setBuffer:mp.batch_topp_part_cnt_buf offset:0 atIndex:1];
                        [enc setBuffer:mp.batch_topp_states_buf   offset:0 atIndex:2];
                        [enc setBuffer:mp.batch_tok_to_stream_buf offset:0 atIndex:3];
                        [enc setBuffer:mp.batch_stream_meta_buf   offset:0 atIndex:4];
                        [enc setBytes:&prm length:sizeof(prm) atIndex:5];
                        [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)B, 1)
                            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                        ++n_dispatches;
                    }
                    {
                        BatchSoftmaxPrm prm{V_UNPAD, n_tg_vec, n_tg_vec, s, 0, 1, max_k, 1};
                        [enc setComputePipelineState:mp.pso_batch_topp_mask_sum];
                        [enc setBuffer:mp.batch_q_probs_buf       offset:0 atIndex:0];
                        [enc setBuffer:mp.batch_topp_states_buf   offset:0 atIndex:1];
                        [enc setBuffer:mp.batch_softmax_part_buf  offset:0 atIndex:2];
                        [enc setBuffer:mp.batch_tok_to_stream_buf offset:0 atIndex:3];
                        [enc setBuffer:mp.batch_stream_meta_buf   offset:0 atIndex:4];
                        [enc setBytes:&prm length:sizeof(prm) atIndex:5];
                        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tg_vec, (NSUInteger)B, 1)
                            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        ++n_dispatches;

                        [enc setComputePipelineState:mp.pso_batch_softmax_sum_p2];
                        [enc setBuffer:mp.batch_softmax_part_buf  offset:0 atIndex:0];
                        [enc setBuffer:mp.batch_row_inv_sum_buf   offset:0 atIndex:1];
                        [enc setBuffer:mp.batch_tok_to_stream_buf offset:0 atIndex:2];
                        [enc setBuffer:mp.batch_stream_meta_buf   offset:0 atIndex:3];
                        [enc setBytes:&prm length:sizeof(prm) atIndex:4];
                        [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)B, 1)
                            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        ++n_dispatches;
                    }
                }
                {
                    BatchGumbelPrm prm{V_UNPAD, n_tg_vec, s, max_k};
                    [enc setComputePipelineState:mp.pso_batch_probs_scale_gumbel_p1];
                    [enc setBuffer:mp.batch_q_probs_buf     offset:0 atIndex:0];
                    [enc setBuffer:mp.batch_row_inv_sum_buf offset:0 atIndex:1];
                    [enc setBuffer:mp.batch_stream_meta_buf offset:0 atIndex:2];
                    [enc setBuffer:mp.batch_markov_part_val offset:0 atIndex:3];
                    [enc setBuffer:mp.batch_markov_part_idx offset:0 atIndex:4];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tg_vec, (NSUInteger)B, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    ++n_dispatches;

                    [enc setComputePipelineState:mp.pso_batch_gumbel_p2];
                    [enc setBuffer:mp.batch_markov_part_val offset:0 atIndex:0];
                    [enc setBuffer:mp.batch_markov_part_idx offset:0 atIndex:1];
                    [enc setBuffer:mp.batch_stream_meta_buf offset:0 atIndex:2];
                    [enc setBuffer:mp.batch_d_toks_buf      offset:0 atIndex:3];
                    [enc setBuffer:mp.batch_prev_tok_buf    offset:0 atIndex:4];
                    [enc setBuffer:mp.batch_verify_toks_buf offset:0 atIndex:5];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:6];
                    [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)B, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    ++n_dispatches;
                }
            }
        }

        {
            const float embed_eps = ctx->globals.embed_norm_eps;
            [enc setComputePipelineState:ctx->pso_embed_from_buf];
            [enc setBuffer:ctx->globals.embed_tokens.buf offset:ctx->globals.embed_tokens.offset atIndex:0];
            [enc setBuffer:ctx->globals.embed_norm.buf   offset:ctx->globals.embed_norm.offset   atIndex:1];
            [enc setBuffer:ctx->scratch                  offset:ctx->off_x                       atIndex:2];
            [enc setBuffer:mp.batch_verify_toks_buf      offset:0                                atIndex:3];
            [enc setBytes:&embed_eps length:sizeof(embed_eps) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)m_v, 1)
                threadsPerThreadgroup:MTLSizeMake((NSUInteger)ctx->mix_tg, 1, 1)];
            ++n_dispatches;
        }

        const int32_t RANK_T = std::max(1, ctx->cfg.loop_gate_rank);
        const int32_t E_T = ctx->cfg.n_experts;
        const int32_t K_TOP_T = ctx->cfg.top_k;
        const int32_t n_caps = (int32_t)mp.capture_layer_ids.size();
        const size_t block_stride_bytes = (size_t)m_v * (size_t)D * sizeof(float);

        auto dispatch_copy_bf16 = [&](id<MTLBuffer> sb, size_t so, id<MTLBuffer> db, size_t doff, uint32_t count) {
            if (count == 0) return;
            CopyPrm prm{count};
            [enc setComputePipelineState:ctx->pso_copy_bf16];
            [enc setBuffer:sb offset:so   atIndex:0];
            [enc setBuffer:db offset:doff atIndex:1];
            [enc setBytes:&prm length:sizeof(prm) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake((count + 255) / 256, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            ++n_dispatches;
        };

        auto dispatch_copy_f32 = [&](id<MTLBuffer> sb, size_t so, id<MTLBuffer> db, size_t doff, uint32_t count) {
            if (count == 0) return;
            CopyPrm prm{count};
            [enc setComputePipelineState:ctx->pso_copy_f32];
            [enc setBuffer:sb offset:so   atIndex:0];
            [enc setBuffer:db offset:doff atIndex:1];
            [enc setBytes:&prm length:sizeof(prm) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake((count + 255) / 256, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            ++n_dispatches;
        };

        auto encode_mix_noconv = [&](
            id<MTLBuffer> x_b, size_t x_o,
            id<MTLBuffer> blk_b, size_t blk_o,
            id<MTLBuffer> qry_b, size_t qry_o,
            id<MTLBuffer> res_b, size_t res_o,
            id<MTLBuffer> msg_b, size_t msg_o,
            id<MTLBuffer> dn_b, size_t dn_o,
            id<MTLBuffer> fg_b, size_t fg_o,
            id<MTLBuffer> gn_b, size_t gn_o,
            id<MTLBuffer> nw_b, size_t nw_o,
            id<MTLBuffer> out_b, size_t out_o,
            id<MTLBuffer> part_b, size_t part_o,
            id<MTLBuffer> resid_b, size_t resid_o,
            int K, int Q, int QO, int QN, int MOFF,
            bool add, bool gate, bool norm, float eps)
        {
            if (gate && m_v >= 16 && !no_nax_env) {
                MixGateP1Prm p1{m_v, Q, QO, MOFF, add ? 1 : 0, 0};
                [enc setComputePipelineState:get_mix_gate_p1_pso(ctx, false, K)];
                [enc setBuffer:x_b          offset:x_o                  atIndex:0];
                [enc setBuffer:blk_b        offset:blk_o                atIndex:1];
                [enc setBuffer:qry_b        offset:qry_o                atIndex:2];
                [enc setBuffer:ctx->zero_buf offset:0                   atIndex:3];
                [enc setBuffer:ctx->zero_buf offset:0                   atIndex:4];
                [enc setBuffer:res_b        offset:res_o                atIndex:5];
                [enc setBuffer:msg_b        offset:msg_o                atIndex:6];
                [enc setBuffer:ctx->scratch offset:ctx->off_msg_o       atIndex:7];
                [enc setBuffer:ctx->scratch offset:ctx->off_msg_gate_in atIndex:8];
                [enc setBuffer:part_b       offset:part_o               atIndex:9];
                [enc setBuffer:resid_b      offset:resid_o              atIndex:10];
                [enc setBuffer:ctx->zero_buf offset:0                   atIndex:11];
                [enc setBytes:&p1 length:sizeof(p1) atIndex:12];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)QO, (NSUInteger)m_v, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)ctx->mix_tg, 1, 1)];
                ++n_dispatches;

                MsgDownPrm pd{m_v};
                [enc setComputePipelineState:ctx->pso_msg_down_batched];
                [enc setBuffer:ctx->scratch offset:ctx->off_msg_gate_in atIndex:0];
                [enc setBuffer:dn_b         offset:dn_o                 atIndex:1];
                [enc setBuffer:ctx->scratch offset:ctx->off_msg_low     atIndex:2];
                [enc setBytes:&pd length:sizeof(pd) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((RANK_T + 7) / 8),
                                                      (NSUInteger)QO,
                                                      (NSUInteger)((m_v + 15) / 16))
                    threadsPerThreadgroup:MTLSizeMake(8 * 32, 1, 1)];
                ++n_dispatches;

                MsgFgPrm pf{m_v, QO, MOFF};
                [enc setComputePipelineState:ctx->pso_msg_fg_blend];
                [enc setBuffer:ctx->scratch offset:ctx->off_msg_low atIndex:0];
                [enc setBuffer:fg_b         offset:fg_o             atIndex:1];
                [enc setBuffer:gn_b         offset:gn_o             atIndex:2];
                [enc setBuffer:msg_b        offset:msg_o            atIndex:3];
                [enc setBuffer:ctx->scratch offset:ctx->off_msg_o   atIndex:4];
                [enc setBytes:&pf length:sizeof(pf) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((D + 31) / 32), (NSUInteger)QO, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
                ++n_dispatches;

                RmsMultiPrm pm{QO, QN, eps};
                [enc setComputePipelineState:ctx->pso_rms_multi];
                [enc setBuffer:ctx->scratch offset:ctx->off_msg_o atIndex:0];
                [enc setBuffer:nw_b         offset:nw_o             atIndex:1];
                [enc setBuffer:out_b        offset:out_o            atIndex:2];
                [enc setBytes:&pm length:sizeof(pm) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)QO, (NSUInteger)m_v, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)ctx->mix_tg, 1, 1)];
                ++n_dispatches;
                return;
            }

            MixPrm prm{m_v, Q, QO, QN, MOFF, add ? 1 : 0, norm ? 1 : 0, 0, eps};
            [enc setComputePipelineState:get_mix_pso(ctx, gate, false, K)];
            [enc setBuffer:x_b          offset:x_o     atIndex:0];
            [enc setBuffer:blk_b        offset:blk_o   atIndex:1];
            [enc setBuffer:qry_b        offset:qry_o   atIndex:2];
            [enc setBuffer:ctx->zero_buf offset:0      atIndex:3];
            [enc setBuffer:ctx->zero_buf offset:0      atIndex:4];
            [enc setBuffer:res_b        offset:res_o   atIndex:5];
            [enc setBuffer:msg_b        offset:msg_o   atIndex:6];
            [enc setBuffer:dn_b         offset:dn_o    atIndex:7];
            [enc setBuffer:fg_b         offset:fg_o    atIndex:8];
            [enc setBuffer:gn_b         offset:gn_o    atIndex:9];
            [enc setBuffer:nw_b         offset:nw_o    atIndex:10];
            [enc setBuffer:out_b        offset:out_o   atIndex:11];
            [enc setBuffer:part_b       offset:part_o  atIndex:12];
            [enc setBuffer:resid_b      offset:resid_o atIndex:13];
            [enc setBuffer:ctx->zero_buf offset:0      atIndex:14];
            [enc setBytes:&prm length:sizeof(prm) atIndex:15];
            if (gate) {
                size_t tg_len = align_up((size_t)D * sizeof(float), 16);
                [enc setThreadgroupMemoryLength:tg_len atIndex:0];
                [enc setThreadgroupMemoryLength:tg_len atIndex:1];
            } else {
                [enc setThreadgroupMemoryLength:16 atIndex:0];
                [enc setThreadgroupMemoryLength:16 atIndex:1];
            }
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)QO, (NSUInteger)m_v, 1)
                threadsPerThreadgroup:MTLSizeMake((NSUInteger)ctx->mix_tg, 1, 1)];
            ++n_dispatches;
        };

        auto encode_target_linear = [&](
            id<MTLBuffer> x_b, size_t x_o,
            id<MTLBuffer> w_b, size_t w_o,
            id<MTLBuffer> out_b, size_t out_o,
            int S, int K, int N, int AT3, int AT4)
        {
            if (use_nax_v_qkv && (K % 16) == 0) {
                constexpr int bn = 64;
                const int nt0 = (S > 1) ? ((AT3 + bn - 1) / bn) : ((N + bn - 1) / bn);
                const int nt1 = (S > 1) ? (((AT4 - AT3) + bn - 1) / bn) : 0;
                const int nt2 = (S > 1) ? (((N - AT4) + bn - 1) / bn) : 0;
                const int total_nt = nt0 + nt1 + nt2;
                NaxQkvPrm prm{m_v, S, K, N, AT3, AT4, nt0, nt1, nt2};
                [enc setComputePipelineState:pso_nax_v];
                [enc setBuffer:x_b   offset:x_o   atIndex:0];
                [enc setBuffer:w_b   offset:w_o   atIndex:1];
                [enc setBuffer:out_b offset:out_o atIndex:2];
                [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)total_nt, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(4 * 32, 1, 1)];
                ++n_dispatches;
            } else {
                const int sg = (N >= 4096) ? 8 : 4;
                const int ny = (N + sg - 1) / sg;
                for (int m0 = 0; m0 < m_v; m0 += 8) {
                    const int cm = std::min(8, m_v - m0);
                    QkvPrm prm{S, N, AT3, AT4};
                    [enc setComputePipelineState:get_qkv_pso(ctx, cm, sg, K)];
                    [enc setBuffer:x_b   offset:x_o + (size_t)m0 * (size_t)S * (size_t)K * sizeof(uint16_t) atIndex:0];
                    [enc setBuffer:w_b   offset:w_o                                                         atIndex:1];
                    [enc setBuffer:out_b offset:out_o + (size_t)m0 * (size_t)N * sizeof(uint16_t)           atIndex:2];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)ny, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)sg, 1)];
                    ++n_dispatches;
                }
            }
        };

        int cur_k = 0, pre_loop_k = 0, bank_k = 0;
        for (int step = 0; step < NV; ++step) {
            const VisitEntry& ve = ctx->schedule[step];
            const int layer_idx = ve.layer_idx;
            const int pass_idx = ve.pass_idx;
            const LayerDeviceWeights& L = ctx->layers[layer_idx];
            const size_t step_vmeta_off = (size_t)step * (size_t)B * sizeof(BatchMTPTargetVisitMetaGpu);

            if (layer_idx == ctx->cfg.loop_start) {
                if (pass_idx == 0) {
                    dispatch_copy_bf16(ctx->scratch, ctx->off_x, ctx->scratch, ctx->off_pre_loop_x, (uint32_t)(m_v * D));
                    pre_loop_k = cur_k;
                } else {
                    int done = pre_loop_k;
                    bank_k = std::max(0, cur_k - (done + 1));
                    if (bank_k > 0) {
                        dispatch_copy_f32(
                            ctx->scratch, ctx->off_blocks + (size_t)(done + 1) * block_stride_bytes,
                            ctx->scratch, ctx->off_bank_blocks,
                            (uint32_t)((size_t)bank_k * (size_t)m_v * (size_t)D));
                    }
                    dispatch_copy_bf16(ctx->scratch, ctx->off_x, ctx->scratch, ctx->off_bank_x, (uint32_t)(m_v * D));
                    dispatch_copy_bf16(ctx->scratch, ctx->off_pre_loop_x, ctx->scratch, ctx->off_x, (uint32_t)(m_v * D));
                    cur_k = pre_loop_k;
                }
            }

            for (int ci = 0; ci < n_caps; ++ci) {
                if (mp.capture_layer_ids[(size_t)ci] == step) {
                    CopyCapPrm cprm{m_v, ci, n_caps};
                    [enc setComputePipelineState:ctx->pso_copy_cap];
                    [enc setBuffer:ctx->scratch      offset:ctx->off_x atIndex:0];
                    [enc setBuffer:mp.batch_vcap_buf offset:0          atIndex:1];
                    [enc setBytes:&cprm length:sizeof(cprm) atIndex:2];
                    [enc dispatchThreadgroups:MTLSizeMake(((uint32_t)D + 255) / 256, (NSUInteger)m_v, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    ++n_dispatches;
                }
            }

            const bool gated = (pass_idx > 0);
            if (gated) {
                size_t mq_off = ctx->globals.message_queries.offset
                              + (size_t)(layer_idx - ctx->cfg.loop_start) * 4 * (size_t)D * sizeof(float);
                MixQ4Prm m4prm{m_v};
                [enc setComputePipelineState:get_mix_q4_pso(ctx, bank_k)];
                [enc setBuffer:ctx->scratch                  offset:ctx->off_bank_x      atIndex:0];
                [enc setBuffer:ctx->scratch                  offset:ctx->off_bank_blocks atIndex:1];
                [enc setBuffer:ctx->globals.message_queries.buf offset:mq_off            atIndex:2];
                [enc setBuffer:ctx->scratch                  offset:ctx->off_messages    atIndex:3];
                [enc setBytes:&m4prm length:sizeof(m4prm) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)m_v, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)ctx->mix_tg, 1, 1)];
                ++n_dispatches;
            }

            size_t normed_dst_off = L.is_block_boundary
                                  ? (ctx->off_blocks + (size_t)cur_k * block_stride_bytes)
                                  : ctx->off_scratch_partial;
            int qkv_s = 1;
            if (!L.has_attn_input) {
                Layer0Prm l0{L.attn_eps};
                [enc setComputePipelineState:ctx->pso_layer0_rms];
                [enc setBuffer:ctx->scratch    offset:ctx->off_x        atIndex:0];
                [enc setBuffer:L.attn_norm.buf offset:L.attn_norm.offset atIndex:1];
                [enc setBuffer:ctx->scratch    offset:normed_dst_off    atIndex:2];
                [enc setBuffer:ctx->scratch    offset:ctx->off_qkv_in   atIndex:3];
                [enc setBytes:&l0 length:sizeof(l0) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)m_v, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)ctx->mix_tg, 1, 1)];
                ++n_dispatches;
                qkv_s = 1;
            } else {
                int Q = L.is_sliding ? 1 : 3;
                int QO = gated ? 3 : Q;
                int QN = L.is_sliding ? 1 : 3;
                encode_mix_noconv(
                    ctx->scratch, ctx->off_x,
                    ctx->scratch, ctx->off_blocks,
                    L.attn_query.buf, L.attn_query.offset,
                    ctx->zero_buf, 0,
                    ctx->scratch, ctx->off_messages,
                    L.attn_gate_down.buf, L.attn_gate_down.offset,
                    L.attn_gate_forget.buf, L.attn_gate_forget.offset,
                    L.attn_gate_gain.buf, L.attn_gate_gain.offset,
                    L.attn_norm.buf, L.attn_norm.offset,
                    ctx->scratch, ctx->off_qkv_in,
                    ctx->scratch, normed_dst_off,
                    ctx->scratch, ctx->off_resid,
                    cur_k, Q, QO, QN, 0,
                    false, gated, true, L.attn_eps);
                qkv_s = QO;
            }

            encode_target_linear(
                ctx->scratch, ctx->off_qkv_in,
                L.stack.buf, L.stack.offset,
                ctx->scratch, ctx->off_qkv,
                qkv_s, D, L.qkv_total_rows, L.qkv_at[3], L.qkv_at[4]);

            {
                const int tg = std::max(32, ((L.head_dim + 31) / 32) * 32);
                const bool use_tau = (!L.is_sliding && ctx->cfg.log_scaling_n_floor > 0.0f);
                KvPrepMtpBatchedPrm kprm{};
                kprm.M           = m_v;
                kprm.HQ          = L.n_heads;
                kprm.HKV         = L.n_kv;
                kprm.R           = L.rel_extent;
                kprm.QKV_W       = L.qkv_total_rows;
                kprm.QOFF        = L.qkv_at[0];
                kprm.ROFF        = L.qkv_at[1];
                kprm.KOFF        = L.qkv_at[3];
                kprm.VOFF        = L.qkv_at[4];
                kprm.X_STRIDE    = qkv_s * D;
                kprm.XK_OFF      = (qkv_s == 3) ? D : 0;
                kprm.XV_OFF      = (qkv_s == 3) ? (2 * D) : 0;
                kprm.TAU         = use_tau ? 1 : 0;
                kprm.qk_thr      = ctx->cfg.qk_l2_clip_threshold;
                kprm.rel_thr     = ctx->cfg.rel_l2_clip_threshold;
                kprm.eps         = ctx->cfg.rms_norm_eps;
                kprm.log_n_floor = ctx->cfg.log_scaling_n_floor;
                kprm.log_alpha   = ctx->cfg.log_scaling_alpha;

                [enc setComputePipelineState:get_kv_prep_mtp_batched_pso(ctx, !L.is_sliding, L.head_dim)];
                [enc setBuffer:ctx->scratch             offset:ctx->off_qkv            atIndex:0];
                [enc setBuffer:ctx->scratch             offset:ctx->off_qkv_in         atIndex:1];
                [enc setBuffer:L.sink_w.buf             offset:L.sink_w.offset         atIndex:2];
                [enc setBuffer:L.kdyn_w.buf             offset:L.kdyn_w.offset         atIndex:3];
                [enc setBuffer:L.vdyn_w.buf             offset:L.vdyn_w.offset         atIndex:4];
                [enc setBuffer:L.k_taps.buf             offset:L.k_taps.offset         atIndex:5];
                [enc setBuffer:L.v_taps.buf             offset:L.v_taps.offset         atIndex:6];
                [enc setBuffer:L.q_scale.buf            offset:L.q_scale.offset        atIndex:7];
                [enc setBuffer:L.rel_scale.buf          offset:L.rel_scale.offset      atIndex:8];
                [enc setBuffer:L.rel_proj.buf           offset:L.rel_proj.offset       atIndex:9];
                [enc setBuffer:ctx->scratch             offset:ctx->off_qo             atIndex:10];
                [enc setBuffer:ctx->scratch             offset:ctx->off_bias           atIndex:11];
                [enc setBuffer:ctx->scratch             offset:ctx->off_sinks          atIndex:12];
                [enc setBuffer:mp.batch_tok_to_stream_buf offset:0                     atIndex:13];
                [enc setBuffer:mp.batch_tok_local_pos_buf offset:0                     atIndex:14];
                [enc setBuffer:mp.batch_target_vmeta_buf  offset:step_vmeta_off        atIndex:15];
                [enc setBytes:&kprm length:sizeof(kprm) atIndex:16];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)L.n_heads, (NSUInteger)m_v, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)tg, 1, 1)];
                ++n_dispatches;
            }

            {
                const TargetVisitPlan& vp = vplans[(size_t)step];
                const int G = L.n_heads / L.n_kv;
                const int win_tg = window_tg(L.head_dim);
                const float attn_scale = 1.0f / std::sqrt((float)L.head_dim);
                if (!vp.use_splitk) {
                    AttendMtpBatchedPrm aprm{m_v, L.n_heads, L.n_kv, L.rel_extent,
                                             L.qkv_total_rows, L.qkv_at[2], L.sliding_window, attn_scale};
                    [enc setComputePipelineState:get_attend_mtp_batched_pso(ctx, G, L.head_dim)];
                    [enc setBuffer:ctx->scratch             offset:ctx->off_qo             atIndex:0];
                    [enc setBuffer:ctx->scratch             offset:ctx->off_bias           atIndex:1];
                    [enc setBuffer:ctx->scratch             offset:ctx->off_sinks          atIndex:2];
                    [enc setBuffer:ctx->scratch             offset:ctx->off_qkv            atIndex:3];
                    [enc setBuffer:ctx->scratch             offset:ctx->off_att            atIndex:4];
                    [enc setBuffer:mp.batch_tok_to_stream_buf offset:0                     atIndex:5];
                    [enc setBuffer:mp.batch_tok_local_pos_buf offset:0                     atIndex:6];
                    [enc setBuffer:mp.batch_target_vmeta_buf  offset:step_vmeta_off        atIndex:7];
                    [enc setBytes:&aprm length:sizeof(aprm) atIndex:8];
                    [enc setThreadgroupMemoryLength:align_up((size_t)G * (size_t)win_tg * sizeof(float), 16) atIndex:0];
                    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)L.n_kv, (NSUInteger)m_v, 1)
                        threadsPerThreadgroup:MTLSizeMake((NSUInteger)win_tg, 1, 1)];
                    ++n_dispatches;
                } else {
                    DecodePartMtpBatchedPrm dprm{m_v, L.n_heads, L.n_kv, L.rel_extent,
                                                 vp.bs, vp.nblocks, L.sliding_window, attn_scale};
                    [enc setComputePipelineState:get_dpart_mtp_batched_pso(ctx, G, L.head_dim)];
                    [enc setBuffer:ctx->scratch             offset:ctx->off_qo             atIndex:0];
                    [enc setBuffer:ctx->scratch             offset:ctx->off_bias           atIndex:1];
                    [enc setBuffer:ctx->part_buf            offset:ctx->part_off_max       atIndex:2];
                    [enc setBuffer:ctx->part_buf            offset:ctx->part_off_sum       atIndex:3];
                    [enc setBuffer:ctx->part_buf            offset:ctx->part_off_acc       atIndex:4];
                    [enc setBuffer:mp.batch_tok_to_stream_buf offset:0                     atIndex:5];
                    [enc setBuffer:mp.batch_tok_local_pos_buf offset:0                     atIndex:6];
                    [enc setBuffer:mp.batch_target_vmeta_buf  offset:step_vmeta_off        atIndex:7];
                    [enc setBytes:&dprm length:sizeof(dprm) atIndex:8];
                    [enc setThreadgroupMemoryLength:align_up(8 * (size_t)G * (size_t)L.head_dim * sizeof(float), 16) atIndex:0];
                    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)vp.nblocks, (NSUInteger)L.n_kv, (NSUInteger)m_v)
                        threadsPerThreadgroup:MTLSizeMake(8 * 32, 1, 1)];
                    ++n_dispatches;

                    DecodeCombPrm cprm{L.n_heads, L.head_dim, vp.nblocks, L.qkv_total_rows, L.qkv_at[2]};
                    [enc setComputePipelineState:ctx->pso_decode_comb];
                    [enc setBuffer:ctx->part_buf offset:ctx->part_off_max atIndex:0];
                    [enc setBuffer:ctx->part_buf offset:ctx->part_off_sum atIndex:1];
                    [enc setBuffer:ctx->part_buf offset:ctx->part_off_acc atIndex:2];
                    [enc setBuffer:ctx->scratch  offset:ctx->off_sinks    atIndex:3];
                    [enc setBuffer:ctx->scratch  offset:ctx->off_qkv      atIndex:4];
                    [enc setBuffer:ctx->scratch  offset:ctx->off_att      atIndex:5];
                    [enc setBytes:&cprm length:sizeof(cprm) atIndex:6];
                    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)L.n_heads, (NSUInteger)m_v, 1)
                        threadsPerThreadgroup:MTLSizeMake((NSUInteger)L.head_dim, 1, 1)];
                    ++n_dispatches;
                }
            }

            encode_target_linear(
                ctx->scratch, ctx->off_att,
                L.o_proj.buf, L.o_proj.offset,
                ctx->scratch, ctx->off_att_out,
                1, L.n_heads * L.head_dim, D, 0, 0);

            if (L.is_block_boundary) cur_k += 1;
            {
                MixPrm prm{m_v, 1, 1, 1, 3, !L.is_block_boundary ? 1 : 0, 1, 1, L.mlp_eps};
                [enc setComputePipelineState:get_mix_mtp_batched_pso(ctx, gated, cur_k)];
                [enc setBuffer:ctx->scratch             offset:ctx->off_att_out        atIndex:0];
                [enc setBuffer:ctx->scratch             offset:ctx->off_blocks         atIndex:1];
                [enc setBuffer:L.mlp_query.buf          offset:L.mlp_query.offset      atIndex:2];
                [enc setBuffer:L.attn_taps.buf          offset:L.attn_taps.offset      atIndex:3];
                [enc setBuffer:(L.is_block_boundary ? ctx->zero_buf : ctx->scratch)
                        offset:(L.is_block_boundary ? 0 : ctx->off_x)                  atIndex:4];
                [enc setBuffer:ctx->scratch             offset:ctx->off_messages       atIndex:5];
                [enc setBuffer:L.mlp_gate_down.buf      offset:L.mlp_gate_down.offset  atIndex:6];
                [enc setBuffer:L.mlp_gate_forget.buf    offset:L.mlp_gate_forget.offset atIndex:7];
                [enc setBuffer:L.mlp_gate_gain.buf      offset:L.mlp_gate_gain.offset  atIndex:8];
                [enc setBuffer:L.mlp_norm.buf           offset:L.mlp_norm.offset       atIndex:9];
                [enc setBuffer:ctx->scratch             offset:ctx->off_mlp_in         atIndex:10];
                [enc setBuffer:ctx->scratch             offset:ctx->off_scratch_partial atIndex:11];
                [enc setBuffer:ctx->scratch             offset:ctx->off_resid          atIndex:12];
                [enc setBuffer:mp.batch_tok_to_stream_buf offset:0                     atIndex:13];
                [enc setBuffer:mp.batch_tok_local_pos_buf offset:0                     atIndex:14];
                [enc setBuffer:mp.batch_target_vmeta_buf  offset:step_vmeta_off        atIndex:15];
                [enc setBytes:&prm length:sizeof(prm) atIndex:16];
                if (gated) {
                    size_t tg_len = align_up((size_t)D * sizeof(float), 16);
                    [enc setThreadgroupMemoryLength:tg_len atIndex:0];
                    [enc setThreadgroupMemoryLength:tg_len atIndex:1];
                } else {
                    [enc setThreadgroupMemoryLength:16 atIndex:0];
                    [enc setThreadgroupMemoryLength:16 atIndex:1];
                }
                [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)m_v, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)ctx->mix_tg, 1, 1)];
                ++n_dispatches;
            }

            if (m_v >= 16 && E_T >= 32) {
                RouteBatchedPrm rprm{m_v, ctx->cfg.route_scale};
                [enc setComputePipelineState:ctx->pso_route_scores];
                [enc setBuffer:ctx->scratch   offset:ctx->off_mlp_in       atIndex:0];
                [enc setBuffer:L.router_w.buf offset:L.router_w.offset     atIndex:1];
                [enc setBuffer:ctx->scratch   offset:ctx->off_route_scores atIndex:2];
                [enc setBytes:&rprm length:sizeof(rprm) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)E_T, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_dispatches;

                [enc setComputePipelineState:ctx->pso_route_topk];
                [enc setBuffer:ctx->scratch   offset:ctx->off_route_scores atIndex:0];
                [enc setBuffer:L.router_b.buf offset:L.router_b.offset     atIndex:1];
                [enc setBuffer:ctx->scratch   offset:ctx->off_inds         atIndex:2];
                [enc setBuffer:ctx->scratch   offset:ctx->off_weights      atIndex:3];
                [enc setBytes:&rprm length:sizeof(rprm) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)m_v, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)ctx->bucket_tg, 1, 1)];
                ++n_dispatches;
            } else {
                RoutePrm rprm{ctx->cfg.route_scale};
                [enc setComputePipelineState:ctx->pso_route];
                [enc setBuffer:ctx->scratch   offset:ctx->off_mlp_in  atIndex:0];
                [enc setBuffer:L.router_w.buf offset:L.router_w.offset atIndex:1];
                [enc setBuffer:L.router_b.buf offset:L.router_b.offset atIndex:2];
                [enc setBuffer:ctx->scratch   offset:ctx->off_inds    atIndex:3];
                [enc setBuffer:ctx->scratch   offset:ctx->off_weights atIndex:4];
                [enc setBytes:&rprm length:sizeof(rprm) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)m_v, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)(E_T * ctx->route_parts), 1, 1)];
                ++n_dispatches;
            }

            const bool is_nvfp4 = (L.moe_mode == 0);
            MoePrm mprm{ctx->cfg.mlp_gate_softcap, ctx->cfg.mlp_up_softcap};
            if (m_v <= 2) {
                const int sg13 = is_nvfp4 ? 8 : 4;
                const int sg2  = 8;
                const int nx13 = (ctx->cfg.inter_size + sg13 * 4 - 1) / (sg13 * 4);
                const int nx2  = (D + sg2 * 4 - 1) / (sg2 * 4);
                if (is_nvfp4) {
                    [enc setComputePipelineState:ctx->pso_m1_w13_nvfp4];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_mlp_in     atIndex:0];
                    [enc setBuffer:L.w13_weight.buf offset:L.w13_weight.offset atIndex:1];
                    [enc setBuffer:L.w13_scales.buf offset:L.w13_scales.offset atIndex:2];
                    [enc setBuffer:L.w13_gscale.buf offset:L.w13_gscale.offset atIndex:3];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_inds       atIndex:4];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_moe_h      atIndex:5];
                    [enc setBytes:&mprm length:sizeof(mprm) atIndex:6];
                    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nx13, (NSUInteger)K_TOP_T, (NSUInteger)m_v)
                        threadsPerThreadgroup:MTLSizeMake((NSUInteger)(32 * sg13), 1, 1)];
                    ++n_dispatches;

                    [enc setComputePipelineState:ctx->pso_m1_w2_nvfp4];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_moe_h     atIndex:0];
                    [enc setBuffer:L.w2_weight.buf offset:L.w2_weight.offset atIndex:1];
                    [enc setBuffer:L.w2_scales.buf offset:L.w2_scales.offset atIndex:2];
                    [enc setBuffer:L.w2_gscale.buf offset:L.w2_gscale.offset atIndex:3];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_inds      atIndex:4];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_ys        atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nx2, (NSUInteger)K_TOP_T, (NSUInteger)m_v)
                        threadsPerThreadgroup:MTLSizeMake((NSUInteger)(32 * sg2), 1, 1)];
                    ++n_dispatches;
                } else {
                    [enc setComputePipelineState:ctx->pso_m1_w13_mxfp8];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_mlp_in     atIndex:0];
                    [enc setBuffer:L.w13_weight.buf offset:L.w13_weight.offset atIndex:1];
                    [enc setBuffer:L.w13_scales.buf offset:L.w13_scales.offset atIndex:2];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_inds       atIndex:3];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_moe_h      atIndex:4];
                    [enc setBytes:&mprm length:sizeof(mprm) atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nx13, (NSUInteger)K_TOP_T, (NSUInteger)m_v)
                        threadsPerThreadgroup:MTLSizeMake((NSUInteger)(32 * sg13), 1, 1)];
                    ++n_dispatches;

                    [enc setComputePipelineState:ctx->pso_m1_w2_mxfp8];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_moe_h     atIndex:0];
                    [enc setBuffer:L.w2_weight.buf offset:L.w2_weight.offset atIndex:1];
                    [enc setBuffer:L.w2_scales.buf offset:L.w2_scales.offset atIndex:2];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_inds      atIndex:3];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_ys        atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nx2, (NSUInteger)K_TOP_T, (NSUInteger)m_v)
                        threadsPerThreadgroup:MTLSizeMake((NSUInteger)(32 * sg2), 1, 1)];
                    ++n_dispatches;
                }
            } else {
                const int t_slots = m_v * K_TOP_T;
                const int nx13 = (ctx->cfg.inter_size + sg_grp * 4 - 1) / (sg_grp * 4);
                const int nx2  = (D + sg_grp * 4 - 1) / (sg_grp * 4);
                BucketTilesPrm bprm{t_slots, grp_mt, nx13, nx2};
                [enc setComputePipelineState:ctx->pso_bucket_tiles];
                [enc setBuffer:ctx->scratch offset:ctx->off_inds          atIndex:0];
                [enc setBuffer:ctx->scratch offset:ctx->off_offsets       atIndex:1];
                [enc setBuffer:ctx->scratch offset:ctx->off_sorted_slots  atIndex:2];
                [enc setBuffer:ctx->scratch offset:ctx->off_tile_expert   atIndex:3];
                [enc setBuffer:ctx->scratch offset:ctx->off_tile_t0       atIndex:4];
                [enc setBuffer:ctx->scratch offset:ctx->off_indirect_args atIndex:5];
                [enc setBytes:&bprm length:sizeof(bprm) atIndex:6];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)ctx->bucket_tg, 1, 1)];
                ++n_dispatches;

                if (is_nvfp4) {
                    [enc setComputePipelineState:pso_grp_w13_nv];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_mlp_in       atIndex:0];
                    [enc setBuffer:L.w13_weight.buf offset:L.w13_weight.offset   atIndex:1];
                    [enc setBuffer:L.w13_scales.buf offset:L.w13_scales.offset   atIndex:2];
                    [enc setBuffer:L.w13_gscale.buf offset:L.w13_gscale.offset   atIndex:3];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_offsets      atIndex:4];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_sorted_slots atIndex:5];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_tile_expert  atIndex:6];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_tile_t0      atIndex:7];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_moe_h        atIndex:8];
                    [enc setBytes:&mprm length:sizeof(mprm) atIndex:9];
                    [enc dispatchThreadgroupsWithIndirectBuffer:ctx->scratch
                                           indirectBufferOffset:ctx->off_indirect_args
                                          threadsPerThreadgroup:MTLSizeMake((NSUInteger)(32 * sg_grp), 1, 1)];
                    ++n_dispatches;

                    [enc setComputePipelineState:pso_grp_w2_nv];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_moe_h        atIndex:0];
                    [enc setBuffer:L.w2_weight.buf offset:L.w2_weight.offset    atIndex:1];
                    [enc setBuffer:L.w2_scales.buf offset:L.w2_scales.offset    atIndex:2];
                    [enc setBuffer:L.w2_gscale.buf offset:L.w2_gscale.offset    atIndex:3];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_offsets      atIndex:4];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_sorted_slots atIndex:5];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_tile_expert  atIndex:6];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_tile_t0      atIndex:7];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_ys           atIndex:8];
                    [enc dispatchThreadgroupsWithIndirectBuffer:ctx->scratch
                                           indirectBufferOffset:ctx->off_indirect_args + 16
                                          threadsPerThreadgroup:MTLSizeMake((NSUInteger)(32 * sg_grp), 1, 1)];
                    ++n_dispatches;
                } else {
                    [enc setComputePipelineState:pso_grp_w13_mx];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_mlp_in       atIndex:0];
                    [enc setBuffer:L.w13_weight.buf offset:L.w13_weight.offset   atIndex:1];
                    [enc setBuffer:L.w13_scales.buf offset:L.w13_scales.offset   atIndex:2];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_offsets      atIndex:3];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_sorted_slots atIndex:4];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_tile_expert  atIndex:5];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_tile_t0      atIndex:6];
                    [enc setBuffer:ctx->scratch     offset:ctx->off_moe_h        atIndex:7];
                    [enc setBytes:&mprm length:sizeof(mprm) atIndex:8];
                    [enc dispatchThreadgroupsWithIndirectBuffer:ctx->scratch
                                           indirectBufferOffset:ctx->off_indirect_args
                                          threadsPerThreadgroup:MTLSizeMake((NSUInteger)(32 * sg_grp), 1, 1)];
                    ++n_dispatches;

                    [enc setComputePipelineState:pso_grp_w2_mx];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_moe_h        atIndex:0];
                    [enc setBuffer:L.w2_weight.buf offset:L.w2_weight.offset    atIndex:1];
                    [enc setBuffer:L.w2_scales.buf offset:L.w2_scales.offset    atIndex:2];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_offsets      atIndex:3];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_sorted_slots atIndex:4];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_tile_expert  atIndex:5];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_tile_t0      atIndex:6];
                    [enc setBuffer:ctx->scratch    offset:ctx->off_ys           atIndex:7];
                    [enc dispatchThreadgroupsWithIndirectBuffer:ctx->scratch
                                           indirectBufferOffset:ctx->off_indirect_args + 16
                                          threadsPerThreadgroup:MTLSizeMake((NSUInteger)(32 * sg_grp), 1, 1)];
                    ++n_dispatches;
                }
            }

            {
                CombinePrm cbprm{m_v};
                int tg = std::min(256, D);
                int nx = (D + tg - 1) / tg;
                [enc setComputePipelineState:ctx->pso_combine_mtp_batched];
                [enc setBuffer:ctx->scratch               offset:ctx->off_ys           atIndex:0];
                [enc setBuffer:ctx->scratch               offset:ctx->off_weights      atIndex:1];
                [enc setBuffer:L.mlp_taps.buf             offset:L.mlp_taps.offset    atIndex:2];
                [enc setBuffer:ctx->scratch               offset:ctx->off_resid        atIndex:3];
                [enc setBuffer:ctx->scratch               offset:ctx->off_x            atIndex:4];
                [enc setBuffer:mp.batch_tok_to_stream_buf offset:0                    atIndex:5];
                [enc setBuffer:mp.batch_tok_local_pos_buf offset:0                    atIndex:6];
                [enc setBuffer:mp.batch_target_vmeta_buf  offset:step_vmeta_off       atIndex:7];
                [enc setBytes:&cbprm length:sizeof(cbprm) atIndex:8];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nx, (NSUInteger)m_v, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)tg, 1, 1)];
                ++n_dispatches;
            }
        }

        encode_mix_noconv(
            ctx->scratch, ctx->off_x,
            ctx->scratch, ctx->off_blocks,
            ctx->globals.final_query.buf, ctx->globals.final_query.offset,
            ctx->zero_buf, 0,
            ctx->zero_buf, 0,
            ctx->zero_buf, 0,
            ctx->zero_buf, 0,
            ctx->zero_buf, 0,
            ctx->zero_buf, 0,
            ctx->scratch, ctx->off_pre_norm,
            ctx->scratch, ctx->off_scratch_partial,
            ctx->scratch, ctx->off_resid,
            cur_k, 1, 1, 1, 0,
            false, false, false, 1e-6f);

        int final_cap_slot = -1;
        for (int ci = 0; ci < n_caps; ++ci) {
            if (mp.capture_layer_ids[(size_t)ci] == NV) {
                final_cap_slot = ci;
                break;
            }
        }

        int32_t target_head_ntg = 0;
        FinalNormPrm fprm{final_cap_slot, n_caps, ctx->globals.final_norm_eps};
        if (use_nax_v_head) {
            [enc setComputePipelineState:ctx->pso_final_norm_bf16_cap];
            [enc setBuffer:ctx->scratch                offset:ctx->off_pre_norm           atIndex:0];
            [enc setBuffer:ctx->globals.final_norm.buf offset:ctx->globals.final_norm.offset atIndex:1];
            [enc setBuffer:ctx->scratch                offset:ctx->off_normed_bf16        atIndex:2];
            [enc setBuffer:mp.batch_vcap_buf           offset:0                             atIndex:3];
            [enc setBytes:&fprm length:sizeof(fprm) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)m_v, 1)
                threadsPerThreadgroup:MTLSizeMake((NSUInteger)ctx->mix_tg, 1, 1)];
            ++n_dispatches;

            target_head_ntg = n_tg_target_nax;
            const int ntg_m = (m_v + 15) / 16;
            NaxArgmaxPrm nap{m_v, target_head_ntg};
            [enc setComputePipelineState:ctx->pso_nax_head_logits_max_16x64];
            [enc setBuffer:ctx->scratch                                              offset:ctx->off_normed_bf16 atIndex:0];
            [enc setBuffer:target_lm_head.buf                                        offset:target_lm_head.offset atIndex:1];
            [enc setBuffer:(any_sampled ? mp.batch_p_probs_buf : ctx->zero_buf)      offset:0                     atIndex:2];
            [enc setBuffer:(any_sampled ? mp.batch_softmax_part_buf : ctx->zero_buf) offset:0                     atIndex:3];
            [enc setBuffer:mp.batch_argmax_pv_buf                                    offset:0                     atIndex:4];
            [enc setBuffer:mp.batch_argmax_pi_buf                                    offset:0                     atIndex:5];
            [enc setBuffer:mp.batch_tok_to_stream_buf                                offset:0                     atIndex:6];
            [enc setBuffer:mp.batch_stream_meta_buf                                  offset:0                     atIndex:7];
            [enc setBytes:&nap length:sizeof(nap) atIndex:8];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)target_head_ntg, (NSUInteger)ntg_m, 1)
                threadsPerThreadgroup:MTLSizeMake(4 * 32, 1, 1)];
            ++n_dispatches;

            if (!all_sampled) {
                ArgmaxP2Prm p2{target_head_ntg};
                [enc setComputePipelineState:ctx->pso_argmax_p2];
                [enc setBuffer:mp.batch_argmax_pv_buf offset:0                atIndex:0];
                [enc setBuffer:mp.batch_argmax_pi_buf offset:0                atIndex:1];
                [enc setBuffer:ctx->scratch           offset:ctx->off_out_toks atIndex:2];
                [enc setBytes:&p2 length:sizeof(p2) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)m_v, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_dispatches;
            }
        } else {
            [enc setComputePipelineState:ctx->pso_final_norm];
            [enc setBuffer:ctx->scratch                offset:ctx->off_pre_norm           atIndex:0];
            [enc setBuffer:ctx->globals.final_norm.buf offset:ctx->globals.final_norm.offset atIndex:1];
            [enc setBuffer:ctx->scratch                offset:ctx->off_normed_f32         atIndex:2];
            [enc setBuffer:mp.batch_vcap_buf           offset:0                             atIndex:3];
            [enc setBytes:&fprm length:sizeof(fprm) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)m_v, 1)
                threadsPerThreadgroup:MTLSizeMake((NSUInteger)ctx->mix_tg, 1, 1)];
            ++n_dispatches;

            target_head_ntg = n_tg_target_simt;
            if (!any_sampled && pso_argmax_simt_m != nil) {
                ArgmaxP1Prm p1{target_head_ntg};
                [enc setComputePipelineState:pso_argmax_simt_m];
                [enc setBuffer:ctx->scratch           offset:ctx->off_normed_f32   atIndex:0];
                [enc setBuffer:target_lm_head.buf     offset:target_lm_head.offset atIndex:1];
                [enc setBuffer:mp.batch_argmax_pv_buf offset:0                     atIndex:2];
                [enc setBuffer:mp.batch_argmax_pi_buf offset:0                     atIndex:3];
                [enc setBytes:&p1 length:sizeof(p1) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)target_head_ntg, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(8 * 32, 1, 1)];
                ++n_dispatches;
            } else {
                for (int m0 = 0; m0 < m_v; m0 += 8) {
                    const int cm = std::min(8, m_v - m0);
                    ArgmaxP1Prm p1{target_head_ntg};
                    [enc setComputePipelineState:get_head_logits_max_mixed_p1_pso(ctx, cm)];
                    [enc setBuffer:ctx->scratch
                            offset:ctx->off_normed_f32 + (size_t)m0 * (size_t)D * sizeof(float) atIndex:0];
                    [enc setBuffer:target_lm_head.buf
                            offset:target_lm_head.offset                                         atIndex:1];
                    [enc setBuffer:(any_sampled ? mp.batch_p_probs_buf : ctx->zero_buf)
                            offset:(any_sampled ? (size_t)m0 * (size_t)V_UNPAD * sizeof(float) : 0) atIndex:2];
                    [enc setBuffer:(any_sampled ? mp.batch_softmax_part_buf : ctx->zero_buf)
                            offset:(any_sampled ? (size_t)m0 * (size_t)target_head_ntg * sizeof(float) : 0) atIndex:3];
                    [enc setBuffer:mp.batch_argmax_pv_buf
                            offset:(size_t)m0 * (size_t)target_head_ntg * sizeof(float)          atIndex:4];
                    [enc setBuffer:mp.batch_argmax_pi_buf
                            offset:(size_t)m0 * (size_t)target_head_ntg * sizeof(uint32_t)       atIndex:5];
                    [enc setBuffer:mp.batch_tok_to_stream_buf
                            offset:(size_t)m0 * sizeof(int32_t)                                  atIndex:6];
                    [enc setBuffer:mp.batch_stream_meta_buf
                            offset:0                                                             atIndex:7];
                    [enc setBytes:&p1 length:sizeof(p1) atIndex:8];
                    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)target_head_ntg, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(8 * 32, 1, 1)];
                    ++n_dispatches;
                }
            }
            if (!all_sampled) {
                ArgmaxP2Prm p2{target_head_ntg};
                [enc setComputePipelineState:ctx->pso_argmax_p2];
                [enc setBuffer:mp.batch_argmax_pv_buf offset:0                atIndex:0];
                [enc setBuffer:mp.batch_argmax_pi_buf offset:0                atIndex:1];
                [enc setBuffer:ctx->scratch           offset:ctx->off_out_toks atIndex:2];
                [enc setBytes:&p2 length:sizeof(p2) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)m_v, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_dispatches;
            }
        }

        if (any_sampled) {
            {
                BatchSoftmaxPrm prm{V_UNPAD, target_head_ntg, n_tg_vec, 0, 0, 0, max_k, 0};
                [enc setComputePipelineState:mp.pso_batch_softmax_max_p2];
                [enc setBuffer:mp.batch_softmax_part_buf  offset:0 atIndex:0];
                [enc setBuffer:mp.batch_row_max_buf       offset:0 atIndex:1];
                [enc setBuffer:mp.batch_tok_to_stream_buf offset:0 atIndex:2];
                [enc setBuffer:mp.batch_stream_meta_buf   offset:0 atIndex:3];
                [enc setBytes:&prm length:sizeof(prm) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)m_v, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_dispatches;
            }
            {
                BatchSoftmaxPrm prm{V_UNPAD, n_tg_vec, n_tg_vec, 0, 0, 0, max_k, 0};
                [enc setComputePipelineState:mp.pso_batch_softmax_exp_sum_p1];
                [enc setBuffer:mp.batch_p_probs_buf       offset:0 atIndex:0];
                [enc setBuffer:mp.batch_row_max_buf       offset:0 atIndex:1];
                [enc setBuffer:mp.batch_softmax_part_buf  offset:0 atIndex:2];
                [enc setBuffer:mp.batch_tok_to_stream_buf offset:0 atIndex:3];
                [enc setBuffer:mp.batch_stream_meta_buf   offset:0 atIndex:4];
                [enc setBytes:&prm length:sizeof(prm) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tg_vec, (NSUInteger)m_v, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_dispatches;

                [enc setComputePipelineState:mp.pso_batch_softmax_sum_p2];
                [enc setBuffer:mp.batch_softmax_part_buf  offset:0 atIndex:0];
                [enc setBuffer:mp.batch_row_inv_sum_buf   offset:0 atIndex:1];
                [enc setBuffer:mp.batch_tok_to_stream_buf offset:0 atIndex:2];
                [enc setBuffer:mp.batch_stream_meta_buf   offset:0 atIndex:3];
                [enc setBytes:&prm length:sizeof(prm) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)m_v, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_dispatches;
            }
            if (any_top_p) {
                for (int pass = 0; pass < 6; ++pass) {
                    BatchSoftmaxPrm prm{V_UNPAD, n_tg_vec, n_tg_vec, 0, pass, 0, max_k, 0};
                    [enc setComputePipelineState:mp.pso_batch_topp_hist];
                    [enc setBuffer:mp.batch_p_probs_buf       offset:0 atIndex:0];
                    [enc setBuffer:mp.batch_row_inv_sum_buf   offset:0 atIndex:1];
                    [enc setBuffer:mp.batch_topp_states_buf   offset:0 atIndex:2];
                    [enc setBuffer:mp.batch_softmax_part_buf  offset:0 atIndex:3];
                    [enc setBuffer:mp.batch_topp_part_cnt_buf offset:0 atIndex:4];
                    [enc setBuffer:mp.batch_tok_to_stream_buf offset:0 atIndex:5];
                    [enc setBuffer:mp.batch_stream_meta_buf   offset:0 atIndex:6];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:7];
                    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tg_vec, (NSUInteger)m_v, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    ++n_dispatches;

                    [enc setComputePipelineState:mp.pso_batch_topp_scan];
                    [enc setBuffer:mp.batch_softmax_part_buf  offset:0 atIndex:0];
                    [enc setBuffer:mp.batch_topp_part_cnt_buf offset:0 atIndex:1];
                    [enc setBuffer:mp.batch_topp_states_buf   offset:0 atIndex:2];
                    [enc setBuffer:mp.batch_tok_to_stream_buf offset:0 atIndex:3];
                    [enc setBuffer:mp.batch_stream_meta_buf   offset:0 atIndex:4];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)m_v, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    ++n_dispatches;
                }
                {
                    BatchSoftmaxPrm prm{V_UNPAD, n_tg_vec, n_tg_vec, 0, 0, 0, max_k, 1};
                    [enc setComputePipelineState:mp.pso_batch_topp_mask_sum];
                    [enc setBuffer:mp.batch_p_probs_buf       offset:0 atIndex:0];
                    [enc setBuffer:mp.batch_topp_states_buf   offset:0 atIndex:1];
                    [enc setBuffer:mp.batch_softmax_part_buf  offset:0 atIndex:2];
                    [enc setBuffer:mp.batch_tok_to_stream_buf offset:0 atIndex:3];
                    [enc setBuffer:mp.batch_stream_meta_buf   offset:0 atIndex:4];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tg_vec, (NSUInteger)m_v, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    ++n_dispatches;

                    [enc setComputePipelineState:mp.pso_batch_softmax_sum_p2];
                    [enc setBuffer:mp.batch_softmax_part_buf  offset:0 atIndex:0];
                    [enc setBuffer:mp.batch_row_inv_sum_buf   offset:0 atIndex:1];
                    [enc setBuffer:mp.batch_tok_to_stream_buf offset:0 atIndex:2];
                    [enc setBuffer:mp.batch_stream_meta_buf   offset:0 atIndex:3];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)m_v, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    ++n_dispatches;
                }
            }
            {
                BatchSoftmaxPrm prm{V_UNPAD, n_tg_vec, n_tg_vec, 0, 0, 0, max_k, 0};
                [enc setComputePipelineState:mp.pso_batch_probs_scale];
                [enc setBuffer:mp.batch_p_probs_buf       offset:0 atIndex:0];
                [enc setBuffer:mp.batch_row_inv_sum_buf   offset:0 atIndex:1];
                [enc setBuffer:mp.batch_tok_to_stream_buf offset:0 atIndex:2];
                [enc setBuffer:mp.batch_stream_meta_buf   offset:0 atIndex:3];
                [enc setBytes:&prm length:sizeof(prm) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tg_vec, (NSUInteger)m_v, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_dispatches;
            }
        }

        {
            BatchRejectPrm rjprm{V_UNPAD, n_tg_vec, max_k, 0};
            if (!all_sampled) {
                [enc setComputePipelineState:mp.pso_batch_greedy_accept];
                [enc setBuffer:mp.batch_d_toks_buf      offset:0                atIndex:0];
                [enc setBuffer:ctx->scratch           offset:ctx->off_out_toks atIndex:1];
                [enc setBuffer:mp.batch_stream_meta_buf offset:0                atIndex:2];
                [enc setBuffer:mp.batch_out_toks_buf    offset:0                atIndex:3];
                [enc setBuffer:mp.batch_d_n_acc_buf     offset:0                atIndex:4];
                [enc setBytes:&rjprm length:sizeof(rjprm) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)B, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                ++n_dispatches;
            }
            if (any_sampled) {
                [enc setComputePipelineState:mp.pso_batch_reject_prefix];
                [enc setBuffer:mp.batch_d_toks_buf      offset:0 atIndex:0];
                [enc setBuffer:mp.batch_p_probs_buf     offset:0 atIndex:1];
                [enc setBuffer:mp.batch_q_probs_buf     offset:0 atIndex:2];
                [enc setBuffer:mp.batch_stream_meta_buf offset:0 atIndex:3];
                [enc setBuffer:mp.batch_out_toks_buf    offset:0 atIndex:4];
                [enc setBuffer:mp.batch_d_reject_j_buf  offset:0 atIndex:5];
                [enc setBuffer:mp.batch_d_n_acc_buf     offset:0 atIndex:6];
                [enc setBytes:&rjprm length:sizeof(rjprm) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)B, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                ++n_dispatches;

                [enc setComputePipelineState:mp.pso_batch_reject_sample_p1];
                [enc setBuffer:mp.batch_p_probs_buf     offset:0 atIndex:0];
                [enc setBuffer:mp.batch_q_probs_buf     offset:0 atIndex:1];
                [enc setBuffer:mp.batch_d_reject_j_buf  offset:0 atIndex:2];
                [enc setBuffer:mp.batch_stream_meta_buf offset:0 atIndex:3];
                [enc setBuffer:mp.batch_reject_res_val  offset:0 atIndex:4];
                [enc setBuffer:mp.batch_reject_res_idx  offset:0 atIndex:5];
                [enc setBuffer:mp.batch_reject_p_val    offset:0 atIndex:6];
                [enc setBuffer:mp.batch_reject_p_idx    offset:0 atIndex:7];
                [enc setBuffer:mp.batch_reject_res_sum  offset:0 atIndex:8];
                [enc setBytes:&rjprm length:sizeof(rjprm) atIndex:9];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tg_vec, (NSUInteger)B, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_dispatches;

                [enc setComputePipelineState:mp.pso_batch_reject_sample_p2];
                [enc setBuffer:mp.batch_reject_res_val  offset:0 atIndex:0];
                [enc setBuffer:mp.batch_reject_res_idx  offset:0 atIndex:1];
                [enc setBuffer:mp.batch_reject_p_val    offset:0 atIndex:2];
                [enc setBuffer:mp.batch_reject_p_idx    offset:0 atIndex:3];
                [enc setBuffer:mp.batch_reject_res_sum  offset:0 atIndex:4];
                [enc setBuffer:mp.batch_d_reject_j_buf  offset:0 atIndex:5];
                [enc setBuffer:mp.batch_stream_meta_buf offset:0 atIndex:6];
                [enc setBuffer:mp.batch_out_toks_buf    offset:0 atIndex:7];
                [enc setBytes:&rjprm length:sizeof(rjprm) atIndex:8];
                [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)B, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_dispatches;
            }
        }

        {
            const int32_t row_elems = NCAP * D;
            MtpBatchCommitPrm cprm{B, NV, row_elems};
            if (any_need_staging) {
                [enc setComputePipelineState:ctx->pso_kv_commit_rot_mtp_batched];
                [enc setBuffer:mp.batch_target_vmeta_buf offset:0 atIndex:0];
                [enc setBuffer:mp.batch_d_n_acc_buf      offset:0 atIndex:1];
                [enc setBytes:&cprm length:sizeof(cprm) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((max_commit_elems + 255) / 256),
                                                      (NSUInteger)NV,
                                                      (NSUInteger)B)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_dispatches;
            }
            [enc setComputePipelineState:ctx->pso_copy_conv4_mtp_batched];
            [enc setBuffer:mp.batch_target_vmeta_buf offset:0 atIndex:0];
            [enc setBuffer:mp.batch_d_n_acc_buf      offset:0 atIndex:1];
            [enc setBytes:&cprm length:sizeof(cprm) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((max_conv_elems + 255) / 256),
                                                  (NSUInteger)NV,
                                                  (NSUInteger)B)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            ++n_dispatches;

            [enc setComputePipelineState:ctx->pso_copy_cap_mtp_batched];
            [enc setBuffer:mp.batch_vcap_buf        offset:0 atIndex:0];
            [enc setBuffer:mp.batch_stream_meta_buf offset:0 atIndex:1];
            [enc setBuffer:mp.batch_d_n_acc_buf     offset:0 atIndex:2];
            [enc setBytes:&cprm length:sizeof(cprm) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((row_elems + 255) / 256),
                                                  (NSUInteger)max_mv_b,
                                                  (NSUInteger)B)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            ++n_dispatches;
        }

        [enc endEncoding];
        const uint64_t t_enc1 = mach_absolute_time();

        [cb commit];
        [cb waitUntilCompleted];
        const uint64_t t_wall1 = mach_absolute_time();
        kept_bufs.clear();

        if (cb.status != MTLCommandBufferStatusCompleted) {
            fprintf(stderr, "[inkling_decode] ERROR: batched mtp command buffer failed: %s\n",
                    cb.error ? cb.error.localizedDescription.UTF8String : "unknown");
            return -10;
        }

        const int32_t* d_n_acc_ptr  = (const int32_t*)[mp.batch_d_n_acc_buf contents];
        const int32_t* gpu_out_toks = (const int32_t*)[mp.batch_out_toks_buf contents];
        for (int b = 0; b < B; ++b) {
            InklingMTPStreamDesc& s = streams[b];
            const int32_t M_b = s.num_draft + 1;
            const int32_t n_acc = std::max<int32_t>(1, std::min<int32_t>(M_b, d_n_acc_ptr[b]));
            for (int32_t j = 0; j < n_acc; ++j) {
                s.out_tokens[j] = gpu_out_toks[b * B_BLK + j];
            }
            s.out_n_acc = n_acc;

            for (int v = 0; v < NV; ++v) {
                const BatchMTPTargetVisitMetaGpu& vm = vm_cpu[(size_t)v * (size_t)B + (size_t)b];
                InklingKVMeta& km = s.kv_meta[v];
                if (vm.need_staging != 0) {
                    km.offset = vm.prev_offset + n_acc;
                    km.idx    = (km.offset <= vm.max_size)
                              ? km.offset
                              : (((vm.old_idx + n_acc - 1) % vm.max_size) + 1);
                } else {
                    km.offset = vm.prev_offset + n_acc;
                    km.idx    = vm.prev_offset + n_acc;
                }
            }

            for (int i = 0; i < mc.num_layers; ++i) {
                s.draft_kv_descs[i].offset   = d_plans[b].off_after[i];
                s.draft_kv_descs[i].ring_idx = d_plans[b].next_ring[i];
            }
        }

        if (stats_out) {
            stats_out->encode_ms    = mach_to_ms(t_enc1 - t_enc0, ctx->tb);
            stats_out->gpu_ms       = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
            stats_out->wall_ms      = mach_to_ms(t_wall1 - t_wall0, ctx->tb);
            stats_out->n_dispatches = n_dispatches;
        }
        return 0;
    }
}

int inkling_ctx_mtp_debug_read(
    InklingCtx* ctx,
    int32_t     buf_id,
    void*       dst_ptr,
    size_t      dst_nbytes)
{
    @autoreleasepool {
        if (!ctx || !ctx->mtp.attached || !dst_ptr || dst_nbytes == 0) return -1;
        const MTPAttachedState& mp = ctx->mtp;
        id<MTLBuffer> src_buf = nil;
        size_t src_off = 0;
        switch (buf_id) {
            case 0: src_buf = mp.q_probs_buf;     break;
            case 1: src_buf = mp.p_probs_buf;     break;
            case 2: src_buf = mp.d_toks_buf;      break;
            case 3: src_buf = ctx->scratch; src_off = ctx->off_out_toks; break;
            case 4: src_buf = mp.d_reject_j_buf;  break;
            case 5: src_buf = mp.d_n_acc_buf;     break;
            case 6: src_buf = mp.verify_toks_buf; break;
            case 7: src_buf = mp.out_toks_buf;    break;
            case 8: src_buf = ctx->scratch; src_off = ctx->off_pre_norm; break;
            default: return -2;
        }
        if (!src_buf || src_off + dst_nbytes > (size_t)[src_buf length]) return -3;
        std::memcpy(dst_ptr, (const uint8_t*)[src_buf contents] + src_off, dst_nbytes);
        return 0;
    }
}

}
