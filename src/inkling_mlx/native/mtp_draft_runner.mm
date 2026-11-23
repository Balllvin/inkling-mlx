#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "mtp_draft_runner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mach/vm_page_size.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

static double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static inline size_t align_up(size_t n, size_t align) {
    return ((n + align - 1) / align) * align;
}

static const char* kMTPDraftMSL = R"METAL(
#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

constant int FC_M              [[function_constant(0)]];
constant int FC_K              [[function_constant(1)]];
constant int FC_SG             [[function_constant(2)]];
constant int FC_HAS_RES        [[function_constant(3)]];
constant int FC_HAS_EMBED_NORM [[function_constant(4)]];

constant float LOG2E = 1.4426950408889634f;

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
    constant InitXPrm& p              [[buffer(6)]],
    uint tid                          [[thread_position_in_threadgroup]],
    uint lane                         [[thread_index_in_simdgroup]],
    uint sg                           [[simdgroup_index_in_threadgroup]])
{
    const device bfloat* emb_row = embed_tokens + size_t(p.anchor_token) * size_t(p.D);
    if (FC_HAS_EMBED_NORM != 0) {
        float ss = 0.0f;
        for (int d = int(tid); d < p.D; d += 256) {
            float v = float(emb_row[d]);
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

    if (int(tid) < p.markov_rank) {
        rnn_state[tid] = bfloat(0.0f);
    }
    if (tid == 0) {
        prev_tok[0] = p.anchor_token;
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
    const int r = int(tgid);
    const int src_row = p.row_start + r;
    const device bfloat* h_base = hidden + size_t(src_row) * size_t(p.n_target_layers) * size_t(p.D);
    const device float* a_row = fuse_alpha + size_t(p.alpha_row) * size_t(p.n_target_layers);

    const float a0 = float(bfloat(a_row[0]));
    const float a1 = float(bfloat(a_row[1]));
    const float a2 = float(bfloat(a_row[2]));
    const float a3 = float(bfloat(a_row[3]));

    float vals[16];
    float ss = 0.0f;
    int cnt = 0;
    for (int d = int(tid); d < p.D; d += 256) {
        float v = float(h_base[0 * p.D + d]) * a0
                + float(h_base[1 * p.D + d]) * a1
                + float(h_base[2 * p.D + d]) * a2
                + float(h_base[3 * p.D + d]) * a3;
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
    if (tid == 0) sg_ss[0] = metal::precise::rsqrt(tot / float(p.D) + p.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = sg_ss[0];

    device bfloat* out_row = out + size_t(r) * size_t(p.D);
    int idx = 0;
    for (int d = int(tid); d < p.D; d += 256) {
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
    const int r = int(tgid);
    const int sr = p.src_row_off + r;
    const device bfloat* xr = x_in + size_t(sr) * size_t(p.D);
    const device bfloat* rr = res_in + size_t(sr) * size_t(p.D);
    device bfloat* ro = res_out + size_t(r) * size_t(p.D);
    device bfloat* no = norm_out + size_t(r) * size_t(p.D);

    float vals[16];
    float ss = 0.0f;
    int cnt = 0;
    for (int d = int(tid); d < p.D; d += 256) {
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
    if (tid == 0) sg_ss[0] = metal::precise::rsqrt(tot / float(p.D) + p.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = sg_ss[0];

    int idx = 0;
    for (int d = int(tid); d < p.D; d += 256) {
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
    const device vec<bfloat, 4>* w4 = (const device vec<bfloat, 4>*)(w + size_t(row) * size_t(FC_K));
    const device vec<bfloat, 4>* x4 = (const device vec<bfloat, 4>*)(x);

    float acc[8];
    #pragma clang loop unroll(full)
    for (int m = 0; m < FC_M; ++m) acc[m] = 0.0f;

    for (int base = 0; base < K4; base += 96) {
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
    const device vec<bfloat, 4>* wa4 = (const device vec<bfloat, 4>*)(w + size_t(row0) * size_t(FC_K));
    const device vec<bfloat, 4>* wb4 = (const device vec<bfloat, 4>*)(w + size_t(row1) * size_t(FC_K));
    const device vec<bfloat, 4>* x4  = (const device vec<bfloat, 4>*)(x);

    float acc0[7];
    float acc1[7];
    #pragma clang loop unroll(full)
    for (int m = 0; m < FC_M; ++m) {
        acc0[m] = 0.0f;
        acc1[m] = 0.0f;
    }

    for (int base = 0; base < K4; base += 64) {
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

    for (int base = 0; base < K4; base += 64) {
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
    const int t = int(tgid.x);
    const int h = int(tgid.y);
    const int d = int(tid2.x);

    const device bfloat* row = kv_ctx + size_t(t) * size_t(2 * p.kv_dim);
    const device bfloat* k_src = row + h * 128;
    const device bfloat* v_src = row + p.kv_dim + h * 128;

    const float k0 = float(k_src[d]);
    const float k1 = float(k_src[d + 64]);

    float ss = simd_sum(k0 * k0 + k1 * k1);
    threadgroup float sg_ss[2];
    if (lane == 0) sg_ss[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = metal::precise::rsqrt((sg_ss[0] + sg_ss[1]) / 128.0f + p.eps);

    const float kn0 = float(bfloat(float(bfloat(k0 * inv_rms)) * float(k_norm_w[d])));
    const float kn1 = float(bfloat(float(bfloat(k1 * inv_rms)) * float(k_norm_w[d + 64])));

    const float theta = float(p.offset_before + t) * metal::exp2(p.neg_log2_theta_over_half * float(d));
    const float c = metal::precise::cos(theta);
    const float s = metal::precise::sin(theta);

    const int dst_slot = (p.is_rotating != 0) ? ((p.ring_start + t) % p.window) : (p.offset_before + t);
    device bfloat* k_dst = k_cache + size_t(h) * size_t(p.k_stride_h) + size_t(dst_slot) * size_t(p.k_stride_s);
    device bfloat* v_dst = v_cache + size_t(h) * size_t(p.v_stride_h) + size_t(dst_slot) * size_t(p.v_stride_s);

    k_dst[d]      = bfloat(kn0 * c - kn1 * s);
    k_dst[d + 64] = bfloat(kn1 * c + kn0 * s);
    v_dst[d]      = v_src[d];
    v_dst[d + 64] = v_src[d + 64];
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
    const int t = int(tgid.x);
    const int idx = int(tgid.y);
    const int d = int(tid2.x);

    const device bfloat* row = qkv + size_t(t) * size_t(p.total_dim);

    if (idx >= 32) {
        const int h = idx - 32;
        const device bfloat* src = row + p.q_dim + p.kv_dim + h * 128;
        device bfloat* dst = bv_out + (h * 8 + t) * 128;
        dst[d]      = src[d];
        dst[d + 64] = src[d + 64];
        return;
    }

    const bool is_q = (idx < 24);
    const int h = is_q ? idx : (idx - 24);
    const device bfloat* src = is_q ? (row + h * 128) : (row + p.q_dim + h * 128);
    const device bfloat* nw = is_q ? q_norm_w : k_norm_w;

    const float x0 = float(src[d]);
    const float x1 = float(src[d + 64]);

    float ss = simd_sum(x0 * x0 + x1 * x1);
    threadgroup float sg_ss[2];
    if (lane == 0) sg_ss[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = metal::precise::rsqrt((sg_ss[0] + sg_ss[1]) / 128.0f + p.eps);

    const float n0 = float(bfloat(float(bfloat(x0 * inv_rms)) * float(nw[d])));
    const float n1 = float(bfloat(float(bfloat(x1 * inv_rms)) * float(nw[d + 64])));

    const float theta = float(p.rope_offset + t) * metal::exp2(p.neg_log2_theta_over_half * float(d));
    const float c = metal::precise::cos(theta);
    const float s = metal::precise::sin(theta);

    device bfloat* dst = is_q ? (q_out + (h * 8 + t) * 128) : (bk_out + (h * 8 + t) * 128);
    dst[d]      = bfloat(n0 * c - n1 * s);
    dst[d + 64] = bfloat(n1 * c + n0 * s);
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
    constant SdpaPrm& p,
    int kv_h,
    int k_start,
    int k_end,
    uint tid,
    uint lane,
    uint sg,
    thread vec<float, 4>& o0,
    thread vec<float, 4>& o1,
    thread vec<float, 4>& o2,
    thread float& m0, thread float& m1, thread float& m2,
    thread float& l0, thread float& l1, thread float& l2,
    threadgroup bfloat* K_tg,
    threadgroup bfloat* V_tg)
{
    const int qt = int(sg);
    const int qh0 = kv_h * 3;
    const int col4 = int(lane) * 4;

    const vec<float, 4> q0 = vec<float, 4>(*(const device vec<bfloat, 4>*)(q_buf + ((qh0 + 0) * 8 + qt) * 128 + col4));
    const vec<float, 4> q1 = vec<float, 4>(*(const device vec<bfloat, 4>*)(q_buf + ((qh0 + 1) * 8 + qt) * 128 + col4));
    const vec<float, 4> q2 = vec<float, 4>(*(const device vec<bfloat, 4>*)(q_buf + ((qh0 + 2) * 8 + qt) * 128 + col4));

    o0 = vec<float, 4>(0.0f);
    o1 = vec<float, 4>(0.0f);
    o2 = vec<float, 4>(0.0f);
    m0 = -INFINITY; m1 = -INFINITY; m2 = -INFINITY;
    l0 = 0.0f;      l1 = 0.0f;      l2 = 0.0f;

    const device bfloat* kc_h = k_cache + size_t(kv_h) * size_t(p.k_stride_h);
    const device bfloat* vc_h = v_cache + size_t(kv_h) * size_t(p.v_stride_h);
    const device bfloat* bk_h = bk_buf + size_t(kv_h) * 1024;
    const device bfloat* bv_h = bv_buf + size_t(kv_h) * 1024;

    const int load_row = int(tid >> 5);
    const int load_col4 = int(tid & 31) * 4;

    for (int k_base = k_start; k_base < k_end; k_base += 8) {
        const int valid_n = min(8, k_end - k_base);
        const int gk = k_base + load_row;
        if (gk < k_end) {
            vec<bfloat, 4> kv, vv;
            if (gk < p.valid_ctx) {
                kv = *(const device vec<bfloat, 4>*)(kc_h + size_t(gk) * size_t(p.k_stride_s) + load_col4);
                vv = *(const device vec<bfloat, 4>*)(vc_h + size_t(gk) * size_t(p.v_stride_s) + load_col4);
            } else {
                const int bi = gk - p.valid_ctx;
                kv = *(const device vec<bfloat, 4>*)(bk_h + bi * 128 + load_col4);
                vv = *(const device vec<bfloat, 4>*)(bv_h + bi * 128 + load_col4);
            }
            *(threadgroup vec<bfloat, 4>*)(K_tg + load_row * 128 + load_col4) = kv;
            *(threadgroup vec<bfloat, 4>*)(V_tg + load_row * 128 + load_col4) = vv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float s0[8], s1[8], s2[8];
        float tm0 = -INFINITY, tm1 = -INFINITY, tm2 = -INFINITY;

        if (valid_n == 8) {
            #pragma clang loop unroll(full)
            for (int j = 0; j < 8; ++j) {
                const vec<float, 4> kv = vec<float, 4>(*(threadgroup vec<bfloat, 4>*)(K_tg + j * 128 + col4));
                const float d0 = simd_sum(dot(q0, kv)) * p.scale2;
                const float d1 = simd_sum(dot(q1, kv)) * p.scale2;
                const float d2 = simd_sum(dot(q2, kv)) * p.scale2;
                s0[j] = d0; s1[j] = d1; s2[j] = d2;
                tm0 = max(tm0, d0);
                tm1 = max(tm1, d1);
                tm2 = max(tm2, d2);
            }
        } else {
            for (int j = 0; j < valid_n; ++j) {
                const vec<float, 4> kv = vec<float, 4>(*(threadgroup vec<bfloat, 4>*)(K_tg + j * 128 + col4));
                const float d0 = simd_sum(dot(q0, kv)) * p.scale2;
                const float d1 = simd_sum(dot(q1, kv)) * p.scale2;
                const float d2 = simd_sum(dot(q2, kv)) * p.scale2;
                s0[j] = d0; s1[j] = d1; s2[j] = d2;
                tm0 = max(tm0, d0);
                tm1 = max(tm1, d1);
                tm2 = max(tm2, d2);
            }
        }

        if (k_base == k_start) {
            m0 = tm0; m1 = tm1; m2 = tm2;
        } else {
            const float nm0 = max(m0, tm0);
            const float nm1 = max(m1, tm1);
            const float nm2 = max(m2, tm2);
            const float f0 = metal::precise::exp2(m0 - nm0);
            const float f1 = metal::precise::exp2(m1 - nm1);
            const float f2 = metal::precise::exp2(m2 - nm2);
            o0 *= f0; l0 *= f0;
            o1 *= f1; l1 *= f1;
            o2 *= f2; l2 *= f2;
            m0 = nm0; m1 = nm1; m2 = nm2;
        }

        if (valid_n == 8) {
            #pragma clang loop unroll(full)
            for (int j = 0; j < 8; ++j) {
                const vec<float, 4> vv = vec<float, 4>(*(threadgroup vec<bfloat, 4>*)(V_tg + j * 128 + col4));
                const float p0 = metal::precise::exp2(s0[j] - m0);
                const float p1 = metal::precise::exp2(s1[j] - m1);
                const float p2 = metal::precise::exp2(s2[j] - m2);
                l0 += p0; o0 += p0 * vv;
                l1 += p1; o1 += p1 * vv;
                l2 += p2; o2 += p2 * vv;
            }
        } else {
            for (int j = 0; j < valid_n; ++j) {
                const vec<float, 4> vv = vec<float, 4>(*(threadgroup vec<bfloat, 4>*)(V_tg + j * 128 + col4));
                const float p0 = metal::precise::exp2(s0[j] - m0);
                const float p1 = metal::precise::exp2(s1[j] - m1);
                const float p2 = metal::precise::exp2(s2[j] - m2);
                l0 += p0; o0 += p0 * vv;
                l1 += p1; o1 += p1 * vv;
                l2 += p2; o2 += p2 * vv;
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
    threadgroup bfloat K_tg[8 * 128];
    threadgroup bfloat V_tg[8 * 128];

    const int kv_h = int(tgid);
    vec<float, 4> o0, o1, o2;
    float m0, m1, m2, l0, l1, l2;
    sdpa_core(q_buf, k_cache, v_cache, bk_buf, bv_buf, p, kv_h, 0, p.total_k,
              tid, lane, sg, o0, o1, o2, m0, m1, m2, l0, l1, l2, K_tg, V_tg);

    const int qt = int(sg);
    const int qh0 = kv_h * 3;
    const int col4 = int(lane) * 4;
    const float r0 = 1.0f / l0;
    const float r1 = 1.0f / l1;
    const float r2 = 1.0f / l2;

    device bfloat* row_out = attn_out + size_t(qt) * 3072;
    *(device vec<bfloat, 4>*)(row_out + (qh0 + 0) * 128 + col4) = vec<bfloat, 4>(o0 * r0);
    *(device vec<bfloat, 4>*)(row_out + (qh0 + 1) * 128 + col4) = vec<bfloat, 4>(o1 * r1);
    *(device vec<bfloat, 4>*)(row_out + (qh0 + 2) * 128 + col4) = vec<bfloat, 4>(o2 * r2);
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
    const uint tid = tid2.x;
    threadgroup bfloat K_tg[8 * 128];
    threadgroup bfloat V_tg[8 * 128];

    const int kv_h = int(tgid.x);
    const int sp = int(tgid.y);
    const int k_start = sp * p.split_chunk;
    const int k_end = min(p.total_k, k_start + p.split_chunk);

    const int qt = int(sg);
    const int qh0 = kv_h * 3;
    const int col4 = int(lane) * 4;

    vec<float, 4> o0(0.0f), o1(0.0f), o2(0.0f);
    float m0 = -INFINITY, m1 = -INFINITY, m2 = -INFINITY;
    float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f;

    if (k_start < k_end) {
        sdpa_core(q_buf, k_cache, v_cache, bk_buf, bv_buf, p, kv_h, k_start, k_end,
                  tid, lane, sg, o0, o1, o2, m0, m1, m2, l0, l1, l2, K_tg, V_tg);
    }

    const size_t base0 = (size_t(sp) * 24 + size_t(qh0 + 0)) * 8 + size_t(qt);
    const size_t base1 = (size_t(sp) * 24 + size_t(qh0 + 1)) * 8 + size_t(qt);
    const size_t base2 = (size_t(sp) * 24 + size_t(qh0 + 2)) * 8 + size_t(qt);

    *(device vec<float, 4>*)(part_o + base0 * 128 + col4) = o0;
    *(device vec<float, 4>*)(part_o + base1 * 128 + col4) = o1;
    *(device vec<float, 4>*)(part_o + base2 * 128 + col4) = o2;

    if (lane == 0) {
        part_ml[base0 * 2 + 0] = m0; part_ml[base0 * 2 + 1] = l0;
        part_ml[base1 * 2 + 0] = m1; part_ml[base1 * 2 + 1] = l1;
        part_ml[base2 * 2 + 0] = m2; part_ml[base2 * 2 + 1] = l2;
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
    const int qt = int(tgid.x);
    const int qh = int(tgid.y);
    const int col4 = int(lane) * 4;

    float m_max = -INFINITY;
    for (int sp = 0; sp < p.n_splits; ++sp) {
        const size_t idx = (size_t(sp) * 24 + size_t(qh)) * 8 + size_t(qt);
        const float ls = part_ml[idx * 2 + 1];
        if (ls > 0.0f) {
            m_max = max(m_max, part_ml[idx * 2 + 0]);
        }
    }

    vec<float, 4> o_tot(0.0f);
    float l_tot = 0.0f;
    for (int sp = 0; sp < p.n_splits; ++sp) {
        const size_t idx = (size_t(sp) * 24 + size_t(qh)) * 8 + size_t(qt);
        const float ms = part_ml[idx * 2 + 0];
        const float ls = part_ml[idx * 2 + 1];
        if (ls > 0.0f) {
            const float w = metal::precise::exp2(ms - m_max);
            l_tot += w * ls;
            o_tot += w * (*(const device vec<float, 4>*)(part_o + idx * 128 + col4));
        }
    }

    *(device vec<bfloat, 4>*)(attn_out + size_t(qt) * 3072 + size_t(qh) * 128 + col4) =
        vec<bfloat, 4>(o_tot * (1.0f / l_tot));
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
    const int rank = p.rank;
    const int out_dim = 3 * rank;
    const int j = int(tgid) * 32 + int(lane);

    threadgroup float tg_in[512];
    const int tok = prev_tok[0];
    const device bfloat* emb = markov_w1 + size_t(tok) * size_t(rank);

    #pragma clang loop unroll(full)
    for (int i = int(lane); i < 256; i += 32) {
        tg_in[i] = float(state_in[i]);
        tg_in[256 + i] = float(emb[i]);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    for (int k = 0; k < 512; k += 4) {
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
    constexpr int RANK = 256;
    constexpr int ROWS_PER_SG = 8;
    const int v_base = int(tgid) * (SG * ROWS_PER_SG);

    threadgroup float lat[RANK];
    lat[tid] = float(latent[tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const device bfloat* base_row = base_l + size_t(p.step_idx) * size_t(p.v_limit);

    float best_val = -INFINITY;
    uint best_idx = 0u;

    const int k = int(lane) * 8;
    const float l0 = lat[k + 0], l1 = lat[k + 1], l2 = lat[k + 2], l3 = lat[k + 3];
    const float l4 = lat[k + 4], l5 = lat[k + 5], l6 = lat[k + 6], l7 = lat[k + 7];

    #pragma clang loop unroll(full)
    for (int ri = 0; ri < ROWS_PER_SG; ++ri) {
        const int v = v_base + int(sg) * ROWS_PER_SG + ri;
        if (v < p.v_limit) {
            const device vec<bfloat, 4>* wp = (const device vec<bfloat, 4>*)(fold_w + size_t(v) * RANK + k);
            const vec<bfloat, 4> wa = wp[0];
            const vec<bfloat, 4> wb = wp[1];
            const float dot_v = float(wa[0]) * l0 + float(wa[1]) * l1
                              + float(wa[2]) * l2 + float(wa[3]) * l3
                              + float(wb[0]) * l4 + float(wb[1]) * l5
                              + float(wb[2]) * l6 + float(wb[3]) * l7;
            const float total = simd_sum(dot_v);
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
    device int* out_tokens       [[buffer(2)]],
    device int* prev_tok         [[buffer(3)]],
    constant MarkovArgmaxPrm& p  [[buffer(4)]],
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
        out_tokens[p.step_idx] = int(fi);
        prev_tok[0] = int(fi);
    }
}
)METAL";

struct WrappedBuf {
    id<MTLBuffer> buf = nil;
    size_t offset = 0;
};

struct CachedKVBuf {
    id<MTLBuffer> buf = nil;
    size_t span = 0;
};

struct LayerMetalWeights {
    int32_t is_local = 0;
    int32_t window = 0;
    WrappedBuf dflare_norm;
    WrappedBuf input_layernorm;
    WrappedBuf post_attention_layernorm;
    WrappedBuf kv_ctx_w;
    WrappedBuf k_norm_ctx;
    WrappedBuf qkv_w;
    WrappedBuf q_norm;
    WrappedBuf k_norm;
    WrappedBuf o_proj_w;
    WrappedBuf gu_w;
    WrappedBuf down_proj_w;
};

struct MTPDraftRunner {
    MTPDraftConfig cfg{};
    size_t page_size = 16384;
    std::string device_name;

    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> lib = nil;
    id<MTLBuffer> dummy_buf = nil;

    id<MTLComputePipelineState> pso_init_x[2] = {nil, nil};
    id<MTLComputePipelineState> pso_dflare_mix = nil;
    id<MTLComputePipelineState> pso_add_rmsnorm[2] = {nil, nil};
    id<MTLComputePipelineState> pso_gemv_2304[9] = {};
    id<MTLComputePipelineState> pso_gemv_2304_r2[8] = {};
    id<MTLComputePipelineState> pso_gemv_3072_m8 = nil;
    id<MTLComputePipelineState> pso_gemv_9216_m8 = nil;
    id<MTLComputePipelineState> pso_gu_silu_m8 = nil;
    id<MTLComputePipelineState> pso_kv_ctx_write = nil;
    id<MTLComputePipelineState> pso_qkv_pack = nil;
    id<MTLComputePipelineState> pso_sdpa_single = nil;
    id<MTLComputePipelineState> pso_sdpa_split_p1 = nil;
    id<MTLComputePipelineState> pso_sdpa_split_p2 = nil;
    id<MTLComputePipelineState> pso_markov_ctx_p1[8] = {};
    id<MTLComputePipelineState> pso_markov_ctx_p2[8] = {};
    id<MTLComputePipelineState> pso_rnn_step = nil;
    id<MTLComputePipelineState> pso_markov_p1 = nil;
    id<MTLComputePipelineState> pso_markov_p2 = nil;

    bool has_embed_norm = false;
    WrappedBuf embed_tokens;
    WrappedBuf embed_norm;
    WrappedBuf mask_embed;
    WrappedBuf final_norm;
    WrappedBuf lm_head;
    WrappedBuf fuse_alpha;
    WrappedBuf markov_dflare_norm;
    WrappedBuf markov_w1;
    WrappedBuf rnn_w_ctx_t;
    WrappedBuf rnn_w_step_t;
    WrappedBuf rnn_b_f32;
    WrappedBuf fold_weight;
    WrappedBuf fold_bias;
    std::vector<LayerMetalWeights> layers;

    std::unordered_map<uintptr_t, CachedKVBuf> weight_buf_cache;
    std::mutex kv_mu;
    std::unordered_map<uintptr_t, CachedKVBuf> kv_buf_cache;

    id<MTLBuffer> hidden_buf = nil;
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
    id<MTLBuffer> sdpa_part_o = nil;
    id<MTLBuffer> sdpa_part_ml = nil;
    id<MTLBuffer> mlp_mid_buf = nil;
    id<MTLBuffer> x_out_buf = nil;
    id<MTLBuffer> base_logits_buf = nil;
    id<MTLBuffer> f_rnn_buf = nil;
    id<MTLBuffer> markov_ctx_part_x = nil;
    id<MTLBuffer> markov_ctx_part_f = nil;
    id<MTLBuffer> rnn_ctx_buf = nil;
    id<MTLBuffer> rnn_state_buf[2] = {nil, nil};
    id<MTLBuffer> rnn_latent_buf = nil;
    id<MTLBuffer> markov_part_val = nil;
    id<MTLBuffer> markov_part_idx = nil;
    id<MTLBuffer> prev_tok_buf = nil;
    id<MTLBuffer> out_tokens_buf = nil;
    id<MTLBuffer> inv_freq_buf = nil;
};

static constexpr int32_t kMaxSdpaSplits = 32;

static WrappedBuf wrap_ptr_in_map(
    id<MTLDevice> device,
    size_t page_size,
    id<MTLBuffer> dummy_buf,
    std::unordered_map<uintptr_t, CachedKVBuf>& cache,
    uintptr_t addr,
    size_t nbytes,
    const char* label)
{
    if (addr == 0 || nbytes == 0) {
        return {dummy_buf, 0};
    }
    const uintptr_t base = addr & ~(page_size - 1);
    const size_t off = static_cast<size_t>(addr - base);
    const size_t span = align_up(off + nbytes, page_size);

    auto it = cache.find(base);
    if (it != cache.end() && it->second.span >= span) {
        return {it->second.buf, off};
    }
    id<MTLBuffer> buf = [device newBufferWithBytesNoCopy:reinterpret_cast<void*>(base)
                                                  length:span
                                                 options:MTLResourceStorageModeShared
                                             deallocator:nil];
    if (!buf) {
        std::fprintf(stderr, "[mtp_draft] newBufferWithBytesNoCopy failed (%s): base=0x%lx span=%zu\n",
                     label, static_cast<unsigned long>(base), span);
        return {nil, 0};
    }
    cache[base] = {buf, span};
    return {buf, off};
}

static WrappedBuf wrap_ptr_nocache(
    id<MTLDevice> device,
    size_t page_size,
    id<MTLBuffer> dummy_buf,
    uintptr_t addr,
    size_t nbytes,
    const char* label)
{
    if (addr == 0 || nbytes == 0) {
        return {dummy_buf, 0};
    }
    const uintptr_t base = addr & ~(page_size - 1);
    const size_t off = static_cast<size_t>(addr - base);
    const size_t span = align_up(off + nbytes, page_size);

    id<MTLBuffer> buf = [device newBufferWithBytesNoCopy:reinterpret_cast<void*>(base)
                                                  length:span
                                                 options:MTLResourceStorageModeShared
                                             deallocator:nil];
    if (!buf) {
        std::fprintf(stderr, "[mtp_draft] newBufferWithBytesNoCopy failed (%s): base=0x%lx span=%zu\n",
                     label, static_cast<unsigned long>(base), span);
        return {nil, 0};
    }
    return {buf, off};
}

static id<MTLComputePipelineState> make_pso(
    id<MTLDevice> device,
    id<MTLLibrary> lib,
    const char* fn_name,
    MTLFunctionConstantValues* fcv)
{
    NSError* err = nil;
    id<MTLFunction> fn = nil;
    NSString* ns_name = [NSString stringWithUTF8String:fn_name];
    if (fcv != nil) {
        fn = [lib newFunctionWithName:ns_name constantValues:fcv error:&err];
    } else {
        fn = [lib newFunctionWithName:ns_name];
    }
    if (!fn) {
        std::fprintf(stderr, "[mtp_draft] Function '%s' failed: %s\n", fn_name,
                     err ? [[err localizedDescription] UTF8String] : "not found");
        return nil;
    }
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        std::fprintf(stderr, "[mtp_draft] PSO '%s' failed: %s\n", fn_name,
                     err ? [[err localizedDescription] UTF8String] : "unknown");
        return nil;
    }
    return pso;
}

static id<MTLComputePipelineState> make_gemv_pso(
    id<MTLDevice> device, id<MTLLibrary> lib, const char* fn_name, int m, int k, int sg)
{
    MTLFunctionConstantValues* fcv = [MTLFunctionConstantValues new];
    [fcv setConstantValue:&m type:MTLDataTypeInt atIndex:0];
    [fcv setConstantValue:&k type:MTLDataTypeInt atIndex:1];
    [fcv setConstantValue:&sg type:MTLDataTypeInt atIndex:2];
    return make_pso(device, lib, fn_name, fcv);
}

struct HostInitXPrm {
    int32_t anchor_token, D, block_size, markov_rank;
    float embed_norm_eps;
};
struct HostMixPrm {
    int32_t D, n_target_layers, row_start, alpha_row;
    float eps;
};
struct HostAddNormPrm {
    int32_t D, src_row_off, write_res;
    float eps;
};
struct HostGemvPrm {
    int32_t N;
};
struct HostKvCtxWritePrm {
    int32_t kv_dim, offset_before, is_rotating, window, ring_start;
    int32_t k_stride_h, k_stride_s, v_stride_h, v_stride_s;
    float eps;
    float neg_log2_theta_over_half;
};
struct HostQkvPackPrm {
    int32_t q_dim, kv_dim, total_dim, rope_offset;
    float eps;
    float neg_log2_theta_over_half;
};
struct HostSdpaPrm {
    int32_t valid_ctx, total_k;
    int32_t k_stride_h, k_stride_s, v_stride_h, v_stride_s;
    int32_t n_splits, split_chunk;
    float scale2;
};
struct HostMarkovCtxPrm {
    int32_t D, out_dim;
};
struct HostRnnStepPrm {
    int32_t step_idx, rank;
};
struct HostMarkovArgmaxPrm {
    int32_t v_limit, step_idx, n_part;
};

extern "C" {

MTPDraftRunner* mtp_draft_create(
    const MTPDraftConfig* cfg,
    const MTPDraftGlobalWeights* globals,
    const MTPDraftLayerWeights* layers)
{
    @autoreleasepool {
        if (!cfg || !globals || !layers) {
            std::fprintf(stderr, "[mtp_draft] NULL argument to mtp_draft_create\n");
            return nullptr;
        }
        if (cfg->hidden_size != 2304 || cfg->num_layers != 3 || cfg->num_heads != 24 ||
            cfg->num_kv_heads != 8 || cfg->head_dim != 128 || cfg->inter_size != 9216 ||
            cfg->block_size != 8 || cfg->markov_rank != 256 || cfg->n_target_layers != 4) {
            std::fprintf(stderr, "[mtp_draft] Unsupported MTP config dims (D=%d L=%d H=%d KV=%d HD=%d I=%d B=%d R=%d T=%d)\n",
                         cfg->hidden_size, cfg->num_layers, cfg->num_heads, cfg->num_kv_heads,
                         cfg->head_dim, cfg->inter_size, cfg->block_size, cfg->markov_rank, cfg->n_target_layers);
            return nullptr;
        }

        auto* r = new MTPDraftRunner();
        r->cfg = *cfg;
        r->page_size = vm_page_size;
        r->device = MTLCreateSystemDefaultDevice();
        if (!r->device) {
            std::fprintf(stderr, "[mtp_draft] Failed to create Metal device\n");
            delete r;
            return nullptr;
        }
        r->device_name = [[r->device name] UTF8String];
        r->queue = [r->device newCommandQueue];
        if (!r->queue) {
            std::fprintf(stderr, "[mtp_draft] Failed to create Metal command queue\n");
            delete r;
            return nullptr;
        }

        NSError* err = nil;
        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.languageVersion = MTLLanguageVersion4_0;
        if (@available(macOS 15.0, *)) {
            opts.mathMode = MTLMathModeSafe;
        }
        r->lib = [r->device newLibraryWithSource:[NSString stringWithUTF8String:kMTPDraftMSL]
                                         options:opts
                                           error:&err];
        if (!r->lib) {
            std::fprintf(stderr, "[mtp_draft] Metal 4.0 shader compile failed:\n%s\n",
                         err ? [[err localizedDescription] UTF8String] : "unknown");
            delete r;
            return nullptr;
        }

        for (int hn = 0; hn < 2; ++hn) {
            MTLFunctionConstantValues* fcv = [MTLFunctionConstantValues new];
            [fcv setConstantValue:&hn type:MTLDataTypeInt atIndex:4];
            r->pso_init_x[hn] = make_pso(r->device, r->lib, "mtp_init_x", fcv);
            if (!r->pso_init_x[hn]) { delete r; return nullptr; }
        }
        r->pso_dflare_mix = make_pso(r->device, r->lib, "mtp_dflare_mix", nil);
        for (int hr = 0; hr < 2; ++hr) {
            MTLFunctionConstantValues* fcv = [MTLFunctionConstantValues new];
            [fcv setConstantValue:&hr type:MTLDataTypeInt atIndex:3];
            r->pso_add_rmsnorm[hr] = make_pso(r->device, r->lib, "mtp_add_rmsnorm", fcv);
            if (!r->pso_add_rmsnorm[hr]) { delete r; return nullptr; }
        }
        for (int m = 1; m <= 8; ++m) {
            r->pso_gemv_2304[m] = make_gemv_pso(r->device, r->lib, "mtp_bf16_gemv", m, 2304, 4);
            if (!r->pso_gemv_2304[m]) { delete r; return nullptr; }
        }
        for (int m = 1; m <= 7; ++m) {
            r->pso_gemv_2304_r2[m] = make_gemv_pso(r->device, r->lib, "mtp_bf16_gemv_r2", m, 2304, 4);
            if (!r->pso_gemv_2304_r2[m]) { delete r; return nullptr; }
        }
        r->pso_gemv_3072_m8 = make_gemv_pso(r->device, r->lib, "mtp_bf16_gemv", 8, 3072, 4);
        r->pso_gemv_9216_m8 = make_gemv_pso(r->device, r->lib, "mtp_bf16_gemv", 8, 9216, 4);
        r->pso_gu_silu_m8 = make_gemv_pso(r->device, r->lib, "mtp_bf16_gu_silu", 8, 2304, 4);
        r->pso_kv_ctx_write = make_pso(r->device, r->lib, "mtp_kv_ctx_norm_rope_write", nil);
        r->pso_qkv_pack = make_pso(r->device, r->lib, "mtp_qkv_norm_rope_pack", nil);
        r->pso_sdpa_single = make_pso(r->device, r->lib, "mtp_sdpa_single", nil);
        r->pso_sdpa_split_p1 = make_pso(r->device, r->lib, "mtp_sdpa_split_p1", nil);
        r->pso_sdpa_split_p2 = make_pso(r->device, r->lib, "mtp_sdpa_split_p2", nil);
        for (int m = 1; m <= 7; ++m) {
            MTLFunctionConstantValues* fcv = [MTLFunctionConstantValues new];
            [fcv setConstantValue:&m type:MTLDataTypeInt atIndex:0];
            r->pso_markov_ctx_p1[m] = make_pso(r->device, r->lib, "mtp_markov_context_p1", fcv);
            r->pso_markov_ctx_p2[m] = make_pso(r->device, r->lib, "mtp_markov_context_p2", fcv);
            if (!r->pso_markov_ctx_p1[m] || !r->pso_markov_ctx_p2[m]) { delete r; return nullptr; }
        }
        r->pso_rnn_step = make_pso(r->device, r->lib, "mtp_rnn_step", nil);
        r->pso_markov_p1 = make_pso(r->device, r->lib, "mtp_markov_argmax_p1", nil);
        r->pso_markov_p2 = make_pso(r->device, r->lib, "mtp_markov_argmax_p2", nil);

        if (!r->pso_dflare_mix || !r->pso_gemv_3072_m8 || !r->pso_gemv_9216_m8 ||
            !r->pso_gu_silu_m8 || !r->pso_kv_ctx_write || !r->pso_qkv_pack ||
            !r->pso_sdpa_single || !r->pso_sdpa_split_p1 || !r->pso_sdpa_split_p2 ||
            !r->pso_rnn_step || !r->pso_markov_p1 || !r->pso_markov_p2) {
            delete r;
            return nullptr;
        }

        r->dummy_buf = [r->device newBufferWithLength:256 options:MTLResourceStorageModeShared];
        std::memset([r->dummy_buf contents], 0, 256);

        auto wrapW = [&](const void* ptr, size_t bytes, const char* lbl) -> WrappedBuf {
            return wrap_ptr_in_map(r->device, r->page_size, r->dummy_buf, r->weight_buf_cache,
                                   reinterpret_cast<uintptr_t>(ptr), bytes, lbl);
        };

        r->has_embed_norm = (globals->embed_norm != nullptr && globals->embed_norm_bytes > 0);
        r->embed_tokens       = wrapW(globals->embed_tokens, globals->embed_tokens_bytes, "embed_tokens");
        r->embed_norm         = r->has_embed_norm ? wrapW(globals->embed_norm, globals->embed_norm_bytes, "embed_norm") : WrappedBuf{r->dummy_buf, 0};
        r->mask_embed         = wrapW(globals->mask_embed, globals->mask_embed_bytes, "mask_embed");
        r->final_norm         = wrapW(globals->final_norm, globals->final_norm_bytes, "final_norm");
        r->lm_head            = wrapW(globals->lm_head, globals->lm_head_bytes, "lm_head");
        r->fuse_alpha         = wrapW(globals->fuse_alpha, globals->fuse_alpha_bytes, "fuse_alpha");
        r->markov_dflare_norm = wrapW(globals->markov_dflare_norm, globals->markov_dflare_norm_bytes, "markov_dflare_norm");
        r->markov_w1          = wrapW(globals->markov_w1, globals->markov_w1_bytes, "markov_w1");
        r->rnn_w_ctx_t        = wrapW(globals->rnn_w_ctx_t, globals->rnn_w_ctx_t_bytes, "rnn_w_ctx_t");
        r->rnn_w_step_t       = wrapW(globals->rnn_w_step_t, globals->rnn_w_step_t_bytes, "rnn_w_step_t");
        r->rnn_b_f32          = wrapW(globals->rnn_b_f32, globals->rnn_b_f32_bytes, "rnn_b_f32");
        r->fold_weight        = wrapW(globals->fold_weight, globals->fold_weight_bytes, "fold_weight");
        r->fold_bias          = wrapW(globals->fold_bias, globals->fold_bias_bytes, "fold_bias");

        if (!r->embed_tokens.buf || !r->embed_norm.buf || !r->mask_embed.buf ||
            !r->final_norm.buf || !r->lm_head.buf || !r->fuse_alpha.buf ||
            !r->markov_dflare_norm.buf || !r->markov_w1.buf || !r->rnn_w_ctx_t.buf ||
            !r->rnn_w_step_t.buf || !r->rnn_b_f32.buf || !r->fold_weight.buf || !r->fold_bias.buf) {
            delete r;
            return nullptr;
        }

        r->layers.resize(static_cast<size_t>(cfg->num_layers));
        for (int i = 0; i < cfg->num_layers; ++i) {
            const MTPDraftLayerWeights& lw = layers[i];
            LayerMetalWeights& dst = r->layers[static_cast<size_t>(i)];
            dst.is_local = lw.is_local;
            dst.window = lw.window;
            dst.dflare_norm              = wrapW(lw.dflare_norm, lw.dflare_norm_bytes, "dflare_norm");
            dst.input_layernorm          = wrapW(lw.input_layernorm, lw.input_layernorm_bytes, "input_layernorm");
            dst.post_attention_layernorm = wrapW(lw.post_attention_layernorm, lw.post_attention_layernorm_bytes, "post_attn_ln");
            dst.kv_ctx_w                 = wrapW(lw.kv_ctx_w, lw.kv_ctx_w_bytes, "kv_ctx_w");
            dst.k_norm_ctx               = wrapW(lw.k_norm_ctx, lw.k_norm_ctx_bytes, "k_norm_ctx");
            dst.qkv_w                    = wrapW(lw.qkv_w, lw.qkv_w_bytes, "qkv_w");
            dst.q_norm                   = wrapW(lw.q_norm, lw.q_norm_bytes, "q_norm");
            dst.k_norm                   = wrapW(lw.k_norm, lw.k_norm_bytes, "k_norm");
            dst.o_proj_w                 = wrapW(lw.o_proj_w, lw.o_proj_w_bytes, "o_proj_w");
            dst.gu_w                     = wrapW(lw.gu_w, lw.gu_w_bytes, "gu_w");
            dst.down_proj_w              = wrapW(lw.down_proj_w, lw.down_proj_w_bytes, "down_proj_w");

            if (!dst.dflare_norm.buf || !dst.input_layernorm.buf || !dst.post_attention_layernorm.buf ||
                !dst.kv_ctx_w.buf || !dst.k_norm_ctx.buf || !dst.qkv_w.buf || !dst.q_norm.buf ||
                !dst.k_norm.buf || !dst.o_proj_w.buf || !dst.gu_w.buf || !dst.down_proj_w.buf) {
                delete r;
                return nullptr;
            }
        }

        auto allocShared = [&](size_t bytes) -> id<MTLBuffer> {
            return [r->device newBufferWithLength:align_up(bytes, 256) options:MTLResourceStorageModeShared];
        };
        auto allocPrivate = [&](size_t bytes) -> id<MTLBuffer> {
            return [r->device newBufferWithLength:align_up(bytes, 256) options:MTLResourceStorageModePrivate];
        };

        const size_t D = static_cast<size_t>(cfg->hidden_size);
        const size_t B = static_cast<size_t>(cfg->block_size);
        const size_t H = static_cast<size_t>(cfg->num_heads);
        const size_t KVH = static_cast<size_t>(cfg->num_kv_heads);
        const size_t HD = static_cast<size_t>(cfg->head_dim);
        const size_t INTER = static_cast<size_t>(cfg->inter_size);
        const size_t RANK = static_cast<size_t>(cfg->markov_rank);
        const size_t V_UNPAD = static_cast<size_t>(cfg->unpadded_vocab_size);
        const size_t MAX_K = B - 1;
        const size_t n_markov_tg = (V_UNPAD + 63) / 64;

        r->hidden_buf      = allocShared(B * static_cast<size_t>(cfg->n_target_layers) * D * sizeof(uint16_t));
        r->x_buf           = allocShared(B * D * sizeof(uint16_t));
        r->residual_buf    = allocShared(B * D * sizeof(uint16_t));
        r->norm_buf        = allocShared(B * D * sizeof(uint16_t));
        r->ctx_buf         = allocShared(B * D * sizeof(uint16_t));
        r->kv_ctx_buf      = allocShared(B * 2 * KVH * HD * sizeof(uint16_t));
        r->qkv_buf         = allocShared(B * (H + 2 * KVH) * HD * sizeof(uint16_t));
        r->q_buf           = allocShared(H * B * HD * sizeof(uint16_t));
        r->bk_buf          = allocShared(KVH * B * HD * sizeof(uint16_t));
        r->bv_buf          = allocShared(KVH * B * HD * sizeof(uint16_t));
        r->attn_out_buf    = allocShared(B * H * HD * sizeof(uint16_t));
        r->sdpa_part_o     = allocPrivate(static_cast<size_t>(kMaxSdpaSplits) * H * B * HD * sizeof(float));
        r->sdpa_part_ml    = allocPrivate(static_cast<size_t>(kMaxSdpaSplits) * H * B * 2 * sizeof(float));
        r->mlp_mid_buf     = allocShared(B * INTER * sizeof(uint16_t));
        r->x_out_buf       = allocShared(MAX_K * D * sizeof(uint16_t));
        r->base_logits_buf = allocShared(MAX_K * V_UNPAD * sizeof(uint16_t));
        r->f_rnn_buf       = allocShared(D * sizeof(uint16_t));
        r->markov_ctx_part_x = allocPrivate(8 * MAX_K * 3 * RANK * sizeof(float));
        r->markov_ctx_part_f = allocPrivate(8 * 3 * RANK * sizeof(float));
        r->rnn_ctx_buf     = allocShared(MAX_K * 3 * RANK * sizeof(float));
        r->rnn_state_buf[0]= allocShared(RANK * sizeof(uint16_t));
        r->rnn_state_buf[1]= allocShared(RANK * sizeof(uint16_t));
        r->rnn_latent_buf  = allocShared(RANK * sizeof(uint16_t));
        r->markov_part_val = allocPrivate(n_markov_tg * sizeof(float));
        r->markov_part_idx = allocPrivate(n_markov_tg * sizeof(uint32_t));
        r->prev_tok_buf    = allocShared(64);
        r->out_tokens_buf  = allocShared(B * sizeof(int32_t));

        r->inv_freq_buf = allocShared((HD / 2) * sizeof(float));
        float* invf = reinterpret_cast<float*>([r->inv_freq_buf contents]);
        const float log2_base = std::log2(cfg->rope_theta);
        const float half_hd = static_cast<float>(HD / 2);
        for (size_t d = 0; d < HD / 2; ++d) {
            invf[d] = std::exp2(-log2_base * static_cast<float>(d) / half_hd);
        }

        return r;
    }
}

void mtp_draft_destroy(MTPDraftRunner* runner) {
    @autoreleasepool {
        delete runner;
    }
}

void mtp_draft_clear_kv_cache_map(MTPDraftRunner* runner) {
    if (!runner) return;
    std::lock_guard<std::mutex> lock(runner->kv_mu);
    runner->kv_buf_cache.clear();
}

const char* mtp_draft_device_name(MTPDraftRunner* runner) {
    if (!runner) return "";
    return runner->device_name.c_str();
}

int mtp_draft_greedy_step(
    MTPDraftRunner* r,
    int32_t anchor_token,
    uintptr_t hidden_ptr,
    size_t hidden_nbytes,
    int32_t n_ctx,
    int32_t num_draft,
    MTPKVCacheDesc* kv_caches,
    int32_t* out_tokens,
    MTPStepStats* out_stats)
{
    @autoreleasepool {
        if (!r || !kv_caches || !out_tokens || hidden_ptr == 0) return -1;
        if (n_ctx < 1 || n_ctx > r->cfg.block_size) return -2;
        if (num_draft < 1 || num_draft >= r->cfg.block_size) return -3;
        if (anchor_token < 0 || anchor_token >= r->cfg.vocab_size) return -4;

        const double t_wall0 = now_ms();

        const size_t need_hidden = static_cast<size_t>(n_ctx) * static_cast<size_t>(r->cfg.n_target_layers)
                                 * static_cast<size_t>(r->cfg.hidden_size) * sizeof(uint16_t);
        if (hidden_nbytes < need_hidden) return -5;

        std::memcpy([r->hidden_buf contents], reinterpret_cast<const void*>(hidden_ptr), need_hidden);

        WrappedBuf kv_k[3], kv_v[3];
        int32_t k_stride_h[3], k_stride_s[3], v_stride_h[3], v_stride_s[3];
        int32_t offset_before[3], offset_after[3], ring_start[3], valid_ctx[3];

        for (int i = 0; i < r->cfg.num_layers; ++i) {
            const MTPKVCacheDesc& kd = kv_caches[i];
            if (kd.k_ptr == 0 || kd.v_ptr == 0 || kd.cache_cap <= 0) return -10 - i;
            if (kd.k_ptr == kd.v_ptr) {
                std::fprintf(stderr, "[mtp_draft] Invalid KV cache layer %d: k_ptr == v_ptr (0x%lx)\n",
                             i, static_cast<unsigned long>(kd.k_ptr));
                return -38 - i;
            }
            offset_before[i] = kd.offset;
            offset_after[i] = kd.offset + n_ctx;
            if (kd.is_rotating != 0) {
                const int32_t W = (kd.window > 0) ? kd.window : r->layers[i].window;
                if (offset_after[i] <= W) {
                    if (kd.cache_cap < offset_after[i]) return -20 - i;
                    ring_start[i] = (kd.ring_idx >= 0) ? (kd.ring_idx % W) : offset_before[i];
                    valid_ctx[i] = offset_after[i];
                } else {
                    if (kd.cache_cap < W) return -20 - i;
                    ring_start[i] = (kd.ring_idx >= 0) ? (kd.ring_idx % W) : (offset_before[i] % W);
                    valid_ctx[i] = W;
                }
            } else {
                if (offset_after[i] > kd.cache_cap) return -30 - i;
                ring_start[i] = offset_before[i];
                valid_ctx[i] = offset_after[i];
            }
            k_stride_s[i] = (kd.stride_s > 0) ? kd.stride_s : r->cfg.head_dim;
            v_stride_s[i] = k_stride_s[i];
            k_stride_h[i] = (kd.stride_h > 0) ? kd.stride_h : (kd.cache_cap * k_stride_s[i]);
            v_stride_h[i] = k_stride_h[i];

            const size_t min_bytes = static_cast<size_t>(r->cfg.num_kv_heads) * static_cast<size_t>(k_stride_h[i]) * sizeof(uint16_t);
            if ((kd.k_nbytes > 0 && kd.k_nbytes < min_bytes) || (kd.v_nbytes > 0 && kd.v_nbytes < min_bytes)) {
                std::fprintf(stderr, "[mtp_draft] KV buffer too small for layer %d (stale/aliased ptr?): k_nbytes=%zu v_nbytes=%zu need>=%zu (cap=%d)\n",
                             i, kd.k_nbytes, kd.v_nbytes, min_bytes, kd.cache_cap);
                return -35 - i;
            }
            kv_k[i] = wrap_ptr_nocache(r->device, r->page_size, r->dummy_buf,
                                       kd.k_ptr, std::max(kd.k_nbytes, min_bytes), "kv_k");
            kv_v[i] = wrap_ptr_nocache(r->device, r->page_size, r->dummy_buf,
                                       kd.v_ptr, std::max(kd.v_nbytes, min_bytes), "kv_v");
            if (!kv_k[i].buf || !kv_v[i].buf) return -40 - i;
        }

        id<MTLCommandBuffer> cb = [r->queue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
        int32_t n_dispatches = 0;

        const int32_t D = r->cfg.hidden_size;
        const int32_t B = r->cfg.block_size;
        const int32_t KV_DIM = r->cfg.num_kv_heads * r->cfg.head_dim;
        const int32_t Q_DIM = r->cfg.num_heads * r->cfg.head_dim;
        const int32_t QKV_DIM = Q_DIM + 2 * KV_DIM;
        const int32_t INTER = r->cfg.inter_size;
        const int32_t V_UNPAD = r->cfg.unpadded_vocab_size;
        const float eps = r->cfg.rms_norm_eps;
        const float neg_log2_theta_over_half = -std::log2(r->cfg.rope_theta) / static_cast<float>(r->cfg.head_dim / 2);
        const float sdpa_scale2 = (1.0f / std::sqrt(static_cast<float>(r->cfg.head_dim))) * 1.4426950408889634f;

        {
            HostInitXPrm prm{anchor_token, D, B, r->cfg.markov_rank, r->cfg.embed_norm_eps};
            [enc setComputePipelineState:r->pso_init_x[r->has_embed_norm ? 1 : 0]];
            [enc setBuffer:r->embed_tokens.buf offset:r->embed_tokens.offset atIndex:0];
            [enc setBuffer:r->embed_norm.buf offset:r->embed_norm.offset atIndex:1];
            [enc setBuffer:r->mask_embed.buf offset:r->mask_embed.offset atIndex:2];
            [enc setBuffer:r->x_buf offset:0 atIndex:3];
            [enc setBuffer:r->rnn_state_buf[0] offset:0 atIndex:4];
            [enc setBuffer:r->prev_tok_buf offset:0 atIndex:5];
            [enc setBytes:&prm length:sizeof(prm) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            ++n_dispatches;
        }

        for (int i = 0; i < r->cfg.num_layers; ++i) {
            const LayerMetalWeights& lw = r->layers[static_cast<size_t>(i)];
            const int32_t W = (kv_caches[i].is_rotating != 0)
                ? ((kv_caches[i].window > 0) ? kv_caches[i].window : lw.window)
                : 0;

            {
                HostMixPrm prm{D, r->cfg.n_target_layers, 0, i, eps};
                [enc setComputePipelineState:r->pso_dflare_mix];
                [enc setBuffer:r->hidden_buf offset:0 atIndex:0];
                [enc setBuffer:r->fuse_alpha.buf offset:r->fuse_alpha.offset atIndex:1];
                [enc setBuffer:lw.dflare_norm.buf offset:lw.dflare_norm.offset atIndex:2];
                [enc setBuffer:r->ctx_buf offset:0 atIndex:3];
                [enc setBytes:&prm length:sizeof(prm) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(n_ctx), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_dispatches;
            }

            {
                HostGemvPrm prm{2 * KV_DIM};
                [enc setComputePipelineState:r->pso_gemv_2304[n_ctx]];
                [enc setBuffer:r->ctx_buf offset:0 atIndex:0];
                [enc setBuffer:lw.kv_ctx_w.buf offset:lw.kv_ctx_w.offset atIndex:1];
                [enc setBuffer:r->kv_ctx_buf offset:0 atIndex:2];
                [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((2 * KV_DIM + 3) / 4), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                ++n_dispatches;
            }

            {
                HostKvCtxWritePrm prm{
                    KV_DIM, offset_before[i], kv_caches[i].is_rotating, W, ring_start[i],
                    k_stride_h[i], k_stride_s[i], v_stride_h[i], v_stride_s[i], eps, neg_log2_theta_over_half
                };
                [enc setComputePipelineState:r->pso_kv_ctx_write];
                [enc setBuffer:r->kv_ctx_buf offset:0 atIndex:0];
                [enc setBuffer:lw.k_norm_ctx.buf offset:lw.k_norm_ctx.offset atIndex:1];
                [enc setBuffer:r->inv_freq_buf offset:0 atIndex:2];
                [enc setBuffer:kv_k[i].buf offset:kv_k[i].offset atIndex:3];
                [enc setBuffer:kv_v[i].buf offset:kv_v[i].offset atIndex:4];
                [enc setBytes:&prm length:sizeof(prm) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(n_ctx), 8, 1)
                    threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                ++n_dispatches;
            }

            {
                HostAddNormPrm prm{D, 0, 1, eps};
                [enc setComputePipelineState:r->pso_add_rmsnorm[i == 0 ? 0 : 1]];
                [enc setBuffer:r->x_buf offset:0 atIndex:0];
                [enc setBuffer:r->residual_buf offset:0 atIndex:1];
                [enc setBuffer:lw.input_layernorm.buf offset:lw.input_layernorm.offset atIndex:2];
                [enc setBuffer:r->residual_buf offset:0 atIndex:3];
                [enc setBuffer:r->norm_buf offset:0 atIndex:4];
                [enc setBytes:&prm length:sizeof(prm) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(B), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_dispatches;
            }

            {
                HostGemvPrm prm{QKV_DIM};
                [enc setComputePipelineState:r->pso_gemv_2304[8]];
                [enc setBuffer:r->norm_buf offset:0 atIndex:0];
                [enc setBuffer:lw.qkv_w.buf offset:lw.qkv_w.offset atIndex:1];
                [enc setBuffer:r->qkv_buf offset:0 atIndex:2];
                [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((QKV_DIM + 3) / 4), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                ++n_dispatches;
            }

            {
                HostQkvPackPrm prm{Q_DIM, KV_DIM, QKV_DIM, offset_after[i], eps, neg_log2_theta_over_half};
                [enc setComputePipelineState:r->pso_qkv_pack];
                [enc setBuffer:r->qkv_buf offset:0 atIndex:0];
                [enc setBuffer:lw.q_norm.buf offset:lw.q_norm.offset atIndex:1];
                [enc setBuffer:lw.k_norm.buf offset:lw.k_norm.offset atIndex:2];
                [enc setBuffer:r->inv_freq_buf offset:0 atIndex:3];
                [enc setBuffer:r->q_buf offset:0 atIndex:4];
                [enc setBuffer:r->bk_buf offset:0 atIndex:5];
                [enc setBuffer:r->bv_buf offset:0 atIndex:6];
                [enc setBytes:&prm length:sizeof(prm) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(B), 40, 1)
                    threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                ++n_dispatches;
            }

            {
                const int32_t total_k = valid_ctx[i] + B;
                if (valid_ctx[i] < 256) {
                    HostSdpaPrm prm{
                        valid_ctx[i], total_k,
                        k_stride_h[i], k_stride_s[i], v_stride_h[i], v_stride_s[i],
                        1, total_k, sdpa_scale2
                    };
                    [enc setComputePipelineState:r->pso_sdpa_single];
                    [enc setBuffer:r->q_buf offset:0 atIndex:0];
                    [enc setBuffer:kv_k[i].buf offset:kv_k[i].offset atIndex:1];
                    [enc setBuffer:kv_v[i].buf offset:kv_v[i].offset atIndex:2];
                    [enc setBuffer:r->bk_buf offset:0 atIndex:3];
                    [enc setBuffer:r->bv_buf offset:0 atIndex:4];
                    [enc setBuffer:r->attn_out_buf offset:0 atIndex:5];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:6];
                    [enc dispatchThreadgroups:MTLSizeMake(8, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    ++n_dispatches;
                } else {
                    int32_t n_splits = std::min<int32_t>(kMaxSdpaSplits, std::max<int32_t>(4, (total_k + 255) / 256));
                    int32_t chunk = ((total_k + n_splits - 1) / n_splits);
                    chunk = ((chunk + 7) / 8) * 8;
                    n_splits = (total_k + chunk - 1) / chunk;
                    HostSdpaPrm prm{
                        valid_ctx[i], total_k,
                        k_stride_h[i], k_stride_s[i], v_stride_h[i], v_stride_s[i],
                        n_splits, chunk, sdpa_scale2
                    };
                    [enc setComputePipelineState:r->pso_sdpa_split_p1];
                    [enc setBuffer:r->q_buf offset:0 atIndex:0];
                    [enc setBuffer:kv_k[i].buf offset:kv_k[i].offset atIndex:1];
                    [enc setBuffer:kv_v[i].buf offset:kv_v[i].offset atIndex:2];
                    [enc setBuffer:r->bk_buf offset:0 atIndex:3];
                    [enc setBuffer:r->bv_buf offset:0 atIndex:4];
                    [enc setBuffer:r->sdpa_part_o offset:0 atIndex:5];
                    [enc setBuffer:r->sdpa_part_ml offset:0 atIndex:6];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:7];
                    [enc dispatchThreadgroups:MTLSizeMake(8, static_cast<NSUInteger>(n_splits), 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    ++n_dispatches;

                    [enc setComputePipelineState:r->pso_sdpa_split_p2];
                    [enc setBuffer:r->sdpa_part_o offset:0 atIndex:0];
                    [enc setBuffer:r->sdpa_part_ml offset:0 atIndex:1];
                    [enc setBuffer:r->attn_out_buf offset:0 atIndex:2];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(B), 24, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    ++n_dispatches;
                }
            }

            {
                HostGemvPrm prm{D};
                [enc setComputePipelineState:r->pso_gemv_3072_m8];
                [enc setBuffer:r->attn_out_buf offset:0 atIndex:0];
                [enc setBuffer:lw.o_proj_w.buf offset:lw.o_proj_w.offset atIndex:1];
                [enc setBuffer:r->x_buf offset:0 atIndex:2];
                [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((D + 3) / 4), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                ++n_dispatches;
            }

            {
                HostAddNormPrm prm{D, 0, 1, eps};
                [enc setComputePipelineState:r->pso_add_rmsnorm[1]];
                [enc setBuffer:r->x_buf offset:0 atIndex:0];
                [enc setBuffer:r->residual_buf offset:0 atIndex:1];
                [enc setBuffer:lw.post_attention_layernorm.buf offset:lw.post_attention_layernorm.offset atIndex:2];
                [enc setBuffer:r->residual_buf offset:0 atIndex:3];
                [enc setBuffer:r->norm_buf offset:0 atIndex:4];
                [enc setBytes:&prm length:sizeof(prm) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(B), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_dispatches;
            }

            {
                HostGemvPrm prm{INTER};
                [enc setComputePipelineState:r->pso_gu_silu_m8];
                [enc setBuffer:r->norm_buf offset:0 atIndex:0];
                [enc setBuffer:lw.gu_w.buf offset:lw.gu_w.offset atIndex:1];
                [enc setBuffer:r->mlp_mid_buf offset:0 atIndex:2];
                [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((INTER + 3) / 4), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                ++n_dispatches;
            }

            {
                HostGemvPrm prm{D};
                [enc setComputePipelineState:r->pso_gemv_9216_m8];
                [enc setBuffer:r->mlp_mid_buf offset:0 atIndex:0];
                [enc setBuffer:lw.down_proj_w.buf offset:lw.down_proj_w.offset atIndex:1];
                [enc setBuffer:r->x_buf offset:0 atIndex:2];
                [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((D + 3) / 4), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                ++n_dispatches;
            }
        }

        {
            HostAddNormPrm prm{D, 1, 0, eps};
            [enc setComputePipelineState:r->pso_add_rmsnorm[1]];
            [enc setBuffer:r->x_buf offset:0 atIndex:0];
            [enc setBuffer:r->residual_buf offset:0 atIndex:1];
            [enc setBuffer:r->final_norm.buf offset:r->final_norm.offset atIndex:2];
            [enc setBuffer:r->dummy_buf offset:0 atIndex:3];
            [enc setBuffer:r->x_out_buf offset:0 atIndex:4];
            [enc setBytes:&prm length:sizeof(prm) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(num_draft), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            ++n_dispatches;
        }

        {
            HostGemvPrm prm{V_UNPAD};
            [enc setComputePipelineState:r->pso_gemv_2304_r2[num_draft]];
            [enc setBuffer:r->x_out_buf offset:0 atIndex:0];
            [enc setBuffer:r->lm_head.buf offset:r->lm_head.offset atIndex:1];
            [enc setBuffer:r->base_logits_buf offset:0 atIndex:2];
            [enc setBytes:&prm length:sizeof(prm) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((V_UNPAD + 7) / 8), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            ++n_dispatches;
        }

        {
            HostMixPrm prm{D, r->cfg.n_target_layers, n_ctx - 1, r->cfg.num_layers, eps};
            [enc setComputePipelineState:r->pso_dflare_mix];
            [enc setBuffer:r->hidden_buf offset:0 atIndex:0];
            [enc setBuffer:r->fuse_alpha.buf offset:r->fuse_alpha.offset atIndex:1];
            [enc setBuffer:r->markov_dflare_norm.buf offset:r->markov_dflare_norm.offset atIndex:2];
            [enc setBuffer:r->f_rnn_buf offset:0 atIndex:3];
            [enc setBytes:&prm length:sizeof(prm) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            ++n_dispatches;
        }

        {
            const int32_t out_dim = 3 * r->cfg.markov_rank;
            HostMarkovCtxPrm prm{D, out_dim};
            [enc setComputePipelineState:r->pso_markov_ctx_p1[num_draft]];
            [enc setBuffer:r->x_out_buf offset:0 atIndex:0];
            [enc setBuffer:r->f_rnn_buf offset:0 atIndex:1];
            [enc setBuffer:r->rnn_w_ctx_t.buf offset:r->rnn_w_ctx_t.offset atIndex:2];
            [enc setBuffer:r->markov_ctx_part_x offset:0 atIndex:3];
            [enc setBuffer:r->markov_ctx_part_f offset:0 atIndex:4];
            [enc setBytes:&prm length:sizeof(prm) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((out_dim + 31) / 32), 16, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            ++n_dispatches;

            [enc setComputePipelineState:r->pso_markov_ctx_p2[num_draft]];
            [enc setBuffer:r->markov_ctx_part_x offset:0 atIndex:0];
            [enc setBuffer:r->markov_ctx_part_f offset:0 atIndex:1];
            [enc setBuffer:r->rnn_b_f32.buf offset:r->rnn_b_f32.offset atIndex:2];
            [enc setBuffer:r->rnn_ctx_buf offset:0 atIndex:3];
            [enc setBytes:&prm length:sizeof(prm) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((out_dim + 31) / 32), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            ++n_dispatches;
        }

        const int32_t n_markov_tg = (V_UNPAD + 63) / 64;
        for (int s = 0; s < num_draft; ++s) {
            {
                HostRnnStepPrm prm{s, r->cfg.markov_rank};
                [enc setComputePipelineState:r->pso_rnn_step];
                [enc setBuffer:r->rnn_state_buf[s & 1] offset:0 atIndex:0];
                [enc setBuffer:r->prev_tok_buf offset:0 atIndex:1];
                [enc setBuffer:r->markov_w1.buf offset:r->markov_w1.offset atIndex:2];
                [enc setBuffer:r->rnn_w_step_t.buf offset:r->rnn_w_step_t.offset atIndex:3];
                [enc setBuffer:r->rnn_ctx_buf offset:0 atIndex:4];
                [enc setBuffer:r->rnn_state_buf[(s & 1) ^ 1] offset:0 atIndex:5];
                [enc setBuffer:r->rnn_latent_buf offset:0 atIndex:6];
                [enc setBytes:&prm length:sizeof(prm) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(8, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                ++n_dispatches;
            }
            {
                HostMarkovArgmaxPrm prm{V_UNPAD, s, n_markov_tg};
                [enc setComputePipelineState:r->pso_markov_p1];
                [enc setBuffer:r->rnn_latent_buf offset:0 atIndex:0];
                [enc setBuffer:r->fold_weight.buf offset:r->fold_weight.offset atIndex:1];
                [enc setBuffer:r->fold_bias.buf offset:r->fold_bias.offset atIndex:2];
                [enc setBuffer:r->base_logits_buf offset:0 atIndex:3];
                [enc setBuffer:r->markov_part_val offset:0 atIndex:4];
                [enc setBuffer:r->markov_part_idx offset:0 atIndex:5];
                [enc setBytes:&prm length:sizeof(prm) atIndex:6];
                [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(n_markov_tg), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_dispatches;
            }
            {
                HostMarkovArgmaxPrm prm{V_UNPAD, s, n_markov_tg};
                [enc setComputePipelineState:r->pso_markov_p2];
                [enc setBuffer:r->markov_part_val offset:0 atIndex:0];
                [enc setBuffer:r->markov_part_idx offset:0 atIndex:1];
                [enc setBuffer:r->out_tokens_buf offset:0 atIndex:2];
                [enc setBuffer:r->prev_tok_buf offset:0 atIndex:3];
                [enc setBytes:&prm length:sizeof(prm) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_dispatches;
            }
        }

        [enc endEncoding];
        const double t_enc1 = now_ms();

        [cb commit];
        [cb waitUntilCompleted];
        const double t_wall1 = now_ms();

        if ([cb status] != MTLCommandBufferStatusCompleted) {
            NSError* cberr = [cb error];
            std::fprintf(stderr, "[mtp_draft] CommandBuffer failed: %s\n",
                         cberr ? [[cberr localizedDescription] UTF8String] : "unknown");
            return -50;
        }

        const int32_t* gpu_toks = reinterpret_cast<const int32_t*>([r->out_tokens_buf contents]);
        for (int s = 0; s < num_draft; ++s) {
            out_tokens[s] = gpu_toks[s];
        }

        for (int i = 0; i < r->cfg.num_layers; ++i) {
            kv_caches[i].offset = offset_after[i];
            if (kv_caches[i].is_rotating != 0 && kv_caches[i].ring_idx >= 0) {
                const int32_t W = (kv_caches[i].window > 0) ? kv_caches[i].window : r->layers[i].window;
                const int32_t next_idx = (ring_start[i] + n_ctx) % W;
                kv_caches[i].ring_idx = (next_idx == 0 && offset_after[i] >= W) ? W : next_idx;
            }
        }

        if (out_stats) {
            out_stats->encode_ms = t_enc1 - t_wall0;
            out_stats->gpu_ms = ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;
            out_stats->wall_ms = t_wall1 - t_wall0;
            out_stats->n_dispatches = n_dispatches;
        }
        return 0;
    }
}

int mtp_draft_bench_layers_gemv(
    MTPDraftRunner* r,
    int32_t n_ctx,
    int32_t warmup,
    int32_t iters,
    double* out_median_gpu_ms,
    double* out_achieved_gbps)
{
    @autoreleasepool {
        if (!r || n_ctx < 1 || n_ctx > 8 || iters < 1) return -1;

        const int32_t D = r->cfg.hidden_size;
        const int32_t KV_DIM = r->cfg.num_kv_heads * r->cfg.head_dim;
        const int32_t QKV_DIM = r->cfg.num_heads * r->cfg.head_dim + 2 * KV_DIM;
        const int32_t INTER = r->cfg.inter_size;

        const double total_bytes = 3.0 * (
            static_cast<double>(2 * KV_DIM) * D +
            static_cast<double>(QKV_DIM) * D +
            static_cast<double>(D) * (r->cfg.num_heads * r->cfg.head_dim) +
            static_cast<double>(2 * INTER) * D +
            static_cast<double>(D) * INTER
        ) * 2.0;

        std::vector<double> gpu_ms;
        gpu_ms.reserve(static_cast<size_t>(iters));

        for (int it = -warmup; it < iters; ++it) {
            id<MTLCommandBuffer> cb = [r->queue commandBufferWithUnretainedReferences];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];

            for (int i = 0; i < r->cfg.num_layers; ++i) {
                const LayerMetalWeights& lw = r->layers[static_cast<size_t>(i)];
                {
                    HostGemvPrm prm{2 * KV_DIM};
                    [enc setComputePipelineState:r->pso_gemv_2304[n_ctx]];
                    [enc setBuffer:r->ctx_buf offset:0 atIndex:0];
                    [enc setBuffer:lw.kv_ctx_w.buf offset:lw.kv_ctx_w.offset atIndex:1];
                    [enc setBuffer:r->kv_ctx_buf offset:0 atIndex:2];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((2 * KV_DIM + 3) / 4), 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                }
                {
                    HostGemvPrm prm{QKV_DIM};
                    [enc setComputePipelineState:r->pso_gemv_2304[8]];
                    [enc setBuffer:r->norm_buf offset:0 atIndex:0];
                    [enc setBuffer:lw.qkv_w.buf offset:lw.qkv_w.offset atIndex:1];
                    [enc setBuffer:r->qkv_buf offset:0 atIndex:2];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((QKV_DIM + 3) / 4), 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                }
                {
                    HostGemvPrm prm{D};
                    [enc setComputePipelineState:r->pso_gemv_3072_m8];
                    [enc setBuffer:r->attn_out_buf offset:0 atIndex:0];
                    [enc setBuffer:lw.o_proj_w.buf offset:lw.o_proj_w.offset atIndex:1];
                    [enc setBuffer:r->x_buf offset:0 atIndex:2];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((D + 3) / 4), 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                }
                {
                    HostGemvPrm prm{INTER};
                    [enc setComputePipelineState:r->pso_gu_silu_m8];
                    [enc setBuffer:r->norm_buf offset:0 atIndex:0];
                    [enc setBuffer:lw.gu_w.buf offset:lw.gu_w.offset atIndex:1];
                    [enc setBuffer:r->mlp_mid_buf offset:0 atIndex:2];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((INTER + 3) / 4), 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                }
                {
                    HostGemvPrm prm{D};
                    [enc setComputePipelineState:r->pso_gemv_9216_m8];
                    [enc setBuffer:r->mlp_mid_buf offset:0 atIndex:0];
                    [enc setBuffer:lw.down_proj_w.buf offset:lw.down_proj_w.offset atIndex:1];
                    [enc setBuffer:r->x_buf offset:0 atIndex:2];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((D + 3) / 4), 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                }
            }

            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if ([cb status] != MTLCommandBufferStatusCompleted) return -2;

            if (it >= 0) {
                gpu_ms.push_back(([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0);
            }
        }

        std::sort(gpu_ms.begin(), gpu_ms.end());
        const double med = gpu_ms[gpu_ms.size() / 2];
        if (out_median_gpu_ms) *out_median_gpu_ms = med;
        if (out_achieved_gbps) *out_achieved_gbps = (total_bytes / 1e9) / (med / 1e3);
        return 0;
    }
}

int mtp_draft_profile_stages(
    MTPDraftRunner* r,
    int32_t anchor_token,
    uintptr_t hidden_ptr,
    size_t hidden_nbytes,
    int32_t n_ctx,
    int32_t num_draft,
    const MTPKVCacheDesc* kv_caches,
    int32_t warmup,
    int32_t iters,
    MTPStageBreakdown* out_bd)
{
    @autoreleasepool {
        if (!r || !kv_caches || !out_bd || iters < 1) return -1;
        auto run_cb = [&](auto&& fn) -> double {
            std::vector<double> ts;
            ts.reserve(static_cast<size_t>(iters));
            for (int it = -warmup; it < iters; ++it) {
                id<MTLCommandBuffer> cb = [r->queue commandBufferWithUnretainedReferences];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
                fn(enc);
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if (it >= 0) ts.push_back(([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0);
            }
            std::sort(ts.begin(), ts.end());
            return ts[ts.size() / 2];
        };

        const int32_t D = r->cfg.hidden_size;
        const int32_t B = r->cfg.block_size;
        const int32_t V_UNPAD = r->cfg.unpadded_vocab_size;
        const int32_t n_markov_tg = (V_UNPAD + 63) / 64;
        const float eps = r->cfg.rms_norm_eps;
        const float sdpa_scale2 = (1.0f / std::sqrt(static_cast<float>(r->cfg.head_dim))) * 1.4426950408889634f;

        MTPKVCacheDesc kv_copy[3];
        int32_t dummy_toks[8];
        std::memcpy(kv_copy, kv_caches, sizeof(MTPKVCacheDesc) * 3);
        mtp_draft_greedy_step(r, anchor_token, hidden_ptr, hidden_nbytes, n_ctx, num_draft, kv_copy, dummy_toks, nullptr);

        WrappedBuf kv_k[3], kv_v[3];
        int32_t k_sh[3], k_ss[3], valid_ctx[3];
        for (int i = 0; i < 3; ++i) {
            k_ss[i] = (kv_caches[i].stride_s > 0) ? kv_caches[i].stride_s : 128;
            k_sh[i] = (kv_caches[i].stride_h > 0) ? kv_caches[i].stride_h : (kv_caches[i].cache_cap * k_ss[i]);
            int32_t off_a = kv_caches[i].offset + n_ctx;
            valid_ctx[i] = kv_caches[i].is_rotating ? std::min(off_a, kv_caches[i].window) : off_a;
            kv_k[i] = wrap_ptr_nocache(r->device, r->page_size, r->dummy_buf, kv_caches[i].k_ptr, kv_caches[i].k_nbytes, "pk");
            kv_v[i] = wrap_ptr_nocache(r->device, r->page_size, r->dummy_buf, kv_caches[i].v_ptr, kv_caches[i].v_nbytes, "pv");
        }

        out_bd->sdpa_only_ms = run_cb([&](id<MTLComputeCommandEncoder> enc) {
            for (int i = 0; i < 3; ++i) {
                int32_t total_k = valid_ctx[i] + B;
                if (valid_ctx[i] < 256) {
                    HostSdpaPrm prm{valid_ctx[i], total_k, k_sh[i], k_ss[i], k_sh[i], k_ss[i], 1, total_k, sdpa_scale2};
                    [enc setComputePipelineState:r->pso_sdpa_single];
                    [enc setBuffer:r->q_buf offset:0 atIndex:0];
                    [enc setBuffer:kv_k[i].buf offset:kv_k[i].offset atIndex:1];
                    [enc setBuffer:kv_v[i].buf offset:kv_v[i].offset atIndex:2];
                    [enc setBuffer:r->bk_buf offset:0 atIndex:3];
                    [enc setBuffer:r->bv_buf offset:0 atIndex:4];
                    [enc setBuffer:r->attn_out_buf offset:0 atIndex:5];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:6];
                    [enc dispatchThreadgroups:MTLSizeMake(8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                } else {
                    int32_t ns = std::min<int32_t>(kMaxSdpaSplits, std::max<int32_t>(4, (total_k + 255) / 256));
                    int32_t ch = (((total_k + ns - 1) / ns + 7) / 8) * 8;
                    ns = (total_k + ch - 1) / ch;
                    HostSdpaPrm prm{valid_ctx[i], total_k, k_sh[i], k_ss[i], k_sh[i], k_ss[i], ns, ch, sdpa_scale2};
                    [enc setComputePipelineState:r->pso_sdpa_split_p1];
                    [enc setBuffer:r->q_buf offset:0 atIndex:0];
                    [enc setBuffer:kv_k[i].buf offset:kv_k[i].offset atIndex:1];
                    [enc setBuffer:kv_v[i].buf offset:kv_v[i].offset atIndex:2];
                    [enc setBuffer:r->bk_buf offset:0 atIndex:3];
                    [enc setBuffer:r->bv_buf offset:0 atIndex:4];
                    [enc setBuffer:r->sdpa_part_o offset:0 atIndex:5];
                    [enc setBuffer:r->sdpa_part_ml offset:0 atIndex:6];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:7];
                    [enc dispatchThreadgroups:MTLSizeMake(8, static_cast<NSUInteger>(ns), 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc setComputePipelineState:r->pso_sdpa_split_p2];
                    [enc setBuffer:r->sdpa_part_o offset:0 atIndex:0];
                    [enc setBuffer:r->sdpa_part_ml offset:0 atIndex:1];
                    [enc setBuffer:r->attn_out_buf offset:0 atIndex:2];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(B), 24, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                }
            }
        });

        out_bd->lm_head_ms = run_cb([&](id<MTLComputeCommandEncoder> enc) {
            HostGemvPrm prm{V_UNPAD};
            [enc setComputePipelineState:r->pso_gemv_2304_r2[num_draft]];
            [enc setBuffer:r->x_out_buf offset:0 atIndex:0];
            [enc setBuffer:r->lm_head.buf offset:r->lm_head.offset atIndex:1];
            [enc setBuffer:r->base_logits_buf offset:0 atIndex:2];
            [enc setBytes:&prm length:sizeof(prm) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((V_UNPAD + 7) / 8), 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        });

        out_bd->markov_ctx_ms = run_cb([&](id<MTLComputeCommandEncoder> enc) {
            HostMixPrm mprm{D, r->cfg.n_target_layers, n_ctx - 1, r->cfg.num_layers, eps};
            [enc setComputePipelineState:r->pso_dflare_mix];
            [enc setBuffer:r->hidden_buf offset:0 atIndex:0];
            [enc setBuffer:r->fuse_alpha.buf offset:r->fuse_alpha.offset atIndex:1];
            [enc setBuffer:r->markov_dflare_norm.buf offset:r->markov_dflare_norm.offset atIndex:2];
            [enc setBuffer:r->f_rnn_buf offset:0 atIndex:3];
            [enc setBytes:&mprm length:sizeof(mprm) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            HostMarkovCtxPrm cprm{D, 3 * r->cfg.markov_rank};
            [enc setComputePipelineState:r->pso_markov_ctx_p1[num_draft]];
            [enc setBuffer:r->x_out_buf offset:0 atIndex:0];
            [enc setBuffer:r->f_rnn_buf offset:0 atIndex:1];
            [enc setBuffer:r->rnn_w_ctx_t.buf offset:r->rnn_w_ctx_t.offset atIndex:2];
            [enc setBuffer:r->markov_ctx_part_x offset:0 atIndex:3];
            [enc setBuffer:r->markov_ctx_part_f offset:0 atIndex:4];
            [enc setBytes:&cprm length:sizeof(cprm) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(24, 16, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

            [enc setComputePipelineState:r->pso_markov_ctx_p2[num_draft]];
            [enc setBuffer:r->markov_ctx_part_x offset:0 atIndex:0];
            [enc setBuffer:r->markov_ctx_part_f offset:0 atIndex:1];
            [enc setBuffer:r->rnn_b_f32.buf offset:r->rnn_b_f32.offset atIndex:2];
            [enc setBuffer:r->rnn_ctx_buf offset:0 atIndex:3];
            [enc setBytes:&cprm length:sizeof(cprm) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(24, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        });

        out_bd->rnn_steps_ms = run_cb([&](id<MTLComputeCommandEncoder> enc) {
            for (int s = 0; s < num_draft; ++s) {
                HostRnnStepPrm prm{s, r->cfg.markov_rank};
                [enc setComputePipelineState:r->pso_rnn_step];
                [enc setBuffer:r->rnn_state_buf[s & 1] offset:0 atIndex:0];
                [enc setBuffer:r->prev_tok_buf offset:0 atIndex:1];
                [enc setBuffer:r->markov_w1.buf offset:r->markov_w1.offset atIndex:2];
                [enc setBuffer:r->rnn_w_step_t.buf offset:r->rnn_w_step_t.offset atIndex:3];
                [enc setBuffer:r->rnn_ctx_buf offset:0 atIndex:4];
                [enc setBuffer:r->rnn_state_buf[(s & 1) ^ 1] offset:0 atIndex:5];
                [enc setBuffer:r->rnn_latent_buf offset:0 atIndex:6];
                [enc setBytes:&prm length:sizeof(prm) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(8, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }
        });

        out_bd->argmax_ms = run_cb([&](id<MTLComputeCommandEncoder> enc) {
            for (int s = 0; s < num_draft; ++s) {
                HostMarkovArgmaxPrm prm{V_UNPAD, s, n_markov_tg};
                [enc setComputePipelineState:r->pso_markov_p1];
                [enc setBuffer:r->rnn_latent_buf offset:0 atIndex:0];
                [enc setBuffer:r->fold_weight.buf offset:r->fold_weight.offset atIndex:1];
                [enc setBuffer:r->fold_bias.buf offset:r->fold_bias.offset atIndex:2];
                [enc setBuffer:r->base_logits_buf offset:0 atIndex:3];
                [enc setBuffer:r->markov_part_val offset:0 atIndex:4];
                [enc setBuffer:r->markov_part_idx offset:0 atIndex:5];
                [enc setBytes:&prm length:sizeof(prm) atIndex:6];
                [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(n_markov_tg), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc setComputePipelineState:r->pso_markov_p2];
                [enc setBuffer:r->markov_part_val offset:0 atIndex:0];
                [enc setBuffer:r->markov_part_idx offset:0 atIndex:1];
                [enc setBuffer:r->out_tokens_buf offset:0 atIndex:2];
                [enc setBuffer:r->prev_tok_buf offset:0 atIndex:3];
                [enc setBytes:&prm length:sizeof(prm) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
        });

        double g_ms = 0.0, g_bw = 0.0;
        mtp_draft_bench_layers_gemv(r, n_ctx, warmup, iters, &g_ms, &g_bw);
        out_bd->layers_ms = g_ms + out_bd->sdpa_only_ms;
        return 0;
    }
}

int mtp_draft_debug_read(
    MTPDraftRunner* r,
    int32_t buf_id,
    void* dst_ptr,
    size_t dst_nbytes)
{
    if (!r || !dst_ptr) return -1;
    id<MTLBuffer> src = nil;
    switch (buf_id) {
        case 0:  src = r->x_out_buf; break;
        case 1:  src = r->base_logits_buf; break;
        case 2:  src = r->f_rnn_buf; break;
        case 3:  src = r->rnn_ctx_buf; break;
        case 4:  src = r->rnn_latent_buf; break;
        case 5:  src = r->attn_out_buf; break;
        case 6:  src = r->mlp_mid_buf; break;
        case 7:  src = r->x_buf; break;
        case 8:  src = r->residual_buf; break;
        case 9:  src = r->q_buf; break;
        case 10: src = r->bk_buf; break;
        case 11: src = r->bv_buf; break;
        case 12: src = r->ctx_buf; break;
        case 13: src = r->kv_ctx_buf; break;
        case 14: src = r->norm_buf; break;
        case 15: src = r->qkv_buf; break;
        default: return -2;
    }
    if (!src || dst_nbytes > [src length]) return -3;
    std::memcpy(dst_ptr, [src contents], dst_nbytes);
    return 0;
}

int mtp_draft_debug_run_up_to_layer(
    MTPDraftRunner* r,
    int32_t anchor_token,
    uintptr_t hidden_ptr,
    size_t hidden_nbytes,
    int32_t n_ctx,
    MTPKVCacheDesc* kv_caches,
    int32_t max_layer)
{
    @autoreleasepool {
        if (!r || !kv_caches || hidden_ptr == 0) return -1;
        const size_t need_hidden = static_cast<size_t>(n_ctx) * static_cast<size_t>(r->cfg.n_target_layers)
                                 * static_cast<size_t>(r->cfg.hidden_size) * sizeof(uint16_t);
        if (hidden_nbytes < need_hidden) return -5;
        std::memcpy([r->hidden_buf contents], reinterpret_cast<const void*>(hidden_ptr), need_hidden);

        WrappedBuf kv_k[3], kv_v[3];
        int32_t k_stride_h[3], k_stride_s[3], v_stride_h[3], v_stride_s[3];
        int32_t offset_before[3], offset_after[3], ring_start[3], valid_ctx[3];
        for (int i = 0; i <= max_layer; ++i) {
            const MTPKVCacheDesc& kd = kv_caches[i];
            offset_before[i] = kd.offset;
            offset_after[i] = kd.offset + n_ctx;
            const int32_t W = (kd.window > 0) ? kd.window : r->layers[i].window;
            if (kd.is_rotating != 0) {
                ring_start[i] = (kd.ring_idx >= 0) ? (kd.ring_idx % W) : (offset_before[i] % W);
                valid_ctx[i] = std::min(offset_after[i], W);
            } else {
                ring_start[i] = offset_before[i];
                valid_ctx[i] = offset_after[i];
            }
            k_stride_s[i] = (kd.stride_s > 0) ? kd.stride_s : 128;
            v_stride_s[i] = k_stride_s[i];
            k_stride_h[i] = (kd.stride_h > 0) ? kd.stride_h : (kd.cache_cap * k_stride_s[i]);
            v_stride_h[i] = k_stride_h[i];
            const size_t min_bytes = 8ULL * static_cast<size_t>(k_stride_h[i]) * sizeof(uint16_t);
            kv_k[i] = wrap_ptr_nocache(r->device, r->page_size, r->dummy_buf, kd.k_ptr, std::max(kd.k_nbytes, min_bytes), "dk");
            kv_v[i] = wrap_ptr_nocache(r->device, r->page_size, r->dummy_buf, kd.v_ptr, std::max(kd.v_nbytes, min_bytes), "dv");
        }
        id<MTLCommandBuffer> cb = [r->queue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
        const int32_t D = r->cfg.hidden_size, B = r->cfg.block_size, KV_DIM = 1024, Q_DIM = 3072, QKV_DIM = 5120, INTER = 9216;
        const float eps = r->cfg.rms_norm_eps;
        const float neg_log2_theta_over_half = -std::log2(r->cfg.rope_theta) / 64.0f;
        const float sdpa_scale2 = (1.0f / std::sqrt(128.0f)) * 1.4426950408889634f;
        {
            HostInitXPrm prm{anchor_token, D, B, r->cfg.markov_rank, r->cfg.embed_norm_eps};
            [enc setComputePipelineState:r->pso_init_x[r->has_embed_norm ? 1 : 0]];
            [enc setBuffer:r->embed_tokens.buf offset:r->embed_tokens.offset atIndex:0];
            [enc setBuffer:r->embed_norm.buf offset:r->embed_norm.offset atIndex:1];
            [enc setBuffer:r->mask_embed.buf offset:r->mask_embed.offset atIndex:2];
            [enc setBuffer:r->x_buf offset:0 atIndex:3];
            [enc setBuffer:r->rnn_state_buf[0] offset:0 atIndex:4];
            [enc setBuffer:r->prev_tok_buf offset:0 atIndex:5];
            [enc setBytes:&prm length:sizeof(prm) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        for (int i = 0; i <= max_layer; ++i) {
            const LayerMetalWeights& lw = r->layers[static_cast<size_t>(i)];
            const int32_t W = (kv_caches[i].is_rotating != 0) ? ((kv_caches[i].window > 0) ? kv_caches[i].window : lw.window) : 0;
            {
                HostMixPrm prm{D, 4, 0, i, eps};
                [enc setComputePipelineState:r->pso_dflare_mix];
                [enc setBuffer:r->hidden_buf offset:0 atIndex:0];
                [enc setBuffer:r->fuse_alpha.buf offset:r->fuse_alpha.offset atIndex:1];
                [enc setBuffer:lw.dflare_norm.buf offset:lw.dflare_norm.offset atIndex:2];
                [enc setBuffer:r->ctx_buf offset:0 atIndex:3];
                [enc setBytes:&prm length:sizeof(prm) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(n_ctx), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            {
                HostGemvPrm prm{2 * KV_DIM};
                [enc setComputePipelineState:r->pso_gemv_2304[n_ctx]];
                [enc setBuffer:r->ctx_buf offset:0 atIndex:0];
                [enc setBuffer:lw.kv_ctx_w.buf offset:lw.kv_ctx_w.offset atIndex:1];
                [enc setBuffer:r->kv_ctx_buf offset:0 atIndex:2];
                [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(512, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
            {
                HostKvCtxWritePrm prm{KV_DIM, offset_before[i], kv_caches[i].is_rotating, W, ring_start[i], k_stride_h[i], k_stride_s[i], v_stride_h[i], v_stride_s[i], eps, neg_log2_theta_over_half};
                [enc setComputePipelineState:r->pso_kv_ctx_write];
                [enc setBuffer:r->kv_ctx_buf offset:0 atIndex:0];
                [enc setBuffer:lw.k_norm_ctx.buf offset:lw.k_norm_ctx.offset atIndex:1];
                [enc setBuffer:r->inv_freq_buf offset:0 atIndex:2];
                [enc setBuffer:kv_k[i].buf offset:kv_k[i].offset atIndex:3];
                [enc setBuffer:kv_v[i].buf offset:kv_v[i].offset atIndex:4];
                [enc setBytes:&prm length:sizeof(prm) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(n_ctx), 8, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            }
            {
                HostAddNormPrm prm{D, 0, 1, eps};
                [enc setComputePipelineState:r->pso_add_rmsnorm[i == 0 ? 0 : 1]];
                [enc setBuffer:r->x_buf offset:0 atIndex:0];
                [enc setBuffer:r->residual_buf offset:0 atIndex:1];
                [enc setBuffer:lw.input_layernorm.buf offset:lw.input_layernorm.offset atIndex:2];
                [enc setBuffer:r->residual_buf offset:0 atIndex:3];
                [enc setBuffer:r->norm_buf offset:0 atIndex:4];
                [enc setBytes:&prm length:sizeof(prm) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            {
                HostGemvPrm prm{QKV_DIM};
                [enc setComputePipelineState:r->pso_gemv_2304[8]];
                [enc setBuffer:r->norm_buf offset:0 atIndex:0];
                [enc setBuffer:lw.qkv_w.buf offset:lw.qkv_w.offset atIndex:1];
                [enc setBuffer:r->qkv_buf offset:0 atIndex:2];
                [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(1280, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
            {
                HostQkvPackPrm prm{Q_DIM, KV_DIM, QKV_DIM, offset_after[i], eps, neg_log2_theta_over_half};
                [enc setComputePipelineState:r->pso_qkv_pack];
                [enc setBuffer:r->qkv_buf offset:0 atIndex:0];
                [enc setBuffer:lw.q_norm.buf offset:lw.q_norm.offset atIndex:1];
                [enc setBuffer:lw.k_norm.buf offset:lw.k_norm.offset atIndex:2];
                [enc setBuffer:r->inv_freq_buf offset:0 atIndex:3];
                [enc setBuffer:r->q_buf offset:0 atIndex:4];
                [enc setBuffer:r->bk_buf offset:0 atIndex:5];
                [enc setBuffer:r->bv_buf offset:0 atIndex:6];
                [enc setBytes:&prm length:sizeof(prm) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(8, 40, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            }
            {
                const int32_t total_k = valid_ctx[i] + B;
                if (valid_ctx[i] < 256) {
                    HostSdpaPrm prm{valid_ctx[i], total_k, k_stride_h[i], k_stride_s[i], v_stride_h[i], v_stride_s[i], 1, total_k, sdpa_scale2};
                    [enc setComputePipelineState:r->pso_sdpa_single];
                    [enc setBuffer:r->q_buf offset:0 atIndex:0];
                    [enc setBuffer:kv_k[i].buf offset:kv_k[i].offset atIndex:1];
                    [enc setBuffer:kv_v[i].buf offset:kv_v[i].offset atIndex:2];
                    [enc setBuffer:r->bk_buf offset:0 atIndex:3];
                    [enc setBuffer:r->bv_buf offset:0 atIndex:4];
                    [enc setBuffer:r->attn_out_buf offset:0 atIndex:5];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:6];
                    [enc dispatchThreadgroups:MTLSizeMake(8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                } else {
                    int32_t ns = std::min<int32_t>(kMaxSdpaSplits, std::max<int32_t>(4, (total_k + 255) / 256));
                    int32_t ch = (((total_k + ns - 1) / ns + 7) / 8) * 8;
                    ns = (total_k + ch - 1) / ch;
                    HostSdpaPrm prm{valid_ctx[i], total_k, k_stride_h[i], k_stride_s[i], v_stride_h[i], v_stride_s[i], ns, ch, sdpa_scale2};
                    [enc setComputePipelineState:r->pso_sdpa_split_p1];
                    [enc setBuffer:r->q_buf offset:0 atIndex:0];
                    [enc setBuffer:kv_k[i].buf offset:kv_k[i].offset atIndex:1];
                    [enc setBuffer:kv_v[i].buf offset:kv_v[i].offset atIndex:2];
                    [enc setBuffer:r->bk_buf offset:0 atIndex:3];
                    [enc setBuffer:r->bv_buf offset:0 atIndex:4];
                    [enc setBuffer:r->sdpa_part_o offset:0 atIndex:5];
                    [enc setBuffer:r->sdpa_part_ml offset:0 atIndex:6];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:7];
                    [enc dispatchThreadgroups:MTLSizeMake(8, static_cast<NSUInteger>(ns), 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc setComputePipelineState:r->pso_sdpa_split_p2];
                    [enc setBuffer:r->sdpa_part_o offset:0 atIndex:0];
                    [enc setBuffer:r->sdpa_part_ml offset:0 atIndex:1];
                    [enc setBuffer:r->attn_out_buf offset:0 atIndex:2];
                    [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(8, 24, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                }
            }
            {
                HostGemvPrm prm{D};
                [enc setComputePipelineState:r->pso_gemv_3072_m8];
                [enc setBuffer:r->attn_out_buf offset:0 atIndex:0];
                [enc setBuffer:lw.o_proj_w.buf offset:lw.o_proj_w.offset atIndex:1];
                [enc setBuffer:r->x_buf offset:0 atIndex:2];
                [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(576, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
            {
                HostAddNormPrm prm{D, 0, 1, eps};
                [enc setComputePipelineState:r->pso_add_rmsnorm[1]];
                [enc setBuffer:r->x_buf offset:0 atIndex:0];
                [enc setBuffer:r->residual_buf offset:0 atIndex:1];
                [enc setBuffer:lw.post_attention_layernorm.buf offset:lw.post_attention_layernorm.offset atIndex:2];
                [enc setBuffer:r->residual_buf offset:0 atIndex:3];
                [enc setBuffer:r->norm_buf offset:0 atIndex:4];
                [enc setBytes:&prm length:sizeof(prm) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            {
                HostGemvPrm prm{INTER};
                [enc setComputePipelineState:r->pso_gu_silu_m8];
                [enc setBuffer:r->norm_buf offset:0 atIndex:0];
                [enc setBuffer:lw.gu_w.buf offset:lw.gu_w.offset atIndex:1];
                [enc setBuffer:r->mlp_mid_buf offset:0 atIndex:2];
                [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(2304, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
            {
                HostGemvPrm prm{D};
                [enc setComputePipelineState:r->pso_gemv_9216_m8];
                [enc setBuffer:r->mlp_mid_buf offset:0 atIndex:0];
                [enc setBuffer:lw.down_proj_w.buf offset:lw.down_proj_w.offset atIndex:1];
                [enc setBuffer:r->x_buf offset:0 atIndex:2];
                [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(576, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return 0;
    }
}

}
