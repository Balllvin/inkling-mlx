#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "flash_attn_nax.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mach/vm_page_size.h>
#include <mutex>
#include <unordered_map>
#include <vector>

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLLibrary> g_lib = nil;

static id<MTLComputePipelineState> g_pso_v1[2] = {nil, nil};
static id<MTLComputePipelineState> g_pso_v2[2] = {nil, nil};
static id<MTLComputePipelineState> g_pso_v3[2] = {nil, nil};
static id<MTLComputePipelineState> g_pso_v4[2] = {nil, nil};
static id<MTLComputePipelineState> g_pso_v5[2] = {nil, nil};
static id<MTLComputePipelineState> g_pso_v6 = nil;

static id<MTLBuffer> g_dummy_buf = nil;

struct CachedBuf {
  id<MTLBuffer> buf = nil;
  size_t span = 0;
  uint64_t lru = 0;
};

static std::mutex g_cache_mu;
static std::unordered_map<uintptr_t, CachedBuf> g_buf_cache;
static uint64_t g_lru_clock = 0;
static constexpr size_t kMaxCachedBufs = 128;

static double now_ms() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

struct WrappedBuffer {
  id<MTLBuffer> buf = nil;
  size_t offset = 0;
};

static WrappedBuffer wrap_ptr_nocopy_cached(uintptr_t addr, size_t nbytes) {
  if (addr == 0 || nbytes == 0) {
    return {g_dummy_buf, 0};
  }
  const size_t page_sz = vm_page_size;
  const uintptr_t base = addr & ~(page_sz - 1);
  const size_t offset = static_cast<size_t>(addr - base);
  const size_t span = ((offset + nbytes + page_sz - 1) / page_sz) * page_sz;

  id<MTLBuffer> buf = [g_device newBufferWithBytesNoCopy:reinterpret_cast<void*>(base)
                                                  length:span
                                                 options:MTLResourceStorageModeShared
                                             deallocator:nil];
  if (!buf) {
    std::fprintf(stderr, "[flash_attn_nax] newBufferWithBytesNoCopy failed: base=0x%lx span=%zu\n",
                 static_cast<unsigned long>(base), span);
    return {nil, 0};
  }
  return {buf, offset};
}

struct MetalFlashParams {
  int32_t B;
  int32_t HQ;
  int32_t HKV;
  int32_t G;
  int32_t L;
  int32_t S;
  int32_t R;
  int32_t window;
  float scale2;
  int32_t sinks_bf16;
  int32_t has_gate;
  int32_t gate_off;
  int32_t gate_stride;
  int32_t pad0;
  int64_t q_s0, q_s1, q_s2;
  int64_t k_s0, k_s1, k_s2;
  int64_t v_s0, v_s1, v_s2;
};

static const char* kFlashAttnMSL = R"METAL(
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

constant bool kAlignQK [[function_constant(0)]];

constant float LOG2E = 1.4426950408889634f;
constant float NEG_INF = -1e30f;

struct MetalFlashParams {
  int B;
  int HQ;
  int HKV;
  int G;
  int L;
  int S;
  int R;
  int window;
  float scale2;
  int sinks_bf16;
  int has_gate;
  int gate_off;
  int gate_stride;
  int pad0;
  long q_s0, q_s1, q_s2;
  long k_s0, k_s1, k_s2;
  long v_s0, v_s1, v_s2;
};

static inline short2 nax_lane_coord(ushort lane) {
  const short qid = short(lane >> 2);
  const short fm = ((qid & 4) | ((short(lane) >> 1) & 3));
  const short fn = ((qid & 2) | (short(lane) & 1)) * 4;
  return short2(fn, fm);
}

static inline void load_frag16x16_preoff(
    thread vec<bfloat, 8>& dst,
    const device bfloat* src,
    int row_stride) {
  vec<bfloat, 4> v0 = *(const device vec<bfloat, 4>*)src;
  vec<bfloat, 4> v1 = *(const device vec<bfloat, 4>*)(src + 8 * row_stride);
  dst[0] = v0[0]; dst[1] = v0[1]; dst[2] = v0[2]; dst[3] = v0[3];
  dst[4] = v1[0]; dst[5] = v1[1]; dst[6] = v1[2]; dst[7] = v1[3];
}

static inline void load_frag16x16_rows_preoff(
    thread vec<bfloat, 8>& dst,
    const device bfloat* src,
    int row_stride,
    int lim_rows_minus_fm) {
  if (0 < lim_rows_minus_fm) {
    vec<bfloat, 4> v0 = *(const device vec<bfloat, 4>*)src;
    dst[0] = v0[0]; dst[1] = v0[1]; dst[2] = v0[2]; dst[3] = v0[3];
  } else {
    dst[0] = bfloat(0.0f); dst[1] = bfloat(0.0f); dst[2] = bfloat(0.0f); dst[3] = bfloat(0.0f);
  }
  if (8 < lim_rows_minus_fm) {
    vec<bfloat, 4> v1 = *(const device vec<bfloat, 4>*)(src + 8 * row_stride);
    dst[4] = v1[0]; dst[5] = v1[1]; dst[6] = v1[2]; dst[7] = v1[3];
  } else {
    dst[4] = bfloat(0.0f); dst[5] = bfloat(0.0f); dst[6] = bfloat(0.0f); dst[7] = bfloat(0.0f);
  }
}

static inline void load_frag16x16(
    thread vec<bfloat, 8>& dst,
    const device bfloat* src,
    long row_stride,
    short fm,
    short fn) {
  vec<bfloat, 4> v0 = *(const device vec<bfloat, 4>*)(src + long(fm) * row_stride + fn);
  vec<bfloat, 4> v1 = *(const device vec<bfloat, 4>*)(src + long(fm + 8) * row_stride + fn);
  dst[0] = v0[0]; dst[1] = v0[1]; dst[2] = v0[2]; dst[3] = v0[3];
  dst[4] = v1[0]; dst[5] = v1[1]; dst[6] = v1[2]; dst[7] = v1[3];
}

static inline void load_frag16x16_rows(
    thread vec<bfloat, 8>& dst,
    const device bfloat* src,
    long row_stride,
    int lim_rows,
    short fm,
    short fn) {
  if (int(fm) < lim_rows) {
    vec<bfloat, 4> v0 = *(const device vec<bfloat, 4>*)(src + long(fm) * row_stride + fn);
    dst[0] = v0[0]; dst[1] = v0[1]; dst[2] = v0[2]; dst[3] = v0[3];
  } else {
    dst[0] = bfloat(0.0f); dst[1] = bfloat(0.0f); dst[2] = bfloat(0.0f); dst[3] = bfloat(0.0f);
  }
  if (int(fm + 8) < lim_rows) {
    vec<bfloat, 4> v1 = *(const device vec<bfloat, 4>*)(src + long(fm + 8) * row_stride + fn);
    dst[4] = v1[0]; dst[5] = v1[1]; dst[6] = v1[2]; dst[7] = v1[3];
  } else {
    dst[4] = bfloat(0.0f); dst[5] = bfloat(0.0f); dst[6] = bfloat(0.0f); dst[7] = bfloat(0.0f);
  }
}

static inline void load_rel_proj_frag16x16(
    thread vec<bfloat, 8>& dst,
    const device bfloat* rel_proj_t,
    int d_start,
    int R,
    short fm,
    short fn) {
  const int r0 = d_start + int(fm);
  const int r1 = d_start + int(fm + 8);
  if (uint(r0) < uint(R)) {
    vec<bfloat, 4> v0 = *(const device vec<bfloat, 4>*)(rel_proj_t + long(r0) * 16 + fn);
    dst[0] = v0[0]; dst[1] = v0[1]; dst[2] = v0[2]; dst[3] = v0[3];
  } else {
    dst[0] = bfloat(0.0f); dst[1] = bfloat(0.0f); dst[2] = bfloat(0.0f); dst[3] = bfloat(0.0f);
  }
  if (uint(r1) < uint(R)) {
    vec<bfloat, 4> v1 = *(const device vec<bfloat, 4>*)(rel_proj_t + long(r1) * 16 + fn);
    dst[4] = v1[0]; dst[5] = v1[1]; dst[6] = v1[2]; dst[7] = v1[3];
  } else {
    dst[4] = bfloat(0.0f); dst[5] = bfloat(0.0f); dst[6] = bfloat(0.0f); dst[7] = bfloat(0.0f);
  }
}

static inline float dot16_bf16(
    const thread float4 r0, const thread float4 r1,
    const thread float4 r2, const thread float4 r3,
    const device bfloat* p) {
  const device vec<bfloat, 4>* pp = (const device vec<bfloat, 4>*)p;
  float4 p0 = float4(pp[0]);
  float4 p1 = float4(pp[1]);
  float4 p2 = float4(pp[2]);
  float4 p3 = float4(pp[3]);
  float b = dot(r0, p0) + dot(r1, p1) + dot(r2, p2) + dot(r3, p3);
  return float(bfloat(b));
}

template <typename CType, typename AType, typename BType, bool transpose_b>
static inline void nax_mma_16x32x16(
    thread vec<CType, 8>& Cn0,
    thread vec<CType, 8>& Cn1,
    const thread vec<AType, 8>& A,
    const thread vec<BType, 8>& Bn0,
    const thread vec<BType, 8>& Bn1) {
  constexpr auto desc = matmul2d_descriptor(
      16, 32, 16, false, transpose_b, true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;

  auto ct_a = op.template get_left_input_cooperative_tensor<AType, BType, CType>();
  auto ct_b = op.template get_right_input_cooperative_tensor<AType, BType, CType>();
  auto ct_c = op.template get_destination_cooperative_tensor<
      metal::remove_addrspace_t<decltype(ct_a)>,
      metal::remove_addrspace_t<decltype(ct_b)>,
      CType>();

  #pragma clang loop unroll(full)
  for (short i = 0; i < 8; ++i) ct_a[i] = A[i];
  #pragma clang loop unroll(full)
  for (short i = 0; i < 8; ++i) {
    ct_b[i] = Bn0[i];
    ct_b[8 + i] = Bn1[i];
  }
  #pragma clang loop unroll(full)
  for (short i = 0; i < 8; ++i) {
    ct_c[i] = Cn0[i];
    ct_c[8 + i] = Cn1[i];
  }
  op.run(ct_a, ct_b, ct_c);
  #pragma clang loop unroll(full)
  for (short i = 0; i < 8; ++i) {
    Cn0[i] = ct_c[i];
    Cn1[i] = ct_c[8 + i];
  }
}

static inline float read_sink(const device void* sinks, long idx, int sinks_bf16) {
  if (sinks_bf16) return float(((const device bfloat*)sinks)[idx]);
  return ((const device float*)sinks)[idx];
}

template <int BK, int B_STRIDE>
static inline void compute_rel_toeplitz_nax(
    threadgroup bfloat* B_sg,
    const thread vec<bfloat, 8>& R_frag,
    const device bfloat* rel_proj_t,
    int d_base,
    int R,
    short fm,
    short fn) {
  constexpr int N_CHUNKS = (BK == 32) ? 2 : 3;
  #pragma clang loop unroll(full)
  for (int ch = 0; ch < N_CHUNKS; ++ch) {
    vec<bfloat, 8> P0, P1;
    const int d0 = d_base + ch * 32;
    const int d1 = d0 + 16;
    if (d0 >= 0 && d0 + 16 <= R) {
      load_frag16x16(P0, rel_proj_t + long(d0) * 16, 16, fm, fn);
    } else {
      load_rel_proj_frag16x16(P0, rel_proj_t, d0, R, fm, fn);
    }
    const bool use_p1 = (ch * 32 + 16 < B_STRIDE);
    if (use_p1) {
      if (d1 >= 0 && d1 + 16 <= R) {
        load_frag16x16(P1, rel_proj_t + long(d1) * 16, 16, fm, fn);
      } else {
        load_rel_proj_frag16x16(P1, rel_proj_t, d1, R, fm, fn);
      }
    } else {
      P1 = vec<bfloat, 8>(bfloat(0.0f));
    }

    vec<float, 8> B0(0.0f), B1(0.0f);
    nax_mma_16x32x16<float, bfloat, bfloat, true>(B0, B1, R_frag, P0, P1);

    const int c0 = ch * 32 + int(fn);
    *(threadgroup vec<bfloat, 4>*)(B_sg + int(fm) * B_STRIDE + c0) =
        vec<bfloat, 4>(bfloat(B0[0]), bfloat(B0[1]), bfloat(B0[2]), bfloat(B0[3]));
    *(threadgroup vec<bfloat, 4>*)(B_sg + int(fm + 8) * B_STRIDE + c0) =
        vec<bfloat, 4>(bfloat(B0[4]), bfloat(B0[5]), bfloat(B0[6]), bfloat(B0[7]));

    if (use_p1) {
      const int c1 = ch * 32 + 16 + int(fn);
      *(threadgroup vec<bfloat, 4>*)(B_sg + int(fm) * B_STRIDE + c1) =
          vec<bfloat, 4>(bfloat(B1[0]), bfloat(B1[1]), bfloat(B1[2]), bfloat(B1[3]));
      *(threadgroup vec<bfloat, 4>*)(B_sg + int(fm + 8) * B_STRIDE + c1) =
          vec<bfloat, 4>(bfloat(B1[4]), bfloat(B1[5]), bfloat(B1[6]), bfloat(B1[7]));
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
}

template <int BK, bool USE_BF16_P, int REL_MODE, bool USE_ID4_BARRIER>
static inline void flash_attn_nax_compact_impl(
    const device bfloat* Q,
    const device bfloat* K,
    const device bfloat* V,
    const device bfloat* r_vec,
    const device bfloat* rel_proj_t,
    const device void* sinks,
    const device int* key_start,
    const device bfloat* gate_proj,
    device bfloat* O,
    constant MetalFlashParams& p,
    threadgroup bfloat* B_sg,
    uint simd_lane_id,
    uint simd_group_id,
    uint3 tgid) {
  constexpr int BQ = 64;
  constexpr int D = 128;
  constexpr int TD = D / 16;
  constexpr int TK = BK / 16;
  constexpr int B_STRIDE = (BK == 32) ? 48 : 80;

  const int q_block = int(tgid.x) * BQ;
  const int head = int(tgid.y);
  const int batch = int(tgid.z);
  const int q_sg0 = q_block + int(simd_group_id) * 16;

  if (!kAlignQK && q_sg0 >= p.L) return;

  const int lim_rows_q = p.L - q_sg0;
  const bool q_full = kAlignQK || (lim_rows_q >= 16);
  const int kv_head = head / p.G;
  const int start = key_start[batch];

  const short2 sc = nax_lane_coord(ushort(simd_lane_id));
  const short fn = sc.x;
  const short fm = sc.y;
  const int lim_q_fm = lim_rows_q - int(fm);

  const int q_stride = int(p.q_s2);
  const int k_stride = int(p.k_s2);
  const int v_stride = int(p.v_s2);

  const device bfloat* Q_pre = Q + long(batch) * p.q_s0 + long(head) * p.q_s1
      + long(q_sg0 + int(fm)) * p.q_s2 + int(fn);
  const device bfloat* K_base_pre = K + long(batch) * p.k_s0 + long(kv_head) * p.k_s1
      + long(fm) * p.k_s2 + int(fn);
  const device bfloat* V_base_pre = V + long(batch) * p.v_s0 + long(kv_head) * p.v_s1
      + long(fm) * p.v_s2 + int(fn);

  const int row0 = q_sg0 + int(fm);
  const int row1 = q_sg0 + int(fm + 8);
  const bool row0_ok = kAlignQK || (row0 < p.L);
  const bool row1_ok = kAlignQK || (row1 < p.L);

  vec<bfloat, 8> R_frag(bfloat(0.0f));
  if constexpr (REL_MODE == 0) {
    if (p.R > 0) {
      const device bfloat* r_sg = r_vec + (long(batch * p.L + q_sg0) * p.HQ + head) * 16;
      const long r_stride = long(p.HQ) * 16;
      if (q_full) load_frag16x16(R_frag, r_sg, r_stride, fm, fn);
      else load_frag16x16_rows(R_frag, r_sg, r_stride, lim_rows_q, fm, fn);
    }
  }

  vec<float, 8> Otile[TD];
  #pragma clang loop unroll(full)
  for (int id = 0; id < TD; ++id) Otile[id] = vec<float, 8>(0.0f);

  float max_score[2];
  max_score[0] = row0_ok ? read_sink(sinks, long(batch * p.L + row0) * p.HQ + head, p.sinks_bf16) * LOG2E : 0.0f;
  max_score[1] = row1_ok ? read_sink(sinks, long(batch * p.L + row1) * p.HQ + head, p.sinks_bf16) * LOG2E : 0.0f;
  float sum_score[2] = {1.0f, 1.0f};

  const int tg_first_diag = p.S - p.L + q_block;
  const int tg_last_diag = p.S - p.L + min(q_block + BQ, p.L) - 1;
  int tg_first_key = (p.window > 0) ? max(0, tg_first_diag - p.window + 1) : 0;
  tg_first_key = max(tg_first_key, start);
  const int k_begin = (tg_first_key / BK) * BK;

  const int sg_first_diag = p.S - p.L + q_sg0;
  const int sg_last_diag = p.S - p.L + min(q_sg0 + 16, p.L) - 1;
  int sg_first_key = (p.window > 0) ? max(0, sg_first_diag - p.window + 1) : 0;
  sg_first_key = max(sg_first_key, start);

  const int loop_end = (USE_ID4_BARRIER && kAlignQK) ? tg_last_diag : sg_last_diag;

  const device bfloat* K_cur = K_base_pre + long(k_begin) * p.k_s2;
  const device bfloat* V_cur = V_base_pre + long(k_begin) * p.v_s2;
  const int k_step = BK * k_stride;
  const int v_step = BK * v_stride;
  const int k_16 = 16 * k_stride;
  const int v_16 = 16 * v_stride;

  const float scale2 = p.scale2;
  const int diag0 = p.S - p.L + row0;
  const int diag1 = p.S - p.L + row1;
  const int r0_loc = int(fm);
  const int r1_loc = int(fm + 8);

  for (int k0 = k_begin; k0 <= loop_end; k0 += BK, K_cur += k_step, V_cur += v_step) {
    if (!(USE_ID4_BARRIER && kAlignQK)) {
      if (k0 + BK <= sg_first_key) continue;
    }

    const int lim_rows_k = p.S - k0;
    const bool k_full = kAlignQK || (lim_rows_k >= BK);

    vec<float, 8> Stile[TK];
    #pragma clang loop unroll(full)
    for (int ik = 0; ik < TK; ++ik) Stile[ik] = vec<float, 8>(0.0f);

    #pragma clang loop unroll(full)
    for (int ik = 0; ik < TK; ik += 2) {
      const device bfloat* K_p0 = K_cur + ik * k_16;
      const device bfloat* K_p1 = K_p0 + k_16;
      #pragma clang loop unroll_count(4)
      for (int id = 0; id < TD; ++id) {
        vec<bfloat, 8> Qf, K0, K1;
        if (q_full) {
          load_frag16x16_preoff(Qf, Q_pre + id * 16, q_stride);
        } else {
          load_frag16x16_rows_preoff(Qf, Q_pre + id * 16, q_stride, lim_q_fm);
        }
        if (k_full) {
          load_frag16x16_preoff(K0, K_p0 + id * 16, k_stride);
          load_frag16x16_preoff(K1, K_p1 + id * 16, k_stride);
        } else {
          load_frag16x16_rows_preoff(K0, K_p0 + id * 16, k_stride, lim_rows_k - ik * 16 - int(fm));
          load_frag16x16_rows_preoff(K1, K_p1 + id * 16, k_stride, lim_rows_k - (ik + 1) * 16 - int(fm));
        }
        nax_mma_16x32x16<float, bfloat, bfloat, true>(Stile[ik], Stile[ik + 1], Qf, K0, K1);
      }
    }

    const bool tile_has_rel = (p.R > 0) && ((sg_last_diag - k0) >= 0) && ((sg_first_diag - (k0 + BK - 1)) < p.R);
    const bool tile_full_vis = q_full && (k0 >= start) && (k0 + BK - 1 <= sg_first_diag)
        && (p.window == 0 || (sg_last_diag - k0) < p.window);

    if (tile_full_vis && !tile_has_rel) {
      #pragma clang loop unroll(full)
      for (int ik = 0; ik < TK; ++ik) Stile[ik] *= scale2;
    } else if constexpr (REL_MODE == 0) {
      if (tile_has_rel) {
        const int d_base = sg_first_diag - (k0 + BK - 1);
        compute_rel_toeplitz_nax<BK, B_STRIDE>(B_sg, R_frag, rel_proj_t, d_base, p.R, fm, fn);
      }

      if (tile_full_vis && tile_has_rel) {
        #pragma clang loop unroll(full)
        for (int ik = 0; ik < TK; ++ik) {
          const int c_base = ik * 16 + int(fn);
          const threadgroup bfloat* b0p = B_sg + r0_loc * B_STRIDE + r0_loc + (BK - 1) - c_base;
          const threadgroup bfloat* b1p = B_sg + r1_loc * B_STRIDE + r1_loc + (BK - 1) - c_base;
          #pragma clang loop unroll(full)
          for (int j = 0; j < 4; ++j) {
            Stile[ik][j] = Stile[ik][j] * scale2 + float(b0p[-j]) * LOG2E;
            Stile[ik][4 + j] = Stile[ik][4 + j] * scale2 + float(b1p[-j]) * LOG2E;
          }
        }
      } else {
        #pragma clang loop unroll(full)
        for (int ik = 0; ik < TK; ++ik) {
          const int c_base = ik * 16 + int(fn);
          #pragma clang loop unroll(full)
          for (int j = 0; j < 4; ++j) {
            const int key = k0 + c_base + j;
            {
              const int d = diag0 - key;
              const bool vis = row0_ok && (d >= 0) && (key >= start) && (p.window == 0 || d < p.window);
              float s = NEG_INF;
              if (vis) {
                s = Stile[ik][j] * scale2;
                if (tile_has_rel && d < p.R) {
                  s += float(B_sg[r0_loc * B_STRIDE + r0_loc + (BK - 1) - (c_base + j)]) * LOG2E;
                }
              }
              Stile[ik][j] = s;
            }
            {
              const int d = diag1 - key;
              const bool vis = row1_ok && (d >= 0) && (key >= start) && (p.window == 0 || d < p.window);
              float s = NEG_INF;
              if (vis) {
                s = Stile[ik][4 + j] * scale2;
                if (tile_has_rel && d < p.R) {
                  s += float(B_sg[r1_loc * B_STRIDE + r1_loc + (BK - 1) - (c_base + j)]) * LOG2E;
                }
              }
              Stile[ik][4 + j] = s;
            }
          }
        }
      }
    } else {
      float4 rv0_0(0), rv0_1(0), rv0_2(0), rv0_3(0);
      float4 rv1_0(0), rv1_1(0), rv1_2(0), rv1_3(0);
      if (tile_has_rel) {
        if (row0_ok) {
          const device vec<bfloat, 4>* rp0 = (const device vec<bfloat, 4>*)(r_vec + (long(batch * p.L + row0) * p.HQ + head) * 16);
          rv0_0 = float4(rp0[0]); rv0_1 = float4(rp0[1]); rv0_2 = float4(rp0[2]); rv0_3 = float4(rp0[3]);
        }
        if (row1_ok) {
          const device vec<bfloat, 4>* rp1 = (const device vec<bfloat, 4>*)(r_vec + (long(batch * p.L + row1) * p.HQ + head) * 16);
          rv1_0 = float4(rp1[0]); rv1_1 = float4(rp1[1]); rv1_2 = float4(rp1[2]); rv1_3 = float4(rp1[3]);
        }
      }

      #pragma clang loop unroll(full)
      for (int ik = 0; ik < TK; ++ik) {
        const int c_base = ik * 16 + int(fn);
        #pragma clang loop unroll(full)
        for (int j = 0; j < 4; ++j) {
          const int key = k0 + c_base + j;
          {
            const int d = diag0 - key;
            const bool vis = row0_ok && (d >= 0) && (key >= start) && (p.window == 0 || d < p.window);
            float s = NEG_INF;
            if (vis) {
              s = Stile[ik][j] * scale2;
              if (tile_has_rel && d < p.R) {
                s += dot16_bf16(rv0_0, rv0_1, rv0_2, rv0_3, rel_proj_t + long(d) * 16) * LOG2E;
              }
            }
            Stile[ik][j] = s;
          }
          {
            const int d = diag1 - key;
            const bool vis = row1_ok && (d >= 0) && (key >= start) && (p.window == 0 || d < p.window);
            float s = NEG_INF;
            if (vis) {
              s = Stile[ik][4 + j] * scale2;
              if (tile_has_rel && d < p.R) {
                s += dot16_bf16(rv1_0, rv1_1, rv1_2, rv1_3, rel_proj_t + long(d) * 16) * LOG2E;
              }
            }
            Stile[ik][4 + j] = s;
          }
        }
      }
    }

    float new_max[2] = {max_score[0], max_score[1]};
    #pragma clang loop unroll(full)
    for (int ik = 0; ik < TK; ++ik) {
      float m0 = max(max(Stile[ik][0], Stile[ik][1]), max(Stile[ik][2], Stile[ik][3]));
      float m1 = max(max(Stile[ik][4], Stile[ik][5]), max(Stile[ik][6], Stile[ik][7]));
      new_max[0] = max(new_max[0], m0);
      new_max[1] = max(new_max[1], m1);
    }
    new_max[0] = max(new_max[0], simd_shuffle_xor(new_max[0], ushort(1)));
    new_max[0] = max(new_max[0], simd_shuffle_xor(new_max[0], ushort(8)));
    new_max[1] = max(new_max[1], simd_shuffle_xor(new_max[1], ushort(1)));
    new_max[1] = max(new_max[1], simd_shuffle_xor(new_max[1], ushort(8)));

    const float factor0 = fast::exp2(max_score[0] - new_max[0]);
    const float factor1 = fast::exp2(max_score[1] - new_max[1]);
    max_score[0] = new_max[0];
    max_score[1] = new_max[1];

    float tile_sum0 = 0.0f;
    float tile_sum1 = 0.0f;
    vec<bfloat, 8> Pbf[USE_BF16_P ? TK : 1];

    #pragma clang loop unroll(full)
    for (int ik = 0; ik < TK; ++ik) {
      #pragma clang loop unroll(full)
      for (int j = 0; j < 4; ++j) {
        float p0 = fast::exp2(Stile[ik][j] - new_max[0]);
        float p1 = fast::exp2(Stile[ik][4 + j] - new_max[1]);
        tile_sum0 += p0;
        tile_sum1 += p1;
        if constexpr (USE_BF16_P) {
          Pbf[ik][j] = bfloat(p0);
          Pbf[ik][4 + j] = bfloat(p1);
        } else {
          Stile[ik][j] = p0;
          Stile[ik][4 + j] = p1;
        }
      }
    }
    tile_sum0 += simd_shuffle_xor(tile_sum0, ushort(1));
    tile_sum0 += simd_shuffle_xor(tile_sum0, ushort(8));
    tile_sum1 += simd_shuffle_xor(tile_sum1, ushort(1));
    tile_sum1 += simd_shuffle_xor(tile_sum1, ushort(8));

    sum_score[0] = sum_score[0] * factor0 + tile_sum0;
    sum_score[1] = sum_score[1] * factor1 + tile_sum1;

    #pragma clang loop unroll(full)
    for (int id = 0; id < TD; ++id) {
      Otile[id][0] *= factor0; Otile[id][1] *= factor0; Otile[id][2] *= factor0; Otile[id][3] *= factor0;
      Otile[id][4] *= factor1; Otile[id][5] *= factor1; Otile[id][6] *= factor1; Otile[id][7] *= factor1;
    }

    simdgroup_barrier(mem_flags::mem_none);

    #pragma clang loop unroll(full)
    for (int id = 0; id < TD; id += 2) {
      if constexpr (USE_ID4_BARRIER) {
        if (kAlignQK && id == 4) {
          threadgroup_barrier(mem_flags::mem_none);
        }
      }
      #pragma clang loop unroll(full)
      for (int ik = 0; ik < TK; ++ik) {
        vec<bfloat, 8> V0, V1;
        const device bfloat* V_p0 = V_cur + ik * v_16 + id * 16;
        if (k_full) {
          load_frag16x16_preoff(V0, V_p0, v_stride);
          load_frag16x16_preoff(V1, V_p0 + 16, v_stride);
        } else {
          const int lim_v = lim_rows_k - ik * 16 - int(fm);
          load_frag16x16_rows_preoff(V0, V_p0, v_stride, lim_v);
          load_frag16x16_rows_preoff(V1, V_p0 + 16, v_stride, lim_v);
        }
        if constexpr (USE_BF16_P) {
          nax_mma_16x32x16<float, bfloat, bfloat, false>(Otile[id], Otile[id + 1], Pbf[ik], V0, V1);
        } else {
          nax_mma_16x32x16<float, float, bfloat, false>(Otile[id], Otile[id + 1], Stile[ik], V0, V1);
        }
      }
    }
  }

  const float rcp0 = 1.0f / sum_score[0];
  const float rcp1 = 1.0f / sum_score[1];
  const long o_row_stride = long(p.HQ) * D;
  device bfloat* O_sg = O + (long(batch * p.L + q_sg0) * p.HQ + head) * D;

  if (p.has_gate) {
    const device bfloat* G_r0 = gate_proj + long(batch * p.L + row0) * p.gate_stride + p.gate_off + head * D;
    const device bfloat* G_r1 = gate_proj + long(batch * p.L + row1) * p.gate_stride + p.gate_off + head * D;
    #pragma clang loop unroll(full)
    for (int id = 0; id < TD; ++id) {
      const int col = id * 16 + int(fn);
      if (row0_ok) {
        vec<bfloat, 4> gv = *(const device vec<bfloat, 4>*)(G_r0 + col);
        vec<bfloat, 4> ov;
        #pragma clang loop unroll(full)
        for (int j = 0; j < 4; ++j) {
          float v = Otile[id][j] * rcp0;
          ov[j] = bfloat(v / (1.0f + metal::precise::exp(-float(gv[j]))));
        }
        *(device vec<bfloat, 4>*)(O_sg + long(fm) * o_row_stride + col) = ov;
      }
      if (row1_ok) {
        vec<bfloat, 4> gv = *(const device vec<bfloat, 4>*)(G_r1 + col);
        vec<bfloat, 4> ov;
        #pragma clang loop unroll(full)
        for (int j = 0; j < 4; ++j) {
          float v = Otile[id][4 + j] * rcp1;
          ov[j] = bfloat(v / (1.0f + metal::precise::exp(-float(gv[j]))));
        }
        *(device vec<bfloat, 4>*)(O_sg + long(fm + 8) * o_row_stride + col) = ov;
      }
    }
  } else {
    #pragma clang loop unroll(full)
    for (int id = 0; id < TD; ++id) {
      const int col = id * 16 + int(fn);
      if (row0_ok) {
        vec<bfloat, 4> ov(
            bfloat(Otile[id][0] * rcp0),
            bfloat(Otile[id][1] * rcp0),
            bfloat(Otile[id][2] * rcp0),
            bfloat(Otile[id][3] * rcp0));
        *(device vec<bfloat, 4>*)(O_sg + long(fm) * o_row_stride + col) = ov;
      }
      if (row1_ok) {
        vec<bfloat, 4> ov(
            bfloat(Otile[id][4] * rcp1),
            bfloat(Otile[id][5] * rcp1),
            bfloat(Otile[id][6] * rcp1),
            bfloat(Otile[id][7] * rcp1));
        *(device vec<bfloat, 4>*)(O_sg + long(fm + 8) * o_row_stride + col) = ov;
      }
    }
  }
}

[[kernel, max_total_threads_per_threadgroup(128)]]
kernel void flash_attn_nax_v1(
    const device bfloat* Q [[buffer(0)]],
    const device bfloat* K [[buffer(1)]],
    const device bfloat* V [[buffer(2)]],
    const device bfloat* r_vec [[buffer(3)]],
    const device bfloat* rel_proj_t [[buffer(4)]],
    const device void* sinks [[buffer(5)]],
    const device int* key_start [[buffer(6)]],
    const device bfloat* gate_proj [[buffer(7)]],
    device bfloat* O [[buffer(8)]],
    constant MetalFlashParams& p [[buffer(9)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
  threadgroup bfloat B_tg[4 * 16 * 48];
  flash_attn_nax_compact_impl<32, false, 0, false>(
      Q, K, V, r_vec, rel_proj_t, sinks, key_start, gate_proj, O, p,
      B_tg + int(simd_group_id) * (16 * 48), simd_lane_id, simd_group_id, tgid);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
kernel void flash_attn_nax_v2(
    const device bfloat* Q [[buffer(0)]],
    const device bfloat* K [[buffer(1)]],
    const device bfloat* V [[buffer(2)]],
    const device bfloat* r_vec [[buffer(3)]],
    const device bfloat* rel_proj_t [[buffer(4)]],
    const device void* sinks [[buffer(5)]],
    const device int* key_start [[buffer(6)]],
    const device bfloat* gate_proj [[buffer(7)]],
    device bfloat* O [[buffer(8)]],
    constant MetalFlashParams& p [[buffer(9)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
  threadgroup bfloat B_tg[4 * 16 * 80];
  flash_attn_nax_compact_impl<64, false, 0, true>(
      Q, K, V, r_vec, rel_proj_t, sinks, key_start, gate_proj, O, p,
      B_tg + int(simd_group_id) * (16 * 80), simd_lane_id, simd_group_id, tgid);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
kernel void flash_attn_nax_v3(
    const device bfloat* Q [[buffer(0)]],
    const device bfloat* K [[buffer(1)]],
    const device bfloat* V [[buffer(2)]],
    const device bfloat* r_vec [[buffer(3)]],
    const device bfloat* rel_proj_t [[buffer(4)]],
    const device void* sinks [[buffer(5)]],
    const device int* key_start [[buffer(6)]],
    const device bfloat* gate_proj [[buffer(7)]],
    device bfloat* O [[buffer(8)]],
    constant MetalFlashParams& p [[buffer(9)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
  threadgroup bfloat B_tg[4 * 16 * 48];
  flash_attn_nax_compact_impl<32, true, 0, false>(
      Q, K, V, r_vec, rel_proj_t, sinks, key_start, gate_proj, O, p,
      B_tg + int(simd_group_id) * (16 * 48), simd_lane_id, simd_group_id, tgid);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
kernel void flash_attn_nax_v4(
    const device bfloat* Q [[buffer(0)]],
    const device bfloat* K [[buffer(1)]],
    const device bfloat* V [[buffer(2)]],
    const device bfloat* r_vec [[buffer(3)]],
    const device bfloat* rel_proj_t [[buffer(4)]],
    const device void* sinks [[buffer(5)]],
    const device int* key_start [[buffer(6)]],
    const device bfloat* gate_proj [[buffer(7)]],
    device bfloat* O [[buffer(8)]],
    constant MetalFlashParams& p [[buffer(9)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
  threadgroup bfloat B_tg[4 * 16 * 80];
  flash_attn_nax_compact_impl<64, true, 0, false>(
      Q, K, V, r_vec, rel_proj_t, sinks, key_start, gate_proj, O, p,
      B_tg + int(simd_group_id) * (16 * 80), simd_lane_id, simd_group_id, tgid);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
kernel void flash_attn_nax_v5(
    const device bfloat* Q [[buffer(0)]],
    const device bfloat* K [[buffer(1)]],
    const device bfloat* V [[buffer(2)]],
    const device bfloat* r_vec [[buffer(3)]],
    const device bfloat* rel_proj_t [[buffer(4)]],
    const device void* sinks [[buffer(5)]],
    const device int* key_start [[buffer(6)]],
    const device bfloat* gate_proj [[buffer(7)]],
    device bfloat* O [[buffer(8)]],
    constant MetalFlashParams& p [[buffer(9)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
  flash_attn_nax_compact_impl<64, false, 1, true>(
      Q, K, V, r_vec, rel_proj_t, sinks, key_start, gate_proj, O, p,
      nullptr, simd_lane_id, simd_group_id, tgid);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
kernel void flash_attn_nax_v6_tg_coop(
    const device bfloat* Q [[buffer(0)]],
    const device bfloat* K [[buffer(1)]],
    const device bfloat* V [[buffer(2)]],
    const device bfloat* r_vec [[buffer(3)]],
    const device bfloat* rel_proj_t [[buffer(4)]],
    const device void* sinks [[buffer(5)]],
    const device int* key_start [[buffer(6)]],
    const device bfloat* gate_proj [[buffer(7)]],
    device bfloat* O [[buffer(8)]],
    constant MetalFlashParams& p [[buffer(9)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
  const uint tid = tid3.x;
  constexpr int BM = 64;
  constexpr int BK = 64;
  constexpr int D = 128;
  constexpr int B_STRIDE = 80;

  const int q0 = int(tgid.x) * BM;
  const int head = int(tgid.y);
  const int batch = int(tgid.z);
  if (q0 >= p.L) return;

  const int kv_head = head / p.G;
  const int start = key_start[batch];
  const int q_rows = min(BM, p.L - q0);

  using ext_t = dextents<int32_t, 2>;
  using str_t = array<int32_t, 2>;
  using dev_bf_t = tensor<device bfloat, ext_t, tensor_inline>;
  using tg_f32_t = tensor<threadgroup float, ext_t, tensor_inline>;
  using tg_bf_t  = tensor<threadgroup bfloat, ext_t, tensor_inline>;

  threadgroup float S_tg[BM * BK];
  threadgroup bfloat BP_tg[4 * 16 * B_STRIDE];
  threadgroup float tg_max[BM];
  threadgroup float tg_sum[BM];
  threadgroup float tg_fac[BM];

  if (tid < uint(BM)) {
    const int r = q0 + int(tid);
    tg_max[tid] = (r < p.L) ? read_sink(sinks, long(batch * p.L + r) * p.HQ + head, p.sinks_bf16) * LOG2E : 0.0f;
    tg_sum[tid] = 1.0f;
    tg_fac[tid] = 1.0f;
  }

  const short2 sc = nax_lane_coord(ushort(simd_lane_id));
  const short fn = sc.x;
  const short fm = sc.y;
  const int q_sg0 = q0 + int(simd_group_id) * 16;
  vec<bfloat, 8> R_frag(bfloat(0.0f));
  if (p.R > 0 && q_sg0 < p.L) {
    const device bfloat* r_sg = r_vec + (long(batch * p.L + q_sg0) * p.HQ + head) * 16;
    const long r_stride = long(p.HQ) * 16;
    if (q_sg0 + 16 <= p.L) load_frag16x16(R_frag, r_sg, r_stride, fm, fn);
    else load_frag16x16_rows(R_frag, r_sg, r_stride, p.L - q_sg0, fm, fn);
  }

  dev_bf_t tQ((device bfloat*)(Q + long(batch) * p.q_s0 + long(head) * p.q_s1 + long(q0) * p.q_s2),
              ext_t(D, q_rows), str_t{1, int(p.q_s2)});
  dev_bf_t tK((device bfloat*)(K + long(batch) * p.k_s0 + long(kv_head) * p.k_s1),
              ext_t(D, p.S), str_t{1, int(p.k_s2)});
  dev_bf_t tV((device bfloat*)(V + long(batch) * p.v_s0 + long(kv_head) * p.v_s1),
              ext_t(D, p.S), str_t{1, int(p.v_s2)});
  tg_f32_t tS(S_tg, ext_t(BK, BM), str_t{1, BK});
  tg_bf_t  tP(BP_tg, ext_t(BK, BM), str_t{1, BK});

  constexpr auto descQK = matmul2d_descriptor(
      BM, BK, D, false, true, true, matmul2d_descriptor::mode::multiply);
  matmul2d<descQK, execution_simdgroups<4>> opQK;

  constexpr auto descPV = matmul2d_descriptor(
      BM, D, BK, false, false, true, matmul2d_descriptor::mode::multiply);
  matmul2d<descPV, execution_simdgroups<4>> opPV;

  auto mP_full = tP.slice<BK, BM>(0, 0);
  auto mV_probe = tV.slice<D, BK>(0, 0);
  auto cO = opPV.get_destination_cooperative_tensor<decltype(mP_full), decltype(mV_probe), float>();
  #pragma clang loop unroll(full)
  for (uint16_t i = 0; i < cO.get_capacity(); ++i) cO[i] = 0.0f;

  const int tg_first_diag = p.S - p.L + q0;
  const int tg_last_diag = p.S - p.L + (q0 + q_rows - 1);
  int tg_first_key = (p.window > 0) ? max(0, tg_first_diag - p.window + 1) : 0;
  tg_first_key = max(tg_first_key, start);
  const int k_begin = (tg_first_key / BK) * BK;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int k0 = k_begin; k0 <= tg_last_diag; k0 += BK) {
    const bool q_full = (q_rows == BM);
    const bool k_full = (k0 + BK <= p.S);

    if (q_full && k_full) {
      auto mQ = tQ.slice<D, BM>(0, 0);
      auto mK = tK.slice<D, BK>(0, k0);
      auto mS = tS.slice<BK, BM>(0, 0);
      opQK.run(mQ, mK, mS);
    } else {
      auto mQ = tQ.slice(0, 0);
      auto mK = tK.slice(0, k0);
      auto mS = tS.slice(0, 0);
      opQK.run(mQ, mK, mS);
    }

    const bool tile_has_rel = (p.R > 0) && ((tg_last_diag - k0) >= 0) && ((tg_first_diag - (k0 + BK - 1)) < p.R);
    if (tile_has_rel && q_sg0 < p.L) {
      const int sg_first_diag = p.S - p.L + q_sg0;
      const int d_base = sg_first_diag - (k0 + BK - 1);
      compute_rel_toeplitz_nax<BK, B_STRIDE>(
          BP_tg + int(simd_group_id) * (16 * B_STRIDE), R_frag, rel_proj_t, d_base, p.R, fm, fn);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
      const int r_loc = int(tid >> 1);
      const int half_id = int(tid & 1);
      const int c_start = half_id * 32;
      const int row = q0 + r_loc;
      const bool row_ok = (row < p.L);
      const int diag = p.S - p.L + row;
      const bool row_full_vis = row_ok && (k0 >= start) && (k0 + BK - 1 <= diag)
          && (p.window == 0 || (diag - k0) < p.window);

      const int r_sg = r_loc >> 4;
      const int r_in_sg = r_loc & 15;
      const threadgroup bfloat* b_row = BP_tg + (r_sg * 16 + r_in_sg) * B_STRIDE;

      float local_max = NEG_INF;
      if (row_full_vis && !tile_has_rel) {
        #pragma clang loop unroll(full)
        for (int c = c_start; c < c_start + 32; ++c) {
          float s = S_tg[r_loc * BK + c] * p.scale2;
          S_tg[r_loc * BK + c] = s;
          local_max = max(local_max, s);
        }
      } else {
        #pragma clang loop unroll(full)
        for (int c = c_start; c < c_start + 32; ++c) {
          const int key = k0 + c;
          const int d = diag - key;
          const bool vis = row_ok && (d >= 0) && (key >= start) && (p.window == 0 || d < p.window);
          float s = NEG_INF;
          if (vis) {
            s = S_tg[r_loc * BK + c] * p.scale2;
            if (tile_has_rel && d < p.R) {
              s += float(b_row[r_in_sg + (BK - 1) - c]) * LOG2E;
            }
          }
          S_tg[r_loc * BK + c] = s;
          local_max = max(local_max, s);
        }
      }

      float tile_max = max(local_max, simd_shuffle_xor(local_max, ushort(1)));
      float old_max = tg_max[r_loc];
      float new_max = max(old_max, tile_max);
      float fac = fast::exp2(old_max - new_max);

      bfloat p_local[32];
      float local_sum = 0.0f;
      #pragma clang loop unroll(full)
      for (int j = 0; j < 32; ++j) {
        float pv = fast::exp2(S_tg[r_loc * BK + c_start + j] - new_max);
        local_sum += pv;
        p_local[j] = bfloat(pv);
      }
      float tile_sum = local_sum + simd_shuffle_xor(local_sum, ushort(1));

      if (half_id == 0) {
        tg_max[r_loc] = new_max;
        tg_sum[r_loc] = tg_sum[r_loc] * fac + tile_sum;
        tg_fac[r_loc] = fac;
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);
      #pragma clang loop unroll(full)
      for (int j = 0; j < 32; ++j) {
        BP_tg[r_loc * BK + c_start + j] = p_local[j];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto cO_step = opPV.get_destination_cooperative_tensor<decltype(mP_full), decltype(mV_probe), float>();
    if (k_full) {
      auto mV = tV.slice<D, BK>(0, k0);
      opPV.run(mP_full, mV, cO_step);
    } else {
      auto mV = tV.slice(0, k0);
      opPV.run(mP_full, mV, cO_step);
    }

    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
      if (cO.is_valid_element(i)) {
        auto idx = cO.get_multidimensional_index(i);
        cO[i] = cO[i] * tg_fac[idx[1]] + cO_step[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const long o_row_stride = long(p.HQ) * D;
  device bfloat* O_base = O + (long(batch * p.L + q0) * p.HQ + head) * D;

  #pragma clang loop unroll(full)
  for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
    if (cO.is_valid_element(i)) {
      auto idx = cO.get_multidimensional_index(i);
      const int col = int(idx[0]);
      const int r_loc = int(idx[1]);
      if (r_loc < q_rows) {
        float val = cO[i] / tg_sum[r_loc];
        if (p.has_gate) {
          const int row = q0 + r_loc;
          float gv = float(gate_proj[long(batch * p.L + row) * p.gate_stride + p.gate_off + head * D + col]);
          val = val / (1.0f + metal::precise::exp(-gv));
        }
        O_base[long(r_loc) * o_row_stride + col] = bfloat(val);
      }
    }
  }
}
)METAL";

static id<MTLComputePipelineState> compile_pso_fc(const char* name, bool align_qk) {
  NSError* err = nil;
  MTLFunctionConstantValues* fcv = [MTLFunctionConstantValues new];
  [fcv setConstantValue:&align_qk type:MTLDataTypeBool atIndex:0];
  id<MTLFunction> fn = [g_lib newFunctionWithName:[NSString stringWithUTF8String:name]
                                   constantValues:fcv
                                            error:&err];
  if (!fn) {
    std::fprintf(stderr, "[flash_attn_nax] Function failed: %s (align=%d): %s\n", name, int(align_qk),
                 err ? [[err localizedDescription] UTF8String] : "unknown");
    return nil;
  }
  id<MTLComputePipelineState> pso = [g_device newComputePipelineStateWithFunction:fn error:&err];
  if (!pso) {
    std::fprintf(stderr, "[flash_attn_nax] PSO failed for %s: %s\n", name,
                 err ? [[err localizedDescription] UTF8String] : "unknown");
  }
  return pso;
}

static id<MTLComputePipelineState> compile_pso(const char* name) {
  NSError* err = nil;
  id<MTLFunction> fn = [g_lib newFunctionWithName:[NSString stringWithUTF8String:name]];
  if (!fn) return nil;
  return [g_device newComputePipelineStateWithFunction:fn error:&err];
}

static size_t compute_strided_span_bf16(
    int32_t d0, int32_t d1, int32_t d2, int32_t d3,
    int64_t s0, int64_t s1, int64_t s2) {
  if (d0 <= 0 || d1 <= 0 || d2 <= 0 || d3 <= 0) return 0;
  int64_t max_elem = int64_t(d0 - 1) * s0 + int64_t(d1 - 1) * s1 + int64_t(d2 - 1) * s2 + int64_t(d3);
  return static_cast<size_t>(std::max<int64_t>(0, max_elem)) * sizeof(uint16_t);
}

static id<MTLComputePipelineState> select_pso(int v, const InklingFlashAttnDesc* desc) {
  if (v == 0) {
    if (desc->window == 0 && desc->S > 2048) {
      v = 2;
    } else {
      v = 1;
    }
  }
  const int bk = (v == 2 || v == 4 || v == 5 || v == 6) ? 64 : 32;
  const int aligned = ((desc->L % 64) == 0 && (desc->S % bk) == 0) ? 1 : 0;
  switch (v) {
    case 1: return g_pso_v1[aligned];
    case 2: return g_pso_v2[aligned];
    case 3: return g_pso_v3[aligned];
    case 4: return g_pso_v4[aligned];
    case 5: return g_pso_v5[aligned];
    case 6: return g_pso_v6;
    default: return g_pso_v1[aligned];
  }
}

extern "C" {

int inkling_flash_attn_init(void) {
  @autoreleasepool {
    if (g_device != nil && g_pso_v1[0] != nil) return 0;
    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) return -1;
    g_queue = [g_device newCommandQueue];
    if (!g_queue) return -2;
    g_dummy_buf = [g_device newBufferWithLength:256 options:MTLResourceStorageModeShared];
    std::memset([g_dummy_buf contents], 0, 256);

    NSError* err = nil;
    MTLCompileOptions* opts = [MTLCompileOptions new];
    opts.languageVersion = MTLLanguageVersion4_0;
    g_lib = [g_device newLibraryWithSource:[NSString stringWithUTF8String:kFlashAttnMSL]
                                   options:opts
                                     error:&err];
    if (!g_lib) {
      std::fprintf(stderr, "[flash_attn_nax] Metal 4.0 compile failed:\n%s\n",
                   err ? [[err localizedDescription] UTF8String] : "unknown");
      return -3;
    }

    for (int a = 0; a < 2; ++a) {
      g_pso_v1[a] = compile_pso_fc("flash_attn_nax_v1", a == 1);
      g_pso_v2[a] = compile_pso_fc("flash_attn_nax_v2", a == 1);
      g_pso_v3[a] = compile_pso_fc("flash_attn_nax_v3", a == 1);
      g_pso_v4[a] = compile_pso_fc("flash_attn_nax_v4", a == 1);
      g_pso_v5[a] = compile_pso_fc("flash_attn_nax_v5", a == 1);
      if (!g_pso_v1[a] || !g_pso_v2[a] || !g_pso_v3[a] || !g_pso_v4[a] || !g_pso_v5[a]) return -4;
    }
    g_pso_v6 = compile_pso("flash_attn_nax_v6_tg_coop");
    if (!g_pso_v6) return -4;
    return 0;
  }
}

void inkling_flash_attn_clear_cache(void) {
  std::lock_guard<std::mutex> lock(g_cache_mu);
  g_buf_cache.clear();
}

const char* inkling_flash_attn_device_name(void) {
  static char buf[256] = {0};
  if (!g_device) inkling_flash_attn_init();
  if (g_device) {
    std::snprintf(buf, sizeof(buf), "%s", [[g_device name] UTF8String]);
  }
  return buf;
}

static int run_flash_attn_impl(
    const InklingFlashAttnDesc* desc,
    uintptr_t q_ptr, size_t q_nbytes,
    uintptr_t k_ptr, size_t k_nbytes,
    uintptr_t v_ptr, size_t v_nbytes,
    uintptr_t r_or_reltbl_ptr, size_t r_or_reltbl_nbytes,
    uintptr_t rel_proj_t_ptr, size_t rel_proj_t_nbytes,
    uintptr_t sinks_ptr, size_t sinks_nbytes,
    uintptr_t key_start_ptr, size_t key_start_nbytes,
    uintptr_t gate_proj_ptr, size_t gate_proj_nbytes,
    uintptr_t out_ptr, size_t out_nbytes,
    int warmup, int iters,
    double* out_median_gpu_ms, double* out_median_wall_ms) {
  @autoreleasepool {
    if (!desc) return -10;
    if (desc->D != 128) return -11;
    if (desc->HQ <= 0 || desc->HKV <= 0 || (desc->HQ % desc->HKV) != 0) return -12;
    if (desc->B <= 0 || desc->L <= 0 || desc->S <= 0) return -13;

    int rc = inkling_flash_attn_init();
    if (rc != 0) return rc;

    id<MTLComputePipelineState> pso = select_pso(desc->variant, desc);
    if (!pso) return -14;

    const size_t q_need = std::max(q_nbytes, compute_strided_span_bf16(desc->B, desc->HQ, desc->L, desc->D, desc->q_s0, desc->q_s1, desc->q_s2));
    const size_t k_need = std::max(k_nbytes, compute_strided_span_bf16(desc->B, desc->HKV, desc->S, desc->D, desc->k_s0, desc->k_s1, desc->k_s2));
    const size_t v_need = std::max(v_nbytes, compute_strided_span_bf16(desc->B, desc->HKV, desc->S, desc->D, desc->v_s0, desc->v_s1, desc->v_s2));

    WrappedBuffer wQ = wrap_ptr_nocopy_cached(q_ptr, q_need);
    WrappedBuffer wK = wrap_ptr_nocopy_cached(k_ptr, k_need);
    WrappedBuffer wV = wrap_ptr_nocopy_cached(v_ptr, v_need);
    WrappedBuffer wR = wrap_ptr_nocopy_cached(r_or_reltbl_ptr, r_or_reltbl_nbytes);
    WrappedBuffer wP = wrap_ptr_nocopy_cached(rel_proj_t_ptr, rel_proj_t_nbytes);
    WrappedBuffer wS = wrap_ptr_nocopy_cached(sinks_ptr, sinks_nbytes);
    WrappedBuffer wKS = wrap_ptr_nocopy_cached(key_start_ptr, key_start_nbytes);
    WrappedBuffer wG = (desc->has_gate && gate_proj_ptr)
        ? wrap_ptr_nocopy_cached(gate_proj_ptr, gate_proj_nbytes)
        : WrappedBuffer{g_dummy_buf, 0};
    WrappedBuffer wO = wrap_ptr_nocopy_cached(out_ptr, out_nbytes);

    if (!wQ.buf || !wK.buf || !wV.buf || !wR.buf || !wP.buf || !wS.buf || !wKS.buf || !wG.buf || !wO.buf) {
      return -15;
    }

    MetalFlashParams mp{};
    mp.B = desc->B;
    mp.HQ = desc->HQ;
    mp.HKV = desc->HKV;
    mp.G = desc->HQ / desc->HKV;
    mp.L = desc->L;
    mp.S = desc->S;
    mp.R = desc->R;
    mp.window = desc->window;
    mp.scale2 = desc->scale * 1.4426950408889634f;
    mp.sinks_bf16 = desc->sinks_is_bf16 ? 1 : 0;
    mp.has_gate = (desc->has_gate && gate_proj_ptr != 0) ? 1 : 0;
    mp.gate_off = desc->gate_off;
    mp.gate_stride = desc->gate_stride;
    mp.pad0 = 0;
    mp.q_s0 = desc->q_s0; mp.q_s1 = desc->q_s1; mp.q_s2 = desc->q_s2;
    mp.k_s0 = desc->k_s0; mp.k_s1 = desc->k_s1; mp.k_s2 = desc->k_s2;
    mp.v_s0 = desc->v_s0; mp.v_s1 = desc->v_s1; mp.v_s2 = desc->v_s2;

    const MTLSize grid = MTLSizeMake(
        static_cast<NSUInteger>((desc->L + 63) / 64),
        static_cast<NSUInteger>(desc->HQ),
        static_cast<NSUInteger>(desc->B));
    const MTLSize tg = MTLSizeMake(128, 1, 1);

    std::vector<double> gpu_times;
    std::vector<double> wall_times;
    gpu_times.reserve(static_cast<size_t>(iters));
    wall_times.reserve(static_cast<size_t>(iters));

    for (int it = -warmup; it < iters; ++it) {
      double t0 = now_ms();
      id<MTLCommandBuffer> cb = [g_queue commandBufferWithUnretainedReferences];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
      [enc setComputePipelineState:pso];
      [enc setBuffer:wQ.buf offset:wQ.offset atIndex:0];
      [enc setBuffer:wK.buf offset:wK.offset atIndex:1];
      [enc setBuffer:wV.buf offset:wV.offset atIndex:2];
      [enc setBuffer:wR.buf offset:wR.offset atIndex:3];
      [enc setBuffer:wP.buf offset:wP.offset atIndex:4];
      [enc setBuffer:wS.buf offset:wS.offset atIndex:5];
      [enc setBuffer:wKS.buf offset:wKS.offset atIndex:6];
      [enc setBuffer:wG.buf offset:wG.offset atIndex:7];
      [enc setBuffer:wO.buf offset:wO.offset atIndex:8];
      [enc setBytes:&mp length:sizeof(mp) atIndex:9];
      [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      double t1 = now_ms();

      if ([cb status] != MTLCommandBufferStatusCompleted) {
        NSError* cberr = [cb error];
        std::fprintf(stderr, "[flash_attn_nax] CommandBuffer failed: %s\n",
                     cberr ? [[cberr localizedDescription] UTF8String] : "unknown");
        return -16;
      }

      if (it >= 0) {
        gpu_times.push_back(([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0);
        wall_times.push_back(t1 - t0);
      }
    }

    std::sort(gpu_times.begin(), gpu_times.end());
    std::sort(wall_times.begin(), wall_times.end());
    if (out_median_gpu_ms && !gpu_times.empty()) {
      *out_median_gpu_ms = gpu_times[gpu_times.size() / 2];
    }
    if (out_median_wall_ms && !wall_times.empty()) {
      *out_median_wall_ms = wall_times[wall_times.size() / 2];
    }
    return 0;
  }
}

int inkling_flash_attn_nax(
    const InklingFlashAttnDesc* desc,
    uintptr_t q_ptr, size_t q_nbytes,
    uintptr_t k_ptr, size_t k_nbytes,
    uintptr_t v_ptr, size_t v_nbytes,
    uintptr_t r_or_reltbl_ptr, size_t r_or_reltbl_nbytes,
    uintptr_t rel_proj_t_ptr, size_t rel_proj_t_nbytes,
    uintptr_t sinks_ptr, size_t sinks_nbytes,
    uintptr_t key_start_ptr, size_t key_start_nbytes,
    uintptr_t gate_proj_ptr, size_t gate_proj_nbytes,
    uintptr_t out_ptr, size_t out_nbytes,
    double* out_gpu_ms) {
  return run_flash_attn_impl(
      desc,
      q_ptr, q_nbytes, k_ptr, k_nbytes, v_ptr, v_nbytes,
      r_or_reltbl_ptr, r_or_reltbl_nbytes, rel_proj_t_ptr, rel_proj_t_nbytes,
      sinks_ptr, sinks_nbytes, key_start_ptr, key_start_nbytes,
      gate_proj_ptr, gate_proj_nbytes, out_ptr, out_nbytes,
      0, 1, out_gpu_ms, nullptr);
}

int inkling_flash_attn_nax_bench(
    const InklingFlashAttnDesc* desc,
    uintptr_t q_ptr, size_t q_nbytes,
    uintptr_t k_ptr, size_t k_nbytes,
    uintptr_t v_ptr, size_t v_nbytes,
    uintptr_t r_or_reltbl_ptr, size_t r_or_reltbl_nbytes,
    uintptr_t rel_proj_t_ptr, size_t rel_proj_t_nbytes,
    uintptr_t sinks_ptr, size_t sinks_nbytes,
    uintptr_t key_start_ptr, size_t key_start_nbytes,
    uintptr_t gate_proj_ptr, size_t gate_proj_nbytes,
    uintptr_t out_ptr, size_t out_nbytes,
    int warmup, int iters,
    double* out_median_gpu_ms, double* out_median_wall_ms) {
  return run_flash_attn_impl(
      desc,
      q_ptr, q_nbytes, k_ptr, k_nbytes, v_ptr, v_nbytes,
      r_or_reltbl_ptr, r_or_reltbl_nbytes, rel_proj_t_ptr, rel_proj_t_nbytes,
      sinks_ptr, sinks_nbytes, key_start_ptr, key_start_nbytes,
      gate_proj_ptr, gate_proj_nbytes, out_ptr, out_nbytes,
      warmup, iters, out_median_gpu_ms, out_median_wall_ms);
}

}
