#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "prefill_moe_nax.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mach/vm_page_size.h>
#include <mutex>
#include <unordered_map>

namespace {

struct CachedBuf {
  id<MTLBuffer> buf;
  size_t span;
};

struct BufSlice {
  id<MTLBuffer> buf;
  size_t offset;
};

struct BucketParamsHost {
  int32_t T;
  int32_t E;
  int32_t BM;
  int32_t n_tiles_w13;
  int32_t n_tiles_w2;
  int32_t grid_order;
};

struct MoeParamsHost {
  int32_t M;
  int32_t K;
  int32_t INTER;
  int32_t D;
  int32_t E;
  int32_t TOP_K;
  int32_t grid_order;
  int32_t _pad;
  float gate_cap;
  float up_cap;
};

struct CombineParamsHost {
  int32_t B;
  int32_t L;
  int32_t D;
  int32_t TOP_K;
  int32_t has_mask;
  int32_t _pad;
};

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;

static id<MTLComputePipelineState> g_pso_bucket = nil;
static id<MTLComputePipelineState> g_pso_fused_combine = nil;

static id<MTLComputePipelineState> g_pso_w13_nvfp4_64x64 = nil;
static id<MTLComputePipelineState> g_pso_w2_nvfp4_64x64 = nil;
static id<MTLComputePipelineState> g_pso_w13_mxfp8_64x64 = nil;
static id<MTLComputePipelineState> g_pso_w2_mxfp8_64x64 = nil;

static id<MTLComputePipelineState> g_pso_w13_nvfp4_128x64 = nil;
static id<MTLComputePipelineState> g_pso_w2_nvfp4_128x64 = nil;
static id<MTLComputePipelineState> g_pso_w13_mxfp8_128x64 = nil;
static id<MTLComputePipelineState> g_pso_w2_mxfp8_128x64 = nil;

static std::mutex g_cache_mu;
static std::unordered_map<uintptr_t, CachedBuf> g_buf_cache;

static id<MTLBuffer> g_scratch_h_sorted = nil;
static size_t g_scratch_h_bytes = 0;

static id<MTLBuffer> g_scratch_y_sorted = nil;
static size_t g_scratch_y_bytes = 0;

static id<MTLBuffer> g_scratch_offsets = nil;
static size_t g_scratch_offsets_bytes = 0;

static id<MTLBuffer> g_scratch_sorted_slots = nil;
static size_t g_scratch_sorted_slots_bytes = 0;

static id<MTLBuffer> g_scratch_inv_order = nil;
static size_t g_scratch_inv_order_bytes = 0;

static id<MTLBuffer> g_scratch_tile_expert = nil;
static id<MTLBuffer> g_scratch_tile_moff = nil;
static size_t g_scratch_tile_bytes = 0;

static id<MTLBuffer> g_scratch_indirect = nil;

static bool ensure_scratch(
    __strong id<MTLBuffer>& buf,
    size_t& cap,
    size_t needed,
    MTLResourceOptions opts = MTLResourceStorageModeShared) {
  if (needed == 0) needed = 64;
  if (buf != nil && cap >= needed) return true;
  size_t alloc_sz = std::max<size_t>(needed, 4096);
  alloc_sz = ((alloc_sz + 16383) / 16384) * 16384;
  id<MTLBuffer> nb = [g_device newBufferWithLength:alloc_sz options:opts];
  if (!nb) return false;
  buf = nb;
  cap = alloc_sz;
  return true;
}

static BufSlice wrap_ptr(uintptr_t addr, size_t nbytes, bool persist) {
  if (addr == 0) return {nil, 0};
  const size_t page_sz = vm_page_size;
  const uintptr_t base = addr & ~(page_sz - 1);
  const size_t offset = static_cast<size_t>(addr - base);
  const size_t span = ((offset + std::max<size_t>(nbytes, 1) + page_sz - 1) / page_sz) * page_sz;

  if (persist) {
    std::lock_guard<std::mutex> lk(g_cache_mu);
    auto it = g_buf_cache.find(base);
    if (it != g_buf_cache.end() && it->second.span >= span) {
      return {it->second.buf, offset};
    }
  }

  id<MTLBuffer> buf = [g_device newBufferWithBytesNoCopy:reinterpret_cast<void*>(base)
                                                  length:span
                                                 options:MTLResourceStorageModeShared
                                             deallocator:nil];
  if (!buf) {
    std::fprintf(stderr, "[prefill_moe_nax] newBufferWithBytesNoCopy failed for ptr=0x%lx span=%zu\n",
                 (unsigned long)addr, span);
    return {nil, 0};
  }

  if (persist) {
    std::lock_guard<std::mutex> lk(g_cache_mu);
    g_buf_cache[base] = {buf, span};
  }
  return {buf, offset};
}

static id<MTLComputePipelineState> compile_pso(id<MTLLibrary> lib, const char* name) {
  NSError* err = nil;
  id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
  if (!fn) {
    std::fprintf(stderr, "[prefill_moe_nax] Function not found: %s\n", name);
    return nil;
  }
  id<MTLComputePipelineState> pso = [g_device newComputePipelineStateWithFunction:fn error:&err];
  if (!pso) {
    std::fprintf(stderr, "[prefill_moe_nax] PSO compile failed for %s: %s\n", name,
                 err ? [[err localizedDescription] UTF8String] : "unknown");
  }
  return pso;
}

static const char* kPrefillMoeMSL = R"(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

struct BucketParams {
  int T;
  int E;
  int BM;
  int n_tiles_w13;
  int n_tiles_w2;
  int grid_order;
};

struct MoeParams {
  int M;
  int K;
  int INTER;
  int D;
  int E;
  int TOP_K;
  int grid_order;
  int _pad;
  float gate_cap;
  float up_cap;
};

#define E2M1_FROW(hi) \
  float2(0.0f, hi), float2(0.5f, hi), float2(1.0f, hi), float2(1.5f, hi), \
  float2(2.0f, hi), float2(3.0f, hi), float2(4.0f, hi), float2(6.0f, hi), \
  float2(-0.0f, hi), float2(-0.5f, hi), float2(-1.0f, hi), float2(-1.5f, hi), \
  float2(-2.0f, hi), float2(-3.0f, hi), float2(-4.0f, hi), float2(-6.0f, hi)

constant float2 E2M1_PAIR_F[256] = {
  E2M1_FROW(0.0f),  E2M1_FROW(0.5f),  E2M1_FROW(1.0f),  E2M1_FROW(1.5f),
  E2M1_FROW(2.0f),  E2M1_FROW(3.0f),  E2M1_FROW(4.0f),  E2M1_FROW(6.0f),
  E2M1_FROW(-0.0f), E2M1_FROW(-0.5f), E2M1_FROW(-1.0f), E2M1_FROW(-1.5f),
  E2M1_FROW(-2.0f), E2M1_FROW(-3.0f), E2M1_FROW(-4.0f), E2M1_FROW(-6.0f)
};

inline float decode_fp8_e4m3_f32(uint8_t value) {
  ushort h = ushort((uint(value) & 0x80u) << 8) | ushort((uint(value) & 0x7Fu) << 7);
  return float(as_type<half>(h)) * 256.0f;
}

inline float decode_fp8_e8m0(uint8_t value) {
  uint32_t decoded = value == 0 ? 0x400000u : uint32_t(value) << 23;
  return as_type<float>(decoded);
}

inline float decode_fp8_e8m0_x256(uint8_t sc) {
  if (sc <= 246u && sc >= 1u) {
    return as_type<float>(uint32_t(sc + 8u) << 23);
  }
  return decode_fp8_e8m0(sc) * 256.0f;
}

inline vec<bfloat, 4> decode4_mxfp8_s256(uint w, float s256) {
  const ushort4 b = ushort4(as_type<uchar4>(w));
  const ushort4 h = ((b & ushort4(0x7Fu)) << 7) | ((b & ushort4(0x80u)) << 8);
  const float4 v = float4(as_type<half4>(h)) * s256;
  return vec<bfloat, 4>(bfloat(v.x), bfloat(v.y), bfloat(v.z), bfloat(v.w));
}

inline void decode16_mxfp8_bf16(uint4 w, uint8_t sc, threadgroup vec<bfloat, 4>* dst) {
  const float s256 = decode_fp8_e8m0_x256(sc);
  dst[0] = decode4_mxfp8_s256(w.x, s256);
  dst[1] = decode4_mxfp8_s256(w.y, s256);
  dst[2] = decode4_mxfp8_s256(w.z, s256);
  dst[3] = decode4_mxfp8_s256(w.w, s256);
}

inline void decode8_mxfp8_bf16(uint2 w, uint8_t sc, threadgroup vec<bfloat, 4>* dst) {
  const float s256 = decode_fp8_e8m0_x256(sc);
  dst[0] = decode4_mxfp8_s256(w.x, s256);
  dst[1] = decode4_mxfp8_s256(w.y, s256);
}

inline void decode8_nvfp4_bf16(uint w, float s, threadgroup vec<bfloat, 4>* dst) {
  float2 p0 = E2M1_PAIR_F[w & 0xFFu] * s;
  float2 p1 = E2M1_PAIR_F[(w >> 8) & 0xFFu] * s;
  float2 p2 = E2M1_PAIR_F[(w >> 16) & 0xFFu] * s;
  float2 p3 = E2M1_PAIR_F[w >> 24] * s;
  dst[0] = vec<bfloat, 4>(bfloat(p0.x), bfloat(p0.y), bfloat(p1.x), bfloat(p1.y));
  dst[1] = vec<bfloat, 4>(bfloat(p2.x), bfloat(p2.y), bfloat(p3.x), bfloat(p3.y));
}

inline float softcapped_swiglu(float gate, float up, float gate_cap, float up_cap) {
  float g = metal::precise::tanh(gate / gate_cap) * gate_cap
            * (1.0f / (1.0f + metal::precise::exp(-gate)));
  float u = metal::precise::tanh(up / up_cap) * up_cap;
  return g * u;
}

kernel void bucket_experts_with_inv(
    device const uint* inds [[buffer(0)]],
    device int* offsets [[buffer(1)]],
    device uint* sorted_slots [[buffer(2)]],
    device uint* inv_order [[buffer(3)]],
    device uint* tile_expert [[buffer(4)]],
    device uint* tile_moff [[buffer(5)]],
    device uint* indirect_args [[buffer(6)]],
    constant BucketParams& p [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup atomic_uint curs[256];
  threadgroup uint sg_tots[8];
  threadgroup uint tg_tots[8];

  atomic_store_explicit(&curs[tid], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int i = int(tid); i < p.T; i += 256) {
    uint e = inds[i];
    atomic_fetch_add_explicit(&curs[e], 1u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint c = (int(tid) < p.E) ? atomic_load_explicit(&curs[tid], memory_order_relaxed) : 0u;
  uint n_mt = (c + uint(p.BM) - 1u) / uint(p.BM);

  uint c_inc = simd_prefix_inclusive_sum(c);
  uint c_exc = c_inc - c;
  uint t_inc = simd_prefix_inclusive_sum(n_mt);
  uint t_exc = t_inc - n_mt;

  if (lane == 31) {
    sg_tots[sg] = c_inc;
    tg_tots[sg] = t_inc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint c_base = 0u;
  uint t_base = 0u;
  for (uint s = 0; s < sg; ++s) {
    c_base += sg_tots[s];
    t_base += tg_tots[s];
  }
  uint off = c_base + c_exc;
  uint toff = t_base + t_exc;

  if (int(tid) < p.E) {
    offsets[tid] = int(off);
    atomic_store_explicit(&curs[tid], off, memory_order_relaxed);
    for (uint t = 0; t < n_mt; ++t) {
      tile_expert[toff + t] = tid;
      tile_moff[toff + t] = t * uint(p.BM);
    }
  }
  if (int(tid) == p.E - 1) {
    offsets[p.E] = int(off + c);
    uint total_mtiles = toff + n_mt;
    if (p.grid_order == 0) {
      indirect_args[0] = uint(p.n_tiles_w13);
      indirect_args[1] = total_mtiles;
      indirect_args[2] = 1u;
      indirect_args[3] = uint(p.n_tiles_w2);
      indirect_args[4] = total_mtiles;
      indirect_args[5] = 1u;
    } else {
      indirect_args[0] = total_mtiles;
      indirect_args[1] = uint(p.n_tiles_w13);
      indirect_args[2] = 1u;
      indirect_args[3] = total_mtiles;
      indirect_args[4] = uint(p.n_tiles_w2);
      indirect_args[5] = 1u;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int i = int(tid); i < p.T; i += 256) {
    uint e = inds[i];
    uint pos = atomic_fetch_add_explicit(&curs[e], 1u, memory_order_relaxed);
    sorted_slots[pos] = uint(i);
    inv_order[i] = pos;
  }
}


template <bool IS_NVFP4>
kernel void w13_64x64x32_db_k(
    device const bfloat* x [[buffer(0)]],
    device const uint* w [[buffer(1)]],
    device const uint8_t* scales [[buffer(2)]],
    device const float* gscale [[buffer(3)]],
    device const int* offsets [[buffer(4)]],
    device const uint* sorted_slots [[buffer(5)]],
    device const uint* tile_expert [[buffer(6)]],
    device const uint* tile_moff [[buffer(7)]],
    device bfloat* h_sorted [[buffer(8)]],
    constant MoeParams& p [[buffer(9)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 32;

  const uint m_tile = (p.grid_order == 0) ? tgid.y : tgid.x;
  const uint n_tile = (p.grid_order == 0) ? tgid.x : tgid.y;
  const uint e = tile_expert[m_tile];
  const int m_off = int(tile_moff[m_tile]);
  const int beg = offsets[e];
  const int cnt = offsets[e + 1] - beg;
  const int rem_m = min(BM, cnt - m_off);
  const int n_off = int(n_tile) * BN;
  const int N_full = 2 * p.INTER;
  const int K = p.K;

  threadgroup bfloat A_tg[2][BM * BK];
  threadgroup bfloat B_tg[2][BN * BK];
  threadgroup int tg_tok[BM];

  if (int(tid) < BM) {
    tg_tok[tid] = (int(tid) < rem_m) ? int(sorted_slots[beg + m_off + int(tid)] / uint(p.TOP_K)) : -1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  using ext_t = dextents<int32_t, 2>;
  using str_t = array<int32_t, 2>;
  tensor<threadgroup bfloat, ext_t, tensor_inline> tA0(A_tg[0], ext_t(BK, BM), str_t{1, BK});
  tensor<threadgroup bfloat, ext_t, tensor_inline> tA1(A_tg[1], ext_t(BK, BM), str_t{1, BK});
  tensor<threadgroup bfloat, ext_t, tensor_inline> tB0(B_tg[0], ext_t(BK, BN), str_t{1, BK});
  tensor<threadgroup bfloat, ext_t, tensor_inline> tB1(B_tg[1], ext_t(BK, BN), str_t{1, BK});
  auto mA0 = tA0.slice<BK, BM>(0, 0);
  auto mA1 = tA1.slice<BK, BM>(0, 0);
  auto mB0 = tB0.slice<BK, BN>(0, 0);
  auto mB1 = tB1.slice<BK, BN>(0, 0);

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, BK, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<4>> op;

  auto cT = op.template get_destination_cooperative_tensor<decltype(mA0), decltype(mB0), float>();
  #pragma clang loop unroll(full)
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) cT[i] = 0.0f;

  const int r = int(tid >> 1);
  const int g_id = int(tid & 1u);
  const int c_off = g_id * 16;
  const int tok = tg_tok[r];
  const device bfloat* x_row = (tok >= 0) ? (x + size_t(tok) * K) : x;

  const size_t w_row = size_t(e) * N_full + size_t(n_off + r);
  const int W_STRIDE = IS_NVFP4 ? (K >> 3) : (K >> 2);
  const int S_STRIDE = IS_NVFP4 ? (K >> 4) : (K >> 5);
  const device uint* w_ptr = w + w_row * W_STRIDE + (IS_NVFP4 ? (g_id * 2) : (g_id * 4));
  const device uint8_t* s_ptr = scales + w_row * S_STRIDE + (IS_NVFP4 ? g_id : 0);

  uint4 reg_x0, reg_x1;
  uint4 reg_w4;
  uint2 reg_w2;
  uint8_t reg_sc;

  auto prefetch_ab = [&](int k) {
    if (tok >= 0) {
      const device uint4* x_src = (const device uint4*)(x_row + k + c_off);
      reg_x0 = x_src[0];
      reg_x1 = x_src[1];
    } else {
      reg_x0 = uint4(0u);
      reg_x1 = uint4(0u);
    }
    if constexpr (IS_NVFP4) {
      reg_w2 = *(const device uint2*)(w_ptr + (k >> 3));
      reg_sc = s_ptr[k >> 4];
    } else {
      reg_w4 = *(const device uint4*)(w_ptr + (k >> 2));
      reg_sc = s_ptr[k >> 5];
    }
  };

  auto store_ab = [&](int buf_idx) {
    threadgroup uint4* a_dst = (threadgroup uint4*)(A_tg[buf_idx] + r * BK + c_off);
    a_dst[0] = reg_x0;
    a_dst[1] = reg_x1;
    threadgroup vec<bfloat, 4>* b_dst = (threadgroup vec<bfloat, 4>*)(B_tg[buf_idx] + r * BK + c_off);
    if constexpr (IS_NVFP4) {
      const float sv = decode_fp8_e4m3_f32(reg_sc);
      decode8_nvfp4_bf16(reg_w2.x, sv, b_dst + 0);
      decode8_nvfp4_bf16(reg_w2.y, sv, b_dst + 2);
    } else {
      decode16_mxfp8_bf16(reg_w4, reg_sc, b_dst);
    }
  };

  prefetch_ab(0);
  store_ab(0);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int k = 0; k < K; k += 64) {
    prefetch_ab(k + 32);
    op.run(mA0, mB0, cT);
    store_ab(1);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (k + 64 < K) {
      prefetch_ab(k + 64);
      op.run(mA1, mB1, cT);
      store_ab(0);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
      op.run(mA1, mB1, cT);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float oscale = IS_NVFP4 ? (gscale[e] * (1.0f / 2688.0f)) : 1.0f;
  auto dOut = op.template get_destination_cooperative_tensor<decltype(mA0), decltype(mB0), bfloat>();
  #pragma clang loop unroll(full)
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
    dOut[i] = static_cast<bfloat>(IS_NVFP4 ? (cT[i] * oscale) : cT[i]);
  }
  tensor<threadgroup bfloat, ext_t, tensor_inline> tC(&A_tg[0][0], ext_t(BN, BM), str_t{1, BN});
  auto mC = tC.slice<BN, BM>(0, 0);
  dOut.store(mC);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (r < rem_m) {
    const int pair_base = g_id * 16;
    device bfloat* h_row = h_sorted + size_t(beg + m_off + r) * p.INTER + (n_off >> 1) + pair_base;
    const threadgroup bfloat* c_row = &A_tg[0][0] + r * BN + pair_base * 2;
    #pragma clang loop unroll(full)
    for (int j = 0; j < 16; ++j) {
      h_row[j] = bfloat(softcapped_swiglu(float(c_row[2 * j]), float(c_row[2 * j + 1]), p.gate_cap, p.up_cap));
    }
  }
}

template <bool IS_NVFP4>
kernel void w2_64x64x32_db_k(
    device const bfloat* h_sorted [[buffer(0)]],
    device const uint* w [[buffer(1)]],
    device const uint8_t* scales [[buffer(2)]],
    device const float* gscale [[buffer(3)]],
    device const int* offsets [[buffer(4)]],
    device const uint* tile_expert [[buffer(5)]],
    device const uint* tile_moff [[buffer(6)]],
    device bfloat* y_sorted [[buffer(7)]],
    constant MoeParams& p [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 32;

  const uint m_tile = (p.grid_order == 0) ? tgid.y : tgid.x;
  const uint n_tile = (p.grid_order == 0) ? tgid.x : tgid.y;
  const uint e = tile_expert[m_tile];
  const int m_off = int(tile_moff[m_tile]);
  const int beg = offsets[e];
  const int cnt = offsets[e + 1] - beg;
  const int rem_m = min(BM, cnt - m_off);
  const int n_off = int(n_tile) * BN;
  const int INTER = p.INTER;
  const int D = p.D;

  threadgroup bfloat B_tg[2][BN * BK];

  using ext_t = dextents<int32_t, 2>;
  using str_t = array<int32_t, 2>;
  tensor<device bfloat, ext_t, tensor_inline> tA(
      (device bfloat*)(h_sorted + size_t(beg + m_off) * INTER), ext_t(INTER, rem_m), str_t{1, INTER});
  tensor<threadgroup bfloat, ext_t, tensor_inline> tB0(B_tg[0], ext_t(BK, BN), str_t{1, BK});
  tensor<threadgroup bfloat, ext_t, tensor_inline> tB1(B_tg[1], ext_t(BK, BN), str_t{1, BK});
  auto mB0 = tB0.slice<BK, BN>(0, 0);
  auto mB1 = tB1.slice<BK, BN>(0, 0);

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, BK, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<4>> op;

  const int r = int(tid >> 1);
  const int g_id = int(tid & 1u);
  const int c_off = g_id * 16;
  const size_t w_row = size_t(e) * D + size_t(n_off + r);
  const int W_STRIDE = IS_NVFP4 ? (INTER >> 3) : (INTER >> 2);
  const int S_STRIDE = IS_NVFP4 ? (INTER >> 4) : (INTER >> 5);
  const device uint* w_ptr = w + w_row * W_STRIDE + (IS_NVFP4 ? (g_id * 2) : (g_id * 4));
  const device uint8_t* s_ptr = scales + w_row * S_STRIDE + (IS_NVFP4 ? g_id : 0);

  uint4 reg_w4;
  uint2 reg_w2;
  uint8_t reg_sc;

  auto prefetch_b = [&](int k) {
    if constexpr (IS_NVFP4) {
      reg_w2 = *(const device uint2*)(w_ptr + (k >> 3));
      reg_sc = s_ptr[k >> 4];
    } else {
      reg_w4 = *(const device uint4*)(w_ptr + (k >> 2));
      reg_sc = s_ptr[k >> 5];
    }
  };

  auto store_b = [&](int buf_idx) {
    threadgroup vec<bfloat, 4>* b_dst = (threadgroup vec<bfloat, 4>*)(B_tg[buf_idx] + r * BK + c_off);
    if constexpr (IS_NVFP4) {
      const float sv = decode_fp8_e4m3_f32(reg_sc);
      decode8_nvfp4_bf16(reg_w2.x, sv, b_dst + 0);
      decode8_nvfp4_bf16(reg_w2.y, sv, b_dst + 2);
    } else {
      decode16_mxfp8_bf16(reg_w4, reg_sc, b_dst);
    }
  };

  prefetch_b(0);
  store_b(0);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float oscale = IS_NVFP4 ? (gscale[e] * (1.0f / 2688.0f)) : 1.0f;
  tensor<device bfloat, ext_t, tensor_inline> tC(
      y_sorted + size_t(beg + m_off) * D + n_off, ext_t(D - n_off, rem_m), str_t{1, D});

  if (rem_m == BM && n_off + BN <= D) {
    auto a_probe = tA.slice<BK, BM>(0, 0);
    auto cT = op.template get_destination_cooperative_tensor<decltype(a_probe), decltype(mB0), float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) cT[i] = 0.0f;

    for (int k = 0; k < INTER; k += 64) {
      prefetch_b(k + 32);
      auto mA0 = tA.slice<BK, BM>(k, 0);
      op.run(mA0, mB0, cT);
      store_b(1);
      threadgroup_barrier(mem_flags::mem_threadgroup);

      auto mA1 = tA.slice<BK, BM>(k + 32, 0);
      if (k + 64 < INTER) {
        prefetch_b(k + 64);
        op.run(mA1, mB1, cT);
        store_b(0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
      } else {
        op.run(mA1, mB1, cT);
      }
    }

    auto dOut = op.template get_destination_cooperative_tensor<decltype(a_probe), decltype(mB0), bfloat>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) dOut[i] = static_cast<bfloat>(IS_NVFP4 ? (cT[i] * oscale) : cT[i]);
    auto mC = tC.slice<BN, BM>(0, 0);
    dOut.store(mC);
  } else {
    auto a_probe = tA.slice(0, 0);
    auto cT = op.template get_destination_cooperative_tensor<decltype(a_probe), decltype(mB0), float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) cT[i] = 0.0f;

    for (int k = 0; k < INTER; k += 64) {
      prefetch_b(k + 32);
      auto mA0 = tA.slice(k, 0);
      op.run(mA0, mB0, cT);
      store_b(1);
      threadgroup_barrier(mem_flags::mem_threadgroup);

      auto mA1 = tA.slice(k + 32, 0);
      if (k + 64 < INTER) {
        prefetch_b(k + 64);
        op.run(mA1, mB1, cT);
        store_b(0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
      } else {
        op.run(mA1, mB1, cT);
      }
    }

    auto dOut = op.template get_destination_cooperative_tensor<decltype(a_probe), decltype(mB0), bfloat>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) dOut[i] = static_cast<bfloat>(IS_NVFP4 ? (cT[i] * oscale) : cT[i]);
    auto mC = tC.slice(0, 0);
    dOut.store(mC);
  }
}


template <bool IS_NVFP4>
kernel void w13_128x64x32_sg8_k(
    device const bfloat* x [[buffer(0)]],
    device const uint* w [[buffer(1)]],
    device const uint8_t* scales [[buffer(2)]],
    device const float* gscale [[buffer(3)]],
    device const int* offsets [[buffer(4)]],
    device const uint* sorted_slots [[buffer(5)]],
    device const uint* tile_expert [[buffer(6)]],
    device const uint* tile_moff [[buffer(7)]],
    device bfloat* h_sorted [[buffer(8)]],
    constant MoeParams& p [[buffer(9)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  constexpr int BM = 128;
  constexpr int BN = 64;
  constexpr int BK = 32;

  const uint m_tile = (p.grid_order == 0) ? tgid.y : tgid.x;
  const uint n_tile = (p.grid_order == 0) ? tgid.x : tgid.y;
  const uint e = tile_expert[m_tile];
  const int m_off = int(tile_moff[m_tile]);
  const int beg = offsets[e];
  const int cnt = offsets[e + 1] - beg;
  const int rem_m = min(BM, cnt - m_off);
  const int n_off = int(n_tile) * BN;
  const int N_full = 2 * p.INTER;
  const int K = p.K;

  threadgroup bfloat smem[2 * (BM * BK + BN * BK)];
  threadgroup int tg_tok[BM];
  threadgroup bfloat* A0 = smem;
  threadgroup bfloat* A1 = smem + BM * BK;
  threadgroup bfloat* B0 = smem + 2 * BM * BK;
  threadgroup bfloat* B1 = smem + 2 * BM * BK + BN * BK;

  if (int(tid) < BM) {
    tg_tok[tid] = (int(tid) < rem_m) ? int(sorted_slots[beg + m_off + int(tid)] / uint(p.TOP_K)) : -1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  using ext_t = dextents<int32_t, 2>;
  using str_t = array<int32_t, 2>;
  tensor<threadgroup bfloat, ext_t, tensor_inline> tA0(A0, ext_t(BK, BM), str_t{1, BK});
  tensor<threadgroup bfloat, ext_t, tensor_inline> tA1(A1, ext_t(BK, BM), str_t{1, BK});
  tensor<threadgroup bfloat, ext_t, tensor_inline> tB0(B0, ext_t(BK, BN), str_t{1, BK});
  tensor<threadgroup bfloat, ext_t, tensor_inline> tB1(B1, ext_t(BK, BN), str_t{1, BK});
  auto mA0 = tA0.slice<BK, BM>(0, 0);
  auto mA1 = tA1.slice<BK, BM>(0, 0);
  auto mB0 = tB0.slice<BK, BN>(0, 0);
  auto mB1 = tB1.slice<BK, BN>(0, 0);

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, BK, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<8>> op;

  auto cT = op.template get_destination_cooperative_tensor<decltype(mA0), decltype(mB0), float>();
  #pragma clang loop unroll(full)
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) cT[i] = 0.0f;

  const int r_a = int(tid >> 1);
  const int c_a = int(tid & 1u) * 16;
  const int tok = tg_tok[r_a];
  const device bfloat* x_row = (tok >= 0) ? (x + size_t(tok) * K) : x;

  const int r_b = int(tid >> 2);
  const int q_b = int(tid & 3u);
  const size_t w_row = size_t(e) * N_full + size_t(n_off + r_b);
  const int W_STRIDE = IS_NVFP4 ? (K >> 3) : (K >> 2);
  const int S_STRIDE = IS_NVFP4 ? (K >> 4) : (K >> 5);
  const device uint* w_ptr = w + w_row * W_STRIDE;
  const device uint8_t* s_ptr = scales + w_row * S_STRIDE;

  uint4 reg_x0, reg_x1;
  uint2 reg_w2;
  uint reg_w1;
  uint8_t reg_sc;

  auto prefetch_ab = [&](int k) {
    if (tok >= 0) {
      const device uint4* x_src = (const device uint4*)(x_row + k + c_a);
      reg_x0 = x_src[0];
      reg_x1 = x_src[1];
    } else {
      reg_x0 = uint4(0u);
      reg_x1 = uint4(0u);
    }
    if constexpr (IS_NVFP4) {
      reg_w1 = w_ptr[(k >> 3) + q_b];
      reg_sc = s_ptr[(k >> 4) + (q_b >> 1)];
    } else {
      reg_w2 = *(const device uint2*)(w_ptr + (k >> 2) + q_b * 2);
      reg_sc = s_ptr[k >> 5];
    }
  };

  auto store_ab = [&](int buf_idx) {
    threadgroup bfloat* Acur = (buf_idx == 0) ? A0 : A1;
    threadgroup bfloat* Bcur = (buf_idx == 0) ? B0 : B1;
    threadgroup uint4* a_dst = (threadgroup uint4*)(Acur + r_a * BK + c_a);
    a_dst[0] = reg_x0;
    a_dst[1] = reg_x1;
    threadgroup vec<bfloat, 4>* b_dst = (threadgroup vec<bfloat, 4>*)(Bcur + r_b * BK + q_b * 8);
    if constexpr (IS_NVFP4) {
      const float sv = decode_fp8_e4m3_f32(reg_sc);
      decode8_nvfp4_bf16(reg_w1, sv, b_dst);
    } else {
      decode8_mxfp8_bf16(reg_w2, reg_sc, b_dst);
    }
  };

  prefetch_ab(0);
  store_ab(0);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int k = 0; k < K; k += 64) {
    prefetch_ab(k + 32);
    op.run(mA0, mB0, cT);
    store_ab(1);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (k + 64 < K) {
      prefetch_ab(k + 64);
      op.run(mA1, mB1, cT);
      store_ab(0);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
      op.run(mA1, mB1, cT);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float oscale = IS_NVFP4 ? (gscale[e] * (1.0f / 2688.0f)) : 1.0f;
  auto dOut = op.template get_destination_cooperative_tensor<decltype(mA0), decltype(mB0), bfloat>();
  #pragma clang loop unroll(full)
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) dOut[i] = static_cast<bfloat>(IS_NVFP4 ? (cT[i] * oscale) : cT[i]);

  tensor<threadgroup bfloat, ext_t, tensor_inline> tC(smem, ext_t(BN, BM), str_t{1, BN});
  auto mC = tC.slice<BN, BM>(0, 0);
  dOut.store(mC);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (r_a < rem_m) {
    const int pair_base = int(tid & 1u) * 16;
    device bfloat* h_row = h_sorted + size_t(beg + m_off + r_a) * p.INTER + (n_off >> 1) + pair_base;
    const threadgroup bfloat* c_row = smem + r_a * BN + pair_base * 2;
    #pragma clang loop unroll(full)
    for (int j = 0; j < 16; ++j) {
      h_row[j] = bfloat(softcapped_swiglu(float(c_row[2 * j]), float(c_row[2 * j + 1]), p.gate_cap, p.up_cap));
    }
  }
}

template <bool IS_NVFP4>
kernel void w2_128x64x32_sg8_k(
    device const bfloat* h_sorted [[buffer(0)]],
    device const uint* w [[buffer(1)]],
    device const uint8_t* scales [[buffer(2)]],
    device const float* gscale [[buffer(3)]],
    device const int* offsets [[buffer(4)]],
    device const uint* tile_expert [[buffer(5)]],
    device const uint* tile_moff [[buffer(6)]],
    device bfloat* y_sorted [[buffer(7)]],
    constant MoeParams& p [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  constexpr int BM = 128;
  constexpr int BN = 64;
  constexpr int BK = 32;

  const uint m_tile = (p.grid_order == 0) ? tgid.y : tgid.x;
  const uint n_tile = (p.grid_order == 0) ? tgid.x : tgid.y;
  const uint e = tile_expert[m_tile];
  const int m_off = int(tile_moff[m_tile]);
  const int beg = offsets[e];
  const int cnt = offsets[e + 1] - beg;
  const int rem_m = min(BM, cnt - m_off);
  const int n_off = int(n_tile) * BN;
  const int INTER = p.INTER;
  const int D = p.D;

  threadgroup bfloat B_tg[2][BN * BK];

  using ext_t = dextents<int32_t, 2>;
  using str_t = array<int32_t, 2>;
  tensor<device bfloat, ext_t, tensor_inline> tA(
      (device bfloat*)(h_sorted + size_t(beg + m_off) * INTER), ext_t(INTER, rem_m), str_t{1, INTER});
  tensor<threadgroup bfloat, ext_t, tensor_inline> tB0(B_tg[0], ext_t(BK, BN), str_t{1, BK});
  tensor<threadgroup bfloat, ext_t, tensor_inline> tB1(B_tg[1], ext_t(BK, BN), str_t{1, BK});
  auto mB0 = tB0.slice<BK, BN>(0, 0);
  auto mB1 = tB1.slice<BK, BN>(0, 0);

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, BK, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<8>> op;

  const int r_b = int(tid >> 2);
  const int q_b = int(tid & 3u);
  const size_t w_row = size_t(e) * D + size_t(n_off + r_b);
  const int W_STRIDE = IS_NVFP4 ? (INTER >> 3) : (INTER >> 2);
  const int S_STRIDE = IS_NVFP4 ? (INTER >> 4) : (INTER >> 5);
  const device uint* w_ptr = w + w_row * W_STRIDE;
  const device uint8_t* s_ptr = scales + w_row * S_STRIDE;

  uint2 reg_w2;
  uint reg_w1;
  uint8_t reg_sc;

  auto prefetch_b = [&](int k) {
    if constexpr (IS_NVFP4) {
      reg_w1 = w_ptr[(k >> 3) + q_b];
      reg_sc = s_ptr[(k >> 4) + (q_b >> 1)];
    } else {
      reg_w2 = *(const device uint2*)(w_ptr + (k >> 2) + q_b * 2);
      reg_sc = s_ptr[k >> 5];
    }
  };

  auto store_b = [&](int buf_idx) {
    threadgroup vec<bfloat, 4>* b_dst = (threadgroup vec<bfloat, 4>*)(B_tg[buf_idx] + r_b * BK + q_b * 8);
    if constexpr (IS_NVFP4) {
      const float sv = decode_fp8_e4m3_f32(reg_sc);
      decode8_nvfp4_bf16(reg_w1, sv, b_dst);
    } else {
      decode8_mxfp8_bf16(reg_w2, reg_sc, b_dst);
    }
  };

  prefetch_b(0);
  store_b(0);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float oscale = IS_NVFP4 ? (gscale[e] * (1.0f / 2688.0f)) : 1.0f;
  tensor<device bfloat, ext_t, tensor_inline> tC(
      y_sorted + size_t(beg + m_off) * D + n_off, ext_t(D - n_off, rem_m), str_t{1, D});

  if (rem_m == BM && n_off + BN <= D) {
    auto a_probe = tA.slice<BK, BM>(0, 0);
    auto cT = op.template get_destination_cooperative_tensor<decltype(a_probe), decltype(mB0), float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) cT[i] = 0.0f;

    for (int k = 0; k < INTER; k += 64) {
      prefetch_b(k + 32);
      auto mA0 = tA.slice<BK, BM>(k, 0);
      op.run(mA0, mB0, cT);
      store_b(1);
      threadgroup_barrier(mem_flags::mem_threadgroup);

      auto mA1 = tA.slice<BK, BM>(k + 32, 0);
      if (k + 64 < INTER) {
        prefetch_b(k + 64);
        op.run(mA1, mB1, cT);
        store_b(0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
      } else {
        op.run(mA1, mB1, cT);
      }
    }

    auto dOut = op.template get_destination_cooperative_tensor<decltype(a_probe), decltype(mB0), bfloat>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) dOut[i] = static_cast<bfloat>(IS_NVFP4 ? (cT[i] * oscale) : cT[i]);
    auto mC = tC.slice<BN, BM>(0, 0);
    dOut.store(mC);
  } else {
    auto a_probe = tA.slice(0, 0);
    auto cT = op.template get_destination_cooperative_tensor<decltype(a_probe), decltype(mB0), float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) cT[i] = 0.0f;

    for (int k = 0; k < INTER; k += 64) {
      prefetch_b(k + 32);
      auto mA0 = tA.slice(k, 0);
      op.run(mA0, mB0, cT);
      store_b(1);
      threadgroup_barrier(mem_flags::mem_threadgroup);

      auto mA1 = tA.slice(k + 32, 0);
      if (k + 64 < INTER) {
        prefetch_b(k + 64);
        op.run(mA1, mB1, cT);
        store_b(0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
      } else {
        op.run(mA1, mB1, cT);
      }
    }

    auto dOut = op.template get_destination_cooperative_tensor<decltype(a_probe), decltype(mB0), bfloat>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) dOut[i] = static_cast<bfloat>(IS_NVFP4 ? (cT[i] * oscale) : cT[i]);
    auto mC = tC.slice(0, 0);
    dOut.store(mC);
  }
}

#define INST_W13(HNAME, KNAME, ARG) \
template [[host_name(#HNAME)]] kernel void KNAME<ARG>( \
    device const bfloat*, device const uint*, device const uint8_t*, device const float*, \
    device const int*, device const uint*, device const uint*, device const uint*, \
    device bfloat*, constant MoeParams&, uint2, uint);

#define INST_W2(HNAME, KNAME, ARG) \
template [[host_name(#HNAME)]] kernel void KNAME<ARG>( \
    device const bfloat*, device const uint*, device const uint8_t*, device const float*, \
    device const int*, device const uint*, device const uint*, \
    device bfloat*, constant MoeParams&, uint2, uint);

INST_W13(nax_grp_w13_nvfp4,        w13_64x64x32_db_k,   true)
INST_W13(nax_grp_w13_mxfp8,        w13_64x64x32_db_k,   false)
INST_W2 (nax_grp_w2_nvfp4,         w2_64x64x32_db_k,    true)
INST_W2 (nax_grp_w2_mxfp8,         w2_64x64x32_db_k,    false)

INST_W13(nax_grp_w13_nvfp4_128x64, w13_128x64x32_sg8_k, true)
INST_W13(nax_grp_w13_mxfp8_128x64, w13_128x64x32_sg8_k, false)
INST_W2 (nax_grp_w2_nvfp4_128x64,  w2_128x64x32_sg8_k,  true)
INST_W2 (nax_grp_w2_mxfp8_128x64,  w2_128x64x32_sg8_k,  false)
)";

static const char* kCombineMSL = R"(
#include <metal_stdlib>
using namespace metal;

struct CombineParams {
  int B;
  int L;
  int D;
  int TOP_K;
  int has_mask;
  int _pad;
};

kernel void fused_unsort_combine_sconv_resid_64ch(
    device const bfloat* ys_sorted [[buffer(0)]],
    device const uint* inv_order [[buffer(1)]],
    device const float* rweights [[buffer(2)]],
    device const float* state [[buffer(3)]],
    device const float* taps [[buffer(4)]],
    device const float* resid [[buffer(5)]],
    device const uint8_t* mask [[buffer(6)]],
    device bfloat* out_x [[buffer(7)]],
    device float* new_state [[buffer(8)]],
    constant CombineParams& p [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 ltid [[thread_position_in_threadgroup]]) {
  const int lx = int(ltid.x);
  const int ly = int(ltid.y);
  const int d0 = int(tgid.x) * 64 + lx * 2;
  const int t0 = int(tgid.y) * 64;
  const int batch = int(tgid.z);
  const int D = p.D;
  const int L = p.L;
  if (d0 >= D) return;

  threadgroup float y_pre[3][64];
  threadgroup bfloat y_tile[64][64];

  const int row_base = batch * L;
  for (int r = ly; r < 67; r += 8) {
    const int t = t0 + r - 3;
    float v0 = 0.0f;
    float v1 = 0.0f;
    if (t < 0) {
      const float2 st = *(const device float2*)(state + (size_t(batch) * 3 + size_t(t + 3)) * D + d0);
      v0 = st.x;
      v1 = st.y;
    } else if (t < L) {
      const int gtok = row_base + t;
      const bool valid = (p.has_mask == 0) || (mask[gtok] != 0);
      if (valid) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        const size_t wbase = size_t(gtok) * size_t(p.TOP_K);
        if (p.TOP_K == 8) {
          const uint4 sp0 = *(const device uint4*)(inv_order + wbase);
          const uint4 sp1 = *(const device uint4*)(inv_order + wbase + 4);
          const float4 rw0 = *(const device float4*)(rweights + wbase);
          const float4 rw1 = *(const device float4*)(rweights + wbase + 4);
          const vec<bfloat, 2> y0 = *(const device vec<bfloat, 2>*)(ys_sorted + size_t(sp0.x) * D + d0);
          const vec<bfloat, 2> y1 = *(const device vec<bfloat, 2>*)(ys_sorted + size_t(sp0.y) * D + d0);
          const vec<bfloat, 2> y2 = *(const device vec<bfloat, 2>*)(ys_sorted + size_t(sp0.z) * D + d0);
          const vec<bfloat, 2> y3 = *(const device vec<bfloat, 2>*)(ys_sorted + size_t(sp0.w) * D + d0);
          const vec<bfloat, 2> y4 = *(const device vec<bfloat, 2>*)(ys_sorted + size_t(sp1.x) * D + d0);
          const vec<bfloat, 2> y5 = *(const device vec<bfloat, 2>*)(ys_sorted + size_t(sp1.y) * D + d0);
          const vec<bfloat, 2> y6 = *(const device vec<bfloat, 2>*)(ys_sorted + size_t(sp1.z) * D + d0);
          const vec<bfloat, 2> y7 = *(const device vec<bfloat, 2>*)(ys_sorted + size_t(sp1.w) * D + d0);
          acc0 += float(y0[0]) * rw0.x; acc1 += float(y0[1]) * rw0.x;
          acc0 += float(y1[0]) * rw0.y; acc1 += float(y1[1]) * rw0.y;
          acc0 += float(y2[0]) * rw0.z; acc1 += float(y2[1]) * rw0.z;
          acc0 += float(y3[0]) * rw0.w; acc1 += float(y3[1]) * rw0.w;
          acc0 += float(y4[0]) * rw1.x; acc1 += float(y4[1]) * rw1.x;
          acc0 += float(y5[0]) * rw1.y; acc1 += float(y5[1]) * rw1.y;
          acc0 += float(y6[0]) * rw1.z; acc1 += float(y6[1]) * rw1.z;
          acc0 += float(y7[0]) * rw1.w; acc1 += float(y7[1]) * rw1.w;
        } else {
          for (int k = 0; k < p.TOP_K; ++k) {
            const uint sp = inv_order[wbase + k];
            const float rw = rweights[wbase + k];
            const vec<bfloat, 2> yk = *(const device vec<bfloat, 2>*)(ys_sorted + size_t(sp) * D + d0);
            acc0 += float(yk[0]) * rw;
            acc1 += float(yk[1]) * rw;
          }
        }
        v0 = float(bfloat(acc0));
        v1 = float(bfloat(acc1));
      }
    }
    if (r < 3) {
      y_pre[r][lx * 2] = v0;
      y_pre[r][lx * 2 + 1] = v1;
    } else {
      *(threadgroup vec<bfloat, 2>*)(&y_tile[r - 3][lx * 2]) = vec<bfloat, 2>(bfloat(v0), bfloat(v1));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float4 tw0 = *(const device float4*)(taps + size_t(d0) * 4);
  const float4 tw1 = *(const device float4*)(taps + size_t(d0 + 1) * 4);

  auto load_y = [&](int idx) -> float2 {
    if (idx < 3) {
      return float2(y_pre[idx][lx * 2], y_pre[idx][lx * 2 + 1]);
    }
    const vec<bfloat, 2> b2 = *(const threadgroup vec<bfloat, 2>*)(&y_tile[idx - 3][lx * 2]);
    return float2(float(b2[0]), float(b2[1]));
  };

  for (int r = ly; r < 64; r += 8) {
    const int t = t0 + r;
    if (t < L) {
      const int gtok = row_base + t;
      const float2 u0 = load_y(r);
      const float2 u1 = load_y(r + 1);
      const float2 u2 = load_y(r + 2);
      const float2 u3 = load_y(r + 3);
      const float sc0 = (tw0.x * u0.x + tw0.y * u1.x + tw0.z * u2.x + tw0.w * u3.x + u3.x) * 0.4472135955f;
      const float sc1 = (tw1.x * u0.y + tw1.y * u1.y + tw1.z * u2.y + tw1.w * u3.y + u3.y) * 0.4472135955f;
      const float2 res2 = *(const device float2*)(resid + size_t(gtok) * D + d0);
      *(device vec<bfloat, 2>*)(out_x + size_t(gtok) * D + d0) = vec<bfloat, 2>(
          bfloat(res2.x + float(bfloat(sc0))),
          bfloat(res2.y + float(bfloat(sc1)))
      );
      if (t >= L - 3) {
        *(device float2*)(new_state + (size_t(batch) * 3 + size_t(t - (L - 3))) * D + d0) = u3;
      }
    }
  }

  if (L < 3 && t0 == 0 && ly < (3 - L)) {
    const int s_dst = ly;
    *(device float2*)(new_state + (size_t(batch) * 3 + size_t(s_dst)) * D + d0) =
        *(const device float2*)(state + (size_t(batch) * 3 + size_t(s_dst + L)) * D + d0);
  }
}
)";

struct VariantConfig {
  id<MTLComputePipelineState> pso_w13;
  id<MTLComputePipelineState> pso_w2;
  int bm;
  int bn;
  int sgs;
  int grid_order;
};

static VariantConfig select_variant(int mode, int variant, int avg_cnt) {
  const bool is_nvfp4 = (mode == 0);
  int eff = variant;
  if (eff < 0) {
    eff = (!is_nvfp4 && avg_cnt >= 128) ? 17 : 16;
  }
  const int grid_order = (eff >= 16) ? 1 : 0;
  const int v = eff & 15;
  if (v == 1) {
    return {
        is_nvfp4 ? g_pso_w13_nvfp4_128x64 : g_pso_w13_mxfp8_128x64,
        is_nvfp4 ? g_pso_w2_nvfp4_128x64 : g_pso_w2_mxfp8_128x64,
        128, 64, 8, grid_order};
  }
  return {
      is_nvfp4 ? g_pso_w13_nvfp4_64x64 : g_pso_w13_mxfp8_64x64,
      is_nvfp4 ? g_pso_w2_nvfp4_64x64 : g_pso_w2_mxfp8_64x64,
      64, 64, 4, grid_order};
}

}

extern "C" {

int prefill_moe_init(void) {
  @autoreleasepool {
    if (g_device != nil && g_pso_bucket != nil) return 0;
    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) {
      std::fprintf(stderr, "[prefill_moe_nax] MTLCreateSystemDefaultDevice() failed\n");
      return -1;
    }
    g_queue = [g_device newCommandQueue];
    if (!g_queue) {
      std::fprintf(stderr, "[prefill_moe_nax] newCommandQueue() failed\n");
      return -2;
    }

    NSError* err = nil;
    MTLCompileOptions* opts = [MTLCompileOptions new];
    opts.languageVersion = MTLLanguageVersion4_0;
    id<MTLLibrary> lib = [g_device newLibraryWithSource:[NSString stringWithUTF8String:kPrefillMoeMSL]
                                                options:opts
                                                  error:&err];
    if (!lib) {
      std::fprintf(stderr, "[prefill_moe_nax] Metal 4.0 MSL compile failed:\n%s\n",
                   err ? [[err localizedDescription] UTF8String] : "unknown");
      return -3;
    }

    MTLCompileOptions* opts_comb = [MTLCompileOptions new];
    opts_comb.languageVersion = MTLLanguageVersion3_2;
    opts_comb.mathMode = MTLMathModeSafe;
    id<MTLLibrary> lib_comb = [g_device newLibraryWithSource:[NSString stringWithUTF8String:kCombineMSL]
                                                     options:opts_comb
                                                       error:&err];
    if (!lib_comb) {
      std::fprintf(stderr, "[prefill_moe_nax] Combine MSL compile failed:\n%s\n",
                   err ? [[err localizedDescription] UTF8String] : "unknown");
      return -3;
    }

    g_pso_bucket = compile_pso(lib, "bucket_experts_with_inv");
    g_pso_fused_combine = compile_pso(lib_comb, "fused_unsort_combine_sconv_resid_64ch");
    g_pso_w13_nvfp4_64x64 = compile_pso(lib, "nax_grp_w13_nvfp4");
    g_pso_w2_nvfp4_64x64 = compile_pso(lib, "nax_grp_w2_nvfp4");
    g_pso_w13_mxfp8_64x64 = compile_pso(lib, "nax_grp_w13_mxfp8");
    g_pso_w2_mxfp8_64x64 = compile_pso(lib, "nax_grp_w2_mxfp8");

    g_pso_w13_nvfp4_128x64 = compile_pso(lib, "nax_grp_w13_nvfp4_128x64");
    g_pso_w2_nvfp4_128x64 = compile_pso(lib, "nax_grp_w2_nvfp4_128x64");
    g_pso_w13_mxfp8_128x64 = compile_pso(lib, "nax_grp_w13_mxfp8_128x64");
    g_pso_w2_mxfp8_128x64 = compile_pso(lib, "nax_grp_w2_mxfp8_128x64");

    g_scratch_indirect = [g_device newBufferWithLength:256 options:MTLResourceStorageModeShared];

    if (!g_pso_bucket || !g_pso_fused_combine || !g_pso_w13_nvfp4_64x64 || !g_pso_w2_nvfp4_64x64 ||
        !g_pso_w13_mxfp8_64x64 || !g_pso_w2_mxfp8_64x64 ||
        !g_pso_w13_nvfp4_128x64 || !g_pso_w2_nvfp4_128x64 ||
        !g_pso_w13_mxfp8_128x64 || !g_pso_w2_mxfp8_128x64 || !g_scratch_indirect) {
      return -4;
    }
    return 0;
  }
}

int prefill_moe_register_buffer(uintptr_t ptr, size_t nbytes) {
  @autoreleasepool {
    if (prefill_moe_init() != 0) return -1;
    BufSlice s = wrap_ptr(ptr, nbytes, true);
    return (s.buf != nil) ? 0 : -2;
  }
}

void prefill_moe_unregister_buffer(uintptr_t ptr) {
  @autoreleasepool {
    const size_t page_sz = vm_page_size;
    const uintptr_t base = ptr & ~(page_sz - 1);
    std::lock_guard<std::mutex> lk(g_cache_mu);
    g_buf_cache.erase(base);
  }
}

void prefill_moe_clear_cache(void) {
  @autoreleasepool {
    std::lock_guard<std::mutex> lk(g_cache_mu);
    g_buf_cache.clear();
  }
}

void prefill_moe_release_scratch(void) {
  @autoreleasepool {
    g_scratch_h_sorted = nil;
    g_scratch_h_bytes = 0;
    g_scratch_y_sorted = nil;
    g_scratch_y_bytes = 0;
    g_scratch_offsets = nil;
    g_scratch_offsets_bytes = 0;
    g_scratch_sorted_slots = nil;
    g_scratch_sorted_slots_bytes = 0;
    g_scratch_inv_order = nil;
    g_scratch_inv_order_bytes = 0;
    g_scratch_tile_expert = nil;
    g_scratch_tile_moff = nil;
    g_scratch_tile_bytes = 0;
  }
}

int prefill_moe_bucket(
    int T, int E,
    uintptr_t inds_ptr,
    uintptr_t offsets_ptr,
    uintptr_t sorted_slots_ptr,
    uintptr_t inv_order_ptr) {
  @autoreleasepool {
    if (prefill_moe_init() != 0) return -1;
    if (E <= 0 || E > 256 || T < 0) return -2;

    const size_t max_tiles = static_cast<size_t>(T / 64 + E + 16);
    if (!ensure_scratch(g_scratch_tile_expert, g_scratch_tile_bytes, max_tiles * sizeof(uint32_t))) return -3;
    if (g_scratch_tile_moff == nil || [g_scratch_tile_moff length] < max_tiles * sizeof(uint32_t)) {
      g_scratch_tile_moff = [g_device newBufferWithLength:g_scratch_tile_bytes options:MTLResourceStorageModeShared];
      if (!g_scratch_tile_moff) return -3;
    }

    BufSlice s_inds = wrap_ptr(inds_ptr, static_cast<size_t>(T) * sizeof(uint32_t), false);
    BufSlice s_off = wrap_ptr(offsets_ptr, static_cast<size_t>(E + 1) * sizeof(int32_t), false);
    BufSlice s_slots = wrap_ptr(sorted_slots_ptr, static_cast<size_t>(T) * sizeof(uint32_t), false);
    BufSlice s_inv = wrap_ptr(inv_order_ptr, static_cast<size_t>(T) * sizeof(uint32_t), false);
    if (!s_inds.buf || !s_off.buf || !s_slots.buf || !s_inv.buf) return -4;

    BucketParamsHost bp{T, E, 64, 1, 1, 0};

    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:g_pso_bucket];
    [enc setBuffer:s_inds.buf offset:s_inds.offset atIndex:0];
    [enc setBuffer:s_off.buf offset:s_off.offset atIndex:1];
    [enc setBuffer:s_slots.buf offset:s_slots.offset atIndex:2];
    [enc setBuffer:s_inv.buf offset:s_inv.offset atIndex:3];
    [enc setBuffer:g_scratch_tile_expert offset:0 atIndex:4];
    [enc setBuffer:g_scratch_tile_moff offset:0 atIndex:5];
    [enc setBuffer:g_scratch_indirect offset:0 atIndex:6];
    [enc setBytes:&bp length:sizeof(bp) atIndex:7];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    return ([cb status] == MTLCommandBufferStatusCompleted) ? 0 : -5;
  }
}

static int run_forward_impl(
    const MoeForwardArgs* a,
    int reps,
    int variant,
    double* out_gpu_ms,
    double* out_w13_ms,
    double* out_w2_ms) {
  @autoreleasepool {
    if (!a) return -10;
    if (prefill_moe_init() != 0) return -1;
    const int mode = a->mode;
    const int M = a->M, K = a->K, INTER = a->INTER, D = a->D, E = a->E, TOP_K = a->TOP_K;
    if (E <= 0 || E > 256 || M <= 0 || TOP_K <= 0) return -2;
    if ((K % 64) != 0 || (INTER % 64) != 0 || (D % 64) != 0) return -3;

    const int T = M * TOP_K;
    VariantConfig vc = select_variant(mode, variant, T / E);
    if (!vc.pso_w13 || !vc.pso_w2) return -4;

    const size_t max_tiles = static_cast<size_t>(T / vc.bm + E + 16);

    if (!ensure_scratch(g_scratch_h_sorted, g_scratch_h_bytes, static_cast<size_t>(T) * INTER * 2,
                        MTLResourceStorageModePrivate)) return -5;
    if (!ensure_scratch(g_scratch_offsets, g_scratch_offsets_bytes, static_cast<size_t>(E + 1) * sizeof(int32_t))) return -5;
    if (!ensure_scratch(g_scratch_sorted_slots, g_scratch_sorted_slots_bytes, static_cast<size_t>(T) * sizeof(uint32_t))) return -5;
    if (!ensure_scratch(g_scratch_inv_order, g_scratch_inv_order_bytes, static_cast<size_t>(T) * sizeof(uint32_t))) return -5;
    if (!ensure_scratch(g_scratch_tile_expert, g_scratch_tile_bytes, max_tiles * sizeof(uint32_t))) return -5;
    if (g_scratch_tile_moff == nil || [g_scratch_tile_moff length] < max_tiles * sizeof(uint32_t)) {
      g_scratch_tile_moff = [g_device newBufferWithLength:g_scratch_tile_bytes options:MTLResourceStorageModeShared];
      if (!g_scratch_tile_moff) return -5;
    }

    const bool is_nvfp4 = (mode == 0);
    const size_t w13_bytes = static_cast<size_t>(E) * (2 * INTER) * (is_nvfp4 ? (K / 8) : (K / 4)) * sizeof(uint32_t);
    const size_t s13_bytes = static_cast<size_t>(E) * (2 * INTER) * (is_nvfp4 ? (K / 16) : (K / 32));
    const size_t w2_bytes = static_cast<size_t>(E) * D * (is_nvfp4 ? (INTER / 8) : (INTER / 4)) * sizeof(uint32_t);
    const size_t s2_bytes = static_cast<size_t>(E) * D * (is_nvfp4 ? (INTER / 16) : (INTER / 32));
    const size_t gs_bytes = static_cast<size_t>(E) * sizeof(float);

    BufSlice b_x = wrap_ptr(a->x_ptr, static_cast<size_t>(M) * K * 2, false);
    BufSlice b_inds = wrap_ptr(a->inds_ptr, static_cast<size_t>(T) * sizeof(uint32_t), false);
    BufSlice b_w13 = wrap_ptr(a->w13_ptr, w13_bytes, true);
    BufSlice b_s13 = wrap_ptr(a->s13_ptr, s13_bytes, true);
    BufSlice b_gs13 = is_nvfp4 ? wrap_ptr(a->gs13_ptr, gs_bytes, true) : BufSlice{g_scratch_indirect, 0};
    BufSlice b_w2 = wrap_ptr(a->w2_ptr, w2_bytes, true);
    BufSlice b_s2 = wrap_ptr(a->s2_ptr, s2_bytes, true);
    BufSlice b_gs2 = is_nvfp4 ? wrap_ptr(a->gs2_ptr, gs_bytes, true) : BufSlice{g_scratch_indirect, 0};
    BufSlice b_y = wrap_ptr(a->y_sorted_ptr, static_cast<size_t>(T) * D * 2, false);

    BufSlice b_inv = a->inv_order_ptr ? wrap_ptr(a->inv_order_ptr, static_cast<size_t>(T) * sizeof(uint32_t), false)
                                      : BufSlice{g_scratch_inv_order, 0};
    BufSlice b_off = a->offsets_ptr ? wrap_ptr(a->offsets_ptr, static_cast<size_t>(E + 1) * sizeof(int32_t), false)
                                    : BufSlice{g_scratch_offsets, 0};
    BufSlice b_slots = a->sorted_slots_ptr ? wrap_ptr(a->sorted_slots_ptr, static_cast<size_t>(T) * sizeof(uint32_t), false)
                                           : BufSlice{g_scratch_sorted_slots, 0};

    if (!b_x.buf || !b_inds.buf || !b_w13.buf || !b_s13.buf || !b_gs13.buf ||
        !b_w2.buf || !b_s2.buf || !b_gs2.buf || !b_y.buf || !b_inv.buf || !b_off.buf || !b_slots.buf) {
      return -6;
    }

    BucketParamsHost bp{T, E, vc.bm, (2 * INTER) / vc.bn, D / vc.bn, vc.grid_order};
    MoeParamsHost mp{M, K, INTER, D, E, TOP_K, vc.grid_order, 0, a->gate_cap, a->up_cap};

    const NSUInteger simd_w = [vc.pso_w13 threadExecutionWidth];
    const MTLSize tg_gemm = MTLSizeMake(simd_w * vc.sgs, 1, 1);

    {
      id<MTLCommandBuffer> cb = [g_queue commandBuffer];

      id<MTLComputeCommandEncoder> enc_b = [cb computeCommandEncoder];
      [enc_b setComputePipelineState:g_pso_bucket];
      [enc_b setBuffer:b_inds.buf offset:b_inds.offset atIndex:0];
      [enc_b setBuffer:b_off.buf offset:b_off.offset atIndex:1];
      [enc_b setBuffer:b_slots.buf offset:b_slots.offset atIndex:2];
      [enc_b setBuffer:b_inv.buf offset:b_inv.offset atIndex:3];
      [enc_b setBuffer:g_scratch_tile_expert offset:0 atIndex:4];
      [enc_b setBuffer:g_scratch_tile_moff offset:0 atIndex:5];
      [enc_b setBuffer:g_scratch_indirect offset:0 atIndex:6];
      [enc_b setBytes:&bp length:sizeof(bp) atIndex:7];
      [enc_b dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [enc_b endEncoding];

      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
      for (int r = 0; r < reps; ++r) {
        [enc setComputePipelineState:vc.pso_w13];
        [enc setBuffer:b_x.buf offset:b_x.offset atIndex:0];
        [enc setBuffer:b_w13.buf offset:b_w13.offset atIndex:1];
        [enc setBuffer:b_s13.buf offset:b_s13.offset atIndex:2];
        [enc setBuffer:b_gs13.buf offset:b_gs13.offset atIndex:3];
        [enc setBuffer:b_off.buf offset:b_off.offset atIndex:4];
        [enc setBuffer:b_slots.buf offset:b_slots.offset atIndex:5];
        [enc setBuffer:g_scratch_tile_expert offset:0 atIndex:6];
        [enc setBuffer:g_scratch_tile_moff offset:0 atIndex:7];
        [enc setBuffer:g_scratch_h_sorted offset:0 atIndex:8];
        [enc setBytes:&mp length:sizeof(mp) atIndex:9];
        [enc dispatchThreadgroupsWithIndirectBuffer:g_scratch_indirect
                               indirectBufferOffset:0
                              threadsPerThreadgroup:tg_gemm];

        [enc setComputePipelineState:vc.pso_w2];
        [enc setBuffer:g_scratch_h_sorted offset:0 atIndex:0];
        [enc setBuffer:b_w2.buf offset:b_w2.offset atIndex:1];
        [enc setBuffer:b_s2.buf offset:b_s2.offset atIndex:2];
        [enc setBuffer:b_gs2.buf offset:b_gs2.offset atIndex:3];
        [enc setBuffer:b_off.buf offset:b_off.offset atIndex:4];
        [enc setBuffer:g_scratch_tile_expert offset:0 atIndex:5];
        [enc setBuffer:g_scratch_tile_moff offset:0 atIndex:6];
        [enc setBuffer:b_y.buf offset:b_y.offset atIndex:7];
        [enc setBytes:&mp length:sizeof(mp) atIndex:8];
        [enc dispatchThreadgroupsWithIndirectBuffer:g_scratch_indirect
                               indirectBufferOffset:3 * sizeof(uint32_t)
                              threadsPerThreadgroup:tg_gemm];
      }
      [enc endEncoding];

      [cb commit];
      [cb waitUntilCompleted];
      if ([cb status] != MTLCommandBufferStatusCompleted) {
        std::fprintf(stderr, "[prefill_moe_nax] CommandBuffer failed: status=%ld\n", (long)[cb status]);
        return -7;
      }
      if (out_gpu_ms) {
        *out_gpu_ms = (([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0) / reps;
      }
    }

    if (out_w13_ms) {
      id<MTLCommandBuffer> cb = [g_queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
      [enc setComputePipelineState:vc.pso_w13];
      [enc setBuffer:b_x.buf offset:b_x.offset atIndex:0];
      [enc setBuffer:b_w13.buf offset:b_w13.offset atIndex:1];
      [enc setBuffer:b_s13.buf offset:b_s13.offset atIndex:2];
      [enc setBuffer:b_gs13.buf offset:b_gs13.offset atIndex:3];
      [enc setBuffer:b_off.buf offset:b_off.offset atIndex:4];
      [enc setBuffer:b_slots.buf offset:b_slots.offset atIndex:5];
      [enc setBuffer:g_scratch_tile_expert offset:0 atIndex:6];
      [enc setBuffer:g_scratch_tile_moff offset:0 atIndex:7];
      [enc setBuffer:g_scratch_h_sorted offset:0 atIndex:8];
      [enc setBytes:&mp length:sizeof(mp) atIndex:9];
      for (int r = 0; r < reps; ++r) {
        [enc dispatchThreadgroupsWithIndirectBuffer:g_scratch_indirect
                               indirectBufferOffset:0
                              threadsPerThreadgroup:tg_gemm];
      }
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      *out_w13_ms = (([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0) / reps;
    }

    if (out_w2_ms) {
      id<MTLCommandBuffer> cb = [g_queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
      [enc setComputePipelineState:vc.pso_w2];
      [enc setBuffer:g_scratch_h_sorted offset:0 atIndex:0];
      [enc setBuffer:b_w2.buf offset:b_w2.offset atIndex:1];
      [enc setBuffer:b_s2.buf offset:b_s2.offset atIndex:2];
      [enc setBuffer:b_gs2.buf offset:b_gs2.offset atIndex:3];
      [enc setBuffer:b_off.buf offset:b_off.offset atIndex:4];
      [enc setBuffer:g_scratch_tile_expert offset:0 atIndex:5];
      [enc setBuffer:g_scratch_tile_moff offset:0 atIndex:6];
      [enc setBuffer:b_y.buf offset:b_y.offset atIndex:7];
      [enc setBytes:&mp length:sizeof(mp) atIndex:8];
      for (int r = 0; r < reps; ++r) {
        [enc dispatchThreadgroupsWithIndirectBuffer:g_scratch_indirect
                               indirectBufferOffset:3 * sizeof(uint32_t)
                              threadsPerThreadgroup:tg_gemm];
      }
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      *out_w2_ms = (([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0) / reps;
    }

    return 0;
  }
}

int prefill_moe_forward(const MoeForwardArgs* a, double* out_gpu_ms) {
  return run_forward_impl(a, 1, -1, out_gpu_ms, nullptr, nullptr);
}

int prefill_moe_forward_bench(
    const MoeForwardArgs* a,
    int reps,
    int variant,
    double* out_gpu_ms,
    double* out_w13_ms,
    double* out_w2_ms) {
  return run_forward_impl(a, std::max(1, reps), variant, out_gpu_ms, out_w13_ms, out_w2_ms);
}

static int run_fused_combine_impl(
    const MoeFusedCombineArgs* a,
    int reps,
    int variant,
    double* out_gpu_ms,
    double* out_combine_ms) {
  @autoreleasepool {
    if (!a) return -10;
    if (prefill_moe_init() != 0) return -1;
    const int mode = a->mode;
    const int B = a->B, L = a->L, K = a->K, INTER = a->INTER, D = a->D, E = a->E, TOP_K = a->TOP_K;
    if (E <= 0 || E > 256 || B <= 0 || L <= 0 || TOP_K <= 0) return -2;
    if ((K % 64) != 0 || (INTER % 64) != 0 || (D % 64) != 0) return -3;

    const int M = B * L;
    const int T = M * TOP_K;
    VariantConfig vc = select_variant(mode, variant, T / E);
    if (!vc.pso_w13 || !vc.pso_w2 || !g_pso_fused_combine) return -4;

    const size_t max_tiles = static_cast<size_t>(T / vc.bm + E + 16);

    if (!ensure_scratch(g_scratch_h_sorted, g_scratch_h_bytes, static_cast<size_t>(T) * INTER * 2,
                        MTLResourceStorageModePrivate)) return -5;
    if (a->y_sorted_ptr == 0) {
      if (!ensure_scratch(g_scratch_y_sorted, g_scratch_y_bytes, static_cast<size_t>(T) * D * 2,
                          MTLResourceStorageModePrivate)) return -5;
    }
    if (!ensure_scratch(g_scratch_offsets, g_scratch_offsets_bytes, static_cast<size_t>(E + 1) * sizeof(int32_t))) return -5;
    if (!ensure_scratch(g_scratch_sorted_slots, g_scratch_sorted_slots_bytes, static_cast<size_t>(T) * sizeof(uint32_t))) return -5;
    if (!ensure_scratch(g_scratch_inv_order, g_scratch_inv_order_bytes, static_cast<size_t>(T) * sizeof(uint32_t))) return -5;
    if (!ensure_scratch(g_scratch_tile_expert, g_scratch_tile_bytes, max_tiles * sizeof(uint32_t))) return -5;
    if (g_scratch_tile_moff == nil || [g_scratch_tile_moff length] < max_tiles * sizeof(uint32_t)) {
      g_scratch_tile_moff = [g_device newBufferWithLength:g_scratch_tile_bytes options:MTLResourceStorageModeShared];
      if (!g_scratch_tile_moff) return -5;
    }

    const bool is_nvfp4 = (mode == 0);
    const size_t w13_bytes = static_cast<size_t>(E) * (2 * INTER) * (is_nvfp4 ? (K / 8) : (K / 4)) * sizeof(uint32_t);
    const size_t s13_bytes = static_cast<size_t>(E) * (2 * INTER) * (is_nvfp4 ? (K / 16) : (K / 32));
    const size_t w2_bytes = static_cast<size_t>(E) * D * (is_nvfp4 ? (INTER / 8) : (INTER / 4)) * sizeof(uint32_t);
    const size_t s2_bytes = static_cast<size_t>(E) * D * (is_nvfp4 ? (INTER / 16) : (INTER / 32));
    const size_t gs_bytes = static_cast<size_t>(E) * sizeof(float);
    const size_t taps_bytes = static_cast<size_t>(D) * 4 * sizeof(float);

    BufSlice b_w13 = wrap_ptr(a->w13_ptr, w13_bytes, true);
    BufSlice b_s13 = wrap_ptr(a->s13_ptr, s13_bytes, true);
    BufSlice b_gs13 = is_nvfp4 ? wrap_ptr(a->gs13_ptr, gs_bytes, true) : BufSlice{g_scratch_indirect, 0};
    BufSlice b_w2 = wrap_ptr(a->w2_ptr, w2_bytes, true);
    BufSlice b_s2 = wrap_ptr(a->s2_ptr, s2_bytes, true);
    BufSlice b_gs2 = is_nvfp4 ? wrap_ptr(a->gs2_ptr, gs_bytes, true) : BufSlice{g_scratch_indirect, 0};
    BufSlice b_taps = wrap_ptr(a->taps_ptr, taps_bytes, true);

    BufSlice b_x = wrap_ptr(a->x_ptr, static_cast<size_t>(M) * K * 2, false);
    BufSlice b_inds = wrap_ptr(a->inds_ptr, static_cast<size_t>(T) * sizeof(uint32_t), false);
    BufSlice b_rw = wrap_ptr(a->rweights_ptr, static_cast<size_t>(T) * sizeof(float), false);
    BufSlice b_state = wrap_ptr(a->state_ptr, static_cast<size_t>(B) * 3 * D * sizeof(float), false);
    BufSlice b_resid = wrap_ptr(a->resid_ptr, static_cast<size_t>(M) * D * sizeof(float), false);
    BufSlice b_mask = (a->has_mask && a->mask_ptr)
        ? wrap_ptr(a->mask_ptr, static_cast<size_t>(M), false)
        : BufSlice{g_scratch_indirect, 0};
    BufSlice b_out_x = wrap_ptr(a->out_x_ptr, static_cast<size_t>(M) * D * 2, false);
    BufSlice b_new_state = wrap_ptr(a->new_state_ptr, static_cast<size_t>(B) * 3 * D * sizeof(float), false);

    BufSlice b_y = a->y_sorted_ptr
        ? wrap_ptr(a->y_sorted_ptr, static_cast<size_t>(T) * D * 2, false)
        : BufSlice{g_scratch_y_sorted, 0};
    BufSlice b_inv = a->inv_order_ptr
        ? wrap_ptr(a->inv_order_ptr, static_cast<size_t>(T) * sizeof(uint32_t), false)
        : BufSlice{g_scratch_inv_order, 0};

    if (!b_x.buf || !b_inds.buf || !b_w13.buf || !b_s13.buf || !b_gs13.buf ||
        !b_w2.buf || !b_s2.buf || !b_gs2.buf || !b_taps.buf ||
        !b_rw.buf || !b_state.buf || !b_resid.buf || !b_mask.buf ||
        !b_out_x.buf || !b_new_state.buf || !b_y.buf || !b_inv.buf) {
      return -6;
    }

    BucketParamsHost bp{T, E, vc.bm, (2 * INTER) / vc.bn, D / vc.bn, vc.grid_order};
    MoeParamsHost mp{M, K, INTER, D, E, TOP_K, vc.grid_order, 0, a->gate_cap, a->up_cap};
    CombineParamsHost cp{B, L, D, TOP_K, (a->has_mask && a->mask_ptr) ? 1 : 0, 0};

    const NSUInteger simd_w = [vc.pso_w13 threadExecutionWidth];
    const MTLSize tg_gemm = MTLSizeMake(simd_w * vc.sgs, 1, 1);
    const MTLSize grid_comb = MTLSizeMake(
        static_cast<NSUInteger>((D + 63) / 64),
        static_cast<NSUInteger>((L + 63) / 64),
        static_cast<NSUInteger>(B));
    const MTLSize tg_comb = MTLSizeMake(32, 8, 1);

    {
      id<MTLCommandBuffer> cb = [g_queue commandBuffer];

      id<MTLComputeCommandEncoder> enc_b = [cb computeCommandEncoder];
      [enc_b setComputePipelineState:g_pso_bucket];
      [enc_b setBuffer:b_inds.buf offset:b_inds.offset atIndex:0];
      [enc_b setBuffer:g_scratch_offsets offset:0 atIndex:1];
      [enc_b setBuffer:g_scratch_sorted_slots offset:0 atIndex:2];
      [enc_b setBuffer:b_inv.buf offset:b_inv.offset atIndex:3];
      [enc_b setBuffer:g_scratch_tile_expert offset:0 atIndex:4];
      [enc_b setBuffer:g_scratch_tile_moff offset:0 atIndex:5];
      [enc_b setBuffer:g_scratch_indirect offset:0 atIndex:6];
      [enc_b setBytes:&bp length:sizeof(bp) atIndex:7];
      [enc_b dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [enc_b endEncoding];

      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
      for (int r = 0; r < reps; ++r) {
        [enc setComputePipelineState:vc.pso_w13];
        [enc setBuffer:b_x.buf offset:b_x.offset atIndex:0];
        [enc setBuffer:b_w13.buf offset:b_w13.offset atIndex:1];
        [enc setBuffer:b_s13.buf offset:b_s13.offset atIndex:2];
        [enc setBuffer:b_gs13.buf offset:b_gs13.offset atIndex:3];
        [enc setBuffer:g_scratch_offsets offset:0 atIndex:4];
        [enc setBuffer:g_scratch_sorted_slots offset:0 atIndex:5];
        [enc setBuffer:g_scratch_tile_expert offset:0 atIndex:6];
        [enc setBuffer:g_scratch_tile_moff offset:0 atIndex:7];
        [enc setBuffer:g_scratch_h_sorted offset:0 atIndex:8];
        [enc setBytes:&mp length:sizeof(mp) atIndex:9];
        [enc dispatchThreadgroupsWithIndirectBuffer:g_scratch_indirect
                               indirectBufferOffset:0
                              threadsPerThreadgroup:tg_gemm];

        [enc setComputePipelineState:vc.pso_w2];
        [enc setBuffer:g_scratch_h_sorted offset:0 atIndex:0];
        [enc setBuffer:b_w2.buf offset:b_w2.offset atIndex:1];
        [enc setBuffer:b_s2.buf offset:b_s2.offset atIndex:2];
        [enc setBuffer:b_gs2.buf offset:b_gs2.offset atIndex:3];
        [enc setBuffer:g_scratch_offsets offset:0 atIndex:4];
        [enc setBuffer:g_scratch_tile_expert offset:0 atIndex:5];
        [enc setBuffer:g_scratch_tile_moff offset:0 atIndex:6];
        [enc setBuffer:b_y.buf offset:b_y.offset atIndex:7];
        [enc setBytes:&mp length:sizeof(mp) atIndex:8];
        [enc dispatchThreadgroupsWithIndirectBuffer:g_scratch_indirect
                               indirectBufferOffset:3 * sizeof(uint32_t)
                              threadsPerThreadgroup:tg_gemm];

        [enc setComputePipelineState:g_pso_fused_combine];
        [enc setBuffer:b_y.buf offset:b_y.offset atIndex:0];
        [enc setBuffer:b_inv.buf offset:b_inv.offset atIndex:1];
        [enc setBuffer:b_rw.buf offset:b_rw.offset atIndex:2];
        [enc setBuffer:b_state.buf offset:b_state.offset atIndex:3];
        [enc setBuffer:b_taps.buf offset:b_taps.offset atIndex:4];
        [enc setBuffer:b_resid.buf offset:b_resid.offset atIndex:5];
        [enc setBuffer:b_mask.buf offset:b_mask.offset atIndex:6];
        [enc setBuffer:b_out_x.buf offset:b_out_x.offset atIndex:7];
        [enc setBuffer:b_new_state.buf offset:b_new_state.offset atIndex:8];
        [enc setBytes:&cp length:sizeof(cp) atIndex:9];
        [enc dispatchThreadgroups:grid_comb threadsPerThreadgroup:tg_comb];
      }
      [enc endEncoding];

      [cb commit];
      [cb waitUntilCompleted];
      if ([cb status] != MTLCommandBufferStatusCompleted) {
        std::fprintf(stderr, "[prefill_moe_nax] Fused-combine CommandBuffer failed: status=%ld\n",
                     (long)[cb status]);
        return -7;
      }
      if (out_gpu_ms) {
        *out_gpu_ms = (([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0) / reps;
      }
    }

    if (out_combine_ms) {
      id<MTLCommandBuffer> cb = [g_queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
      [enc setComputePipelineState:g_pso_fused_combine];
      [enc setBuffer:b_y.buf offset:b_y.offset atIndex:0];
      [enc setBuffer:b_inv.buf offset:b_inv.offset atIndex:1];
      [enc setBuffer:b_rw.buf offset:b_rw.offset atIndex:2];
      [enc setBuffer:b_state.buf offset:b_state.offset atIndex:3];
      [enc setBuffer:b_taps.buf offset:b_taps.offset atIndex:4];
      [enc setBuffer:b_resid.buf offset:b_resid.offset atIndex:5];
      [enc setBuffer:b_mask.buf offset:b_mask.offset atIndex:6];
      [enc setBuffer:b_out_x.buf offset:b_out_x.offset atIndex:7];
      [enc setBuffer:b_new_state.buf offset:b_new_state.offset atIndex:8];
      [enc setBytes:&cp length:sizeof(cp) atIndex:9];
      for (int r = 0; r < reps; ++r) {
        [enc dispatchThreadgroups:grid_comb threadsPerThreadgroup:tg_comb];
      }
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      *out_combine_ms = (([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0) / reps;
    }

    return 0;
  }
}

int inkling_prefill_moe_nax_fused_combine(const MoeFusedCombineArgs* a, double* out_gpu_ms) {
  return run_fused_combine_impl(a, 1, -1, out_gpu_ms, nullptr);
}

int prefill_moe_forward_fused_combine(const MoeFusedCombineArgs* a, double* out_gpu_ms) {
  return run_fused_combine_impl(a, 1, -1, out_gpu_ms, nullptr);
}

int prefill_moe_forward_fused_combine_bench(
    const MoeFusedCombineArgs* a,
    int reps,
    int variant,
    double* out_gpu_ms,
    double* out_combine_ms) {
  return run_fused_combine_impl(a, std::max(1, reps), variant, out_gpu_ms, out_combine_ms);
}

}
