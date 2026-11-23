#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int32_t B;
  int32_t HQ;
  int32_t HKV;
  int32_t L;
  int32_t S;
  int32_t D;
  int32_t R;
  int32_t window;
  float scale;

  int64_t q_s0, q_s1, q_s2;
  int64_t k_s0, k_s1, k_s2;
  int64_t v_s0, v_s1, v_s2;

  int32_t sinks_is_bf16;
  int32_t has_gate;
  int32_t gate_off;
  int32_t gate_stride;
  int32_t variant;
} InklingFlashAttnDesc;

int inkling_flash_attn_init(void);
void inkling_flash_attn_clear_cache(void);
const char* inkling_flash_attn_device_name(void);

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
    double* out_gpu_ms);

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
    double* out_median_gpu_ms, double* out_median_wall_ms);

#ifdef __cplusplus
}
#endif
