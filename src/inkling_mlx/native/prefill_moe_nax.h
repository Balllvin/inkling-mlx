#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MoeForwardArgs {
  int32_t mode;
  int32_t M;
  int32_t K;
  int32_t INTER;
  int32_t D;
  int32_t E;
  int32_t TOP_K;
  int32_t _pad;
  float gate_cap;
  float up_cap;
  uint64_t x_ptr;
  uint64_t inds_ptr;
  uint64_t w13_ptr;
  uint64_t s13_ptr;
  uint64_t gs13_ptr;
  uint64_t w2_ptr;
  uint64_t s2_ptr;
  uint64_t gs2_ptr;
  uint64_t y_sorted_ptr;
  uint64_t inv_order_ptr;
  uint64_t offsets_ptr;
  uint64_t sorted_slots_ptr;
} MoeForwardArgs;

typedef struct MoeFusedCombineArgs {
  int32_t mode;
  int32_t B;
  int32_t L;
  int32_t K;
  int32_t INTER;
  int32_t D;
  int32_t E;
  int32_t TOP_K;
  float gate_cap;
  float up_cap;
  int32_t has_mask;
  int32_t _pad;

  uint64_t x_ptr;
  uint64_t inds_ptr;

  uint64_t w13_ptr;
  uint64_t s13_ptr;
  uint64_t gs13_ptr;
  uint64_t w2_ptr;
  uint64_t s2_ptr;
  uint64_t gs2_ptr;

  uint64_t rweights_ptr;
  uint64_t state_ptr;
  uint64_t taps_ptr;
  uint64_t resid_ptr;
  uint64_t mask_ptr;

  uint64_t out_x_ptr;
  uint64_t new_state_ptr;

  uint64_t y_sorted_ptr;
  uint64_t inv_order_ptr;
} MoeFusedCombineArgs;

int prefill_moe_init(void);

int prefill_moe_register_buffer(uintptr_t ptr, size_t nbytes);
void prefill_moe_unregister_buffer(uintptr_t ptr);
void prefill_moe_clear_cache(void);
void prefill_moe_release_scratch(void);

int prefill_moe_bucket(
    int T, int E,
    uintptr_t inds_ptr,
    uintptr_t offsets_ptr,
    uintptr_t sorted_slots_ptr,
    uintptr_t inv_order_ptr);

int prefill_moe_forward(const MoeForwardArgs* args, double* out_gpu_ms);

int prefill_moe_forward_bench(
    const MoeForwardArgs* args,
    int reps,
    int variant,
    double* out_gpu_ms,
    double* out_w13_ms,
    double* out_w2_ms);

int inkling_prefill_moe_nax_fused_combine(const MoeFusedCombineArgs* args, double* out_gpu_ms);

int prefill_moe_forward_fused_combine(const MoeFusedCombineArgs* args, double* out_gpu_ms);

int prefill_moe_forward_fused_combine_bench(
    const MoeFusedCombineArgs* args,
    int reps,
    int variant,
    double* out_gpu_ms,
    double* out_combine_ms);

#ifdef __cplusplus
}
#endif
