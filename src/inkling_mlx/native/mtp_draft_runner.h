#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MTPDraftConfig {
    int32_t hidden_size;
    int32_t num_layers;
    int32_t num_heads;
    int32_t num_kv_heads;
    int32_t head_dim;
    int32_t inter_size;
    int32_t block_size;
    int32_t markov_rank;
    int32_t n_target_layers;
    int32_t vocab_size;
    int32_t unpadded_vocab_size;
    float   rms_norm_eps;
    float   embed_norm_eps;
    float   rope_theta;
} MTPDraftConfig;

typedef struct MTPDraftGlobalWeights {
    const void* embed_tokens;
    size_t      embed_tokens_bytes;
    const void* embed_norm;
    size_t      embed_norm_bytes;
    const void* mask_embed;
    size_t      mask_embed_bytes;
    const void* final_norm;
    size_t      final_norm_bytes;
    const void* lm_head;
    size_t      lm_head_bytes;
    const void* fuse_alpha;
    size_t      fuse_alpha_bytes;
    const void* markov_dflare_norm;
    size_t      markov_dflare_norm_bytes;
    const void* markov_w1;
    size_t      markov_w1_bytes;
    const void* rnn_w_ctx_t;
    size_t      rnn_w_ctx_t_bytes;
    const void* rnn_w_step_t;
    size_t      rnn_w_step_t_bytes;
    const void* rnn_b_f32;
    size_t      rnn_b_f32_bytes;
    const void* fold_weight;
    size_t      fold_weight_bytes;
    const void* fold_bias;
    size_t      fold_bias_bytes;
} MTPDraftGlobalWeights;

typedef struct MTPDraftLayerWeights {
    int32_t     is_local;
    int32_t     window;
    const void* dflare_norm;
    size_t      dflare_norm_bytes;
    const void* input_layernorm;
    size_t      input_layernorm_bytes;
    const void* post_attention_layernorm;
    size_t      post_attention_layernorm_bytes;
    const void* kv_ctx_w;
    size_t      kv_ctx_w_bytes;
    const void* k_norm_ctx;
    size_t      k_norm_ctx_bytes;
    const void* qkv_w;
    size_t      qkv_w_bytes;
    const void* q_norm;
    size_t      q_norm_bytes;
    const void* k_norm;
    size_t      k_norm_bytes;
    const void* o_proj_w;
    size_t      o_proj_w_bytes;
    const void* gu_w;
    size_t      gu_w_bytes;
    const void* down_proj_w;
    size_t      down_proj_w_bytes;
} MTPDraftLayerWeights;

typedef struct MTPKVCacheDesc {
    uintptr_t k_ptr;
    uintptr_t v_ptr;
    size_t    k_nbytes;
    size_t    v_nbytes;
    int32_t   cache_cap;
    int32_t   offset;
    int32_t   is_rotating;
    int32_t   window;
    int32_t   ring_idx;
    int32_t   stride_h;
    int32_t   stride_s;
    int32_t   _pad;
} MTPKVCacheDesc;

typedef struct MTPStepStats {
    double  encode_ms;
    double  gpu_ms;
    double  wall_ms;
    int32_t n_dispatches;
    int32_t _pad;
} MTPStepStats;

typedef struct MTPDraftRunner MTPDraftRunner;

MTPDraftRunner* mtp_draft_create(
    const MTPDraftConfig* cfg,
    const MTPDraftGlobalWeights* globals,
    const MTPDraftLayerWeights* layers);

void mtp_draft_destroy(MTPDraftRunner* runner);

void mtp_draft_clear_kv_cache_map(MTPDraftRunner* runner);

const char* mtp_draft_device_name(MTPDraftRunner* runner);

int mtp_draft_greedy_step(
    MTPDraftRunner* runner,
    int32_t anchor_token,
    uintptr_t hidden_ptr,
    size_t hidden_nbytes,
    int32_t n_ctx,
    int32_t num_draft,
    MTPKVCacheDesc* kv_caches,
    int32_t* out_tokens,
    MTPStepStats* out_stats);

int mtp_draft_bench_layers_gemv(
    MTPDraftRunner* runner,
    int32_t n_ctx,
    int32_t warmup,
    int32_t iters,
    double* out_median_gpu_ms,
    double* out_achieved_gbps);

typedef struct MTPStageBreakdown {
    double layers_ms;
    double sdpa_only_ms;
    double lm_head_ms;
    double markov_ctx_ms;
    double rnn_steps_ms;
    double argmax_ms;
} MTPStageBreakdown;

int mtp_draft_profile_stages(
    MTPDraftRunner* runner,
    int32_t anchor_token,
    uintptr_t hidden_ptr,
    size_t hidden_nbytes,
    int32_t n_ctx,
    int32_t num_draft,
    const MTPKVCacheDesc* kv_caches,
    int32_t warmup,
    int32_t iters,
    MTPStageBreakdown* out_bd);

int mtp_draft_debug_read(
    MTPDraftRunner* runner,
    int32_t buf_id,
    void* dst_ptr,
    size_t dst_nbytes);

int mtp_draft_debug_run_up_to_layer(
    MTPDraftRunner* runner,
    int32_t anchor_token,
    uintptr_t hidden_ptr,
    size_t hidden_nbytes,
    int32_t n_ctx,
    MTPKVCacheDesc* kv_caches,
    int32_t max_layer);

#ifdef __cplusplus
}
#endif
