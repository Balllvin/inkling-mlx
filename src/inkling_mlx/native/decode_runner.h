#pragma once

#include <stddef.h>
#include <stdint.h>

#include "mtp_draft_runner.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct InklingModelConfig {
    int32_t n_layers;
    int32_t loop_start;
    int32_t loop_end;
    int32_t loop_passes;
    int32_t hidden_size;
    int32_t inter_size;
    int32_t n_experts;
    int32_t top_k;
    int32_t vocab_size;
    int32_t unpadded_vocab_size;
    int32_t d_rel;
    int32_t loop_gate_rank;
    int32_t max_m;
    int32_t max_n_heads;
    int32_t max_n_kv;
    int32_t max_head_dim;
    int32_t max_rel_extent;
    float   rms_norm_eps;
    float   qk_l2_clip_threshold;
    float   rel_l2_clip_threshold;
    float   route_scale;
    float   mlp_gate_softcap;
    float   mlp_up_softcap;
    float   log_scaling_n_floor;
    float   log_scaling_alpha;
} InklingModelConfig;

typedef struct InklingGlobalWeights {
    const void* embed_tokens;
    size_t      embed_tokens_bytes;
    const void* embed_norm;
    size_t      embed_norm_bytes;
    const void* message_queries;
    size_t      message_queries_bytes;
    const void* final_query;
    size_t      final_query_bytes;
    const void* final_norm;
    size_t      final_norm_bytes;
    const void* lm_head;
    size_t      lm_head_bytes;
    float       embed_norm_eps;
    float       final_norm_eps;
} InklingGlobalWeights;

typedef struct InklingLayerWeights {
    int32_t layer_idx;
    int32_t is_sliding;
    int32_t is_block_boundary;
    int32_t has_attn_input;
    int32_t has_message_gates;
    int32_t n_heads;
    int32_t n_kv;
    int32_t head_dim;
    int32_t sliding_window;
    int32_t rel_extent;
    int32_t qkv_at[5];
    int32_t qkv_total_rows;
    int32_t moe_mode;
    float   attn_eps;
    float   mlp_eps;

    const void* stack;
    size_t      stack_bytes;
    const void* o_proj;
    size_t      o_proj_bytes;
    const void* attn_query;
    size_t      attn_query_bytes;
    const void* attn_norm;
    size_t      attn_norm_bytes;
    const void* mlp_query;
    size_t      mlp_query_bytes;
    const void* mlp_norm;
    size_t      mlp_norm_bytes;
    const void* attn_taps;
    size_t      attn_taps_bytes;
    const void* mlp_taps;
    size_t      mlp_taps_bytes;
    const void* k_taps;
    size_t      k_taps_bytes;
    const void* v_taps;
    size_t      v_taps_bytes;
    const void* kdyn_w;
    size_t      kdyn_w_bytes;
    const void* vdyn_w;
    size_t      vdyn_w_bytes;
    const void* attn_gate_down;
    size_t      attn_gate_down_bytes;
    const void* attn_gate_forget;
    size_t      attn_gate_forget_bytes;
    const void* attn_gate_gain;
    size_t      attn_gate_gain_bytes;
    const void* mlp_gate_down;
    size_t      mlp_gate_down_bytes;
    const void* mlp_gate_forget;
    size_t      mlp_gate_forget_bytes;
    const void* mlp_gate_gain;
    size_t      mlp_gate_gain_bytes;
    const void* router_w;
    size_t      router_w_bytes;
    const void* router_b;
    size_t      router_b_bytes;
    const void* sink_w;
    size_t      sink_w_bytes;
    const void* q_scale;
    size_t      q_scale_bytes;
    const void* rel_scale;
    size_t      rel_scale_bytes;
    const void* rel_proj;
    size_t      rel_proj_bytes;
    const void* w13_weight;
    size_t      w13_weight_bytes;
    const void* w13_scales;
    size_t      w13_scales_bytes;
    const void* w13_gscale;
    size_t      w13_gscale_bytes;
    const void* w2_weight;
    size_t      w2_weight_bytes;
    const void* w2_scales;
    size_t      w2_scales_bytes;
    const void* w2_gscale;
    size_t      w2_gscale_bytes;

    const void* attn_gate_fg_bf16;
    size_t      attn_gate_fg_bf16_bytes;
    const void* mlp_gate_fg_bf16;
    size_t      mlp_gate_fg_bf16_bytes;
    const void* router_w_bf16;
    size_t      router_w_bf16_bytes;
} InklingLayerWeights;

typedef struct InklingKVMeta {
    int32_t offset;
    int32_t idx;
    int32_t capacity;
    int32_t max_size;
    int32_t is_rotating;
    int32_t stride_h;
    int32_t stride_s;
    int32_t _pad;
} InklingKVMeta;

typedef struct InklingStepStats {
    double  encode_ms;
    double  gpu_ms;
    double  wall_ms;
    int32_t n_dispatches;
} InklingStepStats;

typedef struct InklingCtx InklingCtx;

InklingCtx* inkling_ctx_create(const InklingModelConfig* cfg);
void        inkling_ctx_destroy(InklingCtx* ctx);

int inkling_ctx_register_globals(InklingCtx* ctx, const InklingGlobalWeights* globals);
int inkling_ctx_register_layer(InklingCtx* ctx, const InklingLayerWeights* layer);

int    inkling_ctx_num_visits(const InklingCtx* ctx);
size_t inkling_ctx_conv_states_floats(const InklingCtx* ctx, int32_t m_tokens);
size_t inkling_ctx_visit_conv_offset(const InklingCtx* ctx, int32_t visit_idx, int32_t m_tokens);
size_t inkling_ctx_scratch_bytes(const InklingCtx* ctx);
void   inkling_ctx_clear_ptr_cache(InklingCtx* ctx);

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
    InklingStepStats*    stats_out
);

int inkling_ctx_commit_rotating(
    InklingCtx*          ctx,
    void* const*         kv_ptrs,
    InklingKVMeta*       kv_meta,
    int32_t              n_accepted
);

int inkling_ctx_commit_verify(
    InklingCtx*          ctx,
    void* const*         kv_ptrs,
    void* const*         conv_ptrs,
    InklingKVMeta*       kv_meta,
    int32_t              n_accepted,
    int32_t              m_tokens
);

typedef struct InklingBatchKVMeta {
    int32_t idx;
    int32_t offset;
    int32_t capacity;
    int32_t max_size;
    int32_t is_rotating;
    int32_t stride_b;
    int32_t stride_h;
    int32_t stride_s;
} InklingBatchKVMeta;

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
    InklingStepStats*       stats_out
);

int inkling_ctx_attach_mtp(
    InklingCtx*                  ctx,
    const MTPDraftConfig*        cfg,
    const MTPDraftGlobalWeights* gw,
    const MTPDraftLayerWeights*  layers,
    const int32_t*               capture_layer_ids,
    int32_t                      n_capture_layers
);

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
    InklingStepStats*    stats_out
);

typedef struct InklingMTPStreamDesc {
    int32_t              anchor_token;
    int32_t              n_ctx;
    int32_t              num_draft;
    int32_t              _pad0;
    const void*          in_hidden_ptr;
    size_t               in_hidden_bytes;
    void*                out_hidden_ptr;
    size_t               out_hidden_bytes;
    float                temperature;
    float                top_p;
    float                logit_softcap;
    int32_t              _pad1;
    uint64_t             rng_seed;
    uint64_t             rng_step;
    void* const*         kv_ptrs;
    void* const*         conv_ptrs;
    InklingKVMeta*       kv_meta;
    MTPKVCacheDesc*      draft_kv_descs;
    int32_t              out_tokens[8];
    int32_t              out_n_acc;
    int32_t              _pad2;
} InklingMTPStreamDesc;

int inkling_ctx_mtp_step_batched(
    InklingCtx*             ctx,
    InklingMTPStreamDesc*   streams,
    int32_t                 batch_size,
    InklingStepStats*       stats_out
);

int inkling_ctx_mtp_debug_read(
    InklingCtx* ctx,
    int32_t     buf_id,
    void*       dst_ptr,
    size_t      dst_nbytes
);

#ifdef __cplusplus
}
#endif
