---
name: mlx-apple-silicon-inference
description: >-
  Optimize MLX and Metal 4 LLM inference on Apple Silicon unified memory across
  prefill and decode (NVFP4, MXFP8, BF16 weights, looped/hybrid MoE architectures,
  256k BF16 KV caches, built-in MTP speculation, and native single-CB Objective-C++
  Metal runners). Triggers: "MLX speed", "Apple Silicon inference", "Metal kernel",
  "NAX", "M5 Pro", "single command buffer", "MTP speculative decoding", "MoE Metal".
---

# MLX Apple Silicon Inference Optimization

Measure before editing, gate every change on numerical parity and interleaved median throughput, and report all four throughput axes (`per-stream prefill`, `total prefill`, `per-stream decode`, `total decode`) alongside percent of hardware roofline.

## 1. Establish Hardware Roofline & Environment Discipline

1. Measure the idle AC-power hardware ceilings first (`thermalState == 0`):
   - **DRAM read ceiling**: sustained read bandwidth across a 2 GB tensor (`sum` / Metal `float4` read kernel; e.g. `282.6 GB/s` on Apple M5 Pro).
   - **BF16 compute ceiling**: large square GEMM peak (`4096 x 4096`; e.g. `29.38 TFLOP/s` on Apple M5 Pro).
2. Calculate the exact byte-per-step floor for decode and FLOPs-per-token floor for prefill across target contexts (`512`, `4k`, `16k`, `256k`) and batch sizes (`B = 1..64`). Compute max safe concurrent streams inside the GPU recommended working set (`~51.8 GiB` on 64 GB unified memory) before changing code.
3. Run A/B benchmarks interleaved (control -> candidate -> control) inside exclusive runs on AC power. Report sample count, median, noise floor, `thermalState`, active DRAM bandwidth, and MLX active/peak memory (`mx.get_active_memory`, `mx.get_peak_memory`).

## 2. Eliminate Command-Buffer & Graph Overhead (Single-CB Metal 4 Runners)

1. Understand MLX flush boundaries: MLX flushes command buffers near `max_ops_per_cb = 50` and `max_mb_per_cb = 50`, incurring ~5–8 ms of host flush and Python graph-build overhead across a deep/looped decode step.
2. Move single-stream decode (`B=1`), MTP drafting, and batched `B=1..4` speculative verify into one Objective-C++ `MTLCommandBuffer` (`MTLDispatchTypeSerial`) per cycle loaded via `ctypes`.
3. Map `mx.array` unified-memory buffers **zero-copy** by rounding pointers down to the `16 KB` `vm_page_size` boundary (`base = addr & ~16383ULL`, `offset = addr - base`, `span = align_up(offset + nbytes, 16384)`) via `newBufferWithBytesNoCopy`.
4. **Never persist VA-keyed `MTLBuffer` wrappers for mutable arrays.** macOS VM reuses freed virtual addresses for newly allocated physical pages across turns or KV growth, causing silent stale reads/writes. Persist `MTLBuffer` mappings **only** for immutable model weights, and bind ephemeral KV/conv/scratch pointers per call (holding strong Python refs and verifying `is` identity so CPython's LIFO freelist cannot alias `id(list)`).
5. Compile performance-critical Metal shaders with **`MTLMathModeSafe`**: on Apple M5 (`applegpu_g17s`), disabling reassociation reduces live register pressure, raises SIMD occupancy by `11–29%` on quantized MoE, and preserves `0.00e+00` bit-exact BF16 parity against MLX.
6. Benchmark native C++ vs async-lookahead MLX honestly: for plain `B = 2..32` batched decode without drafting, well-pipelined MLX (`mx.async_eval` lookahead + coalesced kernels) hides CB flush latency inside the `38–120 ms` GPU step, whereas `B = 1` plain decode and `B = 1..4` speculative cycles win decisively in a single CB.

## 3. Match MoE & Linear Kernels to Token Count `M`

1. Never use sorted `SwitchGLU` / `mx.gather_qmm(..., sorted_indices=True)` blindly for small/medium decode batches (`M < 128`): sorting metadata overhead collapses `B=64` by `2x–4.3x`, while unsorted `gather_qmm` fails to share expert reads across tokens (`B=32..64` reads weights redundantly).
2. Partition execution cleanly by `M`:
   - **`M = 1..2`**: direct 1-expert-per-threadgroup kernel (`moe_m1_experts`) with inline dequantize (`dot8_nvfp4`), saturating `97–99%` of DRAM bandwidth.
   - **`M = 3..112`**: compact `(tile_expert, tile_t0)` tile-table grouped kernel (`moe_grouped_tiles`) with `#pragma unroll` scalar accumulators (eliminates LLVM stack spills), `uint4` loads, and a `256`-entry `half2` E2M1 LUT (`72–89%` DRAM BW, `0.00e+00` invariance).
   - **`M >= 128`**: Metal 4 Neural Accelerator (`mpp::tensor_ops::matmul2d`, `MTLLanguageVersion4_0`) double-buffered grouped MoE with GPU radix bucketing, indirect dispatch, and fused 64-channel 128B-aligned combine+residual inside 1 CB (`21.9–27.1 TFLOP/s`).
3. Keep routers (`e_score_correction_bias`, FP32 projections) and small message gates in FP32 (`NULL` BF16 fallbacks): BF16 router weights flip top-k expert selections on small logit gaps without measurable speedup at `M <= 32`.
4. Read `lm_head` (`>900 MB`) at most once per step (`BM=16` NAX or `R=1`/`R=2` SIMT fused `head_argmax` skipping full-vocab FP32 logits materialization when greedy).

## 4. Drive Apple Neural Accelerators (NAX) Without Mask Bloat

1. On Metal 4 Apple Silicon GPUs, use `mpp::tensor_ops::matmul2d` for causal/sliding-window prefill attention (`flash_attn_nax`) rather than falling back to SIMT kernels whenever custom biases or sinks are present:
   - Compute causal, sliding-window, and ragged `key_start` bounds arithmetically inside tiles to avoid allocating or reading multi-GB dense attention masks (`107 GB` saved at `16k`).
   - Fold hardware NAX Toeplitz relative attention bias and exact FP32 query-dependent sinks directly into online softmax.
2. For batched decode attention (`B >= 2`) with rotating sliding-window caches past wrap-around, index the physical ring buffer with chronological `ring_idx` inside a coalesced split-K kernel (`decode_attention`) rather than falling back to naive uncoalesced attention or temporal copies.

## 5. Run Speculative Decoding End-to-End on the GPU (`B = 1..4`)

1. Unify drafter execution, target batch verification, greedy/Leviathan (`temp > 0`) rejection sampling, KV/conv commit & rollback, and hidden-state capture copy into **one command buffer** so sampled generation never stalls on host `softmax` or per-token readbacks.
2. Keep `B = 1..4` active streams in an **unmerged per-stream KV pool** (`sum(K_i + 1) <= 32`) with batched NAX projections (`M_v >= 9`) and cap adaptive draft length (`K <= 5`) when `B >= 3` so high-acceptance streams do not inflate verification tile cost for the rest of the batch. Only merge into `BatchKVCache` at `B >= 5`.
3. Add a cooldown fallback (`ema_acc < 1.65`) on low-acceptance single-stream greedy turns to run plain 1-CB `M=1` decode while capturing hidden states for cheap draft resume.

## 6. Guard Memory & Parity at 256k Context

1. Pre-allocate `GrowingKVCache` cleanly (`step=2048`, `mx.set_cache_limit` bounded near `4 GB`) and check `PrefixCache` serialized size **before** cloning `>=128`-token KV caches (prevents `+12 GB` transient spikes near `256k`).
2. Share common system prompts (`>= 128` tokens) across concurrent burst arrivals so follower streams skip redundant cold prefill.
3. Gate every speed change against an exact/KL reference (PyTorch/Transformers FP32/BF16 reference: `top-1` where logit gap `> 0.05`, mean `KL <= 1e-5`; single-stream speculative greedy 100% byte-identical to plain; multi-step synthetic parity `0.00e+00` SIMT; live tool-call & needle tests).
