# inkling-mlx

High-throughput MLX + Metal 4 inference engine for [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on Apple Silicon ([Hugging Face model card](https://huggingface.co/balllvin/Inkling-2-Mini-MLX)).

Runs the full released model as-is with **3 loop passes (48 layer visits/token)**, **unquantized 256k BF16 KV cache**, **bit-exact mixed NVFP4 / MXFP8 / BF16 weights**, and the built-in **`mtp.safetensors`** speculative drafter over an OpenAI-compatible HTTP/SSE API (`/v1/chat/completions`, `/v1/models`, `/health`).

> **Hardware target:** Built and tuned specifically for **Apple M5** (per-core Metal 4 Neural Accelerators / NAX). Measured and verified on a **MacBook Pro (`Mac17,9`) with Apple M5 Pro (18-core CPU: 6 Super + 12 Performance, 20-core GPU, Metal 4), 64 GB unified memory, macOS 26.5.1 (Xcode 26.5 SDK), MLX 0.32.4**. See [Chip Compatibility & Porting Guide](#chip-compatibility--porting-guide) for older M1–M4 Macs (`--skip-build` fallback or fork instructions).

---

## Prerequisites, Model & RAM Requirements

### Model to Download

This engine runs the official [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) Hugging Face checkpoint unmodified (`~20.2 GB` download, **`>= 25 GB` free SSD space** recommended):

- `model-00001-of-00001.safetensors` (`~19.53 GB`, mixed NVFP4 / MXFP8 / BF16 text + vision weights)
- `mtp.safetensors` (`~0.64 GB`, built-in 3-layer speculative drafter)
- `config.json` and tokenizer files

Download ahead of time into `~/models/Inkling-2-Mini` with `inkling-mlx download --repo thinkingmachines/Inkling-2-Mini --local-dir ~/models/Inkling-2-Mini`, or pass `--model thinkingmachines/Inkling-2-Mini` directly to `inkling-mlx serve` to fetch and cache it automatically on first launch.

### Unified RAM Requirements

| Unified Memory | Support Status | Recommended Setup / Limits |
|---|---|---|
| **16–24 GB** | Not supported | Resident text + MTP runtime exceeds `20 GB` even before KV cache and prefill scratch. |
| **32 GB** | Minimum | Single-stream (`B=1`) short/medium context (`<= 16k–32k`); use `--no-mtp --cache-limit-gb 2.0`. |
| **48 GB** | Supported | `B=1..4` with MTP up to `64k–128k`, or `B=1` up to `256k` with `--no-mtp`. |
| **64 GB** | **Recommended (Tested)** | Full `256k` BF16 KV context with MTP (`~29.6 GiB` active, `~38.6 GB` peak, `0` swap), `B=1..4` concurrent MTP agents, or up to `B=64` plain batched decode at `ctx=512` (`41.3 GB` peak, `0` swap). |
| **96–128 GB+** | Supported | Multiple concurrent `128k–256k` streams and larger prefix caches. |

### Software Requirements

- **Native Apple M5 Metal 4 + NAX path (recommended):** Apple M5 / M5 Pro / M5 Max / M5 Ultra Mac, **macOS 26+**, Xcode Command Line Tools (`xcode-select --install`), and Python `3.11–3.13` (or `uv`).
- **MLX-only fallback (untested on M1–M4):** Apple Silicon Mac with Metal 3.1+ (macOS 14+), Python `3.11–3.13`, launched with `inkling-mlx serve --skip-build` so Metal 4 libraries are not compiled.

---

## Agent Quickstart Prompt

Copy and paste this prompt into any coding agent on an Apple Silicon Mac:

```text
Set up and verify `inkling-mlx` on this Mac:
1. Check hardware and memory (`system_profiler SPHardwareDataType`): confirm Apple Silicon with >= 32 GB unified RAM (64 GB recommended). Confirm whether this Mac has an Apple M5-family chip and macOS 26+ with Xcode Command Line Tools installed.
2. Clone `https://github.com/Balllvin/inkling-mlx` (or use the current repo directory if already cloned) and `cd` into it.
3. Run `bash scripts/setup.sh` (or create a Python 3.11–3.13 venv, `pip install -e .`, and run `bash scripts/build_native.sh` on Apple M5 / macOS 26+). On older M1–M4 Macs or < macOS 26, skip building the Metal 4 dylibs and use `--skip-build` (MLX fallback, see README Compatibility section).
4. Download `thinkingmachines/Inkling-2-Mini` from Hugging Face into `~/models/Inkling-2-Mini` using `.venv/bin/inkling-mlx download --repo thinkingmachines/Inkling-2-Mini --local-dir ~/models/Inkling-2-Mini` (skip if `config.json`, `model-00001-of-00001.safetensors`, and `mtp.safetensors` are already present).
5. Run the verification suite:
   - `.venv/bin/python -m unittest tests/test_engine.py -v`
   - On Apple M5 / macOS 26+ after `build_native.sh` succeeds:
     - `.venv/bin/python tests/test_flash_attn.py`
     - `.venv/bin/python tests/test_prefill_moe.py`
     - `.venv/bin/python tests/test_mtp_draft.py`
6. Start the OpenAI-compatible server on `127.0.0.1:8095` via `.venv/bin/inkling-mlx serve --model ~/models/Inkling-2-Mini --host 127.0.0.1 --port 8095` (add `--skip-build` on non-M5 / < macOS 26, and `--no-mtp --cache-limit-gb 2.0` if unified RAM is 32 GB) and wait until `http://127.0.0.1:8095/health` reports `"ready": true`.
7. Send one test request to `http://127.0.0.1:8095/v1/chat/completions` and report the `/health` JSON, completion text, usage, and throughput metrics.
```

---

## Manual Setup & Serving

```bash
git clone https://github.com/Balllvin/inkling-mlx.git
cd inkling-mlx

# One-shot setup (Apple M5 + macOS 26+): creates .venv, installs inkling-mlx, compiles Metal 4 .dylibs, and downloads weights
DOWNLOAD_MODEL=1 bash scripts/setup.sh

# Or step by step
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
inkling-mlx build          # Requires Apple M5 + macOS 26+ (omit on M1-M4 / < macOS 26 and pass --skip-build below)
inkling-mlx download --repo thinkingmachines/Inkling-2-Mini --local-dir ~/models/Inkling-2-Mini

# Serve OpenAI-compatible API (auto-downloads from HF if passed a repo ID)
inkling-mlx serve --model ~/models/Inkling-2-Mini --host 127.0.0.1 --port 8095
# On 32 GB Macs: add --no-mtp --cache-limit-gb 2.0
# On M1-M4 / < macOS 26: add --skip-build
```

Example request:

```bash
curl -s http://127.0.0.1:8095/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "inkling-2-mini",
    "messages": [{"role": "user", "content": "Write an O(n) Fibonacci helper in Python."}],
    "max_tokens": 256,
    "temperature": 0.0
  }'
```

---

## Chip Compatibility & Porting Guide

This engine is **built and tuned specifically for Apple M5** (`Apple10` GPU family: M5, M5 Pro, M5 Max, M5 Ultra) and has **only been tested on Apple M5 Pro (`Mac17,9`, 20-core GPU, 64 GB unified RAM, macOS 26.5.1)**.

### Why Older M-Series Chips (M1 / M2 / M3 / M4) Are Not Natively Supported Out of the Box

| Subsystem | Apple M5 (`Apple10`, macOS 26+) | Apple M1 / M2 / M3 / M4 (`Apple7`–`Apple9`) | Why |
|---|---|---|---|
| **Metal 4.0 NAX FlashAttention** (`native/flash_attn_nax.mm`) | Native 4x speedup | Not supported natively | Uses `MetalPerformancePrimitives` (`mpp::tensor_ops::matmul2d`) targeting M5 per-core Neural Accelerators and `MTLLanguageVersion4_0`. |
| **Metal 4.0 NAX Prefill MoE** (`native/prefill_moe_nax.mm`) | Native 21–27 TFLOP/s | Not supported natively | Uses `MTLLanguageVersion4_0` + `mpp::tensor_ops::matmul2d` (`M >= 128`) unavailable on M1–M4 hardware. |
| **Native 1-CB Decode & MTP** (`native/decode_runner.mm`, `native/mtp_draft_runner.mm`) | Compiles & runs (`B=1..4` MTP + batched) | Fails build / init without edits | `mtp_draft_runner.mm` hardcodes `MTLLanguageVersion4_0` even though its shaders are pure SIMT, and `decode_runner.mm` aborts init if `lib_nax` (`MTLLanguageVersion4_0`, used at `M >= 9`) fails to compile. |
| **MLX + Metal 3.1 SIMT Fallback** (`fused.py`, `kernels.py`, `moe_kernels.py`, `mtp.py`) | Available (`--no-native` / fallback) | **Supported in code (untested on M1–M4)** | `kernels.py` and `moe_kernels.py` use standard Metal 3.1 (`simdgroup_bfloat8x8`, `moe_m1_experts`, `moe_grouped_tiles`) and all native `.dylib` loaders return `None` when `.dylib`s are absent. |
| **NVIDIA CUDA / AMD ROCm / Intel / CPU-only** | — | Not supported | Engine depends on Apple Unified Memory, Metal, and `mlx`. |

> **Running on M1 / M2 / M3 / M4 today (untested):** If `inkling-mlx build` or `scripts/build_native.sh` fails on `< macOS 26` or non-M5 hardware, run **`inkling-mlx serve --skip-build --model ...`**. All four native loaders (`try_create_native_decode_runner`, `try_create_native_mtp_runner`, `_get_flash_attn_lib`, `_get_prefill_moe_lib`) return `None` when the `.dylib`s are absent or fail to init, falling back automatically to the pure MLX + Metal 3.1 SIMT kernels (`moe_m1_experts`, `moe_grouped_tiles`, `relative_attention_nax`). Because we do not have M1–M4 machines here, that fallback has not been benchmarked on M1–M4.

### How to Fork / Adapt This Repo for Apple M1–M4 (or Other Chips)

A human or coding agent can unlock native single-command-buffer `B=1` decode and SIMT MTP on **M1 / M2 / M3 / M4** (without M5 NAX) with four focused changes:

1. **Guard Metal 4.0 NAX in `src/inkling_mlx/native/decode_runner.mm`**:
   - Around the `if (cfg->max_m >= 8)` NAX compilation block (`MTLLanguageVersion4_0` / `kMetal4NaxSource`), wrap `MTLLanguageVersion4_0` with `#if defined(__MAC_26_0)` and leave `ctx->lib_nax = nil` on failure instead of returning `nullptr` from init.
   - Gate `use_nax_qkv` and `use_nax_head` (`M >= 9` verification/batched paths) on `(ctx->lib_nax != nil)` instead of returning `-12`, so `M >= 9` dispatches fall through to the existing SIMT paths already in `decode_runner.mm`.
2. **Lower `MTLLanguageVersion` in `src/inkling_mlx/native/mtp_draft_runner.mm`**:
   - Change `opts.languageVersion = MTLLanguageVersion4_0` to `MTLLanguageVersion3_2` (or `MTLLanguageVersion3_1` on macOS 14). The MTP drafter kernels are pure SIMT / GEMV and contain no `mpp::tensor_ops` calls.
3. **Make NAX prefill `.dylib`s optional in `scripts/build_native.sh` and `src/inkling_mlx/cli.py`**:
   - Build only `libinkling_decode.dylib` and `libinkling_mtp_draft.dylib` when targeting M1–M4 or Xcode `< 26`. Python prefill automatically falls back to MLX `relative_attention_nax` (`kernels.py`) and `moe_grouped_tiles` / `mx.gather_qmm` (`moe_kernels.py`) when `libinkling_flash_attn.dylib` and `libinkling_prefill_moe.dylib` are absent.
4. **Re-tune and gate parity on target silicon**:
   - Benchmark `MTLMathModeSafe` vs fast math and threadgroup widths (`128` vs `256`) on `Apple7`/`Apple8`/`Apple9` register files, and verify parity with `tests/test_engine.py` and `tests/test_decode_runner.py`.
   - *(Note on `simdgroup_bfloat8x8`: supported on Metal 3.1+ / M1+ `Apple7` and newer, so `kernels.py` BF16 SIMD-group GEMMs work on M1–M4 under macOS 14+.)*

**Porting to CUDA / ROCm / non-Apple accelerators:** Keep the model spec and scheduling logic from [`model.py`](src/inkling_mlx/model.py), [`config.py`](src/inkling_mlx/config.py), [`mtp.py`](src/inkling_mlx/mtp.py), and [`parser.py`](src/inkling_mlx/parser.py) (3-loop 48-visit core with depth residuals & message gates, causal `ShortConv`, hybrid SWA + Toeplitz relative bias + query-dependent sinks, NVFP4 layers 0–19 / MXFP8 layers 20–23 MoE, built-in DSpark + MarkovHead MTP), and swap the MLX / `.mm` Metal runners for CUDA Graphs, Triton, FlashAttention, and CUTLASS FP4/FP8 grouped GEMMs (e.g. Blackwell).

---

## Architecture & Key Optimizations

Stock MLX execution flushes ~10 GPU command buffers per token (`max_ops_per_cb = 50`), suffers 4.3x `SwitchGLU` sorting cliffs (`sorted_indices=True` at `M >= 8`), reads `lm_head` multiple times at higher batch sizes, and materializes huge dense attention masks during prefill. `inkling-mlx` replaces the critical path while preserving exact numerical equivalence:

1. **Single-CB Native Decode (`native/decode_runner.mm`)**: Executes all 48 layer visits (embedding, RMSNorm, depth residuals, message gating, packed QKV/gate, `ShortConv`, split-K GQA with chronological `ring_idx`, FP32 routing, NVFP4/MXFP8 MoE, `lm_head`, and greedy argmax) in one `MTLCommandBuffer` (`MTLDispatchTypeSerial`) via zero-copy `16 KB` page-aligned `MTLBuffer` wrappers. Compiled with `MTLMathModeSafe` (`11–29%` faster on Apple M5 Pro due to lower register pressure and `0.00e+00` BF16 bit-exactness).
2. **Unified `B=1..4` GPU-Accept MTP (`decode_runner.mm` + `mtp_draft_runner.mm`)**: Drafts `K` tokens with the built-in 3-layer DSpark + folded `MarkovHead`, verifies `sum(K_i + 1) <= 32` tokens across `1..4` concurrent streams, runs greedy or exact Leviathan rejection sampling on the GPU, commits/rolls back KV + draft caches, and captures target hidden states in **1 Metal command buffer**. Uses Metal 4 Neural Accelerator (NAX) kernels when `M >= 9`, caps adaptive `K <= 5` for `B >= 3`, and drops seamlessly into `B >= 5` continuous batching when concurrency exceeds 4.
3. **Regime-Matched MoE Kernels (`moe_kernels.py` & `native/prefill_moe_nax.mm`)**:
   - `M = 1..2`: Direct 1-expert-per-threadgroup kernel (`moe_m1_experts`) with inline `dot8_nvfp4` (`97–99%` DRAM BW).
   - `M = 3..112`: Zero-spill compact tile-table grouped kernel (`moe_grouped_tiles`) with `#pragma unroll` scalar accumulators, vectorized `uint4` loads, and a `256`-entry `half2` E2M1 pair LUT (`72–89%` DRAM BW, `0.00e+00` batch invariance).
   - `M >= 128`: Metal 4.0 NAX (`mpp::tensor_ops::matmul2d`) double-buffered grouped MoE with GPU radix bucketing, indirect dispatch, and a fused 64-channel 128B-aligned combine+residual dispatch (`21.9–27.1 TFLOP/s`).
4. **Metal 4.0 NAX Zero-Mask FlashAttention (`native/flash_attn_nax.mm`)**: Causal/sliding-window prefill attention (`3.6x` SWA, `2.5x` 16k Global) with hardware NAX Toeplitz relative bias, exact FP32 query-dependent sinks without KV concatenation, ragged `key_start`, and zero dense mask DRAM traffic up to `256k`.
5. **Memory & Serving Safety**: Pre-checks `PrefixCache` serialized size before cloning `256k` KV (`-12.5 GB` peak spike eliminated), shares long system-prompt prefixes (`>= 128` tokens) across concurrent burst arrivals, never caches VA-keyed mutable `MTLBuffer` pointers across MLX frees, and validates live array identity to defeat CPython `id()` freelist aliasing.

---

## Measured Performance (Apple M5 Pro 64 GB)

Hardware ceilings measured on this machine (`applegpu_g17s`, 20 GPU cores, macOS 26.5, MLX 0.32.4, AC power):
- **Sustained DRAM Read Bandwidth**: **`282.6 GB/s`**
- **Peak BF16 GEMM Compute**: **`29.38 TFLOP/s`**

### Decode Throughput (`per-stream` and `total aggregate`, vs Stock Baseline)

| Workload | Stock Baseline | `inkling-mlx` | Speedup | % of DRAM BW Ceiling |
|---|---:|---:|---:|---:|
| Plain `B=1` (`ctx=512`) | `33.50 tok/s` | **`48.97 tok/s`** | **`1.46x`** | **`97.0%`** (`50.5` ceil) |
| Plain `B=1` (`ctx=4,096`) | — | **`48.28 tok/s`** | — | **`99.0%`** (`48.8` ceil) |
| Plain `B=1` (`ctx=250k`, BF16 KV) | — | **`19.79 tok/s`** | — | **`97.1%`** (`20.4` ceil) |
| MTP `B=1` (tool / code / 6-prompt med) | — | **`120.19` / `86.86` / `66.79 tok/s`** | **`2.0x–3.6x`** | **`130.5%–236.4%`** |
| MTP `B=1` Sampled (`T=0.7`, `thermalState=1`) | — | **`50.05 tok/s`** med (`97.91/s` tool) | — | **`97.8%–192.6%`** |
| Concurrent MTP `B=2` (`T=0.0`, `thermalState=1`) | ~`64.0 tok/s` | **`94.98 tok/s` agg (`47.49/stream`)** | **`~1.48x`** | **`122.7%`** (plain `B=2` ceil) |
| Concurrent MTP `B=4` (`T=0.0`, `thermalState=1`) | ~`84.3 tok/s` | **`117.68 tok/s` agg (`27.59/stream`)** | **`~1.40x`** | **`108.9%`** (plain `B=4` ceil) |
| Batched Plain `B=8` (`ctx=512`) | ~`47.2 tok/s` | **`160.19 tok/s` agg (`20.03/stream`)** | **`3.4x`** | **`115.9%`** |
| Batched Plain `B=16` (`ctx=512`) | ~`46.3 tok/s` | **`218.33 tok/s` agg (`13.86/stream`)** | **`4.7x`** | **`117.2%`** |
| Batched Plain `B=32` (`ctx=512`) | ~`43.9 tok/s` | **`277.11 tok/s` agg (`8.75/stream`)** | **`6.3x`** | **`100.1%`** |
| Batched Plain `B=48` (`ctx=512`) | — | **`308.17 tok/s` agg (`6.48/stream`)** | — | **`83.4%`** |
| Batched Plain `B=64` (`ctx=512`) | — | **`335.48 tok/s` agg (`5.28/stream`)** | — | **`72.8%`** (`41.3 GB` peak, 0 swap) |
| Batched Plain `B=32` (`ctx=4,096`) | — | **`235.80 tok/s` agg (`7.49/stream`)** | — | **`101.8%`** (`41.3 GB` peak, 0 swap) |
| Batched Plain `B=4` (`ctx=16,384`) | `1.87 tok/s` | **`93.12 tok/s` agg (`23.28/stream`)** | **`49.8x`** | **`108.4%`** |

### Prefill Throughput & Time-to-First-Token (`29.38 TFLOP/s` Ceiling)

| Condition | Stock Baseline | `inkling-mlx` | Speedup | % Compute Ceiling |
|---|---:|---:|---:|---:|
| Cold `16k` `B=1` TTFT / Prefill | `21.4 s` (`762 tok/s`) | **`11.14 s` (`1,464.3 tok/s`)** @ `T=1` | **`1.92x`** | **`47.9%`** (`3,060` ceil) |
| Cold `4k` `B=1` TTFT / Served Prefill | — | **`2.98 s` (`1,381.2 tok/s`)** @ `T=1` | — | **`45.1%`** (`3,060` ceil) |
| In-Process `4k` `B=1` (`chunk=4096`) | — | **`1,739.4 tok/s`** | — | **`54.2%`** (`3,210` ceil) |
| Batched `4k` `B=8` Served Prefill | — | **`183.9/stream`, `1,471.5 total/s`** | — | **`48.1%`** |

### Correctness & Quality Verification

- **Logit & Greedy Parity**: `16/16` chat top-1 agreement (`KL = 2.48e-10` vs PyTorch Transformers MPS reference); `100%` byte-identical single-stream greedy speculative parity + lossless `T=0.7` rejection sampling; `4/4` decode-only `B=2` SIMT exact parity and `B=4` batched prefill `4/4` top-1 (`KL <= 2.88e-10`).
- **Downstream Accuracy & Long Context**: GSM8K-50 **`48–49 / 50` (`96–98%`)**; tool-call parser + HTTP **`20/20 + 6/6`**; **`131,014`-tok and `249,926`-tok needle retrieval `2/2 PASS`** with `-96 MB` swap delta (`29.74 GB` active / `38.64 GB` peak during `250k` soak).

---

## Model Size, Memory Budget & Context Scaling

### Exact Checkpoint Weight Footprint (`thinkingmachines/Inkling-2-Mini`)

| Component | Bytes | GiB | Precision / Notes |
|---|---:|---:|---|
| Routed experts (layers 0–19) | `11,466,199,200` | `10.679` | NVFP4 (E2M1 + E4M3 `[1,16]` scale + FP32 `amax`) |
| Routed experts (layers 20–23) | `4,204,265,472` | `3.916` | MXFP8 (F8_E4M3 + UE8M0 `[1,32]` scale) |
| Attention projections & norms (layers 0–23) | `1,307,510,784` | `1.218` | BF16 (`q/k/v/r/o/gate_proj`, `sconv`, `rel_proj`, norms) |
| Token embeddings (`[201024, 2304]`) | `926,318,592` | `0.863` | BF16 |
| LM head (`[201024, 2304]`) | `926,318,592` | `0.863` | BF16 (accumulated in FP32 without storing FP32 copy) |
| Loop controllers (layers 6–17) | `42,467,328` | `0.040` | BF16 message gates (`down`, `forget`, `gain`) |
| Routers, depth residuals & norms | `16,278,528` | `0.015` | BF16 + FP32 |
| **Text LLM Resident Total** | **`18,889,358,496`** | **`17.592`** | Main file `model-00001-of-00001.safetensors` (`19.53 GB` on disk incl. vision) |
| **MTP Drafter (`mtp.safetensors`)** | **`635,795,744`** | **`0.592`** | BF16 3-layer DSpark + folded `MarkovHead` (`+96 MiB` folded) |
| **Total Text + MTP Resident** | **`19,525,154,240`** | **`18.184`** | **`~20.4 GB`** MLX active after shader/buffer warmup (excl. eager vision) |

### KV Cache & Bandwidth Scaling Across Context Lengths (48 Layer Visits, 3 Loop Passes)

Per-stream BF16 KV & state memory (`n_kv = 8`, `head_dim = 128`, `2` bytes K + `2` bytes V = `4,096 B/tok/visit`):
- **40 sliding-window visits** (`window = 1,024`): capped at **`160 MiB` (`0.156 GiB`)** per stream (`min(ctx, 1024) * 163,840 B`).
- **8 full-attention visits**: grows linearly at **`32 KiB/tok` (`32,768 B/tok`)** (`1 GiB` per `32,768` tokens).
- **ShortConv states**: `~3.8 MiB` per stream (fixed).
- **MTP Draft KV (`B <= 4`)**: `1` global draft layer (`4,096 B/tok` = `1.00 GiB` at `256k`) + `8 MiB` local layers.

Total DRAM read per plain `B=1` decode step = **`5.493 GB` active weights + KV bytes read**:

| Context (`ctx`) | KV / Stream | Total Resident (`B=1` w/ MTP) | 4-Stream KV (`B=4`) | Max Clean Streams (`<= 51.8 GiB` GPU WS) | `B=1` BW Read / Step | Theoretical `B=1` Plain Ceiling (`@ 282.6 GB/s`) |
|---:|---:|---:|---:|---:|---:|---:|
| **`512`** | `0.092 GiB` | `~20.5 GiB` | `0.37 GiB` | `64+` (max-batch / scratch limited) | `5.59 GB` | `50.5 tok/s` |
| **`4,096`** | `0.281 GiB` | `~20.7 GiB` | `1.13 GiB` | `64+` (`32` verified) | `5.79 GB` | `48.8 tok/s` |
| **`16,384`** | `0.656 GiB` | `~21.1 GiB` | `2.63 GiB` | `~39` | `6.20 GB` | `45.6 tok/s` |
| **`32,768`** | `1.156 GiB` | `~21.7 GiB` | `4.63 GiB` | `~22` | `6.74 GB` | `41.9 tok/s` |
| **`65,536`** | `2.156 GiB` | `~22.8 GiB` | `8.63 GiB` | `~11` | `7.81 GB` | `36.2 tok/s` |
| **`131,072` (128k)** | `4.156 GiB` | `~25.1 GiB` | `16.63 GiB` | `~6` | `9.96 GB` | `28.4 tok/s` |
| **`262,144` (256k)** | `8.156 GiB` | `~29.6 GiB` | `32.63 GiB` | `3` (plain, w/ `4 GB` cache) | `14.25 GB` | `19.8 tok/s` |

*Above table uses `--cache-limit-gb 4.0` + `~1.5–2.0 GiB` prefill scratch inside M5 Pro's `51.84 GiB` (`55.66 GB`) recommended GPU working set.*

---

## Included MLX Optimization Skill

See [`skills/mlx-apple-silicon-inference/SKILL.md`](skills/mlx-apple-silicon-inference/SKILL.md) for a reusable Apple Silicon MLX and Metal 4 inference optimization skill covering roofline measurement, single-command-buffer runners, `MTLMathModeSafe`, MoE regime specialization, Metal 4 NAX kernels, zero-mask FlashAttention, GPU speculative acceptance, and macOS VM safety rules.

---

## Testing

```bash
PYTHONPATH=src python -m unittest tests/test_engine.py -v
PYTHONPATH=src python tests/test_flash_attn.py
PYTHONPATH=src python tests/test_prefill_moe.py
PYTHONPATH=src python tests/test_mtp_draft.py
```

All `tests/test_engine.py`, `tests/test_flash_attn.py`, and `tests/test_prefill_moe.py` suites run on tiny synthetic models (`< 64 MB`) without downloading the 19.5 GB checkpoint. `tests/test_mtp_draft.py` tests against `mtp.safetensors` (`636 MB`) if present locally and skips cleanly otherwise.

---

## License

MIT — see [`LICENSE`](LICENSE).
