# inkling-mlx

**Fast local inference for [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on Apple Silicon (MLX + Metal 4).**

- **GitHub (Code & Native Metal 4 Engine):** [`github.com/Balllvin/inkling-mlx`](https://github.com/Balllvin/inkling-mlx)
- **Hugging Face (Model Card & Setup Mirror):** [`huggingface.co/balllvin/Inkling-2-Mini-MLX`](https://huggingface.co/balllvin/Inkling-2-Mini-MLX)
- **Base Weights (Apache 2.0):** [`huggingface.co/thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
- **Included Optimization Playbook:** [`skills/mlx-apple-silicon-inference/SKILL.md`](skills/mlx-apple-silicon-inference/SKILL.md)

---

## Why `inkling-mlx`?

Standard Mac runners lose massive performance on `Inkling-2-Mini` due to its **3-pass looped architecture (48 layer visits per token)**, **mixed NVFP4 + MXFP8 experts**, **unbounded GPU command buffer flushes**, **multi-GB prefill attention masks**, and **unused built-in speculative drafter**.

`inkling-mlx` runs the **exact unmodified `thinkingmachines/Inkling-2-Mini` checkpoint** — same **3 loop passes**, same **unquantized 256k BF16 KV cache**, same **NVFP4 / MXFP8 / BF16 precision**, and the built-in **`mtp.safetensors` drafter** — with custom Metal 4 and MLX kernels behind an OpenAI-compatible HTTP/SSE API (`/v1/chat/completions`, `/v1/models`, `/health`):

- **Up to `4.6x` faster single-agent output (`120 tok/s`):** Unifies GPU draft + verify + rejection sampling into a single Metal command buffer with zero accuracy loss (`100%` byte-identical greedy output).
- **Up to `3.0x` faster for 1–4 concurrent local agents (`117.7 tok/s`):** Built specifically for real local multi-agent workflows (`B=1..4`) without cloud API costs.
- **Up to `7.7x` faster batched throughput (`277–335 tok/s`):** Scales smoothly up to `B=64` with zero swap inside 64 GB unified memory, and fixes a `49.8x` long-context batch collapse (`16k B=4`: `1.87 -> 93.12 tok/s`).
- **Nearly `2x` faster cold prompt prefill:** Cuts cold `16k` time-to-first-token from `21.4 s` to `11.14 s` (`1,464 tok/s` served, `1,739 tok/s` in-process at `4k`).
- **Bit-exact quality (`KL = 2.48e-10`):** Matches PyTorch Transformers logit-for-logit (`16/16` chat top-1, `96–98%` GSM8K-50, `2/2` needle retrieval up to `250k` tokens).

### At-a-Glance Before vs After (`Apple M5 Pro 64 GB`)

| Workload | Before (Baseline) | After (`inkling-mlx`) | Speedup |
|---|---:|---:|---:|
| **Single User (`B=1` Plain, `ctx=512`)** | `25.98 tok/s` *(stock `mlx-vlm`)* | **`48.97 tok/s`** | **`1.88x`** |
| **Single Agent w/ Tools (`B=1` MTP, `T=0.0`)** | `25.98 tok/s` *(stock `mlx-vlm`)* | **`120.19 tok/s`** | **`4.63x`** |
| **4 Concurrent Agents (`B=4` MTP, `ctx=512`)** | `39.31 tok/s` agg *(stock `mlx-vlm`)* | **`117.68 tok/s` agg** (`27.59/stream`) | **`2.99x`** |
| **32 Concurrent Streams (`B=32`, `ctx=512`)** | `36.10 tok/s` agg *(initial MLX)* | **`277.11 tok/s` agg** (`8.75/stream`) | **`7.68x`** |
| **Max Batch Clean (`B=64`, `ctx=512`)** | *(OOM / collapse)* | **`335.48 tok/s` agg** (`41.3 GB` peak, `0` swap) | — |
| **Cold `16k` Prompt TTFT (`B=1`)** | `20.58–21.4 s` | **`11.14 s`** (`1,464.3 tok/s`) | **`1.85x–1.92x`** |
| **Cold `16k` 4-Stream Decode (`B=4`)** | `1.87 tok/s` agg *(initial MLX)* | **`93.12 tok/s` agg** (`23.28/stream`) | **`49.8x`** |

> **Hardware target:** Built and tuned specifically for **Apple M5** (per-core Metal 4 Neural Accelerators / NAX). Measured and verified on a **MacBook Pro (`Mac17,9`) with Apple M5 Pro (18-core CPU: 6 Super + 12 Performance, 20-core GPU, Metal 4), 64 GB unified memory, macOS 26.5.1 (Xcode 26.5 SDK), MLX 0.32.4**. On M1–M4 Macs or `< macOS 26`, pass `--skip-build` for the pure MLX fallback or see [Chip Compatibility & Porting Guide](#chip-compatibility--porting-guide).

---

## Getting Started

### Prerequisites, Model to Download & RAM Requirements

#### Model Weights

`inkling-mlx` downloads and runs Thinking Machines Lab's official [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) checkpoint unmodified (`~20.2 GB` download, **`>= 25 GB` free SSD space** recommended):

- `model-00001-of-00001.safetensors` (`~19.53 GB`, mixed NVFP4 / MXFP8 / BF16 weights)
- `mtp.safetensors` (`~0.64 GB`, built-in 3-layer speculative drafter)
- `config.json` and tokenizer files

Fetch it explicitly into `~/models/Inkling-2-Mini` via `inkling-mlx download`, or pass `--model thinkingmachines/Inkling-2-Mini` directly to `inkling-mlx serve` for automatic download and caching. *(Note: text generation and OpenAI tool calling are supported; the vision encoder is not wired into this MLX server.)*

#### Unified RAM Tiers

| Unified Memory | Status | Recommended Setup / Limits |
|---|---|---|
| **16–24 GB** | Not supported | Resident text + MTP runtime exceeds `20 GB` even before KV cache and prefill scratch. |
| **32 GB** | Minimum | Single-stream (`B=1`) short/medium context (`<= 16k–32k`); run with `--no-mtp --cache-limit-gb 2.0`. |
| **48 GB** | Supported | `B=1..4` with MTP up to `64k–128k`, or `B=1` up to `256k` with `--no-mtp`. |
| **64 GB** | **Recommended (Tested)** | Full `256k` BF16 KV context with MTP (`~29.6 GiB` active, `~38.6 GB` peak, `0` swap), `B=1..4` concurrent MTP agents, or up to `B=64` batched decode at `ctx=512` (`41.3 GB` peak, `0` swap). |
| **96–128 GB+** | Supported | Multiple concurrent `128k–256k` streams and larger prefix caches. |

#### Software Requirements

- **Apple M5 native path (recommended):** Apple M5 / M5 Pro / M5 Max / M5 Ultra Mac, **macOS 26+**, Xcode Command Line Tools (`xcode-select --install`), Python `3.11–3.13` (or `uv`).
- **M1–M4 / older macOS fallback (untested):** Metal 3.1+ (macOS 14+), Python `3.11–3.13`, started with `inkling-mlx serve --skip-build`.

### Agent Quickstart Prompt

Copy and paste this prompt into any coding agent on an Apple Silicon Mac:

```text
Set up and verify `inkling-mlx` on this Mac:
1. Check hardware and memory (`system_profiler SPHardwareDataType`): confirm Apple Silicon with >= 32 GB unified RAM (64 GB recommended). Confirm whether this Mac has an Apple M5-family chip and macOS 26+ with Xcode Command Line Tools installed.
2. Clone `https://github.com/Balllvin/inkling-mlx` (or `https://huggingface.co/balllvin/Inkling-2-Mini-MLX`) and `cd` into it.
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

### Manual Setup & Serving

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

## Verified Benchmarks (Before vs After on Apple M5 Pro 64 GB)

All numbers below were measured on a **MacBook Pro (`Mac17,9`, Apple M5 Pro 20-core GPU, 64 GB RAM, macOS 26.5.1, MLX 0.32.4, AC power)** over the live HTTP server unless noted otherwise. Hardware ceilings measured on this machine:
- **Sustained DRAM Read Bandwidth:** **`282.6 GB/s`** (theoretical `B=1` plain ceilings: `50.5 tok/s` @ `512`, `48.8` @ `4k`, `45.6` @ `16k`, `20.4` @ `250k`)
- **Peak BF16 Compute:** **`29.38 TFLOP/s`** (theoretical prefill ceilings: `3,210 tok/s` @ `<= 4k`, `3,060 tok/s` @ `16k`)

> *Note on host load:* Rows labeled `(thermalState=1)` were measured while the machine was also running unrelated background CPU/GPU work (sustained DRAM bandwidth was `140–195 GB/s` instead of `282.6 GB/s` idle), so idle speeds are equal or higher.

### 1. Interactive & Low-Concurrency Local Agents (`B = 1..4`, `ctx=512`, Cold Prefill)

| Scenario | Before (Baseline) | `inkling-mlx` | Speedup | % Plain BW Ceiling |
|---|---|---|---:|---:|
| **`B=1` Plain Decode** (`T=0.0`) | `25.98 tok/s`, `841 ms` TTFT *(stock `mlx-vlm 0.8.0`)* | **`48.97 tok/s`**, `576 ms` TTFT | **`1.88x`** | **`97.0%`** |
| **`B=1` MTP Tool Calling** (`T=0.0`) | `25.98 tok/s` *(stock `mlx-vlm 0.8.0`)* | **`120.19 tok/s`** (`acc=6.67`) | **`4.63x`** | **`236.4%`** |
| **`B=1` MTP Code Refactor** (`T=0.0`) | `25.98 tok/s` *(stock `mlx-vlm 0.8.0`)* | **`86.86 tok/s`** | **`3.34x`** | **`170.3%`** |
| **`B=1` MTP 6-Prompt Median** (`T=0.0`) | `25.98 tok/s` *(stock `mlx-vlm 0.8.0`)* | **`66.79 tok/s`** | **`2.57x`** | **`130.5%`** |
| **`B=1` MTP Sampled** (`T=0.7`, `thermalState=1`) | — | **`50.05 tok/s`** med (**`97.91/s`** tool) | — | **`97.8%–192.6%`** |
| **`B=2` Plain Batched** (`T=0.0`) | `35.73 tok/s` agg (`18.75/stream`, `mlx-vlm`) | **`64.75 tok/s` agg** (`32.38/stream`) | **`1.81x`** | **`85.1%`** |
| **`B=2` Unified MTP** (`T=0.0`, `thermalState=1`) | `35.73 tok/s` agg *(stock `mlx-vlm 0.8.0`)* | **`94.98 tok/s` agg** (`47.49/stream`) | **`2.66x`** | **`122.7%`** |
| **`B=2` Unified MTP Sampled** (`T=0.7`, `thermalState=1`) | — | **`91.69 tok/s` agg** (`45.84/stream`) | — | **`118.4%`** |
| **`B=4` Plain Batched** (`T=0.0`) | `39.31 tok/s` agg (`14.48/stream`, `mlx-vlm`) | **`107.06 tok/s` agg** (`26.77/stream`) | **`2.72x`** | **`102.0%`** |
| **`B=4` Unified MTP** (`T=0.0`, `thermalState=1`) | `39.31 tok/s` agg *(stock `mlx-vlm 0.8.0`)* | **`117.68 tok/s` agg** (`27.59/stream`) | **`2.99x`** | **`108.9%`** |
| **`B=4` Unified MTP Sampled** (`T=0.7`, `thermalState=1`) | — | **`101.73 tok/s` agg** (`21.91/stream`) | — | **`81.1%`** |

### 2. High-Concurrency Batched Throughput (`B = 8..64`, Cold Prefill)

| Batch & Context | Before (Baseline) | `inkling-mlx` | Speedup | Notes / Memory |
|---|---|---|---:|---|
| **`B=8`, `ctx=512`** | `44.70 tok/s` agg (`10.13/s`), `6.55 s` TTFT *(stock `mlx-vlm`)* | **`160.19 tok/s`** (`20.03/stream`), `3.10 s` TTFT | **`3.58x`** | `115.9%` BW ceil |
| **`B=16`, `ctx=512`** | `35.11 tok/s` agg (`3.40/s`), `10.23 s` TTFT *(initial MLX engine)* | **`218.33 tok/s`** (`13.86/stream`), `6.24 s` TTFT | **`6.22x`** | `117.2%` BW ceil |
| **`B=32`, `ctx=512`** | `36.10 tok/s` agg (`1.83/s`), `18.25 s` TTFT *(initial MLX engine)* | **`277.11 tok/s`** (`8.75/stream`), `12.28 s` TTFT | **`7.68x`** | `100.1%` BW ceil |
| **`B=48`, `ctx=512`** | — | **`308.17 tok/s`** (`6.48/stream`) | — | `thermalState=1`, `0` swap |
| **`B=64`, `ctx=512`** | *(OOM / collapse)* | **`335.48 tok/s`** (`5.28/stream`) | — | `41.3 GB` peak, `0` swap |
| **`B=32`, `ctx=4,096`** | *(collapse)* | **`235.80 tok/s`** (`7.49/stream`) | — | `101.8%` ceil, `41.3 GB` peak, `0` swap |

### 3. Long Context (`4k`, `16k`, `250k`) & Cold Prefill / TTFT

| Condition | Before (Baseline) | `inkling-mlx` | Speedup | % Hardware Ceiling |
|---|---|---|---:|---:|
| **Cold `4k` `B=1` Decode / TTFT** | `23.48 tok/s` / `5.44 s` (`756/s`, `mlx-vlm`) | **`48.28 tok/s`** / **`2.98 s`** (`1,381/s` served; `1,739/s` in-proc) | **`2.06x` dec / `1.83x` TTFT** | `99.0%` dec / `45.1%–54.2%` prefill |
| **Cold `4k` `B=4` Decode / TTFT** | `10.23 tok/s` / `26.09 s` *(stock `mlx-vlm`)* | **`105.94 tok/s`** (`26.50/stream`) / **`12.52 s`** | **`10.36x` dec / `2.08x` TTFT** | `108.4%` dec / `43.0%` prefill |
| **Cold `4k` `B=8` Decode / TTFT** | `10.04 tok/s` / `48.61 s` *(stock `mlx-vlm`)* | **`139.91 tok/s`** (`17.49/stream`) / **`22.38 s`** (`1,471.5/s`) | **`13.94x` dec / `2.17x` TTFT** | `111.1%` dec / `48.1%` prefill |
| **Cold `16k` `B=1` Decode / TTFT** | `24.76 tok/s` / `20.58–21.4 s` (`762–792/s`) | **`42.69 tok/s`** / **`11.14 s`** (`1,464.3/s`, `thermalState=1`) | **`1.72x` dec / `1.85x–1.92x` TTFT** | `93.6%` dec / `47.9%` prefill |
| **Cold `16k` `B=4` Decode** | `1.87 tok/s` agg (`1.57/s`, *initial MLX engine*) | **`93.12 tok/s` agg** (`23.28/stream`) | **`49.8x`** | `108.4%` BW ceil |
| **`250k` Context (`250,102` tok, `B=1`)** | *(untested / OOM)* | **`19.79 tok/s`** (`29.74 GB` active / `38.64 GB` peak) | — | `97.1%` BW ceil (`2/2` needle PASS) |

### 4. End-to-End Multi-Turn Agentic Coding Suite (`20` Real Tool-Using Tasks)

| Engine & Concurrency | Tasks Passed | Median Task Wall | Total Suite Wall | Throughput | Speedup |
|---|---:|---:|---:|---:|---:|
| **Initial MLX Engine (`c=4`)** | `18/20` (`90%`) | `171.40 s` | `1,053.6 s` | `1.14 tasks/min` | `1.0x` |
| **`inkling-mlx` (`c=1` w/ MTP, `5.33` avg acc)** | **`19/20` (`95%`)** | **`34.40 s`** | `979.4 s` | `1.23 tasks/min` | **`4.98x`** / task |
| **`inkling-mlx` (`c=4`)** | **`20/20` (`100%`)** | **`111.64 s`** | **`672.3 s`** | **`1.78 tasks/min`** | **`1.57x`** total |
| **`inkling-mlx` (`c=8`)** | **`19/20` (`95%`)** | `170.26 s` | **`614.6 s`** | **`1.95 tasks/min`** | **`1.71x`** total |

### 5. Quality Verification & Official Upstream Model Benchmarks

Because `inkling-mlx` keeps all released weights (`NVFP4` / `MXFP8` / `BF16`), all `3` loop passes (`48` layer visits), and full `BF16` KV cache precision untouched:
- **Logit & Speculative Equivalence:** `16/16` chat top-1 agreement (`mean KL = 2.48e-10` vs PyTorch Transformers MPS reference); `100%` byte-identical single-stream greedy speculative parity + lossless `T=0.7` rejection sampling; `4/4` decode-only `B=2` SIMT exact parity and `B=4` batched prefill `4/4` top-1 (`KL <= 2.88e-10`).
- **Downstream Verification on M5 Pro:** GSM8K-50 **`48–49 / 50` (`96–98%`)**; tool-call parser + live HTTP **`20/20 + 6/6`**; **`131,014`-tok and `249,926`-tok needle-in-a-haystack retrieval `2/2 PASS`** (`-96 MB` swap delta).
- **Official Thinking Machines Lab Scores (`thinkingmachines/Inkling-2-Mini`):** `26.4B` total / `5.4B` effective active (`5/26B`): **SWEBench Verified `79.2%`**, **SWEBench Pro v2 `77.9%`**, **Vibe Code Bench v1.1 `77.9%`**, **SWEBench Multilingual `73.7%`**, **Terminal Bench v2.1 `71.6%`**, **DeepSearchQA `85.3%`**, **MCP Atlas `82.7%`**, **BrowseComp `79.0%`**, **ARC-AGI v1 / v2 `84.5% / 49.3%`**, **GraphWalks BFS 256K `69.7%`**, **LongBench v2 `66.9%`**. *(See [`balllvin/Inkling-2-Mini-MLX`](https://huggingface.co/balllvin/Inkling-2-Mini-MLX) for the complete official table.)*

---

## Deep Technical Details: How the Optimizations Work

Stock MLX hits four bottlenecks on `Inkling-2-Mini`: ~10 GPU command-buffer flushes per token (`max_ops_per_cb = 50`), redundant expert loads / `4.3x` `SwitchGLU` sorting cliffs across batch sizes, repeated `lm_head` reads at batching, and multi-GB attention masks + unbatched conv/norm dispatches during prefill. `inkling-mlx` solves each stage without sacrificing numerical fidelity:

1. **Single-Command-Buffer Native Metal 4 Decode (`src/inkling_mlx/native/decode_runner.mm`)**:
   - Executes all `48` layer visits (embedding, RMSNorm, attention/MLP depth residuals, message gates, packed QKV/gate, causal `ShortConv`, chronological `ring_idx` split-K GQA, FP32 MoE routing, NVFP4/MXFP8 experts, `lm_head`, and greedy `head_argmax`) inside **one `MTLCommandBuffer`** (`MTLDispatchTypeSerial`) via zero-copy `16 KB` page-aligned `MTLBuffer` wrappers.
   - Eliminates `~5–8 ms` per step of Python graph-building and command-buffer flush latency.
2. **Unified `B=1..4` GPU-Accept MTP Speculative Decoding (`decode_runner.mm` + `mtp_draft_runner.mm`)**:
   - Runs the built-in 3-layer DSpark + folded FP64 `MarkovHead` drafter (`mtp.safetensors`, `636 MB`) in a single native Metal CB, then verifies `sum(K_i + 1) <= 32` draft tokens across `1..4` unmerged concurrent streams in a single target verify CB — running greedy or exact Leviathan rejection sampling on the GPU, committing/rolling back target KV + draft KV + `ShortConv` states, and capturing target hidden states with **zero intermediate CPU waits**.
   - Automatically triggers Metal 4 Neural Accelerator (NAX) batched kernels at `M >= 9`, caps adaptive `K <= 5` for `B >= 3` to bound verification tile cost, uses an adaptive cooldown burst (`ema_acc < 1.65`) when speculation efficiency dips on `B=1`, and seamlessly transitions to `BatchKVCache` continuous batching when `B >= 5`.
3. **Regime-Specialized Quantized MoE Kernels (`src/inkling_mlx/moe_kernels.py` & `native/prefill_moe_nax.mm`)**:
   - **`M = 1..2`**: Direct 1-expert-per-threadgroup kernel (`moe_m1_experts`) with inline `dot8_nvfp4` (`97–99%` DRAM bandwidth).
   - **`M = 3..112`**: Zero-spill compact `(tile_expert, tile_t0)` tile-table grouped kernel (`moe_grouped_tiles`) using `#pragma unroll` scalar accumulators (removing LLVM stack spills), vectorized `uint4` loads, and a `256`-entry `half2` E2M1 LUT (`72–89%` DRAM BW, `0.00e+00` batch invariance across `M`, reads `lm_head` only once per step).
   - **`M >= 128`**: Metal 4.0 NAX (`mpp::tensor_ops::matmul2d`) double-buffered grouped MoE with GPU radix bucketing, indirect dispatch, and a fused 64-channel 128B-aligned combine+residual dispatch (`21.9–27.1 TFLOP/s`).
4. **Metal 4.0 NAX Zero-Mask FlashAttention (`src/inkling_mlx/native/flash_attn_nax.mm`)**:
   - Computes causal bounds, sliding window (`w=1024`), and ragged per-sequence `key_start` arithmetically inside tiles, avoiding `107 GB` of dense mask traffic at `16k` context and scaling cleanly to `256k`.
   - Computes hardware NAX Toeplitz relative attention bias (`16x32x16`) and exact FP32 query-dependent sinks directly in online softmax (`3.6x` faster SWA, `2.5x` faster `16k` global).
5. **`MTLMathModeSafe` Occupancy & Exact Numerical Parity**:
   - Compiling decode/MoE/MTP compute pipelines with `MTLMathModeSafe` disables floating-point reassociation on Apple M5 (`applegpu_g17s`), cutting live register pressure and boosting SIMD occupancy (`11–29%` speedup) while preserving `0.00e+00` BF16 bit-exactness.
6. **Memory, Prefix Sharing & macOS VM Safety (`src/inkling_mlx/engine.py`, `native_decode.py`, `native_mtp.py`)**:
   - Shares long system-prompt prefixes (`>= 128` tokens) across concurrent burst arrivals (`~93%` cache hit rate on multi-turn tool trajectories), pre-allocates `GrowingKVCache` cleanly with `--cache-limit-gb 4.0`, checks `PrefixCache` byte limits **before** cloning `> 8 GB` `256k` KV caches (`-12.5 GB` peak spike eliminated), never caches VA-keyed mutable `MTLBuffer` wrappers across MLX frees, and validates object `is` identity alongside `id()` to defeat CPython LIFO freelist aliasing.

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
| **Total Text + MTP Resident** | **`19,525,154,240`** | **`18.184`** | **`~20.4 GB`** MLX active after warmup (excl. eager vision) |

### KV Cache & Bandwidth Scaling Across Context Lengths (`48` Visits, `3` Loop Passes)

Per-stream BF16 KV & state memory (`n_kv = 8`, `head_dim = 128`, `2` bytes K + `2` bytes V = `4,096 B/tok/visit`):
- **40 sliding-window visits** (`window = 1,024`): capped at **`160 MiB` (`0.156 GiB`)** per stream (`min(ctx, 1024) * 163,840 B`).
- **8 full-attention visits**: grows linearly at **`32 KiB/tok` (`32,768 B/tok`)** (`1 GiB` per `32,768` tokens).
- **ShortConv states**: `~3.8 MiB` per stream; **MTP Draft KV (`B <= 4`)**: `1` global layer (`4,096 B/tok` = `1.00 GiB` at `256k`) + `8 MiB` local layers.

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

*Uses `--cache-limit-gb 4.0` + `~1.5–2.0 GiB` prefill scratch inside M5 Pro's `51.84 GiB` (`55.66 GB`) recommended GPU working set.*

---

## Chip Compatibility & Porting Guide

This engine is **built and tuned specifically for Apple M5** (`Apple10` GPU family: M5, M5 Pro, M5 Max, M5 Ultra) and has **only been tested on Apple M5 Pro (`Mac17,9`, 20-core GPU, 64 GB unified RAM, macOS 26.5.1)**.

### Why Older M-Series Chips (M1 / M2 / M3 / M4) Are Not Natively Supported Out of the Box

| Subsystem | Apple M5 (`Apple10`, macOS 26+) | Apple M1 / M2 / M3 / M4 (`Apple7`–`Apple9`) | Why |
|---|---|---|---|
| **Metal 4.0 NAX FlashAttention** (`native/flash_attn_nax.mm`) | Native 4x speedup | Not supported natively | Uses `MetalPerformancePrimitives` (`mpp::tensor_ops::matmul2d`) targeting M5 per-core Neural Accelerators and `MTLLanguageVersion4_0`. |
| **Metal 4.0 NAX Prefill MoE** (`native/prefill_moe_nax.mm`) | Native 21–27 TFLOP/s | Not supported natively | Uses `MTLLanguageVersion4_0` + `mpp::tensor_ops::matmul2d` (`M >= 128`) unavailable on M1–M4 hardware. |
| **Native 1-CB Decode & MTP** (`native/decode_runner.mm`, `native/mtp_draft_runner.mm`) | Compiles & runs (`B=1..4` MTP + batched) | Fails build / init without edits | `mtp_draft_runner.mm` hardcodes `MTLLanguageVersion4_0` even though its shaders are pure SIMT, and `decode_runner.mm` aborts init if `lib_nax` (`MTLLanguageVersion4_0`, used at `M >= 9`) fails to compile. |
| **MLX + Metal 3.1 SIMT Fallback** (`fused.py`, `kernels.py`, `moe_kernels.py`, `mtp.py`) | Available | **Supported in code (untested on M1–M4)** | `kernels.py` and `moe_kernels.py` use standard Metal 3.1 (`simdgroup_bfloat8x8`, `moe_m1_experts`, `moe_grouped_tiles`) and all native `.dylib` loaders return `None` when `.dylib`s are absent. |
| **NVIDIA CUDA / AMD ROCm / Intel / CPU-only** | — | Not supported | Engine depends on Apple Unified Memory, Metal, and `mlx`. |

> **Running on M1 / M2 / M3 / M4 today (untested):** If `inkling-mlx build` or `scripts/build_native.sh` fails on `< macOS 26` or non-M5 hardware, run **`inkling-mlx serve --skip-build --model ...`**. All four native loaders (`try_create_native_decode_runner`, `try_create_native_mtp_runner`, `_get_flash_attn_lib`, `_get_prefill_moe_lib`) return `None` when the `.dylib`s are absent or fail to init, falling back automatically to the pure MLX + Metal 3.1 SIMT kernels (`moe_m1_experts`, `moe_grouped_tiles`, `relative_attention_nax`). Because we do not have M1–M4 machines here, that fallback has not been benchmarked on M1–M4.

### How to Fork / Adapt This Repo for Apple M1–M4 (or Other Chips)

1. **Guard Metal 4.0 NAX in `src/inkling_mlx/native/decode_runner.mm`**:
   - Wrap the `if (cfg->max_m >= 8)` NAX compilation block (`MTLLanguageVersion4_0` / `kMetal4NaxSource`) in `#if defined(__MAC_26_0)` and leave `ctx->lib_nax = nil` on failure instead of returning `nullptr` from init.
   - Gate `use_nax_qkv` and `use_nax_head` (`M >= 9` paths) on `(ctx->lib_nax != nil)` instead of returning `-12`, so `M >= 9` dispatches fall through to the existing SIMT kernels in `decode_runner.mm`.
2. **Lower `MTLLanguageVersion` in `src/inkling_mlx/native/mtp_draft_runner.mm`**:
   - Change `opts.languageVersion = MTLLanguageVersion4_0` to `MTLLanguageVersion3_2` (or `MTLLanguageVersion3_1` on macOS 14). The MTP drafter kernels are pure SIMT / GEMV and contain no `mpp::tensor_ops` calls.
3. **Make NAX prefill `.dylib`s optional in `scripts/build_native.sh` and `src/inkling_mlx/cli.py`**:
   - Build only `libinkling_decode.dylib` and `libinkling_mtp_draft.dylib` when targeting M1–M4 or Xcode `< 26`. Python prefill already falls back to MLX `relative_attention_nax` (`kernels.py`) and `moe_grouped_tiles` / `mx.gather_qmm` (`moe_kernels.py`) when those two `.dylib`s are absent.
4. **Re-tune and gate parity on target silicon**:
   - Benchmark `MTLMathModeSafe` vs fast math and threadgroup widths (`128` vs `256`) on `Apple7`/`Apple8`/`Apple9` register files, and verify parity with `tests/test_engine.py` and `tests/test_decode_runner.py`.
   - *(Note on `simdgroup_bfloat8x8`: supported on Metal 3.1+ / M1+ `Apple7` and newer, so `kernels.py` BF16 SIMD-group GEMMs work on M1–M4 under macOS 14+.)*
5. **Porting to CUDA / ROCm / non-Apple accelerators**:
   - Reuse the model spec and scheduler in [`model.py`](src/inkling_mlx/model.py), [`config.py`](src/inkling_mlx/config.py), [`mtp.py`](src/inkling_mlx/mtp.py), and [`parser.py`](src/inkling_mlx/parser.py) (3-loop 48-visit core with depth residuals & message gates, causal `ShortConv`, hybrid SWA + Toeplitz relative bias + query-dependent sinks, NVFP4 layers 0–19 / MXFP8 layers 20–23 MoE, built-in DSpark + MarkovHead MTP), replacing MLX + `.mm` Metal runners with CUDA Graphs, Triton, FlashAttention, and CUTLASS FP4/FP8 grouped GEMMs.

---

## Included MLX Optimization Skill, Testing & License

- **Skill:** [`skills/mlx-apple-silicon-inference/SKILL.md`](skills/mlx-apple-silicon-inference/SKILL.md) — reusable Apple Silicon MLX and Metal 4 optimization playbook covering roofline profiling, single-command-buffer runners, `MTLMathModeSafe`, MoE regime specialization, Metal 4 NAX kernels, zero-mask FlashAttention, GPU speculative acceptance, and macOS VM safety.
- **Testing:**

```bash
PYTHONPATH=src python -m unittest tests/test_engine.py -v
PYTHONPATH=src python tests/test_flash_attn.py
PYTHONPATH=src python tests/test_prefill_moe.py
PYTHONPATH=src python tests/test_mtp_draft.py
```

`tests/test_engine.py`, `tests/test_flash_attn.py`, and `tests/test_prefill_moe.py` run on tiny synthetic models (`< 64 MB`) without downloading the 19.5 GB checkpoint. `tests/test_mtp_draft.py` verifies against local `mtp.safetensors` (`636 MB`) if present and skips cleanly otherwise.

- **License:** Engine code is MIT ([`LICENSE`](LICENSE)). Upstream [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) weights remain licensed under **Apache 2.0** by Thinking Machines Lab, Inc.
