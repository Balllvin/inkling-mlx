# Agent Instructions — inkling-mlx

Use this file when building, testing, editing, or serving `inkling-mlx` on Apple Silicon.

Targets **Apple M5 (`Apple10` / Metal 4 NAX on macOS 26+, >= 32 GB unified RAM; tested on Apple M5 Pro 64 GB)** for full native throughput. On older M1–M4 Macs or `< macOS 26`, pass `--skip-build` to `inkling-mlx serve` to fall back to the pure MLX + Metal 3.1 path (untested on M1–M4); see [README.md](README.md#chip-compatibility--porting-guide) for fork instructions.

## Repository Layout

- `src/inkling_mlx/`: Python package implementing the model (`model.py`, `config.py`), fused Metal MLX ops (`kernels.py`, `moe_kernels.py`, `fused.py`), built-in MTP drafter (`mtp.py`), zero-copy native ctypes wrappers (`native_decode.py`, `native_mtp.py`), continuous-batching scheduler & HTTP server (`engine.py`, `serve.py`, `parser.py`, `cli.py`).
- `src/inkling_mlx/native/`: Standalone Objective-C++ & Metal 4 runners (`decode_runner.{h,mm}`, `flash_attn_nax.{h,mm}`, `mtp_draft_runner.{h,mm}`, `prefill_moe_nax.{h,mm}`).
- `scripts/`: `build_native.sh` and `setup.sh`.
- `tests/`: Synthetic engine and native Metal kernel test suites (`test_engine.py`, `test_decode_runner.py`, `test_flash_attn.py`, `test_prefill_moe.py`, `test_mtp_draft.py`).
- `skills/mlx-apple-silicon-inference/SKILL.md`: Apple Silicon MLX & Metal 4 optimization playbook.

## Build, Test, and Serve Commands

- Build all 4 Metal 4 `.dylib` libraries (`0` warnings required):
  `bash scripts/build_native.sh` (or `inkling-mlx build`)
- Run unit and server SSE tests (synthetic, fast):
  `PYTHONPATH=src python -m unittest tests/test_engine.py -v`
- Run native FlashAttention, NAX Prefill MoE, and MTP draft parity suites:
  `PYTHONPATH=src python tests/test_flash_attn.py`
  `PYTHONPATH=src python tests/test_prefill_moe.py`
  `PYTHONPATH=src python tests/test_mtp_draft.py`
- Run full native decode/verify/batched-MTP parity checks (`test_decode_runner.py`).
- Download weights from Hugging Face:
  `inkling-mlx download --repo thinkingmachines/Inkling-2-Mini --local-dir ~/models/Inkling-2-Mini`
- Start OpenAI-compatible server:
  `inkling-mlx serve --model ~/models/Inkling-2-Mini --host 127.0.0.1 --port 8095`

## Immutable Invariants

1. **Zero code comments**: Keep all Python, C/C++/Objective-C++, Metal, and shell source files free of inline comments (`//`, `/* */`, `#` outside line-1 shebangs). Keep concise docstrings only on public modules, classes, and entrypoints when helpful.
2. **Weights, loops, and KV fidelity**: Never requantize released NVFP4/MXFP8/BF16 weights, never drop loop passes (must execute all 3 passes / 48 layer visits), and never quantize the BF16 KV cache.
3. **One resident model process**: Never load multiple copies of the 19.5 GB checkpoint concurrently on a 64 GB unified-memory Mac.
4. **Metal & memory safety rules**:
   - Never persist virtual-address-keyed `MTLBuffer` wrappers for mutable KV, conv, scratch, or activation buffers across calls (macOS VM reuses freed VAs with new physical pages). Cache persistent mappings only for immutable weights.
   - Always hold strong Python object references and verify object `is` identity in pointer caches (CPython LIFO freelists reuse `id()` values on back-to-back requests).
   - Keep `MTLMathModeSafe` on decode/MoE/MTP compute pipelines for register occupancy and bit-exact BF16 rounding.
   - Always enforce `PrefixCache` byte-size checks before cloning KV caches (`256k` caches exceed 8 GB/stream).
5. **Verification before changes land**: After any `.mm`, `.h`, or `.py` edit, rebuild `.dylib` files via `bash scripts/build_native.sh` and verify `tests/test_engine.py`, `tests/test_flash_attn.py`, and `tests/test_prefill_moe.py` pass before committing. Never commit compiled `.dylib` files.
