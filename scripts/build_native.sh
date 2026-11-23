#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NATIVE_DIR="$ROOT/src/inkling_mlx/native"

clang++ -ObjC++ -std=c++20 -O3 -Wall -Wextra -Wno-unused-parameter -fobjc-arc -shared -fPIC \
  -framework Metal -framework Foundation \
  "$NATIVE_DIR/decode_runner.mm" \
  -o "$NATIVE_DIR/libinkling_decode.dylib"

clang++ -ObjC++ -std=c++20 -O3 -fobjc-arc -shared -fPIC \
  -framework Metal -framework Foundation \
  "$NATIVE_DIR/flash_attn_nax.mm" \
  -o "$NATIVE_DIR/libinkling_flash_attn.dylib"

clang++ -ObjC++ -std=c++20 -O3 -Wall -Wextra -Wno-unused-parameter -fobjc-arc -shared -fPIC \
  -framework Metal -framework Foundation \
  "$NATIVE_DIR/mtp_draft_runner.mm" \
  -o "$NATIVE_DIR/libinkling_mtp_draft.dylib"

clang++ -ObjC++ -std=c++20 -O3 -fobjc-arc -shared -fPIC \
  -framework Metal -framework Foundation \
  "$NATIVE_DIR/prefill_moe_nax.mm" \
  -o "$NATIVE_DIR/libinkling_prefill_moe.dylib"

ls -lh "$NATIVE_DIR"/libinkling_*.dylib
