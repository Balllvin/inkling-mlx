#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"

if [ -n "${PYTHON_BIN:-}" ]; then
  if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
  "$VENV_DIR/bin/python" -m pip install --upgrade pip
  "$VENV_DIR/bin/pip" install -e "$ROOT"
elif command -v uv >/dev/null 2>&1; then
  if [ ! -d "$VENV_DIR" ]; then
    uv venv --python 3.12 --seed "$VENV_DIR"
  fi
  uv pip install --python "$VENV_DIR/bin/python" -e "$ROOT"
else
  PY_CMD=""
  for cand in python3.12 python3.13 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
      PY_CMD="$cand"
      break
    fi
  done
  if [ -z "$PY_CMD" ]; then
    echo "Python 3.11+ is required" >&2
    exit 1
  fi
  if [ ! -d "$VENV_DIR" ]; then
    "$PY_CMD" -m venv "$VENV_DIR"
  fi
  "$VENV_DIR/bin/python" -m pip install --upgrade pip
  "$VENV_DIR/bin/pip" install -e "$ROOT"
fi

bash "$ROOT/scripts/build_native.sh"

if [ "${DOWNLOAD_MODEL:-0}" = "1" ]; then
  "$VENV_DIR/bin/inkling-mlx" download --local-dir "${MODEL_DIR:-$HOME/models/Inkling-2-Mini}"
fi
