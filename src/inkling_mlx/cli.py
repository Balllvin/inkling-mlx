"""CLI entrypoint for building native Metal 4 libraries, downloading Inkling 2 Mini, and serving."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_HF_REPO = "thinkingmachines/Inkling-2-Mini"


def native_dir() -> Path:
    return Path(__file__).resolve().parent / "native"


def build_native() -> None:
    ndir = native_dir()
    targets = [
        ("decode_runner.mm", "libinkling_decode.dylib", ["-Wall", "-Wextra", "-Wno-unused-parameter"]),
        ("flash_attn_nax.mm", "libinkling_flash_attn.dylib", []),
        ("mtp_draft_runner.mm", "libinkling_mtp_draft.dylib", ["-Wall", "-Wextra", "-Wno-unused-parameter"]),
        ("prefill_moe_nax.mm", "libinkling_prefill_moe.dylib", []),
    ]
    for src, out_name, extra in targets:
        cmd = [
            "clang++",
            "-ObjC++",
            "-std=c++20",
            "-O3",
            *extra,
            "-fobjc-arc",
            "-shared",
            "-fPIC",
            "-framework",
            "Metal",
            "-framework",
            "Foundation",
            str(ndir / src),
            "-o",
            str(ndir / out_name),
        ]
        print(f"[build] Compiling {out_name}...", flush=True)
        subprocess.run(cmd, check=True)
        print(f"[build] Built {ndir / out_name}", flush=True)


def ensure_native_built() -> None:
    ndir = native_dir()
    libs = [
        "libinkling_decode.dylib",
        "libinkling_flash_attn.dylib",
        "libinkling_mtp_draft.dylib",
        "libinkling_prefill_moe.dylib",
    ]
    if any(not (ndir / name).is_file() for name in libs):
        build_native()


def resolve_model_dir(spec: str) -> Path:
    expanded = Path(spec).expanduser()
    if expanded.is_dir() and (expanded / "config.json").is_file():
        return expanded
    env_dir = os.environ.get("INKLING_MODEL_DIR")
    if env_dir:
        cand = Path(env_dir).expanduser()
        if (cand / "config.json").is_file():
            return cand
    default_local = Path.home() / "models" / "Inkling-2-Mini"
    if spec == DEFAULT_HF_REPO and (default_local / "config.json").is_file():
        return default_local
    from huggingface_hub import snapshot_download

    print(f"[download] Resolving {spec} from Hugging Face Hub...", flush=True)
    return Path(snapshot_download(repo_id=spec))


def resolve_adapter_dir(spec: str) -> Path:
    """Resolve a PEFT LoRA adapter from a local directory or Hugging Face repo ID."""
    expanded = Path(spec).expanduser()
    if expanded.is_dir():
        if (expanded / "adapter_config.json").is_file() or (expanded / "adapter" / "adapter_config.json").is_file():
            return expanded
        raise FileNotFoundError(f"Adapter directory {expanded} does not contain adapter_config.json")
    from huggingface_hub import snapshot_download

    print(f"[download] Resolving adapter {spec} from Hugging Face Hub...", flush=True)
    return Path(snapshot_download(repo_id=spec))


def download_model(repo_id: str, local_dir: str | None) -> Path:
    from huggingface_hub import snapshot_download

    kwargs = {"repo_id": repo_id}
    if local_dir:
        kwargs["local_dir"] = str(Path(local_dir).expanduser())
    print(f"[download] Downloading {repo_id}...", flush=True)
    out = Path(snapshot_download(**kwargs))
    print(f"[download] Ready at {out}", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="inkling-mlx")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build", help="Compile native Metal 4 libraries")

    dl = sub.add_parser("download", help="Download Inkling 2 Mini weights from Hugging Face")
    dl.add_argument("--repo", default=DEFAULT_HF_REPO)
    dl.add_argument("--local-dir", default=None)

    sv = sub.add_parser("serve", add_help=False, help="Run the OpenAI-compatible HTTP server")
    sv.add_argument("--skip-build", action="store_true", help="Skip automatic build of missing native dylibs")

    args, rest = ap.parse_known_args(argv)
    if args.cmd == "build":
        build_native()
        return 0
    if args.cmd == "download":
        download_model(args.repo, args.local_dir)
        return 0
    if args.cmd == "serve":
        if not args.skip_build and "-h" not in rest and "--help" not in rest:
            ensure_native_built()
        from .serve import main as serve_main

        return serve_main(rest)
    return 1


if __name__ == "__main__":
    sys.exit(main())
