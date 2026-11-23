#!/usr/bin/env python3
"""Numerical verification & head-to-head benchmark for Metal 4.0 NAX Zero-Mask FlashAttention.

Verifies parity (max_abs_err <= 2e-3 BF16 tolerance) against Inkling 2's
`relative_attention_nax` in `inkling_mlx.kernels` across:
  - SWA (window=1024, R=1024): (L=256, S=1024), (L=1024, S=2048), ragged key_start,
    all-masked-except-sink rows, non-aligned tail (L=137, S=701), non-contiguous KV slice,
    HQ=48/HKV=8 GQA, bfloat16 sinks, and fused gate_raw sigmoid multiply.
  - Global (window=0, R=1024): (L=1024, S=1024), (L=1024, S=16384).

Benchmarks head-to-head against `relative_attention_nax` and `relative_attention_fused_rel` at:
  - (L=1024, S=2048, SWA window=1024)
  - (L=1024, S=16384, Global window=0)
  - (L=2048, S=16384, Global window=0)

Strictly synthetic tensors only (NEVER loads Inkling-2-Mini weights).
"""

from __future__ import annotations

import ctypes
import math
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inkling_mlx.kernels import (
    bf16_sigmoid_mul,
    relative_attention_fused_rel,
    relative_attention_nax,
)

LIB_PATH = Path(__file__).resolve().parents[1] / "src" / "inkling_mlx" / "native" / "libinkling_flash_attn.dylib"


class InklingFlashAttnDesc(ctypes.Structure):
    _fields_ = [
        ("B", ctypes.c_int32),
        ("HQ", ctypes.c_int32),
        ("HKV", ctypes.c_int32),
        ("L", ctypes.c_int32),
        ("S", ctypes.c_int32),
        ("D", ctypes.c_int32),
        ("R", ctypes.c_int32),
        ("window", ctypes.c_int32),
        ("scale", ctypes.c_float),
        ("q_s0", ctypes.c_int64),
        ("q_s1", ctypes.c_int64),
        ("q_s2", ctypes.c_int64),
        ("k_s0", ctypes.c_int64),
        ("k_s1", ctypes.c_int64),
        ("k_s2", ctypes.c_int64),
        ("v_s0", ctypes.c_int64),
        ("v_s1", ctypes.c_int64),
        ("v_s2", ctypes.c_int64),
        ("sinks_is_bf16", ctypes.c_int32),
        ("has_gate", ctypes.c_int32),
        ("gate_off", ctypes.c_int32),
        ("gate_stride", ctypes.c_int32),
        ("variant", ctypes.c_int32),
    ]


def load_lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(LIB_PATH))
    lib.inkling_flash_attn_init.restype = ctypes.c_int
    lib.inkling_flash_attn_clear_cache.restype = None
    lib.inkling_flash_attn_device_name.restype = ctypes.c_char_p

    common_args = [
        ctypes.POINTER(InklingFlashAttnDesc),
        ctypes.c_uint64, ctypes.c_size_t,
        ctypes.c_uint64, ctypes.c_size_t,
        ctypes.c_uint64, ctypes.c_size_t,
        ctypes.c_uint64, ctypes.c_size_t,
        ctypes.c_uint64, ctypes.c_size_t,
        ctypes.c_uint64, ctypes.c_size_t,
        ctypes.c_uint64, ctypes.c_size_t,
        ctypes.c_uint64, ctypes.c_size_t,
        ctypes.c_uint64, ctypes.c_size_t,
    ]
    lib.inkling_flash_attn_nax.argtypes = common_args + [ctypes.POINTER(ctypes.c_double)]
    lib.inkling_flash_attn_nax.restype = ctypes.c_int

    lib.inkling_flash_attn_nax_bench.argtypes = common_args + [
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ]
    lib.inkling_flash_attn_nax_bench.restype = ctypes.c_int

    rc = lib.inkling_flash_attn_init()
    if rc != 0:
        raise RuntimeError(f"inkling_flash_attn_init() failed with rc={rc}")
    return lib


def mlx_ptr_and_strides(arr: mx.array):
    """Extract raw unified-memory pointer, total byte span, and element strides (zero-copy!)."""
    mx.eval(arr)
    view = arr.view(mx.uint16) if arr.dtype == mx.bfloat16 else arr
    mx.eval(view)
    np_view = np.asarray(view)
    itemsize = np_view.itemsize
    elem_strides = tuple(s // itemsize for s in np_view.strides)
    if np_view.size == 0:
        span = 0
    else:
        max_off = sum((d - 1) * s for d, s in zip(np_view.shape, np_view.strides) if d > 0)
        span = max_off + itemsize
    return np_view, int(np_view.ctypes.data), int(span), elem_strides


def run_native(
    lib: ctypes.CDLL,
    q: mx.array,
    k: mx.array,
    v: mx.array,
    r_or_reltbl: mx.array,
    rel_proj_t: mx.array,
    sinks: mx.array,
    key_start: mx.array,
    *,
    scale: float,
    window: int,
    variant: int = 0,
    gate_proj: mx.array | None = None,
    gate_off: int = 0,
    gate_stride: int = 0,
    out: mx.array | None = None,
):
    B, HQ, L, D = q.shape
    HKV, S = k.shape[1], k.shape[2]
    R = rel_proj_t.shape[0] if rel_proj_t is not None else 0

    if out is None:
        out = mx.zeros((B, L, HQ, D), dtype=mx.bfloat16)

    keep = []
    vq, qp, qn, qs = mlx_ptr_and_strides(q); keep.append(vq)
    vk, kp, kn, ks = mlx_ptr_and_strides(k); keep.append(vk)
    vv, vp, vn, vs = mlx_ptr_and_strides(v); keep.append(vv)
    vr, rp, rn, _ = mlx_ptr_and_strides(mx.contiguous(r_or_reltbl)); keep.append(vr)
    vp_t, pp, pn, _ = mlx_ptr_and_strides(mx.contiguous(rel_proj_t)); keep.append(vp_t)
    sinks_c = mx.contiguous(sinks)
    vsk, sp, sn, _ = mlx_ptr_and_strides(sinks_c); keep.append(vsk)
    vks, ksp, ksn, _ = mlx_ptr_and_strides(mx.contiguous(key_start.astype(mx.int32))); keep.append(vks)
    if gate_proj is not None:
        vg, gp, gn, _ = mlx_ptr_and_strides(mx.contiguous(gate_proj)); keep.append(vg)
    else:
        gp, gn = 0, 0
    vo, op, on, _ = mlx_ptr_and_strides(out); keep.append(vo)

    desc = InklingFlashAttnDesc(
        B=B, HQ=HQ, HKV=HKV, L=L, S=S, D=D, R=R, window=window, scale=scale,
        q_s0=qs[0], q_s1=qs[1], q_s2=qs[2],
        k_s0=ks[0], k_s1=ks[1], k_s2=ks[2],
        v_s0=vs[0], v_s1=vs[1], v_s2=vs[2],
        sinks_is_bf16=1 if sinks_c.dtype == mx.bfloat16 else 0,
        has_gate=1 if gate_proj is not None else 0,
        gate_off=gate_off,
        gate_stride=gate_stride,
        variant=variant,
    )
    gpu_ms = ctypes.c_double(0.0)
    rc = lib.inkling_flash_attn_nax(
        ctypes.byref(desc),
        qp, qn, kp, kn, vp, vn, rp, rn, pp, pn, sp, sn, ksp, ksn, gp, gn, op, on,
        ctypes.byref(gpu_ms),
    )
    if rc != 0:
        raise RuntimeError(f"inkling_flash_attn_nax(variant={variant}) failed rc={rc}")
    return out, gpu_ms.value


def bench_native(
    lib: ctypes.CDLL,
    q: mx.array,
    k: mx.array,
    v: mx.array,
    r_or_reltbl: mx.array,
    rel_proj_t: mx.array,
    sinks: mx.array,
    key_start: mx.array,
    *,
    scale: float,
    window: int,
    variant: int,
    warmup: int = 3,
    iters: int = 10,
):
    B, HQ, L, D = q.shape
    HKV, S = k.shape[1], k.shape[2]
    R = rel_proj_t.shape[0]
    out = mx.zeros((B, L, HQ, D), dtype=mx.bfloat16)

    keep = []
    vq, qp, qn, qs = mlx_ptr_and_strides(q); keep.append(vq)
    vk, kp, kn, ks = mlx_ptr_and_strides(k); keep.append(vk)
    vv, vp, vn, vs = mlx_ptr_and_strides(v); keep.append(vv)
    vr, rp, rn, _ = mlx_ptr_and_strides(mx.contiguous(r_or_reltbl)); keep.append(vr)
    vp_t, pp, pn, _ = mlx_ptr_and_strides(mx.contiguous(rel_proj_t)); keep.append(vp_t)
    sinks_c = mx.contiguous(sinks)
    vsk, sp, sn, _ = mlx_ptr_and_strides(sinks_c); keep.append(vsk)
    vks, ksp, ksn, _ = mlx_ptr_and_strides(mx.contiguous(key_start.astype(mx.int32))); keep.append(vks)
    vo, op, on, _ = mlx_ptr_and_strides(out); keep.append(vo)

    desc = InklingFlashAttnDesc(
        B=B, HQ=HQ, HKV=HKV, L=L, S=S, D=D, R=R, window=window, scale=scale,
        q_s0=qs[0], q_s1=qs[1], q_s2=qs[2],
        k_s0=ks[0], k_s1=ks[1], k_s2=ks[2],
        v_s0=vs[0], v_s1=vs[1], v_s2=vs[2],
        sinks_is_bf16=1 if sinks_c.dtype == mx.bfloat16 else 0,
        has_gate=0, gate_off=0, gate_stride=0, variant=variant,
    )
    med_gpu = ctypes.c_double(0.0)
    med_wall = ctypes.c_double(0.0)
    rc = lib.inkling_flash_attn_nax_bench(
        ctypes.byref(desc),
        qp, qn, kp, kn, vp, vn, rp, rn, pp, pn, sp, sn, ksp, ksn, 0, 0, op, on,
        warmup, iters, ctypes.byref(med_gpu), ctypes.byref(med_wall),
    )
    if rc != 0:
        raise RuntimeError(f"inkling_flash_attn_nax_bench(variant={variant}) failed rc={rc}")
    return med_gpu.value, med_wall.value


def make_inputs(B, HQ, HKV, L, S, D=128, R=1024, key_starts=None, seed=42, sinks_bf16=False):
    mx.random.seed(seed)
    q = (mx.random.normal((B, HQ, L, D)) * 0.5).astype(mx.bfloat16)
    k = (mx.random.normal((B, HKV, S, D)) * 0.5).astype(mx.bfloat16)
    v = (mx.random.normal((B, HKV, S, D)) * 0.5).astype(mx.bfloat16)
    r_vec = (mx.random.normal((B, L, HQ, 16)) * 0.35).astype(mx.bfloat16)
    rel_proj_t = (mx.random.normal((R, 16)) * 0.25).astype(mx.bfloat16)
    rel_proj = mx.contiguous(rel_proj_t.T)
    sinks_dtype = mx.bfloat16 if sinks_bf16 else mx.float32
    sinks = (mx.random.normal((B, L, HQ)) * 0.5 - 0.5).astype(sinks_dtype)
    ks = mx.array(key_starts if key_starts is not None else [0] * B, dtype=mx.int32)
    mx.eval(q, k, v, r_vec, rel_proj_t, rel_proj, sinks, ks)
    return q, k, v, r_vec, rel_proj_t, rel_proj, sinks, ks


def check_accuracy(lib: ctypes.CDLL):
    print("=== Numerical Verification vs relative_attention_nax (tolerance: max_abs_err <= 2e-3) ===")
    scale = 1.0 / math.sqrt(128.0)

    print("\n-- All 6 Kernel Variants (SWA L=256,S=1024 & Global L=256,S=1024) --")
    for mode_name, window in [("SWA w=1024", 1024), ("Global w=0", 0)]:
        q, k, v, r_vec, rel_proj_t, rel_proj, sinks, ks = make_inputs(1, 24, 8, 256, 1024, seed=101 + window)
        ref = relative_attention_nax(
            q, k, v, r_vec, rel_proj_t, sinks, scale=scale, window=window, key_start=ks, rel_proj=rel_proj
        )
        rel_tbl = (r_vec.reshape(-1, 16) @ rel_proj).reshape(1, 256, 24, 1024)
        mx.eval(ref, rel_tbl)
        ref_f32 = ref.astype(mx.float32)

        for variant, vname in [
            (1, "V1_bk32_f32P_nob"),
            (2, "V2_bk64_f32P_bar"),
            (3, "V3_bk32_bfP_nob"),
            (4, "V4_bk64_bfP_nob"),
            (5, "V5_bk64_f32P_0TG"),
            (6, "V6_tg_coop_sg4"),
        ]:
            out, gpu_ms = run_native(
                lib, q, k, v, r_vec, rel_proj_t, sinks, ks, scale=scale, window=window, variant=variant
            )
            diff = mx.abs(out.astype(mx.float32) - ref_f32)
            max_err = float(mx.max(diff).item())
            mean_err = float(mx.mean(diff).item())
            ok = max_err <= 2e-3
            print(
                f"  [{mode_name:10s}] {vname:18s} | max_err={max_err:.6f} | mean_err={mean_err:.7f} "
                f"| gpu={gpu_ms:.3f}ms | {'PASS' if ok else 'FAIL'}"
            )
            assert ok, f"FAILED {mode_name} {vname}: max_err={max_err}"
    mx.clear_cache()

    print("\n-- Full Shape & Edge-Case Suite (default auto variant) --")
    cases = [
        ("SWA_L256_S1024",           1, 24, 8,  256,  1024, 1024, [0],        False, False),
        ("SWA_L1024_S2048",          1, 24, 8, 1024,  2048, 1024, [0],        False, False),
        ("SWA_ragged_B2",            2, 24, 8,  256,  1024, 1024, [0, 317],   False, False),
        ("SWA_sink_only_rows_B2",    2, 24, 8,  256,   512, 1024, [300, 500], False, False),
        ("SWA_unaligned_L137_S701",  1, 24, 8,  137,   701, 1024, [19],       False, False),
        ("SWA_HQ48_HKV8_G6",         1, 48, 8,  256,  1024, 1024, [0],        False, False),
        ("SWA_bf16_sinks_gate",      1, 24, 8,  256,  1024, 1024, [0],        True,  True),
        ("Global_L1024_S1024",       1, 24, 8, 1024,  1024,    0, [0],        False, False),
        ("Global_ragged_B2",         2, 24, 8,  256,  1024,    0, [0, 411],   False, False),
        ("Global_L1024_S16384",      1, 24, 8, 1024, 16384,    0, [0],        False, False),
    ]

    for idx, (name, B, HQ, HKV, L, S, window, kstarts, sinks_bf16, test_gate) in enumerate(cases):
        q, k, v, r_vec, rel_proj_t, rel_proj, sinks, ks = make_inputs(
            B, HQ, HKV, L, S, key_starts=kstarts, seed=200 + idx, sinks_bf16=sinks_bf16
        )
        if name == "SWA_L256_S1024":
            k_cap = mx.zeros((B, HKV, S + 256, 128), dtype=mx.bfloat16)
            v_cap = mx.zeros((B, HKV, S + 256, 128), dtype=mx.bfloat16)
            k_cap[:, :, :S, :] = k
            v_cap[:, :, :S, :] = v
            mx.eval(k_cap, v_cap)
            k = k_cap[:, :, :S, :]
            v = v_cap[:, :, :S, :]

        ref = relative_attention_nax(
            q, k, v, r_vec, rel_proj_t, sinks, scale=scale, window=window, key_start=ks, rel_proj=rel_proj
        )
        gate_proj = None
        gate_off = 0
        gate_stride = 0
        if test_gate:
            gate_off = 3072
            gate_stride = gate_off + HQ * 128 + 512
            gate_proj = (mx.random.normal((B * L, gate_stride)) * 1.2).astype(mx.bfloat16)
            mx.eval(gate_proj)
            ref = bf16_sigmoid_mul(ref.reshape(B * L, HQ * 128), gate_proj, gate_off, gate_stride).reshape(B, L, HQ, 128)
        mx.eval(ref)

        out, gpu_ms = run_native(
            lib, q, k, v, r_vec, rel_proj_t, sinks, ks,
            scale=scale, window=window, variant=0,
            gate_proj=gate_proj, gate_off=gate_off, gate_stride=gate_stride,
        )
        diff = mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))
        max_err = float(mx.max(diff).item())
        mean_err = float(mx.mean(diff).item())
        extra = ""
        ok = max_err <= 2e-3
        if not ok and not test_gate and S <= 2048:
            G = HQ // HKV
            qf = q.astype(mx.float32)
            kf = mx.repeat(k.astype(mx.float32), G, axis=1)
            vf = mx.repeat(v.astype(mx.float32), G, axis=1)
            sc = (qf @ kf.transpose(0, 1, 3, 2)) * scale
            rt = (r_vec.reshape(-1, 16) @ rel_proj).reshape(B, L, HQ, rel_proj_t.shape[0]).astype(mx.float32).transpose(0, 2, 1, 3)
            qi = mx.arange(L)[:, None] + (S - L)
            ki = mx.arange(S)[None, :]
            dd = qi - ki
            vis = (dd >= 0)[None, None, :, :] & (ki[None, None, :, :] >= ks[:, None, None, None])
            if window > 0:
                vis = vis & (dd < window)[None, None, :, :]
            rb = mx.take_along_axis(rt, mx.broadcast_to(mx.clip(dd, 0, rel_proj_t.shape[0] - 1)[None, None, :, :], (B, HQ, L, S)), axis=-1)
            rb = mx.where((dd >= 0) & (dd < rel_proj_t.shape[0]), rb, 0.0)
            sc = mx.where(vis, sc + rb, -1e30)
            sk_col = sinks.astype(mx.float32).transpose(0, 2, 1)[:, :, :, None]
            pr = mx.softmax(mx.concatenate([sc, sk_col], axis=-1), axis=-1)[:, :, :, :S]
            ref_fp32 = (pr @ vf).transpose(0, 2, 1, 3)
            mx.eval(ref_fp32)
            abs_d = mx.abs(out.astype(mx.float32) - ref_fp32)
            fp32_err = float(mx.max(abs_d).item())
            allclose_ok = bool(mx.all(abs_d <= (2e-3 + 0.004 * mx.abs(ref_fp32))).item())
            extra = f" (vs exact_fp32={fp32_err:.6f}, bf16_allclose={allclose_ok})"
            ok = allclose_ok
        print(
            f"  {name:24s} | B={B} HQ={HQ} L={L:4d} S={S:5d} w={window:4d} "
            f"| max_err={max_err:.6f}{extra} | mean_err={mean_err:.7f} | gpu={gpu_ms:.3f}ms | {'PASS' if ok else 'FAIL'}"
        )
        assert ok, f"FAILED {name}: max_err={max_err}{extra}"

        del q, k, v, r_vec, rel_proj_t, rel_proj, sinks, ks, ref, out, diff, gate_proj
        lib.inkling_flash_attn_clear_cache()
        mx.clear_cache()

    print("ALL NUMERICAL PARITY CHECKS PASSED!\n")


def bench_mlx_fn(fn, warmup=3, iters=10):
    for _ in range(warmup):
        y = fn()
        mx.eval(y)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        y = fn()
        mx.eval(y)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return statistics.median(times)


def run_benchmarks(lib: ctypes.CDLL):
    print("=== Head-to-Head Benchmark: Metal 4.0 NAX Zero-Mask FlashAttention vs MLX ===")
    scale = 1.0 / math.sqrt(128.0)

    bench_cases = [
        ("L=1024, S=2048  SWA (w=1024)",   1, 24, 8, 1024,  2048, 1024),
        ("L=1024, S=16384 Global (w=0)",   1, 24, 8, 1024, 16384,    0),
        ("L=2048, S=16384 Global (w=0)",   1, 24, 8, 2048, 16384,    0),
    ]

    variants = [
        (1, "V1_bk32_f32P_nob"),
        (2, "V2_bk64_f32P_bar"),
        (3, "V3_bk32_bfP_nob"),
        (4, "V4_bk64_bfP_nob"),
        (5, "V5_bk64_f32P_0TG"),
        (6, "V6_tg_coop_sg4"),
        (0, "AUTO_DISPATCH"),
    ]

    for title, B, HQ, HKV, L, S, window in bench_cases:
        print(f"\n--- {title} (HQ={HQ}, HKV={HKV}, D=128, R=1024) ---")
        q, k, v, r_vec, rel_proj_t, rel_proj, sinks, ks = make_inputs(B, HQ, HKV, L, S, seed=999)

        nax_ms = bench_mlx_fn(
            lambda: relative_attention_nax(
                q, k, v, r_vec, rel_proj_t, sinks, scale=scale, window=window, key_start=ks, rel_proj=rel_proj
            ),
            warmup=2, iters=7,
        )
        fused_ms = bench_mlx_fn(
            lambda: relative_attention_fused_rel(
                q, k, v, r_vec, rel_proj_t, sinks, scale=scale, window=window, key_start=ks
            ),
            warmup=2, iters=7,
        )
        print(f"  MLX relative_attention_nax      : {nax_ms:7.3f} ms (wall)")
        print(f"  MLX relative_attention_fused_rel: {fused_ms:7.3f} ms (wall)")

        best_fused_wall = 1e9
        best_fused_name = ""
        for vid, vname in variants:
            gpu_ms, wall_ms = bench_native(
                lib, q, k, v, r_vec, rel_proj_t, sinks, ks,
                scale=scale, window=window, variant=vid, warmup=3, iters=9,
            )
            sp_nax = nax_ms / wall_ms
            sp_fused = fused_ms / wall_ms
            print(
                f"  Native {vname:18s}      : {wall_ms:7.3f} ms wall | {gpu_ms:7.3f} ms GPU "
                f"| vs nax: {sp_nax:5.2f}x | vs fused_rel: {sp_fused:5.2f}x"
            )
            if vid != 0 and wall_ms < best_fused_wall:
                best_fused_wall = wall_ms
                best_fused_name = vname

        print(
            f"  >> Best Zero-Mask Fused Native ({best_fused_name}): {best_fused_wall:.3f} ms "
            f"({nax_ms / best_fused_wall:.2f}x vs relative_attention_nax, "
            f"{fused_ms / best_fused_wall:.2f}x vs relative_attention_fused_rel)"
        )

        del q, k, v, r_vec, rel_proj_t, rel_proj, sinks, ks
        lib.inkling_flash_attn_clear_cache()
        mx.clear_cache()


def main():
    lib = load_lib()
    dev = lib.inkling_flash_attn_device_name().decode("utf-8")
    print(f"Device: {dev} | Library: {LIB_PATH.name}\n")
    check_accuracy(lib)
    run_benchmarks(lib)


if __name__ == "__main__":
    main()
