#!/usr/bin/env python3
"""Bit-exact parity & interleaved benchmark for Lever 8 fused 4-dispatch prefill MoE.

Verifies `inkling_prefill_moe_nax_fused_combine` (bucket + w13+SwiGLU + w2 + 64-ch
128B-aligned fused_unsort_combine_sconv_resid in ONE MTLCommandBuffer):
  1. Exact bit-identity (`0.00e+00` max abs diff on BOTH `out_x` BF16 and `new_state` FP32)
     vs `prefill_moe_forward` + MLX `fused_unsort_combine_sconv_resid` across:
       - B=1 L=1024, B=1 L=4096, B=1 L=8192, B=4 L=512 (both has_mask=0 and has_mask=1)
       - Edge cases: B=1 L=2, B=4 L=2, B=1 L=1 (both has_mask=0 and has_mask=1)
       - Both NVFP4 and MXFP8 modes
       - VA-reuse stress test (ephemeral activations/outputs freed & reallocated across calls)
  2. Interleaved per-layer GPU + end-to-end wall benchmark at L=4096 and L=8192
     (NVFP4 and MXFP8), with exact savings projected x48 layer visits.

NEVER loads the 19.5 GB Inkling checkpoint; uses synthetic E=8 quantized experts.
"""

from __future__ import annotations

import ctypes
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inkling_mlx.kernels import fused_unsort_combine_sconv_resid

LIB_PATH = Path(__file__).resolve().parents[1] / "src" / "inkling_mlx" / "native" / "libinkling_prefill_moe.dylib"


class MoeForwardArgs(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int32),
        ("M", ctypes.c_int32),
        ("K", ctypes.c_int32),
        ("INTER", ctypes.c_int32),
        ("D", ctypes.c_int32),
        ("E", ctypes.c_int32),
        ("TOP_K", ctypes.c_int32),
        ("_pad", ctypes.c_int32),
        ("gate_cap", ctypes.c_float),
        ("up_cap", ctypes.c_float),
        ("x_ptr", ctypes.c_uint64),
        ("inds_ptr", ctypes.c_uint64),
        ("w13_ptr", ctypes.c_uint64),
        ("s13_ptr", ctypes.c_uint64),
        ("gs13_ptr", ctypes.c_uint64),
        ("w2_ptr", ctypes.c_uint64),
        ("s2_ptr", ctypes.c_uint64),
        ("gs2_ptr", ctypes.c_uint64),
        ("y_sorted_ptr", ctypes.c_uint64),
        ("inv_order_ptr", ctypes.c_uint64),
        ("offsets_ptr", ctypes.c_uint64),
        ("sorted_slots_ptr", ctypes.c_uint64),
    ]


class MoeFusedCombineArgs(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int32),
        ("B", ctypes.c_int32),
        ("L", ctypes.c_int32),
        ("K", ctypes.c_int32),
        ("INTER", ctypes.c_int32),
        ("D", ctypes.c_int32),
        ("E", ctypes.c_int32),
        ("TOP_K", ctypes.c_int32),
        ("gate_cap", ctypes.c_float),
        ("up_cap", ctypes.c_float),
        ("has_mask", ctypes.c_int32),
        ("_pad", ctypes.c_int32),
        ("x_ptr", ctypes.c_uint64),
        ("inds_ptr", ctypes.c_uint64),
        ("w13_ptr", ctypes.c_uint64),
        ("s13_ptr", ctypes.c_uint64),
        ("gs13_ptr", ctypes.c_uint64),
        ("w2_ptr", ctypes.c_uint64),
        ("s2_ptr", ctypes.c_uint64),
        ("gs2_ptr", ctypes.c_uint64),
        ("rweights_ptr", ctypes.c_uint64),
        ("state_ptr", ctypes.c_uint64),
        ("taps_ptr", ctypes.c_uint64),
        ("resid_ptr", ctypes.c_uint64),
        ("mask_ptr", ctypes.c_uint64),
        ("out_x_ptr", ctypes.c_uint64),
        ("new_state_ptr", ctypes.c_uint64),
        ("y_sorted_ptr", ctypes.c_uint64),
        ("inv_order_ptr", ctypes.c_uint64),
    ]


def load_lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(LIB_PATH))
    lib.prefill_moe_init.argtypes = []
    lib.prefill_moe_init.restype = ctypes.c_int
    lib.prefill_moe_register_buffer.argtypes = [ctypes.c_uint64, ctypes.c_size_t]
    lib.prefill_moe_register_buffer.restype = ctypes.c_int
    lib.prefill_moe_clear_cache.argtypes = []
    lib.prefill_moe_clear_cache.restype = None
    lib.prefill_moe_release_scratch.argtypes = []
    lib.prefill_moe_release_scratch.restype = None

    lib.prefill_moe_forward.argtypes = [
        ctypes.POINTER(MoeForwardArgs),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.prefill_moe_forward.restype = ctypes.c_int

    lib.inkling_prefill_moe_nax_fused_combine.argtypes = [
        ctypes.POINTER(MoeFusedCombineArgs),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.inkling_prefill_moe_nax_fused_combine.restype = ctypes.c_int

    lib.prefill_moe_forward_fused_combine_bench.argtypes = [
        ctypes.POINTER(MoeFusedCombineArgs),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.prefill_moe_forward_fused_combine_bench.restype = ctypes.c_int

    t0 = time.perf_counter()
    rc = lib.prefill_moe_init()
    t1 = time.perf_counter()
    assert rc == 0, f"prefill_moe_init failed rc={rc}"
    print(f"[init] Metal 4.0 NAX pipelines (incl. 64-ch fused combine) compiled in {(t1 - t0) * 1000:.1f} ms")
    return lib


def mlx_ptr(arr: mx.array) -> tuple[int, np.ndarray]:
    mx.eval(arr)
    view = arr.view(mx.uint16) if arr.dtype == mx.bfloat16 else arr
    mx.eval(view)
    nv = np.asarray(view)
    assert nv.flags["C_CONTIGUOUS"], f"Non-contiguous MLX buffer: strides={nv.strides}"
    return int(nv.ctypes.data), nv


def make_synthetic_weights(E: int, K: int, INTER: int, D: int, mode: str, seed: int):
    mx.random.seed(seed)
    gs = 16 if mode == "nvfp4" else 32
    bits = 4 if mode == "nvfp4" else 8
    w13_list, s13_list, w2_list, s2_list = [], [], [], []
    std13 = (2.0 / K) ** 0.5 * 0.5
    std2 = (2.0 / INTER) ** 0.5 * 0.5
    for _ in range(E):
        wf13 = (mx.random.normal((2 * INTER, K), dtype=mx.float32) * std13).astype(mx.bfloat16)
        wq13, sq13 = mx.quantize(wf13, group_size=gs, bits=bits, mode=mode)
        mx.eval(wq13, sq13)
        w13_list.append(wq13)
        s13_list.append(sq13)
        del wf13

        wf2 = (mx.random.normal((D, INTER), dtype=mx.float32) * std2).astype(mx.bfloat16)
        wq2, sq2 = mx.quantize(wf2, group_size=gs, bits=bits, mode=mode)
        mx.eval(wq2, sq2)
        w2_list.append(wq2)
        s2_list.append(sq2)
        del wf2

    w13 = mx.contiguous(mx.stack(w13_list, axis=0))
    s13 = mx.contiguous(mx.stack(s13_list, axis=0))
    w2 = mx.contiguous(mx.stack(w2_list, axis=0))
    s2 = mx.contiguous(mx.stack(s2_list, axis=0))
    taps = mx.contiguous(mx.random.normal((D, 4), dtype=mx.float32) * 0.25)
    del w13_list, s13_list, w2_list, s2_list
    mx.eval(w13, s13, w2, s2, taps)

    if mode == "nvfp4":
        rng = np.random.default_rng(seed + 7)
        gs13 = mx.array(2688.0 * rng.uniform(0.75, 1.35, size=(E,)).astype(np.float32))
        gs2 = mx.array(2688.0 * rng.uniform(0.75, 1.35, size=(E,)).astype(np.float32))
        mx.eval(gs13, gs2)
    else:
        gs13 = gs2 = None
    return w13, s13, gs13, w2, s2, gs2, taps


def verify_bit_identity(
    lib: ctypes.CDLL,
    mode: str,
    E: int = 8,
    K: int = 2304,
    INTER: int = 1152,
    D: int = 2304,
    TOP_K: int = 8,
):
    print(f"\n=== Bit-Identity Verification ({mode.upper()}, E={E}, K={K}, INTER={INTER}, D={D}, TOP_K={TOP_K}) ===")
    mode_id = 0 if mode == "nvfp4" else 1
    w13, s13, gs13, w2, s2, gs2, taps = make_synthetic_weights(E, K, INTER, D, mode, seed=100 + mode_id)

    p_w13, _kw13 = mlx_ptr(w13)
    p_s13, _ks13 = mlx_ptr(s13)
    p_gs13, _kgs13 = mlx_ptr(gs13) if gs13 is not None else (0, None)
    p_w2, _kw2 = mlx_ptr(w2)
    p_s2, _ks2 = mlx_ptr(s2)
    p_gs2, _kgs2 = mlx_ptr(gs2) if gs2 is not None else (0, None)
    p_taps, _ktaps = mlx_ptr(taps)

    cases = [
        (1, 1, 0),
        (1, 1, 1),
        (1, 2, 0),
        (1, 2, 1),
        (4, 2, 1),
        (1, 1024, 0),
        (1, 1024, 1),
        (4, 512, 0),
        (4, 512, 1),
        (1, 4096, 0),
        (1, 4096, 1),
        (1, 8192, 0),
        (1, 8192, 1),
    ]

    for idx_case, (B, L, has_mask) in enumerate(cases):
        M = B * L
        T = M * TOP_K
        rng = np.random.default_rng(1000 + mode_id * 100 + idx_case)
        mx.random.seed(2000 + mode_id * 100 + idx_case)

        x = (mx.random.normal((M, K), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
        inds = mx.array(rng.integers(0, E, size=(M, TOP_K), dtype=np.uint32), dtype=mx.uint32)
        rw_np = rng.uniform(0.05, 0.4, size=(M, TOP_K)).astype(np.float32)
        rw_np /= rw_np.sum(axis=-1, keepdims=True)
        rweights = mx.array(rw_np, dtype=mx.float32)
        state = mx.random.normal((B, 3, D), dtype=mx.float32) * 0.3
        resid = mx.random.normal((M, D), dtype=mx.float32) * 0.7
        if has_mask:
            mask_np = rng.integers(0, 2, size=(B, L), dtype=np.uint8)
            mask_np[:, L // 2 :] = 1
            mask = mx.array(mask_np.reshape(M), dtype=mx.uint8)
        else:
            mask = None
        mx.eval(x, inds, rweights, state, resid)
        if mask is not None:
            mx.eval(mask)

        ys_ref = mx.zeros((T, D), dtype=mx.bfloat16)
        inv_ref = mx.zeros((T,), dtype=mx.uint32)
        p_x, _kx = mlx_ptr(x)
        p_inds, _ki = mlx_ptr(inds)
        p_ys_ref, _kysr = mlx_ptr(ys_ref)
        p_inv_ref, _kinvr = mlx_ptr(inv_ref)

        fargs = MoeForwardArgs(
            mode=mode_id, M=M, K=K, INTER=INTER, D=D, E=E, TOP_K=TOP_K, _pad=0,
            gate_cap=30.0, up_cap=30.0,
            x_ptr=p_x, inds_ptr=p_inds,
            w13_ptr=p_w13, s13_ptr=p_s13, gs13_ptr=p_gs13,
            w2_ptr=p_w2, s2_ptr=p_s2, gs2_ptr=p_gs2,
            y_sorted_ptr=p_ys_ref, inv_order_ptr=p_inv_ref, offsets_ptr=0, sorted_slots_ptr=0,
        )
        rc = lib.prefill_moe_forward(ctypes.byref(fargs), None)
        assert rc == 0, f"prefill_moe_forward failed rc={rc}"

        ref_out_x, ref_new_state = fused_unsort_combine_sconv_resid(
            ys_ref, inv_ref, rweights, state, taps, resid,
            b=B, length=L, top_k=TOP_K, mask=mask,
        )
        mx.eval(ref_out_x, ref_new_state)

        got_out_x = mx.zeros((M, D), dtype=mx.bfloat16)
        got_new_state = mx.zeros((B, 3, D), dtype=mx.float32)
        p_rw, _krw = mlx_ptr(rweights)
        p_st, _kst = mlx_ptr(state)
        p_res, _kres = mlx_ptr(resid)
        p_mask, _kmask = mlx_ptr(mask) if mask is not None else (0, None)
        p_out_x, _kox = mlx_ptr(got_out_x)
        p_new_st, _knst = mlx_ptr(got_new_state)

        cargs = MoeFusedCombineArgs(
            mode=mode_id, B=B, L=L, K=K, INTER=INTER, D=D, E=E, TOP_K=TOP_K,
            gate_cap=30.0, up_cap=30.0, has_mask=has_mask, _pad=0,
            x_ptr=p_x, inds_ptr=p_inds,
            w13_ptr=p_w13, s13_ptr=p_s13, gs13_ptr=p_gs13,
            w2_ptr=p_w2, s2_ptr=p_s2, gs2_ptr=p_gs2,
            rweights_ptr=p_rw, state_ptr=p_st, taps_ptr=p_taps, resid_ptr=p_res, mask_ptr=p_mask,
            out_x_ptr=p_out_x, new_state_ptr=p_new_st,
            y_sorted_ptr=0, inv_order_ptr=0,
        )
        gpu_ms = ctypes.c_double(0.0)
        rc = lib.inkling_prefill_moe_nax_fused_combine(ctypes.byref(cargs), ctypes.byref(gpu_ms))
        assert rc == 0, f"inkling_prefill_moe_nax_fused_combine failed rc={rc}"

        ref_x_u16 = np.asarray(ref_out_x.view(mx.uint16))
        got_x_u16 = np.asarray(got_out_x.view(mx.uint16))
        ref_x_f32 = np.asarray(ref_out_x.astype(mx.float32))
        got_x_f32 = np.asarray(got_out_x.astype(mx.float32))
        ref_s_f32 = np.asarray(ref_new_state)
        got_s_f32 = np.asarray(got_new_state)

        x_diff = float(np.abs(got_x_f32 - ref_x_f32).max())
        s_diff = float(np.abs(got_s_f32 - ref_s_f32).max())
        x_bit_eq = bool(np.array_equal(ref_x_u16, got_x_u16))
        s_bit_eq = bool(np.array_equal(ref_s_f32.view(np.uint32), got_s_f32.view(np.uint32)))

        status = "PASS" if (x_diff == 0.0 and s_diff == 0.0 and x_bit_eq and s_bit_eq) else "FAIL"
        print(
            f"  [{status}] B={B} L={L:4d} mask={has_mask} | "
            f"out_x max_diff={x_diff:.2e} (bit_eq={x_bit_eq}) | "
            f"new_state max_diff={s_diff:.2e} (bit_eq={s_bit_eq}) | gpu={gpu_ms.value:.3f} ms"
        )
        assert status == "PASS", (
            f"Mismatch on {mode} B={B} L={L} mask={has_mask}: "
            f"x_diff={x_diff} x_bit_eq={x_bit_eq} s_diff={s_diff} s_bit_eq={s_bit_eq}"
        )

        del x, inds, rweights, state, resid, mask, ys_ref, inv_ref, ref_out_x, ref_new_state, got_out_x, got_new_state
        mx.clear_cache()

    for it in range(6):
        B, L = 1, 512
        M, T = B * L, B * L * TOP_K
        mx.random.seed(9000 + it)
        x = (mx.random.normal((M, K), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
        inds = mx.array(np.random.default_rng(9000 + it).integers(0, E, size=(M, TOP_K), dtype=np.uint32))
        rweights = mx.ones((M, TOP_K), dtype=mx.float32) * (1.0 / TOP_K)
        state = mx.random.normal((B, 3, D), dtype=mx.float32) * 0.2
        resid = mx.random.normal((M, D), dtype=mx.float32) * 0.5
        out_x = mx.zeros((M, D), dtype=mx.bfloat16)
        new_st = mx.zeros((B, 3, D), dtype=mx.float32)
        ys_ref = mx.zeros((T, D), dtype=mx.bfloat16)
        inv_ref = mx.zeros((T,), dtype=mx.uint32)
        mx.eval(x, inds, rweights, state, resid, out_x, new_st, ys_ref, inv_ref)

        p_x, _ = mlx_ptr(x); p_i, _ = mlx_ptr(inds); p_rw, _ = mlx_ptr(rweights)
        p_s, _ = mlx_ptr(state); p_r, _ = mlx_ptr(resid)
        p_ox, _ = mlx_ptr(out_x); p_ns, _ = mlx_ptr(new_st)
        p_yr, _ = mlx_ptr(ys_ref); p_ir, _ = mlx_ptr(inv_ref)

        fargs = MoeForwardArgs(
            mode=mode_id, M=M, K=K, INTER=INTER, D=D, E=E, TOP_K=TOP_K, _pad=0,
            gate_cap=30.0, up_cap=30.0, x_ptr=p_x, inds_ptr=p_i,
            w13_ptr=p_w13, s13_ptr=p_s13, gs13_ptr=p_gs13,
            w2_ptr=p_w2, s2_ptr=p_s2, gs2_ptr=p_gs2,
            y_sorted_ptr=p_yr, inv_order_ptr=p_ir, offsets_ptr=0, sorted_slots_ptr=0,
        )
        assert lib.prefill_moe_forward(ctypes.byref(fargs), None) == 0
        rx, rs = fused_unsort_combine_sconv_resid(ys_ref, inv_ref, rweights, state, taps, resid, b=B, length=L, top_k=TOP_K)
        mx.eval(rx, rs)

        cargs = MoeFusedCombineArgs(
            mode=mode_id, B=B, L=L, K=K, INTER=INTER, D=D, E=E, TOP_K=TOP_K,
            gate_cap=30.0, up_cap=30.0, has_mask=0, _pad=0,
            x_ptr=p_x, inds_ptr=p_i,
            w13_ptr=p_w13, s13_ptr=p_s13, gs13_ptr=p_gs13,
            w2_ptr=p_w2, s2_ptr=p_s2, gs2_ptr=p_gs2,
            rweights_ptr=p_rw, state_ptr=p_s, taps_ptr=p_taps, resid_ptr=p_r, mask_ptr=0,
            out_x_ptr=p_ox, new_state_ptr=p_ns, y_sorted_ptr=0, inv_order_ptr=0,
        )
        assert lib.inkling_prefill_moe_nax_fused_combine(ctypes.byref(cargs), None) == 0
        assert np.array_equal(np.asarray(rx.view(mx.uint16)), np.asarray(out_x.view(mx.uint16)))
        assert np.array_equal(np.asarray(rs), np.asarray(new_st))
        del x, inds, rweights, state, resid, out_x, new_st, ys_ref, inv_ref, rx, rs
        mx.clear_cache()
    print(f"  [PASS] VA-reuse stress test (6 consecutive fresh alloc/clear_cache cycles): 0.00e+00 bit-exact!")

    lib.prefill_moe_clear_cache()
    lib.prefill_moe_release_scratch()
    del w13, s13, gs13, w2, s2, gs2, taps
    mx.clear_cache()


def benchmark_interleaved(
    lib: ctypes.CDLL,
    mode: str,
    E: int = 8,
    K: int = 2304,
    INTER: int = 1152,
    D: int = 2304,
    TOP_K: int = 8,
):
    print(f"\n=== Interleaved Benchmark ({mode.upper()}, B=1, E={E}, K={K}, INTER={INTER}, D={D}, TOP_K={TOP_K}) ===")
    mode_id = 0 if mode == "nvfp4" else 1
    w13, s13, gs13, w2, s2, gs2, taps = make_synthetic_weights(E, K, INTER, D, mode, seed=500 + mode_id)

    p_w13, _kw13 = mlx_ptr(w13)
    p_s13, _ks13 = mlx_ptr(s13)
    p_gs13, _kgs13 = mlx_ptr(gs13) if gs13 is not None else (0, None)
    p_w2, _kw2 = mlx_ptr(w2)
    p_s2, _ks2 = mlx_ptr(s2)
    p_gs2, _kgs2 = mlx_ptr(gs2) if gs2 is not None else (0, None)
    p_taps, _ktaps = mlx_ptr(taps)

    for L in (4096, 8192):
        B = 1
        M = B * L
        T = M * TOP_K
        rng = np.random.default_rng(7000 + L)
        mx.random.seed(7000 + L)

        x = (mx.random.normal((M, K), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
        inds_flat = np.repeat(np.arange(E, dtype=np.uint32), T // E)
        rng.shuffle(inds_flat)
        inds = mx.array(inds_flat.reshape(M, TOP_K), dtype=mx.uint32)
        rw_np = rng.uniform(0.05, 0.4, size=(M, TOP_K)).astype(np.float32)
        rw_np /= rw_np.sum(axis=-1, keepdims=True)
        rweights = mx.array(rw_np, dtype=mx.float32)
        state = mx.random.normal((B, 3, D), dtype=mx.float32) * 0.3
        resid = mx.random.normal((M, D), dtype=mx.float32) * 0.7

        ys_scratch = mx.zeros((T, D), dtype=mx.bfloat16)
        inv_scratch = mx.zeros((T,), dtype=mx.uint32)
        out_x_buf = mx.zeros((M, D), dtype=mx.bfloat16)
        new_st_buf = mx.zeros((B, 3, D), dtype=mx.float32)
        mx.eval(x, inds, rweights, state, resid, ys_scratch, inv_scratch, out_x_buf, new_st_buf)

        p_x, _kx = mlx_ptr(x)
        p_inds, _ki = mlx_ptr(inds)
        p_rw, _krw = mlx_ptr(rweights)
        p_st, _kst = mlx_ptr(state)
        p_res, _kres = mlx_ptr(resid)
        p_ys, _kys = mlx_ptr(ys_scratch)
        p_inv, _kinv = mlx_ptr(inv_scratch)
        p_ox, _kox = mlx_ptr(out_x_buf)
        p_ns, _kns = mlx_ptr(new_st_buf)

        fargs = MoeForwardArgs(
            mode=mode_id, M=M, K=K, INTER=INTER, D=D, E=E, TOP_K=TOP_K, _pad=0,
            gate_cap=30.0, up_cap=30.0,
            x_ptr=p_x, inds_ptr=p_inds,
            w13_ptr=p_w13, s13_ptr=p_s13, gs13_ptr=p_gs13,
            w2_ptr=p_w2, s2_ptr=p_s2, gs2_ptr=p_gs2,
            y_sorted_ptr=p_ys, inv_order_ptr=p_inv, offsets_ptr=0, sorted_slots_ptr=0,
        )
        cargs = MoeFusedCombineArgs(
            mode=mode_id, B=B, L=L, K=K, INTER=INTER, D=D, E=E, TOP_K=TOP_K,
            gate_cap=30.0, up_cap=30.0, has_mask=0, _pad=0,
            x_ptr=p_x, inds_ptr=p_inds,
            w13_ptr=p_w13, s13_ptr=p_s13, gs13_ptr=p_gs13,
            w2_ptr=p_w2, s2_ptr=p_s2, gs2_ptr=p_gs2,
            rweights_ptr=p_rw, state_ptr=p_st, taps_ptr=p_taps, resid_ptr=p_res, mask_ptr=0,
            out_x_ptr=p_ox, new_state_ptr=p_ns,
            y_sorted_ptr=0, inv_order_ptr=0,
        )

        for _ in range(3):
            assert lib.prefill_moe_forward(ctypes.byref(fargs), None) == 0
            ox, ns = fused_unsort_combine_sconv_resid(
                ys_scratch, inv_scratch, rweights, state, taps, resid, b=B, length=L, top_k=TOP_K
            )
            mx.eval(ox, ns)
            assert lib.inkling_prefill_moe_nax_fused_combine(ctypes.byref(cargs), None) == 0

        iters = 11
        split_wall, split_moe_gpu, mlx_comb_wall = [], [], []
        fused_wall, fused_gpu, native_comb_gpu = [], [], []

        for _ in range(iters):
            moe_gpu = ctypes.c_double(0.0)
            t0 = time.perf_counter()
            assert lib.prefill_moe_forward(ctypes.byref(fargs), ctypes.byref(moe_gpu)) == 0
            t1 = time.perf_counter()
            ox, ns = fused_unsort_combine_sconv_resid(
                ys_scratch, inv_scratch, rweights, state, taps, resid, b=B, length=L, top_k=TOP_K
            )
            mx.eval(ox, ns)
            t2 = time.perf_counter()
            split_wall.append((t2 - t0) * 1000.0)
            split_moe_gpu.append(moe_gpu.value)
            mlx_comb_wall.append((t2 - t1) * 1000.0)

            out_x_fresh = mx.zeros((M, D), dtype=mx.bfloat16)
            new_st_fresh = mx.zeros((B, 3, D), dtype=mx.float32)
            mx.eval(out_x_fresh, new_st_fresh)
            p_ox_f, _koxf = mlx_ptr(out_x_fresh)
            p_ns_f, _knsf = mlx_ptr(new_st_fresh)
            cargs.out_x_ptr = p_ox_f
            cargs.new_state_ptr = p_ns_f
            fgpu = ctypes.c_double(0.0)
            t3 = time.perf_counter()
            assert lib.inkling_prefill_moe_nax_fused_combine(ctypes.byref(cargs), ctypes.byref(fgpu)) == 0
            t4 = time.perf_counter()
            fused_wall.append((t4 - t3) * 1000.0)
            fused_gpu.append(fgpu.value)

            c_tot = ctypes.c_double(0.0)
            c_iso = ctypes.c_double(0.0)
            assert lib.prefill_moe_forward_fused_combine_bench(
                ctypes.byref(cargs), 1, -1, ctypes.byref(c_tot), ctypes.byref(c_iso)
            ) == 0
            native_comb_gpu.append(c_iso.value)

        med = lambda xs: sorted(xs)[len(xs) // 2]
        m_split_wall = med(split_wall)
        m_split_moe_gpu = med(split_moe_gpu)
        m_mlx_comb = med(mlx_comb_wall)
        m_fused_wall = med(fused_wall)
        m_fused_gpu = med(fused_gpu)
        m_nat_comb_gpu = med(native_comb_gpu)

        save_wall = m_split_wall - m_fused_wall
        save_gpu_est = (m_split_moe_gpu + m_mlx_comb) - m_fused_gpu
        ys_mb = (T * D * 2) / 1e6
        nat_bw = (ys_mb * 1e-3 / (m_nat_comb_gpu * 1e-3)) if m_nat_comb_gpu > 0 else 0.0

        print(f"\n  [L={L} | T={T} | ys_sorted={ys_mb:.1f} MB]")
        print(f"    Split (3-disp CB + MLX 32-ch combine) : wall={m_split_wall:7.3f} ms  (native 3-disp GPU={m_split_moe_gpu:.3f} ms + MLX combine wall={m_mlx_comb:.3f} ms)")
        print(f"    Fused (4-disp 1-CB + 64-ch combine)   : wall={m_fused_wall:7.3f} ms  (4-disp 1-CB GPU ={m_fused_gpu:.3f} ms | isolated 64-ch combine GPU={m_nat_comb_gpu:.3f} ms, ~{nat_bw:.0f} GB/s ys read)")
        print(f"    Saved / layer                         : wall={save_wall:+7.3f} ms  ({m_split_wall / m_fused_wall:.3f}x) | combine stage: {m_mlx_comb:.3f} -> {m_nat_comb_gpu:.3f} ms ({m_mlx_comb / m_nat_comb_gpu:.2f}x)")
        print(f"    Projected x48 visits (1 chunk L={L})  : wall={save_wall * 48:+7.1f} ms  | 16k TTFT ({16384 // L} chunks): {save_wall * 48 * (16384 // L):+7.1f} ms")

        del x, inds, rweights, state, resid, ys_scratch, inv_scratch, out_x_buf, new_st_buf
        mx.clear_cache()

    lib.prefill_moe_clear_cache()
    lib.prefill_moe_release_scratch()
    del w13, s13, gs13, w2, s2, gs2, taps
    mx.clear_cache()


def main():
    lib = load_lib()
    for mode in ("nvfp4", "mxfp8"):
        verify_bit_identity(lib, mode)
    for mode in ("nvfp4", "mxfp8"):
        benchmark_interleaved(lib, mode)


if __name__ == "__main__":
    main()
