"""Triton kernels of hgq2's quantized dense, y = q_x(x) @ q_w(w) + b, with the quantizers applied on load.

The quantizers are `quantizers.triton`'s fixed-point device functions, selected by an operand's constexpr form (Q, RND,
OVF, NULL): Q 'quantize' (the stateless quantizer's call), 'round' (rounding only: the trainable-WRAP training call) or
'none' (identity), with the quantizer's round mode, overflow mode and null_xgrad_on_zero_width.

Layouts (row-major, contiguous): x (M, K), w (K, N), y and g (M, N). Bit widths of an operand (R, C) are a compact
(RB, CB) array read periodically, bw[r % RB, c % CB]: per-tensor (1, 1), per-channel (1, C), per-element with the batch
broadcast (T, C), or a column period (1, hd). ROWS / COLS (constexpr) say whether they vary along rows / columns; the
bit-width blocks are loaded at that reduced shape ((1, 1), (1, BC), (BR, 1), (BR, BC)), so subexpressions of the traced
quantizer that depend on bit widths only are evaluated at that shape, not per element.

Problem dimensions (M, N, K; R, C) and loop controls (S, SPLIT) are compile-time constants (from the Spec), so loops,
masks and staged copies are specialized to the shape; the bit-width periods stay runtime integers.

Kernels:
- qdense_fwd: y = q(x) @ q(w) + b (FP32 accumulate, tl.dot input_precision PREC), optional trainable-WRAP abs-max of
  q(x) (AMAX 1: per row block, (cdiv(M, BM), K) partials; AMAX 2: atomic max into the (RB, CB) int32 view, used when
  the bit widths vary along rows) and optionally q(x) itself (STORE_Q, checks only).
- qdense_dx: dq = g @ q(w)^T, then the x quantizer's vjp in the epilogue: dx, and the per-element i / f gradient terms
  summed over rows into partials. S programs stride over the M tiles; with bit widths varying along rows (XR) every
  tile of a program starts at the same row offset in the period (S * BM % RB == 0), so the program keeps (BM, BK)
  partials in registers: (S / (RB / BM), RB, K) partials; else (S, K).
- qdense_dw: dwq = q(x)^T @ g, split over M into SPLIT partials (SPLIT, K, N), and the bias gradient (SPLIT, N).
- qvjp2d: an operand's quantizer vjp elementwise (the weight's, when w is quantized on load): dw and the per-element
  i / f gradient terms (folded to the bit-width shape by the caller; weights are small).
"""

from functools import lru_cache

import triton
import triton.language as tl
from quantizers.triton import fixed_quantize, fixed_quantize_vjp, fixed_round, fixed_round_vjp


@triton.jit
def _nanmax(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _bw(ptr, r, c, RB, CB, ROWS: tl.constexpr, COLS: tl.constexpr):
    """Bit widths of the elements (r, c) from the periodic (RB, CB) array; always in bounds (no mask needed)."""
    if ROWS:
        ro = (r % RB).to(tl.int64)[:, None] * CB
    else:
        ro = tl.zeros((1, 1), tl.int64)
    if COLS:
        co = (c % CB).to(tl.int64)[None, :]
    else:
        co = tl.zeros((1, 1), tl.int64)
    return tl.load(ptr + (ro + co))


@triton.jit
def _quantize(v, k, i, f, Q: tl.constexpr, RND: tl.constexpr, OVF: tl.constexpr, NULL: tl.constexpr):
    if Q == 'round':
        v = fixed_round(v, f, RND)
    else:
        v = fixed_quantize(v, k, i, f, RND, OVF, NULL)
    return v


@triton.jit
def _quantize_vjp(v, k, i, f, g, Q: tl.constexpr, RND: tl.constexpr, OVF: tl.constexpr, NULL: tl.constexpr):
    """(dv, di, df) of `_quantize` for the cotangent g, at the broadcast shape."""
    if Q == 'round':
        dv, df = fixed_round_vjp(v, f, g, RND)
        di = tl.zeros_like(df)
    else:
        dv, _, di, df = fixed_quantize_vjp(v, k, i, f, g, RND, OVF, NULL)
    return dv, di, df


@triton.jit
def _load_q(
    ptr, k_ptr, i_ptr, f_ptr, r, c, R, C, RB, CB, ROWS: tl.constexpr, COLS: tl.constexpr,
    Q: tl.constexpr, RND: tl.constexpr, OVF: tl.constexpr, NULL: tl.constexpr,
):  # fmt: skip
    """Block (r, c) of a row-major (R, C) operand, quantized on load; 0 outside the operand."""
    mask = (r < R)[:, None] & (c < C)[None, :]
    v = tl.load(ptr + r.to(tl.int64)[:, None] * C + c[None, :], mask=mask, other=0.0)
    if Q != 'none':
        k = _bw(k_ptr, r, c, RB, CB, ROWS, COLS)
        i = _bw(i_ptr, r, c, RB, CB, ROWS, COLS)
        f = _bw(f_ptr, r, c, RB, CB, ROWS, COLS)
        v = tl.where(mask, _quantize(v, k, i, f, Q, RND, OVF, NULL), 0.0)
    return v, mask


@triton.jit
def qdense_fwd(
    x_ptr, kx_ptr, ix_ptr, fx_ptr, w_ptr, kw_ptr, iw_ptr, fw_ptr, b_ptr,
    y_ptr, amax_ptr, qx_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, RX, CX, RW, CW,
    QX: tl.constexpr, RNDX: tl.constexpr, OVFX: tl.constexpr, NULLX: tl.constexpr, XR: tl.constexpr, XC: tl.constexpr,
    QW: tl.constexpr, RNDW: tl.constexpr, OVFW: tl.constexpr, NULLW: tl.constexpr, WR: tl.constexpr, WC: tl.constexpr,
    HAS_BIAS: tl.constexpr, AMAX: tl.constexpr, STORE_Q: tl.constexpr, PREC: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
):  # fmt: skip
    pid = tl.program_id(0)
    n_m = tl.cdiv(M, BM)
    n_n = tl.cdiv(N, BN)
    width = GROUP_M * n_n  # grouped launch order: GROUP_M row blocks share their w column blocks in L2
    first_m = (pid // width) * GROUP_M
    gsize = tl.minimum(n_m - first_m, GROUP_M)
    pid_m = first_m + (pid % width) % gsize
    pid_n = (pid % width) // gsize
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    lead = pid_n == 0  # the programs that write per-x outputs (abs-max, q(x))
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        xq, xmask = _load_q(x_ptr, kx_ptr, ix_ptr, fx_ptr, rm, rk, M, K, RX, CX, XR, XC, QX, RNDX, OVFX, NULLX)
        if AMAX == 1:
            a = tl.reduce(tl.abs(xq), 0, _nanmax)
            tl.store(amax_ptr + pid_m * K + rk, a, mask=(rk < K) & lead)
        if AMAX == 2:
            # |q(x)| >= 0 (or NaN), so its int32 bit pattern orders as the float does (NaN above inf)
            ai = tl.abs(xq).to(tl.int32, bitcast=True)
            offs = (rm % RX).to(tl.int64)[:, None] * CX + (rk % CX)[None, :]
            tl.atomic_max(amax_ptr + offs, ai, mask=xmask & lead)
        if STORE_Q:
            tl.store(qx_ptr + rm.to(tl.int64)[:, None] * K + rk[None, :], xq, mask=xmask & lead)
        wq, _ = _load_q(w_ptr, kw_ptr, iw_ptr, fw_ptr, rk, rn, K, N, RW, CW, WR, WC, QW, RNDW, OVFW, NULLW)
        acc = tl.dot(xq, wq, acc, input_precision=PREC)
    if HAS_BIAS:
        acc += tl.load(b_ptr + rn, mask=rn < N, other=0.0)[None, :]
    ymask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(y_ptr + rm.to(tl.int64)[:, None] * N + rn[None, :], acc, mask=ymask)


@triton.jit
def qdense_dx(
    g_ptr, w_ptr, kw_ptr, iw_ptr, fw_ptr, x_ptr, kx_ptr, ix_ptr, fx_ptr,
    dx_ptr, pi_ptr, pf_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, RX, CX, RW, CW, S: tl.constexpr,
    QX: tl.constexpr, RNDX: tl.constexpr, OVFX: tl.constexpr, NULLX: tl.constexpr, XR: tl.constexpr, XC: tl.constexpr,
    QW: tl.constexpr, RNDW: tl.constexpr, OVFW: tl.constexpr, NULLW: tl.constexpr, WR: tl.constexpr, WC: tl.constexpr, PREC: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):  # fmt: skip
    pid_s = tl.program_id(0)
    pid_k = tl.program_id(1)
    rk = pid_k * BK + tl.arange(0, BK)
    kmask = rk < K
    n_m = tl.cdiv(M, BM)
    if XR:
        acc_i = tl.zeros((BM, BK), tl.float32)
        acc_f = tl.zeros((BM, BK), tl.float32)
    else:
        acc_i = tl.zeros((BK,), tl.float32)
        acc_f = tl.zeros((BK,), tl.float32)
    for j in range(pid_s, n_m, S):
        rm = j * BM + tl.arange(0, BM)
        acc = tl.zeros((BM, BK), tl.float32)
        for n0 in range(0, N, BN):
            rn = n0 + tl.arange(0, BN)
            gmask = (rm < M)[:, None] & (rn < N)[None, :]
            g = tl.load(g_ptr + rm.to(tl.int64)[:, None] * N + rn[None, :], mask=gmask, other=0.0)
            wq, _ = _load_q(w_ptr, kw_ptr, iw_ptr, fw_ptr, rk, rn, K, N, RW, CW, WR, WC, QW, RNDW, OVFW, NULLW)
            acc = tl.dot(g, tl.trans(wq), acc, input_precision=PREC)
        xmask = (rm < M)[:, None] & kmask[None, :]
        xoffs = rm.to(tl.int64)[:, None] * K + rk[None, :]
        if QX != 'none':
            x = tl.load(x_ptr + xoffs, mask=xmask, other=0.0)
            kx = _bw(kx_ptr, rm, rk, RX, CX, XR, XC)
            ix = _bw(ix_ptr, rm, rk, RX, CX, XR, XC)
            fx = _bw(fx_ptr, rm, rk, RX, CX, XR, XC)
            dx, di, df = _quantize_vjp(x, kx, ix, fx, acc, QX, RNDX, OVFX, NULLX)
            di = tl.where(xmask, di, 0.0)
            df = tl.where(xmask, df, 0.0)
            if XR:
                acc_i += di
                acc_f += df
            else:
                acc_i += tl.sum(di, 0)
                acc_f += tl.sum(df, 0)
        else:
            dx = acc
        tl.store(dx_ptr + xoffs, dx, mask=xmask)
    if QX != 'none':
        if XR:
            per = RX // BM
            t0 = (pid_s % per) * BM
            offs = (pid_s // per).to(tl.int64) * RX * K + (t0 + tl.arange(0, BM))[:, None] * K + rk[None, :]
            tl.store(pi_ptr + offs, acc_i, mask=kmask[None, :])
            tl.store(pf_ptr + offs, acc_f, mask=kmask[None, :])
        else:
            tl.store(pi_ptr + pid_s * K + rk, acc_i, mask=kmask)
            tl.store(pf_ptr + pid_s * K + rk, acc_f, mask=kmask)


@triton.jit
def qdense_dw(
    x_ptr, kx_ptr, ix_ptr, fx_ptr, g_ptr,
    pw_ptr, pb_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, RX, CX, SPLIT: tl.constexpr,
    QX: tl.constexpr, RNDX: tl.constexpr, OVFX: tl.constexpr, NULLX: tl.constexpr, XR: tl.constexpr, XC: tl.constexpr, HAS_BIAS: tl.constexpr, PREC: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):  # fmt: skip
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)
    rk = pid_k * BK + tl.arange(0, BK)
    rn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BK, BN), tl.float32)
    accb = tl.zeros((BN,), tl.float32)
    for j in range(pid_s, tl.cdiv(M, BM), SPLIT):
        rm = j * BM + tl.arange(0, BM)
        xq, _ = _load_q(x_ptr, kx_ptr, ix_ptr, fx_ptr, rm, rk, M, K, RX, CX, XR, XC, QX, RNDX, OVFX, NULLX)
        gmask = (rm < M)[:, None] & (rn < N)[None, :]
        g = tl.load(g_ptr + rm.to(tl.int64)[:, None] * N + rn[None, :], mask=gmask, other=0.0)
        acc = tl.dot(tl.trans(xq), g, acc, input_precision=PREC)
        if HAS_BIAS:
            accb += tl.sum(g, 0)
    wmask = (rk < K)[:, None] & (rn < N)[None, :]
    tl.store(pw_ptr + pid_s.to(tl.int64) * K * N + rk[:, None] * N + rn[None, :], acc, mask=wmask)
    if HAS_BIAS:
        tl.store(pb_ptr + pid_s * N + rn, accb, mask=(rn < N) & (pid_k == 0))


@triton.jit
def qvjp2d(
    w_ptr, k_ptr, i_ptr, f_ptr, g_ptr,
    dw_ptr, ti_ptr, tf_ptr,
    R: tl.constexpr, C: tl.constexpr, RB, CB,
    Q: tl.constexpr, RND: tl.constexpr, OVF: tl.constexpr, NULL: tl.constexpr,
    ROWS: tl.constexpr, COLS: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
):  # fmt: skip
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    mask = (r < R)[:, None] & (c < C)[None, :]
    offs = r.to(tl.int64)[:, None] * C + c[None, :]
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0)
    k = _bw(k_ptr, r, c, RB, CB, ROWS, COLS)
    i = _bw(i_ptr, r, c, RB, CB, ROWS, COLS)
    f = _bw(f_ptr, r, c, RB, CB, ROWS, COLS)
    dw, ti, tf = _quantize_vjp(w, k, i, f, g, Q, RND, OVF, NULL)
    tl.store(dw_ptr + offs, dw, mask=mask)
    tl.store(ti_ptr + offs, ti, mask=mask)
    tl.store(tf_ptr + offs, tf, mask=mask)


KERNELS = {'fwd': qdense_fwd, 'dx': qdense_dx, 'dw': qdense_dw, 'vjp2d': qvjp2d}


def compile_spec(spec, kernel, target, num_stages=None):
    """``kernel`` ('fwd', 'dx', 'dw', 'vjp2d') of ``spec`` compiled for ``target`` (a triton GPUTarget) as the bindings
    launch it: the Spec's constexprs, runtime integers specialized as Triton and jax-triton do (1 a constant, multiples
    of 16 divisible), 16-byte aligned operands; at ``num_stages`` (default the Spec's). Compiles only (no device)."""
    from triton.compiler import ASTSource

    fn = KERNELS[kernel]
    consts = getattr(spec, f'{kernel}_consts')()
    constexprs = {fn.arg_names[i] for i in fn.constexprs}
    sig, const, attrs = {}, {}, {}
    for i, name in enumerate(fn.arg_names):
        if name.endswith('_ptr'):
            sig[name] = '*i32' if name == 'amax_ptr' and spec.amax_mode == 2 else '*fp32'
            attrs[(i,)] = [['tt.divisibility', 16]]
        elif name in constexprs or consts[name] == 1:
            sig[name], const[name] = 'constexpr', consts[name]
        else:
            sig[name] = 'i32'
            if consts[name] % 16 == 0:
                attrs[(i,)] = [['tt.divisibility', 16]]
    c = spec.cfg(kernel)
    options = dict(num_warps=c['num_warps'], num_stages=num_stages or c['num_stages'])
    return triton.compile(ASTSource(fn, sig, const, attrs), target=target, options=options)


@lru_cache(maxsize=None)
def fit_stages(spec, target, limit):
    """``spec`` with each kernel's num_stages lowered until the kernel, compiled for ``target``, needs at most ``limit``
    bytes of shared memory (the device's per-block limit); a kernel at one stage is not compiled here (nothing to fall
    back to: its launch reports an overflow)."""
    for kernel in KERNELS:
        n = spec.cfg(kernel)['num_stages']
        while n > 1 and compile_spec(spec, kernel, target, n).metadata.shared > limit:
            n -= 1
        spec = spec.with_cfg(kernel, num_stages=n)
    return spec


def fit_device(spec, index, arch):
    """fit_stages for CUDA device ``index`` of compute capability ``arch`` (e.g. 89), against the per-block shared memory
    limit that Triton's launcher checks (the device's opt-in maximum)."""
    from triton.backends.compiler import GPUTarget
    from triton.runtime import driver

    limit = driver.active.utils.get_device_properties(index)['max_shared_mem']
    return fit_stages(spec, GPUTarget('cuda', arch, 32), limit)
