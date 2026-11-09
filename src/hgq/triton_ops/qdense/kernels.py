"""Triton kernels of hgq2's quantized dense, y = q_x(x) @ w + b, with x's quantizer applied on load (w comes quantized).

The quantizers are `quantizers.triton`'s fixed-point device functions, selected by an operand's constexpr form (Q, RND,
OVF, NULL): Q 'quantize' (the stateless quantizer's call), 'round' (rounding only: the trainable-WRAP training call) or
'none' (identity), with the quantizer's round mode, overflow mode and null_xgrad_on_zero_width.

Layouts: a batched GEMM over G (a grid axis), x (G, M, K) @ w (G, K, N) -> y (G, M, N). x (and dx) is a (G, M, K)
buffer, or (M, G, K) with XMG; y (and its cotangent g) (G, M, N), or (M, G, N) with YMG; w (G, K, N) and b (G, N)
contiguous. Bit widths of x are a compact (RG, RB, CB) array read periodically, bw[g % RG, m % RB, k % CB]:
per-tensor (1, 1, 1), per-channel (1, 1, K), per-element with the batch broadcast (1, T, K), per-head (G, 1, K), or a
column period (1, 1, hd). ROWS / COLS (constexpr) say whether they vary along rows / columns; the bit-width blocks are
loaded at that reduced shape ((1, 1), (1, BK), (BM, 1), (BM, BK)), so subexpressions of the traced quantizer that
depend on bit widths only are evaluated at that shape, not per element.

The GEMM's G, N, K and buffer layouts are compile-time constants (from the Spec), so the loops over N and K, their
masks, staged copies and the row strides are specialized to the shape (G 1 drops the group offsets); the row count M,
the program counts along it (S, SPLIT) and the bit-width periods are runtime integers, so a new batch size reuses the
compiled kernels.

Kernels:
- qdense_fwd: y = q(x) @ w + b (FP32 accumulate, tl.dot input_precision PREC), optional trainable-WRAP abs-max of
  q(x) (AMAX 1: per row block, (G * cdiv(M, BM), K) partials; AMAX 2: atomic max into the (RG, RB, CB) int32 view, used when
  the bit widths vary along rows; NEG, a tracing call's: a second plane of whether any q(x) < 0, 0.0 / 1.0, AMAX 2
  as 1.0's bit pattern) and optionally q(x) itself (STORE_Q, checks only).
- qdense_dx: dq = g @ w^T, then the x quantizer's vjp in the epilogue: dx, and the per-element i / f gradient terms
  summed over rows into partials. S programs stride over the M tiles; with bit widths varying along rows (XR) every
  tile of a program starts at the same row offset in the period (S * BM % RB == 0), so the program keeps (BM, BK)
  partials in registers: (G, S / (RB / BM), RB, K) partials; else (G, S, K).
- qdense_dw: dw = q(x)^T @ g, split over M into SPLIT partials (G, SPLIT, K, N), and the bias gradient (G, SPLIT, N).
"""

import functools

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
def _load(ptr, r, c, R, C, RS):
    """Block (r, c) of an (R, C) operand with row stride RS and its mask; 0 outside the operand."""
    mask = (r < R)[:, None] & (c < C)[None, :]
    return tl.load(ptr + r.to(tl.int64)[:, None] * RS + c[None, :], mask=mask, other=0.0), mask


@triton.jit
def _load_q(
    ptr, k_ptr, i_ptr, f_ptr, r, c, R, C, RS, RB, CB, ROWS: tl.constexpr, COLS: tl.constexpr,
    Q: tl.constexpr, RND: tl.constexpr, OVF: tl.constexpr, NULL: tl.constexpr,
):  # fmt: skip
    """Block (r, c) of an (R, C) operand with row stride RS, quantized on load; 0 outside the operand."""
    v, mask = _load(ptr, r, c, R, C, RS)
    if Q != 'none':
        k = _bw(k_ptr, r, c, RB, CB, ROWS, COLS)
        i = _bw(i_ptr, r, c, RB, CB, ROWS, COLS)
        f = _bw(f_ptr, r, c, RB, CB, ROWS, COLS)
        v = tl.where(mask, _quantize(v, k, i, f, Q, RND, OVF, NULL), 0.0)
    return v, mask


@triton.jit
def qdense_fwd(
    x_ptr, kx_ptr, ix_ptr, fx_ptr, w_ptr, b_ptr,
    amax_ptr,  # in-out, zeroed (a JAX Ref: jax-triton 0.4.1 misbinds a Ref after an output parameter)
    y_ptr, qx_ptr,
    M, N: tl.constexpr, K: tl.constexpr, G: tl.constexpr, RG: tl.constexpr, XMG: tl.constexpr, YMG: tl.constexpr, RX, CX,
    QX: tl.constexpr, RNDX: tl.constexpr, OVFX: tl.constexpr, NULLX: tl.constexpr, XR: tl.constexpr, XC: tl.constexpr,
    HAS_BIAS: tl.constexpr, AMAX: tl.constexpr, NEG: tl.constexpr, STORE_Q: tl.constexpr, PREC: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
):  # fmt: skip
    pid = tl.program_id(0)
    g = tl.program_id(1).to(tl.int64) if G > 1 else 0
    SXM, SYM = G * K if XMG else K, G * N if YMG else N  # row strides of x and y
    x_ptr += g * K if XMG else g * M * K
    y_ptr += g * N if YMG else g * M * N
    w_ptr += g * K * N
    b_ptr += g * N
    gb = (g % RG) * RX * CX  # this g's bit widths
    kx_ptr += gb
    ix_ptr += gb
    fx_ptr += gb
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
        xq, xmask = _load_q(x_ptr, kx_ptr, ix_ptr, fx_ptr, rm, rk, M, K, SXM, RX, CX, XR, XC, QX, RNDX, OVFX, NULLX)
        if AMAX == 1:  # (planes, G * n_m, K)
            a = tl.reduce(tl.abs(xq), 0, _nanmax)
            tl.store(amax_ptr + (g * n_m + pid_m) * K + rk, a, mask=(rk < K) & lead)
            if NEG:
                neg = tl.max((xq < 0).to(tl.float32), 0)
                tl.store(amax_ptr + ((G + g) * n_m + pid_m) * K + rk, neg, mask=(rk < K) & lead)
        if AMAX == 2:
            # |q(x)| >= 0 (or NaN), so its int32 bit pattern orders as the float does (NaN above inf)
            ai = tl.abs(xq).to(tl.int32, bitcast=True)
            offs = gb + (rm % RX).to(tl.int64)[:, None] * CX + (rk % CX)[None, :]  # (planes, RG, RX, CX)
            tl.atomic_max(amax_ptr + offs, ai, mask=xmask & lead)
            if NEG:
                neg = (xq < 0).to(tl.float32).to(tl.int32, bitcast=True)
                tl.atomic_max(amax_ptr + RG * RX * CX + offs, neg, mask=xmask & lead)
        if STORE_Q:  # (G, M, K)
            tl.store(qx_ptr + (g * M + rm).to(tl.int64)[:, None] * K + rk[None, :], xq, mask=xmask & lead)
        w, _ = _load(w_ptr, rk, rn, K, N, N)
        acc = tl.dot(xq, w, acc, input_precision=PREC)
    if HAS_BIAS:
        acc += tl.load(b_ptr + rn, mask=rn < N, other=0.0)[None, :]
    ymask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(y_ptr + rm.to(tl.int64)[:, None] * SYM + rn[None, :], acc, mask=ymask)


@triton.jit
def qdense_dx(
    g_ptr, w_ptr, x_ptr, kx_ptr, ix_ptr, fx_ptr,
    dx_ptr, pi_ptr, pf_ptr,
    M, N: tl.constexpr, K: tl.constexpr, G: tl.constexpr, RG: tl.constexpr, XMG: tl.constexpr, YMG: tl.constexpr, RX, CX, S,
    QX: tl.constexpr, RNDX: tl.constexpr, OVFX: tl.constexpr, NULLX: tl.constexpr, XR: tl.constexpr, XC: tl.constexpr, PREC: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):  # fmt: skip
    pid_s = tl.program_id(0)
    pid_k = tl.program_id(1)
    g = tl.program_id(2).to(tl.int64) if G > 1 else 0
    SXM, SYM = G * K if XMG else K, G * N if YMG else N
    g_ptr += g * N if YMG else g * M * N
    w_ptr += g * K * N
    x_ptr += g * K if XMG else g * M * K
    dx_ptr += g * K if XMG else g * M * K
    gb = (g % RG) * RX * CX
    kx_ptr += gb
    ix_ptr += gb
    fx_ptr += gb
    rk = pid_k * BK + tl.arange(0, BK)
    kmask = rk < K
    n_m = tl.cdiv(M, BM)
    if XR:
        acc_i = tl.zeros((BM, BK), tl.float32)
        acc_f = tl.zeros((BM, BK), tl.float32)
    else:
        acc_i = tl.zeros((BK,), tl.float32)
        acc_f = tl.zeros((BK,), tl.float32)
    if QX != 'none':  # the same for every tile of the program (XR: S * BM % RX == 0)
        rm0 = pid_s * BM + tl.arange(0, BM)
        kx = _bw(kx_ptr, rm0, rk, RX, CX, XR, XC)
        ix = _bw(ix_ptr, rm0, rk, RX, CX, XR, XC)
        fx = _bw(fx_ptr, rm0, rk, RX, CX, XR, XC)
    for j in range(pid_s, n_m, S):
        rm = j * BM + tl.arange(0, BM)
        acc = tl.zeros((BM, BK), tl.float32)
        for n0 in range(0, N, BN):
            rn = n0 + tl.arange(0, BN)
            gy, _ = _load(g_ptr, rm, rn, M, N, SYM)
            w, _ = _load(w_ptr, rk, rn, K, N, N)
            acc = tl.dot(gy, tl.trans(w), acc, input_precision=PREC)
        xmask = (rm < M)[:, None] & kmask[None, :]
        xoffs = rm.to(tl.int64)[:, None] * SXM + rk[None, :]
        if QX != 'none':
            x = tl.load(x_ptr + xoffs, mask=xmask, other=0.0)
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
    if QX != 'none':  # (G, S / (RX / BM), RX, K) partials with XR, else (G, S, K)
        if XR:
            per = RX // BM
            t0 = (pid_s % per) * BM
            offs = (g * (S // per) + pid_s // per).to(tl.int64) * RX * K + (t0 + tl.arange(0, BM))[:, None] * K + rk[None, :]
            tl.store(pi_ptr + offs, acc_i, mask=kmask[None, :])
            tl.store(pf_ptr + offs, acc_f, mask=kmask[None, :])
        else:
            tl.store(pi_ptr + (g * S + pid_s) * K + rk, acc_i, mask=kmask)
            tl.store(pf_ptr + (g * S + pid_s) * K + rk, acc_f, mask=kmask)


@triton.jit
def qdense_dw(
    x_ptr, kx_ptr, ix_ptr, fx_ptr, g_ptr,
    pw_ptr, pb_ptr,
    M, N: tl.constexpr, K: tl.constexpr, G: tl.constexpr, RG: tl.constexpr, XMG: tl.constexpr, YMG: tl.constexpr, RX, CX, SPLIT,
    QX: tl.constexpr, RNDX: tl.constexpr, OVFX: tl.constexpr, NULLX: tl.constexpr, XR: tl.constexpr, XC: tl.constexpr, HAS_BIAS: tl.constexpr, PREC: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):  # fmt: skip
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2) % SPLIT if G > 1 else tl.program_id(2)  # axis 2: G * SPLIT programs
    g = (tl.program_id(2) // SPLIT).to(tl.int64) if G > 1 else 0
    SXM, SYM = G * K if XMG else K, G * N if YMG else N
    x_ptr += g * K if XMG else g * M * K
    g_ptr += g * N if YMG else g * M * N
    gb = (g % RG) * RX * CX
    kx_ptr += gb
    ix_ptr += gb
    fx_ptr += gb
    rk = pid_k * BK + tl.arange(0, BK)
    rn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BK, BN), tl.float32)
    accb = tl.zeros((BN,), tl.float32)
    for j in range(pid_s, tl.cdiv(M, BM), SPLIT):
        rm = j * BM + tl.arange(0, BM)
        xq, _ = _load_q(x_ptr, kx_ptr, ix_ptr, fx_ptr, rm, rk, M, K, SXM, RX, CX, XR, XC, QX, RNDX, OVFX, NULLX)
        gy, _ = _load(g_ptr, rm, rn, M, N, SYM)
        acc = tl.dot(tl.trans(xq), gy, acc, input_precision=PREC)
        if HAS_BIAS:
            accb += tl.sum(gy, 0)
    wmask = (rk < K)[:, None] & (rn < N)[None, :]
    part = (g * SPLIT + pid_s).to(tl.int64)  # (G, SPLIT, K, N) and (G, SPLIT, N) partials
    tl.store(pw_ptr + part * K * N + rk[:, None] * N + rn[None, :], acc, mask=wmask)
    if HAS_BIAS:
        tl.store(pb_ptr + part * N + rn, accb, mask=(rn < N) & (pid_k == 0))


KERNELS = {'fwd': qdense_fwd, 'dx': qdense_dx, 'dw': qdense_dw}


def compile_spec(spec, kernel, target, num_stages=None, M=1 << 16):
    """``kernel`` ('fwd', 'dx', 'dw') of ``spec`` compiled for ``target`` (a triton GPUTarget) as the bindings
    launch it on ``M`` rows: the Spec's constexprs, runtime integers specialized as Triton's launcher does (1 a constant,
    multiples of 16 divisible; jax-triton 0.4.1 specializes 1 only), 16-byte aligned operands; at ``num_stages`` (default
    the Spec's). Compiles only (no device)."""
    from triton.compiler import ASTSource

    fn = KERNELS[kernel]
    consts = getattr(spec, f'{kernel}_consts')(M)
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


_FITTED = {}


def fit_stages(spec, target, limit):
    """``spec`` with each kernel's num_stages lowered until the kernel, compiled for ``target``, needs at most ``limit``
    bytes of shared memory (the device's per-block limit; it depends on the tiles, not on M); a kernel at one stage is
    not compiled here (nothing to fall back to: its launch reports an overflow). Cached, the target by repr (an XPU
    target's arch is a dict)."""
    key = spec, repr(target), limit
    if key not in _FITTED:
        for kernel in KERNELS:
            n = spec.cfg(kernel)['num_stages']
            while n > 1 and compile_spec(spec, kernel, target, n).metadata.shared > limit:
                n -= 1
            spec = spec.with_cfg(kernel, num_stages=n)
        _FITTED[key] = spec
    return _FITTED[key]


def device_properties(index=None) -> dict:
    """The active Triton driver's properties of device ``index`` (default its current device): max_shared_mem, the
    per-block limit its launcher checks (CUDA: the opt-in maximum), multiprocessor_count (CUDA SMs, ROCm CUs), ..."""
    from triton.runtime import driver

    return _device_properties(driver.active.get_current_device() if index is None else index)


@functools.cache
def _device_properties(index) -> dict:
    """Cached: the driver query takes ~0.7 ms (CUDA), three times per eager layer call otherwise."""
    from triton.runtime import driver

    return driver.active.utils.get_device_properties(index)


def fit_device(spec, index=None):
    """fit_stages for the active Triton driver's current target, which Triton and jax-triton compile for ('cuda',
    compute capability, 32; 'hip', gfx arch, wavefront size; 'xpu'), against device ``index``'s shared memory limit."""
    from triton.runtime import driver

    return fit_stages(spec, driver.active.get_current_target(), device_properties(index)['max_shared_mem'])
