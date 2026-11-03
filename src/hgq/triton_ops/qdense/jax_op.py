"""hgq2's fused quantized dense in JAX: a jax.custom_vjp around the Triton kernels of qdense.kernels, launched with
jax_triton.triton_call on the GPU.

qdense(x (M, K), kx, ix, fx (rx, cx), w (K, N), kw, iw, fw (rw, cw), b (N,), spec) -> (y (M, N), amax (rx, cx), qx)
(see qdense.torch_op for the meaning; the two bindings launch the same kernels with the same Spec).
"""

import functools

import jax
import jax.numpy as jnp
import jax_triton

from . import kernels as KS
from .spec import Spec


def _launch(kernel, grid, ins, outs, consts, cfg):
    """Run ``kernel`` on operands ``ins`` (arrays and Refs, the leading kernel parameters); ``outs`` (ShapeDtypeStructs)
    bind to the following pointer parameters."""
    return jax_triton.triton_call(
        *ins,
        kernel=kernel,
        out_type=tuple(outs),
        grid=grid,
        num_warps=cfg['num_warps'],
        num_stages=cfg['num_stages'],
        **consts,
    )


def _fitted(s):
    """The Spec with its stages fitted to the shared memory of the first device, which jax-triton compiles for
    (qdense.kernels.fit_device)."""
    d = jax.devices()[0]
    major, minor = (int(v) for v in d.compute_capability.split('.'))
    return KS.fit_device(s, d.local_hardware_id or 0, major * 10 + minor)


def _fold_cols(p, c):
    return jnp.sum(p.reshape(*p.shape[:-1], p.shape[-1] // c, c), -2)


def _fwd(s, x, kx, ix, fx, w, kw, iw, fw, b):
    s, M = _fitted(s), x.shape[0]
    f32 = jnp.float32
    ashape, adtype = s.amax_buffer(M)
    amax = jax.new_ref(jnp.zeros(ashape, jnp.dtype(adtype)))  # in-out: the atomic max starts at 0
    outs = (jax.ShapeDtypeStruct((M, s.N), f32), jax.ShapeDtypeStruct((M, s.K) if s.store_q else (1,), f32))
    c = s.cfg('fwd')
    consts = s.fwd_consts(M)
    y, qx = _launch(KS.qdense_fwd, s.fwd_grid(M), (x, kx, ix, fx, w, kw, iw, fw, b, amax), outs, consts, c)
    amax = jax.freeze(amax)
    if s.amax_mode == 1:  # (planes, n_m, K) -> (planes, cx); jnp.max propagates NaN, as XLA's max in hgq's call
        amax = jnp.max(amax.reshape(amax.shape[0], -1, s.cx), 1)
    elif s.amax_mode == 2:
        amax = jax.lax.bitcast_convert_type(amax, f32)
    else:
        amax = jnp.zeros((s.rx, s.cx), f32)
    return y, amax.reshape(s.amax_shape), qx


def _bwd(s, g, x, kx, ix, fx, w, kw, iw, fw):
    s, M = _fitted(s), x.shape[0]
    f32 = jnp.float32
    S = jax.ShapeDtypeStruct
    c = s.cfg('dx')
    consts = s.dx_consts(M)
    part = S(s.dx_partials(M), f32)
    dx, pi, pf = _launch(KS.qdense_dx, s.dx_grid(M), (g, w, kw, iw, fw, x, kx, ix, fx), (S((M, s.K), f32), part, part), consts, c)
    if s.qx[0] == 'none':
        dix = dfx = jnp.zeros((s.rx, s.cx), f32)
    else:
        dix = _fold_cols(jnp.sum(pi, 0), s.cx).reshape(s.rx, s.cx)
        dfx = _fold_cols(jnp.sum(pf, 0), s.cx).reshape(s.rx, s.cx)
    c, split = s.cfg('dw'), s.dw_split(M)
    consts = s.dw_consts(M)
    outs = (S((split, s.K, s.N), f32), S((split, s.N) if s.bias else (1,), f32))
    pw, pb = _launch(KS.qdense_dw, s.dw_grid(M), (x, kx, ix, fx, g), outs, consts, c)
    dwq = jnp.sum(pw, 0)
    db = jnp.sum(pb, 0) if s.bias else jnp.zeros((s.N,), f32)
    if s.qw[0] == 'none':
        return dx, dix, dfx, dwq, jnp.zeros((s.rw, s.cw), f32), jnp.zeros((s.rw, s.cw), f32), db
    c = s.cfg('vjp2d')
    consts = s.vjp2d_consts()
    full = S((s.K, s.N), f32)
    dw, ti, tf = _launch(KS.qvjp2d, s.vjp2d_grid(), (w, kw, iw, fw, dwq), (full, full, full), consts, c)
    diw = jnp.sum(ti.reshape(s.K // s.rw, s.rw, s.N // s.cw, s.cw), (0, 2))
    dfw = jnp.sum(tf.reshape(s.K // s.rw, s.rw, s.N // s.cw, s.cw), (0, 2))
    return dx, dix, dfx, dw, diw, dfw, db


@functools.cache
def _op(s: Spec):
    @jax.custom_vjp
    def op(x, kx, ix, fx, w, kw, iw, fw, b):
        return _fwd(s, x, kx, ix, fx, w, kw, iw, fw, b)

    def op_fwd(x, kx, ix, fx, w, kw, iw, fw, b):
        return _fwd(s, x, kx, ix, fx, w, kw, iw, fw, b), (x, kx, ix, fx, w, kw, iw, fw, b)

    def op_bwd(res, cts):
        if not s.dx_fits:
            raise ValueError(f'no dx row tile divides the row period {s.rx}: this inference Spec has no backward')
        x, kx, ix, fx, w, kw, iw, fw, b = res
        gy = cts[0]  # amax and qx are not differentiated (hgq stop-gradients the integer-bit update)
        dx, dix, dfx, dw, diw, dfw, db = _bwd(s, gy, x, kx, ix, fx, w, kw, iw, fw)
        z = jnp.zeros_like
        return (
            dx,
            z(kx),
            dix,
            dfx,
            dw,
            z(kw),
            diw if s.qw[0] != 'none' else z(iw),
            dfw if s.qw[0] != 'none' else z(fw),
            db if s.bias else z(b),
        )

    op.defvjp(op_fwd, op_bwd)
    return op


def qdense(x, kx, ix, fx, w, kw, iw, fw, b, spec: Spec):
    return _op(spec)(x, kx, ix, fx, w, kw, iw, fw, b)


def n_sm():
    d = jax.devices()[0]
    return getattr(d, 'core_count', None) or 108
