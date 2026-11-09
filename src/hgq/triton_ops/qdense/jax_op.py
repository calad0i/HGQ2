"""hgq2's fused quantized dense in JAX: a jax.custom_vjp around the Triton kernels of qdense.kernels, launched with
jax_triton.triton_call on the GPU.

qdense(x (G, M, K), kx, ix, fx (rg, rx, cx), w (G * K, N), b (G * N,), spec) -> (y (G, M, N), amax (rg, rx, cx), qx)
(see qdense.torch_op for the meaning and the (M, G, .) layouts; the two bindings launch the same kernels with the same
Spec).
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
    return KS.fit_device(s, jax.devices()[0].local_hardware_id or 0)


def _fold_cols(p, c):
    return jnp.sum(p.reshape(*p.shape[:-1], p.shape[-1] // c, c), -2)


def _fold_bw(s, p):
    """(G, programs, [rx,] K) dx partials -> (rg, rx, cx)."""
    return jnp.sum(_fold_cols(jnp.sum(p, 1), s.cx).reshape(s.G // s.rg, *s.bw_shape), 0)


def _fwd(s, x, kx, ix, fx, w, b):
    s, M = _fitted(s), x.shape[0 if s.xmg else 1]
    f32 = jnp.float32
    ashape, adtype = s.amax_buffer(M)
    amax = jax.new_ref(jnp.zeros(ashape, jnp.dtype(adtype)))  # in-out: the atomic max starts at 0
    y_shape = (M, s.G, s.N) if s.ymg else (s.G, M, s.N)
    outs = (jax.ShapeDtypeStruct(y_shape, f32), jax.ShapeDtypeStruct((s.G, M, s.K) if s.store_q else (1,), f32))
    c = s.cfg('fwd')
    consts = s.fwd_consts(M)
    y, qx = _launch(KS.qdense_fwd, s.fwd_grid(M), (x, kx, ix, fx, w, b, amax), outs, consts, c)
    amax = jax.freeze(amax)
    if s.amax_mode == 1:  # (planes, G * n_m, K) -> (planes, rg, cx); jnp.max propagates NaN, as XLA's max in hgq's call
        amax = jnp.max(amax.reshape(amax.shape[0], s.G // s.rg, s.rg, -1, s.cx), (1, 3))
    elif s.amax_mode == 2:
        amax = jax.lax.bitcast_convert_type(amax, f32)
    else:
        amax = jnp.zeros(s.bw_shape, f32)
    return y, amax.reshape(s.amax_shape), qx


def _bwd(s, g, x, kx, ix, fx, w):
    s, M = _fitted(s), x.shape[0 if s.xmg else 1]
    f32 = jnp.float32
    S = jax.ShapeDtypeStruct
    c = s.cfg('dx')
    consts = s.dx_consts(M)
    part = S(s.dx_partials(M), f32)
    dx, pi, pf = _launch(KS.qdense_dx, s.dx_grid(M), (g, w, x, kx, ix, fx), (S(x.shape, f32), part, part), consts, c)
    if s.qx[0] == 'none':
        dix = dfx = jnp.zeros(s.bw_shape, f32)
    else:
        dix, dfx = _fold_bw(s, pi), _fold_bw(s, pf)
    c, split = s.cfg('dw'), s.dw_split(M)
    consts = s.dw_consts(M)
    outs = (S((s.G, split, s.K, s.N), f32), S((s.G, split, s.N) if s.bias else (1,), f32))
    pw, pb = _launch(KS.qdense_dw, s.dw_grid(M), (x, kx, ix, fx, g), outs, consts, c)
    dw = jnp.sum(pw, 1).reshape(s.G * s.K, s.N)
    db = jnp.sum(pb, 1).reshape(-1) if s.bias else jnp.zeros((s.G * s.N,), f32)
    return dx, dix, dfx, dw, db


@functools.cache
def _op(s: Spec):
    @jax.custom_vjp
    def op(x, kx, ix, fx, w, b):
        return _fwd(s, x, kx, ix, fx, w, b)

    def op_fwd(x, kx, ix, fx, w, b):
        return _fwd(s, x, kx, ix, fx, w, b), (x, kx, ix, fx, w, b)

    def op_bwd(res, cts):
        if not s.dx_fits:
            raise ValueError(f'no dx row tile divides the row period {s.rx}: this inference Spec has no backward')
        x, kx, ix, fx, w, b = res
        gy = cts[0]  # amax and qx are not differentiated (hgq stop-gradients the integer-bit update)
        dx, dix, dfx, dw, db = _bwd(s, gy, x, kx, ix, fx, w)
        return dx, jnp.zeros_like(kx), dix, dfx, dw, db if s.bias else jnp.zeros_like(b)

    op.defvjp(op_fwd, op_bwd)
    return op


def qdense(x, kx, ix, fx, w, b, spec: Spec):
    return _op(spec)(x, kx, ix, fx, w, b)


def n_sm() -> int:
    """The multiprocessor count of the first device (qdense.kernels.device_properties)."""
    return KS.device_properties(jax.devices()[0].local_hardware_id or 0)['multiprocessor_count']
