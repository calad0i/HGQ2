"""hgq2's fused quantized dense as torch custom ops: torch.library.triton_op + register_autograd, so dynamo/inductor
trace it (fullgraph) and CUDA graphs can capture it.

qdense(x (G, M, K), kx, ix, fx (rg, rx, cx), w (G * K, N), b (G * N,), spec) -> (y (G, M, N), amax, qx), a batched GEMM;
x (M, G, K) when spec.xmg, y (M, G, N) when spec.ymg.
- spec: Spec.dumps() (hgq.triton_ops.qdense.spec), free of M (x's rows: symbolic under a dynamic batch); amax: the trainable-WRAP abs-max of q(x) at (rg, rx, cx) (zeros when
  spec.amax is off; spec.negative, (2, rg, rx, cx): then whether any q(x) < 0, 0 / 1); qx: q(x) (G, M, K) as computed in the forward GEMM when spec.store_q (checks), else a 1-element dummy.
- Gradients: x, ix, fx, w, b; k has none (as in hgq).
"""

import dataclasses

import torch
from torch.library import triton_op, wrap_triton

from . import kernels as KS
from .spec import Spec


def _modes_lower(consts):
    """``consts`` with the string constexprs (quantizer modes, forms) lower-cased, for wrap_triton."""
    return {k: v.lower() if isinstance(v, str) else v for k, v in consts.items()}


def _launch(kernel, grid, args, consts, cfg):
    wrap_triton(kernel)[grid](*args, **_modes_lower(consts), num_warps=cfg['num_warps'], num_stages=cfg['num_stages'])


def _fitted(s, dev):
    """The Spec with its stages fitted to the shared memory of device ``dev`` (qdense.kernels.fit_device)."""
    return KS.fit_device(s, dev.index)


@torch.compiler.assume_constant_result  # dynamo skips Triton's driver
def n_sm(index) -> int:
    """The multiprocessor count of device ``index`` (qdense.kernels.device_properties)."""
    return KS.device_properties(index)['multiprocessor_count']


def fold_cols(p, c):
    """(..., K) column partials -> (..., c): the periodic column layout's sum (c divides K)."""
    return p.reshape(*p.shape[:-1], p.shape[-1] // c, c).sum(-2)


def _fold_bw(s, p):
    """(G, programs, [rx,] K) dx partials -> (rg, rx, cx): summed over the programs, the column periods and g // rg."""
    return fold_cols(p.sum(1), s.cx).reshape(s.G // s.rg, *s.bw_shape).sum(0)


def _fwd_impl(s, x, kx, ix, fx, w, b):
    dev, M = x.device, x.shape[0 if s.xmg else 1]
    s = _fitted(s, dev)
    y = torch.empty((M, s.G, s.N) if s.ymg else (s.G, M, s.N), dtype=torch.float32, device=dev)
    ashape, adtype = s.amax_buffer(M)
    amax = torch.zeros(ashape, dtype=getattr(torch, adtype), device=dev)
    qx = torch.empty((s.G, M, s.K) if s.store_q else (1,), dtype=torch.float32, device=dev)
    c = s.cfg('fwd')
    consts = s.fwd_consts(M)
    _launch(KS.qdense_fwd, s.fwd_grid(M), (x, kx, ix, fx, w, b, amax, y, qx), consts, c)
    if s.amax_mode == 1:
        amax = _fold_max(s, amax)
    elif s.amax_mode == 2:
        amax = amax.view(torch.float32).clone()  # a fresh output, not a view
    else:
        amax = torch.zeros(s.bw_shape, dtype=torch.float32, device=dev)
    return y, amax.reshape(s.amax_shape), qx


def _fold_max(s, p):
    """(planes, G * n_m, K) per-row-block partials -> (planes, rg, cx), NaN-propagating like XLA's max."""
    return torch.amax(p.reshape(p.shape[0], s.G // s.rg, s.rg, -1, s.cx), (1, 3))


@triton_op('hgqtri::qdense_fwd', mutates_args=())
def qdense_fwd(
    x: torch.Tensor, kx: torch.Tensor, ix: torch.Tensor, fx: torch.Tensor, w: torch.Tensor, b: torch.Tensor, spec: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    return _fwd_impl(Spec.loads(spec), x, kx, ix, fx, w, b)


@triton_op('hgqtri::qdense_bwd', mutates_args=())
def qdense_bwd(
    g: torch.Tensor, x: torch.Tensor, kx: torch.Tensor, ix: torch.Tensor, fx: torch.Tensor, w: torch.Tensor, spec: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    s = Spec.loads(spec)
    dev, M = x.device, x.shape[0 if s.xmg else 1]
    s = _fitted(s, dev)
    f32 = dict(dtype=torch.float32, device=dev)
    # dx = vjp_x(g @ w^T), with the i / f gradient partials
    dx = torch.empty_like(x)
    pi = torch.empty(s.dx_partials(M), **f32)
    pf = torch.empty(s.dx_partials(M), **f32)
    c = s.cfg('dx')
    consts = s.dx_consts(M)
    _launch(KS.qdense_dx, s.dx_grid(M), (g, w, x, kx, ix, fx, dx, pi, pf), consts, c)
    if s.qx[0] == 'none':
        dix = dfx = torch.zeros(s.bw_shape, **f32)
    else:
        dix, dfx = _fold_bw(s, pi), _fold_bw(s, pf)
    # dw = q(x)^T @ g split over M, and db
    c, split = s.cfg('dw'), s.dw_split(M)
    pw = torch.empty((s.G, split, s.K, s.N), **f32)
    pb = torch.empty((s.G, split, s.N) if s.bias else (1,), **f32)
    consts = s.dw_consts(M)
    _launch(KS.qdense_dw, s.dw_grid(M), (x, kx, ix, fx, g, pw, pb), consts, c)
    dw = pw.sum(1).reshape(s.G * s.K, s.N)
    db = pb.sum(1).reshape(-1) if s.bias else torch.zeros((s.G * s.N,), **f32)
    return dx, dix, dfx, dw, db


def _setup(ctx, inputs, output):
    x, kx, ix, fx, w, b, spec = inputs
    ctx.save_for_backward(x, kx, ix, fx, w)
    ctx.spec = spec


def _backward(ctx, gy, gamax, gqx):
    s = Spec.loads(ctx.spec)
    if not s.dx_fits:
        raise ValueError(f'no dx row tile divides the row period {s.rx}: this inference Spec has no backward')
    dx, dix, dfx, dw, db = qdense_bwd(gy.contiguous(), *ctx.saved_tensors, ctx.spec)
    return dx, None, dix, dfx, dw, (db if s.bias else None), None


qdense_fwd.register_autograd(_backward, setup_context=_setup)


_FIELDS = tuple(f.name for f in dataclasses.fields(Spec))


@torch._dynamo.assume_constant_result
def _dumps(*values):
    """Spec(*values).dumps(), a trace-time constant for dynamo (json / dataclasses are not traceable). It takes the field
    values, not the Spec: a Spec constructed inside the trace reaches a constant-result function without its fields
    (dynamo keeps them as tracked side effects), while reading its fields is traced."""
    return Spec(*values).dumps()


def _spec_str(spec):
    """The op's string argument."""
    return spec if isinstance(spec, str) else _dumps(*(getattr(spec, f) for f in _FIELDS))


def qdense(x, kx, ix, fx, w, b, spec):
    """spec: a Spec or its dumps() string."""
    f = [t.contiguous() for t in (x, kx, ix, fx, w, b)]
    return qdense_fwd(*f, _spec_str(spec))
