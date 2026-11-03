"""hgq2's fused quantized dense as torch custom ops: torch.library.triton_op + register_autograd, so dynamo/inductor
trace it (fullgraph) and CUDA graphs can capture it.

qdense(x (M, K), kx, ix, fx (rx, cx), w (K, N), kw, iw, fw (rw, cw), b (N,), spec) -> (y (M, N), amax, qx)
- spec: Spec.dumps() (hgq.triton_ops.qdense.spec); amax: the trainable-WRAP abs-max of q(x) at (rx, cx) (zeros when
  spec.amax is off); qx: q(x) as computed in the forward GEMM when spec.store_q (checks), else a 1-element dummy.
- Gradients: x, ix, fx, w (and iw, fw when the weight is quantized on load), b; k has none (as in hgq).
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
    """The Spec with its stages fitted to the shared memory of CUDA device ``dev`` (qdense.kernels.fit_device)."""
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(index)
    return KS.fit_device(s, index, major * 10 + minor)


def fold_cols(p, c):
    """(..., K) column partials -> (..., c): the periodic column layout's sum (c divides K)."""
    return p.reshape(*p.shape[:-1], p.shape[-1] // c, c).sum(-2)


def _fwd_impl(s, x, kx, ix, fx, w, kw, iw, fw, b):
    dev = x.device
    s = _fitted(s, dev)
    y = torch.empty((s.M, s.N), dtype=torch.float32, device=dev)
    ashape, adtype = s.amax_buffer()
    amax = torch.zeros(ashape, dtype=getattr(torch, adtype), device=dev)
    qx = torch.empty((s.M, s.K) if s.store_q else (1,), dtype=torch.float32, device=dev)
    c = s.cfg('fwd')
    consts = s.fwd_consts()
    _launch(KS.qdense_fwd, s.fwd_grid(), (x, kx, ix, fx, w, kw, iw, fw, b, y, amax, qx), consts, c)
    if s.amax_mode == 1:
        amax = _fold_max(amax, s.cx)
    elif s.amax_mode == 2:
        amax = amax.view(torch.float32).clone()  # a fresh output, not a view
    else:
        amax = torch.zeros((s.rx, s.cx), dtype=torch.float32, device=dev)
    return y, amax, qx


def _fold_max(p, c):
    """(n_m, K) per-row-block abs-max partials -> (1, c), NaN-propagating like XLA's max."""
    m = p.reshape(-1, p.shape[-1] // c, c).flatten(0, 1)
    return torch.amax(m, 0, keepdim=True)


@triton_op('hgqtri::qdense_fwd', mutates_args=())
def qdense_fwd(
    x: torch.Tensor, kx: torch.Tensor, ix: torch.Tensor, fx: torch.Tensor, w: torch.Tensor, kw: torch.Tensor,
    iw: torch.Tensor, fw: torch.Tensor, b: torch.Tensor, spec: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    return _fwd_impl(Spec.loads(spec), x, kx, ix, fx, w, kw, iw, fw, b)


@triton_op('hgqtri::qdense_bwd', mutates_args=())
def qdense_bwd(
    g: torch.Tensor, x: torch.Tensor, kx: torch.Tensor, ix: torch.Tensor, fx: torch.Tensor, w: torch.Tensor,
    kw: torch.Tensor, iw: torch.Tensor, fw: torch.Tensor, spec: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    dev = x.device
    s = _fitted(Spec.loads(spec), dev)
    f32 = dict(dtype=torch.float32, device=dev)
    # dx = vjp_x(g @ q(w)^T), with the i / f gradient partials
    dx = torch.empty((s.M, s.K), **f32)
    pi = torch.empty(s.dx_partials(), **f32)
    pf = torch.empty(s.dx_partials(), **f32)
    c = s.cfg('dx')
    consts = s.dx_consts()
    _launch(KS.qdense_dx, s.dx_grid(), (g, w, kw, iw, fw, x, kx, ix, fx, dx, pi, pf), consts, c)
    if s.qx[0] == 'none':
        dix = dfx = torch.zeros((s.rx, s.cx), **f32)
    else:
        dix = fold_cols(pi.sum(0), s.cx).reshape(s.rx, s.cx)
        dfx = fold_cols(pf.sum(0), s.cx).reshape(s.rx, s.cx)
    # dwq = q(x)^T @ g split over M, and db
    c = s.cfg('dw')
    pw = torch.empty((c['SPLIT'], s.K, s.N), **f32)
    pb = torch.empty((c['SPLIT'], s.N) if s.bias else (1,), **f32)
    consts = s.dw_consts()
    _launch(KS.qdense_dw, s.dw_grid(), (x, kx, ix, fx, g, pw, pb), consts, c)
    dwq = pw.sum(0)
    db = pb.sum(0) if s.bias else torch.zeros((s.N,), **f32)
    if s.qw[0] == 'none':
        return dx, dix, dfx, dwq, torch.zeros((s.rw, s.cw), **f32), torch.zeros((s.rw, s.cw), **f32), db
    # weight quantized on load: its vjp elementwise on (K, N), the i / f terms folded to (rw, cw)
    dw = torch.empty((s.K, s.N), **f32)
    ti = torch.empty((s.K, s.N), **f32)
    tf = torch.empty((s.K, s.N), **f32)
    c = s.cfg('vjp2d')
    consts = s.vjp2d_consts()
    _launch(KS.qvjp2d, s.vjp2d_grid(), (w, kw, iw, fw, dwq, dw, ti, tf), consts, c)
    diw = ti.reshape(s.K // s.rw, s.rw, s.N // s.cw, s.cw).sum((0, 2))
    dfw = tf.reshape(s.K // s.rw, s.rw, s.N // s.cw, s.cw).sum((0, 2))
    return dx, dix, dfx, dw, diw, dfw, db


def _setup(ctx, inputs, output):
    x, kx, ix, fx, w, kw, iw, fw, b, spec = inputs
    ctx.save_for_backward(x, kx, ix, fx, w, kw, iw, fw)
    ctx.spec = spec


def _backward(ctx, gy, gamax, gqx):
    s = Spec.loads(ctx.spec)
    dx, dix, dfx, dw, diw, dfw, db = qdense_bwd(gy.contiguous(), *ctx.saved_tensors, ctx.spec)
    wq_grads = (diw, dfw) if s.qw[0] != 'none' else (None, None)
    return dx, None, dix, dfx, dw, None, *wq_grads, (db if s.bias else None), None


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


def qdense(x, kx, ix, fx, w, kw, iw, fw, b, spec):
    """spec: a Spec or its dumps() string."""
    f = [t.contiguous() for t in (x, kx, ix, fx, w, kw, iw, fw, b)]
    return qdense_fwd(*f, _spec_str(spec))
