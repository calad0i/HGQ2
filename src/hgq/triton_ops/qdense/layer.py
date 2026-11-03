"""Opt-in fused Triton route for QDense / QEinsumDense training calls.

Switch (default off): hgq.triton_ops.enable(qdense=True), or env HGQ_TRITON_QDENSE=1. Where `applies`, the layer's call
runs as `call`; when off, or when any condition below fails, the layer runs its keras-ops path unchanged.

Fused when all hold (else gated to the keras-ops path):
- backend jax or torch, on a CUDA GPU; training is True (inference and 'tracing' calls keep keras ops); float32 inputs
  and kernel;
- QDense, or a QEinsumDense whose equation is dense-like (input = batch + contracted axes, kernel = contracted + free,
  output = batch + free, in order; bias axes within the free axes);
- the input quantizer is enabled and is a fixed-point KIF or KBI quantizer with the default bit-width mapper, no
  scaler / qnoise / affine, deterministic rounding; KBI with trainable WRAP is gated (its update reads the unrounded
  inputs before rounding); the bit widths are periodic over the flattened (M, K) input (per-tensor, per-channel,
  per-element with the batch axis homogeneous, ...);
- with bit widths varying along rows, some dx row tile divides the row period.

Forms: the stateless quantizer's training call (every deterministic round x overflow mode, either
null_xgrad_on_zero_width) and, for KIF trainable WRAP, `round` plus the abs-max of the rounded inputs (computed in the
forward kernel), from which the integer bits update as in `FixedPointQuantizerKIF.call` (`minimal_i_given_xf`, then
`_updated_i`). The kernel quantizer runs in keras ops (w pre-quantized) unless env HGQ_QDENSE_WLOAD=1 asks for
quantization on load, which applies to non-WRAP fixed-point forms only. EBOPs and the bias quantizer stay in keras ops.
Dot precision: env HGQ_QDENSE_PREC (default 'ieee', FP32 FMA).

Fallback: with the switch on, a call that cannot take the kernel route (no CUDA device, an unsupported quantizer or
layout) runs on keras ops with a warning naming the reason (once per reason under Python's default warning filter; at
trace time under jax.jit). Only these checks fall back: missing modules raise when the switch is turned on (`check`),
errors raised while building or running the kernels propagate. Every check runs before the layer's quantizers are
called, so a fallback updates their state once.
"""

import os
import warnings
from math import prod

import keras
from keras import ops

from ... import triton_ops
from ...quantizer.internal.base import DefaultBitwidthMapper
from ...quantizer.internal.fixed_point_quantizer import (
    FixedPointQuantizerKBI,
    FixedPointQuantizerKIF,
    _updated_i,
    minimal_i_given_xf,
)
from .spec import NONE, dx_tile_fits, make_spec


def _gate(reason) -> bool:
    warnings.warn(f'hgq triton_qdense: running on keras ops ({reason})', stacklevel=2)
    return False


def check():
    """Import the route on the current backend (before any torch.compile tracing); ImportError if it is unavailable."""
    import triton  # noqa: F401
    from quantizers import triton as qt  # noqa: F401

    if keras.backend.backend() == 'torch':
        from . import torch_op  # noqa: F401
    else:
        from . import jax_op  # noqa: F401  (imports jax_triton)


def _lead(bw_dims) -> int:
    """Index of the first bit-width dim that is not 1: the bit widths are periodic over the operand when the rest are full,
    with period the product of the operand's dims from there."""
    return next((n for n, b in enumerate(bw_dims) if b != 1), len(bw_dims))


def _parts(q, shape, n_rows):
    """(bit-width dims, operand dims) of the row axes and of the column axes."""
    bw_shape = tuple(int(d) for d in q.quantizer._k.shape)
    return (bw_shape[:n_rows], shape[:n_rows]), (bw_shape[n_rows:], shape[n_rows:])


def _quantizer_ok(q, shape, n_rows, training) -> bool:
    """Whether the kernels run hgq Quantizer ``q`` on an operand of ``shape`` whose first ``n_rows`` axes flatten to rows
    and the rest to columns; a reason they do not is warned."""
    if q.scaler is not None or q.qnoise_factor is not None or q.affine is not None:
        return _gate('scaler/qnoise/affine')
    iq = q.quantizer
    if type(iq) not in (FixedPointQuantizerKIF, FixedPointQuantizerKBI):
        return _gate(f'quantizer {type(iq).__name__}')
    if type(iq.bw_mapper) is not DefaultBitwidthMapper:
        return _gate('bw mapper')
    if iq._stochastic:
        return _gate('stochastic rounding')
    if iq.overflow_mode == 'WRAP' and iq.trainable and training is True and type(iq) is FixedPointQuantizerKBI:
        return _gate('KBI trainable WRAP')
    if len(iq._k.shape) != len(shape):
        return _gate('bw rank')
    if not all(bw[_lead(bw) :] == dims[_lead(bw) :] for bw, dims in _parts(q, shape, n_rows)):
        return _gate('bw layout not periodic')
    return True


def _plan(q, shape, n_rows, training):
    """(form, rb, cb, track) of a quantizer `_quantizer_ok` admits: its kernel form, the row and column periods of its bit
    widths, whether its integer bits update from the kernel's abs-max (trainable WRAP)."""
    iq, sq = q.quantizer, q.quantizer.stateless_quantizer
    track = iq.overflow_mode == 'WRAP' and iq.trainable and training is True
    rb, cb = (prod(dims[_lead(bw) :]) for bw, dims in _parts(q, shape, n_rows))
    return ('round' if track else 'quantize', sq.round_mode, sq.overflow_mode, bool(sq.null_xgrad_on_zero_width)), rb, cb, track


def _kif(q, rb, cb, track):
    """The bit widths (k, i, f) of a `_plan` as (rb, cb) tensors."""
    iq = q.quantizer
    k, i, f = (iq.f,) * 3 if track else iq.kif
    return tuple(ops.reshape(ops.cast(ops.broadcast_to(v, iq._k.shape), 'float32'), (rb, cb)) for v in (k, i, f))


def _w_load(layer, k_shape, n_kc, training) -> bool:
    """Whether the kernel quantizer runs in the kernels (env HGQ_QDENSE_WLOAD=1; non-WRAP fixed-point forms only)."""
    if os.environ.get('HGQ_QDENSE_WLOAD', '0') != '1' or not _quantizer_ok(layer.kq, k_shape, n_kc, training):
        return False
    form = _plan(layer.kq, k_shape, n_kc, training)[0]
    if form[0] == 'round' or form[2] == 'WRAP':
        return _gate('w on load: WRAP form')
    return True


def _einsum(equation):
    """(input, kernel, output, batch, contracted, free) letters of a two-operand einsum."""
    lhs, out = equation.split('->')
    a, b = lhs.split(',')
    contract = ''.join(ch for ch in a if ch in b and ch not in out)
    batch = ''.join(ch for ch in a if ch not in b)
    free = ''.join(ch for ch in b if ch not in a)
    return a, b, out, batch, contract, free


def _dense_like(layer) -> bool:
    """Input = batch + contracted axes, kernel = contracted + free, output = batch + free, in order; bias axes free."""
    a, b, out, batch, contract, free = _einsum(layer._equation)
    bias_axes = layer.bias_axes if layer.bias is not None else None
    return (
        all(len(set(s)) == len(s) for s in (a, b, out))
        and (a, b, out) == (batch + contract, contract + free, batch + free)
        and not (bias_axes and any(ch not in free for ch in bias_axes))
    )


def _gemm(layer, inputs, kind):
    """(input shape, kernel shape, row axes, contracted kernel axes, output shape) of the layer's call as a GEMM."""
    shape, k_shape = tuple(int(d) for d in inputs.shape), tuple(int(d) for d in layer._kernel.shape)
    if kind == 'dense':
        return shape, k_shape, len(shape) - 1, 1, shape[:-1] + (layer.units,)
    _, _, _, batch, contract, _ = _einsum(layer._equation)
    return shape, k_shape, len(batch), len(contract), shape[: len(batch)] + k_shape[len(contract) :]


def _device_ok(x):
    if keras.backend.backend() == 'torch':
        return bool(getattr(x, 'is_cuda', False))
    import jax

    return jax.default_backend() == 'gpu'


def _tuning(inputs) -> dict:
    """make_spec's ``prec`` (env HGQ_QDENSE_PREC, default 'ieee': FP32 FMA) and ``n_sm`` (the device's SM count)."""
    if keras.backend.backend() == 'torch':
        import torch

        n_sm = torch.cuda.get_device_properties(inputs.device).multi_processor_count
    else:
        from . import jax_op

        n_sm = jax_op.n_sm()
    return dict(prec=os.environ.get('HGQ_QDENSE_PREC', 'ieee'), n_sm=n_sm)


def applies(layer, inputs, training) -> bool:
    """Whether the training call of QDense / QEinsumDense ``layer`` on ``inputs`` runs as `call`; a reason it does not is
    warned. Every check runs before the layer's quantizers are called, so a fallback updates their state once."""
    if not triton_ops.enabled('qdense') or training is not True:
        return False
    if not _device_ok(inputs):
        return _gate('no CUDA device')
    if not layer.enable_iq:
        return _gate('no input quantizer')
    if ops.dtype(inputs) != 'float32' or ops.dtype(layer._kernel) != 'float32':
        return _gate('dtype')
    kind = 'einsum' if isinstance(layer, keras.layers.EinsumDense) else 'dense'
    if kind == 'einsum' and not _dense_like(layer):
        return _gate('einsum not dense-like')
    shape, k_shape, n_rows, n_kc, _ = _gemm(layer, inputs, kind)
    if prod(shape[n_rows:]) != prod(k_shape[:n_kc]):
        return _gate('shape')
    if not _quantizer_ok(layer.iq, shape, n_rows, training):
        return False
    if not dx_tile_fits(_plan(layer.iq, shape, n_rows, training)[1]):
        return _gate('no dx tile divides the row period')
    return True


def call(layer, inputs, kind):
    """The fused training call of ``layer`` (kind 'dense' or 'einsum') where `applies`."""
    shape, k_shape, n_rows, n_kc, out_shape = _gemm(layer, inputs, kind)
    M, K, N = prod(shape[:n_rows]), prod(shape[n_rows:]), prod(k_shape[n_kc:])
    qx, rx, cx, track = _plan(layer.iq, shape, n_rows, True)
    kx, ix, fx = _kif(layer.iq, rx, cx, track)
    wq_train = None if kind == 'dense' else True  # QDense.qkernel passes no training flag; QEinsumDense does
    if _w_load(layer, k_shape, n_kc, wq_train):
        qw, rw, cw, _ = _plan(layer.kq, k_shape, n_kc, wq_train)
        (kw, iw, fw), kernel = _kif(layer.kq, rw, cw, False), layer._kernel
    else:  # the kernel quantizer in keras ops, then the bias, as in the layers' calls
        qw, rw, cw, (kw, iw, fw) = NONE, 1, 1, (ops.zeros((1, 1)),) * 3
        kernel = layer.qkernel if kind == 'dense' else layer.kq(layer._kernel, training=True)
    if layer.bias is not None:
        qb = layer.qbias if kind == 'dense' else layer.bq(layer.bias, training=True)
        b = ops.reshape(qb if kind == 'dense' else ops.broadcast_to(qb, k_shape[n_kc:]), (N,))
    else:
        b = ops.zeros((N,))
    spec = make_spec(M, N, K, qx, rx, cx, qw, rw, cw, bias=layer.bias is not None, amax=track, **_tuning(inputs))
    if keras.backend.backend() == 'torch':
        from .torch_op import qdense
    else:
        from .jax_op import qdense
    y, amax, _ = qdense(ops.reshape(inputs, (M, K)), kx, ix, fx, ops.reshape(kernel, (K, N)), kw, iw, fw, b, spec)
    if track:  # FixedPointQuantizerKIF.call's training update, from the kernel's abs-max of the rounded inputs
        iq = layer.iq.quantizer
        xr = ops.reshape(amax, tuple(int(d) for d in iq._k.shape))
        iq._i.assign(_updated_i(iq._i, minimal_i_given_xf(xr, iq.f, iq.symmetric), iq.i_decay_speed, iq._i.constraint))
    y = ops.reshape(y, out_shape)
    if layer.activation is not None:
        y = layer.activation(y)
    return y
