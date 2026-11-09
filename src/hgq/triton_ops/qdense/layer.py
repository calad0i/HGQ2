"""Opt-in fused Triton route for QDense / QEinsumDense training, tracing and inference calls.

Switch (default off): hgq.triton_ops.enable(qdense=True), or env HGQ_TRITON_QDENSE=1. Where `applies`, the layer's call
runs as `call`; when off, or when any condition below fails, the layer runs its keras-ops path unchanged.

Fused when all hold (else gated to the keras-ops path):
- backend jax or torch, on a GPU Triton targets (`hgq.triton_ops.on_gpu`); float32 inputs and kernel;
- QDense, or a QEinsumDense with no label repeated within an operand nor summed within one operand only (see `_Canon`;
  the bias may be on any output axes);
- the input quantizer is enabled and is a fixed-point KIF or KBI quantizer with the default bit-width mapper, no
  scaler / qnoise / affine, deterministic rounding; KBI with trainable WRAP is gated in training and tracing (its
  update reads the unrounded inputs before rounding); its bit widths, permuted with the input to (G, M, K), are read
  periodically over each of G, M and K flattened, from their first non-1 axis on (per-tensor, per-channel, per-element
  with the batch axis homogeneous, ...);
- in training and tracing, with bit widths varying along rows, some dx row tile divides the row period (inference
  needs no backward; differentiating it then raises).

Forms: the stateless quantizer's training call (every deterministic round x overflow mode, either
null_xgrad_on_zero_width) and, for KIF trainable WRAP, `round` plus the abs-max of the rounded inputs (computed in the
forward kernel), from which the integer bits update as in `FixedPointQuantizerKIF.call` (`minimal_i_given_xf`, then
`_updated_i`). A tracing call (training 'tracing') runs the training forms, the kernel also flagging whether any
rounded input is negative per bit-width element, and updates as the tracing call does: i <- max(i, constraint(required
i)) from the rounded i, k <- k | the flags. An inference call (training False or None) runs every input quantizer in
its quantize form, trainable WRAP included, with no abs-max and no state update. The kernel and bias quantizers and EBOPs
stay in keras ops; the kernels take the quantized kernel as given.
Dot precision (tl.dot input_precision of the forward, dx and dw GEMMs; fp32 accumulation): `hgq.triton_ops.enable`'s
``precision`` ('tf32' by default) in training calls; in inference and tracing calls 'ieee' (fp32) when
force_fp32_inference (default), else ``precision``. Read when the call runs or is traced.

Fallback: with the switch on, a call that cannot take the kernel route (not on a supported GPU, an unsupported quantizer or
layout) runs on keras ops with a warning naming the reason (once per reason under Python's default warning filter; at
trace time under jax.jit). Only these checks fall back: missing modules raise when the switch is turned on (`check`),
errors raised while building or running the kernels propagate. Every check runs before the layer's quantizers are
called, so a fallback updates their state once.
"""

import warnings
from math import prod

import keras
from keras import ops
from keras.src.backend import in_symbolic_scope

from ... import triton_ops
from ...quantizer.internal.base import DefaultBitwidthMapper
from ...quantizer.internal.fixed_point_quantizer import (
    FixedPointQuantizerKBI,
    FixedPointQuantizerKIF,
    _updated_i,
    minimal_i_given_xf,
)
from .spec import dx_tile_fits, make_spec


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


def _layout(q, c):
    """{label: bit-width size} of the input quantizer ``q`` as the kernels read it: per group (G, M, K) from the first
    non-1 bit-width axis on the operand's sizes (the periodic layout, its 1s expanded), else 1."""
    bw = dict(zip(c.a, (int(d) for d in q.quantizer._k.shape)))
    out = {}
    for grp in (c.G, c.M, c.K):
        lead = _lead([bw[ch] for ch in grp])
        out.update((ch, c.sizes[ch] if n >= lead else 1) for n, ch in enumerate(grp))
    return out


def _quantizer_ok(q, rank, training) -> bool:
    """Whether the kernels run hgq Quantizer ``q`` on an input of ``rank``; a reason they do not is warned."""
    if q.scaler is not None or q.qnoise_factor is not None or q.affine is not None:
        return _gate('scaler/qnoise/affine')
    iq = q.quantizer
    if type(iq) not in (FixedPointQuantizerKIF, FixedPointQuantizerKBI):
        return _gate(f'quantizer {type(iq).__name__}')
    if type(iq.bw_mapper) is not DefaultBitwidthMapper:
        return _gate('bw mapper')
    if iq._stochastic:
        return _gate('stochastic rounding')
    stateful = training is True or training == 'tracing'  # KBI's tracing update also reads the unrounded inputs
    if iq.overflow_mode == 'WRAP' and iq.trainable and stateful and type(iq) is FixedPointQuantizerKBI:
        return _gate('KBI trainable WRAP')
    if len(iq._k.shape) != rank:
        return _gate('bw rank')
    return True


def _plan(q, c, training):
    """(form, (rg, rx, cx), track) of a quantizer `_quantizer_ok` admits: its kernel form, the periods of its bit widths
    over G, M and K (`_layout`), whether its state updates from the kernel's abs-max (trainable WRAP, training or
    tracing)."""
    iq, sq = q.quantizer, q.quantizer.stateless_quantizer
    track = iq.overflow_mode == 'WRAP' and iq.trainable and (training is True or training == 'tracing')
    lay = _layout(q, c)
    periods = tuple(prod([lay[ch] for ch in grp]) for grp in (c.G, c.M, c.K))
    return ('round' if track else 'quantize', sq.round_mode, sq.overflow_mode, bool(sq.null_xgrad_on_zero_width)), periods, track


def _perm(labels, order):
    return [labels.index(ch) for ch in order]


def _kif(q, c, periods, track):
    """The bit widths (k, i, f) of a `_plan`, permuted to G + M + K and expanded to `_layout`, as (rg, rx, cx) tensors."""
    iq, lay, order = q.quantizer, _layout(q, c), c.G + c.M + c.K
    k, i, f = (iq.f,) * 3 if track else iq.kif
    bw = (ops.transpose(ops.cast(ops.broadcast_to(v, iq._k.shape), 'float32'), _perm(c.a, order)) for v in (k, i, f))
    return tuple(ops.reshape(ops.broadcast_to(v, [lay[ch] for ch in order]), periods) for v in bw)


def _own(q, c, v):
    """(rg, rx, cx) abs-max or flags of `_kif`'s layout -> the quantizer's bit-width shape (max over the expanded axes)."""
    lay, order = _layout(q, c), c.G + c.M + c.K
    v = ops.transpose(ops.reshape(v, [lay[ch] for ch in order]), _perm(order, c.a))
    bw = tuple(int(d) for d in q.quantizer._k.shape)
    expanded = [n for n, ch in enumerate(c.a) if bw[n] != lay[ch]]
    return ops.max(v, axis=tuple(expanded), keepdims=True) if expanded else v


class _Canon:
    """A two-operand einsum ``a,b->out`` as the batched GEMM (G, M, K) @ (G, K, N) -> (G, M, N), as torch.einsum (permuted
    into bmm) and jnp.einsum (lax.dot_general's batch and contracting dims) reduce it. Labels: G in input, kernel and
    output; M input and output only; K input and kernel only; N kernel and output only; ``one``: summed within one
    operand (refused: the input's is summed after its quantizer, inside the GEMM). G, M and K keep the input's label
    order, N the output's. The kernels read x as (G, M, K) or (M, G, K) in place (xmg) when its labels are in that block
    order, else x is transposed to G + M + K; y likewise (ymg), else transposed back; q(w) is transposed to G + K + N."""

    def __init__(self, equation, x_shape, k_shape):
        lhs, self.out = equation.split('->')
        self.a, self.b = a, b = lhs.split(',')
        out = self.out
        self.G, self.M, self.K = (''.join(ch for ch in a if ((ch in b), (ch in out)) == io) for io in ((1, 1), (0, 1), (1, 0)))
        self.N = ''.join(ch for ch in out if ch in b and ch not in a)
        self.one = ''.join(ch for ch in a + b if ch not in out and (ch in a) != (ch in b))
        self.diagonal = any(len(set(s)) != len(s) for s in (a, b, out))
        self.sizes = {**dict(zip(b, k_shape)), **dict(zip(a, x_shape))}
        self.consistent = all(self.sizes[ch] == d for ch, d in zip(b, k_shape))
        self.xmg = a != self.G + self.M + self.K and a == self.M + self.G + self.K
        self.ymg = out != self.G + self.M + self.N and out == self.M + self.G + self.N
        self.dims = {g: prod([self.sizes[ch] for ch in getattr(self, g)]) for g in 'GMKN'}

    def x3(self, x):
        """x as the kernels' (G, M, K) or (M, G, K) buffer."""
        G, M, K = (self.dims[g] for g in 'GMK')
        if self.xmg:
            return ops.reshape(x, (M, G, K))
        order = self.G + self.M + self.K
        return ops.reshape(x if self.a == order else ops.transpose(x, _perm(self.a, order)), (G, M, K))

    def w2(self, w):
        """q(w) as the kernels' (G * K, N)."""
        order = self.G + self.K + self.N
        w = w if self.b == order else ops.transpose(w, _perm(self.b, order))
        return ops.reshape(w, (self.dims['G'] * self.dims['K'], self.dims['N']))

    def y(self, y):
        """The kernels' y buffer at the output's shape and label order."""
        if self.ymg or self.out == self.G + self.M + self.N:
            return ops.reshape(y, [self.sizes[ch] for ch in self.out])
        order = self.G + self.M + self.N
        return ops.transpose(ops.reshape(y, [self.sizes[ch] for ch in order]), _perm(order, self.out))

    def bias(self, qb):
        """keras' bias (the output's trailing axes, 1 off its bias axes), which varies along G and N only, as (G * N,)."""
        qb = ops.reshape(qb, [1] * (len(self.out) - len(qb.shape)) + list(qb.shape))
        qb = ops.transpose(qb, _perm(self.out, self.G + self.N + self.M))
        return ops.reshape(ops.broadcast_to(qb, [self.sizes[ch] for ch in self.G + self.N] + [1] * len(self.M)), (-1,))


def _canon(layer, inputs, kind) -> tuple[_Canon, str]:
    """The call's `_Canon` and its bias axes ('' without a bias)."""
    x_shape, k_shape = tuple(int(d) for d in inputs.shape), tuple(int(d) for d in layer._kernel.shape)
    if kind == 'dense':  # x (..., K) @ w (K, N)
        a = ''.join(chr(ord('a') + n) for n in range(len(x_shape)))
        return _Canon(f'{a},{a[-1]}Z->{a[:-1]}Z', x_shape, k_shape), 'Z' * (layer.bias is not None)
    return _Canon(layer._equation, x_shape, k_shape), (layer.bias_axes or '') * (layer.bias is not None)


def _tuning(inputs, training) -> dict:
    """make_spec's ``prec`` (`hgq.triton_ops.dot_precision` of the call) and ``n_sm`` (the device's SM count)."""
    if keras.backend.backend() == 'torch':
        from . import torch_op

        tuning = dict(n_sm=torch_op.n_sm(inputs.device.index))
    else:
        from . import jax_op

        tuning = dict(n_sm=jax_op.n_sm())
    return dict(tuning, prec=triton_ops.dot_precision(training))


def applies(layer, inputs, training) -> bool:
    """Whether the call of QDense / QEinsumDense ``layer`` on ``inputs`` runs as `call`; a reason it does not is warned
    (not in keras' symbolic build, whose placeholder inputs have no device). Every check runs before the layer's
    quantizers are called, so a fallback updates their state once."""
    if not triton_ops.enabled('qdense') or in_symbolic_scope():
        return False
    if not triton_ops.on_gpu(inputs):
        return _gate('not placed on a supported GPU')
    if not layer.enable_iq:
        return _gate('no input quantizer')
    if ops.dtype(inputs) != 'float32' or ops.dtype(layer._kernel) != 'float32':
        return _gate('dtype')
    kind = 'einsum' if isinstance(layer, keras.layers.EinsumDense) else 'dense'
    c = _canon(layer, inputs, kind)[0]
    if c.diagonal:
        return _gate('einsum diagonal (a label repeated in an operand)')
    if c.one:
        return _gate('einsum label summed within one operand')
    if not c.consistent:
        return _gate('shape')
    if c.dims['G'] > 1 << 15:
        return _gate('G beyond the launch grid')
    if not _quantizer_ok(layer.iq, len(c.a), training):
        return False
    if (training is True or training == 'tracing') and not dx_tile_fits(_plan(layer.iq, c, training)[1][1]):
        return _gate('no dx tile divides the row period')
    return True


def call(layer, inputs, kind, training):
    """The fused call of ``layer`` (kind 'dense' or 'einsum') where `applies`, in training, tracing or inference."""
    c, bias_axes = _canon(layer, inputs, kind)
    G, M, K, N = (c.dims[g] for g in 'GMKN')
    qx, (rg, rx, cx), track = _plan(layer.iq, c, training)
    kx, ix, fx = _kif(layer.iq, c, (rg, rx, cx), track)
    # the kernel quantizer in keras ops, then the bias, as in the layers' calls (kq gets the training they give it)
    kernel = layer.qkernel if kind == 'dense' else layer.kq(layer._kernel, training=training)
    qb = (layer.qbias if kind == 'dense' else layer.bq(layer.bias, training=training)) if bias_axes else None
    fused = bool(bias_axes) and set(bias_axes) <= set(c.G + c.N)  # else added to y in keras ops (along M)
    b = c.bias(qb) if fused else ops.zeros((G * N,))
    tracing = training == 'tracing'
    spec = make_spec(
        N, K, qx, rx, cx, bias=fused, amax=track, negative=track and tracing, backward=training is True or tracing,
        G=G, rg=rg, xmg=c.xmg, ymg=c.ymg, **_tuning(inputs, training),
    )  # fmt: skip
    if keras.backend.backend() == 'torch':
        from .torch_op import qdense
    else:
        from .jax_op import qdense
    y, amax, _ = qdense(c.x3(inputs), kx, ix, fx, c.w2(kernel), b, spec)
    if track:  # FixedPointQuantizerKIF.call's update, from the kernel's abs-max (and any-negative flags) of q(x)
        iq = layer.iq.quantizer
        required = minimal_i_given_xf(_own(layer.iq, c, amax[0] if tracing else amax), iq.f, iq.symmetric)
        if tracing:
            iq._k.assign(ops.cast(ops.cast(iq.k, 'bool') | (_own(layer.iq, c, amax[1]) > 0), iq._k.dtype))
            iq._i.assign(_updated_i(iq.i, required, 0.0, iq._i.constraint))
        else:
            iq._i.assign(_updated_i(iq._i, required, iq.i_decay_speed, iq._i.constraint))
    y = c.y(y)
    if bias_axes and not fused:
        y = y + qb
    if layer.activation is not None:
        y = layer.activation(y)
    return y
