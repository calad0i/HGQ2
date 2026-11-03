"""hgq's own Triton device functions, next to `quantizers.triton`'s quantizers: the stable softmax exp and inverse
tables as QUnaryFunctionLUT computes them in training (input quantizer, QSoftmax's exp or inverse function, output
quantizer) with their vjps, and the integer-bit transition of a trainable-WRAP FixedPointQuantizerKIF training call
(`minimal_i_given_xf`, then `_updated_i`). libdevice's exp and log2 are the IEEE functions keras ops call; tl.exp and
tl.log2 approximate."""

import triton
import triton.language as tl
from triton.language.extra import libdevice

# Imported by name: inductor regenerates a user kernel's source with the jit functions it names, not module attributes.
from .qdense.kernels import _quantize, _quantize_vjp


@triton.jit
def exp_table(
    x, s, ki, ii, fi, ko, io, fo,
    RND_I: tl.constexpr, OVF_I: tl.constexpr, NULL_I: tl.constexpr,
    RND_O: tl.constexpr, OVF_O: tl.constexpr, NULL_O: tl.constexpr,
):  # fmt: skip
    """oq(exp(-iq(x) * s)), iq and oq stateless quantizer calls."""
    xq = _quantize(x, ki, ii, fi, 'quantize', RND_I, OVF_I, NULL_I)
    return _quantize(libdevice.exp(-xq * s), ko, io, fo, 'quantize', RND_O, OVF_O, NULL_O)


@triton.jit
def exp_table_vjp(
    x, s, ki, ii, fi, ko, io, fo, g,
    RND_I: tl.constexpr, OVF_I: tl.constexpr, NULL_I: tl.constexpr,
    RND_O: tl.constexpr, OVF_O: tl.constexpr, NULL_O: tl.constexpr,
):  # fmt: skip
    """(dx, d ii, d fi, d io, d fo) of `exp_table` for the cotangent g, at the broadcast shape."""
    xq = _quantize(x, ki, ii, fi, 'quantize', RND_I, OVF_I, NULL_I)
    y = libdevice.exp(-xq * s)
    gy, dio, dfo = _quantize_vjp(y, ko, io, fo, g, 'quantize', RND_O, OVF_O, NULL_O)
    dx, dii, dfi = _quantize_vjp(x, ki, ii, fi, -(gy * y * s), 'quantize', RND_I, OVF_I, NULL_I)
    return dx, dii, dfi, dio, dfo


@triton.jit
def inv_table(
    x, eps, ki, ii, fi, ko, io, fo,
    Q_I: tl.constexpr, RND_I: tl.constexpr, OVF_I: tl.constexpr, NULL_I: tl.constexpr,
    RND_O: tl.constexpr, OVF_O: tl.constexpr, NULL_O: tl.constexpr,
):  # fmt: skip
    """oq(1 / (iq(x) + eps)), oq a stateless quantizer call, iq one ('quantize') or a trainable-WRAP rounding (Q_I
    'round'); IEEE division, as keras ops divide."""
    xq = _quantize(x, ki, ii, fi, Q_I, RND_I, OVF_I, NULL_I)
    return _quantize(tl.div_rn(1.0, xq + eps), ko, io, fo, 'quantize', RND_O, OVF_O, NULL_O)


@triton.jit
def inv_table_vjp(
    x, eps, ki, ii, fi, ko, io, fo, g,
    Q_I: tl.constexpr, RND_I: tl.constexpr, OVF_I: tl.constexpr, NULL_I: tl.constexpr,
    RND_O: tl.constexpr, OVF_O: tl.constexpr, NULL_O: tl.constexpr,
):  # fmt: skip
    """(dx, d ii, d fi, d io, d fo) of `inv_table` for the cotangent g, at the broadcast shape."""
    xq = _quantize(x, ki, ii, fi, Q_I, RND_I, OVF_I, NULL_I)
    y = tl.div_rn(1.0, xq + eps)
    gy, dio, dfo = _quantize_vjp(y, ko, io, fo, g, 'quantize', RND_O, OVF_O, NULL_O)
    dx, dii, dfi = _quantize_vjp(x, ki, ii, fi, -(gy * y * y), Q_I, RND_I, OVF_I, NULL_I)
    return dx, dii, dfi, dio, dfo


@triton.jit
def wrap_integer_bits(i, xr, f, decay, lo, hi, CONSTRAINT: tl.constexpr):
    """`_updated_i(i, minimal_i_given_xf(xr, f), decay, constraint)` for a call's rounded-input abs-maximum xr >= 0 (where
    the symmetric and asymmetric minimal_i agree) and the rounded f; CONSTRAINT 'none' or 'minmax' (MinMax(lo, hi))."""
    required = tl.ceil(libdevice.log2(xr + tl.exp2(-f)))  # 2 ** -f: exact for integral f
    if CONSTRAINT == 'minmax':
        required = tl.minimum(tl.maximum(required, lo, propagate_nan=tl.PropagateNan.ALL), hi, propagate_nan=tl.PropagateNan.ALL)
    return tl.maximum(i - decay, required, propagate_nan=tl.PropagateNan.ALL)
