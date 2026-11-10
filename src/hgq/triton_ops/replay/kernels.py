"""Triton integer-bit replay of trainable-WRAP KIF quantizers: the state the quantizer's training calls leave, from the
old raw integer bits and the attention kernel's per-call abs-maxima of the call's rounded input.

One launch per quantizer, one program: per bit-width element, a sequential loop over the calls in call order applying
`FixedPointQuantizerKIF.call`'s training transition (device.wrap_integer_bits), each call ``REPEATS`` times in a row;
TRACING, its tracing transition, which reads the rounded i (decay 0 passed).
The loop is the quantizer's call sequence, not a closed form: each step reads the previous step's state. Each call's
maxima (H, L) are reduced to the bit-width shape as DefaultBitwidthMapper.x_to_bw_absmax does (they are abs-maxima,
non-negative, so the mapper's abs-max is their NaN-propagating max); elements of a coarser bit width are replicated
over the (H, L) tile and computed alike.

Layout (replay.spec): ``m`` (calls, H, L) int32, the float32 bit patterns of the attention kernel's maxima buffers;
``i`` the raw _i, ``f`` the rounded f (bit-width shape (1, H or 1, 1, L or 1), flattened), ``decay`` i_decay_speed
(one element).
"""

import triton
import triton.language as tl
from triton.language.extra import libdevice

# Imported by name: inductor regenerates a user kernel's source with the jit functions it names, not module attributes.
from ..device import wrap_integer_bits


@triton.jit
def _max(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def replay_i(
    m_ptr, i_ptr, f_ptr, decay_ptr, lo, hi, out_ptr,
    CALLS: tl.constexpr, H: tl.constexpr, L: tl.constexpr, HP: tl.constexpr, LP: tl.constexpr, PER_HEAD: tl.constexpr,
    PER_LANE: tl.constexpr, REPEATS: tl.constexpr, CONSTRAINT: tl.constexpr, TRACING: tl.constexpr,
):  # fmt: skip
    """out = the integer bits after the calls, from i. lo, hi: the MinMax bounds (unused for CONSTRAINT 'none'); HP, LP:
    H, L rounded up to powers of two."""
    heads = tl.arange(0, HP)[:, None]
    lanes = tl.arange(0, LP)[None, :]
    mask = (heads < H) & (lanes < L)
    e = (heads if PER_HEAD else heads * 0) * (L if PER_LANE else 1) + (lanes if PER_LANE else lanes * 0)
    i = tl.load(i_ptr + e, mask=mask, other=0.0)
    f = tl.load(f_ptr + e, mask=mask, other=0.0)
    decay = tl.load(decay_ptr)
    row = m_ptr + heads * L + lanes
    for _ in range(CALLS):
        x = tl.load(row, mask=mask, other=0).to(tl.float32, bitcast=True)
        if not PER_LANE:
            x = tl.broadcast_to(tl.reduce(x, 1, _max, keep_dims=True), (HP, LP))
        if not PER_HEAD:
            x = tl.broadcast_to(tl.reduce(x, 0, _max, keep_dims=True), (HP, LP))
        for _r in range(REPEATS):
            if TRACING:
                i = libdevice.rint(i)  # round_conv: half to even
            i = wrap_integer_bits(i, x, f, decay, lo, hi, CONSTRAINT)
        row += H * L
    tl.store(out_ptr + e, i, mask=mask)
