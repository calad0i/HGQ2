"""Triton integer-bit replay of trainable-WRAP KIF quantizers: the state the quantizer's training calls leave, from the
old raw integer bits and the attention kernel's per-round partial abs-maxima of each call's rounded input.

One launch per quantizer, grid (calls, heads or 1): program (c, h) reduces call c's partial maxima (B, H, NB, L), its
head h's (all heads for per-tensor-head bit widths), to the bit-width shape as DefaultBitwidthMapper.x_to_bw_absmax does
(the partials are abs-maxima, non-negative, so the mapper's abs-max is their NaN-propagating max) and writes them to a
(calls, n) scratch. The program that takes the last ticket of a zeroed counter then runs, per bit-width element, a
sequential loop over the calls in call order applying `FixedPointQuantizerKIF.call`'s training transition
(device.wrap_integer_bits), each call ``REPEATS`` times in a row. The loop is the quantizer's call sequence, not a closed
form: each step reads the previous step's state.

Layout (replay.spec): ``m0``, ``m1`` (R, B, H, NB, L) float32, round r's maxima of the round's first and (CALLS_PER_ROUND
2) second call, so call c reads round c // CALLS_PER_ROUND of m0 or m1; ``i`` the raw _i, ``f`` the rounded f (bit-width
shape (1, H or 1, 1, L or 1), flattened), ``decay`` i_decay_speed (one element).
"""

import triton
import triton.language as tl

# Imported by name: inductor regenerates a user kernel's source with the jit functions it names, not module attributes.
from ..device import wrap_integer_bits


@triton.jit
def _max(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def replay_i(
    m0_ptr, m1_ptr, i_ptr, f_ptr, decay_ptr, lo, hi, out_ptr, scratch_ptr, count_ptr,
    B: tl.constexpr, H: tl.constexpr, NB: tl.constexpr, L: tl.constexpr, PER_HEAD: tl.constexpr, PER_LANE: tl.constexpr,
    CALLS_PER_ROUND: tl.constexpr, REPEATS: tl.constexpr, BB: tl.constexpr, NP: tl.constexpr, CONSTRAINT: tl.constexpr,
):  # fmt: skip
    """out = the integer bits after the calls, from i. lo, hi: the MinMax bounds (unused for CONSTRAINT 'none'); count:
    one int32, zero at launch; scratch (calls, n) per-call maxima in the bit-width shape."""
    c, h = tl.program_id(0), tl.program_id(1)
    calls, heads = tl.num_programs(0), tl.num_programs(1)
    LB: tl.constexpr = L if PER_LANE else 1
    n = heads * LB
    rnd = (c // CALLS_PER_ROUND).to(tl.int64) * (B * H * NB * L)
    if c % CALLS_PER_ROUND == 0:
        base = m0_ptr + rnd
    else:
        base = m1_ptr + rnd
    cols = tl.arange(0, NB * L)
    rows = tl.arange(0, BB)
    acc = tl.full((NB * L,), float('-inf'), tl.float32)
    for g in range(1 if PER_HEAD else H):
        if PER_HEAD:
            head = h
        else:
            head = g
        for b0 in range(0, B, BB):
            b = b0 + rows
            ptrs = base + ((b.to(tl.int64) * H + head) * (NB * L))[:, None] + cols[None, :]
            x = tl.load(ptrs, mask=(b < B)[:, None], other=float('-inf'))
            acc = _max(acc, tl.reduce(x, 0, _max))
    m = tl.reduce(tl.reshape(acc, (NB, L)), 0, _max)
    if not PER_LANE:
        m = tl.reduce(m, 0, _max, keep_dims=True)
    tl.store(scratch_ptr + c * n + h * LB + tl.arange(0, LB), m)

    tl.debug_barrier()  # every thread's scratch store before the release
    if tl.atomic_add(count_ptr, 1, sem='acq_rel', scope='gpu') == calls * heads - 1:
        e = tl.arange(0, NP)
        mask = e < n
        i = tl.load(i_ptr + e, mask=mask, other=0.0)
        f = tl.load(f_ptr + e, mask=mask, other=0.0)
        decay = tl.load(decay_ptr)
        row = scratch_ptr + e
        for _ in range(calls):
            xr = tl.load(row, mask=mask, other=0.0, cache_modifier='.cg')
            for _r in range(REPEATS):
                i = wrap_integer_bits(i, xr, f, decay, lo, hi, CONSTRAINT)
            row += n
        tl.store(out_ptr + e, i, mask=mask)
