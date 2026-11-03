"""Triton forward and backward kernels of hgq2's quantized attention (QMultiHeadAttention, softmax='1pass' or '2pass').

One program per (batch b, head h, block of TB queries). Round j (key j) of the recurrence, for the block's queries:

1. s = Qs(where(valid, sum(q * k_j), 0)), the score quantizer's training call (stateless, or a trainable-WRAP rounding)
2. m' = where(valid, max(m, s), m)
3. r = E(m' - m); p = where(valid, E(where(valid, m' - s, 0)), 0), E the exp table (device.exp_table)
4. a = Qa(r * o); b = Qa(p * where(valid, v_j, 0)); c = Ql(r * l), Qa and Ql trainable-WRAP roundings (in an
   inference call, their quantize)
5. o' = where(valid, a + b, o); l' = where(valid, c + p, l)

m starts at the quantized score of each query's first kept key (`first`), computed from that key directly
(`_first_score`): the score quantizer is stateless, so the other rounds of a scan for it have no effect. The key loops
run to the CTA-uniform bound n (one past the last key kept by any query of the block); per-query validity is the mask.
Every round takes the block's maxima of |a|, |b| (per value lane) and |c| into the integer-bit replay's buffers of Qa
and Ql (`_amax`); rounds from n on keep o and l, and their maxima are the quiet round's: |Qa(E(0) o)|, |Qa(0)|,
|Ql(E(0) l)|. Each program also takes its maximum of |s| over the block's queries and keys (rounds from n on see Qs(0)
= 0 when Qs rounds), for the replay of a rounding Qs.

2-pass (TWO_PASS): the first pass is the rounds above without o, with c = Ql(where(valid, r * l, 0)). The second runs
the rounds again from the final (m, l): P = Qp(where(valid, E(where(valid, m - s, 0)) * I(l), 0)), I the inverse table
(device.inv_table, its input quantizer Qi stateless or a trainable-WRAP rounding) and Qp the probability quantizer
(likewise), then a = Qa(where(valid, o, 0)), b = Qa(P v_j), o' = where(valid, a + b, o); the maxima of |a|, |b| and |P|
feed the replay, and rounds from n on see 0 in every call; so does each program's maximum of |Qi(l)|, which every round
of the second pass evaluates. o is the normalized context.

Quantizers are `quantizers.triton`'s (through qdense.kernels' `_quantize`), the exp table `device`'s; widths are
tensor operands, modes constexprs: (round, overflow, null_xgrad_on_zero_width) of the score, exp-table, context and
weight quantizers, the forms ('quantize' or 'round') of Qs, Qa, Ql, Qi and Qp. TRAINING off (an inference call: every
form 'quantize'), attn_fwd writes o and l only.

TRAINING (a training or tracing call) also flags, per head (and value lane for Qa), whether any output of Qa, Ql, Qs,
Qp and Qi was negative over the calls, the tracing call's k update: each program ORs its own and atomically maxes them
into the int32 ``negative`` buffer (`_any`).

The maxima buffers are int32 views of float32, zero at launch, per (call, head[, value lane]); each program atomically
maxes its block's maxima in: the values are |x| >= 0 (or NaN), whose int32 bit patterns order as the floats do (NaN
above inf), so the result is the max over programs regardless of their order. Arithmetic is elementwise FP32 (no tl.dot).
"""

import triton
import triton.language as tl
from quantizers.triton import fixed_round, fixed_round_vjp

# Imported by name: inductor regenerates a user kernel's source with the jit functions it names, not module attributes.
from ..device import exp_table, exp_table_vjp, inv_table, inv_table_vjp
from ..qdense.kernels import _quantize, _quantize_vjp


@triton.jit
def _first_score(
    q, k_base, sks, valid_base, sms, first, n, ks, i_s, fs,
    Q_S: tl.constexpr, RND_S: tl.constexpr, OVF_S: tl.constexpr, NULL_S: tl.constexpr,
):  # fmt: skip
    """The running maximum's start: each query's quantized score of its first kept key, gathered directly, equal to a
    scan of the rounds [0, n) keeping round `first`'s score. A row without a kept key (first 0) gets its masked score
    Qs(0) as the scan's round 0 gives, and every row 0 when n is 0 (the scan visits no round)."""
    live = (first >= 0) & (first < n)
    valid = tl.load(valid_base + first * sms, mask=live, other=0.0) != 0.0
    key = tl.load(k_base[None, :] + first[:, None] * sks, mask=live[:, None], other=0.0)
    return tl.where(live, _quantize(tl.where(valid, tl.sum(q * key, axis=1), 0.0), ks, i_s, fs, Q_S, RND_S, OVF_S, NULL_S), 0.0)


@triton.jit
def _round_scores(
    q, k_base, sks, valid_base, sms, j, ks, i_s, fs,
    Q_S: tl.constexpr, RND_S: tl.constexpr, OVF_S: tl.constexpr, NULL_S: tl.constexpr,
):  # fmt: skip
    """Round j's validity, score quantizer input (masked raw scores) and quantized scores."""
    valid = tl.load(valid_base + j * sms) != 0.0
    x = tl.where(valid, tl.sum(q * tl.load(k_base + j * sks)[None, :], axis=1), 0.0)
    return valid, x, _quantize(x, ks, i_s, fs, Q_S, RND_S, OVF_S, NULL_S)


@triton.jit
def _inverse(
    l, qp, eps,
    Q_I: tl.constexpr, RND_I: tl.constexpr, OVF_I: tl.constexpr, NULL_I: tl.constexpr,
    RND_O: tl.constexpr, OVF_O: tl.constexpr, NULL_O: tl.constexpr,
):  # fmt: skip
    """The inverse table of the weights l, its bit widths at columns 12-17 of the head's qp row."""
    ki, ii, fi = tl.load(qp + 12), tl.load(qp + 13), tl.load(qp + 14)
    ko, io, fo = tl.load(qp + 15), tl.load(qp + 16), tl.load(qp + 17)
    return inv_table(l, eps, ki, ii, fi, ko, io, fo, Q_I, RND_I, OVF_I, NULL_I, RND_O, OVF_O, NULL_O)


@triton.jit
def _inverse_vjp(
    l, qp, eps, g,
    Q_I: tl.constexpr, RND_I: tl.constexpr, OVF_I: tl.constexpr, NULL_I: tl.constexpr,
    RND_O: tl.constexpr, OVF_O: tl.constexpr, NULL_O: tl.constexpr,
):  # fmt: skip
    """`_inverse`'s vjp: (dl, d ii, d fi, d io, d fo)."""
    ki, ii, fi = tl.load(qp + 12), tl.load(qp + 13), tl.load(qp + 14)
    ko, io, fo = tl.load(qp + 15), tl.load(qp + 16), tl.load(qp + 17)
    return inv_table_vjp(l, eps, ki, ii, fi, ko, io, fo, g, Q_I, RND_I, OVF_I, NULL_I, RND_O, OVF_O, NULL_O)


@triton.jit
def _amax(ptr, x):
    """Atomic max of |x|'s maxima ``x`` (>= 0 or NaN) into the int32 view ``ptr``."""
    tl.atomic_max(ptr, x.to(tl.int32, bitcast=True), sem='relaxed')


@triton.jit
def _any(ptr, negative):
    """Atomic max of whether any row of the boolean ``negative`` is set into the int32 ``ptr``."""
    tl.atomic_max(ptr, tl.max(negative.to(tl.int32), axis=0), sem='relaxed')


@triton.jit
def attn_fwd(
    q_ptr,  # (B, T, H, DK) projected queries
    k_ptr,  # (B, S, H, DK) projected keys
    v_ptr,  # (B, S, H, DV) projected values
    valid_ptr,  # (S, B, T) float, nonzero where query t keeps key j
    first_ptr,  # (B, T) int32, each query's first kept key (0 for a row without one)
    n_ptr,  # (B, NB, *) int32, column 0: keys to visit per query block
    qp_ptr,  # (H, *) per-head (k, i, f) of the score, exp-table input, exp-table output and weight quantizers;
    # 2-pass, then of the inverse table's input and output quantizers and the probability quantizer (columns 12-20)
    akif_ptr,  # (3, H, DV) the context quantizer's (k, i, f) per value lane
    scaler_ptr,  # (1,) the exp table's input scaler
    context_ptr,  # in-out int32 (2 S, H, DV) per call max |a| (call 2 j) and |b| (call 2 j + 1) of round j (training)
    held_ptr,  # in-out int32 (S, H) per round max |c|
    served_ptr,  # in-out int32 (S, H) 2-pass: per round max |P| (else unused)
    rounded_ptr,  # in-out int32 (2, H) max |s| over the queries and keys; 2-pass, max |Qi(l)| over the queries
    negative_ptr,  # in-out int32 (5, H, DV) any Qa output negative (per lane), then (lane 0) any Ql, Qs, Qp, Qi one
    o_ptr,  # out (after every input, as jax_triton passes outputs) (B, T, H, DV) context accumulator o
    l_ptr,  # out (B, T, H) weight accumulator l
    B: tl.constexpr,
    T: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    NB: tl.constexpr,  # query blocks, T / TB
    W: tl.constexpr,  # columns of qp
    NW: tl.constexpr,  # columns of n
    TB: tl.constexpr,
    RND_S: tl.constexpr,
    OVF_S: tl.constexpr,
    NULL_S: tl.constexpr,
    RND_EI: tl.constexpr,
    OVF_EI: tl.constexpr,
    NULL_EI: tl.constexpr,
    RND_EO: tl.constexpr,
    OVF_EO: tl.constexpr,
    NULL_EO: tl.constexpr,
    RND_A: tl.constexpr,
    RND_L: tl.constexpr,
    Q_S: tl.constexpr,
    TWO_PASS: tl.constexpr,
    RND_II: tl.constexpr,
    OVF_II: tl.constexpr,
    NULL_II: tl.constexpr,
    Q_II: tl.constexpr,
    RND_IO: tl.constexpr,
    OVF_IO: tl.constexpr,
    NULL_IO: tl.constexpr,
    Q_P: tl.constexpr,
    RND_P: tl.constexpr,
    OVF_P: tl.constexpr,
    NULL_P: tl.constexpr,
    EPSILON: tl.constexpr,
    Q_A: tl.constexpr,
    OVF_A: tl.constexpr,
    NULL_A: tl.constexpr,
    Q_L: tl.constexpr,
    OVF_L: tl.constexpr,
    NULL_L: tl.constexpr,
    TRAINING: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    t = tl.program_id(2)
    rows = t * TB + tl.arange(0, TB)
    dk = tl.arange(0, DK)
    dv = tl.arange(0, DV)
    # Every array is contiguous (row-major) in the shape given at its pointer: its strides are compile-time constants.
    sqb: tl.constexpr = T * H * DK  # q
    sqt: tl.constexpr = H * DK
    sqh: tl.constexpr = DK
    skb: tl.constexpr = S * H * DK  # k
    sks: tl.constexpr = H * DK
    skh: tl.constexpr = DK
    svb: tl.constexpr = S * H * DV  # v
    svs: tl.constexpr = H * DV
    svh: tl.constexpr = DV
    sms: tl.constexpr = B * T  # valid
    smb: tl.constexpr = T
    sfb: tl.constexpr = T  # first
    snb: tl.constexpr = NB * NW  # n
    snt: tl.constexpr = NW
    sph: tl.constexpr = W  # qp
    sak: tl.constexpr = H * DV  # akif
    sah: tl.constexpr = DV
    sob: tl.constexpr = T * H * DV  # o
    sot: tl.constexpr = H * DV
    soh: tl.constexpr = DV
    slb: tl.constexpr = T * H  # l
    slt: tl.constexpr = H
    sxs: tl.constexpr = 2 * H * DV  # context: per round, its two calls
    sxh: tl.constexpr = DV
    sys: tl.constexpr = H  # held, served, rounded

    q = tl.load(q_ptr + b * sqb + rows[:, None] * sqt + h * sqh + dk[None, :])
    first = tl.load(first_ptr + b * sfb + rows)
    n = tl.load(n_ptr + b * snb + t * snt)
    ks = tl.load(qp_ptr + h * sph + 0)
    i_s = tl.load(qp_ptr + h * sph + 1)
    fs = tl.load(qp_ptr + h * sph + 2)
    ke = tl.load(qp_ptr + h * sph + 3)
    ie = tl.load(qp_ptr + h * sph + 4)
    fe = tl.load(qp_ptr + h * sph + 5)
    ko = tl.load(qp_ptr + h * sph + 6)
    io = tl.load(qp_ptr + h * sph + 7)
    fo = tl.load(qp_ptr + h * sph + 8)
    kl = tl.load(qp_ptr + h * sph + 9)
    il = tl.load(qp_ptr + h * sph + 10)
    fl = tl.load(qp_ptr + h * sph + 11)
    ka = tl.load(akif_ptr + h * sah + dv)[None, :]
    ia = tl.load(akif_ptr + sak + h * sah + dv)[None, :]
    fa = tl.load(akif_ptr + 2 * sak + h * sah + dv)[None, :]
    sc = tl.load(scaler_ptr)

    k_base = k_ptr + b * skb + h * skh + dk
    v_base = v_ptr + b * svb + h * svh + dv
    valid_base = valid_ptr + b * smb + rows

    # The running maximum's start: the quantized score of each query's first kept key.
    m = _first_score(q, k_base, sks, valid_base, sms, first, n, ks, i_s, fs, Q_S, RND_S, OVF_S, NULL_S)

    o = tl.zeros((TB, DV), tl.float32)
    l = tl.zeros((TB,), tl.float32)
    scored = tl.zeros((TB,), tl.float32)
    neg_s = tl.zeros((TB,), tl.int1)
    neg_c = tl.zeros((TB,), tl.int1)
    neg_a = tl.zeros((TB, DV), tl.int1)
    kept_base = context_ptr + h * sxh + dv
    added_base = kept_base + H * DV
    held_base = held_ptr + h
    for j in range(0, n):
        valid = tl.load(valid_base + j * sms) != 0.0
        raw = tl.sum(q * tl.load(k_base + j * sks)[None, :], axis=1)
        s = _quantize(tl.where(valid, raw, 0.0), ks, i_s, fs, Q_S, RND_S, OVF_S, NULL_S)
        scored = tl.maximum(scored, tl.abs(s), propagate_nan=tl.PropagateNan.ALL)
        neg_s = neg_s | (s < 0)
        next_m = tl.where(valid, tl.maximum(m, s, propagate_nan=tl.PropagateNan.ALL), m)
        r = exp_table(next_m - m, sc, ke, ie, fe, ko, io, fo, RND_EI, OVF_EI, NULL_EI, RND_EO, OVF_EO, NULL_EO)
        p = exp_table(
            tl.where(valid, next_m - s, 0.0), sc, ke, ie, fe, ko, io, fo, RND_EI, OVF_EI, NULL_EI, RND_EO, OVF_EO, NULL_EO
        )
        p = tl.where(valid, p, 0.0)
        if TWO_PASS:
            c = _quantize(tl.where(valid, r * l, 0.0), kl, il, fl, Q_L, RND_L, OVF_L, NULL_L)
            if TRAINING:
                _amax(held_base + j * sys, tl.max(tl.abs(c), axis=0))
                neg_c = neg_c | (c < 0)
        else:
            vj = tl.load(v_base + j * svs)
            a = _quantize(r[:, None] * o, ka, ia, fa, Q_A, RND_A, OVF_A, NULL_A)
            bq = _quantize(p[:, None] * tl.where(valid[:, None], vj[None, :], 0.0), ka, ia, fa, Q_A, RND_A, OVF_A, NULL_A)
            c = _quantize(r * l, kl, il, fl, Q_L, RND_L, OVF_L, NULL_L)
            if TRAINING:
                _amax(kept_base + j * sxs, tl.max(tl.abs(a), axis=0))
                _amax(added_base + j * sxs, tl.max(tl.abs(bq), axis=0))
                _amax(held_base + j * sys, tl.max(tl.abs(c), axis=0))
                neg_a = neg_a | (a < 0) | (bq < 0)
                neg_c = neg_c | (c < 0)
            o = tl.where(valid[:, None], a + bq, o)
        l = tl.where(valid, c + p, l)
        m = next_m
    if TRAINING:
        _amax(rounded_ptr + h, tl.max(scored, axis=0))

    if TWO_PASS:
        kp = tl.load(qp_ptr + h * sph + 18)
        ip = tl.load(qp_ptr + h * sph + 19)
        fp = tl.load(qp_ptr + h * sph + 20)
        inv = _inverse(l, qp_ptr + h * sph, EPSILON, Q_II, RND_II, OVF_II, NULL_II, RND_IO, OVF_IO, NULL_IO)
        ki, ii, fi = tl.load(qp_ptr + h * sph + 12), tl.load(qp_ptr + h * sph + 13), tl.load(qp_ptr + h * sph + 14)
        if TRAINING:
            lq = _quantize(l, ki, ii, fi, Q_II, RND_II, OVF_II, NULL_II)
            _amax(rounded_ptr + H + h, tl.max(tl.abs(lq), axis=0))
            _any(negative_ptr + (4 * H + h) * DV, lq < 0)
        neg_p = tl.zeros((TB,), tl.int1)
        served_base = served_ptr + h
        for j in range(0, n):
            valid, x, s = _round_scores(q, k_base, sks, valid_base, sms, j, ks, i_s, fs, Q_S, RND_S, OVF_S, NULL_S)
            e = tl.where(valid, m - s, 0.0)
            share = exp_table(e, sc, ke, ie, fe, ko, io, fo, RND_EI, OVF_EI, NULL_EI, RND_EO, OVF_EO, NULL_EO)
            prob = _quantize(tl.where(valid, share * inv, 0.0), kp, ip, fp, Q_P, RND_P, OVF_P, NULL_P)
            a = _quantize(tl.where(valid[:, None], o, 0.0), ka, ia, fa, Q_A, RND_A, OVF_A, NULL_A)
            bq = _quantize(prob[:, None] * tl.load(v_base + j * svs)[None, :], ka, ia, fa, Q_A, RND_A, OVF_A, NULL_A)
            if TRAINING:
                _amax(kept_base + j * sxs, tl.max(tl.abs(a), axis=0))
                _amax(added_base + j * sxs, tl.max(tl.abs(bq), axis=0))
                _amax(served_base + j * sys, tl.max(tl.abs(prob), axis=0))
                neg_a = neg_a | (a < 0) | (bq < 0)
                neg_p = neg_p | (prob < 0)
            o = tl.where(valid[:, None], a + bq, o)

    tl.store(o_ptr + b * sob + rows[:, None] * sot + h * soh + dv[None, :], o)
    tl.store(l_ptr + b * slb + rows * slt + h, l)

    if TRAINING:
        if TWO_PASS:
            # Rounds from n on mask every row: each quantizer call sees 0.
            zero = tl.zeros((TB,), tl.float32)
            kept = tl.max(tl.abs(fixed_round(tl.zeros((TB, DV), tl.float32), fa, RND_A)), axis=0)
            held = tl.max(tl.abs(fixed_round(zero, fl, RND_L)), axis=0)
            served = tl.max(tl.abs(_quantize(zero, kp, ip, fp, Q_P, RND_P, OVF_P, NULL_P)), axis=0)
            for j in range(n, S):
                _amax(kept_base + j * sxs, kept)
                _amax(added_base + j * sxs, kept)
                _amax(held_base + j * sys, held)
                _amax(served_base + j * sys, served)
            _any(negative_ptr + (3 * H + h) * DV, neg_p)
        else:
            # Rounds from n on keep every state; their quantizer calls see the rescale E(0) and no share.
            quiet = exp_table(
                tl.zeros((TB,), tl.float32), sc, ke, ie, fe, ko, io, fo, RND_EI, OVF_EI, NULL_EI, RND_EO, OVF_EO, NULL_EO
            )
            a = fixed_round(quiet[:, None] * o, fa, RND_A)
            c = fixed_round(quiet * l, fl, RND_L)
            kept, held = tl.max(tl.abs(a), axis=0), tl.max(tl.abs(c), axis=0)
            added = tl.max(tl.abs(fixed_round(tl.zeros((TB, DV), tl.float32), fa, RND_A)), axis=0)
            for j in range(n, S):
                _amax(kept_base + j * sxs, kept)
                _amax(added_base + j * sxs, added)
                _amax(held_base + j * sys, held)
                neg_a = neg_a | (a < 0)
                neg_c = neg_c | (c < 0)
        _any(negative_ptr + h * DV + dv, neg_a)
        _any(negative_ptr + (H + h) * DV, neg_c)
        _any(negative_ptr + (2 * H + h) * DV, neg_s)


@triton.jit
def attn_bwd(
    q_ptr,  # (B, T, H, DK) projected queries
    k_ptr,  # (B, S, H, DK) projected keys
    v_ptr,  # (B, S, H, DV) projected values
    valid_ptr,  # (S, B, T) float, nonzero where query t keeps key j
    first_ptr,  # (B, T) int32, each query's first kept key
    n_ptr,  # (B, NB, *) int32, column 0: keys to visit per query block
    qp_ptr,  # (H, *) per-head (k, i, f) of the score, exp-table input, exp-table output and weight quantizers (and
    # 2-pass, columns 12-20 as attn_fwd's)
    akif_ptr,  # (3, H, DV) the context quantizer's (k, i, f) per value lane
    scaler_ptr,  # (1,) the exp table's input scaler
    go_ptr,  # (B, T, H, DV) cotangent of o
    gl_ptr,  # (B, T, H) cotangent of l
    dquery_ptr,  # out (after every input, as jax_triton passes outputs) (B, T, H, DK) query cotangent
    dkey_ptr,  # out (NB, B, S, H, DK) this block's share of the key cotangent
    dvalue_ptr,  # out (NB, B, S, H, DV) this block's share of the value cotangent
    df_ptr,  # out (B, H, NB, DV) this block's share of the context quantizer's f cotangent per lane
    dq_ptr,  # out (B, H, NB, W) this block's share of qp's cotangent, in qp's column order
    later_ptr,  # scratch (S, B, H, T): r_j (sweep A), then R_j (sweep B), then ds_j (sweep C), the cotangent of round
    # j's quantized score through its share, in place of R_j after round j of sweep C reads it
    dm_ptr,  # scratch (S, B, H, T): cotangent of round j's running maximum from the rounds after it
    max_ptr,  # scratch (S, B, H, T): round j's running maximum (2-pass: before sweep C, round j's score cotangent
    # through the second pass)
    B: tl.constexpr,
    T: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    NB: tl.constexpr,  # query blocks, T / TB
    W: tl.constexpr,  # columns of qp (at least 12) and of the bit-width partials
    NW: tl.constexpr,  # columns of n
    TB: tl.constexpr,
    RND_S: tl.constexpr,
    OVF_S: tl.constexpr,
    NULL_S: tl.constexpr,
    RND_EI: tl.constexpr,
    OVF_EI: tl.constexpr,
    NULL_EI: tl.constexpr,
    RND_EO: tl.constexpr,
    OVF_EO: tl.constexpr,
    NULL_EO: tl.constexpr,
    RND_A: tl.constexpr,
    RND_L: tl.constexpr,
    Q_S: tl.constexpr,
    TWO_PASS: tl.constexpr,
    RND_II: tl.constexpr,
    OVF_II: tl.constexpr,
    NULL_II: tl.constexpr,
    Q_II: tl.constexpr,
    RND_IO: tl.constexpr,
    OVF_IO: tl.constexpr,
    NULL_IO: tl.constexpr,
    Q_P: tl.constexpr,
    RND_P: tl.constexpr,
    OVF_P: tl.constexpr,
    NULL_P: tl.constexpr,
    EPSILON: tl.constexpr,
    Q_A: tl.constexpr,
    OVF_A: tl.constexpr,
    NULL_A: tl.constexpr,
    Q_L: tl.constexpr,
    OVF_L: tl.constexpr,
    NULL_L: tl.constexpr,
    TRAINING: tl.constexpr,
):
    """Four sweeps over the rounds sharing per-round values through this program's slice of the scratch outputs, with a
    barrier of the program's threads at each of six sync sites. (A) the rescales r_j; (B) reversed, the products R_j of
    r over later valid rounds (0 on a skipped round); (C) the rounds again: o_j's and l_j's cotangents are R_j go and
    R_j gl (the context and weight quantizers round only, with identity input derivatives), the vjps of Qa, Ql and both
    exp-table calls give their inputs' and bit widths' cotangents, and those reaching each round's running maximum and
    score are stored; (D) reversed, the running maximum's cotangent (a tie splits it equally), the extra path into the
    first kept score, the score quantizer's vjp, dQ and this block's dK share. Rounds
    from n on change nothing: zero dK and dV shares. 2-pass: sweep A also runs the first pass's statistics and sweep P,
    the second pass's vjps, follows it; sweep C then has no context term, its l cotangent gl includes the one through
    the inverse table, and the final maximum's cotangent the second pass's."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    t = tl.program_id(2)
    rows = t * TB + tl.arange(0, TB)
    dk = tl.arange(0, DK)
    dv = tl.arange(0, DV)
    # Every array is contiguous (row-major) in the shape given at its pointer: its strides are compile-time constants.
    sqb: tl.constexpr = T * H * DK  # q, dquery
    sqt: tl.constexpr = H * DK
    sqh: tl.constexpr = DK
    skb: tl.constexpr = S * H * DK  # k
    sks: tl.constexpr = H * DK
    skh: tl.constexpr = DK
    svb: tl.constexpr = S * H * DV  # v
    svs: tl.constexpr = H * DV
    svh: tl.constexpr = DV
    sms: tl.constexpr = B * T  # valid
    smb: tl.constexpr = T
    sfb: tl.constexpr = T  # first
    snb: tl.constexpr = NB * NW  # n
    snt: tl.constexpr = NW
    sph: tl.constexpr = W  # qp
    sak: tl.constexpr = H * DV  # akif
    sah: tl.constexpr = DV
    sgb: tl.constexpr = T * H * DV  # go
    sgt: tl.constexpr = H * DV
    sgh: tl.constexpr = DV
    sglb: tl.constexpr = T * H  # gl
    sglt: tl.constexpr = H
    sdkt: tl.constexpr = B * S * H * DK  # dkey
    sdkb: tl.constexpr = S * H * DK
    sdks: tl.constexpr = H * DK
    sdkh: tl.constexpr = DK
    sdvt: tl.constexpr = B * S * H * DV  # dvalue
    sdvb: tl.constexpr = S * H * DV
    sdvs: tl.constexpr = H * DV
    sdvh: tl.constexpr = DV
    sdfb: tl.constexpr = H * NB * DV  # df
    sdfh: tl.constexpr = NB * DV
    sdft: tl.constexpr = DV
    sdqb: tl.constexpr = H * NB * W  # dq
    sdqh: tl.constexpr = NB * W
    sdqt: tl.constexpr = W
    sxs: tl.constexpr = B * H * T  # scratch: later (then ds), dm, max
    sxb: tl.constexpr = H * T
    sxh: tl.constexpr = T

    q = tl.load(q_ptr + b * sqb + rows[:, None] * sqt + h * sqh + dk[None, :])
    go = tl.load(go_ptr + b * sgb + rows[:, None] * sgt + h * sgh + dv[None, :])
    gl = tl.load(gl_ptr + b * sglb + rows * sglt + h)
    first = tl.load(first_ptr + b * sfb + rows)
    n = tl.load(n_ptr + b * snb + t * snt)
    ks = tl.load(qp_ptr + h * sph + 0)
    i_s = tl.load(qp_ptr + h * sph + 1)
    fs = tl.load(qp_ptr + h * sph + 2)
    ke = tl.load(qp_ptr + h * sph + 3)
    ie = tl.load(qp_ptr + h * sph + 4)
    fe = tl.load(qp_ptr + h * sph + 5)
    ko = tl.load(qp_ptr + h * sph + 6)
    io = tl.load(qp_ptr + h * sph + 7)
    fo = tl.load(qp_ptr + h * sph + 8)
    fl = tl.load(qp_ptr + h * sph + 11)
    fa = tl.load(akif_ptr + 2 * sak + h * sah + dv)[None, :]
    sc = tl.load(scaler_ptr)

    k_base = k_ptr + b * skb + h * skh + dk
    v_base = v_ptr + b * svb + h * svh + dv
    valid_base = valid_ptr + b * smb + rows
    scratch = b * sxb + h * sxh + rows
    later_base = later_ptr + scratch
    dm_base = dm_ptr + scratch
    max_base = max_ptr + scratch
    ds_base = later_base  # R_j's last read is round j of sweep C, which then stores ds_j in its place
    dkey_base = dkey_ptr + t * sdkt + b * sdkb + h * sdkh + dk
    dvalue_base = dvalue_ptr + t * sdvt + b * sdvb + h * sdvh + dv

    # The running maximum's start: the quantized score of each query's first kept key. Its cotangent reaches that
    # score in sweep D (round `first`), with the score quantizer's vjp of the same masked raw score.
    m0 = _first_score(q, k_base, sks, valid_base, sms, first, n, ks, i_s, fs, Q_S, RND_S, OVF_S, NULL_S)

    zero = tl.sum(tl.zeros((TB,), tl.float32), axis=0)
    df = tl.zeros((DV,), tl.float32)
    dqe_i = zero
    dqe_f = zero
    dqo_i = zero
    dqo_f = zero

    # (A) the rescales r_j; 2-pass: also the first pass's statistics (m, l)
    m = m0
    l = tl.zeros((TB,), tl.float32)
    for j in range(0, n):
        valid, x, s = _round_scores(q, k_base, sks, valid_base, sms, j, ks, i_s, fs, Q_S, RND_S, OVF_S, NULL_S)
        next_m = tl.where(valid, tl.maximum(m, s, propagate_nan=tl.PropagateNan.ALL), m)
        r = exp_table(next_m - m, sc, ke, ie, fe, ko, io, fo, RND_EI, OVF_EI, NULL_EI, RND_EO, OVF_EO, NULL_EO)
        tl.store(later_base + j * sxs, r)
        if TWO_PASS:
            difference = tl.where(valid, next_m - s, 0.0)
            p = exp_table(difference, sc, ke, ie, fe, ko, io, fo, RND_EI, OVF_EI, NULL_EI, RND_EO, OVF_EO, NULL_EO)
            l = tl.where(valid, fixed_round(r * l, fl, RND_L) + p, l)
        m = next_m

    if TWO_PASS:
        # (P) the second pass with its vjps: the value cotangent, the cotangents of the maximum m and of l (through
        # the inverse table, added to gl) and, stored in max's place until sweep C reads it, each round's score
        # cotangent. The context quantizer rounds only (identity input derivative): every round's context cotangent
        # is go.
        kp = tl.load(qp_ptr + h * sph + 18)
        ip = tl.load(qp_ptr + h * sph + 19)
        fp = tl.load(qp_ptr + h * sph + 20)
        inv = _inverse(l, qp_ptr + h * sph, EPSILON, Q_II, RND_II, OVF_II, NULL_II, RND_IO, OVF_IO, NULL_IO)
        o = tl.zeros((TB, DV), tl.float32)
        g_max = tl.zeros((TB,), tl.float32)
        g_inv = tl.zeros((TB,), tl.float32)
        dqp_i = zero
        dqp_f = zero
        for j in range(0, n):
            valid, x, s = _round_scores(q, k_base, sks, valid_base, sms, j, ks, i_s, fs, Q_S, RND_S, OVF_S, NULL_S)
            e = tl.where(valid, m - s, 0.0)
            u = exp_table(e, sc, ke, ie, fe, ko, io, fo, RND_EI, OVF_EI, NULL_EI, RND_EO, OVF_EO, NULL_EO)
            y = tl.where(valid, u * inv, 0.0)
            prob = _quantize(y, kp, ip, fp, Q_P, RND_P, OVF_P, NULL_P)
            vj = tl.load(v_base + j * svs)[None, :]
            g_o = tl.where(valid[:, None], go, 0.0)
            x_kept = tl.where(valid[:, None], o, 0.0)
            x_added = prob[:, None] * vj
            _, df_kept = fixed_round_vjp(x_kept, fa, g_o, RND_A)
            g_added, df_added = fixed_round_vjp(x_added, fa, g_o, RND_A)
            tl.store(dvalue_base + j * sdvs, tl.sum(g_added * prob[:, None], axis=0))
            g_y, dip, dfp = _quantize_vjp(y, kp, ip, fp, tl.sum(g_added * vj, axis=1), Q_P, RND_P, OVF_P, NULL_P)
            g_y = tl.where(valid, g_y, 0.0)
            g_inv = g_inv + g_y * u
            g_e, die, dfe, dio, dfo = exp_table_vjp(
                e, sc, ke, ie, fe, ko, io, fo, g_y * inv, RND_EI, OVF_EI, NULL_EI, RND_EO, OVF_EO, NULL_EO
            )
            g_e = tl.where(valid, g_e, 0.0)
            g_max = g_max + g_e
            tl.store(max_base + j * sxs, -g_e)
            dqe_i = dqe_i + tl.sum(die, axis=0)
            dqe_f = dqe_f + tl.sum(dfe, axis=0)
            dqo_i = dqo_i + tl.sum(dio, axis=0)
            dqo_f = dqo_f + tl.sum(dfo, axis=0)
            dqp_i = dqp_i + tl.sum(dip, axis=0)
            dqp_f = dqp_f + tl.sum(dfp, axis=0)
            df = df + tl.sum(df_kept, axis=0) + tl.sum(df_added, axis=0)
            o = tl.where(valid[:, None], fixed_round(x_kept, fa, RND_A) + fixed_round(x_added, fa, RND_A), o)
        g_l, dii_t, dfi_t, dio_t, dfo_t = _inverse_vjp(
            l, qp_ptr + h * sph, EPSILON, g_inv, Q_II, RND_II, OVF_II, NULL_II, RND_IO, OVF_IO, NULL_IO
        )
        gl = gl + g_l
    tl.debug_barrier()  # sync 1: A's stores of r before B's reads

    # (B) reversed: R_j, the product of the rescales of the valid rounds after j
    product = tl.full((TB,), 1.0, tl.float32)
    for i in range(0, n):
        j = n - 1 - i
        valid = tl.load(valid_base + j * sms) != 0.0
        r = tl.load(later_base + j * sxs)
        tl.debug_barrier()  # sync 2: every thread holding r_j has read it before it is overwritten in place
        tl.store(later_base + j * sxs, tl.where(valid, product, 0.0))
        product = tl.where(valid, product * r, product)
    tl.debug_barrier()  # sync 3: B's stores of R before C's reads

    # (C) the rounds again, with the vjps of every quantizer and exp-table call
    m = m0
    o = tl.zeros((TB, DV), tl.float32)
    l = tl.zeros((TB,), tl.float32)
    pending = tl.zeros((TB,), tl.float32)
    first_rise = tl.zeros((TB,), tl.float32)
    dfl = zero
    for j in range(0, n):
        valid, x, s = _round_scores(q, k_base, sks, valid_base, sms, j, ks, i_s, fs, Q_S, RND_S, OVF_S, NULL_S)
        next_m = tl.where(valid, tl.maximum(m, s, propagate_nan=tl.PropagateNan.ALL), m)
        rise = next_m - m
        difference = tl.where(valid, next_m - s, 0.0)
        r = exp_table(rise, sc, ke, ie, fe, ko, io, fo, RND_EI, OVF_EI, NULL_EI, RND_EO, OVF_EO, NULL_EO)
        p = tl.where(
            valid, exp_table(difference, sc, ke, ie, fe, ko, io, fo, RND_EI, OVF_EI, NULL_EI, RND_EO, OVF_EO, NULL_EO), 0.0
        )
        factor = tl.load(later_base + j * sxs)
        if TWO_PASS:
            ds_second = tl.load(max_base + j * sxs)  # before sync 6, after which next_m takes its place
            x_held = r * l
            c = fixed_round(x_held, fl, RND_L)
            g_held, dfl_held = fixed_round_vjp(x_held, fl, factor * gl, RND_L)
            g_r = g_held * l
            g_q = factor * gl
        else:
            vj = tl.where(valid[:, None], tl.load(v_base + j * svs)[None, :], 0.0)
            x_kept = r[:, None] * o
            x_added = p[:, None] * vj
            x_held = r * l
            a = fixed_round(x_kept, fa, RND_A)
            bq = fixed_round(x_added, fa, RND_A)
            c = fixed_round(x_held, fl, RND_L)
            g_kept, df_kept = fixed_round_vjp(x_kept, fa, factor[:, None] * go, RND_A)
            g_added, df_added = fixed_round_vjp(x_added, fa, factor[:, None] * go, RND_A)
            g_held, dfl_held = fixed_round_vjp(x_held, fl, factor * gl, RND_L)
            g_r = tl.sum(g_kept * o, axis=1) + g_held * l
            g_q = tl.sum(g_added * vj, axis=1) + factor * gl
            tl.store(dvalue_base + j * sdvs, tl.sum(g_added * p[:, None], axis=0))
        g_rise, die_r, dfe_r, dio_r, dfo_r = exp_table_vjp(
            rise, sc, ke, ie, fe, ko, io, fo, g_r, RND_EI, OVF_EI, NULL_EI, RND_EO, OVF_EO, NULL_EO
        )
        g_difference, die_q, dfe_q, dio_q, dfo_q = exp_table_vjp(
            difference, sc, ke, ie, fe, ko, io, fo, g_q, RND_EI, OVF_EI, NULL_EI, RND_EO, OVF_EO, NULL_EO
        )
        g_difference = tl.where(valid, g_difference, 0.0)
        # round j-1's maximum is round j's previous one: its cotangent is complete once rise_j's is known
        tl.store(dm_base + tl.maximum(j - 1, 0) * sxs, pending - g_rise)
        tl.debug_barrier()  # sync 6: every thread holding R_j has read it before ds_j overwrites it in place
        if TWO_PASS:
            tl.store(ds_base + j * sxs, ds_second - g_difference)
        else:
            tl.store(ds_base + j * sxs, -g_difference)
        tl.store(max_base + j * sxs, next_m)
        first_rise = tl.where(j == 0, -g_rise, first_rise)
        dqe_i = dqe_i + tl.sum(die_r, axis=0) + tl.sum(die_q, axis=0)
        dqe_f = dqe_f + tl.sum(dfe_r, axis=0) + tl.sum(dfe_q, axis=0)
        dqo_i = dqo_i + tl.sum(dio_r, axis=0) + tl.sum(dio_q, axis=0)
        dqo_f = dqo_f + tl.sum(dfo_r, axis=0) + tl.sum(dfo_q, axis=0)
        if not TWO_PASS:
            o = tl.where(valid[:, None], a + bq, o)
        l = tl.where(valid, c + p, l)
        if not TWO_PASS:
            df = df + tl.sum(df_kept, axis=0) + tl.sum(df_added, axis=0)
        dfl = dfl + tl.sum(dfl_held, axis=0)
        pending = g_rise + g_difference
        m = next_m
    tl.store(df_ptr + b * sdfb + h * sdfh + t * sdft + dv, df)
    tl.debug_barrier()  # sync 4: with one round, the loop's store of row 0 comes first
    if TWO_PASS:
        pending = pending + g_max  # the final maximum's use in the second pass
    tl.store(dm_base + tl.maximum(n - 1, 0) * sxs, pending)
    tl.debug_barrier()  # sync 5: C's stores of dm, max and ds before D's reads

    # (D) reversed: the running maximum's cotangent, the score quantizer's vjp, dQ and this block's dK share
    d = tl.zeros((TB,), tl.float32)
    dquery = tl.zeros((TB, DK), tl.float32)
    dqs_i = zero
    dqs_f = zero
    for i in range(0, n):
        j = n - 1 - i
        d = d + tl.load(dm_base + j * sxs)
        valid = tl.load(valid_base + j * sms) != 0.0
        kj = tl.load(k_base + j * sks)
        x = tl.where(valid, tl.sum(q * kj[None, :], axis=1), 0.0)
        s = _quantize(x, ks, i_s, fs, Q_S, RND_S, OVF_S, NULL_S)
        mj = tl.load(max_base + j * sxs)
        previous = tl.where(j == 0, m0, tl.load(max_base + tl.maximum(j - 1, 0) * sxs))
        to_score = tl.where(valid & (s == mj), d * tl.where(previous == mj, 0.5, 1.0), 0.0)
        to_previous = tl.where(valid, tl.where(previous == mj, d * tl.where(s == mj, 0.5, 1.0), 0.0), d)
        # The first kept key's score also starts the running maximum (all earlier rounds are skipped): its share of
        # the cotangent is complete here, the rounds before add only round j-1's.
        to_start = to_previous + tl.where(j > 0, tl.load(dm_base + tl.maximum(j - 1, 0) * sxs), first_rise)
        g_s = to_score + tl.load(ds_base + j * sxs) + tl.where(j == first, to_start, 0.0)
        g_x, di_s, df_s = _quantize_vjp(x, ks, i_s, fs, g_s, Q_S, RND_S, OVF_S, NULL_S)
        g_raw = tl.where(valid, g_x, 0.0)
        tl.store(dkey_base + j * sdks, tl.sum(g_raw[:, None] * q, axis=0))
        dquery = dquery + g_raw[:, None] * kj[None, :]
        dqs_i = dqs_i + tl.sum(di_s, axis=0)
        dqs_f = dqs_f + tl.sum(df_s, axis=0)
        d = to_previous
    tl.store(dquery_ptr + b * sqb + rows[:, None] * sqt + h * sqh + dk[None, :], dquery)

    # qp's columns: (k, i, f) of the score, exp-table input, exp-table output and weight quantizers; none for k, which
    # has none in the quantizers' calls, nor for the weight quantizer's i, which its rounding does not read
    dq_base = dq_ptr + b * sdqb + h * sdqh + t * sdqt
    tl.store(dq_base + 0, zero)
    tl.store(dq_base + 1, dqs_i)
    tl.store(dq_base + 2, dqs_f)
    tl.store(dq_base + 3, zero)
    tl.store(dq_base + 4, dqe_i)
    tl.store(dq_base + 5, dqe_f)
    tl.store(dq_base + 6, zero)
    tl.store(dq_base + 7, dqo_i)
    tl.store(dq_base + 8, dqo_f)
    tl.store(dq_base + 9, zero)
    tl.store(dq_base + 10, zero)
    tl.store(dq_base + 11, dfl)
    if TWO_PASS:
        # 2-pass: (k, i, f) of the inverse table's input and output quantizers and of the probability quantizer
        tl.store(dq_base + 12, zero)
        tl.store(dq_base + 13, tl.sum(dii_t, axis=0))
        tl.store(dq_base + 14, tl.sum(dfi_t, axis=0))
        tl.store(dq_base + 15, zero)
        tl.store(dq_base + 16, tl.sum(dio_t, axis=0))
        tl.store(dq_base + 17, tl.sum(dfo_t, axis=0))
        tl.store(dq_base + 18, zero)
        tl.store(dq_base + 19, dqp_i)
        tl.store(dq_base + 20, dqp_f)
        for c in range(21, W):
            tl.store(dq_base + c, zero)
    else:
        for c in range(12, W):
            tl.store(dq_base + c, zero)

    for j in range(n, S):
        tl.store(dkey_base + j * sdks, tl.zeros((DK,), tl.float32))
        tl.store(dvalue_base + j * sdvs, tl.zeros((DV,), tl.float32))
