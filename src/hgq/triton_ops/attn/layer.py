"""Opt-in plain-Triton route of QMultiHeadAttention's 1-pass and 2-pass attention on JAX and torch, training, tracing and
inference calls.

Switch (default off): hgq.triton_ops.enable(attention=True), or env HGQ_TRITON_ATTENTION=1; switching it on imports the
route (`check`) and raises if it is unavailable. The switch is read when the layer call runs (or is traced), so a jitted /
compiled function keeps the setting it was traced with (retrace to change it). Where `applies`, QMultiHeadAttention.call
takes `attention`: the kernels of attn.jax_op / attn.torch_op on the projections, the integer-bit replay of the rounding
(trainable WRAP) quantizers from the per-call maxima the forward kernel atomically maxes into small zeroed buffers (the
replay kernel, replay.jax_op / replay.torch_op, one launch per quantizer) and, 1-pass, the normalization by the inverse
table in keras ops (2-pass: in the kernel). A tracing call (training 'tracing') runs the training kernels, its replay
with decay 0 from the rounded integer bits (FixedPointQuantizerKIF.call's tracing transition), and ORs into each
rounding quantizer's k whether the kernel saw any of its outputs negative. An inference call (training False or None)
launches the forward kernel only, every quantizer in its quantize form.

Fallback: with the switch on, a call that cannot take the route runs the keras-ops scan, with a warning per
distinct reason, once, when the gate runs (traced): a configuration the kernels do not reproduce (`applies`), the
computation not placed on a CUDA GPU, a replayed quantizer with a bit-width mapper other than DefaultBitwidthMapper or an
integer-bit constraint other than None or MinMax. Errors raised while building or running the kernels propagate.

Placement: torch, the tensor's device (``is_cuda``). JAX, a concrete operand's devices; inside a trace (jax.jit, the
jax.checkpoint around the attention) operands carry no device, so the device JAX places uncommitted computation on
(jax_default_device, else the default backend's first device). A jitted call on inputs committed elsewhere fails to
lower the Triton call rather than running wrongly.
"""

import warnings
from math import prod

import keras
from keras import ops

from ... import triton_ops
from ...constraints import MinMax
from ...quantizer import Quantizer
from ...quantizer.internal.base import DefaultBitwidthMapper
from ...quantizer.internal.fixed_point_quantizer import FixedPointQuantizerKBI, FixedPointQuantizerKIF
from .spec import attention_settings, query_block

_WARNED = set()
_SAVED = 'triton_attention_outputs'  # checkpoint_name of the kernel's outputs, which the JAX backward keeps


def check():
    """Import the route on the current backend (before any torch.compile tracing); ImportError if it is unavailable."""
    import triton  # noqa: F401
    from quantizers import triton as qt  # noqa: F401

    from ..replay import kernels as replay_kernels  # noqa: F401
    from . import kernels  # noqa: F401

    if keras.backend.backend() == 'jax':
        import jax_triton  # noqa: F401

        from ..replay import jax_op as replay_op  # noqa: F401
        from . import jax_op  # noqa: F401
    else:
        from ..replay import torch_op as replay_op  # noqa: F401
        from . import torch_op  # noqa: F401


def _kind(q, training: bool) -> str:
    """The form of fixed-point KIF or KBI quantizer ``q``'s call: in training or tracing (``training``), 'round' for a
    trainable KIF WRAP one (its call rounds, then updates the state), 'quantize' for a stateless one; in inference,
    'quantize' (a trainable-WRAP weight quantizer's, which only rounds, excepted). 'other' for any other (subclasses,
    scaler, noise mix, affine map, stochastic rounding, trainable KBI WRAP in training or tracing)."""
    if type(q) is not Quantizer or q.scaler is not None or q.qnoise_factor is not None or q.affine is not None:
        return 'other'
    inner = q.quantizer
    if type(inner) not in (FixedPointQuantizerKIF, FixedPointQuantizerKBI) or inner._stochastic:
        return 'other'
    if inner.overflow_mode == 'WRAP' and inner.trainable:
        if not training:
            return 'other' if inner._is_weight else 'quantize'
        return 'round' if type(inner) is FixedPointQuantizerKIF else 'other'
    return 'quantize'


def _per_head(q, lanes: int) -> bool:
    """Bit-width shape (1, heads or 1, 1, ``lanes`` or 1)."""
    shape = q.quantizer._i.shape
    return len(shape) == 4 and shape[0] == shape[2] == 1 and shape[3] in (1, lanes)


def _gate(reason) -> bool:
    """Warn once per distinct reason that the keras-ops attention runs instead."""
    if reason not in _WARNED:
        _WARNED.add(reason)
        warnings.warn(f'hgq triton_attention: running the keras-ops attention ({reason})', stacklevel=3)
    return False


def applies(layer, query, key, value, training) -> bool:
    """Whether QMultiHeadAttention ``layer``'s call on the projections runs as `attention`: a call with the switch on, of
    a 1-pass or 2-pass layer whose call the kernels reproduce; a reason they do not is warned once. Admitted: one device
    (the kernel call is not split by batch like the ops around it); the (B, T, heads, hd) projections, attention over T;
    key, query, value-lane and key-depth counts powers of two; in training (and tracing, alike), context and weight
    quantizers rounding, the score quantizer either kind, the exp table's quantizers stateless (`_kind`); in inference,
    every quantizer a fixed-point one the kernels quantize; bit widths at most per head (and value lane for the
    context); the exp table without EBOPs. 2-pass also: the inverse table without EBOPs, its output quantizer stateless
    (training), its input and the probability quantizer either kind, at most per head. In training, the replayed
    (rounding) quantizers with the default bit-width mapper (the replay kernel reduces as it does) and no or a MinMax
    integer-bit constraint. Then placement on a CUDA GPU."""
    if not triton_ops.enabled('attention') or layer._softmax_kind == 'comb' or layer._fuse != 'none':
        return False
    training = training is True or training == 'tracing'
    softmax, table = layer._softmax, layer._softmax.exp_table
    rounds, queries, lanes, depth = key.shape[1], query.shape[1], value.shape[-1], key.shape[-1]
    distribution = keras.distribution.distribution()
    if distribution is not None and prod(distribution.device_mesh.shape) > 1:
        return _gate('multi-device distribution')
    if layer._dot_product_equation != 'aecd,abcd->acbe':
        return _gate(f'attention equation {layer._dot_product_equation}')
    if not all(n > 0 and n & (n - 1) == 0 for n in (rounds, queries, lanes, depth)):
        return _gate(f'keys, queries, value and key dims {rounds}, {queries}, {lanes}, {depth} not all powers of two')
    if not (softmax.enable_iq and table.enable_iq and table.enable_oq) or table.enable_ebops:
        return _gate('score or exp-table quantizer disabled, or exp-table EBOPs on')
    if not _kind(softmax.aq, training) == _kind(softmax.lq, training) == ('round' if training else 'quantize'):
        if training:
            return _gate('context or weight quantizer not a trainable fixed-point KIF WRAP quantizer')
        return _gate('context or weight quantizer not a fixed-point KIF or KBI quantizer (inference)')
    if _kind(softmax.iq, training) == 'other':
        return _gate('score quantizer neither a stateless nor a trainable KIF WRAP fixed-point quantizer')
    if not _kind(table.iq, training) == _kind(table.oq, training) == 'quantize':
        return _gate('exp-table quantizer not a stateless fixed-point quantizer')
    if not (_per_head(softmax.aq, lanes) and all(_per_head(q, 1) for q in (softmax.lq, softmax.iq, table.iq, table.oq))):
        return _gate('bit widths finer than per head (and value lane for the context quantizer)')
    replayed = {'context': softmax.aq, 'weight': softmax.lq, 'score': softmax.iq}
    if softmax.impl == '2pass':
        inv, oq = softmax.inv_table, softmax.oq
        if not (inv.enable_iq and inv.enable_oq) or inv.enable_ebops:
            return _gate('inverse-table quantizer disabled, or inverse-table EBOPs on')
        if _kind(inv.oq, training) != 'quantize':
            return _gate('inverse-table output quantizer not a stateless fixed-point quantizer')
        if 'other' in (_kind(inv.iq, training), _kind(oq, training)):
            return _gate('inverse-table input or probability quantizer neither a stateless nor a trainable KIF WRAP quantizer')
        if not all(_per_head(q, 1) for q in (inv.iq, inv.oq, oq)):
            return _gate('inverse-table or probability bit widths finer than per head')
        replayed |= {'probability': oq, 'inverse-table input': inv.iq}
    for name, q in replayed.items():
        if _kind(q, training) != 'round':
            continue
        mapper, constraint = q.quantizer.bw_mapper, q.quantizer._i.constraint
        if type(mapper) is not DefaultBitwidthMapper:
            return _gate(f'{name} bit-width mapper {type(mapper).__name__} (the replay reduces as DefaultBitwidthMapper)')
        if constraint is not None and type(constraint) is not MinMax:
            return _gate(f'{name} integer-bit constraint {type(constraint).__name__} (the replay takes None or MinMax)')
    if not triton_ops.on_cuda(query):
        return _gate('not placed on a CUDA GPU')
    return True


def _replay(quantizer, maxima, negative, tracing: bool, repeats=1):
    """Assign a rounding quantizer the state its training or ``tracing`` calls leave (FixedPointQuantizerKIF.call's
    transition per call), from the kernel's per-call abs-maxima ``(calls, heads[, lanes])``, each call ``repeats`` times
    in a row; tracing, the integer bits with decay 0 and k ORed with ``negative`` (heads, lanes), whether any of the
    calls' outputs was negative."""
    if keras.backend.backend() == 'jax':
        from ..replay.jax_op import replay_integer_bits
    else:
        from ..replay.torch_op import replay_integer_bits
    q = quantizer.quantizer
    c = q._i.constraint  # None or MinMax (`applies`)
    decay = ops.zeros_like(q.i_decay_speed) if tracing else q.i_decay_speed
    i, f, decay, maxima = (ops.stop_gradient(x) for x in (q._i.value, q.f, decay, maxima))
    bounds = None if c is None else (c.min_value, c.max_value)
    q._i.assign(replay_integer_bits(i, maxima, f, decay, bounds=bounds, repeats=repeats, tracing=tracing))
    if tracing:
        flag = ops.reshape(negative, (1, negative.shape[0], 1, negative.shape[1]))
        axes = tuple(a for a in (1, 3) if q._k.shape[a] == 1)
        if axes:
            flag = ops.amax(flag, axis=axes, keepdims=True)
        q._k.assign(ops.cast(ops.cast(q.k, 'bool') | (flag > 0), q._k.dtype))


def _attention(softmax, query, key, value, attention_mask, training):
    """The rounds of the kernels (scores, score quantizer, running maximum from the first kept key's quantized score, exp
    table, weight and context recurrences; 2-pass also the second pass); 1-pass, the normalization by the inverse table.
    Training and tracing, the replay of the rounding quantizers' state from the kernel's per-call maxima (and negative
    flags); inference, the forward kernel alone, every quantizer in its quantize form (no state change)."""
    if keras.backend.backend() == 'jax':
        from . import jax_op as binding
    else:
        from . import torch_op as binding
    batch, queries, heads, _ = query.shape
    rounds, lanes = key.shape[1], value.shape[-1]
    stateful, tracing = training is True or training == 'tracing', training == 'tracing'
    settings, block = attention_settings(softmax, stateful), query_block(queries, None)
    nb = queries // block
    keep = ops.ones((batch, 1, rounds), 'bool') if attention_mask is None else ops.cast(attention_mask, 'bool')
    valid = ops.cast(ops.broadcast_to(ops.moveaxis(keep, -1, 0), (rounds, batch, queries)), 'float32')
    first = ops.cast(ops.broadcast_to(ops.argmax(ops.cast(keep, 'int32'), axis=-1), (batch, queries)), 'int32')
    # Keys to visit per batch element and query block: up to the last one kept for any of its queries.
    kept_any = ops.any(ops.reshape(valid, (rounds, batch, nb, block)) != 0.0, axis=3)
    n = ops.amax(ops.where(kept_any, ops.arange(1, rounds + 1, dtype='int32')[:, None, None], 0), axis=0)
    n = ops.broadcast_to(n[..., None], (batch, nb, 16))
    quantizers = [softmax.iq, softmax.exp_table.iq, softmax.exp_table.oq, softmax.lq]
    if settings.two_pass:
        quantizers += [softmax.inv_table.iq, softmax.inv_table.oq, softmax.oq]
    kifs = [ops.broadcast_to(ops.cast(v, 'float32')[0, :, 0, 0], (heads,)) for q in quantizers for v in q.quantizer.kif]
    qp = ops.stack([*kifs, *[ops.zeros((heads,), 'float32')] * ((24 if settings.two_pass else 16) - len(kifs))], axis=1)
    akif = ops.stack([ops.broadcast_to(ops.cast(v, 'float32')[0, :, 0, :], (heads, lanes)) for v in softmax.aq.quantizer.kif])

    if stateful:
        outs = binding.attention(query, key, value, qp, akif, valid, first, n, settings, block)
    else:
        outs = binding.attention_forward(query, key, value, valid, first, n, qp, akif, settings, block)
    context, weight, maxima, held, served, rounded, negative = outs
    context, weight = ops.swapaxes(context, 1, 2), ops.swapaxes(weight, 1, 2)  # (B, T, H, ...) -> (B, H, T, ...)
    if stateful:
        if keras.backend.backend() == 'jax':
            from jax.ad_checkpoint import checkpoint_name

            context, weight = checkpoint_name((context, weight), _SAVED)
        # Calls in call order: context two per round, weight one per round, the score once; 2-pass, probability one per
        # round and inverse-table input once per round (the same maxima every round).
        _replay(softmax.aq, maxima, negative[0], tracing)
        _replay(softmax.lq, held, negative[1], tracing)
        if settings.q_s == 'round':
            _replay(softmax.iq, rounded[:1], negative[2], tracing)
        if settings.two_pass and settings.q_p == 'round':
            _replay(softmax.oq, served, negative[3], tracing)
        if settings.two_pass and settings.q_ii == 'round':
            _replay(softmax.inv_table.iq, rounded[1:], negative[4], tracing, repeats=rounds)
    if not settings.two_pass:
        context = context * softmax.inv_table(weight[..., None], training=training)
    return ops.transpose(context, (0, 2, 1, 3))


def attention(layer, query, key, value, attention_mask, training):
    """The attention output ``(B, T, heads, dv)`` of QMultiHeadAttention ``layer``'s call where `applies`, from the
    projections ``query`` ``(B, T, heads, hd)``, ``key`` ``(B, S, heads, hd)``, ``value`` ``(B, S, heads, dv)``. An
    inference call runs the forward kernel only. Training and tracing, JAX: under jax.checkpoint, autodiff keeps only the
    projections and the kernel's outputs and the backward recomputes the rest; the softmax's variable assignments apply
    after the call, as in hgq.layers.fsoftmax.scan_rounds."""
    softmax = layer._softmax
    if not (training is True or training == 'tracing') or keras.backend.backend() == 'torch':
        return _attention(softmax, query, key, value, attention_mask, training)
    import jax

    variables, written = softmax.variables, []

    def pure(reads, args):
        with keras.StatelessScope(list(zip(variables, reads))) as scope:
            out = _attention(softmax, *args, training)
        values = [scope.get_current_value(v) for v in variables]
        changed = [value is not read for value, read in zip(values, reads)]
        written[:] = [v for v, new in zip(variables, changed) if new]
        return out, tuple(value for value, new in zip(values, changed) if new)

    policy = jax.checkpoint_policies.save_only_these_names(_SAVED)
    out, values = jax.checkpoint(pure, policy=policy)(tuple(v.value for v in variables), (query, key, value, attention_mask))
    for variable, value in zip(written, values):
        variable.assign(value)
    return out
