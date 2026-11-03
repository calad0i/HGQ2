"""Opt-in plain-Triton route of QMultiHeadAttention's 1-pass and 2-pass training attention on JAX and torch.

Switch (default off): hgq.triton_ops.enable(attention=True), or env HGQ_TRITON_ATTENTION=1; switching it on imports the
route (`check`) and raises if it is unavailable. The switch is read when the layer call runs (or is traced), so a jitted /
compiled function keeps the setting it was traced with (retrace to change it). Where `applies`, QMultiHeadAttention.call
takes `attention`: the kernels of attn.jax_op / attn.torch_op on the projections, the integer-bit replay of the rounding
(trainable WRAP) quantizers from the kernel's per-round partial maxima (the replay kernel, replay.jax_op / replay.torch_op,
one launch per quantizer) and, 1-pass, the normalization by the inverse table in keras ops (2-pass: in the kernel).

Fallback: with the switch on, a training call that cannot take the route runs the keras-ops scan, with a warning per
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


def _kind(q) -> str:
    """'round' for a trainable fixed-point KIF WRAP quantizer (its training call only rounds), 'quantize' for a stateless
    fixed-point KIF or KBI one, 'other' for any other (subclasses, scaler, noise mix, affine map, stochastic rounding)."""
    if type(q) is not Quantizer or q.scaler is not None or q.qnoise_factor is not None or q.affine is not None:
        return 'other'
    inner = q.quantizer
    if type(inner) not in (FixedPointQuantizerKIF, FixedPointQuantizerKBI) or inner._stochastic:
        return 'other'
    if inner.overflow_mode == 'WRAP' and inner.trainable:
        return 'round' if type(inner) is FixedPointQuantizerKIF else 'other'
    return 'quantize'


def _per_head(q, lanes: int) -> bool:
    """Bit-width shape (1, heads or 1, 1, ``lanes`` or 1)."""
    shape = q.quantizer._i.shape
    return len(shape) == 4 and shape[0] == shape[2] == 1 and shape[3] in (1, lanes)


def _on_cuda(x) -> bool:
    if keras.backend.backend() == 'torch':
        return bool(getattr(x, 'is_cuda', False))
    import jax

    default = jax.config.jax_default_device
    if isinstance(x, jax.Array) and not isinstance(x, jax.core.Tracer):
        devices = x.devices()
    elif isinstance(default, str):
        devices = jax.devices(default)[:1]
    else:
        devices = [default if default is not None else jax.devices()[0]]
    return all(d.platform == 'gpu' and 'cuda' in d.client.platform_version.lower() for d in devices)


def _gate(reason) -> bool:
    """Warn once per distinct reason that the keras-ops attention runs instead."""
    if reason not in _WARNED:
        _WARNED.add(reason)
        warnings.warn(f'hgq triton_attention: running the keras-ops attention ({reason})', stacklevel=3)
    return False


def applies(layer, query, key, value, training) -> bool:
    """Whether QMultiHeadAttention ``layer``'s call on the projections runs as `attention`: a training call with the switch
    on, of a 1-pass or 2-pass layer whose training call the kernels reproduce; a reason they do not is warned once.
    Admitted: one device (the kernel call is not split by batch like the ops around it); the (B, T, heads, hd)
    projections, attention over T; key, query, value-lane and key-depth counts powers of two; context and weight
    quantizers rounding, the score quantizer either kind, the exp table's quantizers stateless (`_kind`), bit widths at
    most per head (and value lane for the context); the exp table without EBOPs. 2-pass also: the inverse table without
    EBOPs, its output quantizer stateless, its input and the probability quantizer either kind, at most per head. The
    replayed (rounding) quantizers with the default bit-width mapper (the replay kernel reduces as it does) and no or a
    MinMax integer-bit constraint. Then placement on a CUDA GPU."""
    if training is not True or not triton_ops.enabled('attention') or layer._softmax_kind == 'comb' or layer._fuse != 'none':
        return False
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
    if not _kind(softmax.aq) == _kind(softmax.lq) == 'round':
        return _gate('context or weight quantizer not a trainable fixed-point KIF WRAP quantizer')
    if _kind(softmax.iq) == 'other':
        return _gate('score quantizer neither a stateless nor a trainable KIF WRAP fixed-point quantizer')
    if not _kind(table.iq) == _kind(table.oq) == 'quantize':
        return _gate('exp-table quantizer not a stateless fixed-point quantizer')
    if not (_per_head(softmax.aq, lanes) and all(_per_head(q, 1) for q in (softmax.lq, softmax.iq, table.iq, table.oq))):
        return _gate('bit widths finer than per head (and value lane for the context quantizer)')
    replayed = {'context': softmax.aq, 'weight': softmax.lq, 'score': softmax.iq}
    if softmax.impl == '2pass':
        inv, oq = softmax.inv_table, softmax.oq
        if not (inv.enable_iq and inv.enable_oq) or inv.enable_ebops:
            return _gate('inverse-table quantizer disabled, or inverse-table EBOPs on')
        if _kind(inv.oq) != 'quantize':
            return _gate('inverse-table output quantizer not a stateless fixed-point quantizer')
        if 'other' in (_kind(inv.iq), _kind(oq)):
            return _gate('inverse-table input or probability quantizer neither a stateless nor a trainable KIF WRAP quantizer')
        if not all(_per_head(q, 1) for q in (inv.iq, inv.oq, oq)):
            return _gate('inverse-table or probability bit widths finer than per head')
        replayed |= {'probability': oq, 'inverse-table input': inv.iq}
    for name, q in replayed.items():
        if _kind(q) != 'round':
            continue
        mapper, constraint = q.quantizer.bw_mapper, q.quantizer._i.constraint
        if type(mapper) is not DefaultBitwidthMapper:
            return _gate(f'{name} bit-width mapper {type(mapper).__name__} (the replay reduces as DefaultBitwidthMapper)')
        if constraint is not None and type(constraint) is not MinMax:
            return _gate(f'{name} integer-bit constraint {type(constraint).__name__} (the replay takes None or MinMax)')
    if not _on_cuda(query):
        return _gate('not placed on a CUDA GPU')
    return True


def _replay(quantizer, *maxima, repeats=1):
    """Assign a rounding quantizer the integer bits its training calls leave (FixedPointQuantizerKIF.call's transition
    per call), from the kernel's per-round partial abs-maxima ``(R, B, heads, NB[, lanes])`` of the round's call (two
    arrays: its first and second call), each call ``repeats`` times in a row (replay.torch_op's layout)."""
    if keras.backend.backend() == 'jax':
        from ..replay.jax_op import replay_integer_bits
    else:
        from ..replay.torch_op import replay_integer_bits
    q = quantizer.quantizer
    c = q._i.constraint  # None or MinMax (`applies`)
    i, f, decay, *maxima = (ops.stop_gradient(x) for x in (q._i.value, q.f, q.i_decay_speed, *maxima))
    bounds = None if c is None else (c.min_value, c.max_value)
    q._i.assign(replay_integer_bits(i, maxima, f, decay, bounds=bounds, repeats=repeats))


def _attention(softmax, query, key, value, attention_mask, training):
    """The rounds of the kernels (scores, score quantizer, running maximum from the first kept key's quantized score, exp
    table, weight and context recurrences; 2-pass also the second pass) and the replay of the rounding quantizers' integer
    bits from the kernel's per-round maxima; 1-pass, the normalization by the inverse table."""
    backend = keras.backend.backend()
    batch, queries, heads, _ = query.shape
    rounds, lanes = key.shape[1], value.shape[-1]
    settings, block = attention_settings(softmax), query_block(queries, None)
    nb = queries // block
    keep = ops.ones((batch, 1, rounds), 'bool') if attention_mask is None else ops.cast(attention_mask, 'bool')
    valid = ops.cast(ops.broadcast_to(ops.moveaxis(keep, -1, 0), (rounds, batch, queries)), 'float32')
    first = ops.cast(ops.broadcast_to(ops.argmax(ops.cast(keep, 'int32'), axis=-1), (batch, queries)), 'int32')
    # Keys to visit per batch element and query block: up to the last one kept for any of its queries.
    kept_any = ops.any(ops.reshape(valid, (rounds, batch, nb, block)) != 0.0, axis=3)
    n = ops.max(ops.where(kept_any, ops.arange(1, rounds + 1, dtype='int32')[:, None, None], 0), axis=0)
    n = ops.broadcast_to(n[..., None], (batch, nb, 16))
    quantizers = [softmax.iq, softmax.exp_table.iq, softmax.exp_table.oq, softmax.lq]
    if settings.two_pass:
        quantizers += [softmax.inv_table.iq, softmax.inv_table.oq, softmax.oq]
    kifs = [ops.broadcast_to(ops.cast(v, 'float32')[0, :, 0, 0], (heads,)) for q in quantizers for v in q.quantizer.kif]
    qp = ops.stack([*kifs, *[ops.zeros((heads,), 'float32')] * ((24 if settings.two_pass else 16) - len(kifs))], axis=1)
    akif = ops.stack([ops.broadcast_to(ops.cast(v, 'float32')[0, :, 0, :], (heads, lanes)) for v in softmax.aq.quantizer.kif])

    if backend == 'jax':
        from .jax_op import attention as kernel
    else:
        from .torch_op import attention as kernel

    context, weight, kept, added, held, served, rounded = kernel(query, key, value, qp, akif, valid, first, n, settings, block)
    context, weight = ops.swapaxes(context, 1, 2), ops.swapaxes(weight, 1, 2)  # (B, T, H, ...) -> (B, H, T, ...)
    if backend == 'jax':
        from jax.ad_checkpoint import checkpoint_name

        context, weight = checkpoint_name((context, weight), _SAVED)

    # Calls per round: context (kept, then added) and weight; the score once; 2-pass, probability and inverse-table input
    # (the same maxima every round).
    _replay(softmax.aq, kept, added)
    _replay(softmax.lq, held)
    if settings.q_s == 'round':
        _replay(softmax.iq, rounded[:1])
    if settings.two_pass and settings.q_p == 'round':
        _replay(softmax.oq, served)
    if settings.two_pass and settings.q_ii == 'round':
        _replay(softmax.inv_table.iq, rounded[1:], repeats=rounds)
    if not settings.two_pass:
        context = context * softmax.inv_table(weight[..., None], training=training)
    return ops.transpose(context, (0, 2, 1, 3))


def attention(layer, query, key, value, attention_mask, training):
    """The attention output ``(B, T, heads, dv)`` of QMultiHeadAttention ``layer``'s training call where `applies`, from
    the projections ``query`` ``(B, T, heads, hd)``, ``key`` ``(B, S, heads, hd)``, ``value`` ``(B, S, heads, dv)``. JAX:
    under jax.checkpoint, autodiff keeps only the projections and the kernel's outputs and the backward recomputes the
    rest; the softmax's variable assignments apply after the call, as in hgq.layers.fsoftmax.scan_rounds."""
    softmax = layer._softmax
    if keras.backend.backend() == 'torch':
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
