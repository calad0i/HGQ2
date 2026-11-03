"""Static settings of hgq2's quantized attention kernels (attn.kernels; softmax '1pass' or '2pass'), shared by the JAX
(attn.jax_op) and torch (attn.torch_op) bindings. Pure Python: no torch / JAX / triton import."""

import os
from dataclasses import astuple, dataclass, fields
from math import prod

# Queries per program and warps per program (env HGQ_TRITON_ATTENTION_QUERY_BLOCK, HGQ_TRITON_ATTENTION_WARPS).
QUERY_BLOCK = int(os.environ.get('HGQ_TRITON_ATTENTION_QUERY_BLOCK', '32'))
WARPS = int(os.environ.get('HGQ_TRITON_ATTENTION_WARPS', '1'))


@dataclass(frozen=True)
class AttentionSettings:
    """The kernels' quantizer modes: (round, overflow, null_xgrad_on_zero_width) of the score (s), exp-table input (ei)
    and output (eo) quantizers' stateless calls, the round mode of the context (a) and weight (l) quantizers' roundings;
    the exp table's input scaler and the score quantizer's form (q_s: 'round', a trainable-WRAP rounding, or 'quantize').
    2-pass only (two_pass): the modes of the inverse table's input (ii) and output (io) quantizers, the forms of the
    inverse table's input (q_ii) and of the probability quantizer (q_p) and its modes (p), the inverse table's epsilon;
    the defaults stand for the 1-pass kernels, which do not read them."""

    rnd_s: str
    ovf_s: str
    null_s: bool
    rnd_ei: str
    ovf_ei: str
    null_ei: bool
    rnd_eo: str
    ovf_eo: str
    null_eo: bool
    rnd_a: str
    rnd_l: str
    scaler: float
    q_s: str = 'quantize'
    two_pass: bool = False
    rnd_ii: str = 'RND'
    ovf_ii: str = 'WRAP'
    null_ii: bool = False
    q_ii: str = 'quantize'
    rnd_io: str = 'RND'
    ovf_io: str = 'WRAP'
    null_io: bool = False
    q_p: str = 'round'
    rnd_p: str = 'RND'
    ovf_p: str = 'WRAP'
    null_p: bool = False
    epsilon: float = 0.0

    def args(self) -> tuple:
        return astuple(self)

    def consts(self) -> dict:
        """The kernels' mode constexprs (RND_S, OVF_S, ...)."""
        return {f.name.upper(): getattr(self, f.name) for f in fields(self) if f.name != 'scaler'}


def _modes(*quantizers) -> list:
    modes = []
    for q in quantizers:
        sq = q.quantizer.stateless_quantizer
        modes += [sq.round_mode, sq.overflow_mode, bool(sq.null_xgrad_on_zero_width)]
    return modes


def _form(quantizer) -> str:
    """'round' for a trainable-WRAP quantizer (its training call rounds only), else 'quantize'."""
    inner = quantizer.quantizer
    return 'round' if inner.overflow_mode == 'WRAP' and inner.trainable else 'quantize'


def attention_settings(softmax) -> AttentionSettings:
    """`AttentionSettings` of a QFSoftmax that attn.layer admits."""
    table = softmax.exp_table
    rounding = (q.quantizer.stateless_quantizer.round_mode for q in (softmax.aq, softmax.lq))
    two_pass = {}
    if softmax.impl == '2pass':
        from keras.config import epsilon

        modes = _modes(softmax.inv_table.iq, softmax.inv_table.oq, softmax.oq)
        names = ('rnd_ii', 'ovf_ii', 'null_ii', 'rnd_io', 'ovf_io', 'null_io', 'rnd_p', 'ovf_p', 'null_p')
        forms = dict(q_ii=_form(softmax.inv_table.iq), q_p=_form(softmax.oq))
        two_pass = dict(zip(names, modes), two_pass=True, **forms, epsilon=float(epsilon()))
    scaler = float(softmax.input_scaler)
    return AttentionSettings(*_modes(softmax.iq, table.iq, table.oq), *rounding, scaler=scaler, q_s=_form(softmax.iq), **two_pass)


def query_block(queries: int, block: int | None) -> int:
    """Queries per program: ``block`` (default `QUERY_BLOCK`), at most ``queries``."""
    return min(queries, block or QUERY_BLOCK)


def kernel_dims(query_shape, key_shape, value_shape, n_shape, qp_shape, block: int) -> dict[str, int]:
    """The kernels' compile-time dimensions (constexpr keyword arguments), from the operands' shapes and the query block.
    Every operand and output is contiguous (row-major), so the kernels derive their strides from these."""
    batch, queries, heads, depth = query_shape
    return dict(
        B=batch, T=queries, S=key_shape[1], H=heads, DK=depth, DV=value_shape[-1], NB=queries // block, W=qp_shape[1],
        NW=prod(n_shape[2:]), TB=block,
    )  # fmt: skip
