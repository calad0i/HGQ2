import numpy as np
from da4ml.trace import FixedVariableArray
from da4ml.trace.ops import einsum

from hgq.layers import (
    QLinformerAttention,
    QMultiHeadAttention,
)

from ._base import ReplayOperationBase, mirror_quantizer
from .activation import ReplayQSoftmax
from .dense import ReplayQDense


def _compute_attention_mask(
    query,
    value,
    query_mask=None,
    value_mask=None,
    key_mask=None,
    attention_mask=None,
    use_causal_mask=False,
):
    masks = []
    if query_mask is not None:
        masks.append(np.expand_dims(query_mask, -1))  # [Q, 1]
    if value_mask is not None:
        masks.append(np.expand_dims(value_mask, -2))  # [1, V]
    if key_mask is not None:
        masks.append(np.expand_dims(key_mask, -2))  # [1, V]
    if use_causal_mask:
        q = query.shape[0]
        v = q if value is None else value.shape[0]
        masks.append(np.tril(np.ones((q, v), dtype='uint8')))  # [Q, V]
    masks.append(attention_mask)
    if not masks:
        return None

    if any(isinstance(m, FixedVariableArray) for m in masks):
        return np.prod(np.stack(masks, axis=0), axis=0)
    else:
        return None


def _masked_softmax(op, attention_scores, attention_mask=None):
    # Normalize the attention scores to probabilities.
    # attention_scores = [B, N, T, S]
    if attention_mask is not None:
        # The expand dim happens starting from the `num_heads` dimension,
        # (<batch_dims>, num_heads, <query_attention_dims,
        # key_attention_dims>)
        mask_expansion_axis = -len(op._attention_axes) * 2 - 1
        for _ in range(len(attention_scores.shape) - len(attention_mask.shape)):
            attention_mask = np.expand_dims(attention_mask, axis=mask_expansion_axis)
    return ReplayQSoftmax(op._softmax)(attention_scores[0], mask=attention_mask)[0][None]


def _compute_attention(op: QMultiHeadAttention, query, key, value, attention_mask=None, training=None):
    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    attention_scores = einsum(op._dot_product_equation, key, query)

    attention_scores = _masked_softmax(op, attention_scores, attention_mask)

    # `context_layer` = [B, T, N, H]
    attention_output = einsum(op._combine_equation, attention_scores, value)
    return attention_output, attention_scores


class ReplayMHA(ReplayOperationBase):
    handles = (QMultiHeadAttention,)
    __input_quantizer_handled__ = True
    __output_quantizer_handled__ = True

    def call(
        self,
        query: FixedVariableArray,
        value: FixedVariableArray,
        key=None,
        query_mask=None,
        value_mask=None,
        key_mask=None,
        attention_mask=None,
        return_attention_scores=False,
        use_causal_mask=False,
    ):
        op: QMultiHeadAttention = self.op

        if key is None:
            key = value

        _attention_mask = _compute_attention_mask(
            query,
            value,
            query_mask=query_mask,
            value_mask=value_mask,
            key_mask=key_mask,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
        )

        query = ReplayQDense(op._query_dense)(query)[0][None]
        key = ReplayQDense(op._key_dense)(key)[0][None]
        value = ReplayQDense(op._value_dense)(value)[0][None]

        attention_output, attention_scores = _compute_attention(op, query, key, value, _attention_mask)
        attention_output = ReplayQDense(op._output_dense)(attention_output[0])[0]

        if op.enable_oq:
            attention_output = mirror_quantizer(op.oq, attention_output)

        if return_attention_scores:
            return attention_output, attention_scores[0]
        return attention_output


class ReplayQLinformerAttention(ReplayMHA):
    handles = (QLinformerAttention,)

    def call(
        self,
        query,
        value,
        key=None,
        query_mask=None,
        value_mask=None,
        key_mask=None,
        attention_mask=None,
        return_attention_scores=False,
        use_causal_mask=False,
    ):
        assert use_causal_mask is False, 'Causal mask is not supported in QLinformerAttention.'
        key = key if key is not None else value
        op: QLinformerAttention = self.op
        key = ReplayQDense(op._lin_k_proj)(key)[0]
        value = ReplayQDense(op._lin_v_proj)(value)[0]
        return super().call(
            query,
            value,
            key,
            query_mask=query_mask,
            value_mask=value_mask,
            key_mask=key_mask,
            attention_mask=attention_mask,
            return_attention_scores=return_attention_scores,
        )


__all__ = ['ReplayMHA', 'ReplayQLinformerAttention']
