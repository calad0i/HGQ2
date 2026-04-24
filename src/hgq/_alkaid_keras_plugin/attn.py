import numpy as np
from alkaid.trace import FVArray

from hgq.layers.attn import (
    QLinformerAttention,
    QMultiHeadAttention,
    QSALTAttention,
)

from ._base import ReplayOperationBase, mirror_quantizer, to_np_arr
from .activation import _QSoftmax
from .core import _QConv, _QDense


def _fused_qkv_proj(query: FVArray, key: FVArray, value: FVArray, op: QMultiHeadAttention):
    query_qk, query_qb = op.query_dense.qkernel, op.query_dense.qbias
    query_qk = to_np_arr(query_qk)
    query_qb = to_np_arr(query_qb) if query_qb is not None else np.zeros_like(query_qk[0])

    key_qk, key_qb = op.key_dense.qkernel, op.key_dense.qbias
    key_qk = to_np_arr(key_qk)
    key_qb = to_np_arr(key_qb) if key_qb is not None else np.zeros_like(key_qk[0])

    value_qk, value_qb = op.value_dense.qkernel, op.value_dense.qbias
    value_qk = to_np_arr(value_qk)
    value_qb = to_np_arr(value_qb) if value_qb is not None else np.zeros_like(value_qk[0])

    if op._fuse == 'qkv':
        assert query is key and key is value, 'Fused QKV projection only works when query, key and value are the same.'
        iq = op.query_dense.iq if op.enable_iq else None
        query = mirror_quantizer(iq, query) if iq is not None else query

        to_QKV_kernel = np.concatenate([query_qk, key_qk, value_qk], axis=-1)
        to_QKV_bias = np.concatenate([query_qb, key_qb, value_qb], axis=-1)

        QKV = np.einsum(op.query_dense.equation, query, to_QKV_kernel) + to_QKV_bias
        Q, K, V = np.split(QKV, 3, axis=-1)  # type: ignore

        Q = mirror_quantizer(op._query_dense.oq, Q) if op._query_dense.oq else Q  # type: ignore
        K = mirror_quantizer(op._key_dense.oq, K) if op._key_dense.enable_oq else K  # type: ignore
        V = mirror_quantizer(op._value_dense.oq, V) if op._value_dense.enable_oq else V  # type: ignore

    elif op._fuse == 'kv':
        assert key is value, 'Fused KV projection only works when key and value are the same.'
        iq = op._key_dense.iq if op.enable_iq else None
        key = mirror_quantizer(iq, key) if iq is not None else key

        to_KV_kernel = np.concatenate([key_qk, value_qk], axis=-1)
        to_KV_bias = np.concatenate([key_qb, value_qb], axis=-1)
        KV = np.einsum(op._key_dense.equation, key, to_KV_kernel) + to_KV_bias
        K, V = np.split(KV, 2, axis=-1)  # type: ignore

        K = mirror_quantizer(op._key_dense.oq, K) if op._key_dense.enable_oq else K
        V = mirror_quantizer(op._value_dense.oq, V) if op._value_dense.enable_oq else V

        q_query = mirror_quantizer(op._query_dense.iq, query) if op._query_dense.iq is not None else query
        Q = np.einsum(op._query_dense.equation, q_query, query_qk) + query_qb
        Q = mirror_quantizer(op._query_dense.oq, Q) if op._query_dense.enable_oq else Q

    else:
        q_query = mirror_quantizer(op._query_dense.iq, query) if op._query_dense.iq is not None else query
        Q = np.einsum(op._query_dense.equation, q_query, query_qk) + query_qb
        Q = mirror_quantizer(op._query_dense.oq, Q) if op._query_dense.enable_oq else Q

        q_key = mirror_quantizer(op._key_dense.iq, key) if op._key_dense.iq is not None else key
        K = np.einsum(op._key_dense.equation, q_key, key_qk) + key_qb
        K = mirror_quantizer(op._key_dense.oq, K) if op._key_dense.enable_oq else K

        q_value = mirror_quantizer(op._value_dense.iq, value) if op._value_dense.iq is not None else value
        V = np.einsum(op._value_dense.equation, q_value, value_qk) + value_qb
        V = mirror_quantizer(op._value_dense.oq, V) if op._value_dense.enable_oq else V

    return Q, K, V


class ReplayMHA(ReplayOperationBase):
    handles = (QMultiHeadAttention,)
    __input_quantizer_handled__ = True
    __output_quantizer_handled__ = True

    def _compute_attention_mask(
        self,
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
            q = query.shape[1]
            v = q if value is None else value.shape[1]
            masks.append(np.tril(np.ones((1, q, v), dtype='uint8')))  # [1, Q, V]
        if attention_mask is not None:
            masks.append(attention_mask)
        if not masks:
            return None

        return np.prod(np.stack(masks, axis=0), axis=0)

    def _masked_softmax(self, op, attention_scores, attention_mask=None):
        # Normalize the attention scores to probabilities.
        # attention_scores = [B, N, T, S]
        if attention_mask is not None:
            # The expand dim happens starting from the `num_heads` dimension,
            # (<batch_dims>, num_heads, <query_attention_dims,
            # key_attention_dims>)
            mask_expansion_axis = -len(op._attention_axes) * 2 - 1
            for _ in range(len(attention_scores.shape) - len(attention_mask.shape)):
                attention_mask = np.expand_dims(attention_mask, axis=mask_expansion_axis)

        return _QSoftmax(op._softmax)(attention_scores, mask=attention_mask)['final'][0]

    def _compute_attention(self, op: QMultiHeadAttention, query, key, value, attention_mask=None, training=None):
        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        attention_scores = np.einsum(op._dot_product_equation, key, query)

        attention_scores = self._masked_softmax(op, attention_scores, attention_mask)

        # `context_layer` = [B, T, N, H]
        attention_output = np.einsum(op._combine_equation, attention_scores, value)
        return attention_output, attention_scores

    def call(
        self,
        query: FVArray,
        value: None | FVArray = None,
        key: None | FVArray = None,
        query_mask=None,
        value_mask=None,
        key_mask=None,
        attention_mask=None,
        return_attention_scores=False,
        use_causal_mask=False,
    ):
        op: QMultiHeadAttention = self.op

        value = query if value is None else value
        key = value if key is None else key

        _attention_mask = self._compute_attention_mask(
            query,
            value,
            query_mask=query_mask,
            value_mask=value_mask,
            key_mask=key_mask,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
        )

        query, key, value = _fused_qkv_proj(query, key, value, op)  # type: ignore

        attention_output, attention_scores = self._compute_attention(op, query, key, value, _attention_mask)
        attention_output = _QDense(op._output_dense)(attention_output)['final'][0]

        if op.enable_oq:
            attention_output = mirror_quantizer(op.oq, attention_output)

        if return_attention_scores:
            return attention_output, attention_scores
        return attention_output


class _QLinformerAttention(ReplayMHA):
    handles = (QLinformerAttention,)

    def call(
        self,
        query,
        value=None,
        key=None,
        query_mask=None,
        value_mask=None,
        key_mask=None,
        attention_mask=None,
        return_attention_scores=False,
        use_causal_mask=False,
    ):
        assert use_causal_mask is False, 'Causal mask is not supported in QLinformerAttention.'
        value = value if value is not None else query
        key = key if key is not None else value
        op: QLinformerAttention = self.op
        key = _QDense(op._lin_k_proj)(key)['final'][0]
        value = _QDense(op._lin_v_proj)(value)['final'][0]
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


class ReplaySALTAttention(ReplayMHA):
    handles = (QSALTAttention,)

    def _masked_softmax(self, op, attention_scores, attention_mask=None):
        self.op: QSALTAttention
        if self.op.conv_size > 0:
            attention_scores = _QConv(self.op.conv)(attention_scores)['final'][0]
        return super()._masked_softmax(op, attention_scores, attention_mask)

    def call(
        self,
        query,
        value=None,
        key=None,
        query_mask=None,
        value_mask=None,
        key_mask=None,
        attention_mask=None,
        return_attention_scores=False,
        use_causal_mask=False,
    ):
        assert use_causal_mask is False, 'Causal mask is not supported in QLinformerAttention.'
        value = value if value is not None else query
        key = key if key is not None else value
        op = self.op

        if self.op.cluster_k_proj:
            key = np.pad(key, [[0, 0], [0, self.op.n_k_pad], [0, 0]], mode='constant', constant_values=0)  # type: ignore
            key = np.reshape(key, self.op._k_reshape_to)  # type: ignore

        if self.op.n_v_pad > 0:
            value = np.pad(value, [[0, 0], [0, self.op.n_v_pad], [0, 0]], mode='constant', constant_values=0)  # type: ignore
            value = np.reshape(value, self.op._v_reshape_to)  # type: ignore
        key = _QDense(op._lin_k_proj)(key)['final'][0]
        value = _QDense(op._lin_v_proj)(value)['final'][0]
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


__all__ = ['ReplayMHA', '_QLinformerAttention', 'ReplaySALTAttention']
