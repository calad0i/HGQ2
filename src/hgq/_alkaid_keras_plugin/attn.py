from typing import cast

import numpy as np
from alkaid.converter.builtin.keras.layers import ReplayOperationBase
from alkaid.trace import FVArray
from alkaid.trace.ops import quantize

from hgq.layers import QFSoftmax, QUnaryFunctionLUT
from hgq.layers.attn import QLinformerAttention, QLinformerAttentionT, QMultiHeadAttention, QMultiHeadAttentionT, QSALTAttention
from hgq.layers.core.base import Quantizer

from ._base import mirror_quantizer, to_np_arr
from .activation import _QFunctionLUT, _QSoftmax
from .core import _QConv, _QDense
from .table import _QEinsumDenseTable

try:
    from alkaid.opsched.frontend import affine_scan, cut, named, set_token_dim
except ImportError:
    raise RuntimeError('alkaid>=0.9.0beta1 is required for this version of hgq2. Please upgrade alkaid or install hgq2<0.3.')


class _QMHA(ReplayOperationBase):
    handles = (QMultiHeadAttention, QLinformerAttention)
    __activation_handled__ = True

    def _at_head(self, quantizer: Quantizer, value, head: int, axis: int = 1, lanes: int = 0):
        """Select a head's precision; a shared scan/table read has ``lanes`` and no query/key variation."""
        qi = quantizer.quantizer
        if lanes:
            precisions = []
            for bw in qi.kif:
                shape = tuple(bw.shape)
                precision = to_np_arr(bw).astype(np.int8).reshape(shape[1], -1, shape[-1] if shape[-1] == lanes else 1)
                assert (precision == precision[:, :1]).all(), (
                    'QMultiHeadAttention aq varies by query; shared precision needed. '
                    'Use softmax_aq_conf=QuantizerConfig(heterogeneous_axis=(-1,)).'
                    if quantizer is getattr(self.op._softmax, 'aq', None)
                    else f'{quantizer.name} varies by query/key; shared precision needed. Set heterogeneous_axis=(1,).'
                )
                precisions.append(np.broadcast_to(precision[:, 0], (self.op._num_heads, lanes))[head])
            return quantize(value, *precisions, overflow_mode=qi.overflow_mode, round_mode=qi.round_mode)
        if quantizer.scaler is not None:
            value = value * (1.0 / quantizer.scaler)
        stated = (1, *value.shape[:axis], self.op._num_heads, *value.shape[axis:])
        precisions = [to_np_arr(qi.bw_mapper.bw_to_x(bw, stated)).astype(np.int8)[0].take(head, axis=axis) for bw in qi.kif]
        landed = quantize(value, *precisions, overflow_mode=qi.overflow_mode, round_mode=qi.round_mode)
        return landed * quantizer.affine[0] + quantizer.affine[1] if quantizer.affine else landed

    def _table(self, table: QUnaryFunctionLUT, value, head: int):
        lanes = int(len(getattr(value, 'token_shape', value.shape)) < len(value.shape) or len(value.shape) == 1)
        replay = _QFunctionLUT(table)
        replay.__input_quantizer_handled__ = replay.__output_quantizer_handled__ = True
        value = self._at_head(table.iq, value, head, lanes=lanes)
        return self._at_head(table.oq, replay(value)['final'][0], head, lanes=lanes)

    def _qkv(self, op: QMultiHeadAttention, query: FVArray, key: FVArray, value: FVArray) -> tuple[FVArray, ...]:
        denses = (op._query_dense, op._key_dense, op._value_dense)
        if op._fuse == 'qkv':
            assert query is key and key is value, 'Fused QKV projection only works when query, key and value are the same.'
            groups = [((0, 1, 2), query, 'qkv')]
        elif op._fuse == 'kv':
            assert key is value, 'Fused KV projection only works when key and value are the same.'
            groups = [((1, 2), key, 'kv'), ((0,), query, 'query')]
        else:
            groups = [((0,), query, 'query'), ((1,), key, 'key'), ((2,), value, 'value')]
        outputs = [query, key, value]
        for indices, inputs, name in groups:
            dense = denses[indices[0]]
            if op.enable_iq or len(indices) == 1:
                inputs = mirror_quantizer(dense.iq, inputs) if dense.iq is not None else inputs
            kernels = [to_np_arr(denses[i].qkernel) for i in indices]
            biases = []
            for i, kernel in zip(indices, kernels):
                biases.append(to_np_arr(denses[i].qbias) if denses[i].qbias is not None else np.zeros_like(kernel[0]))
            projected = named(
                np.einsum(dense.equation, inputs, np.concatenate(kernels, axis=-1)) + np.concatenate(biases, axis=-1),
                f'{op.name}_{name}',
            )
            parts = np.split(projected, len(indices), axis=-1) if len(indices) > 1 else [projected]
            for i, part in zip(indices, parts):
                part = cast(FVArray, part)
                outputs[i] = mirror_quantizer(denses[i].oq, part) if denses[i].enable_oq else part
        return tuple(outputs)

    def _matrix_attention(self, op: QMultiHeadAttention, query, key, value, mask):
        softmax = op._softmax
        head = op._dot_product_equation.split(',')[1].split('->')[0][-2]
        scored = op._dot_product_equation.replace(head, '')
        combined = op._combine_equation.replace(head, '')
        axes = tuple(axis - 1 for axis in softmax.axes)
        contexts, attends = [], []
        for h in range(op._num_heads):
            scores = np.einsum(scored, key[..., h, :], query[..., h, :])
            if softmax.stable:
                if mask is not None:
                    scores = np.where(mask, scores, np.min(scores.lhs[0]) - 1)  # type: ignore
                scores = np.amax(scores, axis=axes, keepdims=True) - scores  # type: ignore
            weights = self._table(softmax.exp_table, scores, h)
            if mask is not None:
                weights = mask * weights
            divisor = self._table(softmax.inv_table, np.sum(weights, axis=axes, keepdims=True), h)  # type: ignore
            attends.append(self._at_head(softmax.oq, weights * divisor, h, 1))
            contexts.append(np.einsum(combined, attends[h], value[..., h, :]))
        return contexts, attends

    def _online_attention(self, op: QMultiHeadAttention, query, key, value, mask):
        assert op._fuse == 'none', f'{op.name}: fused qkv projection is not supported in flash attn impl'
        assert mask is None, "Scan attention does not support masks; drop the mask or set softmax='comb'."
        softmax: QFSoftmax = op._softmax
        head = op._dot_product_equation.split(',')[1].split('->')[0][-2]
        scored = op._dot_product_equation.replace(head, '')
        depth = np.shape(value)[-1]
        arriving, carrying = 1 + depth, 2 if softmax.impl == '2pass' else 2 + depth
        value = cut(value, f'{op.name}_value')
        lanes = []
        for h in range(op._num_heads):
            scores = self._at_head(softmax.iq, np.einsum(scored, key[..., h, :], query[..., h, :]), h, lanes=1)
            lanes += [scores[..., None], np.broadcast_to(value[..., h, :][:, None], (*np.shape(scores), depth))]
        arrival = named(np.concatenate(lanes, axis=-1), f'{op.name}_arrival')

        def statistics(score, running, weight, h):
            peak = np.maximum(running, score)
            rescale = self._table(softmax.exp_table, peak - running, h)
            share = self._table(softmax.exp_table, peak - score, h)
            return peak, self._at_head(softmax.lq, rescale * weight, h, lanes=1) + share, rescale, share

        def cell(token, state):
            lanes = []
            for h in range(op._num_heads):
                stride = 1 if softmax.impl == '2pass' else arriving
                score = token[h * stride : h * stride + 1]
                running = state[h * carrying : h * carrying + 1]
                weight = state[h * carrying + 1 : h * carrying + 2]
                peak, weight, rescale, share = statistics(score, running, weight, h)
                lanes += [peak, weight]
                if softmax.impl == '1pass':
                    served = token[h * arriving + 1 : (h + 1) * arriving]
                    output = state[h * carrying + 2 : (h + 1) * carrying]
                    lanes.append(
                        self._at_head(softmax.aq, rescale * output, h, lanes=depth)
                        + self._at_head(softmax.aq, share * served, h, lanes=depth)
                    )
            return np.concatenate(lanes, axis=-1)

        reach = sum(int(np.max(to_np_arr(dense.oq.quantizer.kif[1]))) for dense in (op._query_dense, op._key_dense))
        floor = -float(np.shape(query)[-1]) * 2.0**reach
        sequence = named(np.concatenate(lanes[::2], axis=-1), f'{op.name}_scores') if softmax.impl == '2pass' else arrival
        carried = affine_scan(cell, sequence, np.tile([floor, *np.zeros(carrying - 1)], op._num_heads), name=f'{op.name}_combine')

        if softmax.impl == '2pass':
            final = np.broadcast_to(carried[..., -1:, :], (*arrival.shape[:-1], 2 * op._num_heads))

            def accumulate(token, state):
                outputs = []
                for h in range(op._num_heads):
                    score = token[h * arriving : h * arriving + 1]
                    served = token[h * arriving + 1 : (h + 1) * arriving]
                    offset = op._num_heads * arriving + 2 * h
                    peak, weight = token[offset : offset + 1], token[offset + 1 : offset + 2]
                    share = self._table(softmax.exp_table, peak - score, h)
                    probability = self._at_head(softmax.oq, share * self._table(softmax.inv_table, weight, h), h, lanes=1)
                    output = state[h * depth : (h + 1) * depth]
                    outputs.append(
                        self._at_head(softmax.aq, output, h, lanes=depth)
                        + self._at_head(softmax.aq, probability * served, h, lanes=depth)
                    )
                return np.concatenate(outputs)

            carried = affine_scan(
                accumulate,
                np.concatenate([arrival, final], axis=-1),
                np.zeros(depth * op._num_heads),
                name=f'{op.name}_accumulate',
            )

        contexts = []
        for h in range(op._num_heads):
            if softmax.impl == '2pass':
                output = carried[..., -1, h * depth : (h + 1) * depth]
            else:
                weight = carried[..., -1, h * carrying + 1 : h * carrying + 2]
                output = carried[..., -1, h * carrying + 2 : (h + 1) * carrying]
                output = output * self._table(softmax.inv_table, weight, h)
            contexts.append(cut(output, f'{op.name}_context{h}'))
        return contexts, None

    def _project(self, op: QMultiHeadAttention, contexts):
        dense = op._output_dense
        kernel = to_np_arr(dense.qkernel)
        equation = dense.equation.replace(dense.equation.split(',')[0][-2], '')
        parts = []
        for h, context in enumerate(contexts):
            context = self._at_head(dense.iq, context, h, len(context.shape) - 1)
            parts.append(named(np.einsum(equation, context, kernel[h]), f'{op.name}_output{h}'))
        projected = cast(FVArray, sum(parts[1:], parts[0]))
        if dense.qbias is not None:
            projected = projected + to_np_arr(dense.qbias)
        return mirror_quantizer(dense.oq, projected) if dense.enable_oq else projected

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
        op = self.op
        value = query if value is None else value
        key = value if key is None else key
        if hasattr(op, '_lin_k_proj'):
            assert use_causal_mask is False, 'Causal mask is not supported in QLinformerAttention.'
            if getattr(op, 'cluster_k_proj', False):
                key = cast(FVArray, np.pad(key, ((0, 0), (0, op.n_k_pad), (0, 0))).reshape(op._k_reshape_to))
            if getattr(op, 'n_v_pad', 0) > 0:
                value = cast(FVArray, np.pad(value, ((0, 0), (0, op.n_v_pad), (0, 0))).reshape(op._v_reshape_to))
            project = _QEinsumDenseTable if getattr(op, '_lin_kv_proj_mode', 'dense') == 'dense_t' else _QDense
            key = cast(FVArray, project(op._lin_k_proj)(key)['final'][0])
            value = cast(FVArray, project(op._lin_v_proj)(value)['final'][0])
            key, value = set_token_dim(key, -1), set_token_dim(value, -1)
        masks = []
        for mask, axis in ((query_mask, -1), (value_mask, -2), (key_mask, -2)):
            if mask is not None:
                masks.append(np.expand_dims(mask, axis))
        if use_causal_mask:
            masks.append(np.tril(np.ones((1, query.shape[1], value.shape[1]), dtype='uint8')))
        if attention_mask is not None:
            masks.append(attention_mask)
        mask = np.prod(np.stack(masks), axis=0) if masks else None
        query, key, value = self._qkv(op, query, key, value)

        composed = self._online_attention if op._softmax_kind != 'comb' else self._matrix_attention
        contexts, attends = composed(op, query, key, value, mask)
        attention_output = self._project(op, contexts)

        if op.enable_oq:
            attention_output = mirror_quantizer(op.oq, attention_output)
        if not return_attention_scores:
            return attention_output
        assert attends is not None, "Scan attention returns no score matrix; set return_attention_scores=False or softmax='comb'."
        return attention_output, cast(FVArray, np.concatenate([np.expand_dims(attend, 1) for attend in attends], axis=1))


class _QMHAT(_QMHA):
    handles = (QMultiHeadAttentionT, QLinformerAttentionT)

    def _qkv(self, op: QMultiHeadAttentionT, query, key, value):
        denses = (op._query_dense, op._key_dense, op._value_dense)
        return tuple(_QEinsumDenseTable(dense)(x)['final'][0] for dense, x in zip(denses, (query, key, value)))

    def _project(self, op: QMultiHeadAttentionT, contexts):
        return _QEinsumDenseTable(op._output_dense)(np.stack(contexts, axis=-2))['final'][0]


class ReplaySALTAttention(_QMHA):
    handles = (QSALTAttention,)

    def _matrix_attention(self, op: QSALTAttention, query, key, value, mask):
        scores = np.einsum(op._dot_product_equation, key, query)
        if op.conv_size > 0:
            scores = _QConv(op.conv)(scores)['final'][0]
        if mask is not None:
            for _ in range(len(np.shape(scores)) - len(np.shape(mask))):
                mask = np.expand_dims(mask, axis=-len(op._attention_axes) * 2 - 1)
        attend = _QSoftmax(op._softmax)(scores, mask=mask)['final'][0]
        context = np.einsum(op._combine_equation, attend, value)
        heads = range(op._num_heads)
        return [context[..., h, :] for h in heads], [attend[:, h] for h in heads]
