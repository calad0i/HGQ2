from typing import cast

import keras
import numpy as np
from alkaid.converter.builtin.keras.layers._base import ReplayOperationBase
from alkaid.trace import FVArray

from hgq.layers import QAffinedUnaryFunctionLUT, QFSoftmax, QSoftmax, QUnaryFunctionLUT

from ._base import QLayerMixin, mirror_quantizer

try:
    from alkaid.opsched.frontend import affine_scan, apply, configure, cut, scope, set_token_dim
except ImportError:
    raise RuntimeError('alkaid>=0.9.0beta1 is required for this version of hgq2. Please upgrade alkaid or install hgq2<0.3.')


class _QFunctionLUT(QLayerMixin, ReplayOperationBase):
    __activation_handled__ = True
    handles = (QUnaryFunctionLUT, QAffinedUnaryFunctionLUT)

    def call(self, x: FVArray) -> FVArray:
        op = self.op

        def activation(y: np.ndarray) -> np.ndarray:
            ky = keras.ops.convert_to_tensor(y[None])
            if isinstance(op, QAffinedUnaryFunctionLUT):
                ky = ky * op.scale + op.bias
            return keras.ops.convert_to_numpy(op.activation(ky)[0])  # type: ignore

        return apply(x, activation)  # type: ignore


class _QSoftmax(QLayerMixin, ReplayOperationBase):
    __activation_handled__ = True
    handles = (QSoftmax,)

    def call(self, inputs: FVArray, mask: FVArray | None = None) -> FVArray:
        op: QSoftmax = self.op  # type: ignore

        if op.stable:
            if mask is not None:
                low = np.min(inputs.lhs[0])
                inputs = np.where(mask, inputs, low)  # type: ignore
            inputs = np.amax(inputs, axis=op.axes, keepdims=True) - inputs  # type: ignore

        exp_inp = _QFunctionLUT(op.exp_table)(inputs)['final'][0]

        if mask is not None:
            exp_inp = mask * exp_inp

        sums = np.sum(exp_inp, axis=op.axes, keepdims=True)  # type: ignore
        divisor = _QFunctionLUT(op.inv_table)(sums)['final'][0]

        return exp_inp * divisor


class _QFSoftmax(QLayerMixin, ReplayOperationBase):
    __activation_handled__ = True
    handles = (QFSoftmax,)

    def call(self, inputs: FVArray, mask: FVArray | None = None) -> FVArray:
        op: QFSoftmax = self.op  # type: ignore
        order = tuple(axis for axis in range(inputs.ndim) if axis != op.axis) + (op.axis,)
        with scope(inputs, 'arrival'):
            sequence = set_token_dim(np.transpose(inputs, order)[..., None], -1)
        bits = [] if mask is None else [set_token_dim(np.transpose(mask, order)[..., None], -1)]
        exponential = _QFunctionLUT(op.exp_table)

        def statistics(token, peak, weight, seen):
            previous = peak
            peak = np.where(seen, np.maximum(peak, token), token)
            rescale = exponential(peak - previous)['final'][0]
            share = exponential(peak - token)['final'][0]
            weight = mirror_quantizer(op.lq, rescale * weight) + share
            return peak, weight, rescale, share

        if op.impl == '2pass':
            seen = np.broadcast_to((np.arange(inputs.shape[op.axis]) != 0)[:, None], sequence.shape)

            def stats(token, state):
                if mask is None:
                    return np.concatenate(statistics(token[:1], state[:1], state[1:2], token[1:2])[:2])
                # Masked, an earlier kept round is not the position but a third lane the first kept round sets.
                new = np.concatenate([*statistics(token[:1], state[:1], state[1:2], state[2:3])[:2], np.ones(1)])
                return np.where(token[1:2], new, state)

            scanned = np.concatenate([sequence, *(bits or [seen])], axis=-1)
            carried = affine_scan(stats, scanned, np.zeros(2 + len(bits)), name=op.name)
            configure(carried, parallel_firings=op.parallelization_factor)
            final = np.broadcast_to(carried[..., -1:, :2], (*sequence.shape[:-1], 2))

            def normalize(token):
                p = exponential(token[1:2] - token[:1])['final'][0] * _QFunctionLUT(op.inv_table)(token[2:3])['final'][0]
                return p if mask is None else np.where(token[3:4], p, 0)

            normalized = affine_scan(
                normalize, np.concatenate([sequence, final, *bits], axis=-1), None, name=f'{op.name}_normalize'
            )
            normalized = cut(normalized)
            configure(normalized, parallel_firings=op.parallelization_factor)
            return cast(FVArray, np.transpose(normalized[..., 0], np.argsort(order)))

        def cell(token, state):
            peak, weight, rescale, share = statistics(token[:1], state[:1], state[1:2], state[-1:])
            held = mirror_quantizer(op.aq, rescale * state[3:-1])
            new = np.concatenate([peak, weight, held, mirror_quantizer(op.aq, share), np.ones(1)])
            if mask is None:
                return new
            # The slots shift one lane a round, so a masked round still shifts, rescaling nothing, in an empty slot.
            return np.where(token[1:2], new, np.concatenate([state[:2], state[3:-1], np.zeros(1), state[-1:]]))

        scanned = np.concatenate([sequence, *bits], axis=-1) if bits else sequence
        carried = affine_scan(cell, scanned, np.zeros(inputs.shape[op.axis] + 3), name=op.name)
        configure(carried, parallel_firings=op.parallelization_factor)
        final = carried[..., -1, :]
        normalized = final[..., 2:-1] * _QFunctionLUT(op.inv_table)(final[..., 1:2])['final'][0]
        return cast(FVArray, np.transpose(normalized, np.argsort(order)))
