import keras
import numpy as np
from alkaid.converter.builtin.keras.layers._base import ReplayOperationBase
from alkaid.opsched.frontend import affine_scan, configure, scope, set_token_dim
from alkaid.trace import FVArray

from hgq.layers import QAffinedUnaryFunctionLUT, QFSoftmax, QSoftmax, QUnaryFunctionLUT

from ._base import QLayerMixin, mirror_quantizer


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

        return x.apply(activation)


class _QSoftmax(QLayerMixin, ReplayOperationBase):
    __activation_handled__ = True
    handles = (QSoftmax,)

    def call(self, inputs: FVArray, mask: FVArray | None = None) -> FVArray:
        op: QSoftmax = self.op  # type: ignore

        if op.stable:
            if mask is not None:
                low = np.min(inputs.lhs[0]) - 1
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

    def call(self, inputs: FVArray) -> FVArray:
        op: QFSoftmax = self.op  # type: ignore
        order = tuple(axis for axis in range(inputs.ndim) if axis != op.axis) + (op.axis,)
        with scope(inputs, 'arrival'):
            sequence = set_token_dim(np.transpose(inputs, order)[..., None], -1)
        exponential = _QFunctionLUT(op.exp_table)

        def cell(token, state):
            peak = np.where(state[-1:], np.maximum(state[:1], token), token)
            rescale = exponential(peak - state[:1])['final'][0]
            share = exponential(peak - token)['final'][0]
            weight = mirror_quantizer(op.lq, rescale * state[1:2]) + share
            held = mirror_quantizer(op.aq, rescale * state[3:-1])
            return np.concatenate([peak, weight, held, mirror_quantizer(op.aq, share), np.ones(1)])

        carried = affine_scan(cell, sequence, np.zeros(inputs.shape[op.axis] + 3), name=op.name)
        configure(carried, parallel_firings=op.parallelization_factor)
        final = carried[..., -1, :]
        normalized = final[..., 2:-1] * _QFunctionLUT(op.inv_table)(final[..., 1:2])['final'][0]
        return np.transpose(normalized, np.argsort(order))
