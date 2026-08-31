import keras
import numpy as np
from alkaid.converter.builtin.keras.layers._base import ReplayOperationBase
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
    """The online softmax written round by round for FA1"""

    __activation_handled__ = True
    handles = (QFSoftmax,)

    def call(self, inputs: FVArray) -> FVArray:
        op: QFSoftmax = self.op  # type: ignore
        exponential = _QFunctionLUT(op.exp_table)
        lane = lambda i: inputs[(slice(None),) * op.axis + (slice(i, i + 1),)]

        m = lane(0)
        weight = np.zeros(m.shape)
        numerator = np.zeros(inputs.shape)
        for i, slot in enumerate(op._slots):
            score = lane(i)
            m, previous = np.maximum(m, score), m  # type: ignore
            rescale = exponential(m - previous)['final'][0]
            share = exponential(m - score)['final'][0]
            weight = mirror_quantizer(op.lq, rescale * weight) + share
            numerator = np.where(  # type: ignore
                slot, mirror_quantizer(op.aq, share), mirror_quantizer(op.aq, rescale * numerator)
            )

        return numerator * _QFunctionLUT(op.inv_table)(weight)['final'][0]  # type: ignore
