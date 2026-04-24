import keras
import numpy as np
from alkaid.converter.builtin.keras.layers._base import ReplayOperationBase
from alkaid.trace import FVArray

from hgq.layers import QAffinedUnaryFunctionLUT, QSoftmax, QUnaryFunctionLUT

from ._base import QLayerMixin


class _QFunctionLUT(QLayerMixin, ReplayOperationBase):
    __activation_handled__ = True
    handles = (QUnaryFunctionLUT, QAffinedUnaryFunctionLUT)

    def call(self, x: FVArray) -> FVArray:
        op = self.op

        def activation(y: np.ndarray) -> np.ndarray:
            ky = keras.ops.convert_to_tensor(y[None])
            if isinstance(op, QAffinedUnaryFunctionLUT):
                ky = ky * op.scale + op.bias
            return keras.ops.convert_to_numpy(op.activation(ky)[0])

        return x.apply(activation)


class _QSoftmax(QLayerMixin, ReplayOperationBase):
    __activation_handled__ = True
    handles = (QSoftmax,)

    def call(self, inputs: FVArray, mask: FVArray | None = None) -> FVArray:
        # Alkaid passes inputs with the batch dim preserved as size-1, unlike
        # da4ml which strips it. ``op.axes`` is expressed relative to the
        # full (batched) shape, so we apply it directly without prepending
        # an extra dim.
        op: QSoftmax = self.op

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
