import keras
import numpy as np
from da4ml.trace import FixedVariableArray
from da4ml.trace.ops import relu
from keras.layers import LeakyReLU, PReLU, ReLU

from hgq.layers import (
    QAffinedUnaryFunctionLUT,
    QSoftmax,
    QUnaryFunctionLUT,
)

from ._base import ReplayOperationBase, to_np_arr


class ReplayReLU(ReplayOperationBase):
    handles = (ReLU, LeakyReLU, PReLU)

    def call(self, inputs: FixedVariableArray) -> FixedVariableArray:
        op = self.op
        if isinstance(op, ReLU):
            th, neg, maxv = op.threshold, op.negative_slope, op.max_value
        elif isinstance(op, LeakyReLU):
            th, neg, maxv = 0, op.negative_slope, None
        elif isinstance(op, PReLU):
            th, neg, maxv = 0, to_np_arr(op.alpha), None
        else:
            raise TypeError(f'Unsupported activation layer: {type(op)}')

        if th == 0 and np.all(neg == 0) and maxv is None:
            return relu(inputs)

        pos_part = inputs if maxv is None else np.minimum(inputs, maxv)  # type: ignore
        pos_part = pos_part._vars.ravel()

        if th != 0:
            z_cond = (inputs - (th + 2.0 ** (-inputs.kif[2] - 1)))._vars.ravel()
        else:
            z_cond = inputs._vars.ravel()

        neg_part = ((inputs[None] - th) * neg)._vars.ravel()
        out = np.array([c.msb_mux(n, p) if c.low < 0 else p for c, n, p in zip(z_cond, neg_part, pos_part)])

        return FixedVariableArray(out.reshape(inputs.shape), inputs.solver_options)


class ReplayQFunctionLUT(ReplayOperationBase):
    __activation_handled__ = True
    handles = (QUnaryFunctionLUT, QAffinedUnaryFunctionLUT)

    def call(self, x: FixedVariableArray) -> FixedVariableArray:
        op: QUnaryFunctionLUT = self.op

        def activation(x) -> np.ndarray:
            kx = keras.ops.convert_to_tensor(x[None])
            if isinstance(op, QAffinedUnaryFunctionLUT):
                kx = kx * op.scale + op.bias
            kx = op.activation(kx)
            return keras.ops.convert_to_numpy(kx[0])  # type: ignore

        return x.apply(activation)


class ReplayQSoftmax(ReplayOperationBase):
    handles = (QSoftmax,)

    def call(self, inputs: FixedVariableArray, mask: None | FixedVariableArray = None) -> FixedVariableArray:
        op: QSoftmax = self.op
        inputs = inputs[None]

        if op.stable:
            inputs = np.amax(inputs, axis=op.axes, keepdims=True) - inputs  # type: ignore

        exp_inp = ReplayQFunctionLUT(op.exp_table)(inputs[0])[0]

        if mask is not None:
            exp_inp = mask[0] * exp_inp

        sums = np.sum(exp_inp[None], axis=op.axes, keepdims=True)[0]  # type: ignore
        divisor = ReplayQFunctionLUT(op.inv_table)(sums)[0]

        return exp_inp * divisor


__all__ = ['ReplayReLU', 'ReplayQFunctionLUT', 'ReplayQSoftmax']
