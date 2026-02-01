from collections.abc import Sequence

import keras
import numpy as np
from da4ml.trace import FixedVariableArray
from da4ml.trace.ops import einsum
from keras.src.ops.numpy import (
    Abs,
    Absolute,
    Add,
    Concatenate,
    Divide,
    Dot,
    Einsum,
    GetItem,
    Matmul,
    Max,
    Maximum,
    Min,
    Minimum,
    Moveaxis,
    Multiply,
    Ravel,
    Repeat,
    Reshape,
    Subtract,
    Sum,
    Transpose,
    TrueDivide,
)

from hgq.layers import (
    QAdd,
    QAveragePow2,
    QDot,
    QEinsum,
    QMaximum,
    QMeanPow2,
    QMinimum,
    QMultiply,
    QSubtract,
    QSum,
)

from ._base import ReplayOperationBase


class ReplayReshape(ReplayOperationBase):
    handles = (keras.layers.Reshape, keras.layers.Flatten, Reshape, Ravel)

    def call(self, inputs: FixedVariableArray) -> FixedVariableArray:
        if isinstance(self.op, (keras.layers.Flatten, Ravel)):
            return inputs.ravel()
        elif isinstance(self.op, keras.layers.Reshape):
            return inputs.reshape(self.op.target_shape)
        elif isinstance(self.op, Reshape):
            return inputs.reshape(self.op.newshape[1:])
        else:
            raise TypeError(f'Unsupported layer type: {type(self.op)}')


class ReplayMerge(ReplayOperationBase):
    handles = (keras.layers.Add, keras.layers.Concatenate, QAdd, QMultiply, QSubtract, QMaximum, QMinimum, QAveragePow2)

    def call(self, inputs: tuple[FixedVariableArray, ...]) -> FixedVariableArray:
        op = self.op
        name = op.__class__.__name__
        if name.startswith('Q'):
            name = name[1:]
        _inputs: FixedVariableArray = np.stack(np.broadcast_arrays(*inputs), axis=0)  # type: ignore
        match name:
            case 'Add':
                return np.sum(_inputs, axis=0)  # type: ignore
            case 'AveragePow2':
                return np.sum(_inputs, axis=0) * op._scale  # type: ignore
            case 'Subtract':
                assert len(_inputs) == 2, 'Subtract operation requires exactly two inputs'
                return _inputs[0] - _inputs[1]
            case 'Multiply':
                return np.prod(_inputs, axis=0)  # type: ignore
            case 'Maximum':
                return np.amax(_inputs, axis=0)  # type: ignore
            case 'Minimum':
                return np.amin(_inputs, axis=0)  # type: ignore
            case 'Concatenate':
                return np.concatenate(_inputs, axis=op.axis)  # type: ignore

            case _:
                raise TypeError(f'Unsupported layer type: {type(op)}')


class ReplayRepeatVector(ReplayOperationBase):
    handles = (keras.layers.RepeatVector,)

    def call(self, inputs: FixedVariableArray) -> FixedVariableArray:
        layer: keras.layers.RepeatVector = self.op
        if layer.n == 1:
            return inputs
        # return FixedVariableArray(np.repeat(inputs._vars, layer.n, axis=0), inputs.solver_options)
        return np.repeat(inputs[None], layer.n, axis=0)[0]  # type: ignore


class ReplayGetItem(ReplayOperationBase):
    handles = (GetItem,)

    def call(self, x: FixedVariableArray, key):
        if isinstance(key, list):
            key = tuple(key)
        return x[None][key][0]


class ReplayReduction(ReplayOperationBase):
    handles = (Sum, Max, Min)

    def call(self, x: FixedVariableArray, axis=None, keepdims=False):
        if isinstance(self.op, Sum):
            op = np.sum
        elif isinstance(self.op, Max):
            op = np.amax
        elif isinstance(self.op, Min):
            op = np.amin
        return op(x[None], axis=axis, keepdims=keepdims)[0]  # type: ignore


class ReplayQReduction(ReplayOperationBase):
    handles = (QSum, QMeanPow2)

    def call(self, x: FixedVariableArray):
        layer: QSum = self.op
        axes, scale, keepdims = layer.axes, layer.scale, layer.keepdims
        return np.sum(x[None], axis=axes, keepdims=keepdims)[0] * scale  # type: ignore


class ReplayArithmetic(ReplayOperationBase):
    handles = (Add, Subtract, Multiply, TrueDivide, Divide, Maximum, Minimum)

    def call(self, x1: FixedVariableArray, x2: FixedVariableArray):
        name = self.op.__class__.__name__
        match name:
            case 'Add':
                return x1 + x2
            case 'Subtract':
                return x1 - x2
            case 'Multiply':
                return x1 * x2
            case 'TrueDivide' | 'Divide':
                return x1 / x2
            case 'Maximum':
                return np.maximum(x1, x2)  # type: ignore
            case 'Minimum':
                return np.minimum(x1, x2)  # type: ignore
            case _:
                raise TypeError(f'Unsupported arithmetic operation: {type(self.op)}')


class ReplayConcatenate(ReplayOperationBase):
    handles = (Concatenate,)

    def call(self, xs: Sequence[FixedVariableArray]):
        axis = self.op.axis
        # return backend.numpy.concatenate(xs, axis=self.axis)
        # return FixedVariableArray(np.concatenate([x._vars[None] for x in xs], axis=axis)[0], xs[0].solver_options)
        return np.concatenate([x[None] for x in xs], axis=axis)[0]  # type: ignore


class ReplayRepeat(ReplayOperationBase):
    handles = (Repeat,)

    def call(self, x: FixedVariableArray):
        repeats, axis = self.op.repeats, self.op.axis
        # return FixedVariableArray(np.repeat(x._vars[None], repeats, axis=axis)[0], x.solver_options)
        return np.repeat(x[None], repeats, axis=axis)[0]  # type: ignore


class ReplayTranspose(ReplayOperationBase):
    handles = (Transpose,)

    def call(self, x: FixedVariableArray):
        axes = self.op.axes
        return np.transpose(x, axes)  # type: ignore


class ReplayMoveaxis(ReplayOperationBase):
    handles = (Moveaxis,)

    def call(self, x: FixedVariableArray):
        source, destination = self.op.source, self.op.destination
        return np.moveaxis(x[None], source, destination)[0]  # type: ignore


class ReplayNoOp(ReplayOperationBase):
    __noop_layers = []
    for k, v in keras.layers.__dict__.items():
        name = k.lower()
        if 'dropout' in name or 'random' in name or 'noise' in name:
            __noop_layers.append(v)

    handles = tuple(__noop_layers)

    def call(self, x: FixedVariableArray, training=False) -> FixedVariableArray:
        assert not training, 'Training mode is not supported in mirror operation'
        return x


class ReplayEinsum(ReplayOperationBase):
    handles = (QEinsum, Einsum, QDot, keras.layers.Dot)

    def call(self, *_inputs: tuple[FixedVariableArray, FixedVariableArray] | FixedVariableArray) -> FixedVariableArray:
        op = self.op
        inputs: tuple[FixedVariableArray, FixedVariableArray]
        if isinstance(_inputs[0], tuple):
            assert len(_inputs) == 1, 'Einsum with multiple input tuples is not supported'
            inputs = _inputs[0]
        else:
            inputs = _inputs  # type: ignore
        assert len(inputs) == 2, 'Only (Q)Einsum operations with exactly two inputs are supported'
        if isinstance(op, QEinsum):
            eq = op.equation
        elif isinstance(op, Einsum):
            eq = op.subscripts
        else:  # QDot/Dot
            dim0, dim1 = inputs[0].ndim + 1, inputs[1].ndim + 1
            letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'[0 : dim0 + dim1]
            sub0, sub1 = letters[:dim0], letters[dim0 : dim0 + dim1]
            axes = list(op.axes) if not isinstance(op.axes, int) else [op.axes, op.axes]
            idx0, idx1 = axes[0] if axes[0] >= 0 else axes[0] % dim0, axes[1] if axes[1] >= 0 else axes[1] % dim1
            sub1 = sub1[:idx1] + sub0[idx0] + sub1[idx1 + 1 :]
            sub_out = list(sub0 + sub1)
            sub_out.remove(sub0[idx0])
            sub_out.remove(sub0[idx0])
            sub_out = ''.join(sub_out)
            eq = f'{sub0},{sub1}->{sub_out}'
        return einsum(eq, inputs[0][None], inputs[1][None])[0]


class ReplayMatmul(ReplayOperationBase):
    handles = (Matmul, Dot)

    def call(self, x1: FixedVariableArray, x2: FixedVariableArray) -> FixedVariableArray:
        return x1 @ x2


class ReplayAbs(ReplayOperationBase):
    handles = (Absolute, Abs)

    def call(self, x: FixedVariableArray) -> FixedVariableArray:
        return np.abs(x)  # type: ignore
