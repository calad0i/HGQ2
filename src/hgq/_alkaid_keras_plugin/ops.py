import numpy as np
from alkaid.converter.builtin.keras.layers._base import ReplayOperationBase
from alkaid.converter.builtin.keras.layers.ops import ReplayMerge
from alkaid.trace import FVArray
from alkaid.trace.ops import einsum

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

from ._base import QLayerMixin


class _QMerge(QLayerMixin, ReplayMerge):
    handles = (QAdd, QSubtract, QMultiply, QMaximum, QMinimum)


class _QAveragePow2(QLayerMixin, ReplayMerge):
    handles = (QAveragePow2,)

    def call(self, inputs: tuple[FVArray, ...]) -> FVArray:
        stacked: FVArray = np.stack(np.broadcast_arrays(*inputs), axis=0)  # type: ignore
        return np.sum(stacked, axis=0) * self.op._scale  # type: ignore


class _QEinsum(QLayerMixin, ReplayOperationBase):
    handles = (QEinsum, QDot)

    def call(self, *_inputs) -> FVArray:
        op = self.op
        if len(_inputs) == 1 and isinstance(_inputs[0], (tuple, list)):
            inputs = tuple(_inputs[0])
        else:
            inputs = _inputs
        assert len(inputs) == 2
        if isinstance(op, QEinsum):
            # Alkaid preserves the batch dim as size-1, and QEinsum.equation
            # already includes a batch letter — use inputs as-is.
            return einsum(op.equation, inputs[0], inputs[1])  # type: ignore
        # QDot: build the equation from shapes, mirroring Alkaid's handling of
        # keras.layers.Dot (shared batch letter at index 0).
        dim0, dim1 = inputs[0].ndim, inputs[1].ndim
        letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'[: dim0 + dim1 - 1]
        sub0 = letters[:dim0]
        _sub1 = list(sub0[0] + letters[dim0:])
        axes = list(op.axes) if not isinstance(op.axes, int) else [op.axes, op.axes]
        idx0, idx1 = axes[0] % dim0, axes[1] % dim1
        contracted = sub0[idx0]
        _sub1[idx1] = contracted
        sub1 = ''.join(_sub1)
        sub_out = ''.join(c for c in sub0 if c != contracted) + ''.join(c for c in sub1[1:] if c != contracted)
        eq = f'{sub0},{sub1}->{sub_out}'
        return einsum(eq, inputs[0], inputs[1])  # type: ignore


class _QReduction(QLayerMixin, ReplayOperationBase):
    handles = (QSum, QMeanPow2)

    def call(self, x: FVArray) -> FVArray:
        op = self.op
        return np.sum(x, axis=op.axes, keepdims=op.keepdims) * op.scale  # type: ignore
