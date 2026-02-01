import keras
from da4ml.trace import FixedVariableArray
from da4ml.trace.ops import einsum

from hgq.layers import (
    QBatchNormDense,
    QDense,
    QEinsumDense,
    QEinsumDenseBatchnorm,
)

from ._base import ReplayOperationBase, to_np_arr


class ReplayQDense(ReplayOperationBase):
    handles = (QDense, QEinsumDense, QEinsumDenseBatchnorm, QBatchNormDense, keras.layers.EinsumDense)

    def call(self, inputs: FixedVariableArray) -> FixedVariableArray:
        op = self.op
        if isinstance(op, (QDense, QBatchNormDense)):
            qkernel = op.qkernel
            qbias = op.qbias
            eq = '...c,cC->...C'
        elif isinstance(op, (QEinsumDense, QEinsumDenseBatchnorm)):
            qkernel = op.qkernel
            qbias = op.qbias
            eq = op.equation
        elif isinstance(op, keras.layers.EinsumDense):
            qkernel = op.kernel
            qbias = op.bias
            eq = op.equation
        else:
            raise TypeError(f'Unsupported layer type: {type(op)}')

        qkernel = to_np_arr(qkernel)
        qbias = to_np_arr(qbias) if qbias is not None else None
        return (einsum(eq, inputs[None], qkernel) + qbias)[0]


__all__ = ['ReplayQDense']
