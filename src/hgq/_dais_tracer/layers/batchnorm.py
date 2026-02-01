import numpy as np
from da4ml.trace import FixedVariableArray

from hgq.layers import QBatchNormalization

from ._base import ReplayOperationBase


class ReplayQBatchNormalization(ReplayOperationBase):
    handles = (QBatchNormalization,)

    def call(self, inputs: FixedVariableArray) -> FixedVariableArray:
        layer: QBatchNormalization = self.op
        scale, bias = map(np.array, layer.qscaler_and_qoffset)
        shape = layer._shape[1:]
        return inputs * scale.reshape(shape) + bias.reshape(shape)
