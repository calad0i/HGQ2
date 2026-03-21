from da4ml.trace import FixedVariableArray
from keras import ops

from hgq.layers import QBatchNormalization

from ._base import ReplayOperationBase


class ReplayQBatchNormalization(ReplayOperationBase):
    handles = (QBatchNormalization,)

    def call(self, inputs: FixedVariableArray, mask=None) -> FixedVariableArray:
        layer: QBatchNormalization = self.op
        scale, bias = map(ops.convert_to_numpy, layer.qscaler_and_qoffset)
        shape = layer._shape[1:]
        return inputs * scale.reshape(shape) + bias.reshape(shape)  # type: ignore
