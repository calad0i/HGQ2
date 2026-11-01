from keras import ops

from ...utils.misc import expand_ellipsis
from ..core import QLayerBaseMultiInputs


def _einsum_mask(equation: str, idx: int, mask):
    if mask is None:
        return None
    lhs, out = equation.split('->')
    n = len(mask.shape)
    pos = [out.find(ax) for ax in lhs.split(',')[idx][:n]]
    if sorted(pos) != list(range(n)):
        return None
    perm = tuple(pos.index(i) for i in range(n))
    return mask if perm == tuple(range(n)) else ops.transpose(mask, perm)


# class QMatmul(QLayerBaseMultiInputs):
#     def build(self, input_shape):
#         assert len(input_shape) == 2, 'QMatmul requires exactly 2 inputs.'
#         super().build(input_shape)

#     def call(self, inputs, training=None):
#         if self.enable_iq:
#             inputs = self.iq(inputs, training=training)
#         inp1, inp2 = inputs
#         return ops.matmul(inp1, inp2)

#     def _compute_ebops(self, *shapes):
#         bits0, bits1 = (iq.bits_(shape) for iq, shape in zip(self.iq, shapes))
#         ebops = ops.sum(ops.matmul(bits0, bits1))
#         return ebops


class QEinsum(QLayerBaseMultiInputs):
    def __init__(self, equation, **kwargs):
        super().__init__(**kwargs)
        self.equation = equation

    def build(self, input_shape):
        super().build(input_shape)
        self.equation = expand_ellipsis(self.equation, [len(shape) for shape in input_shape])
        self._ebops_equation = self.equation.split('->', 1)[0] + '->'

    def call(self, inputs, training=None):
        if self.enable_iq:
            inputs = self.iq(inputs, training=training)
        return ops.einsum(self.equation, *inputs)

    def compute_mask(self, inputs, previous_mask=None):
        if previous_mask is None:
            return None
        masks = [_einsum_mask(self.equation, i, m) for i, m in enumerate(previous_mask)]
        masks = [m for m in masks if m is not None]
        if not masks:
            return None
        ndim, mask = max(len(ops.shape(m)) for m in masks), None  # type: ignore
        for m in masks:
            while len(ops.shape(m)) < ndim:  # type: ignore
                m = ops.expand_dims(m, -1)
            mask = m if mask is None else ops.logical_and(mask, m)
        return mask

    def _compute_ebops(self, *shapes):
        bitss = [iq.bits_(shape) for iq, shape in zip(self.iq, shapes)]
        ebops = ops.einsum(self._ebops_equation, *bitss)
        return ebops

    def get_config(self):
        config = super().get_config()
        config['equation'] = self.equation
        return config
