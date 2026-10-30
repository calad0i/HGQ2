from keras import ops

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
        equation = self.equation
        inp_idxs = [idx.strip() for idx in equation.split('->', 1)[0].split(',')]
        out_idx = equation.split('->', 1)[1].strip()
        inp_ndims = [len(shape) for shape in input_shape]
        assert len(input_shape) == len(inp_idxs), f'Expected {len(inp_idxs)} inputs, but got {len(input_shape)}.'

        wildcard_ndim = None
        available_idxs = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        for inp_idx, ndim in zip(inp_idxs, inp_ndims):
            if len(inp_idx) != ndim:
                assert '...' in inp_idx
                if wildcard_ndim is None:
                    wildcard_ndim = ndim - (len(inp_idx) - 3)
                    assert wildcard_ndim >= 0, 'Invalid ellipsis notation: expanded to negative dimensions.'
                assert wildcard_ndim == ndim - (len(inp_idx) - 3), 'Inconsistent ellipsis expansion dimensions across inputs.'
                available_idxs -= set(inp_idx.replace('...', ''))

        if wildcard_ndim is not None:
            rep_indices = list(available_idxs)[:wildcard_ndim]
            rep_indices.sort()
            rep_indices = ''.join(rep_indices)
            for i, inp_idx in enumerate(inp_idxs):
                inp_idxs[i] = inp_idx.replace('...', rep_indices)
            out_idx = out_idx.replace('...', rep_indices)
        self._ebops_equation = ','.join(inp_idxs) + '->'
        self.equation = self._ebops_equation + out_idx

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
