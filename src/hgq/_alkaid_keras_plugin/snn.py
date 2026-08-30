import numpy as np
from alkaid.converter.builtin.keras.layers._base import to_np_arr
from alkaid.trace import FVArray
from alkaid.trace.ops import quantize

from hgq.layers.snn import QLIF, QLIFCell, QSimpleSNN, QSimpleSNNCell

from ._base import QLayerMixin, mirror_quantizer
from .rnn import _QRNNReplay


class _QSNNReplay(QLayerMixin, _QRNNReplay):
    handles = (QSimpleSNN, QLIF)
    __activation_handled__ = True
    __input_quantizer_handled__ = True
    __output_quantizer_handled__ = True

    def _step(self, x: FVArray, state: FVArray):
        op: QSimpleSNN | QLIF = self.op  # type: ignore
        cell: QSimpleSNNCell = op.cell  # type: ignore
        leaked = to_np_arr(cell.qlif_beta) * state if isinstance(cell, QLIFCell) else state  # type: ignore
        membrane = leaked + (mirror_quantizer(cell.iq, x) if cell.enable_iq else x)
        score = membrane - to_np_arr(cell.threshold)
        over = lambda v: (v > 0) * 1.0
        mapped = getattr(score, 'apply', None)
        fired = quantize(over(score) if mapped is None else mapped(over), 0, 1, 0)

        spikes = fired * to_np_arr(cell.qgraded_spikes_factor)
        if cell.enable_oq:
            spikes = mirror_quantizer(cell.oq, spikes)  # type: ignore
        if cell.reset_mechanism == 'subtract':
            kept = membrane - fired * to_np_arr(cell.threshold)
        elif cell.reset_mechanism == 'zero':
            kept = membrane * (1.0 - fired)
        else:
            kept = membrane
        kept = mirror_quantizer(cell.sq, kept) if cell.enable_sq else kept  # type: ignore
        return (np.concatenate([spikes, kept], axis=-1) if op.return_state else spikes), kept

    def call(self, inputs: FVArray, initial_state=None, mask=None):  # type: ignore
        op: QSimpleSNN | QLIF = self.op  # type: ignore
        cell: QSimpleSNNCell = op.cell  # type: ignore
        assert mask is None, 'Masked SNN replay is not supported.'
        assert not cell.inhibition, (
            f'{op.__class__.__name__} {op.name}: inhibition=True is not supported by the alkaid conversion'
        )

        answered = super().call(inputs, initial_state, mask)
        if not op.return_state:
            return answered
        emitted, last = answered
        return emitted[..., : cell.units], last[..., cell.units :]
