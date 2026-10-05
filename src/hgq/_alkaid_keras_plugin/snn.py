import numpy as np
from alkaid.converter.builtin.keras.layers._base import to_np_arr
from alkaid.opsched.frontend import affine_scan, named, set_token_dim
from alkaid.trace import FVArray
from alkaid.trace.ops import quantize

from hgq.layers.snn import QLIF, QLIFMHA, QLIFCell, QSimpleSNN, QSimpleSNNCell

from ._base import QLayerMixin, mirror_quantizer
from .attn import _QMHA
from .rnn import _QRNNReplay


class _QSNNReplay(QLayerMixin, _QRNNReplay):
    handles = (QSimpleSNN, QLIF)
    __activation_handled__ = True
    __input_quantizer_handled__ = True
    __output_quantizer_handled__ = True

    def _fire(self, x: FVArray, state: FVArray):
        """One timestep's 1-bit pulses and the state they leave."""
        cell: QSimpleSNNCell = self.op.cell  # type: ignore
        leaked = to_np_arr(cell.qlif_beta) * state if isinstance(cell, QLIFCell) else state  # type: ignore
        membrane = leaked + (mirror_quantizer(cell.iq, x) if cell.enable_iq else x)
        score = membrane - to_np_arr(cell.threshold)
        fired = quantize(score > 0, 0, 1, 0)  # type: ignore

        if cell.reset_mechanism == 'subtract':
            kept = np.where(fired, score, membrane)
        elif cell.reset_mechanism == 'zero':
            kept = np.where(fired, 0.0, membrane)
        else:
            kept = membrane
        kept = mirror_quantizer(cell.sq, kept) if cell.enable_sq else kept  # type: ignore
        return fired, kept

    def _step(self, x: FVArray, state: FVArray):
        op: QSimpleSNN | QLIF = self.op  # type: ignore
        cell: QSimpleSNNCell = op.cell  # type: ignore
        fired, kept = self._fire(x, state)
        spikes = fired * to_np_arr(cell.qgraded_spikes_factor)
        if cell.enable_oq:
            spikes = mirror_quantizer(cell.oq, spikes)  # type: ignore
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


class _QLIFMHA(_QMHA):
    """QLIFMHA on :class:`_QMHA`'s fused and output projections, with :class:`_QSNNReplay` firing the pulses."""

    handles = (QLIFMHA,)

    def _pulse_train(self, op: QLIFMHA, lif: QLIF, projected):
        """A QLIF's 1-bit pulses over S"""
        batch, seq, heads, dim = np.shape(projected)
        replay = _QSNNReplay(lif)
        flat = np.reshape(projected, (batch, seq, heads * dim))
        pulses = affine_scan(replay._fire, flat, replay._init(None), name=f'{op.name}_{lif.name}')
        return np.reshape(pulses, (batch, seq, heads, dim))

    def _at_head(self, quantizer, value, head: int, axis: int = 1, lanes: int = 0):
        return set_token_dim(super()._at_head(quantizer, value, head, axis, lanes), -1)

    def _attend(self, op: QLIFMHA, query, key, value, gain: float, h: int):
        """One head's context: count coincident key and value pulses over the stream, contract every query token
        with the counts, and apply the constant gain once."""
        batch, _, key_dim = np.shape(key)
        value_dim = np.shape(value)[-1]

        def accumulate(token, count):
            k, v = token[:key_dim], token[key_dim:]
            return count + np.reshape(k[:, None] * v, -1)

        pairs = np.concatenate([key, value], axis=-1)
        counted = affine_scan(accumulate, pairs, np.zeros(key_dim * value_dim), name=f'{op.name}_count{h}')
        count = np.reshape(counted[:, -1, :], (batch, key_dim, value_dim))
        return named(np.einsum('bsk,bkv->bsv', query, count) * gain, f'{op.name}_context{h}')

    def call(self, inputs: FVArray, mask=None):  # type: ignore
        op: QLIFMHA = self.op  # type: ignore
        # always fused qkv proj
        projected = self._qkv(op, inputs, inputs, inputs, fuse='qkv')  # type: ignore
        lifs = (op._query_lif, op._key_lif, op._value_lif)
        query, key, value = (self._pulse_train(op, lif, x) for lif, x in zip(lifs, projected))
        if mask is not None:
            valid = mask[..., None, None]
            query, key = query * valid, key * valid
        # delayed gain application
        factors = [to_np_arr(lif.cell.qgraded_spikes_factor) for lif in lifs]
        assert all(np.ndim(factor) == 0 for factor in factors), f'{op.name}: graded_spikes_factor must be a scalar'
        gain = op.scale * np.prod(factors)
        heads = range(op.num_heads)
        contexts = [self._attend(op, query[..., h, :], key[..., h, :], value[..., h, :], gain, h) for h in heads]
        output = self._project(op, contexts)  # type: ignore
        return mirror_quantizer(op.oq, output) if op.enable_oq else output
