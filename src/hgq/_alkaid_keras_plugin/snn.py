import numpy as np
from alkaid.converter.builtin.keras.layers._base import ReplayOperationBase, to_np_arr
from alkaid.trace import FVArray

from hgq.layers.snn import QLIF, QLIFCell, QSimpleSNN, QSimpleSNNCell

from ._base import QLayerMixin, mirror_quantizer


class _QSNNReplay(QLayerMixin, ReplayOperationBase):
    handles = (QSimpleSNN, QLIF)
    __activation_handled__ = True
    __input_quantizer_handled__ = True
    __output_quantizer_handled__ = True

    @staticmethod
    def _zero_state(inputs: FVArray, units: int) -> FVArray:
        state_shape = (*inputs.shape[:-2], units)
        return FVArray(np.zeros(state_shape, dtype=np.float32), inputs.solver_options, hwconf=inputs.hwconf)

    @staticmethod
    def _state(initial_state, inputs: FVArray, units: int) -> FVArray:
        if initial_state is None:
            return _QSNNReplay._zero_state(inputs, units)
        if isinstance(initial_state, (tuple, list)):
            assert len(initial_state) == 1, 'SNN cells have exactly one recurrent state.'
            return initial_state[0]
        return initial_state

    @staticmethod
    def _spike(cell: QSimpleSNNCell, state: FVArray):
        score = state - to_np_arr(cell.threshold)
        fire = score > 0
        if cell.inhibition:
            winners = []
            for i in range(cell.units):
                winner = fire[..., i]
                if i:
                    winner = winner & np.all(score[..., i, None] > score[..., :i], axis=-1)
                if i + 1 < cell.units:
                    winner = winner & np.all(score[..., i, None] >= score[..., i + 1 :], axis=-1)
                winners.append(winner)
            fire = np.stack(winners, axis=-1)
        return np.where(fire, to_np_arr(cell.qgraded_spikes_factor), 0.0), fire

    @staticmethod
    def _reset(cell: QSimpleSNNCell, state: FVArray, fire: FVArray):
        if cell.reset_mechanism == 'subtract':
            return np.where(fire, state - to_np_arr(cell.threshold), state)
        if cell.reset_mechanism == 'zero':
            return np.where(fire, 0.0, state)
        return state

    def _step(self, x: FVArray, state_tm1: FVArray):
        cell: QSimpleSNNCell = self.op.cell  # type: ignore
        x = mirror_quantizer(cell.iq, x) if cell.enable_iq else x
        if isinstance(cell, QLIFCell):
            state: FVArray = to_np_arr(cell.qlif_beta) * state_tm1 + x  # type: ignore
        else:
            state = state_tm1 + x
        output, fire = self._spike(cell, state)
        if cell.enable_oq:
            output = mirror_quantizer(cell.oq, output)  # type: ignore
        new_state = self._reset(cell, state, fire)  # type: ignore
        if getattr(cell, 'enable_sq', False):
            new_state = mirror_quantizer(cell.sq, new_state)  # type: ignore
        return output, new_state

    def call(self, inputs: FVArray, initial_state=None, mask=None):  # type: ignore
        op: QSimpleSNN | QLIF = self.op  # type: ignore
        assert mask is None, 'Masked SNN replay is not supported.'
        assert inputs.ndim in (2, 3), f'SNN replay expects rank-2 traced or rank-3 batched inputs, got shape {inputs.shape}.'

        time_axis = inputs.ndim - 2
        seq_len = inputs.shape[time_axis]
        assert seq_len is not None, 'SNN replay requires a fixed sequence length.'

        state = self._state(initial_state, inputs, op.cell.units)
        outputs = []
        steps = range(seq_len - 1, -1, -1) if op.go_backwards else range(seq_len)
        for t in steps:
            step_input = inputs[t, :] if inputs.ndim == 2 else inputs[:, t, :]
            output, state = self._step(step_input, state)  # type: ignore
            outputs.append(output)

        final_output = np.stack(outputs, axis=time_axis) if op.return_sequences else outputs[-1]  # type: ignore
        if op.return_state:
            return final_output, state
        return final_output
