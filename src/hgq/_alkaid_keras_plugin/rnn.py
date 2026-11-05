import numpy as np
from alkaid.converter.builtin.keras.layers._base import ReplayOperationBase, to_np_arr
from alkaid.converter.builtin.keras.layers.activation import keras_unary_to_numpy
from alkaid.trace import FVArray

from hgq.layers.rnn import QGRU, QGRUCell, QSimpleRNN, QSimpleRNNCell

from ._base import mirror_quantizer


class _QRNNReplay(ReplayOperationBase):
    handles = (QSimpleRNN, QGRU)
    __activation_handled__ = True

    @staticmethod
    def _zero_state(inputs: FVArray, units: int) -> FVArray:
        return FVArray(np.zeros((inputs.shape[0], units), dtype=np.float32), inputs.solver_options, hwconf=inputs.hwconf)

    @staticmethod
    def _state(initial_state, inputs: FVArray, units: int) -> FVArray:
        if initial_state is None:
            return _QRNNReplay._zero_state(inputs, units)
        if isinstance(initial_state, (tuple, list)):
            assert len(initial_state) == 1, 'QSimpleRNN and QGRU have exactly one recurrent state.'
            return initial_state[0]
        return initial_state

    def _step(self, x: FVArray, state: FVArray):
        raise NotImplementedError

    def call(self, inputs: FVArray, initial_state=None, mask=None):  # type: ignore
        op = self.op
        if mask is not None:
            raise NotImplementedError(f'{op.__class__.__name__} replay does not support masks.')
        if inputs.ndim != 3:
            raise ValueError(f'{op.__class__.__name__} replay expects rank-3 inputs, got shape {inputs.shape}.')

        state = self._state(initial_state, inputs, op.cell.units)
        outputs = []
        steps = range(inputs.shape[1] - 1, -1, -1) if op.go_backwards else range(inputs.shape[1])
        for t in steps:
            output, state = self._step(inputs[:, t, :], state)
            outputs.append(output)

        final_output = np.stack(outputs, axis=1) if op.return_sequences else outputs[-1]  # type: ignore
        if op.return_state:
            return final_output, state
        return final_output


class _QSimpleRNN(_QRNNReplay):
    handles = (QSimpleRNN,)

    def __init__(self, op: QSimpleRNN):
        super().__init__(op)
        cell: QSimpleRNNCell = op.cell  # type: ignore
        self.activation = keras_unary_to_numpy(cell.activation)

    def _step(self, x: FVArray, state: FVArray):
        cell: QSimpleRNNCell = self.op.cell  # type: ignore
        qx = mirror_quantizer(cell.iq, x)
        qstate = mirror_quantizer(cell.sq, state) if cell.enable_sq else state

        y = qx @ to_np_arr(cell.qkernel)
        if cell.use_bias:
            y = y + to_np_arr(cell.qbias)
        y = y + qstate @ to_np_arr(cell.qrecurrent_kernel)
        y = mirror_quantizer(cell.paq, y)
        y = self.activation(y)
        return y, y


class _QGRU(_QRNNReplay):
    handles = (QGRU,)

    def __init__(self, op: QGRU):
        super().__init__(op)
        cell: QGRUCell = op.cell  # type: ignore
        self.activation = keras_unary_to_numpy(cell.activation)
        self.recurrent_activation = keras_unary_to_numpy(cell.recurrent_activation)

    def _step(self, x: FVArray, state: FVArray):
        cell: QGRUCell = self.op.cell  # type: ignore
        units = cell.units
        qstate = mirror_quantizer(cell.sq, state)
        qx = mirror_quantizer(cell.iq, x) if cell.enable_iq else x

        if cell.use_bias:
            if cell.reset_after:
                input_bias, recurrent_bias = to_np_arr(cell.qbias)
            else:
                input_bias, recurrent_bias = to_np_arr(cell.qbias), 0
        else:
            input_bias, recurrent_bias = 0, 0

        matrix_x = qx @ to_np_arr(cell.qkernel) + input_bias
        x_zr, x_h = matrix_x[:, : 2 * units], matrix_x[:, 2 * units :]
        recurrent_kernel = to_np_arr(cell.qrecurrent_kernel)

        if cell.reset_after:
            matrix_inner = qstate @ recurrent_kernel + recurrent_bias
        else:
            matrix_inner = qstate @ recurrent_kernel[:, : 2 * units]

        zr = mirror_quantizer(cell.praq, self.recurrent_activation(x_zr + matrix_inner[:, : 2 * units]))
        z, r = np.split(zr, 2, axis=-1)  # type: ignore

        if cell.reset_after:
            recurrent_h = r * mirror_quantizer(cell.rhq, matrix_inner[:, 2 * units :])
        else:
            recurrent_h = mirror_quantizer(cell.rhq, r * qstate) @ recurrent_kernel[:, 2 * units :]  # type: ignore

        hh = mirror_quantizer(cell.paq, self.activation(x_h + recurrent_h))
        state = z * qstate + (1 - z) * hh  # type: ignore
        return state, state


__all__ = ['_QSimpleRNN', '_QGRU']
