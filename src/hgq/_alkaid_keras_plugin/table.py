"""Table-layer handlers. Structurally a direct port of ``hgq._dais_tracer``'s
``table.py`` — the only substantive differences are the module paths
(``da4ml.trace`` -> ``alkaid.trace``) and the batch-dim adjustments needed
because Alkaid carries a leading size-1 batch while da4ml strips it.
"""

from collections.abc import Callable
from math import prod, sqrt

import keras
import numpy as np
from alkaid.converter.builtin.keras.layers._base import ReplayOperationBase, to_np_arr
from alkaid.trace import FVariable, FVArray
from alkaid.trace.ops import _quantize, extract_patches
from keras import ops

from hgq.layers.table import QConvT1D, QConvT2D, QConvTBase, QDenseT
from hgq.quantizer.internal import FixedPointQuantizerBase

from ._base import QLayerMixin, mirror_quantizer


def keras_act_to_numpy(act: Callable) -> Callable:
    match act:
        case keras.activations.relu:
            return lambda x: np.maximum(0, x)
        case keras.activations.tanh:
            return np.tanh
        case keras.activations.softmax:
            raise ValueError('Non-local activation must not be used')
        case keras.activations.linear:
            return lambda x: x
        case keras.activations.sigmoid:
            return lambda x: 1 / (1 + np.exp(-x))
        case keras.activations.swish:
            return lambda x: x / (1 + np.exp(-x))
        case keras.activations.gelu:
            return lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
        case keras.activations.elu:
            return lambda x: np.where(x > 0, x, np.exp(x) - 1)
        case keras.activations.selu:
            alpha = 1.6732632423543772
            scale = 1.0507009873554805
            return lambda x: scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))
        case keras.activations.softplus:
            return lambda x: np.log1p(np.exp(x))
        case keras.activations.softsign:
            return lambda x: x / (1 + np.abs(x))
        case keras.activations.exponential:
            return lambda x: np.exp(x)
        case keras.activations.hard_silu:
            return lambda x: x * np.minimum(1, np.maximum(0, (x + 1) / 2))
        case _:
            return lambda x: ops.convert_to_numpy(act(ops.convert_to_tensor(x)))


def gather_weights_and_activation(model: keras.Sequential):
    ws: list[np.ndarray] = []
    bs: list[np.ndarray | None] = []
    acts: list[Callable[[np.ndarray], np.ndarray]] = []
    for layer in model.layers:
        layer: keras.layers.EinsumDense
        w, *b = layer.get_weights()
        act = keras_act_to_numpy(layer.activation)
        if len(b) != 0:
            assert len(b) == 1
            b = b[0]
        else:
            b = None
        if w.ndim == 3:
            w = w[..., None]
            if b is not None:
                b = b[..., None]
        ws.append(w)
        bs.append(b)
        acts.append(act)
    return ws, bs, acts


class _QDenseTable(QLayerMixin, ReplayOperationBase):
    handles = (QDenseT,)
    __input_quantizer_handled__ = True

    def call(self, inputs: FVArray) -> FVArray:
        op: QDenseT = self.op  # type: ignore

        out: FVArray = np.broadcast_to(inputs[..., None], inputs.shape + (op.n_out,))  # type: ignore
        out = mirror_quantizer(op.iq, out)

        l, h, s = out.lhs

        table_sizes: np.ndarray = np.round((h - l) / s).astype(np.uint32) + 1

        model = op.module

        ws, bs, acts = gather_weights_and_activation(model)

        out_shape: tuple[int, ...] = inputs.shape + (op.n_out,)
        tables: list[np.ndarray] = [None] * prod(out_shape)  # type: ignore
        n, loc = np.unique(table_sizes, return_inverse=True)

        work_dtype = op.dtype

        for i in range(n.size):
            mask: np.ndarray = loc == i
            _l, _h = l[mask], h[mask]
            inp = np.linspace(_l, _h, n[i], dtype=work_dtype)

            _out = inp[..., None]

            idxs = np.where(mask.ravel())[0]
            mask = mask.reshape(-1, *mask.shape[-2:])

            for w, b, act in zip(ws, bs, acts):
                w = np.concatenate([w[_mask] for _mask in mask], axis=0)
                if b is not None:
                    b = np.concatenate([b[_mask] for _mask in mask], axis=0)
                else:
                    b = 0
                _out = act(np.einsum('...ni,nij->...nj', _out, w, optimize='optimal') + b)
            _out = _out[..., 0]

            for j, idx in enumerate(idxs):
                tables[idx] = _out[..., j]

        if op.enable_bn:
            bn = op.bn_module
            beta: np.ndarray = ops.convert_to_numpy(bn.beta) if bn.center else 1  # type: ignore
            gamma: np.ndarray = ops.convert_to_numpy(bn.gamma) if bn.scale else 1  # type: ignore
            m_mean: np.ndarray = ops.convert_to_numpy(bn.moving_mean)  # type: ignore
            m_var: np.ndarray = ops.convert_to_numpy(bn.moving_variance)  # type: ignore
            epsilon = bn.epsilon
            scaler = gamma / np.sqrt(m_var + epsilon)
            offset = beta - m_mean * scaler

            for i in range(len(tables)):
                tables[i][:] = (tables[i] * scaler[i % op.n_out] + offset[i % op.n_out]) / sqrt(op.n_in)

        assert all(v is not None for v in tables), tables

        toq = op.toq
        toq_internal: FixedPointQuantizerBase = toq.quantizer
        kk, ki, kf = toq_internal.kif

        _shape = (1,) + out.shape
        kk = toq_internal.bw_mapper.bw_to_x(kk, _shape)
        ki = toq_internal.bw_mapper.bw_to_x(ki, _shape)
        kf = toq_internal.bw_mapper.bw_to_x(kf, _shape)

        k, i, f = map(lambda x: to_np_arr(x).astype(np.int32).ravel(), (kk, ki, kf))

        round_mode, overflow_mode = toq_internal.round_mode, toq_internal.overflow_mode
        round_mode = round_mode[2:] if round_mode.startswith('S_') else round_mode
        for arr, _k, _i, _f in zip(tables, k, i, f):
            arr[:] = _quantize(arr, _k, _i, _f, overflow_mode, round_mode)

        ret_vars: list[FVariable] = [None] * len(tables)  # type: ignore
        # Alkaid's ``FVArray`` IS an ``ndarray`` of ``FVariable`` (object dtype),
        # so plain indexing gives the variable without da4ml's ``._vars`` accessor.
        flat = out.ravel()
        for i in range(len(tables)):
            ret_vars[i] = flat[i].lookup(tables[i])
        out = FVArray(np.array(ret_vars).reshape(out_shape), out.solver_options, hwconf=out.hwconf)
        out = np.sum(out, axis=-2)  # type: ignore
        return out


class _QConvTable(_QDenseTable):
    handles = (QConvT2D, QConvT1D, QConvTBase)

    def call(self, inputs: FVArray):
        op: QConvTBase = self.op

        if op.rank == 1:
            inputs = inputs[..., None, :]

        inputs = extract_patches(inputs, **op.im2col_params)

        if op.rank == 1:
            inputs = inputs[..., 0, :]

        return super().call(inputs)
