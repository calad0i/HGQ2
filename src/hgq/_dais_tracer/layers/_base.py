import typing
from collections.abc import Sequence
from typing import Any

import keras
import numpy as np
from da4ml.trace import FixedVariable, FixedVariableArray
from da4ml.trace.ops import quantize, relu
from keras.ops import convert_to_numpy

import hgq
from hgq.layers.core.base import MultipleQuantizers, Quantizer
from hgq.quantizer.internal import FixedPointQuantizerBase


def to_np_arr(x: Any) -> np.ndarray:
    return np.asarray(convert_to_numpy(x))


def mirror_quantizer(q: Quantizer, v: FixedVariableArray) -> FixedVariableArray:
    if q.scaler is not None:
        v = v * (1.0 / q.scaler)
    q_internal: FixedPointQuantizerBase = q.quantizer
    kk, ki, kf = q_internal.kif
    shape = (1,) + v.shape
    kk = q_internal.bw_mapper.bw_to_x(kk, shape)
    ki = q_internal.bw_mapper.bw_to_x(ki, shape)
    kf = q_internal.bw_mapper.bw_to_x(kf, shape)
    k, i, f = (to_np_arr(x).astype(np.int8)[0] for x in (kk, ki, kf))
    round_mode, overflow_mode = q_internal.round_mode, q_internal.overflow_mode
    rq = quantize(v, k, i, f, overflow_mode=overflow_mode, round_mode=round_mode)
    if q.affine:
        rq = rq * q.affine[0] + q.affine[1]
    return rq


_registry: dict[type, 'type[ReplayOperationBase]'] = {}


class HandlerRegMeta(type):
    """Metaclass for automatic registration of handler classes."""

    def __new__(mcs, name: str, bases: tuple[type, ...], namespace: dict[str, typing.Any]):
        cls = super().__new__(mcs, name, bases, namespace)
        if name == 'ReplayOperationBase':
            return cls

        handles: type | tuple[type, ...] = namespace['handles']
        if not isinstance(handles, tuple):
            handles = (handles,)

        for handle in handles:
            _registry[handle] = cls  # type: ignore
        return cls


class ReplayOperationBase(metaclass=HandlerRegMeta):
    handles: tuple[type, ...] = ()
    __activation_handled__ = False
    __input_quantizer_handled__ = False
    __output_quantizer_handled__ = False

    def __init__(self, layer: 'keras.Operation'):
        assert isinstance(layer, self.handles)
        self.op: Any = layer

    def call(self, *args, **kwargs) -> tuple[FixedVariableArray, ...] | FixedVariableArray: ...

    def __call__(self, *args, **kwargs) -> tuple[FixedVariableArray, ...]:
        assert all(not isinstance(a, FixedVariableArray) for a in kwargs.values())

        if not isinstance(self.op, hgq.layers.QLayerBase):
            r = self.call(*args, **kwargs)
            return r if isinstance(r, tuple) else (r,)

        layer: hgq.layers.QLayerBase = self.op
        assert kwargs.pop('training', False) is False, 'Training mode is not supported in mirror operation'
        assert kwargs.pop('mask', None) is None, 'Masking is not supported in mirror operation'

        if not self.__input_quantizer_handled__:
            assert len(args) == 1
            inputs = args[0]

            if layer.enable_iq:
                if isinstance(inputs, Sequence):
                    assert isinstance(layer.iq, MultipleQuantizers)
                    inputs = tuple(mirror_quantizer(q, v) for q, v in zip(layer.iq.quantizers, inputs))
                else:
                    assert isinstance(layer.iq, Quantizer), f'Expected iq to be a Quantizer, got {type(layer.iq)}'
                    inputs = mirror_quantizer(layer.iq, inputs)

            outputs = self.call(inputs, **kwargs)
        else:
            outputs = self.call(*args, **kwargs)
        if isinstance(outputs, FixedVariable):
            outputs = FixedVariableArray(np.array([outputs]))

        if not self.__activation_handled__:
            activation = getattr(layer, 'activation', keras.activations.linear)
            if activation is not keras.activations.linear:
                if activation is keras.activations.relu:
                    if isinstance(outputs, tuple):
                        assert len(outputs) == 1, 'ReLU activation is expected to have a single output'
                        outputs = (relu(outputs[0]),)
                    else:
                        outputs = relu(outputs)
                else:
                    raise NotImplementedError(f'Activation {activation} is not supported in mirror operation')

        if layer.enable_oq and not self.__output_quantizer_handled__:
            if isinstance(outputs, tuple):
                assert isinstance(layer.oq, MultipleQuantizers)
                outputs = tuple(mirror_quantizer(q, v) for q, v in zip(layer.oq.quantizers, outputs))
            else:
                assert isinstance(layer.oq, Quantizer)
                outputs = mirror_quantizer(layer.oq, outputs)

        if isinstance(outputs, (FixedVariableArray, np.ndarray)):
            outputs = (outputs,)

        return outputs


class ReplayQuantizer(ReplayOperationBase):
    handles = (Quantizer,)

    def __init__(self, op: 'Quantizer'):
        super().__init__(op)
        assert isinstance(op.quantizer, FixedPointQuantizerBase)

    def call(self, inputs: FixedVariableArray) -> FixedVariableArray:
        return mirror_quantizer(self.op, inputs)
