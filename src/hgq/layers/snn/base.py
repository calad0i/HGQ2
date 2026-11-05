from __future__ import annotations

from warnings import warn

import numpy as np
from keras import ops
from keras.initializers import Constant
from keras.layers import Layer
from keras.saving import deserialize_keras_object, register_keras_serializable, serialize_keras_object
from keras.src.layers.input_spec import InputSpec

from ...config import QuantizerConfig
from ...config.layer import global_config
from ...constraints import MinMax
from ...quantizer import Quantizer
from ..core.base import QLayerBase, QLayerBaseSingleInput
from ..rnn.simple_rnn import QRNN


@register_keras_serializable(package='hgq')
class ATan:
    """Arc-tangent surrogate gradient for a Heaviside spike function."""

    def __init__(self, alpha: float = 2.0):
        self.alpha = float(alpha)

    def __call__(self, inputs):
        dtype = ops.dtype(inputs)
        alpha = ops.cast(self.alpha, dtype)
        pi = ops.cast(np.pi, dtype)
        hard = ops.cast(inputs > 0, dtype)
        soft = ops.arctan(pi / 2.0 * alpha * inputs) / pi + 0.5
        return ops.stop_gradient(hard - soft) + soft

    def get_config(self):
        return {'alpha': self.alpha}


def atan(alpha: float = 2.0):
    """Return an arc-tangent surrogate gradient callable."""
    return ATan(alpha=alpha)


@register_keras_serializable(package='hgq')
class SpikingNeuralCell(Layer):
    """Integrate-and-fire RNN cell."""

    reset_mechanisms = ('subtract', 'zero', 'none')

    def __init__(
        self,
        units: int,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        inhibition=False,
        learn_threshold=False,
        reset_mechanism='subtract',
        detach_reset=True,
        graded_spikes_factor=1.0,
        learn_graded_spikes_factor=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if units <= 0:
            raise ValueError('units must be a positive integer.')
        if reset_mechanism not in self.reset_mechanisms:
            raise ValueError("reset_mechanism must be set to either 'subtract', 'zero', or 'none'.")
        if inhibition:
            warn(
                'Inhibition is an unstable feature that has only been tested '
                'for dense (fully-connected) layers. Use with caution!',
                UserWarning,
            )

        threshold_value = np.asarray(threshold)
        graded_spikes_factor_value = np.asarray(graded_spikes_factor)
        self.units = int(units)
        self.state_size = self.units
        self.output_size = self.units
        self._threshold_init = threshold
        self._threshold_config = threshold_value.item() if threshold_value.shape == () else threshold_value.tolist()
        self._graded_spikes_factor_init = graded_spikes_factor
        self._graded_spikes_factor_config = (
            graded_spikes_factor_value.item() if graded_spikes_factor_value.shape == () else graded_spikes_factor_value.tolist()
        )
        self._spike_grad_arg = spike_grad
        self.surrogate_disable = surrogate_disable
        self.inhibition = inhibition
        self.learn_threshold = learn_threshold
        self.reset_mechanism = reset_mechanism
        self.detach_reset = detach_reset
        self.learn_graded_spikes_factor = learn_graded_spikes_factor

        if surrogate_disable:
            self.spike_grad = self._surrogate_bypass
        elif spike_grad is None:
            self.spike_grad = atan()
        else:
            self.spike_grad = spike_grad

    def build(self, input_shape):
        if input_shape[-1] is not None and input_shape[-1] != self.units:
            raise ValueError(f'Expected input feature size {self.units}, got {input_shape[-1]}.')

        threshold_value = np.asarray(self._threshold_init)
        graded_spikes_factor_value = np.asarray(self._graded_spikes_factor_init)
        for name, value in (
            ('threshold', threshold_value),
            ('graded_spikes_factor', graded_spikes_factor_value),
        ):
            if value.shape not in ((), (self.units,)):
                raise ValueError(f'{name} must be a scalar or have shape ({self.units},), got {value.shape}.')

        self.threshold = self.add_weight(
            name='threshold',
            shape=threshold_value.shape,
            initializer=Constant(self._threshold_init),
            trainable=self.learn_threshold,
        )
        self.graded_spikes_factor = self.add_weight(
            name='graded_spikes_factor',
            shape=graded_spikes_factor_value.shape,
            initializer=Constant(self._graded_spikes_factor_init),
            trainable=self.learn_graded_spikes_factor,
        )
        super().build(input_shape)

    def fire(self, mem):
        spk = self.spike_grad(mem - self.threshold)
        if self.inhibition:
            index = ops.argmax(mem - self.threshold, axis=1)
            spk = spk * ops.one_hot(index, self.units, dtype=ops.dtype(spk))
        return spk * self.graded_spikes_factor, spk

    def reset(self, mem, spk):
        reset = ops.stop_gradient(spk) if self.detach_reset else spk
        if self.reset_mechanism == 'subtract':
            return mem - reset * self.threshold
        if self.reset_mechanism == 'zero':
            return mem * (1 - reset)
        return mem

    def call(self, inputs, states, training=None):
        del training
        mem_tm1 = states[0] if isinstance(states, (list, tuple)) else states
        mem = mem_tm1 + inputs
        output, spk = self.fire(mem)
        return output, [self.reset(mem, spk)]

    def get_initial_state(self, batch_size=None):
        return [ops.zeros((batch_size, self.units), dtype=self.compute_dtype)]

    @staticmethod
    def _surrogate_bypass(inputs):
        return ops.cast(inputs > 0, ops.dtype(inputs))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'units': self.units,
                'threshold': self._threshold_config,
                'spike_grad': self._spike_grad_arg
                if self._spike_grad_arg is False or self._spike_grad_arg is None
                else serialize_keras_object(self._spike_grad_arg),
                'surrogate_disable': self.surrogate_disable,
                'inhibition': self.inhibition,
                'learn_threshold': self.learn_threshold,
                'reset_mechanism': self.reset_mechanism,
                'detach_reset': self.detach_reset,
                'graded_spikes_factor': self._graded_spikes_factor_config,
                'learn_graded_spikes_factor': self.learn_graded_spikes_factor,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        if config.get('spike_grad') is not False and config.get('spike_grad') is not None:
            config['spike_grad'] = deserialize_keras_object(config['spike_grad'])
        return cls(**config)


@register_keras_serializable(package='hgq')
class LIFCell(SpikingNeuralCell):
    """Leaky integrate-and-fire RNN cell."""

    def __init__(
        self,
        units: int,
        beta=1.0,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        inhibition=False,
        learn_beta=False,
        learn_threshold=False,
        reset_mechanism='subtract',
        detach_reset=True,
        graded_spikes_factor=1.0,
        learn_graded_spikes_factor=False,
        **kwargs,
    ):
        super().__init__(
            units=units,
            threshold=threshold,
            spike_grad=spike_grad,
            surrogate_disable=surrogate_disable,
            inhibition=inhibition,
            learn_threshold=learn_threshold,
            reset_mechanism=reset_mechanism,
            detach_reset=detach_reset,
            graded_spikes_factor=graded_spikes_factor,
            learn_graded_spikes_factor=learn_graded_spikes_factor,
            **kwargs,
        )
        beta_value = np.asarray(beta)
        self._beta_init = beta
        self._beta_config = beta_value.item() if beta_value.shape == () else beta_value.tolist()
        self.learn_beta = learn_beta

    def build(self, input_shape):
        super().build(input_shape)
        beta_value = np.asarray(self._beta_init)
        if beta_value.shape not in ((), (self.units,)):
            raise ValueError(f'beta must be a scalar or have shape ({self.units},), got {beta_value.shape}.')

        self.beta = self.add_weight(
            name='beta',
            shape=beta_value.shape,
            initializer=Constant(self._beta_init),
            trainable=self.learn_beta,
            constraint=MinMax(0, 1),
        )

    def call(self, inputs, states, training=None):
        state_tm1 = states[0] if isinstance(states, (list, tuple)) else states
        state = self.beta * state_tm1 + inputs
        output, pulse = self.fire(state)
        new_states = self.reset(state, pulse)
        new_states = [new_states] if isinstance(states, (list, tuple)) else new_states
        return output, new_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'beta': self._beta_config,
                'learn_beta': self.learn_beta,
            }
        )
        return config


class QSimpleSNNCell(QLayerBaseSingleInput, SpikingNeuralCell):
    """Quantization-aware integrate-and-fire RNN cell.

    The membrane accumulator is not quantized by default. Input, emitted
    spikes, scalar parameters, and optionally recurrent state may be quantized.
    """

    __output_quantizer_handled__ = True

    def __init__(
        self,
        units: int,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        inhibition=False,
        learn_threshold=False,
        reset_mechanism='subtract',
        detach_reset=True,
        graded_spikes_factor=1.0,
        learn_graded_spikes_factor=False,
        iq_conf: QuantizerConfig | None = None,
        sq_conf: QuantizerConfig | None = None,
        oq_conf: QuantizerConfig | None = None,
        graded_spikes_factor_q_conf: QuantizerConfig | None = None,
        standalone: bool = True,
        enable_ebops: bool | None = None,
        enable_iq: bool | None = None,
        enable_sq: bool | None = None,
        enable_oq: bool | None = None,
        beta0: float | None = None,
        ebops_factor: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            units=units,
            threshold=threshold,
            spike_grad=spike_grad,
            surrogate_disable=surrogate_disable,
            inhibition=inhibition,
            learn_threshold=learn_threshold,
            reset_mechanism=reset_mechanism,
            detach_reset=detach_reset,
            graded_spikes_factor=graded_spikes_factor,
            learn_graded_spikes_factor=learn_graded_spikes_factor,
            iq_conf=iq_conf,
            oq_conf=oq_conf,
            enable_ebops=enable_ebops if enable_ebops is not None else global_config['enable_ebops'],
            enable_iq=enable_iq if enable_iq is not None else global_config['enable_iq'],
            enable_oq=enable_oq if enable_oq is not None else global_config['enable_oq'],
            beta0=beta0,
            ebops_factor=ebops_factor,
            **kwargs,
        )
        self._graded_spikes_factor_q = Quantizer(
            graded_spikes_factor_q_conf or QuantizerConfig('default', 'weight'),
            name=f'{self.name}_graded_spikes_factor_q',
        )
        self._enable_sq = enable_sq if enable_sq is not None else global_config['enable_sq']
        if self.enable_sq:
            self._sq = Quantizer(sq_conf or QuantizerConfig(place='datalane'), name=f'{self.name}_sq')
        self.standalone = standalone
        self._ebops_sequence_length = 1

    @property
    def enable_sq(self):
        return self._enable_sq

    @property
    def sq(self):
        if not self.enable_sq:
            raise ValueError('State Quantizer is not enabled.')
        return self._sq

    @property
    def graded_spikes_factor_q(self):
        return self._graded_spikes_factor_q

    @property
    def qgraded_spikes_factor(self):
        return self.graded_spikes_factor_q(self.graded_spikes_factor)

    @property
    def enable_ebops(self):
        return self._enable_ebops and self.standalone

    def _set_weight_cache(self):
        pass

    def _invalidate_weight_cache(self):
        pass

    def build(self, input_shape):
        super().build(input_shape)
        if self.enable_sq and not self.sq.built:
            self.sq.build((input_shape[0], self.units))
        self.graded_spikes_factor_q.build(self.graded_spikes_factor.shape)
        if self.enable_oq and not self.oq.built:
            self.oq.build((input_shape[0], self.units))

    def fire(self, mem):
        spk = self.spike_grad(mem - self.threshold)
        if self.inhibition:
            index = ops.argmax(mem - self.threshold, axis=1)
            spk = spk * ops.one_hot(index, self.units, dtype=ops.dtype(spk))
        return spk * self.qgraded_spikes_factor, spk

    def reset(self, mem, spk):
        reset = ops.stop_gradient(spk) if self.detach_reset else spk
        if self.reset_mechanism == 'subtract':
            return mem - reset * self.threshold
        if self.reset_mechanism == 'zero':
            return mem * (1 - reset)
        return mem

    def _quantize_input(self, inputs, training):
        return self.iq(inputs, training=training) if self.enable_iq else inputs

    def _quantize_output(self, output, training):
        return self.oq(output, training=training) if self.enable_oq else output

    def _quantize_state(self, state, training):
        return self.sq(state, training=training) if self.enable_sq else state

    def _accumulator_bits(self, shape):
        if self.enable_sq:
            return self.sq.bits_(shape)
        if self.enable_iq:
            bits = self.iq.bits_(shape)
        else:
            bits = ops.ones(shape, dtype=self.dtype)
        extra = int(np.ceil(np.log2(max(self._ebops_sequence_length, 1))))
        return bits + extra

    def _graded_spikes_factor_bits(self, shape):
        return ops.broadcast_to(self.graded_spikes_factor_q.bits_(self.graded_spikes_factor.shape), shape)

    @staticmethod
    def _add_ebops(bits0, bits1):
        return ops.sum(bits0 + bits1 - ops.minimum(bits0, bits1)) * 0.65

    def _fire_reset_ebops(self, shape, mem_bits):
        ebops = ops.sum(mem_bits) * 0.65
        pulse_bits = ops.ones(shape, dtype=self.dtype)
        ebops = ebops + ops.sum(pulse_bits * self._graded_spikes_factor_bits(shape))  # type: ignore
        if self.reset_mechanism == 'subtract':
            ebops = ebops + ops.sum(mem_bits) * 0.65  # type: ignore
        elif self.reset_mechanism == 'zero':
            ebops = ebops + ops.sum(mem_bits * pulse_bits)  # type: ignore
        return ebops

    def call(self, inputs, states, training=None):
        mem_tm1 = states[0] if isinstance(states, (list, tuple)) else states
        mem = mem_tm1 + self._quantize_input(inputs, training)
        output, spk = self.fire(mem)
        output = self._quantize_output(output, training)
        new_state = self._quantize_state(self.reset(mem, spk), training)
        return output, [new_state] if isinstance(states, (list, tuple)) else new_state

    def _compute_ebops(self, shape):
        if not self.enable_iq:
            return ops.cast(0, self.dtype)
        input_bits = self.iq.bits_(shape)
        mem_bits = self._accumulator_bits(shape)
        return self._add_ebops(mem_bits, input_bits) + self._fire_reset_ebops(shape, mem_bits)  # type: ignore

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'graded_spikes_factor_q_conf': self.graded_spikes_factor_q.config,
                'sq_conf': self.sq.config if self.enable_sq else None,
                'enable_sq': self.enable_sq,
                'standalone': self.standalone,
            }
        )
        return config


class QLIFCell(QSimpleSNNCell):
    """Quantization-aware leaky integrate-and-fire RNN cell."""

    def __init__(
        self,
        units: int,
        beta=0.5,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        inhibition=False,
        learn_beta=False,
        learn_threshold=False,
        reset_mechanism='subtract',
        detach_reset=True,
        graded_spikes_factor=1.0,
        learn_graded_spikes_factor=False,
        iq_conf: QuantizerConfig | None = None,
        sq_conf: QuantizerConfig | None = None,
        oq_conf: QuantizerConfig | None = None,
        beta_q_conf: QuantizerConfig | None = None,
        graded_spikes_factor_q_conf: QuantizerConfig | None = None,
        standalone: bool = True,
        enable_ebops: bool | None = None,
        enable_iq: bool | None = None,
        enable_sq: bool | None = None,
        enable_oq: bool | None = None,
        beta0: float | None = None,
        ebops_factor: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            units=units,
            threshold=threshold,
            spike_grad=spike_grad,
            surrogate_disable=surrogate_disable,
            inhibition=inhibition,
            learn_threshold=learn_threshold,
            reset_mechanism=reset_mechanism,
            detach_reset=detach_reset,
            graded_spikes_factor=graded_spikes_factor,
            learn_graded_spikes_factor=learn_graded_spikes_factor,
            iq_conf=iq_conf,
            sq_conf=sq_conf,
            oq_conf=oq_conf,
            graded_spikes_factor_q_conf=graded_spikes_factor_q_conf,
            standalone=standalone,
            enable_ebops=enable_ebops,
            enable_iq=enable_iq,
            enable_sq=enable_sq,
            enable_oq=enable_oq,
            beta0=beta0,
            ebops_factor=ebops_factor,
            **kwargs,
        )
        beta_value = np.asarray(beta)
        self._beta_init = beta
        self._beta_config = beta_value.item() if beta_value.shape == () else beta_value.tolist()
        self.learn_beta = learn_beta
        self._beta_q = Quantizer(beta_q_conf or QuantizerConfig('default', 'weight'), name=f'{self.name}_beta_q')

    @property
    def beta_q(self):
        return self._beta_q

    @property
    def qlif_beta(self):
        return self.beta_q(self.lif_beta)

    def build(self, input_shape):
        beta_value = np.asarray(self._beta_init)
        if beta_value.shape not in ((), (self.units,)):
            raise ValueError(f'beta must be a scalar or have shape ({self.units},), got {beta_value.shape}.')
        self.lif_beta = self.add_weight(
            name='beta',
            shape=beta_value.shape,
            initializer=Constant(self._beta_init),
            trainable=self.learn_beta,
            constraint=MinMax(0, 1),
        )
        super().build(input_shape)
        self.beta_q.build(self.lif_beta.shape)

    def call(self, inputs, states, training=None):
        mem_tm1 = states[0] if isinstance(states, (list, tuple)) else states
        mem = self.qlif_beta * mem_tm1 + self._quantize_input(inputs, training)
        output, spk = self.fire(mem)
        output = self._quantize_output(output, training)
        new_state = self._quantize_state(self.reset(mem, spk), training)
        return output, [new_state] if isinstance(states, (list, tuple)) else new_state

    def _compute_ebops(self, shape):
        if not self.enable_iq:
            return ops.cast(0, self.dtype)
        input_bits = self.iq.bits_(shape)
        mem_bits = self._accumulator_bits(shape)
        beta_bits = ops.broadcast_to(self.beta_q.bits_(self.lif_beta.shape), shape)
        ebops = ops.sum(mem_bits * beta_bits)
        ebops = ebops + self._add_ebops(mem_bits, input_bits)  # type: ignore
        return ebops + self._fire_reset_ebops(shape, mem_bits)  # type: ignore

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'beta_q_conf': self.beta_q.config,
                'beta': self._beta_config,
                'learn_beta': self.learn_beta,
            }
        )
        return config


class _QSNN(QRNN):
    def build(self, sequences_shape, initial_state_shape=None):
        seq_len = sequences_shape[1]
        if seq_len is None:
            raise ValueError(f'{self.__class__.__name__} requires a fixed sequence length.')
        self.cell._ebops_sequence_length = int(seq_len)
        super().build(sequences_shape, initial_state_shape)

    def _compute_ebops(self, shape):
        return self.cell._compute_ebops((shape[0], shape[2])) * self.parallelization_factor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'units': self.cell.units,
                'threshold': self.cell._threshold_config,
                'spike_grad': self.cell._spike_grad_arg
                if self.cell._spike_grad_arg is False or self.cell._spike_grad_arg is None
                else serialize_keras_object(self.cell._spike_grad_arg),
                'surrogate_disable': self.cell.surrogate_disable,
                'inhibition': self.cell.inhibition,
                'learn_threshold': self.cell.learn_threshold,
                'reset_mechanism': self.cell.reset_mechanism,
                'detach_reset': self.cell.detach_reset,
                'graded_spikes_factor': self.cell._graded_spikes_factor_config,
                'learn_graded_spikes_factor': self.cell.learn_graded_spikes_factor,
                'iq_conf': self.cell.iq.config if self.cell.enable_iq else None,
                'sq_conf': self.cell.sq.config if self.cell.enable_sq else None,
                'oq_conf': self.cell.oq.config if self.cell.enable_oq else None,
                'enable_sq': self.cell.enable_sq,
                'graded_spikes_factor_q_conf': self.cell.graded_spikes_factor_q.config,
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = dict(config)
        config.pop('cell', None)
        config.pop('zero_output_for_mask', None)
        if callable(config.get('reset_mechanism')):
            config['reset_mechanism'] = config['reset_mechanism'].__name__
        for key in (
            'beta0',
            'dtype',
            'iq_conf',
            'sq_conf',
            'oq_conf',
            'graded_spikes_factor_q_conf',
            'beta_q_conf',
        ):
            if isinstance(config.get(key), dict):
                config[key] = deserialize_keras_object(config[key], custom_objects=custom_objects)
        if isinstance(config.get('spike_grad'), dict):
            config['spike_grad'] = deserialize_keras_object(config['spike_grad'])
        return cls(**config)


class QSimpleSNN(_QSNN):
    """Quantization-aware integrate-and-fire sequence layer."""

    def __init__(
        self,
        units: int,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        inhibition=False,
        learn_threshold=False,
        reset_mechanism='subtract',
        detach_reset=True,
        graded_spikes_factor=1.0,
        learn_graded_spikes_factor=False,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        unroll=False,
        iq_conf: QuantizerConfig | None = None,
        sq_conf: QuantizerConfig | None = None,
        oq_conf: QuantizerConfig | None = None,
        graded_spikes_factor_q_conf: QuantizerConfig | None = None,
        parallelization_factor=-1,
        enable_oq: bool | None = None,
        enable_iq: bool | None = None,
        enable_sq: bool | None = None,
        enable_ebops: bool | None = None,
        beta0: float | None = None,
        **kwargs,
    ):
        cell = QSimpleSNNCell(
            units=units,
            threshold=threshold,
            spike_grad=spike_grad,
            surrogate_disable=surrogate_disable,
            inhibition=inhibition,
            learn_threshold=learn_threshold,
            reset_mechanism=reset_mechanism,
            detach_reset=detach_reset,
            graded_spikes_factor=graded_spikes_factor,
            learn_graded_spikes_factor=learn_graded_spikes_factor,
            dtype=kwargs.get('dtype', None),
            trainable=kwargs.get('trainable', True),
            name='q_spiking_neural_cell',
            iq_conf=iq_conf,
            sq_conf=sq_conf,
            oq_conf=oq_conf,
            graded_spikes_factor_q_conf=graded_spikes_factor_q_conf,
            standalone=False,
            enable_oq=enable_oq,
            enable_iq=enable_iq,
            enable_sq=enable_sq,
            enable_ebops=enable_ebops,
            beta0=beta0,
        )
        super().__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs,
        )
        self.input_spec = InputSpec(ndim=3)
        self.parallelization_factor = parallelization_factor
        self._set_unroll()


class QLIF(_QSNN):
    """Quantization-aware leaky integrate-and-fire sequence layer."""

    def __init__(
        self,
        units: int,
        beta=0.5,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        inhibition=False,
        learn_beta=False,
        learn_threshold=False,
        reset_mechanism='subtract',
        detach_reset=True,
        graded_spikes_factor=1.0,
        learn_graded_spikes_factor=False,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        unroll=False,
        iq_conf: QuantizerConfig | None = None,
        sq_conf: QuantizerConfig | None = None,
        oq_conf: QuantizerConfig | None = None,
        beta_q_conf: QuantizerConfig | None = None,
        graded_spikes_factor_q_conf: QuantizerConfig | None = None,
        parallelization_factor=-1,
        enable_oq: bool | None = None,
        enable_iq: bool | None = None,
        enable_sq: bool | None = None,
        enable_ebops: bool | None = None,
        beta0: float | None = None,
        **kwargs,
    ):
        cell = QLIFCell(
            units=units,
            beta=beta,
            threshold=threshold,
            spike_grad=spike_grad,
            surrogate_disable=surrogate_disable,
            inhibition=inhibition,
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
            reset_mechanism=reset_mechanism,
            detach_reset=detach_reset,
            graded_spikes_factor=graded_spikes_factor,
            learn_graded_spikes_factor=learn_graded_spikes_factor,
            dtype=kwargs.get('dtype', None),
            trainable=kwargs.get('trainable', True),
            name='q_lif_cell',
            iq_conf=iq_conf,
            sq_conf=sq_conf,
            oq_conf=oq_conf,
            beta_q_conf=beta_q_conf,
            graded_spikes_factor_q_conf=graded_spikes_factor_q_conf,
            standalone=False,
            enable_oq=enable_oq,
            enable_iq=enable_iq,
            enable_sq=enable_sq,
            enable_ebops=enable_ebops,
            beta0=beta0,
        )
        super().__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs,
        )
        self.input_spec = InputSpec(ndim=3)
        self.parallelization_factor = parallelization_factor
        self._set_unroll()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'beta': self.cell._beta_config,
                'learn_beta': self.cell.learn_beta,
                'beta_q_conf': self.cell.beta_q.config,
            }
        )
        return config


QLayerBase.register(QSimpleSNNCell)
QLayerBase.register(QLIFCell)
QLayerBase.register(QSimpleSNN)
QLayerBase.register(QLIF)
