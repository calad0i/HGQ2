from typing import overload

from keras import ops
from keras.layers import Layer
from keras.saving import deserialize_keras_object

from .config import QuantizerConfig, all_quantizer_keys


class Quantizer(Layer):
    """The generic quantizer layer, wraps internal quantizers to provide a universal interface. Supports float, fixed-point (KBI, KIF) quantization. Can be initialized with a QuantizerConfig object or with the quantizer type and its parameters."""

    @overload
    def __init__(self, config: QuantizerConfig, **kwargs): ...

    @overload
    def __init__(self, q_type='default', place='datalane', **kwargs): ...

    def __init__(self, *args, **kwargs):
        self.supports_masking = True
        self.config, kwargs = self.get_quantizer_config_kwargs(*args, **kwargs)
        self.qnoise_factor = self.config.qnoise_factor
        self.scaler = self.config.scaler
        self.affine = self.config.affine
        super().__init__(**kwargs)
        self.quantizer = self.config.get_quantizer()

    def build(self, input_shape):
        self.quantizer.build(input_shape)
        super().build(input_shape)

    def get_quantizer_config_kwargs(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], QuantizerConfig):
            return args[0], kwargs
        config = kwargs.pop('config', None)
        if isinstance(config, QuantizerConfig):
            return config, kwargs
        else:
            assert config is None, f'config must be a QuantizerConfig object, got {config}'

        _kwargs = {}
        for k in list(kwargs.keys()):
            if k in all_quantizer_keys:
                _kwargs[k] = kwargs.pop(k)
        config = QuantizerConfig(*args, **_kwargs)
        return config, kwargs

    def call(self, inputs, training=None):
        if self.scaler is not None:
            inputs = inputs / self.scaler
        inputs = ops.cast(inputs, ops.dtype(inputs))  # cast to tensor, for sure... (tf is playing naughty here)
        outputs = self.quantizer.call(inputs, training=training)
        if self.scaler is not None:
            outputs = outputs * self.scaler  # type: ignore
        if self.qnoise_factor is not None and training:
            outputs = inputs + self.qnoise_factor * (outputs - inputs)  # type: ignore
        if self.affine is not None:
            outputs = outputs * self.affine[0] + self.affine[1]  # type: ignore
        return outputs

    def get_config(self):
        config = super().get_config()
        config['config'] = self.config
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**deserialize_keras_object(config))

    @property
    def bits(self):
        return self.quantizer.bits

    @property
    def fbits(self):
        return self.quantizer.fbits

    @property
    def q_type(self):
        return self.config.q_type

    def __repr__(self):
        return f'{self.__class__.__name__}(q_type={self.config.q_type}, name={self.name}, built={self.built})'

    def bits_(self, shape):
        bits = self.bits
        return self.quantizer.bw_mapper.bw_to_x(bits, shape)

    def fbits_(self, shape):
        fbits = self.fbits
        return self.quantizer.bw_mapper.bw_to_x(fbits, shape)

    def min_(self, shape):
        _min = self.quantizer.min
        return self.quantizer.bw_mapper.bw_to_x(_min, shape)

    def max_(self, shape):
        _max = self.quantizer.max
        return self.quantizer.bw_mapper.bw_to_x(_max, shape)

    def epsilon_(self, shape):
        epsilon = self.quantizer.epsilon
        return self.quantizer.bw_mapper.bw_to_x(epsilon, shape)
