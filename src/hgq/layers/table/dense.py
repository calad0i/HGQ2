from collections.abc import Callable
from math import prod

import keras
from keras import ops
from keras.layers import Layer
from keras.saving import register_keras_serializable

from ...quantizer import Quantizer, QuantizerConfig
from ..core import QLayerBaseSingleInput


@register_keras_serializable(package='hgq2')
class QDenseT(QLayerBaseSingleInput):
    def __init__(
        self,
        n_out: int,
        n_hl: int = 1,
        d_hl: int = 8,
        use_bias: bool = True,
        activation: Callable | None | str = None,
        subnn_activation: str | Callable | Layer | None = None,
        toq_conf: QuantizerConfig | None = None,
        parallelization_factor: int = -1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_out = n_out
        self.d_hl = d_hl
        self.use_bias = use_bias
        self.subnn_activation = keras.activations.get(subnn_activation)
        self.parallelization_factor = parallelization_factor

        assert n_hl >= 0

        self.n_hl = n_hl

        toq_conf = toq_conf or QuantizerConfig(place='table')
        self._toq = Quantizer(toq_conf)
        self.activation = keras.activations.get(activation)

    def _build_module(self, n_in: int):
        layers = []
        _shape = (n_in, self.n_out, self.d_hl)
        bias_axes = 'ioD' if self.use_bias else None
        for _ in range(self.n_hl):
            layers.append(
                keras.layers.EinsumDense(
                    '...iod,iodD->...ioD',
                    _shape,
                    self.subnn_activation,
                    bias_axes=bias_axes,
                )
            )
        bias_axes = 'io' if self.use_bias else None
        l_out = keras.layers.EinsumDense('...iod,iod->...io', (n_in, self.n_out), 'linear', bias_axes=bias_axes)
        layers.append(l_out)
        module = keras.models.Sequential(layers)
        return module

    def call(self, x, training=None):
        n_in = keras.ops.shape(x)[-1]
        x = keras.ops.broadcast_to(x[..., None], (*keras.ops.shape(x)[:-1], n_in, self.n_out))  # (B, N_in, N_out)
        if self.enable_iq:
            x = self.iq(x)
        x = x[..., None]  # (B, ..., N_in, N_out, 1)

        return self.activation(ops.sum(self.toq(self.module(x, training=training)), axis=-2))

    def build(self, input_shape):
        input_shape = tuple(input_shape)
        self.n_in = input_shape[-1]

        self.n_parallel = prod(input_shape[1:-1])
        if self.parallelization_factor < 0:
            self.parallelization_factor = self.n_parallel

        if self.enable_iq and not self.iq.built:
            self.iq.build(input_shape + (self.n_out,))
        self.toq.build(input_shape + (self.n_out,))
        self.module = self._build_module(self.n_in)
        self.module.build(input_shape + (self.n_out, 1))
        super().build(input_shape)

    def _compute_ebops(self, shape: tuple[int, ...]):
        q_shape = shape + (self.n_out,)
        bits_in = self.iq.fbits_(q_shape)
        bits_out = self.toq.fbits_(q_shape)

        eff_lut5_count = ops.where(bits_in >= 5, 2 ** (bits_in - 5), 0.2 * bits_in)  # type: ignore

        table_lut5s = ops.dot(ops.ravel(eff_lut5_count), ops.ravel(bits_out)) * 0.5  # type: ignore

        return (table_lut5s + ops.sum(bits_out)) * self.parallelization_factor / self.n_parallel

    @property
    def toq(self):
        return self._toq

    def get_config(self):
        config = {
            'n_out': self.n_out,
            'n_hl': self.n_hl,
            'd_hl': self.d_hl,
            'subnn_activation': self.subnn_activation,
            'activation': self.activation,
            'toq_conf': self.toq.config,
            **super().get_config(),
        }
        return config
