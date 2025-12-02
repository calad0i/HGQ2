from keras import ops
from keras.layers import EinsumDense

from ...quantizer import Quantizer
from ...quantizer.config import QuantizerConfig
from ...utils.misc import gather_vars_to_kwargs
from .base import QLayerBaseSingleInput


class QEinsumDense(QLayerBaseSingleInput, EinsumDense):
    def __init__(
        self,
        equation,
        output_shape,
        activation=None,
        bias_axes=None,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        ebops_scaler: float = 1.0,
        kq_conf: None | QuantizerConfig = None,
        iq_conf: None | QuantizerConfig = None,
        bq_conf: None | QuantizerConfig = None,
        **kwargs,
    ):
        kwargs = gather_vars_to_kwargs('self|kq_conf|bq_conf|ebops_scaler')
        super().__init__(lora_rank=None, **kwargs)

        kq_conf = kq_conf or QuantizerConfig('default', 'weight')
        self._kq = Quantizer(kq_conf, name=f'{self.name}_kq')
        bq_conf = bq_conf or QuantizerConfig('default', 'bias')
        self._bq = None if bias_axes is None else Quantizer(bq_conf, name=f'{self.name}_bq')
        self.ebops_scaler = ebops_scaler

    @property
    def kq(self):
        return self._kq

    @property
    def bq(self):
        return self._bq

    def build(self, input_shape):
        super().build(input_shape)

        self.kq.build(ops.shape(self._kernel))
        if self.bias is not None:
            assert self.bq is not None
            self.bq.build(ops.shape(self.bias))
        self.ebops_equation = self.equation.split('->')[0] + '->'

    def call(self, inputs, training=None):
        qkernel = self.kq(self._kernel, training=training)
        if self.enable_iq:
            inputs = self.iq(inputs, training=training)
        x = ops.einsum(self.equation, inputs, qkernel)
        if self.bias is not None:
            assert self.bq is not None
            x += self.bq(self.bias, training=training)
        if self.activation is not None:
            x = self.activation(x)

        return x

    def _compute_ebops(self, shape):
        # shape = shapes[0]
        bw_inp = self.iq.bits_(shape)
        bw_ker = self.kq.bits_(ops.shape(self.kernel))
        ebops = ops.einsum(self.ebops_equation, bw_inp, bw_ker)
        if self.bq is not None:
            bw_bias = self.bq.bits_(ops.shape(self.bias))
            size = ops.cast(ops.prod(self.full_output_shape[1:]), self.dtype)
            ebops = ebops + ops.mean(bw_bias) * size  # type: ignore
        return ebops * self.ebops_scaler  # type: ignore

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'kq_conf': self.kq.config,
                'bq_conf': self.bq.config if self.bq is not None else None,
            }
        )
        return config

    @property
    def qkernel(self):
        return self.kq(self._kernel)

    @property
    def qbias(self):
        if self.bias is None:
            return None
        assert self.bq is not None
        return self.bq(self.bias)
