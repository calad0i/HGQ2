from keras import ops
from keras.layers import EinsumDense

from ...quantizer import Quantizer
from ...quantizer.config import QuantizerConfig
from ...utils.misc import gather_vars_to_kwargs
from .base import QLayerBaseSingleInput


def _einsum_free0_size(equation: str, input_shape: tuple[int | None, ...]) -> int:
    _inp, _ker_out = equation.split(',', 1)
    _ker, _out = _ker_out.split('->', 1)
    elided_size = len(input_shape) - len(_inp) + 3
    _inp = _inp.replace('...', '?' * elided_size)
    n_parallel = 1
    for ax, dim in zip(_inp[1:], input_shape[1:]):
        if ax not in _ker and (ax in _out or ax == '?'):
            assert dim is not None, f'Cannot determine input loop size for einsum eq: {equation} with input shape {input_shape}'
            n_parallel *= dim
    return n_parallel


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
        kq_conf: None | QuantizerConfig = None,
        iq_conf: None | QuantizerConfig = None,
        bq_conf: None | QuantizerConfig = None,
        parallelization_factor: int = -1,
        **kwargs,
    ):
        kwargs = gather_vars_to_kwargs('self|kq_conf|bq_conf|parallelization_factor')
        super().__init__(lora_rank=None, **kwargs)

        kq_conf = kq_conf or QuantizerConfig('default', 'weight')
        self._kq = Quantizer(kq_conf, name=f'{self.name}_kq')
        bq_conf = bq_conf or QuantizerConfig('default', 'bias')
        self._bq = None if bias_axes is None else Quantizer(bq_conf, name=f'{self.name}_bq')
        self.parallelization_factor = parallelization_factor

    @property
    def kq(self):
        return self._kq

    @property
    def bq(self):
        return self._bq

    def build(self, input_shape):
        super().build(input_shape)

        self.n_parallel = _einsum_free0_size(self.equation, input_shape)
        if self.parallelization_factor < 0:
            self.parallelization_factor = self.n_parallel

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
        bw_inp = self.iq.bits_(shape)
        bw_ker = self.kq.bits_(ops.shape(self.kernel))
        ebops = ops.einsum(self.ebops_equation, bw_inp, bw_ker)
        if self.bq is not None:
            bw_bias = self.bq.bits_(ops.shape(self.bias))
            size = ops.cast(ops.prod(self.full_output_shape[1:]), self.dtype)
            ebops = ebops + ops.mean(bw_bias) * size  # type: ignore
        return ebops * self.parallelization_factor / self.n_parallel  # type: ignore

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'kq_conf': self.kq.config,
                'bq_conf': self.bq.config if self.bq is not None else None,
                'parallelization_factor': self.parallelization_factor,
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
