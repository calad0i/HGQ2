from math import prod

from keras import ops
from keras.src.layers.convolutional.base_depthwise_conv import BaseDepthwiseConv
from keras.src.layers.convolutional.base_separable_conv import BaseSeparableConv

from ..quantizer import Quantizer
from ..quantizer.config import QuantizerConfig
from ..utils.misc import gather_vars_to_kwargs
from .core.base import QLayerBaseSingleInput


def _n_parallel(layer, input_shape):
    outputs_shape = layer.compute_output_shape(input_shape)
    if None in outputs_shape[1:]:
        return None
    return prod(outputs_shape[1:-1]) if layer.data_format == 'channels_last' else prod(outputs_shape[2:])


def _bias_shape(layer, n):
    if layer.data_format == 'channels_last':
        return (1,) * (layer.rank + 1) + (n,)
    return (1, n) + (1,) * layer.rank


class QBaseDepthwiseConv(QLayerBaseSingleInput, BaseDepthwiseConv):
    """Quantized depthwise convolution: each input channel is convolved with its own `depth_multiplier` kernels.

    Quantizers: iq (input), kq (depthwise kernel), bq (bias). EBOPs are summed over the per-channel products only,
    as no channels are mixed.
    """

    def __init__(
        self,
        rank,
        depth_multiplier,
        kernel_size,
        strides=1,
        padding='valid',
        data_format=None,
        dilation_rate=1,
        activation=None,
        use_bias=True,
        depthwise_initializer='glorot_uniform',
        bias_initializer='zeros',
        depthwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        bias_constraint=None,
        parallelization_factor=-1,
        kq_conf: None | QuantizerConfig = None,
        iq_conf: None | QuantizerConfig = None,
        bq_conf: None | QuantizerConfig = None,
        **kwargs,
    ):
        kwargs = gather_vars_to_kwargs('self|kq_conf|bq_conf|parallelization_factor')
        super().__init__(**kwargs)

        kq_conf = kq_conf or QuantizerConfig('default', 'weight')
        self._kq = Quantizer(kq_conf, name=f'{self.name}_kq')
        bq_conf = bq_conf or QuantizerConfig('default', 'bias')
        self._bq = Quantizer(bq_conf, name=f'{self.name}_bq') if use_bias else None
        self.parallelization_factor = parallelization_factor

    @property
    def kq(self):
        return self._kq

    @property
    def bq(self):
        return self._bq

    def build(self, input_shape):
        super().build(input_shape)
        self.kq.build(ops.shape(self.kernel))
        if self.use_bias:
            assert self.bq is not None
            self.bq.build(ops.shape(self.bias))

        self.n_parallel = _n_parallel(self, input_shape)
        if self.parallelization_factor <= 0:
            assert self.n_parallel is not None
            self.parallelization_factor = self.n_parallel

    def convolution_op(self, inputs, kernel):
        return ops.depthwise_conv(
            inputs,
            kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format,
        )

    def call(self, inputs, training=None):
        if self.enable_iq:
            inputs = self.iq(inputs, training=training)
        outputs = self.convolution_op(inputs, self.qkernel)
        if self.use_bias:
            outputs += ops.reshape(self.qbias, _bias_shape(self, ops.shape(self.bias)[0]))  # type: ignore

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def _compute_ebops(self, shape):
        bw_inp = self.iq.bits_(shape)
        bw_ker = self.kq.bits_(ops.shape(self.kernel))
        ebops = ops.sum(self.convolution_op(bw_inp, bw_ker))
        ebops *= self.parallelization_factor / self.n_parallel  # type: ignore

        if self.bq is not None:
            n_out = ops.shape(self.bias)[0]
            size = ops.cast(ops.prod(shape[:-1]) * n_out, self.dtype)
            bw_bias = self.bq.bits_(ops.shape(self.bias))
            ebops = ebops + ops.mean(bw_bias) * size  # type: ignore

        return ebops

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
        return self.kq(self.kernel)

    @property
    def qbias(self):
        if self.bias is None:
            return None
        assert self.bq is not None
        return self.bq(self.bias)


class QDepthwiseConv1D(QBaseDepthwiseConv):
    def __init__(
        self,
        kernel_size,
        strides=1,
        padding='valid',
        depth_multiplier=1,
        data_format=None,
        dilation_rate=1,
        activation=None,
        use_bias=True,
        depthwise_initializer='glorot_uniform',
        bias_initializer='zeros',
        depthwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        bias_constraint=None,
        kq_conf: None | QuantizerConfig = None,
        iq_conf: None | QuantizerConfig = None,
        bq_conf: None | QuantizerConfig = None,
        **kwargs,
    ):
        kwargs = gather_vars_to_kwargs('self')
        super().__init__(rank=1, **kwargs)


class QDepthwiseConv2D(QBaseDepthwiseConv):
    def __init__(
        self,
        kernel_size,
        strides=(1, 1),
        padding='valid',
        depth_multiplier=1,
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        depthwise_initializer='glorot_uniform',
        bias_initializer='zeros',
        depthwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        bias_constraint=None,
        kq_conf: None | QuantizerConfig = None,
        iq_conf: None | QuantizerConfig = None,
        bq_conf: None | QuantizerConfig = None,
        **kwargs,
    ):
        kwargs = gather_vars_to_kwargs('self')
        super().__init__(rank=2, **kwargs)


class QBaseSeparableConv(QLayerBaseSingleInput, BaseSeparableConv):
    """Quantized depthwise-separable convolution: a depthwise convolution followed by a pointwise (1x1) one.

    Quantizers: iq (input), dkq (depthwise kernel), pq (the depthwise result, i.e. the pointwise input), pkq
    (pointwise kernel) and bq (bias, added after the pointwise step). With pq, the intermediate tensor has a
    definite fixed-point type, so the layer maps exactly onto a depthwise + quantizer + pointwise chain in hardware.
    EBOPs are those of the two steps added together.
    """

    def __init__(
        self,
        rank,
        depth_multiplier,
        filters,
        kernel_size,
        strides=1,
        padding='valid',
        data_format=None,
        dilation_rate=1,
        activation=None,
        use_bias=True,
        depthwise_initializer='glorot_uniform',
        pointwise_initializer='glorot_uniform',
        bias_initializer='zeros',
        depthwise_regularizer=None,
        pointwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        pointwise_constraint=None,
        bias_constraint=None,
        parallelization_factor=-1,
        dkq_conf: None | QuantizerConfig = None,
        pkq_conf: None | QuantizerConfig = None,
        iq_conf: None | QuantizerConfig = None,
        pq_conf: None | QuantizerConfig = None,
        bq_conf: None | QuantizerConfig = None,
        **kwargs,
    ):
        kwargs = gather_vars_to_kwargs('self|dkq_conf|pkq_conf|pq_conf|bq_conf|parallelization_factor')
        super().__init__(**kwargs)

        self._dkq = Quantizer(dkq_conf or QuantizerConfig('default', 'weight'), name=f'{self.name}_dkq')
        self._pkq = Quantizer(pkq_conf or QuantizerConfig('default', 'weight'), name=f'{self.name}_pkq')
        self._pq = Quantizer(pq_conf or QuantizerConfig('default', 'datalane'), name=f'{self.name}_pq')
        bq_conf = bq_conf or QuantizerConfig('default', 'bias')
        self._bq = Quantizer(bq_conf, name=f'{self.name}_bq') if use_bias else None
        self.parallelization_factor = parallelization_factor

    @property
    def dkq(self):
        return self._dkq

    @property
    def pkq(self):
        return self._pkq

    @property
    def pq(self):
        return self._pq

    @property
    def bq(self):
        return self._bq

    def build(self, input_shape):
        super().build(input_shape)
        self.dkq.build(ops.shape(self.depthwise_kernel))
        self.pkq.build(ops.shape(self.pointwise_kernel))
        self.pq.build(self._depthwise_output_shape(input_shape))
        if self.use_bias:
            assert self.bq is not None
            self.bq.build(ops.shape(self.bias))

        self.n_parallel = _n_parallel(self, input_shape)
        if self.parallelization_factor <= 0:
            assert self.n_parallel is not None
            self.parallelization_factor = self.n_parallel

    def _depthwise_output_shape(self, input_shape):
        n_mid = (input_shape[-1] if self.data_format == 'channels_last' else input_shape[1]) * self.depth_multiplier
        out = list(self.compute_output_shape(input_shape))
        out[-1 if self.data_format == 'channels_last' else 1] = n_mid
        return tuple(out)

    def depthwise_op(self, inputs, kernel):
        return ops.depthwise_conv(
            inputs,
            kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format,
        )

    def pointwise_op(self, inputs, kernel):
        return ops.conv(inputs, kernel, strides=1, padding='valid', data_format=self.data_format)

    def call(self, inputs, training=None):
        if self.enable_iq:
            inputs = self.iq(inputs, training=training)
        x = self.depthwise_op(inputs, self.qdepthwise_kernel)
        x = self.pq(x, training=training)
        outputs = self.pointwise_op(x, self.qpointwise_kernel)
        if self.use_bias:
            outputs += ops.reshape(self.qbias, _bias_shape(self, self.filters))  # type: ignore

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def _compute_ebops(self, shape):
        bw_inp = self.iq.bits_(shape)
        bw_dk = self.dkq.bits_(ops.shape(self.depthwise_kernel))
        ebops = ops.sum(self.depthwise_op(bw_inp, bw_dk))
        bw_mid = self.pq.bits_(self._depthwise_output_shape(shape))
        bw_pk = self.pkq.bits_(ops.shape(self.pointwise_kernel))
        ebops += ops.sum(self.pointwise_op(bw_mid, bw_pk))
        ebops *= self.parallelization_factor / self.n_parallel  # type: ignore

        if self.bq is not None:
            out_shape = self.compute_output_shape(shape)
            size = ops.cast(prod(out_shape[1:]), self.dtype)
            bw_bias = self.bq.bits_(ops.shape(self.bias))
            ebops = ebops + ops.mean(bw_bias) * size  # type: ignore

        return ebops

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'dkq_conf': self.dkq.config,
                'pkq_conf': self.pkq.config,
                'pq_conf': self.pq.config,
                'bq_conf': self.bq.config if self.bq is not None else None,
                'parallelization_factor': self.parallelization_factor,
            }
        )
        return config

    @property
    def qdepthwise_kernel(self):
        return self.dkq(self.depthwise_kernel)

    @property
    def qpointwise_kernel(self):
        return self.pkq(self.pointwise_kernel)

    @property
    def qbias(self):
        if self.bias is None:
            return None
        assert self.bq is not None
        return self.bq(self.bias)


class QSeparableConv1D(QBaseSeparableConv):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding='valid',
        data_format=None,
        dilation_rate=1,
        depth_multiplier=1,
        activation=None,
        use_bias=True,
        depthwise_initializer='glorot_uniform',
        pointwise_initializer='glorot_uniform',
        bias_initializer='zeros',
        depthwise_regularizer=None,
        pointwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        pointwise_constraint=None,
        bias_constraint=None,
        dkq_conf: None | QuantizerConfig = None,
        pkq_conf: None | QuantizerConfig = None,
        iq_conf: None | QuantizerConfig = None,
        pq_conf: None | QuantizerConfig = None,
        bq_conf: None | QuantizerConfig = None,
        **kwargs,
    ):
        kwargs = gather_vars_to_kwargs('self')
        super().__init__(rank=1, **kwargs)


class QSeparableConv2D(QBaseSeparableConv):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding='valid',
        data_format=None,
        dilation_rate=(1, 1),
        depth_multiplier=1,
        activation=None,
        use_bias=True,
        depthwise_initializer='glorot_uniform',
        pointwise_initializer='glorot_uniform',
        bias_initializer='zeros',
        depthwise_regularizer=None,
        pointwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        pointwise_constraint=None,
        bias_constraint=None,
        dkq_conf: None | QuantizerConfig = None,
        pkq_conf: None | QuantizerConfig = None,
        iq_conf: None | QuantizerConfig = None,
        pq_conf: None | QuantizerConfig = None,
        bq_conf: None | QuantizerConfig = None,
        **kwargs,
    ):
        kwargs = gather_vars_to_kwargs('self')
        super().__init__(rank=2, **kwargs)
