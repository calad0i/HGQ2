from keras import ops
from keras.src.layers.convolutional.base_conv import BaseConv

from ..quantizer import Quantizer
from ..quantizer.config import QuantizerConfig
from ..utils.misc import gather_vars_to_kwargs
from .core.base import QLayerBaseSingleInput


class QBaseConv(QLayerBaseSingleInput, BaseConv):
    def __init__(
        self,
        rank,
        filters,
        kernel_size,
        strides=1,
        padding='valid',
        data_format=None,
        dilation_rate=1,
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
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

    def call(self, inputs, training=None):
        if self.enable_iq:
            inputs = self.iq(inputs, training=training)
        outputs = self.convolution_op(
            inputs,
            self.qkernel,
        )
        if self.use_bias:
            if self.data_format == 'channels_last':
                bias_shape = (1,) * (self.rank + 1) + (self.filters,)
            else:
                bias_shape = (1, self.filters) + (1,) * self.rank
            qbias = ops.reshape(self.qbias, bias_shape)
            outputs += qbias  # type: ignore

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def _compute_ebops(self, shape):
        bw_inp = self.iq.bits_(shape)
        bw_ker = self.kq.bits_(ops.shape(self.kernel))
        if self.parallelization_factor < 0:
            ebops = ops.sum(self.convolution_op(bw_inp, bw_ker))
        else:
            reduce_axis_kernel = tuple(range(0, self.rank + 1))
            if self.data_format == 'channels_last':
                reduce_axis_input = reduce_axis_kernel
            else:
                reduce_axis_input = (0,) + tuple(range(2, self.rank + 2))

            bw_inp = ops.max(bw_inp, axis=reduce_axis_input)  # Keep only maximum per channel
            reduce_axis_kernel = tuple(range(0, self.rank))
            bw_ker = ops.sum(bw_ker, axis=reduce_axis_kernel)  # Keep only sum per channel
            ebops = ops.sum(bw_inp[:, None] * bw_ker)

        if self.bq is not None:
            size = ops.cast(ops.prod(shape), self.dtype)
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
        return self.kq(self._kernel)

    @property
    def qbias(self):
        if self.bias is None:
            return None
        assert self.bq is not None
        return self.bq(self.bias)


class QConv1D(QBaseConv):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding='valid',
        data_format=None,
        dilation_rate=1,
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        kq_conf: None | QuantizerConfig = None,
        iq_conf: None | QuantizerConfig = None,
        bq_conf: None | QuantizerConfig = None,
        **kwargs,
    ):
        kwargs = gather_vars_to_kwargs('self')
        super().__init__(rank=1, **kwargs)

    def _compute_causal_padding(self):
        left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
        if self.data_format == 'channels_last':
            causal_padding = [[0, 0], [left_pad, 0], [0, 0]]
        else:
            causal_padding = [[0, 0], [0, 0], [left_pad, 0]]
        return causal_padding

    def call(self, inputs, training=None):
        padding = self.padding
        if self.padding == 'causal':
            # Apply causal padding to inputs.
            inputs = ops.pad(inputs, self._compute_causal_padding())
            padding = 'valid'

        qinputs = self.iq(inputs, training=training)
        qkernel = self.kq(self._kernel, training=training)

        outputs = ops.conv(
            qinputs,
            qkernel,
            strides=list(self.strides),  # type: ignore
            padding=padding,
            dilation_rate=self.dilation_rate,  # type: ignore
            data_format=self.data_format,
        )

        if self.use_bias:
            if self.data_format == 'channels_last':
                bias_shape = (1,) * (self.rank + 1) + (self.filters,)
            else:
                bias_shape = (1, self.filters) + (1,) * self.rank
            qbias = self.bq(self.bias, training=training)  # type: ignore
            qbias = ops.reshape(qbias, bias_shape)
            outputs += qbias  # type: ignore
        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class QConv2D(QBaseConv):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding='valid',
        data_format=None,
        dilation_rate=(1, 1),
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        kq_conf: None | QuantizerConfig = None,
        iq_conf: None | QuantizerConfig = None,
        bq_conf: None | QuantizerConfig = None,
        **kwargs,
    ):
        kwargs = gather_vars_to_kwargs('self')
        super().__init__(rank=2, **kwargs)


class QConv3D(QBaseConv):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding='valid',
        data_format=None,
        dilation_rate=(1, 1),
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        kq_conf: None | QuantizerConfig = None,
        iq_conf: None | QuantizerConfig = None,
        bq_conf: None | QuantizerConfig = None,
        **kwargs,
    ):
        kwargs = gather_vars_to_kwargs('self')
        super().__init__(rank=3, **kwargs)
