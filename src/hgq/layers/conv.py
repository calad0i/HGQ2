from math import prod

from keras import ops
from keras.src.layers.convolutional.base_conv import BaseConv

from ..quantizer import Quantizer
from ..quantizer.config import QuantizerConfig
from ..utils.misc import gather_vars_to_kwargs
from .core.base import QLayerBase, QLayerBaseSingleInput


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

        outputs_shape = self.compute_output_shape(input_shape)
        if None in outputs_shape[1:]:
            self.n_parallel = None
        else:
            self.n_parallel = prod(outputs_shape[1:-1]) if self.data_format == 'channels_last' else prod(outputs_shape[2:])

        if self.parallelization_factor <= 0:
            assert self.n_parallel is not None
            self.parallelization_factor = self.n_parallel

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
        ebops = ops.sum(self.convolution_op(bw_inp, bw_ker))
        ebops *= self.parallelization_factor / self.n_parallel  # type: ignore

        if self.bq is not None:
            size = ops.cast(ops.prod(shape[:-1]) * self.filters, self.dtype)
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
        is_causal = self.padding == 'causal'

        if is_causal:
            # Apply causal padding to inputs.
            inputs = ops.pad(inputs, self._compute_causal_padding())
            self.padding = 'valid'
        ret = super().call(inputs, training=training)
        if is_causal:
            self.padding = 'causal'
        return ret


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
        strides=(1, 1, 1),
        padding='valid',
        data_format=None,
        dilation_rate=(1, 1, 1),
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


class QBaseDepthwiseConv:
    """Depthwise convolution as a grouped QConv: groups = in_channels and filters = in_channels * depth_multiplier,
    both fixed in build once the input channels are known. Output channel c * depth_multiplier + m is kernel m of
    input channel c, as in keras' DepthwiseConv, but the kernel has the grouped QConv layout
    (*kernel_size, 1, in_channels * depth_multiplier)."""

    def __init__(
        self,
        kernel_size,
        strides,
        padding,
        depth_multiplier,
        data_format,
        dilation_rate,
        activation,
        use_bias,
        depthwise_initializer,
        bias_initializer,
        depthwise_regularizer,
        bias_regularizer,
        activity_regularizer,
        depthwise_constraint,
        bias_constraint,
        **kwargs,
    ):
        super().__init__(  # type: ignore
            filters=depth_multiplier,  # in_channels * depth_multiplier, set in build
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=depthwise_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=depthwise_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=depthwise_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )
        self.depth_multiplier = depth_multiplier

    def build(self, input_shape):
        n_chan = input_shape[-1] if self.data_format == 'channels_last' else input_shape[1]
        self.groups = n_chan
        self.filters = n_chan * self.depth_multiplier
        super().build(input_shape)  # type: ignore

    def get_config(self):
        config = super().get_config()  # type: ignore
        del config['filters'], config['groups']
        for key in ('initializer', 'regularizer', 'constraint'):
            config[f'depthwise_{key}'] = config.pop(f'kernel_{key}')
        config['depth_multiplier'] = self.depth_multiplier
        return config


class QDepthwiseConv1D(QBaseDepthwiseConv, QConv1D):
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
        super().__init__(**kwargs)


class QDepthwiseConv2D(QBaseDepthwiseConv, QConv2D):
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
        super().__init__(**kwargs)


class QBaseSeparableConv(QLayerBase):
    """Separable convolution as two QConv layers in series: a QDepthwiseConv (`depthwise`, without bias) and a 1x1
    QConv (`pointwise`). The input quantizer of the pointwise conv (pq) quantizes the depthwise result, so the
    intermediate tensor has a definite fixed-point type. Both sublayers count their own EBOPs."""

    _depthwise_cls: type
    _pointwise_cls: type

    def __init__(
        self,
        filters,
        kernel_size,
        strides,
        padding,
        data_format,
        dilation_rate,
        depth_multiplier,
        activation,
        use_bias,
        depthwise_initializer,
        pointwise_initializer,
        bias_initializer,
        depthwise_regularizer,
        pointwise_regularizer,
        bias_regularizer,
        activity_regularizer,
        depthwise_constraint,
        pointwise_constraint,
        bias_constraint,
        dkq_conf: None | QuantizerConfig = None,
        pkq_conf: None | QuantizerConfig = None,
        iq_conf: None | QuantizerConfig = None,
        pq_conf: None | QuantizerConfig = None,
        bq_conf: None | QuantizerConfig = None,
        parallelization_factor=-1,
        **kwargs,
    ):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        common = {
            'enable_ebops': self.enable_ebops,
            'beta0': self._beta0,
            'enable_oq': False,
            'parallelization_factor': parallelization_factor,
        }
        self._depthwise = self._depthwise_cls(
            kernel_size,
            strides=strides,
            padding=padding,
            depth_multiplier=depth_multiplier,
            data_format=data_format,
            dilation_rate=dilation_rate,
            use_bias=False,
            depthwise_initializer=depthwise_initializer,
            depthwise_regularizer=depthwise_regularizer,
            depthwise_constraint=depthwise_constraint,
            kq_conf=dkq_conf,
            iq_conf=iq_conf,
            enable_iq=self.enable_iq,
            name=f'{self.name}_depthwise',
            **common,
        )
        self._pointwise = self._pointwise_cls(
            filters,
            1,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=pointwise_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=pointwise_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=pointwise_constraint,
            bias_constraint=bias_constraint,
            kq_conf=pkq_conf,
            iq_conf=pq_conf,
            bq_conf=bq_conf,
            enable_iq=True,
            name=f'{self.name}_pointwise',
            **common,
        )
        self.filters = filters
        self.depth_multiplier = depth_multiplier
        self.parallelization_factor = parallelization_factor

    @property
    def depthwise(self):
        return self._depthwise

    @property
    def pointwise(self):
        return self._pointwise

    @property
    def dkq(self):
        return self._depthwise.kq

    @property
    def pkq(self):
        return self._pointwise.kq

    @property
    def pq(self):
        return self._pointwise.iq

    @property
    def bq(self):
        return self._pointwise.bq

    def build(self, input_shape):
        self._depthwise.build(input_shape)
        self._pointwise.build(self._depthwise.compute_output_shape(input_shape))
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return self._pointwise.compute_output_shape(self._depthwise.compute_output_shape(input_shape))

    def call(self, inputs, training=None):
        x = self._depthwise(inputs, training=training)
        return self._pointwise(x, training=training)

    def _post_build(self):
        # The input quantizer is the depthwise sublayer's
        for sublayer in self._flatten_layers():
            assert sublayer.built, f'Sublayer {sublayer.name} is not built for {self.name}'

    @property
    def ebops(self):
        return self._depthwise.ebops + self._pointwise.ebops  # type: ignore

    def get_config(self):
        config = super().get_config()
        dw_config, pw_config = self._depthwise.get_config(), self._pointwise.get_config()
        config.update(
            {
                'filters': self.filters,
                'depth_multiplier': self.depth_multiplier,
                'parallelization_factor': self.parallelization_factor,
                'dkq_conf': self.dkq.config,
                'pkq_conf': self.pkq.config,
                'iq_conf': self._depthwise.iq.config if self.enable_iq else None,
                'pq_conf': self.pq.config,
                'bq_conf': self.bq.config if self.bq is not None else None,
            }
        )
        for key in ('kernel_size', 'strides', 'padding', 'data_format', 'dilation_rate'):
            config[key] = dw_config[key]
        for key in ('initializer', 'regularizer', 'constraint'):
            config[f'depthwise_{key}'] = dw_config[f'depthwise_{key}']
            config[f'pointwise_{key}'] = pw_config[f'kernel_{key}']
        for key in ('activation', 'use_bias', 'bias_initializer', 'bias_regularizer', 'bias_constraint'):
            config[key] = pw_config[key]
        return config


class QSeparableConv1D(QBaseSeparableConv):
    _depthwise_cls = QDepthwiseConv1D
    _pointwise_cls = QConv1D

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
        super().__init__(**kwargs)


class QSeparableConv2D(QBaseSeparableConv):
    _depthwise_cls = QDepthwiseConv2D
    _pointwise_cls = QConv2D

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
        super().__init__(**kwargs)
