from math import prod

from keras import constraints, initializers, ops, regularizers
from keras.layers import Dense

from ...quantizer import Quantizer
from ...quantizer.config import QuantizerConfig
from ...utils.misc import gather_vars_to_kwargs
from .base import QLayerBaseSingleInput


class QDense(QLayerBaseSingleInput, Dense):
    def __init__(
        self,
        units,
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
        parallelization_factor=-1,
        **kwargs,
    ):
        kwargs = gather_vars_to_kwargs('self|kq_conf|bq_conf|parallelization_factor')
        super().__init__(lora_rank=None, **kwargs)

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
        n_parallel = prod(input_shape[1:-1])
        self.n_parallel = n_parallel
        self.parallelization_factor = self.parallelization_factor if self.parallelization_factor > 0 else n_parallel

        self.kq.build(ops.shape(self._kernel))
        if self.bias is not None:
            assert self.bq is not None
            self.bq.build(ops.shape(self.bias))

    def call(self, inputs, training=None):
        if self.enable_iq:
            inputs = self.iq(inputs, training=training)
        x = ops.matmul(inputs, self.qkernel)
        if self.bias is not None:
            x = ops.add(x, self.qbias)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def _compute_ebops(self, shape):
        bw_inp = self.iq.bits_(shape)
        bw_ker = self.kq.bits_(ops.shape(self.kernel))
        ebops = ops.sum(ops.matmul(bw_inp, bw_ker))
        ebops = ebops * self.parallelization_factor / self.n_parallel  # type: ignore
        if self.bq is not None:
            bw_bias = self.bq.bits_(ops.shape(self.bias))
            size = ops.cast(ops.prod(shape), self.dtype)
            ebops = ebops + ops.mean(bw_bias) * size  # type: ignore

        return ebops

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


class QBatchNormDense(QDense):
    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        scale=True,
        momentum=0.99,
        epsilon=1e-3,
        bn_gamma_initializer='ones',
        moving_mean_initializer='zeros',
        moving_variance_initializer='ones',
        bn_gamma_regularizer=None,
        bn_gamma_constraint=None,
        synchronized=False,
        kq_conf: None | QuantizerConfig = None,
        iq_conf: None | QuantizerConfig = None,
        bq_conf: None | QuantizerConfig = None,
        **kwargs,
    ):
        assert use_bias, 'BatchNormDense must have `use_bias=True`'
        super().__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            kq_conf=kq_conf,
            iq_conf=iq_conf,
            bq_conf=bq_conf,
            **kwargs,
        )

        self.scale = scale
        self.momentum = float(momentum)
        self.bn_gamma_initializer = initializers.get(bn_gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = initializers.get(moving_variance_initializer)
        self.bn_gamma_regularizer = regularizers.get(bn_gamma_regularizer)
        self.bn_gamma_constraint = constraints.get(bn_gamma_constraint)
        self.epsilon = epsilon
        self.synchronized = synchronized

    def build(self, input_shape):
        super().build(input_shape)
        shape = (input_shape[-1],)
        self.reduction_axis = tuple(range(len(input_shape) - 1))
        if self.scale:
            self.bn_gamma = self.add_weight(
                shape=shape,
                name='bn_gamma',
                initializer=self.bn_gamma_initializer,
                regularizer=self.bn_gamma_regularizer,
                constraint=self.bn_gamma_constraint,
                trainable=True,
                autocast=False,
            )

        self.moving_mean = self.add_weight(
            shape=shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False,
            autocast=False,
        )
        self.moving_variance = self.add_weight(
            shape=shape,
            name='moving_variance',
            initializer=self.moving_variance_initializer,
            trainable=False,
            autocast=False,
        )

    def get_fused_qkernel_and_qbias(self, mean, var):
        assert self.bias is not None
        assert self.bq is not None
        if self.scale:
            bn_gamma = self.bn_gamma
        else:
            bn_gamma = 1
        scaler = bn_gamma / ops.sqrt(var + self.epsilon)  # type: ignore
        kernel = ops.cast(self.kernel, self.kernel.dtype)  # type: ignore
        fused_qkernel = self.kq(scaler[:, None] * kernel)  # type: ignore

        offset = -ops.dot(mean, kernel)
        fused_qbias = self.bq(self.bias + offset)

        return fused_qkernel, fused_qbias

    def call(self, inputs, training=None):
        if self.enable_iq:
            inputs = self.iq(inputs, training=training)

        if training and self.trainable:
            mean, var = ops.moments(inputs, self.reduction_axis, keepdims=False, synchronized=self.synchronized)  # type: ignore
            self.moving_mean.assign(
                self.moving_mean * self.momentum + mean * (1.0 - self.momentum),
            )
            self.moving_variance.assign(
                self.moving_variance * self.momentum + var * (1.0 - self.momentum),
            )
            # _var, _mean = ops.cast(var, self.moving_variance.dtype), ops.cast(mean, self.moving_mean.dtype)
            # var = ops.stop_gradient(_var - var) + var
            # mean = ops.stop_gradient(_mean - mean) + mean
        else:
            var, mean = self.moving_variance, self.moving_mean

        qkernel, qbias = self.get_fused_qkernel_and_qbias(mean, var)
        x = ops.matmul(inputs, qkernel)
        x = ops.add(x, qbias)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'kq_conf': self.kq.config,
                'bq_conf': self.bq.config if self.bq is not None else None,
                'scale': self.scale,
                'momentum': self.momentum,
                'epsilon': self.epsilon,
                'bn_gamma_initializer': self.bn_gamma_initializer,
                'moving_mean_initializer': self.moving_mean_initializer,
                'moving_variance_initializer': self.moving_variance_initializer,
                'bn_gamma_regularizer': self.bn_gamma_regularizer,
                'bn_gamma_constraint': self.bn_gamma_constraint,
                'synchronized': self.synchronized,
            }
        )
        return config

    @property
    def qkernel(self):
        return self.get_fused_qkernel_and_qbias(self.moving_mean, self.moving_variance)[0]

    @property
    def qbias(self):
        return self.get_fused_qkernel_and_qbias(self.moving_mean, self.moving_variance)[1]
