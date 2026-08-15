import numpy as np
from keras import ops
from keras.src import constraints, initializers, regularizers
from keras.src.backend.config import epsilon
from keras.src.layers.core.einsum_dense import _analyze_einsum_string

from ..quantizer.config import QuantizerConfig
from .core.einsum_dense import QEinsumDense


class QEinsumDenseBatchnorm(QEinsumDense):  # type: ignore
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
        normalize_axes: str | None = None,
        momentum=0.99,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer='zeros',
        gamma_initializer='ones',
        moving_mean_initializer='zeros',
        moving_variance_initializer='ones',
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        synchronized=False,
        **kwargs,
    ):
        super().__init__(
            equation=equation,
            output_shape=output_shape,
            activation=activation,
            bias_axes=bias_axes,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            iq_conf=iq_conf,
            kq_conf=kq_conf,
            bq_conf=bq_conf,
            **kwargs,
        )
        self.synchronized = synchronized
        self.momentum = float(momentum)
        self.epsilon = float(epsilon)
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = initializers.get(
            moving_variance_initializer,
        )
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.supports_masking = False

        normalize_axes = normalize_axes or bias_axes
        assert normalize_axes is not None, 'Either normalize_axes or bias_axes must be provided.'
        self.normalize_axes = normalize_axes
        self._check_normalize_axes()

    def _check_normalize_axes(self):
        kernel_axes, output_axes = self.equation.split(',')[1].split('->')
        for c in self.normalize_axes:
            if c not in kernel_axes:
                raise ValueError(f'Axis {c} not found in kernel axes {kernel_axes}')
            if c not in output_axes:
                raise ValueError(f'Axis {c} not found in output axes {output_axes}')
            if self.bias_axes and c not in self.bias_axes:
                raise ValueError(f'Axis {c} not found in bias axes {self.bias_axes}')

    @property
    def kq(self):
        return self._kq

    @property
    def bq(self):
        return self._bq

    def _compute_fused_einsum_specs(self, input_shape):
        # Replace '...' with '0' to match the behavior of keras
        equation = self.equation.replace('...', '0')
        kernel_axes, output_axes = equation.split(',')[1].split('->')
        shape_data = _analyze_einsum_string(equation, self.normalize_axes, input_shape, self.partial_output_shape)
        _kernel_shape, _norm_bias_cast_shape, output_shape, *_io_axes = shape_data
        assert _norm_bias_cast_shape is not None
        if '0' in output_axes:
            _pad_len = len(output_shape) - len(output_axes) + 1
            output_axes = output_axes.replace('0', '.' * _pad_len)

        assert len(output_axes) == len(output_shape), f'{output_axes} != {output_shape}'
        # Order normalize_axes to match kernel_axes
        normalize_axes = ''.join(c for c in kernel_axes if c in self.normalize_axes)

        _reduction_axis = tuple(i for i, c in enumerate(output_axes) if c not in normalize_axes)
        norm_shape = tuple(output_shape[i] for i, c in enumerate(output_axes) if c in normalize_axes)

        assert np.prod(norm_shape) == np.prod(_norm_bias_cast_shape), f'{norm_shape} != {_norm_bias_cast_shape}'

        fused_kernel_equation = f'{kernel_axes},{normalize_axes}->{kernel_axes}'

        self._reduction_axes = _reduction_axis
        self._norm_offset_cast_shape = _norm_bias_cast_shape
        self.fused_kernel_equation = fused_kernel_equation
        return norm_shape

    def build(self, input_shape):
        super().build(input_shape)
        norm_shape = self._compute_fused_einsum_specs(input_shape)

        if self.scale:
            self.bn_gamma = self.add_weight(
                shape=norm_shape,
                name='gamma',
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True,
                autocast=False,
            )
        assert self.bias is not None
        self.bias.trainable = self.center

        self.moving_mean = self.add_weight(
            shape=norm_shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False,
            autocast=False,
        )
        self.moving_variance = self.add_weight(
            shape=norm_shape,
            name='moving_variance',
            initializer=self.moving_variance_initializer,
            trainable=False,
            autocast=False,
        )

    def get_fused_qkernel_and_qbias(self, training, mean, var):
        assert self.bias is not None
        assert self.bq is not None
        if self.scale:
            bn_gamma = self.bn_gamma
        else:
            bn_gamma = 1
        scaler = bn_gamma / ops.sqrt(var + epsilon())  # type: ignore
        kernel = self.kernel
        fused_kernel = ops.einsum(self.fused_kernel_equation, kernel, scaler)
        fused_qkernel = self.kq(fused_kernel, training=training)

        offset = -mean * scaler
        offset = ops.reshape(offset, self._norm_offset_cast_shape)
        fused_qbias = self.bq(self.bias + offset, training=training)
        return fused_qkernel, fused_qbias

    @property
    def qkernel(self):
        mean, var = self.moving_mean, self.moving_variance
        return self.get_fused_qkernel_and_qbias(training=False, mean=mean, var=var)[0]

    @property
    def qbias(self):
        mean, var = self.moving_mean, self.moving_variance
        return self.get_fused_qkernel_and_qbias(training=False, mean=mean, var=var)[1]

    def call(self, inputs, training=None):  # type: ignore
        if self.enable_iq:
            inputs = self.iq(inputs, training=training)

        if training and self.trainable:
            x = ops.einsum(self.equation, inputs, self.kernel)
            mean, var = ops.moments(
                x,
                self._reduction_axes,
                keepdims=False,
                synchronized=self.synchronized,
            )  # type: ignore
            self.moving_mean.assign(
                self.moving_mean * self.momentum + mean * (1.0 - self.momentum),
            )
            self.moving_variance.assign(
                self.moving_variance * self.momentum + var * (1.0 - self.momentum),
            )
        else:
            var, mean = self.moving_variance, self.moving_mean

        qkernel, qbias = self.get_fused_qkernel_and_qbias(training, mean, var)

        x = ops.einsum(self.equation, inputs, qkernel) + qbias
        if self.activation is not None:
            x = self.activation(x)
        return x
