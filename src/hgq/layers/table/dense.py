import string
from collections.abc import Callable
from math import prod, sqrt

import keras
from keras import ops
from keras.layers import Layer
from keras.src.layers.core.einsum_dense import _analyze_einsum_string, _analyze_quantization_info

from ...quantizer import Quantizer, QuantizerConfig
from ..core import QLayerBaseSingleInput
from ..core.einsum_dense import _einsum_free0_size


def _einsum_idxs_from_keras(equation: str, input_shape: tuple[int | None, ...]) -> tuple[str, str, str]:
    custom_gradient_equation = _analyze_quantization_info(equation, input_shape)[8]
    output_idxs, weight_to_input_idxs = custom_gradient_equation.split(',')
    weight_idxs, input_idxs = weight_to_input_idxs.split('->')
    return input_idxs, weight_idxs, output_idxs


class QDenseT(QLayerBaseSingleInput):
    def __init__(
        self,
        n_out: int,
        n_hl: int = 1,
        d_hl: int = 8,
        activation: Callable | None | str = None,
        subnn_activation: str | Callable | Layer | None = 'tanh',
        toq_conf: QuantizerConfig | None = None,
        parallelization_factor: int = -1,
        use_bias: bool = True,
        batch_norm: bool = False,
        bn_center: bool = True,
        bn_scale: bool = True,
        bn_momentum: float = 0.99,
        bn_epsilon: float = 0.001,
        table_idxs: int | tuple[int, int] = (6, 5),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_out = n_out
        self.d_hl = d_hl
        self.use_bias = use_bias
        self.subnn_activation = keras.activations.get(subnn_activation)
        self.parallelization_factor = parallelization_factor
        self.enable_bn = batch_norm
        self.table_idxs = (table_idxs, table_idxs) if isinstance(table_idxs, int) else table_idxs
        self.bn_args = {
            'center': bn_center,
            'scale': bn_scale,
            'momentum': bn_momentum,
            'epsilon': bn_epsilon,
        }
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
            # layers.extend(
            #     [
            #         keras.layers.EinsumDense(
            #             '...iod,iodD->...ioD',
            #             _shape,
            #             bias_axes=bias_axes,
            #         ),
            #         # keras.layers.Reshape((n_in * self.n_out * self.d_hl,)),
            #         keras.layers.BatchNormalization(axis=-1),
            #         # keras.layers.Reshape((n_in, self.n_out, self.d_hl)),
            #         keras.layers.Activation(self.subnn_activation),
            #     ]
            # )
        bias_axes = 'io' if self.use_bias else None
        l_out = keras.layers.EinsumDense('...iod,iod->...io', (n_in, self.n_out), 'linear', bias_axes=bias_axes)
        layers.append(l_out)
        module = keras.models.Sequential(layers)
        return module

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

        if self.enable_bn:
            self.bn_module = keras.layers.BatchNormalization(
                axis=-1,
                **self.bn_args,
            )
            self.bn_module.build(input_shape + (self.n_out,))

        super().build(input_shape)

    def call(self, x, training=None):
        n_in = ops.shape(x)[-1]
        x = ops.broadcast_to(x[..., None], (*ops.shape(x)[:-1], n_in, self.n_out))  # (B, N_in, N_out)
        if self.enable_iq:
            x = self.iq(x)
        x = x[..., None]  # (B, ..., N_in, N_out, 1)
        table_out = self.module(x, training=training)  # (B, ..., N_in, N_out, 1) -> (B, ..., N_in, N_out)

        if self.enable_bn:
            table_out = self.bn_module(table_out, training=training) / sqrt(self.n_in)

        return self.activation(ops.sum(self.toq(table_out), axis=-2))

    def _compute_ebops(self, shape: tuple[int, ...]):
        q_shape = shape + (self.n_out,)
        bits_in = self.iq.fbits_(q_shape)
        bits_out = self.toq.fbits_(q_shape)

        B, b = self.table_idxs

        small_lut_count = ops.where(bits_in >= b, 2 ** (bits_in - b), bits_in / b)  # type: ignore

        large_lut_count = ops.dot(ops.ravel(small_lut_count), ops.ravel(bits_out)) * 2 ** (b - B)  # type: ignore

        return (large_lut_count + ops.sum(bits_out)) * self.parallelization_factor / self.n_parallel

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
            'use_bias': self.use_bias,
            'parallelization_factor': self.parallelization_factor,
            'batch_norm': self.enable_bn,
            'bn_center': self.bn_args['center'],
            'bn_scale': self.bn_args['scale'],
            'bn_momentum': self.bn_args['momentum'],
            'bn_epsilon': self.bn_args['epsilon'],
            'table_idxs': self.table_idxs,
            **super().get_config(),
        }
        return config


class QEinsumDenseT(QLayerBaseSingleInput):
    def __init__(
        self,
        equation: str,
        output_shape: int | tuple[int, ...],
        n_hl: int = 1,
        d_hl: int = 8,
        activation: Callable | None | str = None,
        subnn_activation: str | Callable | Layer | None = 'tanh',
        bias_axes: str | None = None,
        normalize_axes: str | None = None,
        toq_conf: QuantizerConfig | None = None,
        parallelization_factor: int = -1,
        batch_norm: bool = False,
        bn_center: bool = True,
        bn_scale: bool = True,
        bn_momentum: float = 0.99,
        bn_epsilon: float = 0.001,
        table_idxs: int | tuple[int, int] = (6, 5),
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert n_hl >= 0
        self.equation = equation
        self.partial_output_shape = (output_shape,) if isinstance(output_shape, int) else tuple(output_shape)
        self.n_hl = n_hl
        self.d_hl = d_hl
        self.subnn_activation = keras.activations.get(subnn_activation)
        self.activation = keras.activations.get(activation)
        self.bias_axes = bias_axes
        self.normalize_axes = normalize_axes
        self.parallelization_factor = parallelization_factor
        self.enable_bn = batch_norm
        self.table_idxs = (table_idxs, table_idxs) if isinstance(table_idxs, int) else table_idxs
        self.bn_center = bn_center
        self.bn_scale = bn_scale
        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epsilon

        toq_conf = toq_conf or QuantizerConfig(place='table')
        self._toq = Quantizer(toq_conf)

    def _build_module(self, presum_idxs: str, presum_shape: tuple[int | None, ...], weight_idxs: str):
        layers = []
        labels = iter(sorted(set(string.ascii_letters) - set(presum_idxs) - set(weight_idxs)))
        hidden_in = next(labels)
        hidden_out = next(labels)
        hidden_in_dim = 1
        for _ in range(self.n_hl):
            bias_axes = f'{weight_idxs}{hidden_out}' if self.bias_axes is not None and not self.enable_bn else None
            eq = f'{presum_idxs}{hidden_in},{weight_idxs}{hidden_in}{hidden_out}->{presum_idxs}{hidden_out}'
            layers.append(
                keras.layers.EinsumDense(
                    eq,
                    output_shape=presum_shape[1:] + (self.d_hl,),
                    activation=self.subnn_activation,
                    bias_axes=bias_axes,
                )
            )
            hidden_in_dim = self.d_hl
            hidden_in, hidden_out = hidden_out, hidden_in

        bias_axes = weight_idxs if self.bias_axes is not None and not self.enable_bn else None
        layers.append(
            keras.layers.EinsumDense(
                f'{presum_idxs}{hidden_in},{weight_idxs}{hidden_in}->{presum_idxs}',
                output_shape=presum_shape[1:],
                activation='linear',
                bias_axes=bias_axes,
            )
        )

        module = keras.models.Sequential(layers)
        module.build(presum_shape + (hidden_in_dim,))
        return module

    def build(self, input_shape):
        input_shape = tuple(input_shape)
        kernel_shape, _, full_output_shape = _analyze_einsum_string(
            self.equation,
            self.bias_axes,
            input_shape,
            self.partial_output_shape,
        )
        input_idxs, weight_idxs, output_idxs = _einsum_idxs_from_keras(self.equation, input_shape)
        kernel_shape = tuple(kernel_shape)
        full_output_shape = tuple(full_output_shape)
        self.full_output_shape = full_output_shape

        extra_idxs = ''.join(idx for idx in weight_idxs if idx not in input_idxs)
        if any(idx not in output_idxs for idx in extra_idxs):
            raise ValueError(f'Kernel-only contract axes are not supported: equation={self.equation}')
        presum_idxs = input_idxs + extra_idxs
        kernel_dim = dict(zip(weight_idxs, kernel_shape))
        extra_shape = tuple(kernel_dim[idx] for idx in extra_idxs)
        presum_shape = input_shape + extra_shape
        self._contract_axes = tuple(i for i, idx in enumerate(input_idxs) if idx not in output_idxs)
        post_reduction_idxs = ''.join(idx for idx in input_idxs if idx in output_idxs) + extra_idxs
        post_axis = {idx: axis for axis, idx in enumerate(post_reduction_idxs)}
        self._output_transpose = tuple(post_axis[idx] for idx in output_idxs)
        contract_shape = tuple(presum_shape[i] for i in self._contract_axes)
        if any(dim is None for dim in contract_shape):
            raise ValueError(
                f'Cannot build {self.__class__.__name__} with unknown contract axis size: '
                f'equation={self.equation}, input_shape={input_shape}, contract_shape={contract_shape}'
            )
        self.n_in = prod(contract_shape) if contract_shape else 1
        self.n_parallel = _einsum_free0_size(self.equation, input_shape)
        if self.parallelization_factor < 0:
            self.parallelization_factor = self.n_parallel

        presum_axis = {idx: axis for axis, idx in enumerate(presum_idxs)}
        self._kernel_axes = tuple(presum_axis[idx] for idx in weight_idxs)
        self._bcast = (
            input_shape + (1,) * len(extra_idxs),
            presum_shape,
            tuple(i for i, dim in enumerate(input_shape) if dim is None),
        )

        if self.enable_iq and not self.iq.built:
            self.iq.build(presum_shape)
        self.toq.build(presum_shape)
        self.module = self._build_module(presum_idxs, presum_shape, weight_idxs)

        if self.enable_bn:
            normalize_axes = self.normalize_axes
            if normalize_axes is None:
                normalize_axes = self.bias_axes or ''.join(dim for dim in weight_idxs if dim in output_idxs)
            self.normalize_axes = normalize_axes
            self._bn_axes = tuple(presum_axis[idx] for idx in normalize_axes)
            bn_shape = tuple(presum_shape[axis] for axis in self._bn_axes)
            bn_broadcast_shape = [1] * len(presum_shape)
            for axis, dim in zip(self._bn_axes, bn_shape):
                bn_broadcast_shape[axis] = dim
            self._bn = (
                tuple(bn_broadcast_shape),
                tuple(i for i in range(len(presum_shape)) if i not in self._bn_axes),
            )
            if self.bn_scale:
                self.bn_gamma = self.add_weight(
                    shape=bn_shape,
                    name='bn_gamma',
                    initializer='ones',
                    trainable=True,
                    autocast=False,
                )
            if self.bn_center:
                self.bn_beta = self.add_weight(
                    shape=bn_shape,
                    name='bn_beta',
                    initializer='zeros',
                    trainable=True,
                    autocast=False,
                )
            self.moving_mean = self.add_weight(
                shape=bn_shape,
                name='moving_mean',
                initializer='zeros',
                trainable=False,
                autocast=False,
            )
            self.moving_variance = self.add_weight(
                shape=bn_shape,
                name='moving_variance',
                initializer='ones',
                trainable=False,
                autocast=False,
            )

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return tuple(_analyze_einsum_string(self.equation, self.bias_axes, tuple(input_shape), self.partial_output_shape)[2])

    def _broadcast_shapes(self, input_shape: tuple[int, ...] | None):
        reshape_shape, broadcast_shape, dynamic_axes = self._bcast
        if not dynamic_axes:
            return reshape_shape, broadcast_shape
        reshape_shape = list(reshape_shape)
        broadcast_shape = list(broadcast_shape)
        assert input_shape is not None
        for axis in dynamic_axes:
            reshape_shape[axis] = broadcast_shape[axis] = input_shape[axis]
        return tuple(reshape_shape), tuple(broadcast_shape)

    def _apply_batch_norm(self, table_out, training=None):
        if not self.enable_bn:
            return table_out

        bn_broadcast_shape, bn_reduction_axes = self._bn
        if training and self.trainable:
            mean, var = ops.moments(table_out, bn_reduction_axes, keepdims=False)
            self.moving_mean.assign(self.moving_mean * self.bn_momentum + mean * (1.0 - self.bn_momentum))
            self.moving_variance.assign(self.moving_variance * self.bn_momentum + var * (1.0 - self.bn_momentum))
        else:
            mean, var = self.moving_mean, self.moving_variance

        scaler = 1
        if self.bn_scale:
            scaler = self.bn_gamma
        scaler = scaler / ops.sqrt(var + self.bn_epsilon)  # type: ignore
        offset = -mean * scaler
        if self.bn_center:
            offset = offset + self.bn_beta

        scaler = ops.reshape(scaler, bn_broadcast_shape)
        offset = ops.reshape(offset, bn_broadcast_shape)
        return (table_out * scaler + offset) / sqrt(self.n_in)

    def call(self, x, training=None):
        input_shape = ops.shape(x) if self._bcast[2] else None
        reshape_shape, broadcast_shape = self._broadcast_shapes(input_shape)
        x = ops.broadcast_to(ops.reshape(x, reshape_shape), broadcast_shape)
        if self.enable_iq:
            x = self.iq(x, training=training)
        table_out = self.module(x[..., None], training=training)
        table_out = self._apply_batch_norm(table_out, training=training)

        table_out = self.toq(table_out, training=training)
        out = ops.sum(table_out, axis=self._contract_axes)
        out = ops.transpose(out, self._output_transpose)
        return self.activation(out)

    def _compute_ebops(self, shape: tuple[int, ...]):
        _, presum_shape = self._broadcast_shapes(shape)
        bits_in = self.iq.fbits_(presum_shape)
        bits_out = self.toq.fbits_(presum_shape)

        B, b = self.table_idxs
        small_lut_count = ops.where(bits_in >= b, 2 ** (bits_in - b), bits_in / b)  # type: ignore
        large_lut_count = ops.dot(ops.ravel(small_lut_count), ops.ravel(bits_out)) * 2 ** (b - B)  # type: ignore

        return (large_lut_count + ops.sum(bits_out)) * self.parallelization_factor / self.n_parallel

    @property
    def toq(self):
        return self._toq

    def get_config(self):
        config = {
            'equation': self.equation,
            'output_shape': self.partial_output_shape,
            'n_hl': self.n_hl,
            'd_hl': self.d_hl,
            'subnn_activation': self.subnn_activation,
            'activation': self.activation,
            'bias_axes': self.bias_axes,
            'normalize_axes': self.normalize_axes,
            'toq_conf': self.toq.config,
            'parallelization_factor': self.parallelization_factor,
            'batch_norm': self.enable_bn,
            'bn_center': self.bn_center,
            'bn_scale': self.bn_scale,
            'bn_momentum': self.bn_momentum,
            'bn_epsilon': self.bn_epsilon,
            'table_idxs': self.table_idxs,
            **super().get_config(),
        }
        return config
