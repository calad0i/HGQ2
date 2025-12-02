from collections.abc import Callable
from math import prod
from typing import TypedDict

import keras
from keras import ops
from keras.layers import Layer
from keras.saving import register_keras_serializable
from keras.src.ops.operation_utils import compute_conv_output_shape

from ...quantizer import QuantizerConfig
from .dense import QDenseT


class im2col_params_t(TypedDict):
    size: tuple[int, int]
    strides: tuple[int, int]
    dilation_rate: tuple[int, int]
    padding: str
    data_format: str


class QConvTBase(QDenseT):
    def __init__(
        self,
        filters: int,
        kernel_size: int | tuple[int, ...],
        rank: int,
        n_hl: int = 1,
        d_hl: int = 8,
        strides: int | tuple[int, ...] = 1,
        padding='valid',
        data_format=None,
        dilation_rate: int | tuple[int, ...] = 1,
        groups: int = 1,
        activation: Callable | None | str = None,
        subnn_activation: str | Callable | Layer | None = None,
        use_bias=True,
        activity_regularizer=None,
        toq_conf: QuantizerConfig | None = None,
        parallelization_factor: int = -1,
        **kwargs,
    ):
        super().__init__(
            n_out=filters,
            n_hl=n_hl,
            d_hl=d_hl,
            activation=activation,
            subnn_activation=subnn_activation,
            toq_conf=toq_conf,
            parallelization_factor=parallelization_factor,
            use_bias=use_bias,
            **kwargs,
        )

        def _normalize(x):
            if isinstance(x, int):
                return (x,) * rank
            assert len(x) == rank, f'Expected {rank}-D tuple/list, got {x}'
            return tuple(x)

        self.rank = rank
        self.strides = _normalize(strides)
        self.padding = padding
        self.data_format = data_format or 'channels_last'
        self.dilation_rate = _normalize(dilation_rate)
        self.groups = groups
        self.kernel_size = _normalize(kernel_size)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)

        assert groups == 1, 'Table-based grouped convolutions are not supported yet.'

    def call(self, x, training=None):
        if self.rank == 1:
            x = x[:, :, None]
        x = ops.image.extract_patches(x, **self.im2col_params)  # type: ignore
        if self.rank == 1:
            x = x[:, :, 0]
        return super().call(x, training=training)

    def build(self, input_shape):
        input_shape = tuple(input_shape)
        output_shape = compute_conv_output_shape(
            input_shape,
            filters=self.n_out,
            kernel_size=self.kernel_size,
            strides=self.strides,  # type: ignore
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,  # type: ignore
        )
        ch_in = input_shape[-1]
        n_in = ch_in * prod(self.kernel_size)
        _dense_in_shape = output_shape[:-1] + (n_in,)
        self.output_shape = (1, *output_shape[1:])

        self._build_im2col_params()

        super().build(_dense_in_shape)

    def _build_im2col_params(self):
        if self.rank == 2:
            self.im2col_params: im2col_params_t = {  # type: ignore
                'size': self.kernel_size,
                'strides': self.strides,
                'dilation_rate': self.dilation_rate,
                'padding': self.padding,
                'data_format': self.data_format,
            }
        elif self.rank == 1:
            if self.data_format == 'channels_last':
                size = self.kernel_size + (1,)
                strides = self.strides + (1,)
                dilation_rate = self.dilation_rate + (1,)
            else:
                size = (1,) + self.kernel_size
                strides = (1,) + self.strides
                dilation_rate = (1,) + self.dilation_rate

            self.im2col_params: im2col_params_t = {  # type: ignore
                'size': size,
                'strides': strides,
                'dilation_rate': dilation_rate,
                'padding': self.padding,
                'data_format': self.data_format,
            }
        else:
            raise ValueError('Only 1D and 2D convolutions are supported.')

    def _compute_ebops(self, shape: tuple[int, ...]):
        q_shape = self.output_shape[:-1] + (self.n_in,)
        return super()._compute_ebops(q_shape)

    @property
    def toq(self):
        return self._toq

    def get_config(self):
        config = {
            'n_out': self.ch_out,
            'n_hl': self.n_hl,
            'd_hl': self.d_hl,
            'subnn_activation': self.subnn_activation,
            'activation': self.activation,
            'toq_conf': self.toq.config,
            **super().get_config(),
        }
        return config


@register_keras_serializable(package='hgq2')
class QConvT1D(QConvTBase):
    def __init__(
        self,
        filters: int,
        kernel_size: int,
        n_hl: int = 1,
        d_hl: int = 8,
        strides: int | tuple[int] = 1,
        padding='valid',
        data_format=None,
        dilation_rate: int | tuple[int] = 1,
        groups: int = 1,
        activation: Callable | None | str = None,
        subnn_activation: str | Callable | Layer | None = None,
        use_bias=True,
        activity_regularizer=None,
        toq_conf: QuantizerConfig | None = None,
        parallelization_factor: int = -1,
        **kwargs,
    ):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            rank=1,
            n_hl=n_hl,
            d_hl=d_hl,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            subnn_activation=subnn_activation,
            use_bias=use_bias,
            activity_regularizer=activity_regularizer,
            toq_conf=toq_conf,
            parallelization_factor=parallelization_factor,
            **kwargs,
        )


@register_keras_serializable(package='hgq2')
class QConvT2D(QConvTBase):
    def __init__(
        self,
        filters: int,
        kernel_size: int | tuple[int, int],
        n_hl: int = 1,
        d_hl: int = 8,
        strides: int | tuple[int, int] = 1,
        padding='valid',
        data_format=None,
        dilation_rate: int | tuple[int, int] = 1,
        groups: int = 1,
        activation: Callable | None | str = None,
        subnn_activation: str | Callable | Layer | None = None,
        use_bias=True,
        activity_regularizer=None,
        toq_conf: QuantizerConfig | None = None,
        parallelization_factor: int = -1,
        **kwargs,
    ):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            rank=2,
            n_hl=n_hl,
            d_hl=d_hl,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            subnn_activation=subnn_activation,
            use_bias=use_bias,
            activity_regularizer=activity_regularizer,
            toq_conf=toq_conf,
            parallelization_factor=parallelization_factor,
            **kwargs,
        )
