from keras import ops
from keras.layers import (
    AveragePooling1D,
    AveragePooling2D,
    AveragePooling3D,
    GlobalAveragePooling1D,
    GlobalAveragePooling2D,
    GlobalAveragePooling3D,
    GlobalMaxPooling1D,
    GlobalMaxPooling2D,
    GlobalMaxPooling3D,
    MaxPooling1D,
    MaxPooling2D,
    MaxPooling3D,
)

from ..config import QuantizerConfig
from .core.base import QLayerBaseSingleInput


class QBasePooling(QLayerBaseSingleInput):
    def _compute_ebops(self, shape):
        bw_inp = self.iq.bits_(shape)
        return ops.sum(bw_inp)

    def call(self, inputs, training=None):
        if self.enable_iq:
            inputs = self.iq(inputs, training=training)
        return super().call(inputs)


class QAveragePooling1D(QBasePooling, AveragePooling1D):
    def __init__(
        self,
        pool_size,
        strides=None,
        padding='valid',
        data_format=None,
        name=None,
        iq_conf: QuantizerConfig | None = None,
        oq_conf: QuantizerConfig | None = None,
        **kwargs,
    ):
        super().__init__(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            name=name,
            iq_conf=iq_conf,
            oq_conf=oq_conf,
            **kwargs,
        )


class QAveragePooling2D(QBasePooling, AveragePooling2D):
    def __init__(
        self,
        pool_size,
        strides=None,
        padding='valid',
        data_format=None,
        name=None,
        iq_conf: QuantizerConfig | None = None,
        oq_conf: QuantizerConfig | None = None,
        **kwargs,
    ):
        super().__init__(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            name=name,
            iq_conf=iq_conf,
            oq_conf=oq_conf,
            **kwargs,
        )


class QAveragePooling3D(QBasePooling, AveragePooling3D):
    def __init__(
        self,
        pool_size,
        strides=None,
        padding='valid',
        data_format=None,
        name=None,
        iq_conf: QuantizerConfig | None = None,
        oq_conf: QuantizerConfig | None = None,
        **kwargs,
    ):
        super().__init__(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            name=name,
            iq_conf=iq_conf,
            oq_conf=oq_conf,
            **kwargs,
        )


class QMaxPooling1D(QBasePooling, MaxPooling1D):
    def __init__(
        self,
        pool_size,
        strides=None,
        padding='valid',
        data_format=None,
        name=None,
        iq_conf: QuantizerConfig | None = None,
        oq_conf: QuantizerConfig | None = None,
        **kwargs,
    ):
        super().__init__(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            name=name,
            iq_conf=iq_conf,
            oq_conf=oq_conf,
            **kwargs,
        )


class QMaxPooling2D(QBasePooling, MaxPooling2D):
    def __init__(
        self,
        pool_size,
        strides=None,
        padding='valid',
        data_format=None,
        name=None,
        iq_conf: QuantizerConfig | None = None,
        oq_conf: QuantizerConfig | None = None,
        **kwargs,
    ):
        super().__init__(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            name=name,
            iq_conf=iq_conf,
            oq_conf=oq_conf,
            **kwargs,
        )


class QMaxPooling3D(QBasePooling, MaxPooling3D):
    def __init__(
        self,
        pool_size,
        strides=None,
        padding='valid',
        data_format=None,
        name=None,
        iq_conf: QuantizerConfig | None = None,
        oq_conf: QuantizerConfig | None = None,
        **kwargs,
    ):
        super().__init__(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            name=name,
            iq_conf=iq_conf,
            oq_conf=oq_conf,
            **kwargs,
        )


class QGlobalAveragePooling1D(QBasePooling, GlobalAveragePooling1D):  # type: ignore
    def __init__(
        self,
        data_format=None,
        keepdims=False,
        iq_conf: QuantizerConfig | None = None,
        oq_conf: QuantizerConfig | None = None,
        **kwargs,
    ):
        super().__init__(
            data_format=data_format,
            keepdims=keepdims,
            iq_conf=iq_conf,
            oq_conf=oq_conf,
            **kwargs,
        )


class QGlobalAveragePooling2D(QBasePooling, GlobalAveragePooling2D):
    def __init__(
        self,
        data_format=None,
        keepdims=False,
        iq_conf: QuantizerConfig | None = None,
        oq_conf: QuantizerConfig | None = None,
        **kwargs,
    ):
        super().__init__(
            data_format=data_format,
            keepdims=keepdims,
            iq_conf=iq_conf,
            oq_conf=oq_conf,
            **kwargs,
        )


class QGlobalAveragePooling3D(QBasePooling, GlobalAveragePooling3D):
    def __init__(
        self,
        data_format=None,
        keepdims=False,
        iq_conf: QuantizerConfig | None = None,
        oq_conf: QuantizerConfig | None = None,
        **kwargs,
    ):
        super().__init__(
            data_format=data_format,
            keepdims=keepdims,
            iq_conf=iq_conf,
            oq_conf=oq_conf,
            **kwargs,
        )


class QGlobalMaxPooling1D(QBasePooling, GlobalMaxPooling1D):
    def __init__(
        self,
        data_format=None,
        keepdims=False,
        iq_conf: QuantizerConfig | None = None,
        oq_conf: QuantizerConfig | None = None,
        **kwargs,
    ):
        super().__init__(
            data_format=data_format,
            keepdims=keepdims,
            iq_conf=iq_conf,
            oq_conf=oq_conf,
            **kwargs,
        )


class QGlobalMaxPooling2D(QBasePooling, GlobalMaxPooling2D):
    def __init__(
        self,
        data_format=None,
        keepdims=False,
        iq_conf: QuantizerConfig | None = None,
        oq_conf: QuantizerConfig | None = None,
        **kwargs,
    ):
        super().__init__(
            data_format=data_format,
            keepdims=keepdims,
            iq_conf=iq_conf,
            oq_conf=oq_conf,
            **kwargs,
        )


class QGlobalMaxPooling3D(QBasePooling, GlobalMaxPooling3D):
    def __init__(
        self,
        data_format=None,
        keepdims=False,
        iq_conf: QuantizerConfig | None = None,
        oq_conf: QuantizerConfig | None = None,
        **kwargs,
    ):
        super().__init__(
            data_format=data_format,
            keepdims=keepdims,
            iq_conf=iq_conf,
            oq_conf=oq_conf,
            **kwargs,
        )


__all__ = [
    'QMaxPooling1D',
    'QMaxPooling2D',
    'QMaxPooling3D',
    'QAveragePooling1D',
    'QAveragePooling2D',
    'QAveragePooling3D',
    'QGlobalMaxPooling1D',
    'QGlobalMaxPooling2D',
    'QGlobalMaxPooling3D',
    'QGlobalAveragePooling1D',
    'QGlobalAveragePooling2D',
    'QGlobalAveragePooling3D',
]
