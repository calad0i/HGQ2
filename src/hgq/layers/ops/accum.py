from collections.abc import Sequence
from math import log2, prod

from keras import ops

from ...quantizer.config import QuantizerConfig
from ..core.base import QLayerBaseSingleInput


class QSum(QLayerBaseSingleInput):
    def __init__(
        self,
        iq_conf: QuantizerConfig | None = None,
        axes: int | Sequence[int] = -1,
        scale: float = 1.0,
        keepdims: bool = False,
        **kwargs,
    ):
        super().__init__(iq_conf=iq_conf, **kwargs)
        assert kwargs.get('axis', None) is None, f'Use "axes" instead of "axis" in {self.__class__.__name__} ({self.name}).'
        self.axes = tuple(axes) if isinstance(axes, Sequence) else (axes,)
        assert log2(scale).is_integer(), 'Scale must be a power of 2.'
        self._scale = scale
        self._keepdims = keepdims

    def build(self, input_shape):
        super().build(input_shape)
        axes = sorted(i if i >= 0 else i + len(input_shape) for i in self.axes)
        self.axes = tuple(axes)

    @property
    def scale(self):
        return self._scale

    @property
    def keepdims(self):
        return self._keepdims

    def _compute_ebops(self, shape):
        bits = self.iq.bits_(shape)
        ebops = ops.sum(bits) - ops.sum(ops.min(bits, axis=self.axes))  # type: ignore
        ebops = ebops * 0.65  # TODO: better ebops cost model for accumulators
        return ebops

    def call(self, inputs, training=None):
        if self.enable_iq:
            inputs = self.iq(inputs, training=training)
        r = ops.sum(inputs, axis=self.axes, keepdims=self.keepdims) * self.scale  # type: ignore
        return r

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'axes': self.axes,
                'scale': self.scale,
                'keepdims': self.keepdims,
            }
        )
        return config


class QMeanPow2(QSum):
    def __init__(
        self,
        iq_conf: QuantizerConfig | None = None,
        axes: int | Sequence[int] = -1,
        keepdims: bool = False,
        **kwargs,
    ):
        super().__init__(iq_conf=iq_conf, axes=axes, keepdims=keepdims, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        scale = 1.0 / prod([input_shape[i] for i in self.axes])
        self._scale = 2.0 ** round(log2(scale))

    def get_config(self):
        config = super().get_config()
        config.pop('scale')
        return config
