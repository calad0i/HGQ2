from collections.abc import Sequence
from math import log2, prod

from keras import ops

from ...utils.config.quantizer import QuantizerConfig
from ..core.base import QLayerBaseSingleInput


class QSum(QLayerBaseSingleInput):
    def __init__(
            self,
            iq_confs: Sequence[QuantizerConfig] | QuantizerConfig | None = None,
            axis: int | Sequence[int] = -1,
            pow2_scale: float = 1.0,
            keepdims: bool = False,
            **kwargs
    ):
        super().__init__(iq_confs=iq_confs, **kwargs)
        self.axis = tuple(axis) if isinstance(axis, Sequence) else (axis,)
        self._scale = float(2.**log2(pow2_scale))
        self._keepdims = keepdims

    @property
    def scale(self):
        return self._scale

    @property
    def keepdims(self):
        return self._keepdims

    def _compute_ebops(self, shape):
        bits = self.iq.bits_((1,) + shape[1:])
        ebops = ops.sum(bits) - ops.sum(ops.min(bits, axis=self.axis))  # type: ignore
        ebops = ebops * 0.65  # TODO: better ebops cost model for accumulators
        self._ebops.assign(ops.cast(ebops, self._ebops.dtype))  # type: ignore
        self.add_loss(self.beta * ebops)

    def call(self, inputs, training=None):
        if self.enable_ebops and training:
            self._compute_ebops(ops.shape(inputs))
        qinput = self.iq(inputs, training=training)
        r = ops.sum(qinput, axis=self.axis, keepdims=self.keepdims) * self.scale  # type: ignore
        return r


class QMeanPow2(QSum):
    def __init__(
            self,
            iq_confs: Sequence[QuantizerConfig] | QuantizerConfig | None = None,
            axis: int | Sequence[int] = -1,
            keepdims: bool = False,
            **kwargs
    ):
        super().__init__(iq_confs=iq_confs, axis=axis, keepdims=keepdims, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        scale = 1.0 / prod([input_shape[i] for i in self.axis])
        self._scale = float(2.**log2(scale))
