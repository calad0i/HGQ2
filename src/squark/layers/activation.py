from collections.abc import Callable

from keras import ops
from keras.api.layers import Activation
from keras.src import backend

from ..quantizer import QuantizerConfig
from .core.base import QLayerBaseSingleInput


def _large_negative_number(dtype):
    """Return a Large negative number based on dtype."""
    if backend.standardize_dtype(dtype) == 'float16':
        return -3e4
    return -1e9


class QUnaryFunctionLUT(Activation, QLayerBaseSingleInput):
    def __init__(
        self,
        activation: Callable | str,
        iq_conf: QuantizerConfig | None = None,
        oq_conf: QuantizerConfig | None = None,
        enable_oq=True,
        allow_heterogeneous_table: bool = False,
        override_oq_k0_to_0: bool = False,
        **kwargs,
    ):
        act_name = activation.__name__ if isinstance(activation, Callable) else activation
        assert act_name not in ('softmax', 'log_softmax'), f'activation {act_name} is not unary'

        if enable_oq:
            oq_conf = oq_conf or QuantizerConfig('default', 'table')
            if override_oq_k0_to_0:
                if 'k0' in oq_conf.config:
                    oq_conf.config['k0'] = False
            if not allow_heterogeneous_table:
                oq_conf.config['homogeneous_axis'] = None
                oq_conf.config['heterogeneous_axis'] = ()
        super().__init__(activation=activation, iq_conf=iq_conf, oq_conf=oq_conf, enable_oq=enable_oq, **kwargs)

    def call(self, inputs, training=None):
        if self.enable_iq:
            inputs = self.iq(inputs, training=training)
        x = super().call(inputs)
        if self.enable_oq:
            x = self.oq(x, training=training)
        return x

    def _compute_ebops(self, shape):
        bw_inp = self.iq.bits_(shape)
        bw_out = self.oq.bits_(shape)
        # TODO: more realistic cost for lookup tables
        return ops.sum((2.0**bw_inp) * bw_out) * 0.01  # type: ignore


class QPositiveUnaryFunctionLUT(QUnaryFunctionLUT):
    def __init__(
        self,
        activation: Callable | str,
        iq_conf: QuantizerConfig | None = None,
        oq_conf: QuantizerConfig | None = None,
        allow_heterogeneous_table: bool = False,
        **kwargs,
    ):
        assert kwargs.pop(
            'enable_out_quantizer', True
        ), 'enable_out_quantizer must be True for QPositiveUnaryFunctionLUT, if set.'
        super().__init__(
            activation=activation,
            iq_conf=iq_conf,
            oq_conf=oq_conf,
            enable_oq=True,
            allow_heterogeneous_table=allow_heterogeneous_table,
            override_oq_k0_to_0=True,
            **kwargs,
        )

    def call(self, inputs, training=None):
        x = super().call(inputs, training=training)
        eps = self.oq.epsilon_(ops.shape(inputs))
        return ops.maximum(x, eps)
