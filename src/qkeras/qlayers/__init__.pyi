from typing import Sequence

from hgq import layers
from hgq.config import QuantizerConfig

class QDense(layers.QDense):
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
        kernel_quantizer: None | QuantizerConfig | str = None,
        input_quantizer: None | QuantizerConfig | str = None,
        bias_quantizer: None | QuantizerConfig | str = None,
        output_quantizer: None | QuantizerConfig | str = None,
        **kwargs,
    ): ...

class QEinsumDense(layers.QEinsumDense):
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
        kernel_quantizer: None | QuantizerConfig | str = None,
        input_quantizer: None | QuantizerConfig | str = None,
        bias_quantizer: None | QuantizerConfig | str = None,
        output_quantizer: None | QuantizerConfig | str = None,
        **kwargs,
    ): ...

class QConv1D(layers.QConv1D):
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
        kernel_quantizer: None | QuantizerConfig | str = None,
        input_quantizer: None | QuantizerConfig | str = None,
        bias_quantizer: None | QuantizerConfig | str = None,
        output_quantizer: None | QuantizerConfig | str = None,
        **kwargs,
    ): ...

class QConv2D(layers.QConv2D):
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
        kernel_quantizer: None | QuantizerConfig | str = None,
        input_quantizer: None | QuantizerConfig | str = None,
        bias_quantizer: None | QuantizerConfig | str = None,
        output_quantizer: None | QuantizerConfig | str = None,
        **kwargs,
    ): ...

class QConv3D(layers.QConv3D):
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
        kernel_quantizer: None | QuantizerConfig | str = None,
        input_quantizer: None | QuantizerConfig | str = None,
        bias_quantizer: None | QuantizerConfig | str = None,
        output_quantizer: None | QuantizerConfig | str = None,
        **kwargs,
    ): ...

class QSoftmax(layers.QSoftmax):
    def __init__(
        self,
        axis: int | Sequence[int] = -1,
        iq_conf: None | QuantizerConfig = None,
        stable=False,
        exp_input_quantizer: None | QuantizerConfig | str = None,
        exp_output_quantizer: None | QuantizerConfig | str = None,
        inv_input_quantizer: None | QuantizerConfig | str = None,
        inv_output_quantizer: None | QuantizerConfig | str = None,
        allow_heterogeneous_table: bool = False,
        input_scaler: float = 1.0,
        **kwargs,
    ): ...

class QActivation(layers.QUnaryFunctionLUT):
    def __init__(
        self,
        activation,
        input_quantizer: None | QuantizerConfig | str = None,
        output_quantizer: None | QuantizerConfig | str = None,
        allow_heterogeneous_table: bool = False,
        **kwargs,
    ): ...

class QMeanPow2(layers.QMeanPow2):
    pass

class QSum(layers.QSum):
    pass

class QAdd(layers.QAdd):
    pass

class QAveragePow2(layers.QAveragePow2):
    pass

class QDot(layers.QDot):
    pass

class QEinsum(layers.QEinsum):
    pass

class QMaximum(layers.QMaximum):
    pass

class QMinimum(layers.QMinimum):
    pass

class QMultiply(layers.QMultiply):
    pass

class QSubtract(layers.QSubtract):
    pass
