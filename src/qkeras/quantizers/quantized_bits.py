import inspect
import types
from collections.abc import Sequence
from functools import wraps

import keras
import numpy as np
from keras import ops
from keras.saving import register_keras_serializable
from quantizers.fixed_point.fixed_point_ops import FixedPointQuantizer

from hgq.config import QuantizerConfig
from hgq.constraints import Min, MinMax
from hgq.quantizer.config import KBIConfig
from hgq.quantizer.internal import FixedPointQuantizerKBI
from hgq.utils.misc import numbers

from .frozen_quantizer import FrozenFixedPointQuantizer
from .utils import parse_string_to_args


@register_keras_serializable(package='hgq/qkeras')
class quantized_bits(QuantizerConfig):
    def __init__(
        self,
        bits: int = 8,
        integer: int = 0,
        symmetric: int = 0,
        keep_negative: bool = True,
        alpha: numbers | str | None = None,
        use_stochastic_rounding: bool = False,
        scale_axis: Sequence[int] | None = None,
        qnoise_factor: float | None = None,
        var_name: None = None,
        use_ste: bool = True,
        use_variables: bool | None = None,
        bc=Min(0),
        br=None,
        ic=MinMax(-12, 12),
        ir=None,
    ):
        """**Configuration** for the creating a qkeras-like quantized_bits quantizer. Can be called to quantize arrays, but does not support training. Pass to the constructor of Quantizer to create a quantizer, or call get_quantizer() to get the internal quantizer (missing certain features realized in the wrapper).

        Parameters
        ----------
        bits : int, optional
            The bit width of the quantized values excluding the sign bit, by default 8
        integer : int, optional
            The number of integer bits excluding the sign bit, by default 0
        symmetric : int, optional
            Whether to use symmetric quantization, by default False
        keep_negative : bool, optional
            Whether to keep negative values, by default True
        alpha : numbers | str | None, optional
            The scaling factor for the quantization. If a number, use the number as the fixed scaling factor. If 'trainable_po2', make number of integer bits trainable. If None, skip scaling, by default 'trainable_po2'
        use_stochastic_rounding : bool, optional
            Whether to use stochastic rounding, by default False
        scale_axis : Sequence[int] | None, optional
            The axes to apply heterogenous quantization, only used if alpha is 'trainable_po2', by default None
        qnoise_factor : float, optional
            The fraction of quantization strength. If None, treat as full quantization noise, by default None
        var_name : None, optional
            Not supported. Must be None, by default None
        use_ste : bool, optional
            Not supported. Must be True, by default True
        use_variables : bool | None, optional
            Whether to use variables for the quantization parameters. Must be True or None if alpha is 'trainable_po2', by default None
        bc : Constraint, optional
           The constraint for the bitwidth. Only used if bitwidth is trainable, by default Min(0)
        br : Regularizer, optional
            The regularizer for the bitwidth. Only used if bitwidth is trainable, by default None
        ic : Constraint, optional
            The constraint for the integer width. Only used if integer width is trainable, by default MinMax(-12, 12)
        ir : Regularizer, optional
            The regularizer for the integer width. Only used if integer width is trainable, by default None
        """

        assert bits > 0, 'bits must be greater than 0'
        if isinstance(alpha, numbers):
            assert alpha > 0, 'alpha must be greater than 0 if it is a number'
        if isinstance(alpha, str):
            assert alpha == 'trainable_po2', "alpha must be 'trainable_po2' if it is a string"
        self.trainable = alpha == 'trainable_po2'

        if use_variables is None:
            use_variables = self.trainable
        if self.trainable:
            assert use_variables, "use_variables must be True if alpha is set to 'trainable_po2'"
        self.use_variables = use_variables

        assert var_name is None, 'var_name is not supported'
        assert use_ste, 'only use_ste=True is supported'

        scaler = alpha if isinstance(alpha, numbers) else None
        heterogeneous_axis = ()
        if scale_axis is not None and self.trainable:
            heterogeneous_axis = tuple(scale_axis)
        overflow_mode = 'SAT_SYM' if symmetric else 'SAT'
        round_mode = 'S_RND_CONV' if use_stochastic_rounding else 'RND_CONV'

        super().__init__(
            'kbi',
            'datalane',
            k0=keep_negative,
            b0=bits,
            i0=integer,
            round_mode=round_mode,
            overflow_mode=overflow_mode,
            bc=bc,
            ic=ic,
            br=br,
            ir=ir,
            i_decay_speed=-1,
            heterogeneous_axis=heterogeneous_axis,
            scaler=scaler,
            qnoise_factor=qnoise_factor,
            trainable=self.trainable,
        )
        self.config: KBIConfig
        self.stateless_quantizer = FixedPointQuantizer(round_mode.removeprefix('S_'), overflow_mode)

    @property
    def bits(self):
        b = self.config['b0']
        assert isinstance(b, int)
        return b

    @property
    def integers(self):
        i = self.config['i0']
        assert isinstance(i, int)
        return i

    @property
    def symmetric(self):
        return 'SYM' in self.config['overflow_mode'].upper()

    @property
    def keep_negative(self):
        return bool(self.config['k0'])

    def get_quantizer(self):
        if not self.use_variables:
            return FrozenFixedPointQuantizer(**self.config)  # type: ignore

        quantizer = FixedPointQuantizerKBI(**self.config)
        if not self.trainable:
            return quantizer

        # Make the bits non-trainable by default to match qkeras behavior
        original_setattr_hook = quantizer._setattr_hook

        @wraps(original_setattr_hook)
        def _setattr_hook(self, name, value):
            if name == '_b':
                value.trainable = False
            return original_setattr_hook(name, value)

        quantizer._setattr_hook = types.MethodType(_setattr_hook, quantizer)

        return quantizer

    def __str__(self):
        k, b, i = self.config['k0'], self.config['b0'], self.config['i0']
        symmetric = 'SYM' in self.config['overflow_mode'].upper()
        alpha = self.scaler
        if alpha is None:
            alpha = 'trainable_po2' if self.trainable else 1
        stochastic = 'S_RND' in self.config['round_mode']
        cls_str = self.__class__.__name__
        return (
            f'{cls_str}({b}, {i}, symmetric={symmetric}, keep_negative={k}, alpha={alpha}, use_stochastic_rounding={stochastic})'
        )

    def __repr__(self):
        return str(self)

    def __call__(self, inputs, training=None):
        assert training is None, (
            'quantized_bits in HGQ2 is merely a configuration. It does not support training. Call get_quantizer() to get the actual quantizer.'
        )
        return_np = False
        if not ops.is_tensor(inputs):
            return_np = True
            inputs = np.asarray(inputs)
            if inputs.dtype not in (np.float16, np.float32):
                inputs = inputs.astype(np.float32)
        k, b, i = self.keep_negative, self.bits, self.integers

        with keras.device('cpu:0'):
            if self.scaler is not None:
                inputs /= self.scaler
            outputs = self.stateless_quantizer(inputs, k, b, i)
            if self.scaler is not None:
                outputs *= self.scaler
            if self.qnoise_factor != 1.0:
                outputs = inputs + self.qnoise_factor * (outputs - inputs)  # type: ignore
        if return_np:
            outputs = np.asarray(outputs)
        return outputs

    @property
    def max(self):
        b, i = self.bits, self.integers
        f = b - i
        return 2.0**i - 2.0**-f

    @property
    def min(self):
        if not self.keep_negative:
            return 0.0
        if self.symmetric:
            return -self.max
        else:
            return -(2.0**self.integers)

    def get_config(self):  # type: ignore
        config = dict(
            bits=self.bits,
            integer=self.integers,
            symmetric=self.symmetric,
            keep_negative=self.keep_negative,
            alpha=self.scaler if not self.trainable else 'trainable_po2',
            use_stochastic_rounding='S_RND' in self.config['round_mode'],
            scale_axis=self.config['heterogeneous_axis'],
            qnoise_factor=self.qnoise_factor,
            var_name=None,
            use_ste=True,
            use_variables=self.trainable,
            bc=self.config['bc'],
            br=self.config['br'],
            ic=self.config['ic'],
            ir=self.config['ir'],
        )
        return config

    @classmethod
    def from_string(cls, string: str):
        """Create a quantized_bits configuration from a string representation."""
        if not string.startswith('quantized_bits(') or not string.endswith(')'):
            raise ValueError(f'Invalid string representation for quantized_bits: {string}')
        args, kwargs = parse_string_to_args(string.split('(', 1)[1][:-1])
        signature = inspect.signature(cls.__init__)
        # remove self
        signature = signature.replace(parameters=list(signature.parameters.values())[1:])
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return cls(**bound.arguments)


class quantized_relu(quantized_bits):
    def __init__(
        self,
        bits: int = 8,
        integer: int = 0,
        alpha: numbers | str | None = 1.0,
        use_stochastic_rounding: bool = False,
        scale_axis: Sequence[int] | None = None,
        qnoise_factor: float | None = None,
        var_name: None = None,
        use_ste: bool = True,
        use_variables: bool | None = None,
        bc=Min(0),
        br=None,
        ic=MinMax(-12, 12),
        ir=None,
    ):
        """Grammatical sugar for quantized_bits with keep_negative=False"""
        ...

    def __new__(cls, *args, **kwargs):
        return quantized_bits(*args, **kwargs, keep_negative=False)
