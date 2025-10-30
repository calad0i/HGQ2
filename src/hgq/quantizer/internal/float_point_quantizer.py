import keras
from keras import ops
from keras.constraints import Constraint
from keras.initializers import Constant, Initializer
from keras.regularizers import Regularizer
from quantizers.fixed_point._fixed_point_ops import round_conv
from quantizers.minifloat.float_point_ops import float_quantize

from ...constraints import Min
from .base import TrainableQuantizerBase, numbers


class FloatPointQuantizer(TrainableQuantizerBase):
    """Internal class for float-point quantizer. Follows IEEE 754 standard with subnormal numbers, but without NaNs and infinities. The quantizer is defined by three parameters: mantissa, exponent, and exponent offset. Mantissa bits are excluding the sign bit. Exponent bits are including the sign bit. Exponent offset is added to the signed exponent.

    The sign bit always presents in this quantizer. However, as the number of mantissa bits reaches -1, the quantizer will always produce zero.

    Can be used as a quantizer in Keras layers, but is usually wrapped by a `Quantizer` class to provide a consistent interface.

    Parameters
    ----------
    m0 : numbers | Initializer
        Initial value of the number of mantissa bits. Trainable.
    e0 : numbers | Initializer
        Initial value of the number of exponent bits. Trainable.
    e00 : numbers | Initializer, optional
        Initial value of the exponent offset. Default is 0. Trainable.
    mc : Constraint, optional
        Constraint for the number of mantissa bits. Default is Min(-1).
    ec : Constraint, optional
        Constraint for the number of exponent bits. Default is keras.constraints.NonNeg().
    e0c : Constraint, optional
        Constraint for the exponent offset. Default is None.
    mr : Regularizer, optional
        Regularizer for the number of mantissa bits. Default is None.
    er : Regularizer, optional
        Regularizer for the number of exponent bits. Default is None.
    e0r : Regularizer, optional
        Regularizer for the exponent offset. Default is None.
    """

    def __init__(
        self,
        m0: numbers | Initializer,
        e0: numbers | Initializer,
        e00: numbers | Initializer = 0,
        mc: Constraint | None = Min(-1),
        ec: Constraint | None = keras.constraints.NonNeg(),
        e0c: Constraint | None = None,
        mr: Regularizer | None = None,
        er: Regularizer | None = None,
        e0r: Regularizer | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._m0 = m0 if isinstance(m0, Initializer) else Constant(float(m0))
        self._e0 = e0 if isinstance(e0, Initializer) else Constant(float(e0))
        self._e00 = e00 if isinstance(e00, Initializer) else Constant(float(e00))
        self.m_constraint = mc
        self.e_constraint = ec
        self.e0_constraint = e0c
        self.m_regularizer = mr
        self.e_regularizer = er
        self.e0_regularizer = e0r

    def build(self, input_shape):
        super().build(input_shape)
        bw_shape = self.bw_mapper.inference_weight_shape(input_shape)
        self._m = self.add_weight(
            name='m',
            shape=bw_shape,
            initializer=self._m0,
            trainable=True,
            constraint=self.m_constraint,
            regularizer=self.m_regularizer,
        )
        self._e = self.add_weight(
            name='e',
            shape=bw_shape,
            initializer=self._e0,
            trainable=True,
            constraint=self.e_constraint,
            regularizer=self.e_regularizer,
        )
        self._e0 = self.add_weight(
            name='e0',
            shape=bw_shape,
            initializer=self._e00,
            trainable=True,
            constraint=self.e0_constraint,
            regularizer=self.e0_regularizer,
        )
        super().build(input_shape)

    @property
    def m(self):
        return round_conv(ops.cast(self._m, ops.dtype(self._m)))  # type: ignore

    @property
    def e(self):
        return round_conv(ops.cast(self._e, ops.dtype(self._e)))  # type: ignore

    @property
    def e0(self):
        return round_conv(ops.cast(self._e0, ops.dtype(self._e0)))  # type: ignore

    @property
    def bits(self):
        return self.m + self.e + 1.0  # type: ignore

    @property
    def fbits(self):
        return self.bits

    @property
    def min(self):
        return -self.max

    @property
    def max(self):
        return 2.0 ** (2 ** (self.e - 1.0) + self.e0 - 1) * (2 - 2.0**-self.m)  # type: ignore

    @property
    def epsilon(self):
        return 2.0 ** (-(2 ** (self.e - 1)) + self.e0 + 1) * (2.0**-self.m)  # type: ignore

    def call(self, inputs, training=None):
        m = self.bw_mapper.bw_to_x(self.m, ops.shape(inputs))
        e = self.bw_mapper.bw_to_x(self.e, ops.shape(inputs))
        e0 = self.bw_mapper.bw_to_x(self.e0, ops.shape(inputs))
        return float_quantize(inputs, m, e, e0)

    def __repr__(self):
        if not self.built:
            return f'{self.__class__.__name__}(name={self.name}, built=False)'
        mstd, estd, e0std = float(ops.std(self.m)), float(ops.std(self.e)), float(ops.std(self.e0))  # type: ignore
        mmean, emean, e0mean = float(ops.mean(self.m)), float(ops.mean(self.e)), float(ops.mean(self.e0))  # type: ignore
        mstr = f'{mmean:.2f}±{mstd:.2f}'
        estr = f'{emean:.2f}±{estd:.2f}'
        e0str = f'{e0mean:.2f}±{e0std:.2f}'
        return f'{self.__class__.__name__}(m={mstr}, e={estr}, e0={e0str}, name={self.name})'
