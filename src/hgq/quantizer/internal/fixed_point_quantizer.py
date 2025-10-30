import keras
from keras import ops
from keras.constraints import Constraint
from keras.initializers import Constant, Initializer
from keras.regularizers import Regularizer
from keras.src import backend
from keras.src.backend.config import epsilon
from quantizers.fixed_point._fixed_point_ops import _get_fixed_quantizer as get_fixed_quantizer
from quantizers.fixed_point._fixed_point_ops import round_conv

from ...constraints import MinMax
from .base import TrainableQuantizerBase, numbers


def minimal_i_given_xb(x, b, symmetric=False):
    eps = epsilon()
    if symmetric:
        return ops.ceil(ops.log2((x + eps) / (1 - 2.0**-b + eps)))
    i_pos = ops.ceil(ops.log2((x + eps) / (1 - 2.0**-b + eps)))
    i_neg = ops.ceil(ops.log2(-x - eps))
    return ops.where(b > 0, ops.where(x >= 0, i_pos, i_neg), 32)


def minimal_i_given_xf(x, f, symmetric=False):
    eps = epsilon()
    if symmetric:
        return ops.ceil(ops.log2(ops.abs(x) + 2**-f))
    i_pos = ops.ceil(ops.log2(x + 2**-f))
    i_neg = ops.ceil(ops.log2(-x - eps))
    return ops.where(x >= 0, i_pos, i_neg)


class FixedPointQuantizerBase(TrainableQuantizerBase):
    """Abstract base class for all fixed-point quantizers."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._seed_gen = None
        self._is_weight = False

    @property
    def round_mode(self) -> str:
        return self._round_mode

    @property
    def overflow_mode(self) -> str:
        return self._overflow_mode

    @property
    def seed_gen(self):
        return self._seed_gen

    def build(self, input_shape):
        super().build(input_shape)
        if self.round_mode.startswith('S_'):
            self._seed_gen = keras.random.SeedGenerator(self._seed)
        self.stateless_quantizer = get_fixed_quantizer(self.round_mode, self.overflow_mode)
        if self.overflow_mode == 'WRAP':
            init = Constant(self._i_decay_speed0) if not isinstance(self._i_decay_speed0, Initializer) else self._i_decay_speed0
            self._i_decay_speed = self.add_weight(name='i_decay_speed', shape=(), initializer=init, trainable=False)
        self._symmetric = self.overflow_mode.endswith('SYM')

    @property
    def symmetric(self):
        return self._symmetric

    @property
    def i_decay_speed(self):
        return ops.cast(self._i_decay_speed, self._i_decay_speed.dtype)

    @property
    def kif(self):
        raise NotImplementedError

    @property
    def k(self):
        raise NotImplementedError

    @property
    def b(self):
        raise NotImplementedError

    @property
    def i(self):
        raise NotImplementedError

    @property
    def f(self):
        raise NotImplementedError

    @property
    def bits(self):
        if self.overflow_mode != 'SAT':
            return self.b
        else:
            return self.b + self.k

    @property
    def fbits(self):
        b = self.b
        if self.overflow_mode == 'WRAP':
            return b + self.k * ops.cast(b > 0, self.dtype)
        else:
            return b + self.k

    @property
    def min(self):
        if self.symmetric:
            return -self.k * (2.0**self.i - 2.0**-self.f)
        else:
            return -self.k * (2.0**self.i)

    @property
    def max(self):
        return 2.0**self.i - 2.0**-self.f

    @property
    def epsilon(self):
        return 2.0**-self.f

    def get_any_k(self, inputs):
        return self.bw_mapper.x_to_bw_sign(inputs)

    def __repr__(self) -> str:
        if not self.built:
            return f'{self.__class__.__name__}({self.round_mode}, {self.overflow_mode}, name={self.name}, built=False)'
        k, i, f = self.k, self.i, self.f
        kstd, istd, fstd = float(ops.std(k)), float(ops.std(i)), float(ops.std(f))  # type: ignore
        kmean, imean, fmean = float(ops.mean(k)), float(ops.mean(i)), float(ops.mean(f))  # type: ignore
        kstr = f'{kmean:.2f}±{kstd:.2f}'
        istr = f'{imean:.2f}±{istd:.2f}'
        fstr = f'{fmean:.2f}±{fstd:.2f}'
        return (
            f'{self.__class__.__name__}(k={kstr}, i={istr}, f={fstr}, {self.round_mode}, {self.overflow_mode}, name={self.name})'
        )

    def get_minimum_i(self, inputs):
        raise NotImplementedError

    def call(self, inputs, training=None):  # type: ignore
        k, i, f = self.kif
        k = self.bw_mapper.bw_to_x(k, ops.shape(inputs))
        i = self.bw_mapper.bw_to_x(i, ops.shape(inputs))
        f = self.bw_mapper.bw_to_x(f, ops.shape(inputs))
        ret = self.stateless_quantizer(inputs, k, i, f, training is True, self.seed_gen)
        if not training:
            ret = ops.where(k + i + f > 0, ret, ops.zeros_like(ret))  # type: ignore
        return ret


class FixedPointQuantizerKBI(FixedPointQuantizerBase):
    """Internal quantizer for fixed-point quantization parameterized by keep_negative, bits, and integer bits.
    Can be used as a quantizer in Keras layers, but is usually wrapped by a `Quantizer` class to provide a consistent interface.

    Parameters
    ----------
    k0 : numbers | bool | Initializer
        Initial value for the keep_negative parameter. Not trained, but can be manually updated.
    b0 : numbers | Initializer
        Initial value for the number of bits. Trainable.
    i0 : numbers | Initializer
        Initial value for the number of integer bits. Trainable.
    round_mode : str
        Rounding mode, one of 'RND', 'TRN', 'RND_CONV', 'S_RND', 'S_RND_CONV'.
    overflow_mode : str
        Overflow mode, one of 'WRAP', 'SAT', 'SYM', 'SAT_SYM'.
    bc : Constraint | None
        Constraint for the number of bits.
    ic : Constraint | None
        Constraint for the number of integer bits.
    br : Regularizer | None
        Regularizer for the number of bits.
    ir : Regularizer | None
        Regularizer for the number of integer bits.
    i_decay_speed : numbers
        Speed of decay for the integer bits in WRAP mode, per step. If set to negative, enable tracing of maximum number even not in training mode.
    """

    def __init__(
        self,
        k0: numbers | bool | Initializer,
        b0: numbers | Initializer,
        i0: numbers | Initializer,
        round_mode: str,
        overflow_mode: str,
        bc: Constraint | None = MinMax(0, 12),
        ic: Constraint | None = None,
        br: Regularizer | None = None,
        ir: Regularizer | None = None,
        i_decay_speed: numbers = float('inf'),
        is_weight: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not isinstance(k0, Initializer) and not isinstance(b0, Initializer) and not isinstance(i0, Initializer):
            k0 = int(k0)
            assert k0 == 0 or k0 == 1, f'Invalid k0 value {k0}: must be 0 or 1.'
            assert b0 >= 0, f'Invalid b0 value {b0}: must be non-negative.'
        self._k0 = k0 if isinstance(k0, Initializer) else Constant(float(k0))
        self._b0 = b0 if isinstance(b0, Initializer) else Constant(float(b0))
        self._i0 = i0 if isinstance(i0, Initializer) else Constant(float(i0))
        self._i_decay_speed0 = i_decay_speed if isinstance(i_decay_speed, Initializer) else Constant(float(i_decay_speed))
        self._round_mode = round_mode.upper()
        self._overflow_mode = overflow_mode.upper()
        self.b_constraint = bc
        self.i_constraint = ic
        self.b_regularizer = br
        self.i_regularizer = ir
        self._is_weight = is_weight

        self.validate_config()

    def build(self, input_shape):
        bw_shape = self.bw_mapper.inference_weight_shape(input_shape)

        self._k = self.add_weight(
            name='k',
            shape=bw_shape,
            initializer=self._k0,
            trainable=False,
            dtype='uint8',
        )
        self._b = self.add_weight(
            name='b',
            shape=bw_shape,
            initializer=self._b0,
            trainable=True,
            constraint=self.b_constraint,
            regularizer=self.b_regularizer,
        )
        i_trainable = self.overflow_mode != 'WRAP'
        self._i = self.add_weight(
            name='i',
            shape=bw_shape,
            initializer=self._i0,
            trainable=i_trainable,
            constraint=self.i_constraint,
            regularizer=self.i_regularizer,
        )
        super().build(input_shape)

    @property
    def k(self):
        return backend.convert_to_tensor(self._k, self.dtype)

    @property
    def b(self):
        return round_conv(ops.cast(self._b, ops.dtype(self._b)))

    @property
    def i(self):
        return round_conv(ops.cast(self._i, ops.dtype(self._i)))

    @property
    def f(self):
        return self.b - self.i  # type: ignore

    @property
    def kif(self):
        k = self.k
        b = self.b
        i = self.i
        return k, i, b - i  # type: ignore

    def get_minimal_i(self, inputs):
        xr = self.bw_mapper.x_to_bw_absmax(inputs)
        return minimal_i_given_xb(xr, self.b, self.symmetric)

    def validate_config(self):
        assert self.b_constraint is None or isinstance(self.b_constraint, Constraint)
        assert self.i_constraint is None or isinstance(self.i_constraint, Constraint)
        assert self.b_regularizer is None or isinstance(self.b_regularizer, Regularizer)
        assert self.i_regularizer is None or isinstance(self.i_regularizer, Regularizer)

    def call(self, inputs, training=None):
        if self.overflow_mode == 'WRAP' and self.trainable:
            stochastic = self.stateless_quantizer.stochastic and training is True
            if training or training == 'tracing':  # noqa: E712, training maybe a special wrapper object
                new_i = self.get_minimal_i(inputs)
                if training:
                    new_i = ops.stop_gradient(ops.maximum((self.i - self.i_decay_speed), new_i))  # type: ignore
                    if self._i.constraint is not None:
                        new_i = self._i.constraint(new_i)
                    self._i.assign(new_i)
                    f = self.bw_mapper.bw_to_x(self.f, ops.shape(inputs))
                    return self.stateless_quantizer.round(inputs, f, stochastic, self.seed_gen)
                else:
                    new_i = ops.stop_gradient(ops.maximum(self.i, new_i))  # type: ignore
                    if self._i.constraint is not None:
                        new_i = self._i.constraint(new_i)

                    f = self.bw_mapper.bw_to_x(self.b - new_i, ops.shape(inputs))  # type: ignore
                    rinputs = self.stateless_quantizer.round(inputs, f, stochastic, self.seed_gen)
                    new_k = self.get_any_k(rinputs) & (self.b > 0)  # type: ignore
                    new_k = ops.cast(ops.cast(self.k, 'bool') | new_k, self._k.dtype)  # type: ignore
                    self._k.assign(new_k)
                    self._i.assign(new_i)
                    return rinputs

            if self._is_weight:
                f = self.bw_mapper.bw_to_x(self.f, ops.shape(inputs))
                if self.stateless_quantizer.round_mode == 'RND':
                    return self.stateless_quantizer.round(inputs, f, False, self.seed_gen)

        return super().call(inputs, training is True)


class FixedPointQuantizerKIF(FixedPointQuantizerBase):
    """Internal quantizer for fixed-point quantization parameterized by keep_negative, integer bits, and fractional bits.
    Can be used as a quantizer in Keras layers, but is usually wrapped by a `Quantizer` class to provide a consistent interface.

    Parameters
    ----------
    k0 : numbers | bool | Initializer
        Initial value for the keep_negative parameter. Not trained, but can be manually updated.
    i0 : numbers | Initializer
        Initial value for the number of integer bits. Trainable.
    f0 : numbers | Initializer
        Initial value for the number of fractional bits. Trainable.
    round_mode : str
        Rounding mode, one of 'RND', 'TRN', 'RND_CONV', 'S_RND', 'S_RND_CONV'.
    overflow_mode : str
        Overflow mode, one of 'WRAP', 'SAT', 'SYM', 'SAT_SYM'.
    ic : Constraint | None
        Constraint for the number of integer bits.
    fc : Constraint | None
        Constraint for the number of fractional bits.
    ir : Regularizer | None
        Regularizer for the number of integer bits.
    fr : Regularizer | None
        Regularizer for the number of fractional bits.
    i_decay_speed : numbers
        Speed of decay for the integer bits in WRAP mode, per step. If set to negative, enable tracing of maximum number even not in training mode.
    """

    def __init__(
        self,
        k0: numbers | bool,
        i0: numbers,
        f0: numbers,
        round_mode: str,
        overflow_mode: str,
        ic: Constraint | None = None,
        fc: Constraint | None = None,
        ir: Regularizer | None = None,
        fr: Regularizer | None = None,
        i_decay_speed: numbers = float('inf'),
        is_weight: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not isinstance(k0, Initializer) and not isinstance(i0, Initializer) and not isinstance(f0, Initializer):
            k0 = int(k0)
            assert k0 == 0 or k0 == 1, f'Invalid k0 value {k0}: must be 0 or 1.'
            assert i0 + f0 >= 0, f'Invalid i0+f0 value {i0 + f0}: must be non-negative.'
        self._k0 = k0 if isinstance(k0, Initializer) else Constant(float(k0))
        self._i0 = i0 if isinstance(i0, Initializer) else Constant(float(i0))
        self._f0 = f0 if isinstance(f0, Initializer) else Constant(float(f0))
        self._i_decay_speed0 = i_decay_speed if isinstance(i_decay_speed, Initializer) else Constant(float(i_decay_speed))
        self._round_mode = round_mode.upper()
        self._overflow_mode = overflow_mode.upper()
        self.i_constraint = ic
        self.f_constraint = fc
        self.i_regularizer = ir
        self.f_regularizer = fr
        self._is_weight = is_weight
        self.validate_config()

    def build(self, input_shape):
        bw_shape = self.bw_mapper.inference_weight_shape(input_shape)

        self._k = self.add_weight(
            name='k',
            shape=bw_shape,
            initializer=self._k0,
            trainable=False,
            dtype='uint8',
        )
        i_trainable = self.overflow_mode != 'WRAP'
        self._i = self.add_weight(
            name='i',
            shape=bw_shape,
            initializer=self._i0,
            trainable=i_trainable,
            constraint=self.i_constraint,
            regularizer=self.i_regularizer,
        )
        self._f = self.add_weight(
            name='f',
            shape=bw_shape,
            initializer=self._f0,
            trainable=True,
            constraint=self.f_constraint,
            regularizer=self.f_regularizer,
        )

        super().build(input_shape)

    def call(self, inputs, training=None):  # type: ignore
        if self.overflow_mode == 'WRAP' and self.trainable:
            f = self.bw_mapper.bw_to_x(self.f, ops.shape(inputs))
            stochastic = self.stateless_quantizer.stochastic and training is True
            rinputs = self.stateless_quantizer.round(inputs, f, stochastic, self.seed_gen)

            if training or training == 'tracing':  # noqa: E712, training maybe a special wrapper object
                _new_i = self.get_minimal_i(rinputs)
                if training:
                    new_i = ops.stop_gradient(ops.maximum((self.i - self.i_decay_speed), _new_i))  # type: ignore
                    if self._i.constraint is not None:
                        _new_i = self._i.constraint(_new_i)
                    self._i.assign(new_i)
                else:
                    # tracing
                    if self._i.constraint is not None:
                        _new_i = self._i.constraint(_new_i)
                    new_i = ops.stop_gradient(ops.maximum(self.i, _new_i))  # type: ignore
                    new_k = self.get_any_k(rinputs)
                    new_k = ops.cast(ops.cast(self.k, 'bool') | new_k, self._k.dtype)  # type: ignore
                    self._k.assign(new_k)
                    self._i.assign(new_i)

                return rinputs

            if self._is_weight:
                return rinputs

        return super().call(inputs, training is True)

    @property
    def k(self):
        return ops.cast(self._k, self.dtype)

    @property
    def b(self):
        return ops.relu(self.i + self.f)  # type: ignore

    @property
    def i(self):
        return round_conv(ops.cast(self._i, self._i.dtype))

    @property
    def f(self):
        return round_conv(ops.cast(self._f, self._f.dtype))

    @property
    def kif(self):
        return self.k, self.i, self.f

    def get_minimal_i(self, inputs):
        xr = self.bw_mapper.x_to_bw_absmax(inputs)
        return minimal_i_given_xf(xr, self.f, self.symmetric)

    def validate_config(self):
        assert self.f_constraint is None or isinstance(self.f_constraint, Constraint)
        assert self.i_constraint is None or isinstance(self.i_constraint, Constraint)
        assert self.f_regularizer is None or isinstance(self.f_regularizer, Regularizer)
        assert self.i_regularizer is None or isinstance(self.i_regularizer, Regularizer)
