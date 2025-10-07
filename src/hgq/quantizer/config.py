from collections.abc import Mapping, Sequence
from typing import TypedDict, overload

from keras.constraints import Constraint
from keras.initializers import Initializer
from keras.regularizers import Regularizer
from keras.saving import deserialize_keras_object, register_keras_serializable

from ..constraints import Constant, Min, MinMax
from ..regularizers import MonoL1
from ..utils.misc import numbers
from .internal import (
    BitwidthMapperBase,
    DummyQuantizer,
    FixedPointQuantizerKBI,
    FixedPointQuantizerKIF,
    FloatPointQuantizer,
    TrainableQuantizerBase,
)

default_q_type = {
    'weight': 'kbi',
    'datalane': 'kif',
    'bias': 'kbi',
    'table': 'kbi',
}


class QuantizerConfigBase(TypedDict):
    homogeneous_axis: Sequence[int] | None
    heterogeneous_axis: Sequence[int] | None
    bw_mapper: BitwidthMapperBase | None
    trainable: bool
    is_weight: bool


class KBIConfig(QuantizerConfigBase):
    k0: numbers | bool | Initializer
    b0: numbers | Initializer
    i0: numbers | Initializer
    round_mode: str
    overflow_mode: str
    bc: Constraint | None
    ic: Constraint | None
    br: Regularizer | None
    ir: Regularizer | None
    i_decay_speed: numbers


class KIFConfig(QuantizerConfigBase):
    k0: numbers | bool | Initializer
    i0: numbers | Initializer
    f0: numbers | Initializer
    round_mode: str
    overflow_mode: str
    ic: Constraint | None
    ir: Regularizer | None
    fc: Constraint | None
    fr: Regularizer | None
    i_decay_speed: numbers


class FloatConfig(QuantizerConfigBase):
    m0: numbers | Initializer
    e0: numbers | Initializer
    e00: numbers | Initializer
    mc: Constraint | None
    ec: Constraint | None
    e0c: Constraint | None
    mr: Regularizer | None
    er: Regularizer | None
    e0r: Regularizer | None


kbi_weight_default = KBIConfig(
    k0=True,
    b0=4,
    i0=2,
    round_mode='RND',
    overflow_mode='SAT_SYM',
    bc=MinMax(0, 23),
    ic=None,
    br=MonoL1(1e-8),
    ir=None,
    i_decay_speed=float('inf'),
    homogeneous_axis=(),
    heterogeneous_axis=None,
    bw_mapper=None,
    trainable=True,
    is_weight=True,
)


kbi_datalane_default = KBIConfig(
    k0=True,
    b0=4,
    i0=2,
    round_mode='RND',
    overflow_mode='WRAP',
    bc=MinMax(0, 24),
    ic=None,
    br=MonoL1(1e-8),
    ir=None,
    i_decay_speed=0.01,
    homogeneous_axis=(0,),
    heterogeneous_axis=None,
    bw_mapper=None,
    trainable=True,
    is_weight=False,
)

kif_weight_default = KIFConfig(
    k0=True,
    i0=4,
    f0=2,
    round_mode='RND',
    overflow_mode='SAT_SYM',
    ic=MinMax(-23, 23),
    ir=MonoL1(1e-8),
    fc=MinMax(-24, 24),
    fr=MonoL1(1e-8),
    i_decay_speed=float('inf'),
    homogeneous_axis=(),
    heterogeneous_axis=None,
    bw_mapper=None,
    trainable=True,
    is_weight=True,
)


kif_datalane_default = KIFConfig(
    k0=True,
    i0=4,
    f0=2,
    round_mode='RND',
    overflow_mode='WRAP',
    ic=MinMax(-23, 23),
    ir=MonoL1(1e-8),
    fc=MinMax(-24, 24),
    fr=MonoL1(1e-8),
    i_decay_speed=0.01,
    homogeneous_axis=(0,),
    heterogeneous_axis=None,
    bw_mapper=None,
    trainable=True,
    is_weight=False,
)

float_weight_default = FloatConfig(
    m0=2,
    e0=4,
    e00=0,
    mc=MinMax(-1, 8),
    ec=MinMax(0, 4),
    e0c=MinMax(-16, 16),
    mr=MonoL1(1e-8),
    er=MonoL1(1e-8),
    e0r=None,
    homogeneous_axis=(),
    heterogeneous_axis=None,
    bw_mapper=None,
    trainable=True,
    is_weight=True,
)


float_datalane_default = FloatConfig(
    m0=2,
    e0=4,
    e00=0,
    mc=MinMax(-1, 8),
    ec=MinMax(0, 4),
    e0c=MinMax(-16, 16),
    mr=MonoL1(1e-8),
    er=MonoL1(1e-8),
    e0r=None,
    homogeneous_axis=None,
    heterogeneous_axis=(),
    bw_mapper=None,
    trainable=True,
    is_weight=False,
)

kbi_table_default = kbi_weight_default.copy()
kif_table_default = kif_weight_default.copy()
float_table_default = float_weight_default.copy()

kbi_table_default['homogeneous_axis'] = None
kbi_table_default['heterogeneous_axis'] = ()
kbi_table_default['is_weight'] = False
kif_table_default['homogeneous_axis'] = None
kif_table_default['heterogeneous_axis'] = ()
kif_table_default['is_weight'] = False
float_table_default['homogeneous_axis'] = None
float_table_default['heterogeneous_axis'] = ()
float_table_default['is_weight'] = False


kbi_bias_default = kbi_datalane_default.copy()
kif_bias_default = kif_datalane_default.copy()
float_bias_default = float_datalane_default.copy()

kbi_bias_default['homogeneous_axis'] = ()
kif_bias_default['homogeneous_axis'] = ()
float_bias_default['homogeneous_axis'] = ()

default_configs: dict[tuple[str, str], KIFConfig | KBIConfig | FloatConfig] = {
    ('kbi', 'weight'): kbi_weight_default,
    ('kbi', 'bias'): kbi_bias_default,
    ('kbi', 'table'): kbi_table_default,
    ('kbi', 'datalane'): kbi_datalane_default,
    ('kif', 'weight'): kif_weight_default,
    ('kif', 'bias'): kif_bias_default,
    ('kif', 'table'): kif_table_default,
    ('kif', 'datalane'): kif_datalane_default,
    ('float', 'weight'): float_weight_default,
    ('float', 'bias'): float_bias_default,
    ('float', 'table'): float_table_default,
    ('float', 'datalane'): float_datalane_default,
}

all_quantizer_keys = {k for v in default_configs.values() for k in v.keys()} | {
    'q_type',
    'place',
    'scaler',
    'affine',
    'qnoise_factor',
}


def all_quantizer_types():
    return {k[0] for k in default_configs.keys()}


def all_places():
    return {k[1] for k in default_configs.keys()}


@register_keras_serializable(package='hgq')
class QuantizerConfig(Mapping):
    @overload
    def __init__(
        self,
        q_type: str = 'kbi',
        place: str = 'datalane',
        *,
        k0: numbers | bool | Initializer | None = True,
        b0: numbers | Initializer | None = 4,
        i0: numbers | Initializer | None = 2,
        round_mode: str | None = 'RND',
        overflow_mode: str | None = 'WRAP',
        bc: Constraint | None = MinMax(0, 12),
        ic: Constraint | None = None,
        br: Regularizer | None = None,
        ir: Regularizer | None = None,
        i_decay_speed: numbers = -1,
        homogeneous_axis: Sequence[int] | None = None,
        heterogeneous_axis: Sequence[int] | None = None,
        bw_mapper: BitwidthMapperBase | None = None,
        scaler: numbers | None = None,
        affine: tuple[numbers, numbers] | None = None,
        qnoise_factor: float | None = None,
        **kwargs,
    ) -> None:
        """Fixed point quantizer config with KBI parametrization.

        Parameters
        ----------
        q_type : str
            The type of the quantizer. 'kbi' for this implementation.
        place : str
            Where the quantizer is expected to be place. Only affects default config. One of 'weight', 'datalane', 'bias', and 'table'.
        k0 : numbers | bool | Initializer, optional
            If the quantizer allows negative values, by default True
        b0 : numbers | Initializer, optional
            The initial value of the number of bits (excl. sign), by default 4
        i0 : numbers | Initializer, optional
            The initial value of the number of integer bits (excl. sign), by default 2
        round_mode : str, optional
            Rounding mode. One of 'RND', 'RND_CONV', 'TRN', 'S_RND', 'S_RND_CONV', by default 'RND'
        overflow_mode : str, optional
            Overflow mode. One of 'WRAP', 'SAT', 'SAT_SYM', by default 'WRAP'
        bc : Constraint | None, optional
            Constraint for the number of bits, by default MinMax(0, 12)
        ic : Constraint | None, optional
            Constraint for the number of integer bits, by default None
        br : Regularizer | None, optional
            Regularizer for the number of bits, by default None
        ir : Regularizer | None, optional
            Regularizer for the number of integer bits, by default None
        i_decay_speed : numbers, optional
            The decay speed of the integer. Only used if `round_mode` is 'WRAP', by default -1
        homogeneous_axis : Sequence[int] | None, optional
            The axes that are quantized homogeneously. Mutually exclusive with `heterogeneous_axis`. Only used if `bw_mapper` is not set, by default None
        heterogeneous_axis : Sequence[int] | None, optional
            The axes that are quantized heterogeneously. Mutually exclusive with `homogeneous_axis`. Only used if `bw_mapper` is not set, by default None
        bw_mapper : BitwidthMapperBase | None, optional
            The bitwidth mapper to be used. Must be a subclass of `BitwidthMapperBase`. If None, the default bitwidth mapper is used with `homogeneous_axis` and `heterogeneous_axis` as arguments, by default None
        scaler : numbers | None, optional
            The scaling factor to be used. If None, no scaling is applied, by default None
        affine : tuple[numbers, numbers] | None, optional
            Post-quantization affine transformation (scale, shift). If None, no affine transformation is applied, by default None
        qnoise_factor : float | None, optional
            The fraction of quantization strength. If None, treat as full quantization noise, by default None
        """
        ...

    @overload
    def __init__(
        self,
        q_type: str = 'kif',
        place: str = 'datalane',
        *,
        k0: numbers | bool | Initializer = True,
        i0: numbers | Initializer = 4,
        f0: numbers | Initializer = 2,
        round_mode: str | None = 'RND',
        overflow_mode: str | None = 'WRAP',
        ic: Constraint | None = MinMax(-12, 12),
        ir: Regularizer | None = None,
        fc: Constraint | None = MinMax(-10, 10),
        fr: Regularizer | None = None,
        i_decay_speed: numbers = 0.01,
        homogeneous_axis: Sequence[int] | None = (0,),
        heterogeneous_axis: Sequence[int] | None = None,
        bw_mapper: BitwidthMapperBase | None = None,
        scaler: numbers | None = None,
        qnoise_factor: float | None = None,
        **kwargs,
    ) -> None:
        """Fixed point quantizer config with KIF parametrization.

        Parameters
        ----------
        q_type : str
            The type of the quantizer. 'kif' for this implementation.
        place : str
            Where the quantizer is expected to be place. Only affects default config. One of 'weight', 'datalane', 'bias', and 'table'.
        k0 : numbers | bool | Initializer, optional
            If the quantizer allows negative values, by default True
        i0 : numbers | Initializer, optional
            The initial value of the number of integer bits (excl. sign), by default 4
        f0 : numbers | Initializer, optional
            The initial value of the number of fraction bits, by default 2
        round_mode : str, optional
            Rounding mode. One of 'RND', 'RND_CONV', 'TRN', 'S_RND', 'S_RND_CONV', by default 'RND'
        overflow_mode : str, optional
            Overflow mode. One of 'WRAP', 'SAT', 'SAT_SYM', by default 'SAT'
        ic : Constraint | None, optional
            Constraint for the number of integer bits, by default MinMax(-12, 12)
        ir : Regularizer | None, optional
            Regularizer for the number of integer bits, by default None
        fc : Constraint | None, optional
            Constraint for the number of fraction bits, by default MinMax(-12, 12)
        fr : Regularizer | None, optional
            Regularizer for the number of fraction bits, by default None
        i_decay_speed : numbers, optional
            The decay speed of the integer. Only used if `round_mode` is 'WRAP', by default 0.01
        homogeneous_axis : Sequence[int] | None, optional
            The axes that are quantized homogeneously. Mutually exclusive with `heterogeneous_axis`. Only used if `bw_mapper` is not set, by default (0,)
        heterogeneous_axis : Sequence[int] | None, optional
            The axes that are quantized heterogeneously. Mutually exclusive with `homogeneous_axis`. Only used if `bw_mapper` is not set, by default None
        bw_mapper : BitwidthMapperBase | None, optional
            The bitwidth mapper to be used. Must be a subclass of `BitwidthMapperBase`. If None, the default bitwidth mapper is used with `homogeneous_axis` and `heterogeneous_axis` as arguments, by default None
        scaler : numbers | None, optional
            The scaling factor to be used. If None, no scaling is applied, by default None
        affine : tuple[numbers, numbers] | None, optional
            Post-quantization affine transformation (scale, shift). If None, no affine transformation is applied, by default None
        qnoise_factor : float | None, optional
            The fraction of quantization strength. If None, treat as full quantization noise, by default None
        """
        ...

    @overload
    def __init__(
        self,
        q_type: str = 'float',
        place: str = 'datalane',
        *,
        m0: numbers | Initializer | None = 2,
        e0: numbers | Initializer | None = 1,
        e00: numbers | Initializer | None = 0,
        mc: Constraint | None = Min(-1),
        ec: Constraint | None = MinMax(0, 4),
        e0c: Constraint | None = MinMax(-8, 8),
        mr: Regularizer | None = None,
        er: Regularizer | None = None,
        e0r: Regularizer | None = None,
        homogeneous_axis: Sequence[int] | None = (),
        heterogeneous_axis: Sequence[int] | None = None,
        bw_mapper: BitwidthMapperBase | None = None,
        scaler: numbers | None = None,
        qnoise_factor: float | None = None,
        **kwargs,
    ) -> None:
        """Floating point quantizer config.

        Parameters
        ----------
        q_type : str
            The type of the quantizer. 'float' for this implementation.
        place : str
            Where the quantizer is expected to be place. Only affects default config. One of 'weight', 'datalane', 'bias', and 'table'.
        m0 : numbers | Initializer, optional
            The initial value of the number of mantissa bits, by default 2
        e0 : numbers | Initializer, optional
            The initial value of the number of exponent bits, by default 1
        e00 : numbers | Initializer, optional
            The initial value of the number of exponent bits for the first axis, by default 0
        mc : Constraint | None, optional
            Constraint for the number of mantissa bits, by default Min(-1)
        ec : Constraint | None, optional
            Constraint for the number of exponent bits, by default MinMax(0, 4)
        e0c : Constraint | None, optional
            Constraint for the number of exponent bits for the first axis, by default MinMax(-8, 8)
        mr : Regularizer | None, optional
            Regularizer for the number of mantissa bits, by default None
        er : Regularizer | None, optional
            Regularizer for the number of exponent bits, by default None
        e0r : Regularizer | None, optional
            Regularizer for the number of exponent bits for the first axis, by default None
        homogeneous_axis : Sequence[int] | None, optional
            The axes that are quantized homogeneously. Mutually exclusive with `heterogeneous_axis`. Only used if `bw_mapper` is not set, by default ()
        heterogeneous_axis : Sequence[int] | None, optional
            The axes that are quantized heterogeneously. Mutually exclusive with `homogeneous_axis`. Only used if `bw_mapper` is not set, by default None
        bw_mapper : BitwidthMapperBase | None, optional
            The bitwidth mapper to be used. Must be a subclass of `BitwidthMapperBase`. If None, the default bitwidth mapper is used with `homogeneous_axis` and `heterogeneous_axis` as arguments, by default None
        scaler : numbers | None, optional
            The scaling factor to be used. If None, no scaling is applied, by default None
        affine : tuple[numbers, numbers] | None, optional
            Post-quantization affine transformation (scale, shift). If None, no affine transformation is applied, by default None
        qnoise_factor : float | None, optional
            The fraction of quantization strength. If None, treat as full quantization noise, by default None
        """
        ...

    def __init__(
        self,
        q_type: str = 'default',
        place: str = 'datalane',
        scaler: numbers | None = None,
        affine: tuple[numbers, numbers] | None = None,
        qnoise_factor: float | None = None,
        **kwargs,
    ) -> None:
        """Universal quantizer config. The type of the quantizer is specified by the `type` argument.

        Parameters
        ----------
        q_type : str
            The type of the quantizer. One of 'kbi', 'kif', 'float', 'default'. If 'default', the default quantizer type is used, by default 'kbi'. Can be overridden by the `default_q_type` argument of `QuantizerConfigScope`.
        place : str, optional
            The default config to be loaded of the quantizer. One of 'weight', 'datalane', by default 'weight'
        scaler : numbers | None, optional
            The scaling factor to be used. If None, no scaling is applied, by default None
        qnoise_factor : float | None, optional
            The fraction of quantization strength. If None, treat as full quantization noise, by default None
        affine : tuple[numbers, numbers] | None, optional
            Post-quantization affine transformation (scale, shift). If None, no affine transformation is applied, by default None

        **kwargs : Specific parameters for different quantizer types.
        """

        place = place.lower()
        q_type = q_type.lower()
        if q_type == 'default':
            q_type = default_q_type[place]
        if q_type != 'dummy':
            assert (q_type, place) in default_configs, f'Default config for ({q_type}, {place}) not found.'
        self.place = place
        self.q_type = q_type

        self.scaler = None
        self.qnoise_factor = None
        self.affine = None
        if scaler is not None:
            assert scaler != 0, 'scaler must not be 0.'
            self.scaler = float(scaler)
        if affine is not None:
            assert len(affine) == 2, 'affine must be a tuple of (scale, shift).'
            assert affine[0] != 0, 'affine scale must not be 0.'
            self.affine = (float(affine[0]), float(affine[1]))
        if qnoise_factor is not None:
            assert 0 <= qnoise_factor <= 1, 'qnoise_factor must be between 0 and 1.'
            self.qnoise_factor = float(qnoise_factor) if qnoise_factor is not None else None

        if q_type == 'dummy':  # Special case for dummy quantizer
            self.config = {}
            return

        assert kwargs.get('homogeneous_axis') is None or kwargs.get('heterogeneous_axis') is None, (
            'homogeneous_axis and heterogeneous_axis are mutually exclusive. Set only one of them.'
        )

        if kwargs.get('homogeneous_axis') is not None:
            kwargs['heterogeneous_axis'] = None
        if kwargs.get('heterogeneous_axis') is not None:
            kwargs['homogeneous_axis'] = None

        config = default_configs.get((q_type, place))
        assert config is not None, f'Default config for ({q_type}, {place}) not found.'
        self.config = config.copy()

        if self.config is not None:
            for k, v in kwargs.items():
                if k not in self.config:
                    raise ValueError(f'{k} is not a valid parameter for {q_type} quantizer config.')
                self.config[k] = v

    def __getitem__(self, key):
        return self.config[key]

    def __iter__(self):
        return iter(self.config)

    def __len__(self) -> int:
        return len(self.config)

    def __bool__(self):
        return True

    def get_config(self):
        return {
            'q_type': self.q_type,
            'place': self.place,
            'scaler': self.scaler,
            'qnoise_factor': self.qnoise_factor,
            'affine': self.affine,
            **self.config,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**deserialize_keras_object(config))

    def get_quantizer(self) -> TrainableQuantizerBase:
        match self.q_type:
            case 'float':
                return FloatPointQuantizer(**self)  # type: ignore
            case 'kif':
                return FixedPointQuantizerKIF(**self)  # type: ignore
            case 'kbi':
                return FixedPointQuantizerKBI(**self)  # type: ignore
            case 'dummy':
                return DummyQuantizer()
            case _:
                raise ValueError(f'Unknown quantizer type: {self.q_type}')


class QuantizerConfigScope:
    def __init__(
        self,
        q_type: str | Sequence[str] | set[str] = 'all',
        place: str | Sequence[str] | set[str] = 'all',
        default_q_type=None,
        **kwargs,
    ):
        """Override default quantizer config within a context.

        Parameters
        ----------
        q_type : str
            The type of the quantizers.
        place : str
            The location of the quantizers.
        default_q_type : str, optional
            The default quantizer type to be used. If None, the default quantizer type is not changed. One of 'kbi', 'kif', 'float', by default None
        """

        if q_type == 'all':
            q_type = all_quantizer_types()
        if place == 'all':
            place = all_places()

        q_type = (q_type,) if isinstance(q_type, str) else q_type
        place = (place,) if isinstance(place, str) else place
        q_type = {_q_type.lower() for _q_type in q_type}
        place = {_place.lower() for _place in place}

        for _q_type in q_type:
            for _place in place:
                assert (_q_type, _place) in default_configs, f'Default config for ({_q_type}, {_place}) not found.'

        assert kwargs.get('homogeneous_axis') is None or kwargs.get('heterogeneous_axis') is None, (
            'homogeneous_axis and heterogeneous_axis are mutually exclusive. Set only one of them.'
        )

        if kwargs.get('homogeneous_axis') is not None:
            kwargs['heterogeneous_axis'] = None
        if kwargs.get('heterogeneous_axis') is not None:
            kwargs['homogeneous_axis'] = None

        i, f, b = kwargs.get('i0', None), kwargs.get('f0', None), kwargs.get('b0', None)
        if sum((i is not None, f is not None, b is not None)) == 2:
            if i is None:
                kwargs['i0'] = b - f  # type: ignore
            if f is None:
                kwargs['f0'] = b - i  # type: ignore
            if b is None:
                kwargs['b0'] = i + f  # type: ignore

        for k in kwargs:
            if k not in all_quantizer_keys:
                raise ValueError(f'{k} is not a valid parameter for any known quantizer configs.')

        self.q_types = q_type
        self.places = place
        self.kwargs = kwargs
        self.default_q_type = default_q_type
        self._tmp_storage = {}
        self.original_default_q_type = None

    def __enter__(self):
        for (q_type, place), default_conf in default_configs.items():
            if q_type in self.q_types and place in self.places:
                self._tmp_storage[(q_type, place)] = default_conf.copy()
                for k, v in self.kwargs.items():
                    if k in default_conf:
                        default_conf[k] = v
        if self.default_q_type is not None:
            self.original_default_q_type = default_q_type.copy()
            for place in self.places:
                default_q_type[place] = self.default_q_type

    def __exit__(self, exc_type, exc_value, traceback):
        if self.original_default_q_type is not None:
            default_q_type.clear()
            default_q_type.update(self.original_default_q_type)
            self.original_default_q_type = None

        for q_type, place in self._tmp_storage:
            default_configs[(q_type, place)].update(self._tmp_storage[(q_type, place)])
        self._tmp_storage.clear()

    def override(self):
        """Override the default quantizer config."""
        self.__enter__()
        self._tmp_storage.clear()


@register_keras_serializable(package='hgq')
class HardTanhConfig(QuantizerConfig):
    """Grammar sugar for hard tanh quantizer config.
    Equivalent to QuantizerConfig with q_type='kif', k0=1, ic=Constant(0).
    """

    def __init__(
        self,
        q_type: str = 'kif',
        place: str = 'datalane',
        symmetric: bool = False,
        **kwargs,
    ) -> None:
        assert not any(k in kwargs for k in ('k0', 'ic', 'affine', 'overflow_node')), (
            'k0, ic, affine, overflow_mode must not be set for HardTanhConfig.'
        )
        kwargs.pop('i0', None)  # i0 is not used in HardTanhConfig

        overflow_mode = 'SAT_SYM' if symmetric else 'SAT'
        super().__init__(
            q_type=q_type,
            place=place,
            k0=1,
            ic=Constant(0),
            i0=0,
            overflow_mode=overflow_mode,
            affine=None,
            **kwargs,
        )

    def get_config(self):
        conf = super().get_config()
        del conf['ic']
        del conf['k0']
        del conf['i0']
        del conf['overflow_mode']
        del conf['affine']
        return conf


@register_keras_serializable(package='hgq')
class HardSigmoidConfig(QuantizerConfig):
    """Grammar sugar for HardSigmoid quantizer config.
    Equivalent to QuantizerConfig with fixed k0=1, ic=Constant(1), affine=(0.25, 0.5).
    """

    def __init__(
        self,
        q_type: str = 'kif',
        place: str = 'datalane',
        **kwargs,
    ) -> None:
        assert not any(k in kwargs for k in ('k0', 'ic', 'affine', 'overflow_mode')), (
            'k0, ic, affine, overflow_mode must not be set for HardSigmoidConfig.'
        )
        kwargs.pop('i0', None)  # i0 is not used in HardSigmoidConfig

        super().__init__(
            q_type=q_type,
            place=place,
            k0=1,
            ic=Constant(1),
            i0=1,
            overflow_mode='SAT',
            affine=(0.25, 0.5),
            **kwargs,
        )

    def get_config(self):
        conf = super().get_config()
        del conf['ic']
        del conf['k0']
        del conf['i0']
        del conf['overflow_mode']
        del conf['affine']
        return conf
