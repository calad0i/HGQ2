from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import TypedDict, overload

from keras.api.constraints import Constraint
from keras.api.initializers import Initializer
from keras.api.regularizers import Regularizer
from keras.api.saving import deserialize_keras_object, register_keras_serializable

from ...quantizer.base import BitwidthMapperBase
from ...utils.constraints import Min, MinMax
from ..misc import numbers
from ..regularizers import MonoL1

default_q_type = {
    'weight': 'kbi',
    'input': 'kif',
    'bias': 'kbi',
    'table': 'kbi',
}


class QuantizerConfigBase(TypedDict):
    homogeneous_axis: Sequence[int] | None
    heterogeneous_axis: Sequence[int] | None
    bw_mapper: BitwidthMapperBase | None
    trainable: bool


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
    overflow_mode='WRAP',
    bc=MinMax(0, 12),
    ic=None,
    br=MonoL1(1e-6),
    ir=None,
    i_decay_speed=float('inf'),
    homogeneous_axis=(),
    heterogeneous_axis=None,
    bw_mapper=None,
    trainable=True,
)


kbi_input_default = KBIConfig(
    k0=True,
    b0=4,
    i0=2,
    round_mode='RND',
    overflow_mode='SAT_SYM',
    bc=MinMax(0, 12),
    ic=None,
    br=MonoL1(1e-6),
    ir=None,
    i_decay_speed=0.01,
    homogeneous_axis=(0,),
    heterogeneous_axis=None,
    bw_mapper=None,
    trainable=True,
)

kif_weight_default = KIFConfig(
    k0=True,
    i0=4,
    f0=2,
    round_mode='RND',
    overflow_mode='SAT_SYM',
    ic=MinMax(-12, 12),
    ir=MonoL1(1e-6),
    fc=MinMax(-12, 12),
    fr=MonoL1(1e-6),
    i_decay_speed=float('inf'),
    homogeneous_axis=(),
    heterogeneous_axis=None,
    bw_mapper=None,
    trainable=True,
)


kif_input_default = KIFConfig(
    k0=True,
    i0=4,
    f0=2,
    round_mode='RND',
    overflow_mode='SAT_SYM',
    ic=MinMax(-12, 12),
    ir=MonoL1(1e-6),
    fc=MinMax(-12, 12),
    fr=MonoL1(1e-6),
    i_decay_speed=0.01,
    homogeneous_axis=(0,),
    heterogeneous_axis=None,
    bw_mapper=None,
    trainable=True,
)

float_weight_default = FloatConfig(
    m0=2,
    e0=4,
    e00=0,
    mc=MinMax(-1, 8),
    ec=MinMax(0, 4),
    e0c=MinMax(-16, 16),
    mr=MonoL1(1e-6),
    er=MonoL1(1e-6),
    e0r=None,
    homogeneous_axis=(),
    heterogeneous_axis=None,
    bw_mapper=None,
    trainable=True,
)


float_input_default = FloatConfig(
    m0=2,
    e0=4,
    e00=0,
    mc=MinMax(-1, 8),
    ec=MinMax(0, 4),
    e0c=MinMax(-16, 16),
    mr=MonoL1(1e-6),
    er=MonoL1(1e-6),
    e0r=None,
    homogeneous_axis=(0,),
    heterogeneous_axis=None,
    bw_mapper=None,
    trainable=True,
)

default_configs: dict[tuple[str, str], KIFConfig | KBIConfig | FloatConfig] = {
    ('kbi', 'weight'): kbi_weight_default,
    ('kbi', 'bias'): kbi_weight_default.copy(),
    ('kbi', 'table'): kbi_weight_default.copy(),
    ('kbi', 'input'): kbi_input_default,

    ('kif', 'weight'): kif_weight_default,
    ('kif', 'bias'): kif_weight_default.copy(),
    ('kif', 'table'): kif_weight_default.copy(),
    ('kif', 'input'): kif_input_default,

    ('float', 'weight'): float_weight_default,
    ('float', 'bias'): float_weight_default.copy(),
    ('float', 'table'): float_weight_default.copy(),
    ('float', 'input'): float_input_default,
}

all_quantizer_keys = {k for v in default_configs.values() for k in v.keys()} | {'q_type', 'place'}


def all_quantizer_types():
    return {k[0] for k in default_configs.keys()}


def all_places():
    return {k[1] for k in default_configs.keys()}


@register_keras_serializable(package='qkeras_next')
class QuantizerConfig(Mapping):

    @overload
    def __init__(
        self,
        q_type: str = 'default',
        place: str = 'input',
        *,
        k0: numbers | bool | Initializer = True,
        b0: numbers | Initializer = 4,
        i0: numbers | Initializer = 2,
        round_mode: str = 'RND',
        overflow_mode: str = 'WRAP',
        bc: Constraint | None = MinMax(0, 12),
        ic: Constraint | None = None,
        br: Regularizer | None = None,
        ir: Regularizer | None = None,
        i_decay_speed: numbers = -1,
        homogeneous_axis: Sequence[int] | None = None,
        heterogeneous_axis: Sequence[int] | None = None,
        bw_mapper: BitwidthMapperBase | None = None,
    ) -> None:
        """Fixed point quantizer config with KBI parametrization.

        Parameters
        ----------
        q_type : str
            The type of the quantizer. 'kbi' for this implementation.
        place : str
            Where the quantizer is expected to be place. Only affects default config. One of 'weight', 'input', and 'bias'.
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
            The axes that are quantized homogeneously. Mutually exclusive with `heterogeneous_axis`. Only used if `bw_mapper` is `DefaultBitwidthMapper`, by default None
        heterogeneous_axis : Sequence[int] | None, optional
            The axes that are quantized heterogeneously. Mutually exclusive with `homogeneous_axis`. Only used if `bw_mapper` is `DefaultBitwidthMapper`, by default None
        bw_mapper : BitwidthMapperBase | None, optional
            The bitwidth mapper to be used. Must be a subclass of `BitwidthMapperBase`. If None, the default bitwidth mapper is used with `homogeneous_axis` and `heterogeneous_axis` as arguments, by default None
        """
        ...

    @overload
    def __init__(
        self,
        q_type: str = 'default',
        place: str = 'input',
        *,
        k0: numbers | bool | Initializer = True,
        i0: numbers | Initializer = 4,
        f0: numbers | Initializer = 2,
        round_mode: str = 'RND',
        overflow_mode: str = 'SAT',
        ic: Constraint | None = MinMax(-12, 12),
        ir: Regularizer | None = None,
        fc: Constraint | None = MinMax(-12, 12),
        fr: Regularizer | None = None,
        i_decay_speed: numbers = 0.01,
        homogeneous_axis: Sequence[int] | None = (0,),
        heterogeneous_axis: Sequence[int] | None = None,
    ) -> None:
        """Fixed point quantizer config with KIF parametrization.

        Parameters
        ----------
        q_type : str
            The type of the quantizer. 'kif' for this implementation.
        place : str
            Where the quantizer is expected to be place. Only affects default config. One of 'weight', 'input', and 'bias'.
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
            The axes that are quantized homogeneously. Mutually exclusive with `heterogeneous_axis`. Only used if `bw_mapper` is `DefaultBitwidthMapper`, by default (0,)
        heterogeneous_axis : Sequence[int] | None, optional
            The axes that are quantized heterogeneously. Mutually exclusive with `homogeneous_axis`. Only used if `bw_mapper` is `DefaultBitwidthMapper`, by default None
        """
        ...

    @overload
    def __init__(
        self,
        q_type: str = 'default',
        place: str = 'input',
        *,
        m0: numbers | Initializer = 2,
        e0: numbers | Initializer = 1,
        e00: numbers | Initializer = 0,
        mc: Constraint | None = Min(-1),
        ec: Constraint | None = MinMax(0, 4),
        e0c: Constraint | None = MinMax(-8, 8),
        mr: Regularizer | None = None,
        er: Regularizer | None = None,
        e0r: Regularizer | None = None,
        homogeneous_axis: Sequence[int] | None = (),
        heterogeneous_axis: Sequence[int] | None = None,
    ) -> None:
        """Floating point quantizer config.

        Parameters
        ----------
        q_type : str
            The type of the quantizer. 'float' for this implementation.
        place : str
            Where the quantizer is expected to be place. Only affects default config. One of 'weight', 'input', and 'bias'.
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
            The axes that are quantized homogeneously. Mutually exclusive with `heterogeneous_axis`. Only used if `bw_mapper` is `DefaultBitwidthMapper`, by default ()
        heterogeneous_axis : Sequence[int] | None, optional
            The axes that are quantized heterogeneously. Mutually exclusive with `homogeneous_axis`. Only used if `bw_mapper` is `DefaultBitwidthMapper`, by default None
        """
        ...

    def __init__(self, q_type: str = 'default', place: str = 'input', **kwargs) -> None:
        """Universal quantizer config. The type of the quantizer is specified by the `type` argument.

        Parameters
        ----------
        q_type : str
            The type of the quantizer. One of 'kbi', 'kif', 'float', 'default'. If 'default', the default quantizer type is used, by default 'kbi'. Can be overridden by the `default_q_type` argument of `QuantizerConfigScope`.
        place : str, optional
            The default config to be loaded of the quantizer. One of 'weight', 'input', by default 'weight'
        """

        place = place.lower()
        q_type = q_type.lower()
        if q_type == 'default':
            q_type = default_q_type[place]

        assert (q_type, place) in default_configs, f"Default config for ({q_type}, {place}) not found."
        self.place = place
        self.q_type = q_type

        if q_type == 'dummy':  # Special case for dummy quantizer
            self.config = {}

            return

        assert kwargs.get('homogeneous_axis') is None or kwargs.get('heterogeneous_axis') is None, \
            "homogeneous_axis and heterogeneous_axis are mutually exclusive. Set only one of them."

        if kwargs.get('homogeneous_axis') is not None:
            kwargs['heterogeneous_axis'] = None
        if kwargs.get('heterogeneous_axis') is not None:
            kwargs['homogeneous_axis'] = None

        config = default_configs.get((q_type, place))
        assert config is not None, f"Default config for ({q_type}, {place}) not found."
        self.config = config.copy()

        if self.config is not None:
            for k, v in kwargs.items():
                if k not in self.config:
                    raise ValueError(f"{k} is not a valid parameter for {q_type} quantizer config.")
                self.config[k] = v

        self.kwargs = kwargs

    def __getitem__(self, key):
        return self.config[key]

    def __iter__(self):
        return iter(self.config)

    def __len__(self) -> int:
        return len(self.config)

    def get_config(self):
        return {
            'q_type': self.q_type,
            'place': self.place,
            **self.config
        }

    @classmethod
    def from_config(cls, config):
        return cls(**deserialize_keras_object(config))


class QuantizerConfigScope:
    def __init__(self, q_type: str | Sequence[str] | set[str] = 'all', place: str | Sequence[str] | set[str] = 'all', default_q_type=None, **kwargs):
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
                assert (_q_type, _place) in default_configs, f"Default config for ({_q_type}, {_place}) not found."

        assert kwargs.get('homogeneous_axis') is None or kwargs.get('heterogeneous_axis') is None, \
            "homogeneous_axis and heterogeneous_axis are mutually exclusive. Set only one of them."

        if kwargs.get('homogeneous_axis') is not None:
            kwargs['heterogeneous_axis'] = None
        if kwargs.get('heterogeneous_axis') is not None:
            kwargs['homogeneous_axis'] = None

        i, f, b = kwargs.get('i0', None), kwargs.get('f0', None), kwargs.get('b0', None)
        if sum((i is not None, f is not None, b is not None)) == 2:
            if i is None:
                kwargs['i0'] = b - f
            if f is None:
                kwargs['f0'] = b - i
            if b is None:
                kwargs['b0'] = i + f

        for k in kwargs:
            if k not in all_quantizer_keys:
                raise ValueError(f"{k} is not a valid parameter for any known quantizer configs.")

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

        for (q_type, place) in self._tmp_storage:
            default_configs[(q_type, place)].update(self._tmp_storage[(q_type, place)])
        self._tmp_storage.clear()

    def override(self):
        """Override the default quantizer config."""
        self.__enter__()
        self._tmp_storage.clear()
