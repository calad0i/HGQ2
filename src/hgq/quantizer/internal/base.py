from collections.abc import Sequence
from math import prod

import keras
import numpy as np
from keras import ops
from keras.backend import standardize_dtype
from keras.saving import register_keras_serializable
from keras.src import backend
from keras.src.layers import Layer

numbers = int | float | np.integer | np.floating


def _large_number(dtype):
    """Return a Large negative number based on dtype."""
    if standardize_dtype(dtype) == 'float16':
        return 3e4
    return 1e9


@ops.custom_gradient
def round_conv(x):
    qx = ops.round(x)

    def grad(*args, upstream=None):
        if upstream is None:
            (upstream,) = args
        return upstream

    return qx, grad


class BitwidthMapperBase:
    """Abstract base class for mapping bitwidth tensor to input tensors for HG quantizers."""

    def bw_to_x(self, bw, x_shape):
        raise NotImplementedError

    def x_to_bw_absmax(self, x):
        raise NotImplementedError

    def inference_weight_shape(self, input_shape) -> tuple[int, ...]:
        raise NotImplementedError

    def get_config(self):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def check_axis(axis: Sequence[int], ndim: int):
    """Given a list of axis, check that they are valid for a tensor of ndim dimensions. If valid, return the axis as a list of positive integers.

    Parameters
    ----------
    axis : Sequence[int]
        List of axis to check.
    ndim : int
        Number of dimensions of the tensor.

    Returns
    -------
    axis : list[int]
        List of positive integers representing the axis.
    """
    axis = tuple(int(a if a >= 0 else a + ndim) for a in axis)
    assert all(0 <= a < ndim for a in axis), f'Invalid axis {axis} for shape {ndim}.'
    return axis


def to_sparse_pblock(
    *,
    heterogeneous_axis: Sequence[int] | None = None,
    homogeneous_axis: Sequence[int] | None = None,
    pblock: Sequence[int] | Sequence[tuple[int, int]] | dict[int, int] | None = None,
) -> tuple[tuple[tuple[int, int], ...], int | None]:
    """Normalize to the (pblock, rest) repr of `DefaultBitwidthMapper`"""
    assert (heterogeneous_axis is not None) + (homogeneous_axis is not None) + (pblock is not None) == 1, (
        'Exactly one of heterogeneous_axis, homogeneous_axis and pblock must be specified.'
    )
    if pblock is not None:
        if isinstance(pblock, dict):
            spblock = tuple(sorted(pblock.items()))
        else:
            assert isinstance(pblock, (list, tuple))
            if all(isinstance(p, int) for p in pblock):
                spblock = tuple((-i - 1, int(p)) for i, p in enumerate(pblock[::-1]))  # type: ignore
            else:
                assert all(isinstance(p, (list, tuple)) and len(p) == 2 for p in pblock)
                spblock = tuple((int(a), int(p)) for a, p in pblock)  # type: ignore
        return spblock, 1
    if heterogeneous_axis is not None:
        return tuple((a, -1) for a in heterogeneous_axis), 1
    if homogeneous_axis is not None:
        return tuple((a, 1) for a in homogeneous_axis), -1
    return (), -1  # none given: per element along every axis


class TrainableQuantizerBase(Layer):
    """Abstract base class for all quantizers."""

    __dummy__ = False

    def __init__(self, **kwargs):
        pblock = to_sparse_pblock(
            heterogeneous_axis=kwargs.pop('heterogeneous_axis', None),
            homogeneous_axis=kwargs.pop('homogeneous_axis', None),
            pblock=kwargs.pop('pblock', None),
        )
        self.bw_mapper: BitwidthMapperBase = kwargs.pop('bw_mapper', None) or DefaultBitwidthMapper(*pblock)
        self._seed = kwargs.pop('seed', int(np.random.randint(0, 2**31)))
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, training=None): ...

    def __repr__(self) -> str: ...

    def quantize(self, mode):  # type: ignore
        raise ValueError('Quantize method is built-in for keras v3. This method is disabled in this package.')

    def compute_output_shape(self, input_shape):
        return input_shape

    @property
    def bits(self):
        raise NotImplementedError

    @property
    def fbits(self):
        raise NotImplementedError

    @property
    def min(self):
        raise NotImplementedError

    @property
    def max(self):
        raise NotImplementedError

    @property
    def epsilon(self):
        raise NotImplementedError


class DummyQuantizer(TrainableQuantizerBase):
    __dummy__ = True

    def __init__(self, **kwargs):
        super().__init__(heterogeneous_axis=(), **kwargs)  # one zero width over the whole input

    def build(self, input_shape):
        self._bw_shape = self.bw_mapper.inference_weight_shape(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        return inputs

    @property
    def bits(self):
        return ops.zeros(self._bw_shape)

    @property
    def fbits(self):
        return ops.zeros(self._bw_shape)

    @property
    def min(self):
        return -_large_number(self.dtype)

    @property
    def max(self):
        return _large_number(self.dtype)

    @property
    def epsilon(self):
        return backend.epsilon()

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name}, built={self.built})'


@register_keras_serializable(package='hgq')
class DefaultBitwidthMapper(BitwidthMapperBase):
    """Bitwidth mapper sharing widths over a pblock of the input.

    Parameters
    ----------
    pblock : Sequence[tuple[int, int]]
        Sparse repr of the pblock. (ax, tile_size) pairs: use tile_size for axis ax. -1 tile_size means use full axis size. ax can be negative, but no ax can be selected more than once. tile_size does not necessarily divide the axis size.
    rest : int | None, optional
        For axes not listed in pblock, use this tile_size.
    """

    strided = False  # any 1 < p < size: set with the shape

    def __init__(self, pblock: Sequence[tuple[int, int]], rest: int | None = None):
        self.pblock = tuple((int(a), int(p)) for a, p in pblock)
        self.rest = None if rest is None else int(rest)
        assert all(p == -1 or p > 0 for _, p in self.pblock), f'pblock {self.pblock} has a p neither -1 nor positive.'
        assert self.rest is None or self.rest == -1 or self.rest > 0, f'rest {self.rest} is neither -1 nor positive.'

    def inference_weight_shape(self, input_shape):
        N, shape = len(input_shape), tuple(input_shape)
        assert self.rest is not None or len(self.pblock) == N, f'pblock {self.pblock} does not match the rank of {shape}.'
        axes = check_axis([a for a, _ in self.pblock], N)
        assert len(set(axes)) == len(axes), f'pblock {self.pblock} lists an axis of {shape} more than once.'
        by_axis = {a: s for a, (_, s) in zip(axes, self.pblock)}
        if self.rest is None:
            pblock = tuple(by_axis[a] for a in range(N))
        else:
            pblock = tuple(by_axis.get(a, self.rest) for a in range(N))
        for a, (n, p) in enumerate(zip(input_shape, pblock)):
            assert p == 1 or n is not None, f'Unable to heterogeneously quantize axis {a} of unknown size: {shape}.'
        self.bw_shape = tuple((n if p == -1 else min(p, n or 1)) for n, p in zip(input_shape, pblock))
        self.strided = any(1 < p < n for n, p in zip(input_shape, self.bw_shape) if n is not None)
        self._reduce_axis = tuple(a for a, p in enumerate(self.bw_shape) if p == 1)
        return self.bw_shape

    def bw_to_x(self, bw, x_shape):
        """``bw`` mapped onto ``x_shape``, under ``keras.remat`` when ``x_shape`` is dynamic or larger than ``bw``."""
        assert len(x_shape) == len(self.bw_shape), f'x shape {tuple(x_shape)} does not match the rank of widths {self.bw_shape}.'
        if all(isinstance(s, int) for s in x_shape) and prod(bw.shape) >= prod(x_shape):
            return self._bw_to_x(bw, x_shape)
        return keras.remat(lambda bw: self._bw_to_x(bw, x_shape))(bw)

    def _bw_to_x(self, bw, x_shape):
        if not self.strided:
            return ops.broadcast_to(bw, x_shape)
        trim = [s if n > 1 else 1 for n, s in zip(bw.shape, x_shape)]  # each strided axis tiled ceil(s / n) times, trimmed to s
        bw = ops.tile(bw, [-(-s // n) for n, s in zip(bw.shape, trim)])
        return ops.broadcast_to(ops.slice(bw, (0,) * len(trim), trim), x_shape)

    def _reduce(self, x, fn):
        shape: tuple[int, ...] = ops.shape(x)  # type: ignore
        assert len(shape) == len(self.bw_shape), f'x shape {tuple(shape)} does not match the rank of widths {self.bw_shape}.'
        if not self.strided:
            return fn(x, axis=self._reduce_axis, keepdims=True)
        # zeros, neutral for abs-max and any-negative, pad each strided axis to whole tiles
        x = ops.pad(x, [(0, -s % p if p > 1 else 0) for s, p in zip(shape, self.bw_shape)])
        split = [d for s, p in zip(ops.shape(x), self.bw_shape) for d in (s // p, p)]  # type: ignore
        return fn(ops.reshape(x, split), axis=tuple(range(0, 2 * len(shape), 2)))

    def x_to_bw_absmax(self, x):
        return self._reduce(ops.abs(x), ops.amax)

    def x_to_bw_sign(self, x):
        return self._reduce(x < 0, ops.any)

    def get_config(self):
        return {'pblock': self.pblock, 'rest': self.rest}

    @classmethod
    def from_config(cls, config):
        if 'pblock' in config:
            return cls(config['pblock'], config['rest'])
        # compat for < 1.0
        het = config['heterogeneous_axis']
        if het is not None:
            return cls(*to_sparse_pblock(heterogeneous_axis=het))
        return cls(*to_sparse_pblock(homogeneous_axis=config['homogeneous_axis']))
