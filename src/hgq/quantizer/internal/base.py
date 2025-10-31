from collections.abc import Sequence

import numpy as np
from keras import ops
from keras.saving import register_keras_serializable
from keras.src import backend
from keras.src.layers import Layer

numbers = int | float | np.integer | np.floating


def _large_number(dtype):
    """Return a Large negative number based on dtype."""
    if backend.standardize_dtype(dtype) == 'float16':
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
    axis = [a if a >= 0 else a + ndim for a in axis]
    assert all(0 <= a < ndim for a in axis), f'Invalid axis {axis} for shape {ndim}.'
    return axis


class TrainableQuantizerBase(Layer):
    """Abstract base class for all quantizers."""

    __dummy__ = False

    def __init__(self, **kwargs):
        homogeneous_axis = kwargs.pop('homogeneous_axis', ())
        heterogeneous_axis = kwargs.pop('heterogeneous_axis', None)
        bw_mapper: BitwidthMapperBase = kwargs.pop('bw_mapper', None) or DefaultBitwidthMapper(
            heterogeneous_axis, homogeneous_axis
        )
        self.bw_mapper = bw_mapper
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

    def call(self, inputs, training=None):
        return inputs

    @property
    def bits(self):
        return ops.convert_to_tensor(0, 'float32')

    @property
    def fbits(self):
        return ops.convert_to_tensor(0, 'float32')

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
    """Default bitwidth mapper for HG quantizers."""

    def __init__(self, heterogeneous_axis: Sequence[int] | None = None, homogeneous_axis: Sequence[int] | None = None, **kwargs):
        _shape_inferenced = kwargs.pop('_shape_inferenced', False)
        if not _shape_inferenced:
            assert (heterogeneous_axis is None) ^ (homogeneous_axis is None), (
                'Only one of quantize_dims and skip_dims can be specified.'
            )
        self.heterogeneous_axis = heterogeneous_axis
        self.homogeneous_axis = homogeneous_axis
        self._shape_inferenced = _shape_inferenced
        super().__init__(**kwargs)

    def inference_weight_shape(self, input_shape):
        N = len(input_shape)
        axis = np.arange(N)
        if self.heterogeneous_axis is not None:
            self.heterogeneous_axis = check_axis(self.heterogeneous_axis, N)  # type: ignore
            self.homogeneous_axis = tuple(np.setdiff1d(axis, self.heterogeneous_axis))
        elif self.homogeneous_axis is not None:
            self.homogeneous_axis = check_axis(self.homogeneous_axis, N)  # type: ignore
            self.heterogeneous_axis = tuple(np.setdiff1d(axis, self.homogeneous_axis))

        weight_shape = [1] * N
        for i in self.heterogeneous_axis:  # type: ignore
            assert input_shape[i] is not None, (
                f'Unable to heterogeneously quantize axis {i} with unknown shape. Input shape: {input_shape}.'
            )
            weight_shape[i] = input_shape[i]
        self._shape_inferenced = True
        return tuple(weight_shape)

    def bw_to_x(self, bw, x_shape):
        return ops.broadcast_to(bw, x_shape)

    def x_to_bw_absmax(self, x):
        return ops.max(ops.abs(x), axis=self.homogeneous_axis, keepdims=True)

    def x_to_bw_sign(self, x):
        return ops.any(x < 0, axis=self.homogeneous_axis, keepdims=True)

    def get_config(self):
        return dict(
            heterogeneous_axis=self.heterogeneous_axis,
            homogeneous_axis=self.homogeneous_axis,
            _shape_inferenced=self._shape_inferenced,
        )
