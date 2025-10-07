from keras import ops
from keras.constraints import Constraint
from keras.saving import register_keras_serializable
from keras.src import backend

from ..utils.misc import numbers

__all__ = ['MinMax', 'Min', 'Max', 'Constant']


@register_keras_serializable(package='hgq')
class MinMax(Constraint):
    """Constrains the weights to between min_value and max_value."""

    def __init__(self, min_value: numbers, max_value: numbers):
        self.min_value = float(min_value)
        self.max_value = float(max_value)

    def __call__(self, w):
        w = backend.convert_to_tensor(w)
        return ops.clip(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value, 'max_value': self.max_value}


@register_keras_serializable(package='hgq')
class Constant(Constraint):
    """Constrains the weights to a constant value."""

    def __init__(self, value: numbers):
        self.value = float(value)

    def __call__(self, w):
        w = backend.convert_to_tensor(w)
        return ops.full_like(w, self.value)

    def get_config(self):
        return {'value': self.value}


@register_keras_serializable(package='hgq')
class Min(Constraint):
    """Constrains the weights to greater or equal than min_value."""

    def __init__(self, min_value: numbers):
        self.min_value = float(min_value)

    def __call__(self, w):
        w = backend.convert_to_tensor(w)
        return ops.maximum(w, self.min_value)

    def get_config(self):
        return {'min_value': self.min_value}


@register_keras_serializable(package='hgq')
class Max(Constraint):
    """Constrains the weights to less or equal than max_value."""

    def __init__(self, max_value: numbers):
        self.max_value = float(max_value)

    def __call__(self, w):
        w = backend.convert_to_tensor(w)
        return ops.minimum(w, self.max_value)

    def get_config(self):
        return {'max_value': self.max_value}
