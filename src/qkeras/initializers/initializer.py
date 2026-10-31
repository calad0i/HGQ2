from keras import ops
from keras.initializers import Initializer

from hgq.quantizer import Quantizer


class QInitializer(Initializer):
    """QKeras compatible initializer."""

    def __init__(self, initializer: Initializer, quantizer: Quantizer):
        self.initializer = initializer
        self.quantizer = quantizer

    def __call__(self, shape, dtype=None):
        x = self.initializer(shape, dtype)

        # max_x = ops.amax(ops.abs(x))
        std_x = ops.std(x)
        delta = ops.mean(self.quantizer.epsilon_(ops.shape(x)))

        # delta is the minimum resolution of the number system.
        # we want to make sure we have enough values.
        if delta > std_x and hasattr(self.initializer, 'scale'):  # type: ignore
            q = self.quantizer(x)
            max_q = ops.amax(ops.abs(q))
            scale = 1.0
            if max_q == 0.0:
                xx = ops.mean(x * x)
                scale = ops.mean(self.quantizer.max_(ops.shape(x))) / ops.sqrt(xx)  # type: ignore
            else:
                qx = ops.sum(q * x)
                qq = ops.sum(q * q)

                scale = qq / qx  # type: ignore

            self.initializer.scale *= max(scale, 1)  # type: ignore
            x = self.initializer(shape, dtype)

        _min, _max = self.quantizer.min_(ops.shape(x)), self.quantizer.max_(ops.shape(x))
        _min = ops.amax(_min, -_max)  # type: ignore
        return ops.clip(x, _min, _max)

    def get_config(self):
        # TODO: either avoid (stateful) quantizers in the class, or make the init ephemeral. Avoid saving (stateful) quantizers.
        # Error is tentatively raised to avoid saving (stateful) quantizers.
        raise ValueError('QInitializer is ephemeral in hgq/qkeras. It should not be saved.')
