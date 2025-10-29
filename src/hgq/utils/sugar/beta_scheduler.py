from collections.abc import Callable, Sequence

import numpy as np
from keras import ops
from keras.callbacks import Callback
from keras.models import Model


class BetaScheduler(Callback):
    """Schedule the beta value of the Q Layers.

    Parameters
    ----------
    beta_fn : Callable[[int], float]
        A function that takes the current epoch and returns the beta value.
    """

    def __init__(self, beta_fn: Callable[[int], float]):
        self.beta_fn = beta_fn

    def on_epoch_begin(self, epoch, logs=None):
        assert isinstance(self.model, Model)

        beta = self.beta_fn(epoch)
        for layer in self.model._flatten_layers():
            if hasattr(layer, '_beta'):
                layer._beta.assign(ops.convert_to_tensor(beta, dtype=layer._beta.dtype))

    def on_epoch_end(self, epoch, logs=None):
        assert isinstance(logs, dict)
        logs['beta'] = self.beta_fn(epoch)


class PieceWiseSchedule:
    """Get interpolated schedule from key points.

    Parameters
    ----------
    intervals : sequence of tuple[epoch:int, beta:float, interp:str]
        The key points of the schedule. Each tuple contains the starting epoch, beta, and interpolation for the interval.

        epoch: the epoch number
        beta: the beta value at that epoch
        interp: the interpolation type in the interval after that epoch, one of 'linear', 'log', 'constant'. After the last epoch defined in the intervals, the beta value will always be constant disregarding the interpolation type.

        Example: `[(0, 0, 'linear'), (10, 1e-5, 'log'), (20, 1e-3, 'constant')]` will start with beta=0, then increase to 1e-5 in 10 epochs linearly, and increase to 1e-3 in another 10 epochs logarithmically. beta will stay at 1e-3 after 20 epochs.
    """

    def __init__(self, intervals: Sequence[tuple[int, float, str]]):
        intervals = sorted(intervals, key=lambda v: v[0])
        epochs = [v[0] for v in intervals]
        betas = [v[1] for v in intervals]
        interpolations = [v[2] for v in intervals]
        assert all(interp in ('linear', 'log', 'constant') for interp in interpolations)

        self.epochs = epochs
        self.betas = betas
        self.interpolations = interpolations

    def __call__(self, epoch):
        idx0 = np.searchsorted(self.epochs, epoch, side='right') - 1
        idx1 = idx0 + 1
        idx0 = max(0, min(idx0, len(self.epochs) - 1))
        idx1 = max(0, min(idx1, len(self.epochs) - 1))
        beta0, beta1 = self.betas[idx0], self.betas[idx1]
        epoch0, epoch1 = self.epochs[idx0], self.epochs[idx1]
        interp = self.interpolations[idx0]

        eps = 1e-9
        match interp:
            case 'linear':
                beta = beta0 + (beta1 - beta0) * (epoch - epoch0) / (epoch1 - epoch0 + eps)
            case 'log':
                beta = beta0 * (beta1 / beta0) ** ((epoch - epoch0) / (epoch1 - epoch0 + eps))
            case 'constant':
                beta = beta0
            case _:
                raise ValueError(f'Invalid interpolation type: {interp}')
        return float(beta)
