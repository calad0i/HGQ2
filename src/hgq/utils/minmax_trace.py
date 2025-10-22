from collections.abc import Sequence

import keras
import numpy as np
from keras import ops
from keras.utils import PyDataset
from numpy.typing import ArrayLike, NDArray
from tqdm import tqdm

from .dataset import Dataset


def _reset_minmax(layer: keras.Layer):
    if hasattr(layer, '_i_decay_speed') and layer.trainable:
        # WRAP-like overflow mode
        shape, dtype = layer._i.shape, layer._i.dtype
        layer._i.assign(keras.ops.full(shape, -1e9, dtype=dtype))
        shape, dtype = layer._k.shape, layer._k.dtype
        layer._k.assign(keras.ops.zeros(shape, dtype=dtype))
    for sublayer in layer._layers:
        _reset_minmax(sublayer)


class TrainingFlagWrapper:
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return self.value == other

    def __bool__(self):
        return self.value is True


def trace_minmax(
    model: keras.Model,
    data: ArrayLike | Sequence[ArrayLike] | PyDataset,
    reset=True,
    batch_size=1024,
    verbose: int | bool = 0,
    return_results=False,
) -> int | NDArray | tuple[NDArray, ...]:
    """With a calibration dataset, find the necessary integer bits required for the quantizers in a model.
    Only needed if `WRAP` overflow mode is used anywhere for the activation quantizers.

    Parameters
    ==========
    model: keras.Model
        The model to trace.
    data: ArrayLike or Sequence[ArrayLike] or PyDataset
        The calibration dataset.
    reset: bool, default True
        Whether to reset the min/max values before tracing. Set to False if you want to continue tracing from previous values.
    batch_size: int, default 1024
        The batch size to use for tracing.
    verbose: int or bool, default 0
        If not o or False, print the EBOPs for each layer after tracing.
        If > 1 or True, show a progress bar during tracing.
    return_results: bool, default False
        If True, return the model outputs on the calibration dataset. If False, return the total EBOPs.
    """
    n_outputs = len(model.outputs)

    if not isinstance(data, PyDataset):
        data = Dataset(data, batch_size=batch_size, device='none')

    if reset:
        _reset_minmax(model)
    record: dict[str, int] = {}

    results = []
    use_pbar = verbose is True or verbose > 1
    n_batch = len(data)  # type: ignore
    n_outputs = len(model.outputs)

    with tqdm(total=n_batch, leave=False, disable=not use_pbar) as pbar:  # type: ignore
        for i in range(n_batch):
            r = model(data[i][0], training=TrainingFlagWrapper('tracing'))
            if return_results:
                results.append(ops.convert_to_numpy(r))
            pbar.update(1)

    record = {}
    for layer in model.layers:
        if getattr(layer, 'enable_ebops', False):
            record[layer.name] = int(layer.ebops)  # type: ignore

    if verbose:
        width = max(max(map(len, record.keys())), 5)
        for k, v in record.items():
            print(f'{k:{width}}: {v}')
        print(f'Total: {sum(record.values())}')

    if return_results:
        if n_outputs == 1:
            return np.concatenate([r for r in results])
        return tuple(np.concatenate([r[i] for r in results]) for i in range(n_outputs))
    else:
        return int(sum(record.values()))
