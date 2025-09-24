from math import ceil
from warnings import warn

import keras
from keras import ops
from keras.utils import PyDataset


class Dataset(PyDataset):
    def __init__(self, x_set, y_set=None, batch_size=None, device: str = 'cpu:0', drop_last=False, **kwargs):
        super().__init__(**kwargs)

        n_inp = 1
        if isinstance(x_set, (list, tuple)):
            n_inp = len(x_set)
            if n_inp > 10:
                warn(f'You have {n_inp} input tensors. Are you sure this is correct?')

        n_out = 1 if y_set is not None else None
        if isinstance(y_set, (list, tuple)):
            n_out = len(y_set)
            if n_out > 10:
                warn(f'You have {n_out} output tensors. Are you sure this is correct?')

        self.n_inp = n_inp
        self.n_out = n_out

        to_tensor = ops.convert_to_tensor if device != 'none' else lambda x: x
        self.device = device

        device = 'cpu:0' if device == 'none' else device

        with keras.device(device):
            if self.n_inp == 1:
                self.x = to_tensor(x_set)
                self.data_len = len(x_set)
            else:
                self.x = tuple(to_tensor(x) for x in x_set)
                self.data_len = len(x_set[0])
                assert all(len(x) == self.data_len for x in x_set), 'All input tensors must have the same length'
            if y_set is not None:
                if self.n_out == 1:
                    self.y = to_tensor(y_set)
                    assert len(y_set) == self.data_len, 'Output tensor must have the same length as input tensor'
                else:
                    self.y = tuple(to_tensor(y) for y in y_set)
                    assert all(len(y) == self.data_len for y in y_set), (
                        'All output tensors must have the same length as input tensor'
                    )
            else:
                self.y = None

        self.batch_size = batch_size
        self.drop_last = drop_last

    def __len__(self):
        assert self.batch_size is not None, 'batch_size must be set'
        if self.drop_last:
            return self.data_len // self.batch_size
        return ceil(self.data_len / self.batch_size)

    def batch(self, batch_size):
        self.batch_size = batch_size

    def __getitem__(self, idx):
        if idx < 0:
            idx = idx % len(self)
        low = idx * self.batch_size
        high = min(low + self.batch_size, self.data_len)
        if self.n_inp == 1:
            batch_x = self.x[low:high]  # type: ignore
        else:
            batch_x = tuple(x[low:high] for x in self.x)  # type: ignore
        if self.y is not None:
            if self.n_out == 1:
                batch_y = self.y[low:high]  # type: ignore
            else:
                batch_y = tuple(y[low:high] for y in self.y)  # type: ignore
        else:
            batch_y = float('nan')
        return batch_x, batch_y
