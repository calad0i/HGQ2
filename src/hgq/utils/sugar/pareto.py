import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import keras
import numpy as np
from keras.callbacks import Callback


class ParetoFront(Callback):
    def __init__(
        self,
        path: str | Path,
        metrics: list[str],
        sides: list[int],
        fname_format: str | None = None,
        enable_if: Callable[[dict[str, Any]], bool] | None = None,
    ):
        if fname_format is None:
            fname_format = 'epoch={epoch}'
            for metric in metrics:
                fname_format += f'_{metric}={{{metric}}}'
            fname_format += '.keras'
        else:
            fname_format = fname_format.strip()
        self.path = Path(path)
        self.paths = []
        self.record = []
        self.metrics = metrics
        self.sides = np.array(sides)
        self.enable_if = enable_if
        self.fname_format = fname_format

        _fname_format = fname_format.lower()
        if _fname_format.endswith('.weights.h5') or _fname_format.endswith('.weights.json'):
            self._save_weights = True
        else:
            self._save_weights = False

    def on_train_begin(self, logs=None):
        os.makedirs(self.path, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        assert isinstance(self.model, keras.models.Model)
        assert isinstance(logs, dict)

        logs = logs.copy()
        logs['epoch'] = epoch

        if self.enable_if is not None and not self.enable_if(logs):
            return
        new_metrics = np.array([logs[metric_name] for metric_name in self.metrics])
        _rm_idx = []
        for i, old_metrics in enumerate(self.record):
            _old_metrics = self.sides * old_metrics
            _new_metrics = self.sides * new_metrics
            if np.all(_new_metrics <= _old_metrics):
                return
            if np.all(_new_metrics > _old_metrics):
                _rm_idx.append(i)
        for i in _rm_idx[::-1]:
            self.record.pop(i)
            p = self.paths.pop(i)
            os.remove(p)

        path = self.path / self.fname_format.format(**logs)
        self.record.append(new_metrics)
        self.paths.append(path)
        if self._save_weights:
            self.model.save_weights(path)
        else:
            self.model.save(path)
