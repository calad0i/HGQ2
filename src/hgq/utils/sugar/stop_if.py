from collections.abc import Callable
from typing import Any

import keras
from keras.callbacks import Callback


class StopIf(Callback):
    """
    Stop training when the condition function returns True for `patience_epochs` consecutive epochs or `patience_steps` consecutive steps.

    Parameters
    ----------
    stop_condition : Callable[[dict[str, Any]], bool]
        A function that takes the `logs` dict and returns True if the stop condition is met.
    patience_epochs : int | None, default 0
        Number of consecutive epochs for which the stop condition needs to be met before stopping. If None, epoch-based stopping is disabled.
    patience_steps : int | None, default None
        Number of consecutive steps for which the stop condition needs to be met before stopping. If None, step-based stopping is disabled.
    verbose : int, default 0
        Verbosity, 0 or 1.


    Example
    -------
    To stop training when EBOPs is at or below some value for 3 consecutive epochs:
    `stop_on_collapse = StopIf(lambda logs: logs['ebops'] < thres, patience_epochs=3)`
    Then pass `stop_on_collapse` to callbacks **after** the callback that computes EBOPs, e.g. `FReeEBOPs`:

    ```python
    stop_on_collapse = StopIf(lambda logs: logs['ebops'] < thres, patience_epochs=3)
    ebops = FReeEBOPs()
    model.fit(..., callbacks=[ebops, ..., stop_on_collapse])
    ```

    To stop training when the kernel of layer 'that_layer' is all zero immediately:
    `stop_on_zero_weights = StopIf(lambda logs, model: keras.ops.all(model.get_layer('that_layer').kernel==0), patience_steps=0)`
    """

    def __init__(
        self,
        stop_condition: Callable[[dict[str, Any]], bool] | Callable[[dict[str, Any], keras.Model], bool],
        patience_epochs: int | None = 0,
        patience_steps: int | None = None,
        verbose: int = 0,
    ):
        if stop_condition.__code__.co_argcount == 1:
            _stop_condition = lambda logs, model: stop_condition(logs)  # type: ignore
        else:
            _stop_condition = stop_condition
        self.stop_condition: Callable[[dict[str, Any], keras.Model], bool] = _stop_condition  # type: ignore
        self.patience_epochs = patience_epochs
        self.patience_steps = patience_steps
        self.counter_epochs = 0
        self.counter_steps = 0
        self.verbose = verbose

    def kill(self):
        assert isinstance(self.model, keras.Model)
        if self.verbose > 0:
            if (self.patience_epochs or 0) < self.counter_epochs:
                src, count = 'epoch', self.counter_epochs
            else:
                src, count = 'step', self.counter_steps
            print(f'Stopping training as the stop condition has been met for {count} consecutive {src}s.')
        self.model.stop_training = True

    def on_epoch_end(self, epoch, logs=None):
        if self.patience_epochs is None:
            return
        if self.stop_condition(logs, self.model):  # type: ignore
            self.counter_epochs += 1
            if self.counter_epochs > self.patience_epochs:
                self.kill()
        else:
            self.counter_epochs = 0

    def on_train_batch_end(self, batch, logs=None):
        if self.patience_steps is None:
            return
        if self.stop_condition(logs, self.model):  # type: ignore
            self.counter_steps += 1
            if self.counter_steps > self.patience_steps:
                self.kill()
        else:
            self.counter_steps = 0
