import keras
from keras.callbacks import Callback

class StopOnCollapse(Callback):
    """
    Stop training when the model's EBOPs collapses to beyond threshold.

    Can monitor accuracy, loss, ebops or other metrics.
    The training will stop when the monitored metric meets the specified condition
    (e.g., falls below a certain threshold for 'min' mode or rises above a certain threshold for 'max' mode).

    Examples:

    Monitor EBOPs and stop training when it falls below threshold:
    ```python
    stop_on_collapse = StopOnCollapse(stop_condition=1e6, stop_monitor='ebops', stop_mode='min', patience=0, verbose=1)
    model.fit(..., callbacks=[stop_on_collapse])
    ```

    Monitor accuracy for 5 class classification and stop training when model collapses to random guessing:
    ```python
    stop_on_collapse = StopOnCollapse(stop_condition=0.21, stop_monitor='accuracy', stop_mode='min', patience=0, verbose=1)
    model.fit(..., callbacks=[stop_on_collapse])
    ```

    Parameters
    ----------
    stop_condition : float
    stop_monitor : string
    stop_mode : string
    patience : int
    verbose : int  

    """

    def __init__(
        self,
        stop_condition: float,
        stop_monitor: str = 'ebops',
        stop_mode: str = 'min',
        patience: int = 0,
        verbose: int = 0,
    ):
        self.stop_condition = stop_condition
        self.stop_monitor = stop_monitor
        self.stop_mode = stop_mode
        self.verbose = verbose
        self.patience = patience

    def get_condition(self, monitor, stop_mode):
        if stop_mode == 'min':
            return monitor <= self.stop_condition
        elif stop_mode == 'max':
            return monitor >= self.stop_condition
        else:
            raise ValueError(f'Invalid stop_mode: {stop_mode}. Must be "min" or "max".')    

    def on_epoch_end(self, epoch, logs=None):
        assert isinstance(self.model, keras.Model)

        monitor = logs.get(self.stop_monitor)

        if monitor is not None and self.get_condition(monitor, self.stop_mode):
            if self.verbose > 0:
                print(
                    f'\nEpoch {epoch + 1}: stopping training as {self.stop_monitor} has collapsed to {monitor} (threshold: {self.stop_condition}).'
                )
            self.model.stop_training = True