import keras
from keras import ops
from keras.callbacks import EarlyStopping

from ...layers.core.base import QLayerBase
from ..misc import gather_vars_to_kwargs


class EarlyStoppingWithEbopsThres(EarlyStopping):
    """Vanilla Keras EarlyStopping but only after a given EBOPs threshold is met.

    This callback stops training when:
    - EBOPs is lower than a given threshold, and
    - monitored metric has stopped improving

    Assuming the goal of a training is to minimize the loss. With this, the
    metric to be monitored would be `'loss'`, and mode would be `'min'`. A
    `model.fit()` training loop will check at end of every epoch whether
    the loss is no longer decreasing, considering the `min_delta` and
    `patience` if applicable. Once it's found no longer decreasing,
    `model.stop_training` is marked True and the training terminates.

    The quantity to be monitored needs to be available in `logs` dict.
    To make it so, pass the loss or metrics at `model.compile()`.

    Parameters
    ----------
    ebops_threshold : float
        The target EBOps value. This callback will not stop the
        training until the model's EBOPs is at or below this value.
    monitor : str, default "val_loss"
        Quantity to be monitored.
    min_delta : float, default 0
        Minimum change in the monitored quantity to qualify as an
        improvement, i.e. an absolute change of less than min_delta, will
        count as no improvement.
    patience : int, default 0
        Number of epochs with no improvement after which training will
        be stopped.
    verbose : int, default 0
        Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1 displays
        messages when the callback takes an action.
    mode : {"auto", "min", "max"}, default "auto"
        In `min` mode, training will stop when the quantity monitored has
        stopped decreasing; in `"max"` mode it will stop when the quantity
        monitored has stopped increasing; in `"auto"` mode, the direction
        is automatically inferred from the name of the monitored quantity.
    baseline : float, optional
        Baseline value for the monitored quantity. If not `None`,
        training will stop if the model doesn't show improvement over the
        baseline.
    restore_best_weights : bool, default False
        Whether to restore model weights from the epoch with the best value
        of the monitored quantity. If `False`, the model weights obtained at
        the last step of training are used. An epoch will be restored
        regardless of the performance relative to the `baseline`. If no epoch
        improves on `baseline`, training will run for `patience` epochs and
        restore weights from the best epoch in that set.
    start_from_epoch : int, default 0
        Number of epochs to wait before starting to monitor improvement.
        This allows for a warm-up period in which no improvement is expected
        and thus training will not be stopped.
    """

    def __init__(
        self,
        ebops_threshold: float,
        monitor='val_loss',
        min_delta: float = 0,
        patience: int = 0,
        verbose: int = 0,
        mode='auto',
        baseline: float | None = None,
        restore_best_weights: bool = False,
        start_from_epoch: int = 0,
    ):
        self.ebops_threshold = ebops_threshold
        kwargs = gather_vars_to_kwargs('self|ebops_threshold')
        super().__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        assert isinstance(self.model, keras.Model)
        stop_training = self.model.stop_training

        ebops = 0
        for layer in self.model._flatten_layers():
            if isinstance(layer, QLayerBase) and layer.enable_ebops:
                ebops += int(ops.convert_to_numpy(layer._ebops))  # type: ignore

        super().on_epoch_end(epoch, logs)
        if ebops > self.ebops_threshold:
            self.model.stop_training = stop_training
