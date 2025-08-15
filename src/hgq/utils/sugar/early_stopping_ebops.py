import keras
import numpy as np
import warnings

class EarlyStoppingOnEbops(keras.callbacks.Callback):
    """
    Stops training when a monitored metric has stopped improving, but only after
    the model's EBOps threshold value has been reached.

    When 'restore_best_weights' is True, it only saves weights from epochs
    that both improve the monitored metric AND have an EBOps value below the
    specified threshold.

    Arguments:
        ebops_threshold: The target EBOps value. Early stopping and weight saving
            are conditioned on the model's EBOps being at or below this value.
        monitor: Quantity to be monitored (e.g., 'val_loss').
        min_delta: Minimum change in the monitored quantity to qualify as an
            improvement. Defaults to 0.
        patience: Number of epochs with no improvement after which training
            will be stopped. Defaults to 0.
        verbose: Verbosity mode. 0 = silent, 1 = display messages. Defaults to 0.
        mode: One of {'auto', 'min', 'max'}. Determines how the monitored
            metric's improvement is measured. Defaults to 'auto'.
        baseline: Baseline value for the monitored quantity.
        restore_best_weights: Whether to restore model weights from the best
            valid epoch found. Defaults to False.
        start_from_epoch: Number of epochs to wait before starting to monitor
            for improvement. Defaults to 0.
    """

    def __init__(
        self,
        ebops_threshold,
        monitor="val_loss",
        min_delta=0.0,
        patience=0,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
        start_from_epoch=0,
    ):
        super().__init__()

        self.ebops_threshold = ebops_threshold
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.start_from_epoch = start_from_epoch
        self.restore_best_weights = restore_best_weights

        if mode not in ["auto", "min", "max"]:
            warnings.warn(f"EarlyStopping mode '{mode}' is unknown, fallback to 'auto'.")
            mode = "auto"
        self.mode = mode

        if self.mode == "max":
            self.monitor_op = np.greater
        elif self.mode == "min":
            self.monitor_op = np.less
        else:
            if "acc" in self.monitor or "auc" in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.inf if self.monitor_op == np.less else -np.inf

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.start_from_epoch:
            return

        current_val = logs.get(self.monitor)
        if current_val is None:
            warnings.warn(
                f"Early stopping conditioned on metric `{self.monitor}` which is not "
                f"available. Available metrics are: {','.join(logs.keys())}",
                RuntimeWarning
            )
            return

        current_ebops = self._get_model_ebops()

        if self._is_improvement(current_val, self.best):
            self.best = current_val
            self.wait = 0

            # --- ADJUSTED LOGIC HERE ---
            # Only consider saving weights if EBOps is also below the threshold.
            if self.restore_best_weights and current_ebops <= self.ebops_threshold:
                self.best_weights = self.model.get_weights()
                self.best_epoch = epoch
                if self.verbose > 0:
                     print(
                        f"\nEpoch {epoch + 1}: Found new best weights. "
                        f"{self.monitor}: {current_val:.4f}, EBOps: {current_ebops:.2f}"
                     )

        else:
            self.wait += 1

        # Stopping condition remains the same.
        if current_ebops <= self.ebops_threshold and self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.verbose > 0:
                print(
                    f"\nEpoch {epoch + 1}: EBOps ({current_ebops:.2f}) is below threshold "
                    f"({self.ebops_threshold}) and no improvement for {self.patience} epochs. "
                    "Stopping."
                )

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f"Early stopping at epoch {self.stopped_epoch + 1}.")

        if self.restore_best_weights and self.best_weights is not None:
            if self.verbose > 0:
                print(f"Restoring model weights from the end of the best epoch: {self.best_epoch + 1}.")
            self.model.set_weights(self.best_weights)
        elif self.restore_best_weights:
            if self.verbose > 0:
                print("Could not restore weights. No epoch met both metric improvement and EBOps threshold criteria.")


    def _is_improvement(self, current_val, best_val):
        return self.monitor_op(current_val - self.min_delta, best_val)

    def _get_model_ebops(self):
        total_ebops = 0
        for layer in self.model.layers:
            if hasattr(layer, "ebops"):
                total_ebops += getattr(layer, "ebops")
        return total_ebops