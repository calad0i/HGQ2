from math import log10

from keras import ops
from keras.callbacks import Callback
from keras.models import Model


class PID:
    def __init__(self, p: float, i: float, d: float, neg: bool):
        self.p = p
        self.i = i
        self.d = d
        self.integral = 0.0
        self.prev_error = 0.0
        self.neg = neg

    def update_gains(self, p: float, i: float, d: float):
        self.p = p
        self.i = i
        self.d = d

    def update_gains_KcTiTd(self, Kp: float, Ti: float, Td: float):
        self.p = Kp
        self.i = Kp / Ti
        self.d = Kp * Td

    def __call__(self, sv: float, pv: float) -> float:
        err = sv - pv if not self.neg else pv - sv
        self.integral += err
        derivative = err - self.prev_error
        self.prev_error = err
        return self.p * err + self.i * self.integral + self.d * derivative


class BaseBetaPID(Callback):
    def __init__(self, target_ebops: float, p: float, i: float, d: float = 0.0) -> None:
        assert target_ebops > 0, 'Target EBOPs must be greater than 0.'
        self.pid = PID(p, i, d, neg=True)
        self.target_ebops = target_ebops

    def get_ebops(self):
        assert isinstance(self.model, Model)
        ebops: float = 0.0
        for layer in self.model.layers:
            if hasattr(layer, 'ebops'):
                ebops += float(layer.ebops)
        return ebops

    def set_beta(self, beta: float):
        assert isinstance(self.model, Model)
        for layer in self.model._flatten_layers():
            if getattr(layer, '_beta', None) is not None:
                layer._beta.assign(ops.convert_to_tensor(beta, dtype=layer._beta.dtype))

    def on_epoch_begin(self, epoch, logs: dict | None = None):
        assert isinstance(logs, dict)
        ebops = self.get_ebops()
        beta = self.pid(self.target_ebops, ebops)
        beta = max(beta, 0.0)
        self.set_beta(beta)
        logs['beta'] = beta
        logs['ebops'] = ebops


class BetaPID(BaseBetaPID):
    """
    Control the beta value of the Q Layers using a PID controller to reach a specified target EBOPs.

    Parameters
    ----------
    target_ebops : float
        The target EBOPs to reach.
    init_beta : float, optional
        The initial beta value to set before training starts. If None, the average beta of the model is used.
        If initial beta is set, it will be applied to the model at the beginning of training.
    p : float, default 1.0
        The proportional gain of the PID controller.
    i : float, default 2e-3
        The integral gain of the PID controller.
    d : float, default 0.0
        The derivative gain of the PID controller. As EBOPs is noisy, it is recommended to set this to 0.0 or a very small value.
    warmup : int, default 10
        The number of epochs to warm up the beta value. During this period, the beta value will not be updated.
    log : bool, default True
        If True, the beta value and error in EBOPs will be processed in logarithmic scale.
    max_beta : float, default float('inf')
        The maximum beta value to set. If the computed beta exceeds this value, it will be clamped to this maximum.
    min_beta : float, default 0.0
        The minimum beta value to set. If the computed beta is below this value, it will be clamped to this minimum.
    damp_beta_on_target : float, default 0.0
        A damping factor applied to the beta value when the target EBOPs is reached: beta *= (1 - damp_beta_on_target).
        This can help mitigating beta overshooting.
    """

    def __init__(
        self,
        target_ebops: float,
        init_beta: float | None = None,
        p: float = 1.0,
        i: float = 2e-3,
        d: float = 0.0,
        warmup: int = 10,
        log: bool = True,
        max_beta: float = float('inf'),
        min_beta: float = 0.0,
        damp_beta_on_target: float = 0.0,
    ) -> None:
        super().__init__(target_ebops, p, i, d)
        self.warmup = warmup
        self.init_beta = init_beta
        self.max_beta = max_beta
        self.min_beta = min_beta
        self.damp_beta_on_target = damp_beta_on_target
        self.log = log

    def get_avg_beta(self) -> float:
        assert isinstance(self.model, Model)
        n, c = 0, 0.0
        for layer in self.model._flatten_layers():
            if hasattr(layer, '_beta'):
                c += float(layer._beta.numpy())
                n += 1
        return c / n if n > 0 else 0.0

    def on_train_begin(self, logs=None):
        if self.init_beta is not None:
            self.set_beta(self.init_beta)
        self.beta = self.get_avg_beta() if self.init_beta is None else self.init_beta
        self._ebops = self.get_ebops()

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        assert isinstance(logs, dict)

        if epoch < self.warmup:
            if self.init_beta is not None:
                self.set_beta(self.init_beta)
            return

        ebops = self._ebops
        if epoch == self.warmup:
            # match integral term ST init beta = beta for the first epoch
            # beta = p * P + i * I
            if not self.log:
                err = 1 - ebops / self.target_ebops
                self.pid.integral = (self.beta - self.pid.p * err) / self.pid.i - err
            else:
                err = log10(ebops / self.target_ebops + 1e-9)
                self.pid.integral = (log10(self.beta) - self.pid.p * err) / self.pid.i - err

        if self.log:
            beta = 10.0 ** self.pid(0, log10(ebops / self.target_ebops))
        else:
            beta = self.pid(1, ebops / self.target_ebops)

        if ebops < self.target_ebops:
            beta *= 1 - self.damp_beta_on_target

        beta = max(min(beta, self.max_beta), self.min_beta)
        self.set_beta(beta)
        self.beta = beta

    def on_epoch_end(self, epoch: int, logs: dict | None = None):
        assert isinstance(logs, dict)
        logs['beta'] = self.beta
        ebops = self.get_ebops()
        logs['ebops'] = ebops
        self._ebops = ebops
