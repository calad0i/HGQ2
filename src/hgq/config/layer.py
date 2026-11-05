from typing import TypedDict, overload

from keras.initializers import Initializer


class GlobalConfig(TypedDict):
    beta0: float
    enable_ebops: bool
    enable_oq: bool
    enable_iq: bool
    enable_sq: bool


global_config = GlobalConfig(
    beta0=1e-5,
    enable_ebops=True,
    enable_oq=False,
    enable_iq=True,
    enable_sq=True,
)


class LayerConfigScope:
    @overload
    def __init__(self, *, beta0: float | None | Initializer = None, enable_ebops: bool | None = None): ...

    @overload
    def __init__(self, **kwargs): ...

    def __init__(self, **kwargs):
        self._override = kwargs

    def __enter__(self):
        self._tmp = global_config.copy()
        for k, v in self._override.items():
            global_config[k] = v

    def __exit__(self, exc_type, exc_val, exc_tb):
        for k in self._override:
            del global_config[k]
        global_config.update(self._tmp)
