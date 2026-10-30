import copy
import functools

import keras

MODES = ('default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs')


@functools.cache
def _patch_keras():
    import warnings

    import torch
    from keras.layers import Layer

    clear_losses = Layer._clear_losses

    def _clear_losses(self):
        self._losses[:] = []
        clear_losses(self)

    Layer._clear_losses = _clear_losses
    torch._dynamo.config.reorderable_logging_functions.add(warnings.warn)


def _plain_containers(model):
    from keras.layers import BatchNormalization

    for layer in model._flatten_layers():
        sig = layer._call_signature
        if type(sig._parameters) is not dict:
            layer._call_signature = sig = copy.copy(sig)
            sig._parameters = dict(sig.parameters)
        if isinstance(layer, BatchNormalization) and layer._reduction_axes is not None:
            layer._reduction_axes = tuple(layer._reduction_axes)  # type: ignore


def _wrap(make_function, attr: str):
    @functools.wraps(make_function)
    def wrapper(self, force=False):
        jit_compile = self.jit_compile
        if not isinstance(jit_compile, str) or jit_compile == 'auto' or (getattr(self, attr) is not None and not force):
            return make_function(self, force)
        import torch

        if jit_compile in MODES:
            backend, mode = 'inductor', jit_compile
        elif jit_compile in (torch._dynamo.list_backends(exclude_tags=())):
            backend, mode = jit_compile, None
        else:
            backends = torch._dynamo.list_backends(exclude_tags=())
            raise ValueError(f'jit_compile={jit_compile!r}: expected a bool, "auto", a mode {MODES} or a backend {backends}')
        _patch_keras()
        _plain_containers(self)
        self._jit_compile = False  # Keras builds its plain step function
        try:
            make_function(self, force)
        finally:
            self._jit_compile = jit_compile
        compiled = torch.compile(getattr(self, attr), backend=backend, mode=mode)
        if mode == 'reduce-overhead':

            def step(data):
                torch.compiler.cudagraph_mark_step_begin()
                return compiled(data)

            compiled = step
        setattr(self, attr, compiled)

    return wrapper


@functools.cache
def patch_torch_trainer():
    if keras.backend.backend() != 'torch':
        return
    import torch
    from keras.src.backend.torch.trainer import TorchTrainer
    from quantizers._backends import get_backend

    for x in (torch.zeros(0), torch.nn.Parameter(torch.zeros(0))):
        get_backend(x)
    for kind in ('train', 'test', 'predict'):
        name = f'make_{kind}_function'
        setattr(TorchTrainer, name, _wrap(getattr(TorchTrainer, name), f'{kind}_function'))
