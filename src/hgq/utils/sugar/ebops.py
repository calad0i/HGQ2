from keras import Model, ops
from keras.callbacks import Callback

from ...layers import QLayerBase


class FreeEBOPs(Callback):
    def on_epoch_end(self, epoch, logs=None):
        assert logs is not None
        assert isinstance(self.model, Model)
        ebops = 0
        for layer in self.model._flatten_layers():
            if isinstance(layer, QLayerBase) and layer.enable_ebops:
                ebops += int(ops.convert_to_numpy(layer._ebops))  # type: ignore
        logs['ebops'] = ebops
