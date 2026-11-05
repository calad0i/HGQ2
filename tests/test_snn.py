import keras
import numpy as np
import pytest
from keras import ops

from hgq.layers.snn import QLIF, QSimpleSNN

from .base import LayerTestBase


class SNNTestBase(LayerTestBase):
    hls4ml_not_supported = True
    layer_cls = QSimpleSNN

    @pytest.fixture(params=[(4, 3)])
    def input_shapes(self, request):
        return request.param

    @pytest.fixture
    def input_data(self, input_shapes):
        return np.round(np.random.randn(64, *input_shapes).astype(np.float32).clip(-3, 3) * 16) / 16

    @pytest.fixture(params=[True, False])
    def return_sequences(self, request):
        return request.param

    @pytest.fixture
    def layer_kwargs(self, return_sequences):
        return {
            'units': 3,
            'threshold': [0.75, 1.0, 1.25],
            'learn_graded_spikes_factor': True,
            'return_sequences': return_sequences,
        }

    def test_training(self, model: keras.Model, input_data, overflow_mode: str, *args, **kwargs):
        input_data = ops.convert_to_tensor(input_data, dtype=model.dtype)
        model(input_data, training=True)
        labels = ops.ones(ops.shape(model(input_data)), dtype='float32')

        model.compile(optimizer=keras.optimizers.SGD(), loss=keras.losses.MeanSquaredError())
        loss = model.train_on_batch(input_data, labels)

        assert np.isfinite(loss)

    def cell_kwargs(self, layer_kwargs):
        return {
            'units': layer_kwargs['units'],
            'threshold': layer_kwargs['threshold'],
        }

    def rnn_kwargs(self, layer_kwargs):
        return {
            'return_sequences': layer_kwargs['return_sequences'],
        }


class TestQSNN(SNNTestBase):
    layer_cls = QSimpleSNN


class TestQLIF(SNNTestBase):
    layer_cls = QLIF

    @pytest.fixture
    def layer_kwargs(self, return_sequences):
        return {
            'units': 3,
            'beta': [0.25, 0.5, 0.75],
            'threshold': [0.75, 1.0, 1.25],
            'learn_beta': True,
            'learn_graded_spikes_factor': True,
            'return_sequences': return_sequences,
        }

    def cell_kwargs(self, layer_kwargs):
        kwargs = super().cell_kwargs(layer_kwargs)
        kwargs['beta'] = layer_kwargs['beta']
        return kwargs
