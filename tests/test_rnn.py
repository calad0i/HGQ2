import keras
import numpy as np
import pytest
from keras import layers, ops

from hgq.config import QuantizerConfig, QuantizerConfigScope
from hgq.layers.rnn import QGRU, QSimpleRNN
from hgq.layers.rnn.simple_rnn import QRNN
from tests.base import LayerTestBase


class RNNTestBase(LayerTestBase):
    layer_cls = QRNN
    keras_layer_cls = layers.RNN

    @pytest.fixture(params=((5, 8), (31, 7)))
    def input_shapes(self, request):
        return request.param

    @pytest.fixture
    def layer_kwargs(self, pool_size, strides, padding):
        return {
            'pool_size': pool_size,
            'strides': strides,
            'padding': padding,
        }

    def test_hls4ml_conversion(
        self, model: keras.Model, input_data: np.ndarray, temp_directory: str, use_parallel_io: bool, q_type: str
    ):
        pytest.skip('hls4ml support not yet ready')

    def test_behavior(self, input_data, layer_kwargs):
        raise NotImplementedError()


class TestQSimpleRNN(RNNTestBase):
    layer_cls = QSimpleRNN
    keras_layer_cls = layers.SimpleRNN

    @pytest.fixture(params=[9])
    def units(self, request):
        return request.param

    @pytest.fixture(params=['linear'])
    def activation(self, request):
        return request.param

    @pytest.fixture(params=[True, False])
    def layer_kwargs(self, units, activation, request):
        return {
            'units': units,
            'activation': activation,
            'return_sequences': request.param,
            'use_bias': not request.param,
            'go_backwards': request.param,
        }

    def test_behavior(self, input_data, layer_kwargs):
        layer_kwargs = layer_kwargs.copy()
        layer_kwargs['activation'] = 'tanh'

        keras_layer = self.keras_layer_cls(**layer_kwargs)
        with QuantizerConfigScope(default_q_type='dummy'):
            q_layer = self.layer_cls(**layer_kwargs, enable_ebops=False, paq_conf=QuantizerConfig('dummy'))

        keras_layer.build(input_data.shape)
        q_layer.build(input_data.shape)

        for w0, w1 in zip(keras_layer.weights, q_layer.weights):
            w1.assign(w0)
            assert w0.name == w1.name
        assert len(keras_layer.weights) == len(q_layer.weights)

        keras_output = keras_layer(input_data)
        q_output = q_layer(input_data)

        assert ops.all(keras_output == q_output)


class TestQGRU(RNNTestBase):
    layer_cls = QGRU
    keras_layer_cls = layers.GRU

    @pytest.fixture(params=[9])
    def units(self, request):
        return request.param

    @pytest.fixture(params=['linear'])
    def activation(self, request):
        return request.param

    @pytest.fixture(params=['linear'])
    def recurrent_activation(self, request):
        return request.param

    @pytest.fixture(params=[True, False])
    def layer_kwargs(self, units, activation, recurrent_activation, request):
        return {
            'units': units,
            'activation': activation,
            'recurrent_activation': recurrent_activation,
            'return_sequences': request.param,
            'use_bias': not request.param,
            'go_backwards': request.param,
        }

    def test_behavior(self, input_data, layer_kwargs):
        layer_kwargs['activation'] = 'tanh'
        layer_kwargs['recurrent_activation'] = 'sigmoid'

        keras_layer = self.keras_layer_cls(**layer_kwargs)
        with QuantizerConfigScope(default_q_type='dummy'):
            q_layer = self.layer_cls(
                **layer_kwargs, enable_ebops=False, paq_conf=QuantizerConfig('dummy'), praq_conf=QuantizerConfig('dummy')
            )

        keras_layer.build(input_data.shape)
        q_layer.build(input_data.shape)

        for w0, w1 in zip(keras_layer.weights, q_layer.weights):
            w1.assign(w0)
            assert w0.name == w1.name
        assert len(keras_layer.weights) == len(q_layer.weights)

        keras_output = keras_layer(input_data)
        q_output = q_layer(input_data)

        assert ops.all(keras_output == q_output), f'{keras_output} != {q_output}'
