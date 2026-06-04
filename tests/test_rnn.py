import pytest
from keras import layers, ops

from hgq.config import QuantizerConfig, QuantizerConfigScope
from hgq.layers.rnn import QGRU, QSimpleRNN
from hgq.layers.rnn.simple_rnn import QRNN

from .base import LayerTestBase


class RNNTestBase(LayerTestBase):
    da4ml_not_supported = True
    hls4ml_not_supported = True
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

    def test_weight_quantizers_cached_per_forward(self, input_data, layer_kwargs):
        """Weight quantizers (kq/rkq) must run once per forward, not once per timestep.

        Pre-fix, QGRUCell.qkernel/qrecurrent_kernel re-ran the weight quantizers on
        every timestep of the scan; QGRU.call now pre-quantizes once and caches.
        """
        layer_kwargs = layer_kwargs.copy()
        layer_kwargs['activation'] = 'tanh'
        layer_kwargs['recurrent_activation'] = 'sigmoid'

        with QuantizerConfigScope(default_q_type='dummy'):
            q_layer = self.layer_cls(
                **layer_kwargs, enable_ebops=False, paq_conf=QuantizerConfig('dummy'), praq_conf=QuantizerConfig('dummy')
            )
        q_layer.build(input_data.shape)

        cell = q_layer.cell
        counts: dict[str, int] = {}
        for name in ('kq', 'rkq'):
            quantizer = getattr(cell, name)
            original_call = quantizer.call

            def counting(inputs, training=None, _orig=original_call, _name=name):
                counts[_name] = counts.get(_name, 0) + 1
                return _orig(inputs, training=training)

            quantizer.call = counting

        q_layer(input_data)

        timesteps = input_data.shape[1]
        assert timesteps > 1
        assert counts.get('kq') == 1, f'kernel quantizer ran {counts.get("kq")}x over {timesteps} timesteps'
        assert counts.get('rkq') == 1, f'recurrent-kernel quantizer ran {counts.get("rkq")}x over {timesteps} timesteps'
