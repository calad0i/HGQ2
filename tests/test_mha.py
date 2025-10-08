import keras
import numpy as np
import pytest
from keras import ops

from hgq.layers import QLinformerAttention, QMultiHeadAttention
from hgq.quantizer.internal import FixedPointQuantizerKBI, FixedPointQuantizerKIF
from tests.base import LayerTestBase


class TestMultiHeadAttention(LayerTestBase):
    layer_cls = QMultiHeadAttention

    @pytest.fixture(params=[1, 2])  # Test different head counts
    def num_heads(self, request):
        return request.param

    @pytest.fixture(params=[8])  # Test different key dimensions
    def key_dim(self, request):
        return request.param

    @pytest.fixture(params=['none'])  # Test different fusion modes
    def fuse(self, request):
        # qkv and kv fused projection not implemented in hls4ml yet
        return request.param

    @pytest.fixture
    def input_shapes(self, fuse: str):
        # Seq..., dim
        if fuse == 'none':
            return (2, 4, 8), (2, 4, 9), (2, 4, 10)
        elif fuse == 'qkv':
            return (2, 4, 8)
        elif fuse == 'kv':
            return (2, 4, 3), (2, 4, 5)
        raise ValueError(f'Invalid fusion mode: {fuse}')

    @pytest.fixture
    def value_dim(self, key_dim):
        return key_dim if self.fuse in ('qkv', 'kv') else key_dim + 1

    @pytest.fixture
    def layer_kwargs(self, num_heads, key_dim, fuse):
        return {'num_heads': num_heads, 'key_dim': key_dim, 'fuse': fuse}

    @pytest.fixture(params=[True])
    def use_parallel_io(self, request) -> bool:
        return request.param

    @pytest.fixture
    def model(self, layer, input_shapes, use_parallel_io):
        """Create test model with the layer"""
        if isinstance(input_shapes[0], int):
            input_shapes = (input_shapes,)
        inputs = [keras.layers.Input(shape=shape) for shape in input_shapes]

        outputs = layer(*inputs)  # Differs to other layers, MHA takes 2-3 inputs (q, v, k=v) not in a list
        model = keras.Model(inputs, outputs)

        self.perturbe_bw(use_parallel_io, model)
        return model

    def perturbe_bw(self, use_parallel_io, model):
        if use_parallel_io:
            for _layer in model._flatten_layers(False):
                if isinstance(_layer, FixedPointQuantizerKBI):
                    b = np.random.randint(4, 8, _layer._b.shape)
                    i = ops.convert_to_numpy(ops.stop_gradient(_layer.i))
                    b = np.minimum(b, 12 - i)
                    if np.all(b == 0):
                        b.ravel()[0] = 1
                    _layer._b.assign(ops.array(b))
                if isinstance(_layer, FixedPointQuantizerKIF):
                    f = np.random.randint(2, 8, _layer._f.shape)
                    i = ops.convert_to_numpy(ops.stop_gradient(_layer.i))
                    f = np.minimum(f, 12 - i)
                    if np.all(i + f == 0):
                        f.ravel()[0] = 1
                    _layer._f.assign(ops.array(f))
        for _layer in model._flatten_layers(False):
            # Randomize activation values
            if hasattr(_layer, 'bias') and isinstance(_layer.bias, keras.Variable):
                bias = np.random.randn(*_layer.bias.shape)
                _layer.bias.assign(ops.array(bias))

    @pytest.fixture
    def input_data(self, input_shapes, N: int = 5000):
        return tuple(np.random.randn(N, *shape).astype(np.float32) * 3 for shape in input_shapes)

    def assert_equal(self, keras_output, hls_output):
        return np.testing.assert_allclose(keras_output, hls_output, atol=1e-6)


class TestLinformerAttention(TestMultiHeadAttention):
    layer_cls = QLinformerAttention

    @pytest.fixture
    def layer_kwargs(self, num_heads, key_dim, fuse, input_shapes):
        lin_kv_proj_dim = [2 for _ in range(len(input_shapes[0]) - 1)]
        return {'num_heads': num_heads, 'key_dim': key_dim, 'fuse': fuse, 'lin_kv_proj_dim': lin_kv_proj_dim}
