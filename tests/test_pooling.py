from math import log2, prod

import keras
import numpy as np
import pytest
from keras import layers

from hgq.layers import QLayerBase
from hgq.layers.pooling import (
    QAveragePooling1D,
    QAveragePooling2D,
    QGlobalAveragePooling1D,
    QGlobalAveragePooling2D,
    QGlobalMaxPooling1D,
    QGlobalMaxPooling2D,
    QMaxPooling1D,
    QMaxPooling2D,
)

from .base import LayerTestBase


class PoolingTestBase(LayerTestBase):
    layer_cls = QLayerBase
    keras_layer_cls = layers.Layer
    dim = -1

    @pytest.fixture
    def input_shapes(self):
        shape = (5, 8, 5)[: self.dim + 1]
        return shape

    @pytest.fixture(params=['valid', 'same'])
    def padding(self, request):
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
        layer = model.layers[-1]
        if not use_parallel_io and getattr(layer, 'padding', None) == 'same':
            pytest.skip("hls4ml does not support 'same' padding without parallel IO")
        if self.dim == 2 and isinstance(layer, (QAveragePooling2D, QMaxPooling2D)) and layer.strides != layer.pool_size:
            pytest.skip('Known bug in hls4ml vivado/vitis + io stream with 2d pooling with non-trivial strides')
        super().test_hls4ml_conversion(
            model=model,
            input_data=input_data,
            temp_directory=temp_directory,
            use_parallel_io=use_parallel_io,
            q_type=q_type,
        )

    def test_da4ml_conversion(self, model: keras.Model, input_data, overflow_mode: str, temp_directory: str):
        layer = model.layers[-1]

        if isinstance(layer, (QAveragePooling2D, QAveragePooling1D)):
            if getattr(layer, 'padding', None) == 'same':
                pytest.skip('non-pow-2 pool size at boundary')
            if log2(prod(layer.pool_size)) % 1 != 0:
                pytest.skip('non-pow-2 pool size')

        super()._test_da4ml_conversion(
            model=model,
            input_data=input_data,
            overflow_mode=overflow_mode,
            temp_directory=temp_directory,
        )


class GlobalPoolingTestBase(PoolingTestBase):
    @pytest.fixture
    def input_shapes(self):
        shape = (4, 8, 5)[: self.dim + 1]
        return shape

    @pytest.fixture
    def layer_kwargs(self):
        return {'data_format': 'channels_last'}


class TestQMaxPooling1D(PoolingTestBase):
    layer_cls = QMaxPooling1D
    keras_layer_cls = layers.MaxPooling1D
    dim = 1

    @pytest.fixture(params=[2, 4])
    def pool_size(self, request):
        return request.param

    @pytest.fixture(params=[3])
    def strides(self, request):
        return request.param


class TestQAveragePooling1D(PoolingTestBase):
    layer_cls = QAveragePooling1D
    keras_layer_cls = layers.AveragePooling1D
    dim = 1

    @pytest.fixture(params=[2, 4])
    def pool_size(self, request):
        return request.param

    @pytest.fixture(params=[3])
    def strides(self, request):
        return request.param

    def assert_equal(self, keras_output, hls_output):
        return np.testing.assert_allclose(keras_output, hls_output, rtol=0, atol=5e-6)


class TestQMaxPooling2D(PoolingTestBase):
    layer_cls = QMaxPooling2D
    keras_layer_cls = layers.MaxPooling2D
    dim = 2

    @pytest.fixture(params=[(2, 4)])
    def pool_size(self, request):
        return request.param

    @pytest.fixture(params=[(3, 2), (2, 4)])
    def strides(self, request):
        return request.param

    @pytest.fixture(params=['channels_last'])
    def layer_kwargs(self, request, pool_size, strides, padding):
        return {
            'pool_size': pool_size,
            'strides': strides,
            'padding': padding,
            'data_format': request.param,
        }


class TestQAveragePooling2D(PoolingTestBase):
    layer_cls = QAveragePooling2D
    keras_layer_cls = layers.AveragePooling2D
    dim = 2

    @pytest.fixture(params=[(2, 4)])
    def pool_size(self, request):
        return request.param

    @pytest.fixture(params=[(3, 2), (2, 4)])
    def strides(self, request):
        return request.param

    @pytest.fixture(params=['channels_last'])
    def layer_kwargs(self, request, pool_size, strides, padding):
        return {
            'pool_size': pool_size,
            'strides': strides,
            'padding': padding,
            'data_format': request.param,
        }

    def assert_equal(self, keras_output, hls_output):
        return np.testing.assert_allclose(keras_output, hls_output, rtol=0, atol=5e-6)


class TestQGlobalMaxPooling1D(GlobalPoolingTestBase):
    layer_cls = QGlobalMaxPooling1D
    keras_layer_cls = layers.GlobalMaxPooling1D
    dim = 1


class TestQGlobalMaxPooling2D(GlobalPoolingTestBase):
    layer_cls = QGlobalMaxPooling2D
    keras_layer_cls = layers.GlobalMaxPooling2D
    dim = 2


class TestQGlobalAveragePooling1D(GlobalPoolingTestBase):
    layer_cls = QGlobalAveragePooling1D
    keras_layer_cls = layers.GlobalAveragePooling1D
    dim = 1

    def assert_equal(self, keras_output, hls_output):
        return np.testing.assert_allclose(keras_output, hls_output, rtol=0, atol=5e-6)


class TestQGlobalAveragePooling2D(GlobalPoolingTestBase):
    layer_cls = QGlobalAveragePooling2D
    keras_layer_cls = layers.GlobalAveragePooling2D
    dim = 2

    def assert_equal(self, keras_output, hls_output):
        return np.testing.assert_allclose(keras_output, hls_output, rtol=0, atol=5e-6)
