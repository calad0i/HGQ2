import keras
import numpy as np
import pytest

from hgq.layers import QConv1D, QConv2D

from .base import LayerTestBase


class TestConv1D(LayerTestBase):
    layer_cls = QConv1D

    @pytest.fixture(params=[1, 4])
    def ch_out(self, request):
        return request.param

    @pytest.fixture(
        params=[
            {'kernel_size': 1, 'strides': 2, 'parallelization_factor': -1},
            {'kernel_size': 3, 'strides': 1, 'parallelization_factor': -1},
            {'kernel_size': 2, 'strides': 2, 'parallelization_factor': 1},
        ]
    )
    def conv_params(self, request):
        return request.param

    @pytest.fixture()
    def input_shapes(self):
        return (6, 2)

    @pytest.fixture(params=['valid', 'same'])
    def padding(self, request):
        return request.param

    @pytest.fixture
    def layer_kwargs(self, ch_out, conv_params):
        return {'filters': ch_out, **conv_params}

    def assert_equal(self, keras_output, hls_output):
        if keras.backend.backend() == 'torch':
            # Torch conv operator introduces some extra numerical error
            return np.testing.assert_allclose(keras_output, hls_output, atol=1e-4)
        return np.testing.assert_allclose(keras_output, hls_output, atol=1e-6)

    def test_da4ml_conversion(self, model: keras.Model, input_data, overflow_mode: str, temp_directory: str):
        super()._test_da4ml_conversion(
            model=model,
            input_data=input_data,
            overflow_mode=overflow_mode,
            temp_directory=temp_directory,
        )


class TestConv2D(LayerTestBase):
    layer_cls = QConv2D

    @pytest.fixture(params=[1, 4])
    def ch_out(self, request):
        return request.param

    @pytest.fixture(
        params=[
            {'kernel_size': (1, 1), 'strides': (3, 5), 'parallelization_factor': -1},
            {'kernel_size': (3, 3), 'strides': (1, 1), 'parallelization_factor': -1},
            {'kernel_size': (3, 2), 'strides': (1, 3), 'parallelization_factor': 1},
            {'kernel_size': (2, 3), 'strides': (4, 2), 'parallelization_factor': 1},
        ]
    )
    def conv_params(self, request):
        return request.param

    @pytest.fixture()
    def input_shapes(self):
        return (6, 6, 2)

    @pytest.fixture(params=['valid', 'same'])
    def padding(self, request):
        return request.param

    @pytest.fixture
    def layer_kwargs(self, ch_out, conv_params):
        return {'filters': ch_out, **conv_params}

    def assert_equal(self, keras_output, hls_output):
        # Conv operations has some numerical error in ops.conv operations.
        if keras.backend.backend() == 'torch':
            # Torch conv operator introduces some extra numerical error
            return np.testing.assert_allclose(keras_output, hls_output, atol=5e-4)
        return np.testing.assert_allclose(keras_output, hls_output, atol=1e-6)

    def test_training(self, model: keras.Model, input_data: np.ndarray, overflow_mode, ch_out: int):
        if keras.backend.backend() == 'torch' and ch_out == 1:
            pytest.skip('Torch runtime error for unknown reason when ch_out is 1.')
        return super().test_training(model, input_data, overflow_mode)

    def test_da4ml_conversion(self, model: keras.Model, input_data, overflow_mode: str, temp_directory: str):
        super()._test_da4ml_conversion(
            model=model,
            input_data=input_data,
            overflow_mode=overflow_mode,
            temp_directory=temp_directory,
        )
