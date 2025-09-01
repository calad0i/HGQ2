import keras
import numpy as np
import pytest
from keras import ops
from keras.initializers import RandomNormal, RandomUniform

from hgq.config import QuantizerConfigScope
from hgq.layers import QBatchNormalization
from tests.base import LayerTestBase


class TestBatchNorm(LayerTestBase):
    layer_cls = QBatchNormalization

    @pytest.fixture(params=[2])
    def axis(self, request) -> int:
        return request.param

    @pytest.fixture(params=[True])
    def scale(self, request) -> bool:
        return request.param

    @pytest.fixture(params=[True])
    def center(self, request) -> bool:
        return request.param

    @pytest.fixture
    def layer_kwargs(self, axis, scale, center):
        return {
            'axis': axis,
            'scale': scale,
            'center': center,
            'momentum': 0.99,
            'synchronized': False,
            'moving_mean_initializer': RandomNormal(stddev=1.0),
            'moving_variance_initializer': RandomUniform(minval=0.0, maxval=8.0),
        }

    @pytest.fixture
    def input_shapes(self):
        return (6, 8)

    def test_behavior(self, input_data, layer_kwargs):
        with QuantizerConfigScope(default_q_type='dummy'):
            bn = QBatchNormalization(**{**layer_kwargs, 'momentum': 0.0})  # type: ignore

        input_data = ops.convert_to_tensor(input_data, dtype=bn.dtype)  # type: ignore
        hgq_output = bn(input_data, training=True)
        hgq_output_test = bn(input_data, training=False)
        mean, var = ops.moments(input_data, axes=layer_kwargs['axis'], keepdims=True)  # type: ignore
        ref_output = (input_data - mean) / ops.sqrt(var + bn.epsilon)  # type: ignore

        hgq_output_np: np.ndarray = ops.convert_to_numpy(hgq_output)  # type: ignore
        ref_output_np: np.ndarray = ops.convert_to_numpy(ref_output)  # type: ignore
        hgq_output_test_np: np.ndarray = ops.convert_to_numpy(hgq_output_test)  # type: ignore

        np.allclose(hgq_output_np, ref_output_np, atol=1e-6)
        np.allclose(hgq_output_test_np, ref_output_np)

    def test_da4ml_conversion(self, model: keras.Model, input_data, overflow_mode: str, temp_directory: str):
        super()._test_da4ml_conversion(
            model=model,
            input_data=input_data,
            overflow_mode=overflow_mode,
            temp_directory=temp_directory,
        )
