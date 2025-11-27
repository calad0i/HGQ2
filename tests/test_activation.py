import numpy as np
import pytest
from keras import ops

from hgq.config import QuantizerConfigScope
from hgq.layers import QSoftmax, QUnaryFunctionLUT

from .base import LayerTestBase


def custom_fn(x):
    return ops.tanh(x) - ops.sin(x) + ops.log(ops.abs(x) + 1)  # type: ignore


class TestQUnaryFunctionLUT(LayerTestBase):
    layer_cls = QUnaryFunctionLUT
    custom_objects = {'custom_fn': custom_fn}

    @pytest.fixture(params=['sigmoid', 'tanh', custom_fn])
    def layer_kwargs(self, request):
        return {'activation': request.param}

    @pytest.fixture
    def input_shapes(self):
        return (8,)


class TestSoftmax(LayerTestBase):
    layer_cls = QSoftmax

    @pytest.fixture(params=[-1, (-2, -1)])
    def axis(self, request):  # return request.param
        return request.param

    @pytest.fixture(params=[(8,), (4, 4)])
    def input_shapes(self, request, axis):
        if axis == (-2, -1) and request.param != (4, 4):
            pytest.skip('Only support 2D input with axis=(-2, -1)')
        return request.param

    @pytest.fixture(params=[True, False])
    def stable(self, request):
        return request.param

    @pytest.fixture
    def layer_kwargs(self, axis, stable, use_parallel_io):
        if not use_parallel_io and axis != -1:
            pytest.skip('hls4ml only support axis=-1 with io_stream')

        return {
            'axis': axis,
            'stable': stable,
        }

    def test_behavior(self, input_data, layer_kwargs):
        with QuantizerConfigScope(default_q_type='dummy'):
            softmax = QSoftmax(**layer_kwargs)

        axis = layer_kwargs['axis']
        hgq_output = softmax(input_data)
        if axis == -1:
            ref_output = ops.nn.softmax(input_data, axis=-1)
        else:
            shape = input_data.shape
            input_data = ops.reshape(input_data, shape[:-2] + (-1,))
            ref_output = ops.reshape(ops.nn.softmax(input_data, axis=-1), shape)

        hgq_output_np: np.ndarray = ops.convert_to_numpy(hgq_output)  # type: ignore
        ref_output_np: np.ndarray = ops.convert_to_numpy(ref_output)  # type: ignore

        np.testing.assert_allclose(hgq_output_np, ref_output_np, atol=1e-6)
