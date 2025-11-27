from math import log2, prod

import numpy as np
import pytest
from keras import layers, ops

from hgq.config import QuantizerConfigScope
from hgq.layers import QLayerBase
from hgq.layers.ops import QAdd, QAveragePow2, QDot, QEinsum, QMaximum, QMeanPow2, QMinimum, QMultiply, QSubtract, QSum

from .base import LayerTestBase


class MergeOpsBase(LayerTestBase):
    layer_cls = QLayerBase
    keras_layer_cls = layers.Layer

    @pytest.fixture(params=[2])
    def input_shapes(self, request):
        shape = (2, 4, 2)
        return [shape] * request.param

    def test_behavior(self, input_data, layer_kwargs, *args, **kwargs):
        with QuantizerConfigScope(default_q_type='dummy'):
            hgq_layer = self.layer_cls(**layer_kwargs)
            keras_layer = self.keras_layer_cls(**layer_kwargs)

        r_hgq: np.ndarray = ops.convert_to_numpy(hgq_layer(input_data))  # type: ignore
        r_keras: np.ndarray = ops.convert_to_numpy(keras_layer(input_data))  # type: ignore
        np.testing.assert_allclose(r_hgq, r_keras)


class TestQAdd(MergeOpsBase):
    layer_cls = QAdd
    keras_layer_cls = layers.Add


class TestQSubtract(MergeOpsBase):
    layer_cls = QSubtract
    keras_layer_cls = layers.Subtract


class TestQDot(MergeOpsBase):
    layer_cls = QDot
    keras_layer_cls = layers.Dot
    # hls4ml qdot supports only 1d vectors

    @pytest.fixture
    def layer_kwargs(self):
        return {'axes': -1}

    @pytest.fixture(params=[2])
    def input_shapes(self, request):
        shape = (8,)
        return [shape] * request.param

    @pytest.fixture(params=[True])
    def use_parallel_io(self, request) -> bool:
        return request.param


class TestQEinsum(MergeOpsBase):
    layer_cls = QEinsum

    @pytest.fixture
    def input_shapes(self, request):
        return ((2, 4, 3), (2, 3, 4))

    @pytest.fixture(params=['Babc,Bacd->Babd', '...abd,...adb->...'])
    def equation(self, request):
        return request.param

    @pytest.fixture
    def layer_kwargs(self, equation):
        return {'equation': equation}

    def test_behavior(self, input_data, layer_kwargs):
        with QuantizerConfigScope(default_q_type='dummy'):
            hgq_layer = self.layer_cls(**layer_kwargs)

        r_hgq: np.ndarray = ops.convert_to_numpy(hgq_layer(input_data))  # type: ignore
        r_keras: np.ndarray = ops.convert_to_numpy(ops.einsum(layer_kwargs['equation'], *input_data))  # type: ignore
        np.testing.assert_allclose(r_hgq, r_keras)

    @pytest.fixture(params=[True])
    def use_parallel_io(self, request) -> bool:
        return request.param


class TestQMaximum(MergeOpsBase):
    layer_cls = QMaximum
    keras_layer_cls = layers.Maximum


class TestQMinimum(MergeOpsBase):
    layer_cls = QMinimum
    keras_layer_cls = layers.Minimum


class TestQMultiply(MergeOpsBase):
    layer_cls = QMultiply
    keras_layer_cls = layers.Multiply


class TestQAveragePow2(MergeOpsBase):
    layer_cls = QAveragePow2

    @pytest.fixture(params=[2])
    def input_shapes(self, request):
        shape = (2, 4, 3)
        return [shape] * request.param

    def test_behavior(self, input_data):
        with QuantizerConfigScope(default_q_type='dummy'):
            hgq_layer = self.layer_cls()

        s = 2.0 ** round(log2(1.0 / len(input_data)))

        r_hgq: np.ndarray = ops.convert_to_numpy(hgq_layer(input_data))  # type: ignore
        r_keras: np.ndarray = ops.convert_to_numpy(ops.sum(input_data, axis=0) * s)  # type: ignore
        np.testing.assert_allclose(r_hgq, r_keras)


# class TestQMatmul(MergeOpsBase):
#     layer_cls = QMatmul

#     @pytest.fixture(params=[((3, 5), (5, 6)), ((3, 4, 5), (3, 5, 6))])
#     def input_shapes(self, request):
#         return request.param

#     def test_behavior(self, input_data):
#         with QuantizerConfigScope(default_q_type='dummy'):
#             hgq_layer = self.layer_cls()

#         r_hgq: np.ndarray = ops.convert_to_numpy(hgq_layer(input_data))  # type: ignore
#         r_keras: np.ndarray = ops.convert_to_numpy(ops.matmul(*input_data))  # type: ignore

#         np.testing.assert_allclose(r_hgq, r_keras)


class TestSum(LayerTestBase):
    layer_cls = QSum
    hls4ml_not_supported = True

    @pytest.fixture(params=[(8,), (2, 3, 4)])
    def input_shapes(self, request):
        return request.param

    @pytest.fixture
    def axes(self, input_shapes):
        if len(input_shapes) == 1:
            return 1
        else:
            return (-2, -1)

    @pytest.fixture
    def layer_kwargs(self, axes):
        return {'axes': axes}

    @pytest.fixture
    def scale(self, input_shapes, axes):
        return 1.0

    def test_behavior(self, input_data, layer_kwargs, scale):
        with QuantizerConfigScope(default_q_type='dummy'):
            hgq_layer = self.layer_cls(**layer_kwargs)

        r_hgq: np.ndarray = ops.convert_to_numpy(hgq_layer(input_data))  # type: ignore
        r_keras: np.ndarray = ops.convert_to_numpy(ops.sum(input_data, axis=layer_kwargs['axes']) * scale)  # type: ignore
        np.testing.assert_allclose(r_hgq, r_keras)


class TestMeanPow2(TestSum):
    layer_cls = QMeanPow2
    hls4ml_not_supported = True

    @pytest.fixture
    def scale(self, input_shapes: tuple[int, ...], axes):
        if isinstance(axes, int):
            scale = 1.0 / input_shapes[axes % (len(input_shapes) + 1) - 1]
        else:
            scale = 1.0 / prod(input_shapes[ax % (len(input_shapes) + 1) - 1] for ax in axes)
        return 2.0 ** round(log2(scale))
