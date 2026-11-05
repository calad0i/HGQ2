import keras
import numpy as np
import pytest
from keras import ops

from hgq.layers import (
    QDepthwiseConv1D,
    QDepthwiseConv2D,
    QSeparableConv1D,
    QSeparableConv2D,
)

from .base import LayerTestBase


class _ConvTestMixin:
    def assert_equal(self, keras_output, hw_output, lsb_step=None, ignore_err=0.0):
        if keras.backend.backend() == 'torch':
            # Torch conv operator introduces some extra numerical error
            lsb_step = lsb_step + 1e-4 if lsb_step is not None else 5e-4
        return super().assert_equal(keras_output, hw_output, lsb_step, ignore_err)  # type: ignore

    def test_hls4ml_conversion(self, model, input_data, temp_directory, use_parallel_io, q_type, ignore_err, depth_multiplier):
        if depth_multiplier > 1:
            pytest.skip('The hls4ml Vivado/Vitis depthwise kernels do not support depth_multiplier > 1')
        super().test_hls4ml_conversion(model, input_data, temp_directory, use_parallel_io, q_type, ignore_err)  # type: ignore

    def test_alkaid_conversion(self, model, input_data, overflow_mode, q_type, ignore_err):
        try:
            super().test_alkaid_conversion(model, input_data, overflow_mode, q_type, ignore_err)  # type: ignore
        except StopIteration:
            # alkaid's cmvm builds an FVArray from constants only (no FVariable to take the hwconf from) when every
            # input of a depthwise group quantizes to 0 bits
            pytest.xfail('alkaid cmvm fails on a depthwise group with only constant inputs')


class TestDepthwiseConv1D(_ConvTestMixin, LayerTestBase):
    layer_cls = QDepthwiseConv1D

    @pytest.fixture(params=[1, 2])
    def depth_multiplier(self, request):
        return request.param

    @pytest.fixture(
        params=[
            {'kernel_size': 3, 'strides': 1, 'parallelization_factor': -1},
            {'kernel_size': 2, 'strides': 2, 'parallelization_factor': 1},
        ]
    )
    def conv_params(self, request):
        return request.param

    @pytest.fixture(params=['valid', 'same'])
    def padding(self, request):
        return request.param

    @pytest.fixture()
    def input_shapes(self):
        return (6, 3)

    @pytest.fixture
    def layer_kwargs(self, depth_multiplier, conv_params, padding):
        return {'depth_multiplier': depth_multiplier, 'padding': padding, **conv_params}


class TestDepthwiseConv2D(TestDepthwiseConv1D):
    layer_cls = QDepthwiseConv2D

    @pytest.fixture(
        params=[
            {'kernel_size': (3, 3), 'strides': (1, 1), 'parallelization_factor': -1},
            {'kernel_size': (3, 2), 'strides': (2, 1), 'parallelization_factor': 1},
        ]
    )
    def conv_params(self, request):
        return request.param

    @pytest.fixture()
    def input_shapes(self):
        return (6, 6, 3)


class TestSeparableConv1D(_ConvTestMixin, LayerTestBase):
    layer_cls = QSeparableConv1D

    @pytest.fixture(params=[1, 4])
    def ch_out(self, request):
        return request.param

    @pytest.fixture(params=[1, 2])
    def depth_multiplier(self, request):
        return request.param

    @pytest.fixture(
        params=[
            {'kernel_size': 3, 'strides': 1, 'parallelization_factor': -1},
            {'kernel_size': 2, 'strides': 2, 'parallelization_factor': 1},
        ]
    )
    def conv_params(self, request):
        return request.param

    @pytest.fixture(params=['valid', 'same'])
    def padding(self, request):
        return request.param

    @pytest.fixture()
    def input_shapes(self):
        return (6, 3)

    @pytest.fixture
    def layer_kwargs(self, ch_out, depth_multiplier, conv_params, padding):
        return {'filters': ch_out, 'depth_multiplier': depth_multiplier, 'padding': padding, **conv_params}

    def test_training(self, model: keras.Model, input_data: np.ndarray, overflow_mode, ch_out: int):
        if ch_out == 1:
            # The pointwise kernel then has only n_chan * depth_multiplier weights; with the random small bit widths
            # they can all quantize to 0, and no gradient reaches the depthwise kernel.
            pytest.skip('Single-filter pointwise kernel may quantize to all zeros.')
        return super().test_training(model, input_data, overflow_mode)  # type: ignore


class TestSeparableConv2D(TestSeparableConv1D):
    layer_cls = QSeparableConv2D

    @pytest.fixture(
        params=[
            {'kernel_size': (3, 3), 'strides': (1, 1), 'parallelization_factor': -1},
            {'kernel_size': (2, 3), 'strides': (2, 2), 'parallelization_factor': 1},
        ]
    )
    def conv_params(self, request):
        return request.param

    @pytest.fixture()
    def input_shapes(self):
        return (6, 6, 3)


def test_depthwise_matches_grouped_conv():
    """QDepthwiseConv2D is a grouped conv whose output channel c * depth_multiplier + m is kernel m of channel c."""
    layer = QDepthwiseConv2D(3, padding='same', depth_multiplier=2)
    x = np.random.randn(4, 6, 6, 3).astype(np.float32)
    y = ops.convert_to_numpy(layer(x))
    xq = ops.convert_to_numpy(layer.iq(x))
    k = ops.convert_to_numpy(layer.qkernel)  # (3, 3, 1, 3 * 2)
    assert k.shape == (3, 3, 1, 6)
    b = ops.convert_to_numpy(layer.qbias)
    ref = np.zeros_like(y)
    xp = np.pad(xq, ((0, 0), (1, 1), (1, 1), (0, 0)))
    for c in range(3):
        for m in range(2):
            for dy in range(3):
                for dx in range(3):
                    ref[..., c * 2 + m] += xp[:, dy : dy + 6, dx : dx + 6, c] * k[dy, dx, 0, c * 2 + m]
            ref[..., c * 2 + m] += b[c * 2 + m]
    np.testing.assert_allclose(y, ref, rtol=1e-5, atol=1e-5)


def test_separable_ebops_is_sum_of_steps():
    """EBOPs of a separable conv are those of its depthwise step plus those of its pointwise step."""
    from hgq.config import LayerConfigScope, QuantizerConfigScope

    with LayerConfigScope(enable_ebops=True, beta0=0.0), QuantizerConfigScope(place='datalane', homogeneous_axis=(0,)):
        sep = QSeparableConv2D(4, 3, padding='same', use_bias=False)
        inp = keras.Input((6, 6, 3))
        model = keras.Model(inp, sep(inp))
    x = np.random.randn(8, 6, 6, 3).astype(np.float32)
    model(x, training=True)
    bw_in = sep.depthwise.iq.bits_((1, 6, 6, 3))
    bw_dk = sep.dkq.bits_(ops.shape(sep.depthwise.kernel))
    bw_mid = sep.pq.bits_((1, 6, 6, 3))
    bw_pk = sep.pkq.bits_(ops.shape(sep.pointwise.kernel))
    dw = ops.sum(ops.depthwise_conv(bw_in, ops.reshape(bw_dk, (3, 3, 3, 1)), padding='same'))
    pw = ops.sum(ops.conv(bw_mid, bw_pk))
    np.testing.assert_allclose(float(sep.ebops), float(dw + pw), rtol=1e-5)


def test_separable_is_depthwise_then_pointwise():
    """QSeparableConv2D equals its QDepthwiseConv2D and 1x1 QConv2D sublayers applied in series."""
    sep = QSeparableConv2D(4, 3, padding='same', depth_multiplier=2, activation='relu')
    x = np.random.randn(4, 6, 6, 3).astype(np.float32)
    y = ops.convert_to_numpy(sep(x))
    ref = ops.convert_to_numpy(sep.pointwise(sep.depthwise(x)))
    np.testing.assert_array_equal(y, ref)
    assert sep.depthwise.kernel.shape == (3, 3, 1, 6) and sep.pointwise.kernel.shape == (1, 1, 6, 4)
    assert sep.depthwise.groups == 3 and not sep.depthwise.use_bias
