import keras
import numpy as np
import pytest
from keras import ops

from hgq.layers import (
    QDepthwiseConv1D,
    QDepthwiseConv2D,
    QPointwiseConv1D,
    QPointwiseConv2D,
    QSeparableConv1D,
    QSeparableConv2D,
)

from .base import LayerTestBase


class _ConvTestMixin:
    alkaid_not_supported = True  # no alkaid replay for these layers yet

    def assert_equal(self, keras_output, hw_output, lsb_step=None, ignore_err=0.0):
        if keras.backend.backend() == 'torch':
            # Torch conv operator introduces some extra numerical error
            lsb_step = lsb_step + 1e-4 if lsb_step is not None else 5e-4
        return super().assert_equal(keras_output, hw_output, lsb_step, ignore_err)  # type: ignore


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
        if ch_out == 1 and overflow_mode == 'SAT':
            # The pointwise kernel then has only n_chan * depth_multiplier weights; with the random small bit widths
            # they can all quantize to 0, and no gradient reaches the depthwise kernel.
            pytest.skip('Single-filter pointwise kernel may quantize to all zeros in SAT mode.')
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


class TestPointwiseConv1D(_ConvTestMixin, LayerTestBase):
    layer_cls = QPointwiseConv1D

    @pytest.fixture(params=[1, 4])
    def ch_out(self, request):
        return request.param

    @pytest.fixture()
    def input_shapes(self):
        return (6, 3)

    @pytest.fixture
    def layer_kwargs(self, ch_out):
        return {'filters': ch_out}


class TestPointwiseConv2D(TestPointwiseConv1D):
    layer_cls = QPointwiseConv2D

    @pytest.fixture()
    def input_shapes(self):
        return (6, 6, 3)


def test_pointwise_kernel_size():
    with pytest.raises(ValueError):
        QPointwiseConv2D(4, kernel_size=3)
    assert QPointwiseConv2D(4, kernel_size=(1, 1)).kernel_size == (1, 1)


def test_depthwise_matches_grouped_conv():
    """QDepthwiseConv2D matches a direct reference: each channel convolved with its own depth_multiplier kernels."""
    layer = QDepthwiseConv2D(3, padding='same', depth_multiplier=2)
    x = np.random.randn(4, 6, 6, 3).astype(np.float32)
    y = ops.convert_to_numpy(layer(x))
    xq = ops.convert_to_numpy(layer.iq(x))
    k = ops.convert_to_numpy(layer.qkernel)
    b = ops.convert_to_numpy(layer.qbias)
    ref = np.zeros_like(y)
    xp = np.pad(xq, ((0, 0), (1, 1), (1, 1), (0, 0)))
    for c in range(3):
        for m in range(2):
            for dy in range(3):
                for dx in range(3):
                    ref[..., c * 2 + m] += xp[:, dy : dy + 6, dx : dx + 6, c] * k[dy, dx, c, m]
            ref[..., c * 2 + m] += b[c * 2 + m]
    np.testing.assert_allclose(y, ref, rtol=1e-5, atol=1e-5)


def test_separable_ebops_is_sum_of_steps():
    """EBOPs of a separable conv are those of its depthwise step plus those of its pointwise step."""
    from hgq.config import LayerConfigScope

    with LayerConfigScope(enable_ebops=True, beta0=0.0):
        sep = QSeparableConv2D(4, 3, padding='same', use_bias=False)
        inp = keras.Input((6, 6, 3))
        model = keras.Model(inp, sep(inp))
    x = np.random.randn(8, 6, 6, 3).astype(np.float32)
    model(x, training=True)
    bw_in = sep.iq.bits_((1, 6, 6, 3))
    bw_dk = sep.dkq.bits_(ops.shape(sep.depthwise_kernel))
    bw_mid = sep.pq.bits_((1, 6, 6, 3))
    bw_pk = sep.pkq.bits_(ops.shape(sep.pointwise_kernel))
    dw = ops.sum(ops.depthwise_conv(bw_in, bw_dk, padding='same'))
    pw = ops.sum(ops.conv(bw_mid, bw_pk))
    np.testing.assert_allclose(float(sep.ebops), float(dw + pw), rtol=1e-5)
