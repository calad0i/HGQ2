import keras
import numpy as np
import pytest
from keras import ops

from hgq.config import QuantizerConfigScope
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

    def assert_equal(self, keras_output, hw_output, lsb_step=None, ignore_err=0.0):
        if keras.backend.backend() == 'torch':
            # Torch conv operator introduces some extra numerical error
            lsb_step = lsb_step + 1e-4 if lsb_step is not None else 5e-4
        return super().assert_equal(keras_output, hw_output, lsb_step, ignore_err)


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

    def assert_equal(self, keras_output, hw_output, lsb_step=None, ignore_err=0.0):
        if keras.backend.backend() == 'torch':
            # Torch conv operator introduces some extra numerical error
            lsb_step = lsb_step + 1e-4 if lsb_step is not None else 5e-4
        return super().assert_equal(keras_output, hw_output, lsb_step, ignore_err)

    def test_training(self, model: keras.Model, input_data: np.ndarray, overflow_mode, ch_out: int):
        if keras.backend.backend() == 'torch' and ch_out == 1:
            pytest.skip('Torch runtime error for unknown reason when ch_out is 1.')
        return super().test_training(model, input_data, overflow_mode)


class TestConv1DNoIQ:
    """Regression: QConv1D.call must guard self.iq when enable_iq=False.

    QConv1D.call overrode QBaseConv.call but read self.iq unconditionally, so
    QConv1D(enable_iq=False) built fine yet crashed at forward (the iq property
    raises when input quantization is disabled). QConv2D/QConv3D inherit the
    guarded base call and were unaffected.
    """

    @pytest.mark.parametrize('kwargs', [
        {'padding': 'causal', 'groups': 4},
        {'padding': 'valid'},
        {'padding': 'same'},
    ])
    def test_build_and_forward(self, kwargs):
        with QuantizerConfigScope(default_q_type='dummy'):
            x = keras.Input((16, 4))
            model = keras.Model(x, QConv1D(4, 3, enable_iq=False, **kwargs)(x))
        xin = np.random.default_rng(0).standard_normal((2, 16, 4)).astype('float32')
        y = ops.convert_to_numpy(model(xin, training=False))
        assert y.shape[0] == 2 and y.shape[-1] == 4

    @pytest.mark.parametrize('fmt', ['keras', 'h5'])
    def test_roundtrip(self, fmt, tmp_path):
        with QuantizerConfigScope(default_q_type='dummy'):
            x = keras.Input((16, 4))
            model = keras.Model(x, QConv1D(4, 3, padding='causal', groups=4, enable_iq=False)(x))
        xin = np.random.default_rng(0).standard_normal((2, 16, 4)).astype('float32')
        ref = ops.convert_to_numpy(model(xin, training=False))
        path = f'{tmp_path}/m.{fmt}'
        model.save(path)
        loaded = keras.models.load_model(path)
        np.testing.assert_array_equal(ref, ops.convert_to_numpy(loaded(xin, training=False)))
