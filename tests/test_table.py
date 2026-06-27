import keras
import numpy as np
import pytest

from hgq.config import LayerConfigScope, QuantizerConfigScope
from hgq.layers import QConvT1D, QConvT2D, QDenseT, QEinsumDenseT

from .base import CtxGlue, LayerTestBase


class TableTestBase(LayerTestBase):
    abs_cap_multiplier = 8.0
    max_lsb_drift_fraction = 5e-4

    @pytest.fixture(params=[True])
    def use_parallel_io(self, request) -> bool:
        return request.param

    @pytest.fixture(params=[True, False])
    def batch_norm(self, request) -> bool:
        return request.param

    @pytest.fixture
    def ctx_scope(self, q_type: str, overflow_mode: str, round_mode: str):
        scope_t = QuantizerConfigScope(place='table', homogeneous_axis=(0,), overflow_mode='SAT')
        scope_w = QuantizerConfigScope(
            default_q_type=q_type,
            heterogeneous_axis=None,
            homogeneous_axis=(),
            overflow_mode=overflow_mode,
            round_mode=round_mode,
            br=None,
            ir=None,
            fr=None,
        )
        scope_a = QuantizerConfigScope(
            default_q_type=q_type, place='datalane', overflow_mode=overflow_mode, round_mode=round_mode, homogeneous_axis=(0,)
        )
        scope_l = LayerConfigScope(beta0=0.0, enable_ebops=True)
        return CtxGlue(scope_w, scope_a, scope_t, scope_l)


class TestDenseT(TableTestBase):
    layer_cls = QDenseT

    @pytest.fixture(params=[8, 12])  # Test different output sizes
    def n_out(self, request):
        return request.param

    @pytest.fixture(params=[None])  # Test with and without activation
    def activation(self, request):
        return request.param

    @pytest.fixture(params=[(8, 8), (12,)])
    def input_shapes(self, request):
        return request.param

    @pytest.fixture
    def layer_kwargs(self, n_out, activation, batch_norm):
        return {'n_out': n_out, 'activation': activation, 'batch_norm': batch_norm}


class TestEinsumDenseT(TableTestBase):
    layer_cls = QEinsumDenseT
    hls4ml_not_supported = True

    @pytest.fixture(params=[False, True])
    def batch_norm(self, request) -> bool:
        return request.param

    @pytest.fixture(
        params=[
            ('Bij,ijk->Bik', (3, 4), (3, 6), 'ik'),
            ('Babc,bcd->Babd', (3, 4, 5), (3, 4, 2), 'bd'),
        ]
    )
    def einsum_case(self, request):
        return request.param

    @pytest.fixture
    def input_shapes(self, einsum_case):
        return einsum_case[1]

    @pytest.fixture
    def layer_kwargs(self, einsum_case, batch_norm):
        equation, _, output_shape, bias_axes = einsum_case
        return {'equation': equation, 'output_shape': output_shape, 'bias_axes': bias_axes, 'n_hl': 0, 'batch_norm': batch_norm}


class TestConvT1D(TableTestBase):
    layer_cls = QConvT1D

    @pytest.fixture(params=[6])
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
    def layer_kwargs(self, ch_out, conv_params, batch_norm):
        return {'filters': ch_out, 'batch_norm': batch_norm, **conv_params}


class TestConvT2D(TableTestBase):
    layer_cls = QConvT2D

    @pytest.fixture(params=[6])
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
        return (4, 5, 2)

    @pytest.fixture(params=['valid', 'same'])
    def padding(self, request):
        return request.param

    @pytest.fixture
    def layer_kwargs(self, ch_out, conv_params, batch_norm):
        return {'filters': ch_out, 'batch_norm': batch_norm, **conv_params}

    def test_training(self, model: keras.Model, input_data: np.ndarray, ch_out: int):
        if keras.backend.backend() == 'torch' and ch_out == 1:
            pytest.skip('Torch runtime error for unknown reason when ch_out is 1.')
        return super().test_training(model, input_data, 'SAT')
