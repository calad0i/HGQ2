import keras
import pytest
from keras import ops

from hgq.config import QuantizerConfigScope
from hgq.layers import QBatchNormDense, QDense, QEinsumDense, QEinsumDenseBatchnorm
from tests.base import LayerTestBase


class TestDense(LayerTestBase):
    da4ml_not_supported = False
    layer_cls = QDense
    keras_layer_cls = keras.layers.Dense

    @pytest.fixture(params=[8])  # Test different output sizes
    def units(self, request):
        return request.param

    @pytest.fixture(params=[None, 'relu'])  # Test with and without activation
    def activation(self, request):
        return request.param

    @pytest.fixture(params=[(8, 8), (12,)])
    def input_shapes(self, request):
        return request.param

    @pytest.fixture
    def layer_kwargs(self, units, activation):
        return {'units': units, 'activation': activation}

    def test_da4ml_conversion(self, model: keras.Model, input_data, overflow_mode: str, temp_directory: str):
        super()._test_da4ml_conversion(
            model=model,
            input_data=input_data,
            overflow_mode=overflow_mode,
            temp_directory=temp_directory,
        )

    def test_behavior(self, input_data, layer_kwargs):
        with QuantizerConfigScope(default_q_type='dummy'):
            hgq_layer = self.layer_cls(**layer_kwargs, enable_ebops=False)
            keras_layer = self.keras_layer_cls(**layer_kwargs)

        hgq_layer.build(input_data.shape)
        keras_layer.build(input_data.shape)

        keras_layer._kernel.assign(hgq_layer.qkernel)
        if hgq_layer.use_bias:
            keras_layer.bias.assign(hgq_layer.qbias)  # type: ignore

        r_hgq = hgq_layer(input_data)
        r_keras = keras_layer(input_data)

        assert ops.all(r_hgq == r_keras), f'{r_hgq} != {r_keras}'


class TestBatchNormDense(TestDense):
    layer_cls = QBatchNormDense


class TestEinsumDense(LayerTestBase):
    layer_cls = QEinsumDense

    @pytest.fixture(params=['Bij,ijk->Bik', 'Babc,bcd->Babd'])
    def equation(self, request):
        return request.param

    @pytest.fixture
    def input_shapes(self, equation: str):
        if equation == 'Bij,ijk->Bik':
            return (3, 4)
        return (3, 4, 5)

    @pytest.fixture
    def output_shape(self, equation: str):
        if equation == 'Bij,ijk->Bik':
            return (3, 6)
        return (3, 4, 2)

    @pytest.fixture(params=[False, True])
    def bias_axes(self, equation: str, request) -> str | None:
        if not request.param:
            return None
        if equation == 'Bij,ijk->Bik':
            return 'ik'
        return 'ad'

    @pytest.fixture
    def layer_kwargs(self, equation: str, output_shape: tuple[int, ...], bias_axes: str | None):
        return {'equation': equation, 'output_shape': output_shape, 'bias_axes': bias_axes, 'activation': None}

    @pytest.fixture(params=[True])
    def use_parallel_io(self, request) -> bool:
        return request.param

    def test_da4ml_conversion(self, model: keras.Model, input_data, overflow_mode: str, temp_directory: str):
        super()._test_da4ml_conversion(
            model=model,
            input_data=input_data,
            overflow_mode=overflow_mode,
            temp_directory=temp_directory,
        )


class TestEinsumDenseBatchnorm(TestEinsumDense):
    layer_cls = QEinsumDenseBatchnorm

    @pytest.fixture()
    def bias_axes(self, equation: str, request):
        if equation == 'Bij,ijk->Bik':
            return 'ik'
        return 'bd'
