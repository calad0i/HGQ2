from math import log2, prod

import keras
import numpy as np
import pytest
from keras import layers, ops

from hgq.config import QuantizerConfigScope
from hgq.layers import QLayerBase, Quantizer
from hgq.layers.ops import QAdd, QAveragePow2, QDot, QEinsum, QMaximum, QMeanPow2, QMinimum, QMultiply, QSubtract, QSum
from hgq.utils import trace_minmax

from .base import LayerTestBase, masked_call


class TestQuantizer(LayerTestBase):
    layer_cls = Quantizer
    hls4ml_not_supported = True  # supported, but can't convert single quantizer

    @pytest.fixture(params=[(12,)])
    def input_shapes(self, request):
        return request.param

    @pytest.mark.parametrize('shape', [(64, 6, 2, 8), (64, 7, 2, 9)])
    @pytest.mark.parametrize(
        'pblock',
        [
            (1, 3, -1, 2),
            (3, -1, 2),
            (-1, 2),
            {1: 3, -1: 2},
            ((1, 3), (-2, -1), (-1, 2)),
        ],
        ids=['full', 'last3', 'last2', 'dict', 'pairs'],
    )
    @pytest.mark.parametrize('training', [False, True])
    def test_tiled_bw_mapper(self, q_type: str, overflow_mode: str, round_mode: str, training: bool, shape: tuple, pblock):
        """A tiled quantizer equals a per-element one whose widths are the pblock's np.tile expansion, trimmed to the input.
        The pblock is one p per axis over the last axes, or sparse (axis, p) pairs or a dict; axes not given share one."""
        rng = np.random.default_rng(0)
        if isinstance(pblock, dict):
            pairs = pblock.items()
        elif all(isinstance(p, int) for p in pblock):
            pairs = zip(range(-len(pblock), 0), pblock)
        else:
            pairs = pblock
        p_of = {a % len(shape): p for a, p in pairs}
        bw_shape = tuple(n if p_of.get(a, 1) == -1 else min(p_of.get(a, 1), n) for a, n in enumerate(shape))
        x = 0.5 * rng.standard_normal(shape).astype(np.float32)
        x[0] = 2 * _tile(_shrink(np.abs(x), bw_shape, np.max), (1, *shape[1:]))[0]  # one absmax per width
        x[1] = -x[0]  # and a negative value everywhere
        conf = dict(q_type=q_type, place='datalane', overflow_mode=overflow_mode, round_mode=round_mode)
        q = Quantizer(**conf, pblock=pblock)
        q_ref = Quantizer(**conf, homogeneous_axis=(0,))
        q.build(shape)
        q_ref.build(shape)
        upper_bound = {'k': 2, 'i': 3, 'f': 6, 'b': 9}
        widths = {n: rng.integers(n != 'k', upper_bound[n], bw_shape) for n in upper_bound}
        widths['i'] = widths['i'] - 1
        for v, v_ref in zip(q.weights, q_ref.weights):
            if v.name in widths:
                v.assign(widths[v.name])
                v_ref.assign(_tile(widths[v.name], v_ref.shape))

        y, gx, gw = _value_and_grads(q, x, training)
        y_ref, gx_ref, gw_ref = _value_and_grads(q_ref, x, training)
        np.testing.assert_array_equal(y, y_ref)
        np.testing.assert_allclose(gx, gx_ref, rtol=1e-6)
        assert len(gw) == len(gw_ref) > 0
        for g, g_ref in zip(gw, gw_ref):
            np.testing.assert_allclose(g, _shrink(g_ref, bw_shape, np.sum), rtol=1e-5, atol=1e-5)  # fp32 sums of ~1e3 terms

        y = trace_minmax(keras.Sequential([keras.Input(shape[1:]), q]), x, return_results=True)
        y_ref = trace_minmax(keras.Sequential([keras.Input(shape[1:]), q_ref]), x, return_results=True)
        np.testing.assert_array_equal(y, y_ref)
        for v, v_ref in zip(q.weights, q_ref.weights):
            if v.name in widths:
                np.testing.assert_array_equal(_tile(ops.convert_to_numpy(v), v_ref.shape), ops.convert_to_numpy(v_ref))
        np.testing.assert_array_equal(
            ops.convert_to_numpy(ops.stop_gradient(q(x))), ops.convert_to_numpy(ops.stop_gradient(q_ref(x)))
        )


def _tile(a, shape):
    """``a`` np.tile-d over ``shape``, dropping last elements to match exact ``shape``."""
    return np.tile(a, [-(-s // n) for s, n in zip(shape, a.shape)])[tuple(map(slice, shape))]


def _shrink(a, shape, fn):
    """inverse layout of _tile - ``fn`` over the elements sharing each width of ``shape``, zeros padding the trimmed tiles."""
    a = np.pad(a, [(0, -s % n) for s, n in zip(a.shape, shape)])
    for axis, n in enumerate(shape):
        a = fn(a.reshape(a.shape[:axis] + (a.shape[axis] // n, n) + a.shape[axis + 1 :]), axis=axis)
    return a


def _value_and_grads(q, x, training) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """q(x), and the gradients of sum(q(x) * c) to x and to q's trainable variables."""
    c = ops.convert_to_tensor(np.random.default_rng(1).standard_normal(x.shape).astype(np.float32))
    match keras.backend.backend():
        case 'jax':
            import jax

            ntv = [v.value for v in q.non_trainable_variables]

            def loss(x, tv):
                y, ntv_new = q.stateless_call(tv, ntv, x, training=training)
                return ops.sum(y * c), (y, ntv_new)

            (_, (y, ntv_new)), (gx, gw) = jax.value_and_grad(loss, (0, 1), has_aux=True)(
                x, [v.value for v in q.trainable_variables]
            )
            for v, n in zip(q.non_trainable_variables, ntv_new):
                v.assign(n)
        case 'tensorflow':
            import tensorflow as tf

            x = tf.convert_to_tensor(x)
            with tf.GradientTape() as tape:
                tape.watch(x)
                y = q(x, training=training)
                l = ops.sum(y * c)
            gx, *gw = tape.gradient(l, [x, *q.trainable_variables])  # type: ignore
        case 'torch':
            import torch

            x = torch.tensor(x, requires_grad=True)
            y = q(x, training=training)
            gx, *gw = torch.autograd.grad(ops.sum(y * c), [x, *(v.value for v in q.trainable_variables)])  # type: ignore
            y = y.detach()
    return ops.convert_to_numpy(y), ops.convert_to_numpy(gx), [ops.convert_to_numpy(g) for g in gw]  # type: ignore


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

    def test_mask_passthrough(self, layer, input_data):
        masks, out_mask = masked_call(layer, input_data)
        xs, dirty = [x[: len(masks[0])] for x in input_data], np.array(False)
        for i, m in enumerate(masks):
            feed = [np.ones_like(x) for x in xs]
            feed[i] = np.broadcast_to(np.expand_dims(~m, tuple(range(2, xs[i].ndim))), xs[i].shape).astype('float32')
            dirty = dirty | (ops.convert_to_numpy(layer(feed)) != 0)
        np.testing.assert_array_equal(out_mask, ~dirty.any(tuple(range(out_mask.ndim, dirty.ndim))))

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


class TestQSum(LayerTestBase):
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


class TestMeanPow2(TestQSum):
    layer_cls = QMeanPow2
    hls4ml_not_supported = True

    @pytest.fixture
    def scale(self, input_shapes: tuple[int, ...], axes):
        if isinstance(axes, int):
            scale = 1.0 / input_shapes[axes % (len(input_shapes) + 1) - 1]
        else:
            scale = 1.0 / prod(input_shapes[ax % (len(input_shapes) + 1) - 1] for ax in axes)
        return 2.0 ** round(log2(scale))
