"""Trivial subclassed handlers. Each class just adds HGQ's iq/oq/qkernel/qbias
logic (via ``QLayerMixin``) on top of an Alkaid scaffolding handler.
"""

import warnings

from alkaid.converter.builtin.keras.layers._base import ReplayOperationBase, to_np_arr
from alkaid.converter.builtin.keras.layers.batchnorm import ReplayBatchNormalization
from alkaid.converter.builtin.keras.layers.conv import ReplayConv
from alkaid.converter.builtin.keras.layers.dense import ReplayDense
from alkaid.converter.builtin.keras.layers.pool import ReplayPool
from alkaid.opsched.frontend import configure
from alkaid.trace import FVArray

import hgq
from hgq.layers import (
    QBatchNormalization,
    QBatchNormDense,
    QConv1D,
    QConv2D,
    QConv3D,
    QDense,
    QEinsumDense,
    QEinsumDenseBatchnorm,
    Quantizer,
)
from hgq.quantizer.internal import DefaultBitwidthMapper, FixedPointQuantizerBase
from hgq.utils.misc import expand_ellipsis

from ._base import QLayerMixin, mirror_quantizer


class ReplayQuantizer(ReplayOperationBase):
    handles = (Quantizer,)

    def __init__(self, op: 'Quantizer'):
        super().__init__(op)
        assert isinstance(op.quantizer, FixedPointQuantizerBase)

    def call(self, inputs: FVArray) -> FVArray:
        return mirror_quantizer(self.op, inputs)


def _strided(op) -> bool:
    mapper = op.kq.quantizer.bw_mapper
    return isinstance(mapper, DefaultBitwidthMapper) and mapper.strided


def _hinted(projected: FVArray, op, row: int, column: int) -> FVArray:
    """``projected`` hinted the MAC tile (d, c) that is the strided pblock of ``op``'s kernel quantizer: d row lanes along
    the kernel axis ``row``, c column lanes along ``column``. A pblock along any other, firing, axis only restricts the
    parallel firings along it, which alkaid takes no hint for yet."""
    pblock = op.kq.quantizer.bw_mapper.bw_shape
    return configure(projected, tiling=(pblock[row], pblock[column]), hint=True)


class _QDense(QLayerMixin, ReplayDense):
    handles = (QDense, QBatchNormDense, QEinsumDense, QEinsumDenseBatchnorm)

    def call(self, inputs: FVArray) -> FVArray:
        projected, op = super().call(inputs), self.op
        if not _strided(op):
            return projected
        if isinstance(op, QEinsumDense):
            inp, ker, out = expand_ellipsis(op.equation, (len(inputs.shape), len(op.kernel.shape))).replace('->', ',').split(',')
            rows, columns = [a for a in inp if a in ker and a not in out], [a for a in out if a in ker and a not in inp]
            if rows != [inp[-1]] or len(columns) != 1:
                warnings.warn(f'{op.name}: {op.equation} is no (last input axis, one output axis) MAC tile; no tiling hint')
                return projected
            return _hinted(projected, op, ker.index(rows[0]), ker.index(columns[0]))
        return _hinted(projected, op, 0, 1)


class _QConv(QLayerMixin, ReplayConv):
    handles = (QConv1D, QConv2D, QConv3D)


class _QBatchNormalization(QLayerMixin, ReplayBatchNormalization):
    handles = (QBatchNormalization,)

    def fused_scale_offset(self):
        op: QBatchNormalization = self.op  # type: ignore
        qs, qo = op.qscaler_and_qoffset
        return to_np_arr(qs), to_np_arr(qo)


class _QPool(QLayerMixin, ReplayPool):
    handles = (
        hgq.layers.QAvgPool1D,
        hgq.layers.QAvgPool2D,
        hgq.layers.QAvgPool3D,
        hgq.layers.QMaxPool1D,
        hgq.layers.QMaxPool2D,
        hgq.layers.QMaxPool3D,
        hgq.layers.QGlobalAveragePooling1D,
        hgq.layers.QGlobalAveragePooling2D,
        hgq.layers.QGlobalAveragePooling3D,
        hgq.layers.QGlobalMaxPooling1D,
        hgq.layers.QGlobalMaxPooling2D,
        hgq.layers.QGlobalMaxPooling3D,
    )
