"""Trivial subclassed handlers. Each class just adds HGQ's iq/oq/qkernel/qbias
logic (via ``QLayerMixin``) on top of an Alkaid scaffolding handler.
"""

from alkaid.converter.builtin.keras.layers._base import to_np_arr
from alkaid.converter.builtin.keras.layers.batchnorm import ReplayBatchNormalization
from alkaid.converter.builtin.keras.layers.conv import ReplayConv
from alkaid.converter.builtin.keras.layers.dense import ReplayDense
from alkaid.converter.builtin.keras.layers.pool import ReplayPool

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
)

from ._base import QLayerMixin


class _QDense(QLayerMixin, ReplayDense):
    handles = (QDense, QBatchNormDense, QEinsumDense, QEinsumDenseBatchnorm)


class _QConv(QLayerMixin, ReplayConv):
    handles = (QConv1D, QConv2D, QConv3D)


class _QBatchNormalization(QLayerMixin, ReplayBatchNormalization):
    handles = (QBatchNormalization,)

    def fused_scale_offset(self):
        qs, qo = self.op.qscaler_and_qoffset
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
