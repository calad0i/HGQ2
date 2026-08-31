import numpy as np
from keras import ops

from ..quantizer import Quantizer, QuantizerConfig
from .core import QLayerBase, QLayerBaseSingleInput
from .softmax import QSoftmax


class QFSoftmax(QSoftmax):
    """Online (flash-attention style) softmax matching alkaid's FA impl."""

    ebops = QLayerBase.ebops  # type: ignore

    def __init__(
        self,
        axis: int = -1,
        iq_conf: None | QuantizerConfig = None,
        lq_conf: None | QuantizerConfig = None,
        aq_conf: None | QuantizerConfig = None,
        stable: bool = True,
        parallelization_factor: int = -1,
        **kwargs,
    ):
        assert isinstance(axis, int), f'QFSoftmax scans exactly one axis, got axis={axis}.'
        assert 'axes' not in kwargs, f'QFSoftmax scans exactly one axis, stated as `axis`; got axes={kwargs["axes"]}.'
        assert stable, 'QFSoftmax carries the running max through the scan, which is always the stable form'

        # TEnforcing homogeneouity config
        iq_conf = iq_conf or QuantizerConfig('default', 'datalane')
        aq_conf = aq_conf or QuantizerConfig('default', 'datalane')
        iq_conf.config['homogeneous_axis'] = (0, axis)  # type: ignore
        iq_conf.config['heterogeneous_axis'] = None  # type: ignore
        aq_conf.config['homogeneous_axis'] = (0, axis)  # type: ignore
        aq_conf.config['heterogeneous_axis'] = None  # type: ignore

        super().__init__(axis=axis, iq_conf=iq_conf, stable=stable, parallelization_factor=parallelization_factor, **kwargs)

        lq_conf = lq_conf or QuantizerConfig('default', 'datalane')

        self.lq = Quantizer(lq_conf, name=f'{self.name}_lq')
        self.aq = Quantizer(aq_conf, name=f'{self.name}_aq')

        self.exp_table._enable_ebops = False
        self.inv_table._enable_ebops = False
        self.supports_masking = False

    @property
    def axis(self) -> int:
        return self.axes[0]

    def build(self, input_shape):
        axis = self.axis if self.axis >= 0 else self.axis + len(input_shape)
        self.axes = (axis,)
        n = input_shape[axis]
        assert n is not None, f'QFSoftmax needs a known length on the scan axis {axis}, got {input_shape}.'

        state_shape = tuple(1 if i == axis else s for i, s in enumerate(input_shape))
        slot_shape = tuple(n if i == axis else 1 for i in range(len(input_shape)))
        self._slots = [np.arange(n).reshape(slot_shape) == i for i in range(n)]

        self.exp_table.build(state_shape)
        self.inv_table.build(state_shape)
        self.lq.build(state_shape)
        self.aq.build(input_shape)

        # QSoftmax shapes its exp table over the whole input; the scan reads one round of it at a time.
        QLayerBaseSingleInput.build(self, input_shape)

    def call(self, inputs):  # type: ignore
        if self.enable_iq:
            inputs = self.iq(inputs)

        m = ops.take(inputs, [0], axis=self.axis)  # the first score, so every round below is identical
        l = ops.zeros_like(m)
        o = ops.zeros_like(inputs)

        for i, slot in enumerate(self._slots):
            s = ops.take(inputs, [i], axis=self.axis)
            m, m_prev = ops.maximum(m, s), m
            dr, dp = m - m_prev, m - s  # type: ignore
            r = self.exp_table(dr)  # EXP[m_prev - m], 1 on a quiet round
            p = self.exp_table(dp)
            l = self.lq(r * l) + p
            o = ops.where(slot, self.aq(p), self.aq(r * o))

        return o * self.inv_table(l)

    def _compute_ebops(self, shape):
        state_shape = tuple(1 if i == self.axis else s for i, s in enumerate(shape))
        n = shape[self.axis]

        inp_bits = self.iq.bits_(shape) if self.enable_iq else self.exp_table.iq.bits_(shape)
        exp_in_bits = self.exp_table.iq.bits_(state_shape)
        exp_bits = self.exp_table.oq.bits_(state_shape)
        l_bits = self.lq.bits_(state_shape)
        acc_bits = self.aq.bits_(shape)  # one grid, spread over the n lanes it rescales
        inv_in_bits = self.inv_table.iq.bits_(state_shape)
        inv_bits = self.inv_table.oq.bits_(state_shape)

        round_ebops = (
            2 * ops.sum((2.0**exp_in_bits) * exp_bits) * 1e-4  # type: ignore
            + ops.sum(exp_bits * l_bits)  # type: ignore
            + ops.sum(l_bits)
            + ops.sum(exp_bits * acc_bits)  # type: ignore
        )

        # Once per row
        final_ebops = ops.sum((2.0**inv_in_bits) * inv_bits) * 1e-4 + ops.sum(acc_bits * inv_bits)  # type: ignore

        # max and the two differences happen once per lane.
        return 3 * ops.sum(inp_bits) + n * round_ebops + final_ebops  # type: ignore

    def get_config(self):
        config = super().get_config()
        del config['axes']
        config.update({'axis': self.axis, 'lq_conf': self.lq.config, 'aq_conf': self.aq.config})
        return config
