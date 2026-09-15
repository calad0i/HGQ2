from collections.abc import Callable
from copy import copy
from math import prod
from typing import Literal, cast

import keras
import numpy as np
from keras import ops

from ..quantizer import Quantizer, QuantizerConfig
from .activation import table_ebops
from .core import QLayerBase, QLayerBaseSingleInput
from .softmax import QSoftmax


def scan_rounds(fn: Callable, init, xs, variables):
    """Roughly equivalent to
    ```python
    def scan_rounds(fn, init, xs):
        carry = init
        for x in zip(*xs):
            carry = fn(carry, x)
        return carry
    ```
    compiled as one traced round.
    """

    def traced(carry, x):
        carry, values = carry
        with keras.StatelessScope(zip(variables, values)) as scope:
            carry = fn(carry, x)
        return (carry, tuple(scope.get_current_value(variable) for variable in variables)), None

    carry = (init, tuple(variable.value for variable in variables))
    if keras.backend.backend() != 'tensorflow':
        carry, _ = ops.scan(traced, carry, xs)  # type: ignore
    else:
        for i in range(int(xs[0].shape[0])):
            carry, _ = traced(carry, tuple(x[i] for x in xs))
    carry, values = carry
    for variable, value in zip(variables, values):
        variable.assign(value)
    return carry


class QFSoftmax(QSoftmax):
    ebops = QLayerBase.ebops  # type: ignore

    def __init__(
        self,
        axis: int = -1,
        iq_conf: None | QuantizerConfig = None,
        lq_conf: None | QuantizerConfig = None,
        aq_conf: None | QuantizerConfig = None,
        stable: bool = True,
        parallelization_factor: int = -1,
        accumulator_shape: tuple[int | None, ...] | None = None,
        impl: Literal['1pass', '2pass'] = '1pass',
        **kwargs,
    ):
        assert isinstance(axis, int), f'QFSoftmax scans exactly one axis, got axis={axis}.'
        assert 'axes' not in kwargs, f'QFSoftmax scans exactly one axis, stated as `axis`; got axes={kwargs["axes"]}.'
        assert stable, 'QFSoftmax carries the running max through the scan, which is always the stable form'

        self.impl = impl
        self._configured_axis: int = axis
        lane_axes = (axis,) if accumulator_shape is None else (-1, len(accumulator_shape) - 1)
        head_axes = () if accumulator_shape is None else (1, 1 - len(accumulator_shape))
        kwargs.update(iq_conf=iq_conf, lq_conf=lq_conf, aq_conf=aq_conf)
        for name in ('iq_conf', 'lq_conf', 'aq_conf', 'exp_iq_conf', 'exp_oq_conf', 'inv_iq_conf', 'inv_oq_conf'):
            conf = copy(kwargs.get(name) or QuantizerConfig('default', 'table' if name.endswith('oq_conf') else 'datalane'))
            conf.config = conf.config.copy()
            conf.config.update(homogeneous_axis=None, bw_mapper=None)
            conf.config['heterogeneous_axis'] = tuple(
                a for a in conf.config.get('heterogeneous_axis') or () if a in head_axes or name == 'aq_conf' and a in lane_axes
            )
            kwargs[name] = conf
        lq_conf = cast(QuantizerConfig, kwargs.pop('lq_conf'))
        aq_conf = cast(QuantizerConfig, kwargs.pop('aq_conf'))
        super().__init__(axis=axis, stable=stable, parallelization_factor=parallelization_factor, **kwargs)

        self.lq = Quantizer(lq_conf, name=f'{self.name}_lq')
        self.aq = Quantizer(aq_conf, name=f'{self.name}_aq')
        self.accumulator_shape = accumulator_shape

        self.exp_table._enable_ebops = False
        self.inv_table._enable_ebops = False
        self.supports_masking = False

    @property
    def impl(self) -> Literal['1pass', '2pass']:
        return self._impl

    @impl.setter
    def impl(self, value: Literal['1pass', '2pass']):
        if value not in ('1pass', '2pass'):
            raise ValueError("impl must be '1pass' or '2pass'.")
        self._impl: Literal['1pass', '2pass'] = value

    @property
    def axis(self) -> int:
        return self.axes[0]

    def build(self, input_shape):
        axis = self.axis if self.axis >= 0 else self.axis + len(input_shape)
        self.axes = (axis,)
        n = input_shape[axis]
        assert n is not None, f'QFSoftmax needs a known length on the scan axis {axis}, got {input_shape}.'

        state_shape = tuple(1 if i == axis else s for i, s in enumerate(input_shape))
        accumulator_shape = self.accumulator_shape if self.accumulator_shape is not None else input_shape

        slot_shape = tuple(n if i == axis else 1 for i in range(len(input_shape)))
        self._lanes = np.arange(n).reshape(slot_shape)

        self.exp_table.build(state_shape)
        self.inv_table.build(state_shape)
        self.lq.build(state_shape)
        self.aq.build(accumulator_shape)

        self.n_parallel = prod(state_shape[1:])  # the rows the scan leaves, one instance of the layer each
        if self.parallelization_factor < 0:
            self.parallelization_factor = self.n_parallel

        # QSoftmax shapes its exp table over the whole input; the scan reads one round of it at a time.
        QLayerBaseSingleInput.build(self, input_shape)

    def call(self, inputs, training=None):  # type: ignore
        if self.enable_iq:
            inputs = self.iq(inputs, training=training)

        def statistics(m, l, s):
            s = ops.expand_dims(s, self.axis)
            m, m_prev = ops.maximum(m, s), m
            dr, dp = m - m_prev, m - s  # type: ignore
            r = self.exp_table(dr, training=training)  # EXP[m_prev - m], 1 on a quiet round
            p = self.exp_table(dp, training=training)
            return m, self.lq(r * l, training=training) + p, r, p

        def round_(carry, xs):
            m, l, o = carry
            s, i = xs
            m, l, r, p = statistics(m, l, s)
            o = ops.where(
                ops.equal(self._lanes, i),
                self.aq(ops.broadcast_to(p, ops.shape(o)), training=training),
                self.aq(r * o, training=training),
            )
            return m, l, o

        m = ops.take(inputs, [0], axis=self.axis)
        if self.impl == '2pass':

            def stats(carry, xs):
                return statistics(carry[0], carry[1], xs[0])[:2]

            m, l = scan_rounds(
                stats,
                (m, ops.zeros_like(m)),
                (ops.moveaxis(inputs, self.axis, 0),),
                [*self.exp_table.variables, *self.lq.variables],
            )
            return self.exp_table(m - inputs, training=training) * self.inv_table(l, training=training)

        init = (m, ops.zeros_like(m), ops.zeros_like(inputs))
        # The scan axis leads, so the round above is traced once and stands for all n of them.
        xs = (ops.moveaxis(inputs, self.axis, 0), ops.arange(self._lanes.size))
        _, l, o = scan_rounds(round_, init, xs, [*self.exp_table.variables, *self.lq.variables, *self.aq.variables])

        return o * self.inv_table(l, training=training)

    def _compute_ebops(self, shape):
        state_shape = tuple(1 if i == self.axis else s for i, s in enumerate(shape))
        n = shape[self.axis]

        factor = self.parallelization_factor / self.n_parallel
        inp_bits = self.iq.bits_(shape) if self.enable_iq else self.exp_table.iq.bits_(shape)
        exp_in_bits = self.exp_table.iq.bits_(state_shape)
        exp_bits = self.exp_table.oq.bits_(state_shape)
        l_bits = self.lq.bits_(state_shape)
        inv_in_bits = self.inv_table.iq.bits_(state_shape)
        inv_bits = self.inv_table.oq.bits_(state_shape)

        if self.impl == '2pass':
            per_key = (
                3 * table_ebops(exp_in_bits, exp_bits)  # type: ignore
                + ops.sum(exp_bits * l_bits)  # type: ignore
                + ops.sum(l_bits)  # type: ignore
                + table_ebops(inv_in_bits, inv_bits)  # type: ignore
                + ops.sum(exp_bits * inv_bits)  # type: ignore
            )
            return (4 * ops.sum(inp_bits) + n * per_key) * factor  # type: ignore

        acc_bits = self.aq.bits_(shape)
        round_ebops = (
            2 * table_ebops(exp_in_bits, exp_bits)  # type: ignore
            + ops.sum(exp_bits * l_bits)  # type: ignore
            + ops.sum(l_bits)
            + ops.sum(exp_bits * acc_bits)  # type: ignore
        )

        # Once per row
        final_ebops = table_ebops(inv_in_bits, inv_bits) + ops.sum(acc_bits * inv_bits)  # type: ignore

        # max and the two differences happen once per lane.
        return (3 * ops.sum(inp_bits) + n * round_ebops + final_ebops) * factor  # type: ignore

    def get_config(self):
        config = super().get_config()
        del config['axes']
        config.update(
            {
                'axis': self._configured_axis,
                'lq_conf': self.lq.config,
                'aq_conf': self.aq.config,
                'accumulator_shape': self.accumulator_shape,
                'impl': self.impl,
            }
        )
        return config
