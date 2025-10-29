from collections.abc import Sequence
from math import prod

from keras import ops
from keras.src import backend

from ..quantizer import QuantizerConfig
from .activation import QUnaryFunctionLUT
from .core import QLayerBaseSingleInput


class QSoftmax(QLayerBaseSingleInput):
    def __init__(
        self,
        axis: int | Sequence[int] | None = None,
        iq_conf: None | QuantizerConfig = None,
        stable=True,
        exp_iq_conf: None | QuantizerConfig = None,
        exp_oq_conf: None | QuantizerConfig = None,
        inv_iq_conf: None | QuantizerConfig = None,
        inv_oq_conf: None | QuantizerConfig = None,
        allow_heterogeneous_table: bool = False,
        input_scaler: float = 1.0,
        parallelization_factor: int = -1,
        **kwargs,
    ):
        # Keras h5 loader pops axis silent when it is a list longer than 1.
        axes = kwargs.pop('axes', None)
        assert axis is None or axes is None, 'Use only one of `axis` or `axes`.'
        self.axes = axes or (tuple(axis) if isinstance(axis, Sequence) else (axis,) if axis is not None else (-1,))

        self.supports_masking = True
        super().__init__(iq_conf=iq_conf, **kwargs)  # type: ignore
        self.stable = stable
        self.parallelization_factor = parallelization_factor

        self._allow_heterogeneous_table = allow_heterogeneous_table

        self.input_scaler = input_scaler

        def _inv(x):
            return 1.0 / (x + backend.epsilon())

        def _exp(x):
            if self.stable:
                return ops.exp(-x * self.input_scaler)
            else:
                return ops.exp(x * self.input_scaler)

        inv_iq_conf = inv_iq_conf or QuantizerConfig('default', 'datalane')
        exp_iq_conf = exp_iq_conf or QuantizerConfig('default', 'datalane')
        exp_oq_conf = exp_oq_conf or QuantizerConfig('default', 'table')
        inv_oq_conf = inv_oq_conf or QuantizerConfig('default', 'table')
        if not self._allow_heterogeneous_table:
            inv_iq_conf.config['heterogeneous_axis'] = ()
            inv_iq_conf.config['homogeneous_axis'] = None
            exp_iq_conf.config['heterogeneous_axis'] = ()
            exp_iq_conf.config['homogeneous_axis'] = None

        if 'k0' in inv_oq_conf.config:
            inv_oq_conf.config['k0'] = 0
        if 'k0' in exp_oq_conf.config:
            exp_oq_conf.config['k0'] = 0

        # hls4ml only supports the following configurations
        inv_oq_conf.config['overflow_mode'] = 'SAT'  # type: ignore
        inv_oq_conf.config['round_mode'] = 'RND_CONV'  # type: ignore
        exp_oq_conf.config['overflow_mode'] = 'SAT'  # type: ignore
        exp_oq_conf.config['round_mode'] = 'RND_CONV'  # type: ignore

        self.inv_table = QUnaryFunctionLUT(
            _inv,
            inv_iq_conf,
            inv_oq_conf,
            enable_iq=True,
            enable_oq=True,
            allow_heterogeneous_table=allow_heterogeneous_table,
            name=f'{self.name}_inv_table',
            enable_ebops=self.enable_ebops,
            beta0=self._beta0.clone(),
        )
        self.exp_table = QUnaryFunctionLUT(
            _exp,
            exp_iq_conf,
            exp_oq_conf,
            enable_iq=self.stable,
            enable_oq=True,
            allow_heterogeneous_table=allow_heterogeneous_table,
            allow_heterogeneous_input=True,
            name=f'{self.name}_exp_table',
            enable_ebops=self.enable_ebops and stable,
            beta0=self._beta0.clone(),
        )

    def build(self, input_shape):
        self.exp_table.build(input_shape)
        axis = sorted(i if i >= 0 else i + len(input_shape) for i in self.axes)  # type: ignore
        self.axes = tuple(axis)

        inv_shape = list(input_shape)
        for i in self.axes:
            inv_shape[i] = 1
        self.inv_table.build(tuple(inv_shape))

        super().build(input_shape)

    def call(self, inputs, training=None, mask=None):  # type: ignore
        if self.enable_iq:
            inputs = self.iq(inputs, training=training)

        if self.stable:
            inputs = ops.max(inputs, axis=self.axes, keepdims=True) - inputs

        exp_inp = self.exp_table(inputs, training=training)

        if mask is not None:
            exp_inp = backend.cast(mask, ops.dtype(inputs)) * exp_inp

        sums = ops.sum(exp_inp, axis=self.axes, keepdims=True)  # type: ignore
        divisor = self.inv_table(sums, training=training)

        return exp_inp * divisor

    def _compute_ebops(self, shape):
        accum_shape = tuple(1 if i in self.axes else s for i, s in enumerate(shape))
        max_instance = prod(accum_shape)
        n_instance = self.parallelization_factor if self.parallelization_factor > 0 else max_instance
        factor = n_instance / max_instance

        inp_bits = self.iq.bits_(shape)
        exp_bits = self.exp_table.oq.bits_(shape)
        inv_bits = self.inv_table.oq.bits_(accum_shape)

        if self.stable:
            substract_ebops = ops.sum(inp_bits)  # type: ignore # TODO: better ebops cost model for add and max
        else:
            substract_ebops = 0

        accum_ebops = ops.sum(exp_bits) - ops.sum(ops.min(exp_bits, axis=self.axes))  # type: ignore
        mult_ebops = ops.sum(exp_bits * inv_bits)  # type: ignore

        ebops = substract_ebops + accum_ebops + mult_ebops

        if not self.stable:
            # iq is disabled for exp table, compute here
            ebops += ops.sum((2.0**inp_bits) * exp_bits) * 1e-4  # type: ignore

        return ebops * factor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'axes': self.axes,
                'stable': self.stable,
                'exp_oq_conf': self.exp_table.oq.config,
                'exp_iq_conf': self.exp_table.iq.config if self.stable else None,
                'inv_oq_conf': self.inv_table.oq.config,
                'inv_iq_conf': self.inv_table.iq.config,
                'allow_heterogeneous_table': self._allow_heterogeneous_table,
                'input_scaler': self.input_scaler,
                'parallelization_factor': self.parallelization_factor,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

    @property
    def ebops(self):
        ebops = sum(
            (  # type: ignore
                ops.convert_to_tensor(self._ebops),
                self.exp_table.ebops,
                self.inv_table.ebops,
            )
        )
        return round(ops.convert_to_numpy(ebops).item())  # type: ignore
