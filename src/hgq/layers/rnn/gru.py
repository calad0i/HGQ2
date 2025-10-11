from keras import ops
from keras.layers import GRU, GRUCell
from keras.saving import register_keras_serializable
from keras.src import tree
from keras.src.layers.input_spec import InputSpec

from ...config import HardSigmoidConfig, HardTanhConfig, QuantizerConfig
from ...quantizer import Quantizer
from ..core.base import QLayerBase
from .simple_rnn import QRNN


@register_keras_serializable(package='hgq')
class QGRUCell(QLayerBase, GRUCell):
    """Cell class for the GRU layer.

    This class processes one step within the whole time sequence input, whereas
    `keras.layer.GRU` processes the whole sequence.

    Parameters
    ----------
    units : int
        Positive integer, dimensionality of the output space.
    activation : str, optional
        Activation function to use. Default: hyperbolic tangent
        (`tanh`). If you pass None, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    recurrent_activation : str, optional
        Activation function to use for the recurrent step.
        Default: sigmoid (`sigmoid`). If you pass `None`, no activation is
        applied (ie. "linear" activation: `a(x) = x`).
    use_bias : bool, optional
        Whether the layer should use a bias vector. Default: True.
    kernel_initializer : str, optional
        Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs. Default: "glorot_uniform".
    recurrent_initializer : str, optional
        Initializer for the `recurrent_kernel` weights matrix, used for the linear transformation
        of the recurrent state. Default: "orthogonal".
    bias_initializer : str, optional
        Initializer for the bias vector. Default: "zeros".
    kernel_regularizer : optional
        Regularizer function applied to the `kernel` weights matrix. Default: None.
    recurrent_regularizer : optional
        Regularizer function applied to the `recurrent_kernel` weights matrix. Default: None.
    bias_regularizer : optional
        Regularizer function applied to the bias vector. Default: None.
    kernel_constraint : optional
        Constraint function applied to the `kernel` weights matrix. Default: None.
    recurrent_constraint : optional
        Constraint function applied to the `recurrent_kernel` weights matrix. Default: None.
    bias_constraint : optional
        Constraint function applied to the bias vector. Default: None.
    dropout : float, optional
        Float between 0 and 1. Fraction of the units to drop for the
        linear transformation of the inputs. Default: 0.
    recurrent_dropout : float, optional
        Float between 0 and 1. Fraction of the units to drop for the
        linear transformation of the recurrent state. Default: 0.
    reset_after : bool, optional
        GRU convention (whether to apply reset gate after or
        before matrix multiplication). False = "before",
        True = "after" (default and cuDNN compatible).
    seed : int, optional
        Random seed for dropout.
    iq_conf : QuantizerConfig or None, optional
        Quantizer configuration for input quantizer. Default: None.
    paq_conf : QuantizerConfig or None, optional
        Quantizer configuration for post-activation quantizer. Default: None.
    praq_conf : QuantizerConfig or None, optional
        Quantizer configuration for pre-recurrent activation quantizer. Default: None.
    sq_conf : QuantizerConfig or None, optional
        Quantizer configuration for state quantizer. Default: None.
    kq_conf : QuantizerConfig or None, optional
        Quantizer configuration for kernel quantizer. Default: None.
    rkq_conf : QuantizerConfig or None, optional
        Quantizer configuration for recurrent kernel quantizer. Default: None.
    bq_conf : QuantizerConfig or None, optional
        Quantizer configuration for bias quantizer. Default: None.
    oq_conf : QuantizerConfig or None, optional
        Quantizer configuration for output quantizer. Default: None.
    rhq_conf : QuantizerConfig or None, optional
        Quantizer configuration for recurrent hidden state quantizer. Default: None.
    standalone : bool, optional
        Whether this cell is used standalone or as part of a larger RNN layer.
        EBOPS computation will be skipped when used as a sublayer.
        Default: True.
    enable_ebops : bool or None, optional
        Whether to enable energy-efficient bit operations. Default: None.
    enable_iq : bool or None, optional
        Whether to enable input quantizer. Default: None.
    enable_oq : bool or None, optional
        Whether to enable output quantizer. Default: None.

    Notes
    -----
    inputs : array_like
        A 2D tensor, with shape `(batch, features)`.
    states : array_like
        A 2D tensor with shape `(batch, units)`, which is the state
        from the previous time step.
    training : bool, optional
        Python boolean indicating whether the layer should behave in
        training mode or in inference mode. Only relevant when `dropout` or
        `recurrent_dropout` is used.
    """

    def __init__(
        self,
        units,
        activation='linear',
        recurrent_activation='linear',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        recurrent_initializer='orthogonal',
        bias_initializer='zeros',
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        reset_after=True,
        seed=None,
        iq_conf: QuantizerConfig | None = None,
        paq_conf: QuantizerConfig | None = None,
        praq_conf: QuantizerConfig | None = None,
        sq_conf: QuantizerConfig | None = None,
        kq_conf: QuantizerConfig | None = None,
        rkq_conf: QuantizerConfig | None = None,
        bq_conf: QuantizerConfig | None = None,
        oq_conf: QuantizerConfig | None = None,
        rhq_conf: QuantizerConfig | None = None,
        standalone: bool = True,
        enable_ebops: bool | None = None,
        enable_iq: bool | None = None,
        enable_oq: bool | None = None,
        **kwargs,
    ):
        super().__init__(
            units=units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            reset_after=reset_after,
            seed=seed,
            oq_conf=oq_conf,
            enable_oq=enable_oq,
            enable_iq=enable_iq,
            enable_ebops=enable_ebops,
            **kwargs,
        )

        paq_conf = paq_conf or HardTanhConfig(place='datalane')
        praq_conf = praq_conf or HardSigmoidConfig(place='datalane')
        iq_conf = iq_conf or QuantizerConfig(place='datalane')
        sq_conf = sq_conf or QuantizerConfig(place='datalane')
        kq_conf = kq_conf or QuantizerConfig(place='weight')
        rkq_conf = rkq_conf or QuantizerConfig(place='weight')
        bq_conf = bq_conf or QuantizerConfig(place='bias')
        rhq_conf = rhq_conf or QuantizerConfig(place='datalane')
        self.standalone = standalone

        if self._enable_iq:
            self._iq = Quantizer(iq_conf, name=f'{self.name}_iq')
        self._paq = Quantizer(paq_conf, name=f'{self.name}_paq')
        self._praq = Quantizer(praq_conf, name=f'{self.name}_praq')
        self._sq = Quantizer(sq_conf, name=f'{self.name}_sq')
        self._kq = Quantizer(kq_conf, name=f'{self.name}_kq')
        self._rkq = Quantizer(rkq_conf, name=f'{self.name}_rkq')
        if self.use_bias:
            self._bq = Quantizer(bq_conf, name=f'{self.name}_bq')
        self._rhq = Quantizer(rhq_conf, name=f'{self.name}_rhq')

    @property
    def paq(self):
        return self._paq

    @property
    def praq(self):
        return self._praq

    @property
    def kq(self):
        return self._kq

    @property
    def bq(self):
        return self._bq

    @property
    def rkq(self):
        return self._rkq

    @property
    def iq(self):
        if not self.enable_iq:
            raise AttributeError(f'iq has been disabled for {self.name}.')
        return self._iq

    @property
    def sq(self):
        return self._sq

    @property
    def rhq(self):
        return self._rhq

    @property
    def qkernel(self):
        return self.kq(self.kernel)

    @property
    def qrecurrent_kernel(self):
        return self.rkq(self.recurrent_kernel)

    @property
    def qbias(self):
        if not self.use_bias:
            raise AttributeError(f'bias has been disabled for {self.name}.')
        assert self.bq is not None
        return self.bq(self.bias)

    def qactivation(self, x):
        return self.paq(self.activation(x))

    def qrecurrent_activation(self, x):
        return self.praq(self.recurrent_activation(x))

    def build(self, input_shape):
        state_shape = (input_shape[0], self.units)
        super().build(input_shape)
        if self._enable_iq:
            self.iq.build(input_shape)
            self.sq.build(state_shape)
        self.kq.build(self.kernel.shape)
        if self.use_bias:
            assert self.bq is not None
            self.bq.build(self.bias.shape)  # type: ignore
        self.rkq.build(self.recurrent_kernel.shape)
        if self._enable_oq:
            self.oq.build(state_shape)
        self.rhq.build(state_shape)
        self.paq.build(state_shape)
        self.praq.build((state_shape[0], 2 * self.units))

    def call(self, inputs, states, training=False):
        h_tm1 = states[0] if tree.is_nested(states) else states  # previous state

        qh_tm1 = self.sq(h_tm1)
        qinputs = self.iq(inputs) if self.enable_iq else inputs

        if self.use_bias:
            if not self.reset_after:
                input_qbias, recurrent_qbias = self.qbias, None
            else:
                input_qbias, recurrent_qbias = self.qbias
        else:
            input_qbias, recurrent_qbias = 0, 0

        if training and 0.0 < self.dropout < 1.0:
            dp_mask = self.get_dropout_mask(qinputs)
            qinputs = qinputs * dp_mask

        matrix_x = qinputs @ self.qkernel + input_qbias

        x_zr = matrix_x[:, : 2 * self.units]
        x_h = matrix_x[:, 2 * self.units :]

        qrecurrent_kernel = self.qrecurrent_kernel
        if self.reset_after:
            # hidden state projected by all gate matrices at once
            matrix_inner = qh_tm1 @ qrecurrent_kernel
            if self.use_bias:
                matrix_inner += recurrent_qbias
        else:
            # hidden state projected separately for update/reset and new
            matrix_inner = qh_tm1 @ qrecurrent_kernel[:, : 2 * self.units]

        recurrent_zr = matrix_inner[:, : 2 * self.units]

        qzr = self.qrecurrent_activation(x_zr + recurrent_zr)
        qz, qr = ops.split(qzr, 2, axis=-1)

        if self.reset_after:
            recurrent_h = qr * self.rhq(matrix_inner[:, self.units * 2 :])
        else:
            recurrent_h = ops.matmul(self.rhq(qr * qh_tm1), qrecurrent_kernel[:, 2 * self.units :])

        qhh = self.qactivation(x_h + recurrent_h)

        # previous and candidate state mixed by update gate
        h = qz * qh_tm1 + (1 - qz) * qhh  # type: ignore
        new_state = [h] if tree.is_nested(states) else h
        return h, new_state

    def get_initial_state(self, batch_size=None):
        return [ops.zeros((batch_size, self.state_size), dtype=self.compute_dtype)]

    def get_config(self):
        config = {
            'paq_conf': self.paq.config,
            'praq_conf': self.praq.config,
            'iq_conf': self.iq.config if self.enable_iq else None,
            'sq_conf': self.sq.config,
            'kq_conf': self.kq.config,
            'rkq_conf': self.rkq.config,
            'bq_conf': self.bq.config if self.use_bias else None,  # type: ignore
            'beta0': self._beta0,
            **super().get_config(),
        }
        return config

    def _compute_ebops(self, shape, state_shape):
        bw_state = self.sq.bits_(state_shape)
        bw_inp = self.iq.bits_(shape) if self.enable_iq else 0
        bw_ker = self.kq.bits_(ops.shape(self.kernel))
        bw_rker = self.rkq.bits_(ops.shape(self.recurrent_kernel))
        bw_rh = self.rhq.bits_(state_shape)
        bw_zr = self.praq.bits_((*shape[:-1], 2 * self.units))

        if self.use_bias:
            bw_bias = self.bq.bits_(ops.shape(self.bias))  # type: ignore
            if not self.reset_after:
                ebops_bias = ops.sum(bw_bias[0])
            else:
                ebops_bias = ops.sum(bw_bias)
        else:
            ebops_bias = 0

        ebops_0 = ops.sum(ops.matmul(bw_inp, bw_ker))

        if self.reset_after:
            ebops1 = ops.sum(ops.matmul(bw_state, bw_rker))
        else:
            ebops1 = ops.sum(ops.matmul(bw_state, bw_rker[:, : 2 * self.units]))

        bw_z, bw_r = ops.split(bw_zr, 2, axis=-1)
        if self.reset_after:
            ebops2 = ops.sum(bw_r * bw_rh)  # type: ignore
        else:
            ebops2 = ops.sum(ops.matmul(bw_rh, bw_rker[:, 2 * self.units :])) + ops.sum(bw_r * bw_state)  # type: ignore

        bw_qhh = self.paq.bits_(state_shape)

        ebops3 = ops.sum(bw_z * (bw_qhh + bw_state))  # type: ignore
        return ebops_0 + ebops1 + ebops2 + ebops3 + ebops_bias  # type: ignore

    @property
    def enable_ebops(self):
        # When used as a sublayer in the RNN layer, standalone is set to False
        # EBOPs computation handled on the higher level RNN layer
        return self._enable_ebops and self.standalone


class QGRU(QRNN, GRU):
    """Gated Recurrent Unit - Cho et al. 2014.

    The QGRU only allows the backend native implementation (no CuDNN kernel).
    When the jax backend is used, if any `WRAP` quantizers are used, unroll will
    be set to `True` to avoid the side effect issue in the `jax.lax.scan` loop.

    Parameters
    ----------
    units : int
        Positive integer, dimensionality of the output space.
    activation : str, optional
        Activation function to use.
        Default: linear, effectively hard_tanh by the pre-activation quantizer.
    recurrent_activation : str, optional
        Activation function to use for the recurrent step.
        Default: linear, effectively hard_sigmoid (slope=0.5) by the pre-activation quantizer.
    use_bias : bool, optional
        Whether the layer should use a bias vector. Default: True.
    kernel_initializer : str, optional
        Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs. Default: "glorot_uniform".
    recurrent_initializer : str, optional
        Initializer for the `recurrent_kernel` weights matrix, used for the linear transformation
        of the recurrent state. Default: "orthogonal".
    bias_initializer : str, optional
        Initializer for the bias vector. Default: "zeros".
    kernel_regularizer : optional
        Regularizer function applied to the `kernel` weights matrix. Default: None.
    recurrent_regularizer : optional
        Regularizer function applied to the `recurrent_kernel` weights matrix. Default: None.
    bias_regularizer : optional
        Regularizer function applied to the bias vector. Default: None.
    activity_regularizer : optional
        Regularizer function applied to the output of the layer (its "activation"). Default: None.
    kernel_constraint : optional
        Constraint function applied to the `kernel` weights matrix. Default: None.
    recurrent_constraint : optional
        Constraint function applied to the `recurrent_kernel` weights matrix. Default: None.
    bias_constraint : optional
        Constraint function applied to the bias vector. Default: None.
    dropout : float, optional
        Float between 0 and 1. Fraction of the units to drop for the
        linear transformation of the inputs. Default: 0.
    recurrent_dropout : float, optional
        Float between 0 and 1. Fraction of the units to drop for the
        linear transformation of the recurrent state. Default: 0.
    seed : int, optional
        Random seed for dropout.
    return_sequences : bool, optional
        Whether to return the last output in the output sequence, or the full sequence. Default: False.
    return_state : bool, optional
        Whether to return the last state in addition to the output. Default: False.
    go_backwards : bool, optional
        If True, process the input sequence backwards and return the reversed sequence. Default: False.
    stateful : bool, optional
        If True, the last state for each sample at index i in a batch will be used as initial
        state for the sample of index i in the following batch. Default: False.
    unroll : bool or None, optional
        None is equivalent to False. However, for the JAX backend, if
        any `WRAP` quantizers are used, unroll will be set to True
        to avoid the side effect issue in the `jax.lax.scan` loop.
        If True, the network will be unrolled,
        else a symbolic loop will be used.
        Unrolling can speed-up a RNN,
        although it tends to be more memory-intensive.
        Unrolling is only suitable for short sequences. Default: None.
    reset_after : bool, optional
        GRU convention (whether to apply reset gate after or
        before matrix multiplication). False is "before",
        True is "after" (default and cuDNN compatible). Default: True.
    iq_conf : QuantizerConfig or None, optional
        Quantizer configuration for input quantizer. Default: None (global default)
    paq_conf : QuantizerConfig or None, optional
        Quantizer configuration for post-activation quantizer.
        Default: None (hard tanh like, w/ global default)
    praq_conf : QuantizerConfig or None, optional
        Quantizer configuration for pre-recurrent activation quantizer.
        Default: None (hard sigmoid like, w/ global default)
    sq_conf : QuantizerConfig or None, optional
        Quantizer configuration for state quantizer. Default: None (global default)
    kq_conf : QuantizerConfig or None, optional
        Quantizer configuration for kernel quantizer. Default: None (global default)
    rkq_conf : QuantizerConfig or None, optional
        Quantizer configuration for recurrent kernel quantizer. Default: None (global default)
    bq_conf : QuantizerConfig or None, optional
        Quantizer configuration for bias quantizer. Default: None (global default)
    oq_conf : QuantizerConfig or None, optional
        Quantizer configuration for output quantizer. Default: None (global default)
    rhq_conf : QuantizerConfig or None, optional
        Quantizer configuration for recurrent hidden state quantizer. Default: None (global default)
    parallelization_factor : int, optional
        Factor for parallelization. Default: 1.
    enable_oq : bool or None, optional
        Whether to enable output quantizer. Default: None (global default)
    enable_iq : bool or None, optional
        Whether to enable input quantizer. Default: None (global default)
    enable_ebops : bool or None, optional
        Whether to enable energy-efficient bit operations. Default: None (global default)
    beta0 : float or None, optional
        Beta0 parameter for quantizer. Default: None (global default)
    enable_ebops : bool or None, optional
        Whether to enable EBOPs resource consumption estimation. Default: None (global default).
    parallelization_factor : int, optional
        Number of cells to be computed in parallel. Default: 1.

    Notes
    -----
    inputs : array_like
        A 3D tensor, with shape `(batch, timesteps, feature)`.
    mask : array_like, optional
        Binary tensor of shape `(samples, timesteps)` indicating whether
        a given timestep should be masked (optional).
        An individual `True` entry indicates that the corresponding timestep
        should be utilized, while a `False` entry indicates that the
        corresponding timestep should be ignored. Defaults to `None`.
    training : bool, optional
        Python boolean indicating whether the layer should behave in
        training mode or in inference mode. This argument is passed to the
        cell when calling it. This is only relevant if `dropout` or
        `recurrent_dropout` is used (optional). Defaults to `None`.
    initial_state : list, optional
        List of initial state tensors to be passed to the first
        call of the cell (optional, `None` causes creation
        of zero-filled initial state tensors). Defaults to `None`.
    """

    def __init__(
        self,
        units,
        activation='linear',
        recurrent_activation='linear',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        recurrent_initializer='orthogonal',
        bias_initializer='zeros',
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        seed=None,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        unroll=None,
        reset_after=True,
        iq_conf: QuantizerConfig | None = None,
        paq_conf: QuantizerConfig | None = None,
        praq_conf: QuantizerConfig | None = None,
        sq_conf: QuantizerConfig | None = None,
        kq_conf: QuantizerConfig | None = None,
        rkq_conf: QuantizerConfig | None = None,
        bq_conf: QuantizerConfig | None = None,
        oq_conf: QuantizerConfig | None = None,
        rhq_conf: QuantizerConfig | None = None,
        parallelization_factor=1,
        enable_oq: bool | None = None,
        enable_iq: bool | None = None,
        enable_ebops: bool | None = None,
        beta0: float | None = None,
        **kwargs,
    ):
        cell = QGRUCell(
            units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            reset_after=reset_after,
            dtype=kwargs.get('dtype', None),
            trainable=kwargs.get('trainable', True),
            name='qgru_cell',
            seed=seed,
            standalone=False,
            iq_conf=iq_conf,
            paq_conf=paq_conf,
            praq_conf=praq_conf,
            sq_conf=sq_conf,
            kq_conf=kq_conf,
            rkq_conf=rkq_conf,
            bq_conf=bq_conf,
            oq_conf=oq_conf,
            rhq_conf=rhq_conf,
            enable_iq=enable_iq,
            enable_oq=enable_oq,
            enable_ebops=enable_ebops,
            beta0=beta0,
        )
        super(GRU, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,  # type: ignore
            activity_regularizer=activity_regularizer,
            **kwargs,
        )
        self.input_spec = InputSpec(ndim=3)
        self.use_cudnn = False
        self.parallelization_factor = parallelization_factor
        self._set_unroll()

    def get_config(self):
        base_config = super().get_config()
        conf = {
            'iq_conf': self.cell.iq.config if self.cell.enable_iq else None,
            'paq_conf': self.cell.paq.config,
            'praq_conf': self.cell.praq.config,
            'sq_conf': self.cell.sq.config,
            'kq_conf': self.cell.kq.config,
            'rkq_conf': self.cell.rkq.config,
            'bq_conf': self.cell.bq.config if self.cell.use_bias else None,
            'oq_conf': self.cell.oq.config if self.cell.enable_oq else None,
            'rhq_conf': self.cell.rhq.config,
        }
        # del base_config['cell']
        return {**base_config, **conf}

    def _compute_ebops(self, shape):
        state_shape = (shape[0], self.cell.units)
        cell_shape = (shape[0], shape[2])  # (batch, features)
        return self.cell._compute_ebops(cell_shape, state_shape) * self.parallelization_factor


QLayerBase.register(QGRU)
