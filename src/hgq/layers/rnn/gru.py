from keras import ops
from keras.layers import GRUCell
from keras.saving import register_keras_serializable, serialize_keras_object
from keras.src import tree

from ...config import HardSigmoidConfig, HardTanhConfig, QuantizerConfig
from ...quantizer import Quantizer
from ..core.base import QLayerBase


@register_keras_serializable(package='hgq')
class QGRUCell(QLayerBase, GRUCell):
    """Cell class for the GRU layer.

    This class processes one step within the whole time sequence input, whereas
    `keras.layer.GRU` processes the whole sequence.

    Args:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use. Default: hyperbolic tangent
            (`tanh`). If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use for the recurrent step.
            Default: sigmoid (`sigmoid`). If you pass `None`, no activation is
            applied (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, (default `True`), whether the layer
            should use a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs. Default:
            `"glorot_uniform"`.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix, used for the linear transformation
            of the recurrent state. Default: `"orthogonal"`.
        bias_initializer: Initializer for the bias vector. Default: `"zeros"`.
        kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_regularizer: Regularizer function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_regularizer: Regularizer function applied to the bias vector.
            Default: `None`.
        kernel_constraint: Constraint function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_constraint: Constraint function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_constraint: Constraint function applied to the bias vector.
            Default: `None`.
        dropout: Float between 0 and 1. Fraction of the units to drop for the
            linear transformation of the inputs. Default: 0.
        recurrent_dropout: Float between 0 and 1. Fraction of the units to drop
            for the linear transformation of the recurrent state. Default: 0.
        reset_after: GRU convention (whether to apply reset gate after or
            before matrix multiplication). False = "before",
            True = "after" (default and cuDNN compatible).
        seed: Random seed for dropout.

    Call arguments:
        inputs: A 2D tensor, with shape `(batch, features)`.
        states: A 2D tensor with shape `(batch, units)`, which is the state
            from the previous time step.
        training: Python boolean indicating whether the layer should behave in
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
        paq_conf: QuantizerConfig | None = None,
        praq_conf: QuantizerConfig | None = None,
        iq_conf: QuantizerConfig | None = None,
        sq_conf: QuantizerConfig | None = None,
        kq_conf: QuantizerConfig | None = None,
        rkq_conf: QuantizerConfig | None = None,
        bq_conf: QuantizerConfig | None = None,
        oq_conf: QuantizerConfig | None = None,
        rhq_conf: QuantizerConfig | None = None,
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

        if self._enable_iq:
            self._iq = Quantizer(iq_conf, name=f'{self.name}_iq')
        self._paq = Quantizer(paq_conf, name=f'{self.name}_paq')
        self._praq = Quantizer(praq_conf, name=f'{self.name}_praq')
        self._sq = Quantizer(sq_conf, name=f'{self.name}_sq')
        self._kq = Quantizer(kq_conf, name=f'{self.name}_kq')
        self._rkq = Quantizer(rkq_conf, name=f'{self.name}_rkq')
        self._bq = Quantizer(bq_conf, name=f'{self.name}_bq') if self.use_bias else None
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
        return self.bq(self.bias)  # type: ignore

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
            'paq_conf': serialize_keras_object(self.paq.config),
            'praq_conf': serialize_keras_object(self.praq.config),
            'iq_conf': serialize_keras_object(self.iq.config) if self.enable_iq else None,
            'sq_conf': serialize_keras_object(self.sq.config),
            'kq_conf': serialize_keras_object(self.kq.config),
            'rkq_conf': serialize_keras_object(self.rkq.config),
            'bq_conf': serialize_keras_object(self.bq.config) if self.use_bias else None,  # type: ignore
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
