from warnings import warn

import keras
from keras import ops
from keras.initializers import Constant
from keras.saving import deserialize_keras_object, register_keras_serializable
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.rnn.simple_rnn import RNN, SimpleRNN, SimpleRNNCell

from ...config import HardTanhConfig, QuantizerConfig
from ...layers.core.base import QLayerBase, QLayerMeta
from ...quantizer import Quantizer
from ...quantizer.internal import FixedPointQuantizerBase
from ..core.base import QLayerBaseSingleInput


class QSimpleRNNCell(QLayerBaseSingleInput, SimpleRNNCell):
    """Cell class for the QSimpleRNN layer.

    This class processes one step within the whole time sequence input, whereas
    `QSimpleRNN` processes the whole sequence.

    Parameters
    ----------
    units : int
        Positive integer, dimensionality of the output space.
    activation : str, optional
        Activation function to use. Default: "linear".
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
    seed : int, optional
        Random seed for dropout.
    enable_sq : bool or None, optional
        Whether to enable state quantizer.
        When the output is already quantized and the state is 0-inited,
        the state quantizer should be disabled to avoid double quantization.
        Default: None, which means it will be set to the value of `standalone`.
    iq_conf : QuantizerConfig or None, optional
        Quantizer configuration for input quantizer. Default: None.
    sq_conf : QuantizerConfig or None, optional
        Quantizer configuration for state quantizer. Default: None.
    kq_conf : QuantizerConfig or None, optional
        Quantizer configuration for kernel quantizer. Default: None.
    rkq_conf : QuantizerConfig or None, optional
        Quantizer configuration for recurrent kernel quantizer. Default: None.
    bq_conf : QuantizerConfig or None, optional
        Quantizer configuration for bias quantizer. Default: None.
    paq_conf : QuantizerConfig or None, optional
        Quantizer configuration for pre-activation quantizer. Default: None.
    standalone : bool, optional
        Whether this cell is used standalone or as part of a larger RNN layer. Default: True.
    """

    @property
    def kq(self):
        "Kernel Quantizer"
        return self._kq

    @property
    def rkq(self):
        "Recurrent Kernel Quantizer"
        return self._rkq

    @property
    def iq(self):
        "Input Quantizer"
        return self._iq

    @property
    def sq(self):
        "State Quantizer"
        if not self.enable_sq:
            raise ValueError('State Quantizer is not enabled.')
        return self._sq

    @property
    def bq(self):
        "Bias Quantizer"
        if not self.use_bias:
            return None
        return self._bq

    @property
    def paq(self):
        "Pre-Activation Quantizer"
        return self._paq

    @property
    def qkernel(self):
        return self.kq(self.kernel)

    @property
    def qrecurrent_kernel(self):
        return self.rkq(self.recurrent_kernel)

    @property
    def qbias(self):
        if not self.use_bias:
            return None
        return self.bq(self.bias)  # type: ignore

    @property
    def enable_sq(self):
        return self._enable_sq

    def __init__(
        self,
        units,
        activation='linear',
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
        seed=None,
        enable_sq: bool | None = None,
        iq_conf: QuantizerConfig | None = None,
        sq_conf: QuantizerConfig | None = None,
        kq_conf: QuantizerConfig | None = None,
        rkq_conf: QuantizerConfig | None = None,
        bq_conf: QuantizerConfig | None = None,
        paq_conf: QuantizerConfig | None = None,
        standalone=True,
        **kwargs,
    ):
        super().__init__(
            units=units,
            activation=activation,
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
            iq_conf=iq_conf,
            seed=seed,
            **kwargs,
        )

        if enable_sq is None:
            enable_sq = standalone
        self._enable_sq = enable_sq

        iq_conf = iq_conf or QuantizerConfig(place='datalane')
        sq_conf = sq_conf or QuantizerConfig(place='datalane')
        kq_conf = kq_conf or QuantizerConfig(place='weight')
        rkq_conf = rkq_conf or QuantizerConfig(place='weight')
        bq_conf = bq_conf or QuantizerConfig(place='bias')
        paq_conf = paq_conf or HardTanhConfig(place='datalane')

        if self.enable_sq:
            self._sq = Quantizer(sq_conf, name='sq')
        self._kq = Quantizer(kq_conf, name='kq')
        self._rkq = Quantizer(rkq_conf, name='rkq')
        if use_bias:
            self._bq = Quantizer(bq_conf, name='bq')
        self._paq = Quantizer(paq_conf, name='paq')

        self.standalone = standalone

    def build(self, input_shape):
        if self.enable_sq:
            self._sq.build((None, self.units))
        self._kq.build((input_shape[-1], self.units))
        self._rkq.build((self.units, self.units))
        if self.use_bias:
            self._bq.build((self.units,))
        self._paq.build((None, self.units))
        super().build(input_shape)

    def call(self, sequence, states, training=False):
        prev_output = states[0] if isinstance(states, (list, tuple)) else states
        dp_mask = self.get_dropout_mask(sequence)
        rec_dp_mask = self.get_recurrent_dropout_mask(prev_output)

        if training and dp_mask is not None:
            sequence = sequence * dp_mask
        if training and rec_dp_mask is not None:
            prev_output = prev_output * rec_dp_mask

        qkernel = self.kq(self.kernel, training=training)
        qrecurrent_kernel = self.rkq(self.recurrent_kernel, training=training)
        if self.enable_sq:
            qstate = self.sq(prev_output, training=training)
        else:
            # If the output of the activation is already quantized, sq shall be disabled.
            qstate = prev_output
        qsequence = self.iq(sequence, training=training)

        h = ops.matmul(qsequence, qkernel)
        if self.bias is not None:
            h += self.bq(self.bias, training=training)  # type: ignore

        output = h + ops.matmul(qstate, qrecurrent_kernel)  # type: ignore

        output = self.paq(output)
        output = self.activation(output)

        new_state = [output] if isinstance(states, (list, tuple)) else output
        return output, new_state

    def get_config(self):
        return {
            'iq_conf': self._iq.config,
            'sq_conf': self._sq.config if self.enable_sq else None,
            'kq_conf': self._kq.config,
            'rkq_conf': self._rkq.config,
            'bq_conf': self._bq.config if self.use_bias else None,
            'paq_conf': self._paq.config,
            'enable_sq': self._enable_sq,
            'standalone': self.standalone,
            **super().get_config(),
        }

    def _compute_ebops(self, shape):
        bw_inp = self.iq.bits_(shape)
        bw_ker = self.kq.bits_(ops.shape(self.kernel))
        ebops1 = ops.sum(ops.matmul(bw_inp, bw_ker))
        if self.enable_sq:
            bw_state = self.sq.bits_((1, self.units))
        else:
            bw_state = self.paq.bits_((1, self.units))
        bw_rker = self.rkq.bits_(ops.shape(self.recurrent_kernel))
        ebops2 = ops.sum(ops.matmul(bw_state, bw_rker))
        ebops = ebops1 + ebops2  # type: ignore
        if self.bq is not None:
            bw_bias = self.bq.bits_(ops.shape(self.bias))
            size = ops.cast(ops.prod(shape), self.dtype)
            ebops = ebops + ops.mean(bw_bias) * size  # type: ignore
        return ebops

    @property
    def enable_ebops(self):
        # When used as a sublayer in the RNN layer, standalone is set to False
        # EBOPs computation handled on the higher level RNN layer
        return self._enable_ebops and self.standalone


def _is_wrap_quantizer(q: Quantizer) -> bool:
    if not isinstance(q, Quantizer):
        return False
    if not isinstance(q.quantizer, FixedPointQuantizerBase):
        return False
    return q.quantizer.overflow_mode == 'WRAP'


@register_keras_serializable(package='hgq')
class QRNN(RNN, metaclass=QLayerMeta):
    __output_quantizer_handled__ = True

    def _set_unroll(self):
        backend = keras.backend.backend()
        has_wrap_quantizers = any(_is_wrap_quantizer(layer) for layer in self._flatten_layers())
        if backend == 'jax' and has_wrap_quantizers:
            # JAX tracer issues when using jax scan with WRAP quantizers (range update is side effect)
            # Force unrolling in this case
            if self.unroll is False:
                warn('JAX backend does not support WRAP quantizers with rolled RNNs. Forcing unrolling.')
            self.unroll = True
        if self.unroll is None:
            self.unroll = False  # Keras default

    def build(self, sequences_shape, initial_state_shape=None):
        seq_len = sequences_shape[1]
        if self.parallelization_factor == -1:
            self.parallelization_factor = seq_len

        if self.enable_ebops:
            self._beta = self.add_weight(
                name='beta',
                shape=(),
                initializer=self.cell._beta0,
                trainable=False,
            )
            self._ebops = self.add_weight(
                name='ebops',
                shape=(),
                initializer=Constant(0.0),
                trainable=False,
                dtype='uint32',
            )
        else:
            self._beta = None
            self._ebops = None

        super().build(sequences_shape, initial_state_shape)

    def get_config(self):
        return {
            'parallelization_factor': self.parallelization_factor,
            'enable_ebops': self.enable_ebops,
            'enable_iq': self.enable_iq,
            'enable_oq': self.enable_oq,
            'beta0': self.cell._beta0,
            **super().get_config(),
        }

    @property
    def enable_ebops(self):
        return self.cell._enable_ebops

    @property
    def enable_iq(self):
        return self.cell._enable_iq

    @property
    def enable_oq(self):
        return self.cell._enable_oq

    @property
    def beta(self):
        if self._beta is None:
            return ops.cast(0, 'float32')
        return ops.cast(self._beta, ops.dtype(self._beta))

    @property
    def ebops(self):
        if self._ebops is None:
            return ops.cast(0, 'uint32')
        return ops.cast(self._ebops, ops.dtype(self._ebops))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = deserialize_keras_object(config, custom_objects=custom_objects)
        return super().from_config(config)


@register_keras_serializable(package='hgq')
class QSimpleRNN(QRNN, SimpleRNN):
    """Quantized Fully-connected RNN where the output is to be fed back as the new input.

    When the jax backend is used, if any `WRAP` quantizers are used, unroll will
    be set to `True` to avoid the side effect issue in the `jax.lax.scan` loop.

    Parameters
    ----------
    units : int
        Positive integer, dimensionality of the output space.
    activation : str, optional
        Activation function to use.
        Default: linear, effectively hard_tanh by the pre-activation quantizer.
    use_bias : bool, optional
        Whether the layer uses a bias vector. Default: True.
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
        Float between 0 and 1. Fraction of the units to drop for the linear transformation
        of the inputs. Default: 0.
    recurrent_dropout : float, optional
        Float between 0 and 1. Fraction of the units to drop for the linear transformation of the
        recurrent state. Default: 0.
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
    seed : int, optional
        Random seed for dropout.
    iq_conf : QuantizerConfig or None, optional
        Quantizer configuration for input quantizer. Default: None.
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
    paq_conf : QuantizerConfig or None, optional
        Quantizer configuration for pre-activation quantizer. Default: None.
    parallelization_factor : int, optional
        Factor for parallelization. Default: -1 (automatic).
    enable_sq : bool or None, optional
        Whether to enable state quantizer.
        When the output is already quantized and the state is 0-inited,
        the state quantizer should be disabled to avoid double quantization.
        Default: False.
    enable_oq : bool or None, optional
        Whether to enable output quantizer. Default: None.
    enable_iq : bool or None, optional
        Whether to enable input quantizer. Default: None.
    enable_ebops : bool or None, optional
        Whether to enable energy-efficient bit operations. Default: None.
    beta0 : float or None, optional
        Beta0 parameter for quantizer. Default: None.

    Notes
    -----
    sequence : array_like
        A 3D tensor, with shape `[batch, timesteps, feature]`.
    mask : array_like, optional
        Binary tensor of shape `[batch, timesteps]` indicating whether
        a given timestep should be masked. An individual `True` entry
        indicates that the corresponding timestep should be utilized,
        while a `False` entry indicates that the corresponding timestep
        should be ignored.
    training : bool, optional
        Python boolean indicating whether the layer should behave in
        training mode or in inference mode.
        This argument is passed to the cell when calling it.
        This is only relevant if `dropout` or `recurrent_dropout` is used.
    initial_state : list, optional
        List of initial state tensors to be passed to the first
        call of the cell.
    """

    def __init__(
        self,
        units,
        activation='linear',
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
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        unroll=None,
        seed=None,
        iq_conf: QuantizerConfig | None = None,
        sq_conf: QuantizerConfig | None = None,
        kq_conf: QuantizerConfig | None = None,
        rkq_conf: QuantizerConfig | None = None,
        bq_conf: QuantizerConfig | None = None,
        oq_conf: QuantizerConfig | None = None,
        paq_conf: QuantizerConfig | None = None,
        parallelization_factor=-1,
        enable_oq: bool | None = None,
        enable_iq: bool | None = None,
        enable_sq: bool | None = False,
        enable_ebops: bool | None = None,
        beta0: float | None = None,
        **kwargs,
    ):
        cell = QSimpleRNNCell(
            units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            seed=seed,
            dtype=kwargs.get('dtype', None),
            trainable=kwargs.get('trainable', True),
            name='simple_rnn_cell',
            iq_conf=iq_conf,
            sq_conf=sq_conf,
            kq_conf=kq_conf,
            rkq_conf=rkq_conf,
            bq_conf=bq_conf,
            oq_conf=oq_conf,
            enable_sq=enable_sq,
            paq_conf=paq_conf,
            standalone=False,
            enable_oq=enable_oq,
            enable_iq=enable_iq,
            enable_ebops=enable_ebops,
            beta0=beta0,
        )
        super(SimpleRNN, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,  # type: ignore
            **kwargs,
        )
        self.input_spec = [InputSpec(ndim=3)]
        self.parallelization_factor = parallelization_factor
        self._set_unroll()

    def _compute_ebops(self, shape):
        cell_shape = (1, *shape[1:])
        ebops = self.cell._compute_ebops(cell_shape) * self.parallelization_factor
        return ebops

    def get_config(self):  # type: ignore
        base_conf = super().get_config()
        conf = {
            'iq_conf': self.cell.iq.config if self.enable_iq else None,
            'sq_conf': self.cell.sq.config if self.cell.enable_sq else None,
            'kq_conf': self.cell.kq.config,
            'rkq_conf': self.cell.rkq.config,
            'bq_conf': self.cell.bq.config if self.cell.use_bias else None,
            'oq_conf': self.cell.oq.config if self.enable_oq else None,
            'paq_conf': self.cell.paq.config,
        }
        return {**base_conf, **conf}


QLayerBase.register(QSimpleRNN)
