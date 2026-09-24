import math
from collections.abc import Sized
from copy import copy
from typing import Literal

import keras
from keras import ops
from keras.initializers import Constant
from keras.layers import Dropout, MultiHeadAttention
from keras.saving import register_keras_serializable
from keras.src.layers.attention.multi_head_attention import _build_attention_equation, _build_proj_equation

from ...quantizer.config import QuantizerConfig
from ...utils.misc import gather_vars_to_kwargs
from ..activation import table_ebops
from ..core.base import QLayerBase
from ..core.einsum_dense import QEinsumDense
from ..fsoftmax import QFSoftmax, scan_rounds
from ..softmax import QSoftmax
from ..table import QEinsumDenseT

try:
    from alkaid.opsched.passes import _cover_firings
except ImportError:

    def _cover_firings(target: int, shape: tuple[int, ...]) -> tuple[int, int]:
        return max(1, target // math.prod(shape)), 0


def _get_output_shape(output_rank, known_last_dims, input_shape):
    n = output_rank - len(known_last_dims)
    return list(input_shape[1 : n + 1]) + list(known_last_dims)


class QMultiHeadAttention(MultiHeadAttention, QLayerBase):
    __output_quantizer_handled__ = True

    def __init__(
        self,
        num_heads,
        key_dim,
        value_dim=None,
        dropout=0.0,
        use_bias=True,
        output_shape=None,
        attention_axes=None,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        seed=None,
        fuse: Literal['none', 'qkv', 'kv'] = 'none',
        qkvo_iq_conf: QuantizerConfig | None = None,
        qkvo_kq_conf: QuantizerConfig | None = None,
        qkvo_bq_conf: QuantizerConfig | None = None,
        qkvo_oq_conf: QuantizerConfig | None = None,
        softmax: Literal['comb', '1pass', '2pass'] = 'comb',
        softmax_iq_conf: QuantizerConfig | None = None,
        softmax_exp_iq_conf: QuantizerConfig | None = None,
        softmax_exp_oq_conf: QuantizerConfig | None = None,
        softmax_inv_iq_conf: QuantizerConfig | None = None,
        softmax_inv_oq_conf: QuantizerConfig | None = None,
        softmax_lq_conf: QuantizerConfig | None = None,
        softmax_aq_conf: QuantizerConfig | None = None,
        softmax_oq_conf: QuantizerConfig | None = None,
        stable_softmax=True,
        softmax_allow_heterogeneous_table: bool = False,
        parallelization_factor: int = -1,
        target_ii: int | None = None,
        **kwargs,
    ):
        kwargs = gather_vars_to_kwargs('self|.+q_conf')

        self._qkvo_iq_conf = qkvo_iq_conf or QuantizerConfig(place='datalane')
        self._qkvo_kq_conf = qkvo_kq_conf or QuantizerConfig(place='weight')
        self._qkvo_bq_conf = qkvo_bq_conf or QuantizerConfig(place='bias')
        self._qkvo_oq_conf = qkvo_oq_conf or QuantizerConfig(place='datalane')
        self._softmax_iq_conf = softmax_iq_conf or QuantizerConfig(place='datalane')
        self._softmax_exp_iq_conf = softmax_exp_iq_conf or QuantizerConfig(place='datalane', overflow_mode='SAT')
        self._softmax_exp_oq_conf = softmax_exp_oq_conf or QuantizerConfig(place='table')
        self._softmax_inv_iq_conf = softmax_inv_iq_conf or QuantizerConfig(place='datalane')
        self._softmax_inv_oq_conf = softmax_inv_oq_conf or QuantizerConfig(place='table')
        self._softmax_lq_conf = softmax_lq_conf or QuantizerConfig(place='datalane')
        self._softmax_aq_conf = softmax_aq_conf or QuantizerConfig(place='datalane')
        self._softmax_oq_conf = softmax_oq_conf or QuantizerConfig(place='datalane')
        self._softmax_allow_heterogeneous_table = kwargs.pop('softmax_allow_heterogeneous_table')
        self.parallelization_factor = kwargs.pop('parallelization_factor')
        self._target_ii: int | None = kwargs.pop('target_ii')
        if self.target_ii is not None and self.target_ii < 1:
            raise ValueError('target_ii must be positive.')
        self._stable_softmax = kwargs.pop('stable_softmax')
        self._softmax_kind: Literal['comb', '1pass', '2pass'] = kwargs.pop('softmax')
        assert self._softmax_kind in ('comb', '1pass', '2pass'), "softmax must be 'comb', '1pass' or '2pass'."
        self._qkv_oq_conf: QuantizerConfig = self._qkvo_oq_conf
        if self._softmax_kind != 'comb':
            self._qkv_oq_conf = copy(self._qkvo_oq_conf)
            self._qkv_oq_conf.config = self._qkv_oq_conf.config.copy()
            self._qkv_oq_conf.config.update(homogeneous_axis=None, bw_mapper=None)
            self._qkv_oq_conf.config['heterogeneous_axis'] = tuple(
                a for a in self._qkv_oq_conf.config.get('heterogeneous_axis') or () if a in (2, 3, -2, -1)
            )
        if self.target_ii is not None and self._softmax_kind == 'comb':
            raise ValueError("target_ii prices scan softmax; set softmax='1pass' or '2pass', or omit target_ii.")
        self._fuse = kwargs.pop('fuse', 'none').lower()
        self._inverse_sqrt_key_dim = 1.0 / math.sqrt(float(key_dim))

        super().__init__(**kwargs)

    @property
    def target_ii(self) -> int | None:
        return self._target_ii

    def _get_common_kwargs_for_sublayer(self):
        common_kwargs: dict = super()._get_common_kwargs_for_sublayer()
        # Inject quantizer and ebops configs to sub QEinsumDense layers.
        common_kwargs.update(
            {
                'iq_conf': self._qkvo_iq_conf,
                'kq_conf': self._qkvo_kq_conf,
                'bq_conf': self._qkvo_bq_conf,
                'oq_conf': self._qkvo_oq_conf,
                'enable_ebops': self.enable_ebops,
                'beta0': self._beta0.clone(),
                'parallelization_factor': self.parallelization_factor,
            }
        )
        return common_kwargs

    def compute_output_spec(self, query, value=None, key=None, *args, **kwargs):
        value = query if value is None else value
        return super().compute_output_spec(*args, query=query, value=value, key=key, **kwargs)

    def build(
        self,
        query_shape,
        value_shape=None,
        key_shape=None,
    ):
        """Builds layers and variables.

        Parameters
        ----------
        query_shape : tuple
            Shape of the `query` tensor.
        value_shape : tuple, optional
            Shape of the `value` tensor.
        key_shape : tuple, optional
            Shape of the `key` tensor.
        """

        # Copied and modified from keras MultiHeadAttention, substituted
        # EinsumDense with QEinsumDense and added sequence length (shape) to its
        # output shape when initializing, if known.
        value_shape = query_shape if value_shape is None else value_shape
        key_shape = value_shape if key_shape is None else key_shape

        if self._fuse == 'qkv':
            assert query_shape == value_shape == key_shape, (
                'For fuse=qkv, query, key and value must have the same shape',
                f'but got query_shape={query_shape}, value_shape={value_shape} and key_shape={key_shape}.',
            )
        if self._fuse == 'kv':
            assert value_shape == key_shape, (
                'For fuse=kv, key and value must have the same shape',
                f'but got value_shape={value_shape} and key_shape={key_shape}.',
            )

        if value_shape[1:-1] != key_shape[1:-1]:
            raise ValueError(
                'All dimensions of `value` and `key`, except the last one, '
                f'must be equal. Received: value_shape={value_shape} and '
                f'key_shape={key_shape}',
            )

        query_rank = len(query_shape)
        key_rank = len(key_shape)
        value_rank = len(value_shape)

        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            query_rank - 1,
            bound_dims=1,
            output_dims=2,
        )
        self._query_dense = QEinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(
                output_rank - 1,
                [self._num_heads, self._key_dim],
                query_shape,
            ),
            bias_axes=bias_axes if self._use_bias else None,
            name='query',
            enable_iq=self.enable_iq and not self._fuse == 'qkv',
            enable_oq=True,
            **{**self._get_common_kwargs_for_sublayer(), 'oq_conf': self._qkv_oq_conf},
        )

        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            key_rank - 1,
            bound_dims=1,
            output_dims=2,
        )
        self._key_dense = QEinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(
                output_rank - 1,
                [self._num_heads, self._key_dim],
                key_shape,
            ),
            bias_axes=None,  # Useless as it will be directly fed to softmax on seq axis
            name='key',
            enable_iq=self.enable_iq and self._fuse not in ('qkv', 'kv'),
            enable_oq=True,
            **{**self._get_common_kwargs_for_sublayer(), 'oq_conf': self._qkv_oq_conf},
        )

        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            value_rank - 1,
            bound_dims=1,
            output_dims=2,
        )
        self._value_dense = QEinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(
                output_rank - 1,
                [self._num_heads, self._value_dim],
                value_shape,
            ),
            bias_axes=bias_axes if self._use_bias else None,
            name='value',
            enable_iq=self.enable_iq,
            enable_oq=True,
            **{**self._get_common_kwargs_for_sublayer(), 'oq_conf': self._qkv_oq_conf},
        )
        self._value_dense.build(value_shape)

        if self._fuse == 'qkv' and self.enable_iq:
            self._query_dense._iq = self._value_dense._iq
            self._query_dense._enable_iq = True
        if self._fuse in ('qkv', 'kv') and self.enable_iq:
            self._key_dense._iq = self._value_dense._iq
            self._key_dense._enable_iq = True

        self._query_dense.build(query_shape)
        self._key_dense.build(key_shape)

        # Builds the attention computations for multi-head dot product
        # attention.  These computations could be wrapped into the keras
        # attention layer once it supports multi-head einsum computations.
        self._build_attention(output_rank, (query_shape, value_shape, key_shape))
        self._output_dense = self._make_output_dense(
            query_shape,
            self._get_common_kwargs_for_sublayer(),
            'attention_output',
        )
        output_dense_input_shape = list(
            self._query_dense.compute_output_shape(query_shape),
        )
        output_dense_input_shape[-1] = self._value_dim
        self._output_dense.build(tuple(output_dense_input_shape))

        if self.enable_ebops:
            self._beta = self.add_weight(
                name='beta',
                shape=(),
                initializer=self._beta0,
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

        self._dot_product_ebops_equation = self._dot_product_equation.split('->', 1)[0] + '->'
        self._combine_ebops_equation = self._combine_equation.split('->', 1)[0] + '->'

        self.n_parallel = math.prod(query_shape[1:-1])
        if self.parallelization_factor < 0:
            if self.target_ii is None:
                self.parallelization_factor = self.n_parallel
            else:
                # temporary alkaid matched impl, used iff pf<0
                denses = (self._query_dense, self._key_dense, self._value_dense, self._output_dense)
                terms = (query_shape[-1], key_shape[-1], value_shape[-1], self._value_dim)
                spans = (query_shape[1:-1], key_shape[1:-1], value_shape[1:-1], query_shape[1:-1])
                for dense, extents, contraction in zip(denses, spans, terms):
                    cycles, copies = _cover_firings(self.target_ii, extents)
                    budget = min(contraction, cycles)
                    if dense.n_parallel == 1:
                        budget = 1
                    unroll = math.ceil(contraction / budget)
                    steps = math.ceil(contraction / unroll)
                    workers = math.ceil(dense.n_parallel / (self.target_ii // steps))
                    dense.parallelization_factor = copies if copies and steps > 1 else workers
                    # fraction of the contraction a serial step in parallel
                    dense.ebops_factor = unroll / contraction
                self._softmax.parallelization_factor = math.ceil(self.n_parallel * value_shape[1] / self.target_ii)
        self.built = True

    def _make_output_dense(self, query_shape, common_kwargs, name=None):
        """Builds the output projection matrix.

        Parameters
        ----------
        query_shape : tuple
            Shape of the query tensor.
        common_kwargs : dict
            Common keyword arguments for the einsum layer.
        name : str, optional
            Name for the projection layer.

        Returns
        -------
        QEinsumDense

        Notes
        -----
        This method is copied and modified from Keras MultiHeadAttention. It substitutes
        EinsumDense with QEinsumDense and adds sequence length (shape) to its output shape
        when initializing, if known.
        """

        query_rank = len(query_shape)
        if self._output_shape:
            if not isinstance(self._output_shape, Sized):
                output_shape = [self._output_shape]
            else:
                output_shape = self._output_shape
        else:
            output_shape = [query_shape[-1]]
        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            query_rank - 1,
            bound_dims=2,
            output_dims=len(output_shape),
        )
        return QEinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(output_rank - 1, output_shape, query_shape),
            bias_axes=bias_axes if self._use_bias else None,
            name=name,
            enable_iq=True,
            enable_oq=self.enable_oq,
            **common_kwargs,
        )

    def _build_attention(self, rank, shapes):  # type: ignore[reportIncompatibleMethodOverride]
        """Builds multi-head dot-product attention computations.

        This function builds attributes necessary for `_compute_attention` to
        customize attention computation to replace the default dot-product
        attention.

        Parameters
        ----------
        rank : int
            The rank of query, key, value tensors.
        """

        # Copied and modified from keras MultiHeadAttention, substituted Softmax with QSoftmax.
        if self._attention_axes is None:
            self._attention_axes = tuple(range(1, rank - 2))
        elif not isinstance(self._attention_axes, tuple):
            self._attention_axes = tuple(self._attention_axes)
        (
            self._dot_product_equation,
            self._combine_equation,
            attn_scores_rank,
        ) = _build_attention_equation(rank, attn_axes=self._attention_axes)
        norm_axes = tuple(
            range(
                attn_scores_rank - len(self._attention_axes),
                attn_scores_rank,
            ),
        )
        q_shape, v_shape, _ = shapes
        attn_score_shape = (None, self._num_heads, *q_shape[1:-1], *v_shape[1:-1])
        if self._softmax_kind != 'comb':
            assert attn_scores_rank == 4, (
                'the online softmax scans the one key axis of a [batch, heads, query, key] score matrix. '
                'Input shapes must be [B, T, D] in 3D.'
            )
            assert not self._dropout, 'dropout is not supported for online softmax'

            context_shape = tuple(s for i, s in enumerate(attn_score_shape) if i != norm_axes[0]) + (self._value_dim,)
            # The score rides the arrival tape in both online paths, so both land it; only the 2-pass path
            # forms the probability inside its loop, so only it lands one.
            self._softmax = QFSoftmax(
                impl=self._softmax_kind,
                enable_iq=True,
                enable_oq=self._softmax_kind == '2pass',
                axis=norm_axes[0],
                dtype=self.dtype_policy,
                iq_conf=self._softmax_iq_conf,
                exp_iq_conf=self._softmax_exp_iq_conf,
                exp_oq_conf=self._softmax_exp_oq_conf,
                inv_iq_conf=self._softmax_inv_iq_conf,
                inv_oq_conf=self._softmax_inv_oq_conf,
                lq_conf=self._softmax_lq_conf,
                aq_conf=self._softmax_aq_conf,
                oq_conf=self._softmax_oq_conf,
                accumulator_shape=context_shape,
                allow_heterogeneous_table=self._softmax_allow_heterogeneous_table,
                input_scaler=self._inverse_sqrt_key_dim,
                enable_ebops=False,
            )
        else:
            self._softmax = QSoftmax(
                enable_oq=True,
                axis=norm_axes,
                dtype=self.dtype_policy,
                stable=self._stable_softmax,
                iq_conf=self._softmax_iq_conf,
                exp_iq_conf=self._softmax_exp_iq_conf,
                exp_oq_conf=self._softmax_exp_oq_conf,
                inv_iq_conf=self._softmax_inv_iq_conf,
                inv_oq_conf=self._softmax_inv_oq_conf,
                oq_conf=self._softmax_oq_conf,
                allow_heterogeneous_table=self._softmax_allow_heterogeneous_table,
                input_scaler=self._inverse_sqrt_key_dim,
                enable_ebops=self.enable_ebops,
            )
        self._dropout_layer = Dropout(
            rate=self._dropout,
            dtype=self.dtype_policy,
            seed=self.seed,
        )
        self._inverse_sqrt_key_dim = 1.0
        self._softmax.build(attn_score_shape)
        self._dropout_layer.build(attn_score_shape)

    def compute_output_shape(self, query_shape, value_shape, key_shape=None):
        return super().compute_output_shape(query_shape, query_shape, None)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'qkvo_iq_conf': self._qkvo_iq_conf,
                'qkvo_kq_conf': self._qkvo_kq_conf,
                'qkvo_bq_conf': self._qkvo_bq_conf,
                'qkvo_oq_conf': self._qkvo_oq_conf,
                'softmax': self._softmax_kind,
                'softmax_iq_conf': self._softmax_iq_conf,
                'softmax_exp_iq_conf': self._softmax_exp_iq_conf,
                'softmax_exp_oq_conf': self._softmax_exp_oq_conf,
                'softmax_inv_iq_conf': self._softmax_inv_iq_conf,
                'softmax_inv_oq_conf': self._softmax_inv_oq_conf,
                'softmax_lq_conf': self._softmax_lq_conf,
                'softmax_aq_conf': self._softmax_aq_conf,
                'softmax_oq_conf': self._softmax_oq_conf,
                'softmax_allow_heterogeneous_table': self._softmax_allow_heterogeneous_table,
                'parallelization_factor': self.parallelization_factor,
                'target_ii': self.target_ii,
                'stable_softmax': self._stable_softmax,
                'fuse': self._fuse,
            }
        )
        return config

    def _post_build(self):
        if self._enable_oq:
            assert hasattr(self, '_oq'), f'Output Quantizer is not defined for {self.name}, but enable_oq is True.'
        for sublayer in self._flatten_layers():
            assert sublayer.built, f'Sublayer {sublayer.name} is not built for {self.name}'

    def _compute_ebops(self, query_shape, value_shape=None, key_shape=None):
        Q_shape = (1,) + self._query_dense.full_output_shape[1:]
        K_shape = (1,) + self._key_dense.full_output_shape[1:]
        V_shape = (1,) + self._value_dense.full_output_shape[1:]

        value_shape = query_shape if value_shape is None else value_shape
        attn_score_shape = (1, self._num_heads, *query_shape[1:-1], *value_shape[1:-1])

        factor = self.parallelization_factor / self.n_parallel

        bw_q = self._query_dense.oq.bits_(Q_shape)
        bw_k = self._key_dense.oq.bits_(K_shape)
        bw_v = self._value_dense.oq.bits_(V_shape)

        ebops_qk = ops.einsum(self._dot_product_ebops_equation, bw_q, bw_k)
        if self._softmax_kind != 'comb':
            softmax: QFSoftmax = self._softmax
            dk = attn_score_shape[-1]
            state_shape = (*attn_score_shape[:-1], 1)
            context_shape = (*attn_score_shape[:-1], self._value_dim)

            exp_in_bits = softmax.exp_table.iq.bits_(state_shape)
            exp_bits = softmax.exp_table.oq.bits_(state_shape)
            l_bits = softmax.lq.bits_(state_shape)
            acc_bits = softmax.aq.bits_(context_shape)
            inv_in_bits = softmax.inv_table.iq.bits_(state_shape)
            inv_bits = softmax.inv_table.oq.bits_(state_shape)

            comparisons = 3 * ops.sum(softmax.iq.bits_(attn_score_shape))  # type: ignore
            if softmax.impl == '2pass':
                comparisons += ops.sum(softmax.iq.bits_(attn_score_shape))  # type: ignore
                probability_bits = softmax.oq.bits_(state_shape)
                round_ebops = (
                    3 * table_ebops(exp_in_bits, exp_bits)  # type: ignore
                    + ops.sum(exp_bits * l_bits)  # type: ignore
                    + ops.sum(l_bits)  # type: ignore
                    + table_ebops(inv_in_bits, inv_bits)  # type: ignore
                    + ops.sum(exp_bits * inv_bits)  # type: ignore
                    + ops.sum(acc_bits)  # type: ignore
                    + ops.einsum(
                        self._combine_ebops_equation,
                        ops.broadcast_to(probability_bits, attn_score_shape),
                        bw_v,
                    )
                    / dk
                )
                final_ebops = 0
            else:
                round_ebops = (
                    2 * table_ebops(exp_in_bits, exp_bits)  # type: ignore
                    + ops.sum(exp_bits * l_bits)  # type: ignore
                    + ops.sum(l_bits)
                    + 2 * ops.sum(exp_bits * acc_bits)  # type: ignore
                )
                final_ebops = table_ebops(inv_in_bits, inv_bits) + ops.sum(acc_bits * inv_bits)  # type: ignore

            if self.parallelization_factor < 0 and self.target_ii is not None:
                cells = softmax.parallelization_factor
                rows = self.n_parallel
                return (
                    ebops_qk * cells / (rows * dk)  # type: ignore
                    + (comparisons / dk + round_ebops) * cells / rows
                    + final_ebops * math.ceil(rows / self.target_ii) / rows  # type: ignore
                )
            return (ebops_qk + (comparisons + dk * round_ebops + final_ebops)) * factor

        bw_attn = self._softmax.oq.bits_(attn_score_shape)

        ebops_av = ops.einsum(self._combine_ebops_equation, bw_attn, bw_v)
        return (ebops_qk + ebops_av) * factor  # type: ignore

    @property
    def ebops(self):
        if self._ebops is None:
            return ops.cast(0, 'uint32')
        ebops = sum(
            (  # type: ignore
                self._query_dense.ebops,
                self._key_dense.ebops,
                self._value_dense.ebops,
                self._softmax.ebops,
                self._output_dense.ebops,
                ops.convert_to_tensor(self._ebops),  # type: ignore
            )
        )
        return ebops  # type: ignore

    def call(
        self,
        query,
        value=None,
        key=None,
        query_mask=None,
        value_mask=None,
        key_mask=None,
        attention_mask=None,
        return_attention_scores=False,
        training=None,
        use_causal_mask=False,
    ):
        # Adapted from _compute_attention in keras 3.5.0

        if value is None:
            value = query
        if key is None:
            key = value

        attention_mask = self._compute_attention_mask(
            query,
            value,
            query_mask=query_mask,
            value_mask=value_mask,
            key_mask=key_mask,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
        )

        #   N = `num_attention_heads`
        #   H = `size_per_head`
        # `query` = [B, T, N ,H]
        query = self._query_dense(query)

        # `key` = [B, S, N, H]
        key = self._key_dense(key)

        # `value` = [B, S, N, H]
        value = self._value_dense(value)

        attention_output, attention_scores = self._compute_attention(query, key, value, attention_mask, training)
        attention_output = self._output_dense(attention_output)

        if self.enable_oq:
            attention_output = self.oq(attention_output, training=training)

        if return_attention_scores:
            return attention_output, attention_scores
        return attention_output

    def _online_attention(self, query, key, value, attention_mask=None, training=None):
        assert self._fuse == 'none', 'Streaming attention requires fuse=none'
        softmax = self._softmax
        scores = ops.einsum(self._dot_product_equation, key, query)
        if attention_mask is None:
            keep = ops.ones((ops.shape(scores)[0], 1, 1, scores.shape[-1]), dtype='bool')
        else:
            keep = ops.expand_dims(ops.cast(attention_mask, 'bool'), axis=1)
        # Excluded pairs must not inflate the calibrated score/table ranges.
        scores = softmax.iq(ops.where(keep, scores, 0.0), training=training)
        first = ops.argmax(ops.cast(keep, 'int32'), axis=-1)
        m = ops.take_along_axis(scores, ops.expand_dims(first, -1), axis=-1)
        context = ops.zeros((*ops.shape(m)[:-1], self._value_dim), dtype=scores.dtype)
        xs = (ops.moveaxis(scores, -1, 0), ops.moveaxis(value, 1, 0), ops.moveaxis(keep, -1, 0))

        if softmax.impl == '2pass':

            def statistics(carry, xs):
                maximum, weight = carry
                score, _, valid = xs
                score, valid = ops.expand_dims(score, -1), ops.expand_dims(valid, -1)
                next_max = ops.where(valid, ops.maximum(maximum, score), maximum)
                rescale = softmax.exp_table(next_max - maximum, training=training)
                difference = ops.where(valid, next_max - score, 0.0)
                share = softmax.exp_table(difference, training=training)
                next_weight = softmax.lq(ops.where(valid, rescale * weight, 0.0), training=training) + share
                return next_max, ops.where(valid, next_weight, weight)

            maximum, weight = scan_rounds(
                statistics,
                (m, ops.zeros_like(m)),
                xs,
                [*softmax.exp_table.variables, *softmax.lq.variables],
            )

            def accumulate(context, xs):
                score, served, valid = xs
                score, valid = ops.expand_dims(score, -1), ops.expand_dims(valid, -1)
                difference = ops.where(valid, maximum - score, 0.0)
                share = softmax.exp_table(difference, training=training)
                probability = softmax.oq(
                    ops.where(valid, share * softmax.inv_table(weight, training=training), 0.0),
                    training=training,
                )
                next_context = softmax.aq(ops.where(valid, context, 0.0), training=training) + softmax.aq(
                    probability * ops.expand_dims(served, -2), training=training
                )
                return ops.where(valid, next_context, context)

            context = scan_rounds(
                accumulate,
                context,
                xs,
                [
                    *softmax.exp_table.variables,
                    *softmax.inv_table.variables,
                    *softmax.oq.variables,
                    *softmax.aq.variables,
                ],
            )
            return ops.transpose(context, (0, 2, 1, 3)), None

        def round_(carry, xs):
            maximum, weight, context = carry
            score, served, valid = xs
            score = ops.expand_dims(score, -1)
            valid = ops.expand_dims(valid, -1)
            next_max = ops.where(valid, ops.maximum(maximum, score), maximum)
            rescale = softmax.exp_table(next_max - maximum, training=training)
            difference = ops.where(valid, next_max - score, 0.0)
            share = softmax.exp_table(difference, training=training)
            share = ops.where(valid, share, 0.0)
            next_weight = softmax.lq(rescale * weight, training=training) + share
            served = ops.where(valid, ops.expand_dims(served, -2), 0.0)
            next_context = softmax.aq(rescale * context, training=training) + softmax.aq(share * served, training=training)
            return (
                next_max,
                ops.where(valid, next_weight, weight),
                ops.where(valid, next_context, context),
            )

        _, weight, context = scan_rounds(
            round_,
            (m, ops.zeros_like(m), context),
            xs,
            [*softmax.exp_table.variables, *softmax.lq.variables, *softmax.aq.variables],
        )
        context = context * softmax.inv_table(weight, training=training)
        return ops.transpose(context, (0, 2, 1, 3)), None

    def _compute_attention(self, query, key, value, attention_mask=None, training=None):  # type: ignore
        # Original _compute_attention in keras 3.5.0
        # Copied for disable to flash-attn that breaks quantization.
        """Applies Dot-product attention with query, key, value tensors.

        This function defines the computation inside `call` with projected
        multi-head Q, K, V inputs. Users can override this function for
        customized attention implementation.

        Parameters
        ----------
        query : tensor
            Projected query tensor of shape `(B, T, N, key_dim)`.
        key : tensor
            Projected key tensor of shape `(B, S, N, key_dim)`.
        value : tensor
            Projected value tensor of shape `(B, S, N, value_dim)`.
        attention_mask : tensor, optional
            A boolean mask of shape `(B, T, S)` that prevents attention to
            certain positions. It is generally not needed if the `query` and
            `value` (and/or `key`) are masked.
        training : bool, optional
            Python boolean indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (doing nothing).

        Returns
        -------
        attention_output : tensor
            Multi-headed outputs of attention computation.
        attention_scores : tensor
            Multi-headed attention weights.
        """
        if self._softmax_kind != 'comb':
            return self._online_attention(query, key, value, attention_mask, training)

        # Note: Applying scalar multiply at the smaller end of einsum improves
        # XLA performance, but may introduce slight numeric differences in
        # the Transformer attention head.
        query = ops.multiply(query, ops.cast(self._inverse_sqrt_key_dim, query.dtype))

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        attention_scores = ops.einsum(self._dot_product_equation, key, query)

        attention_scores = self._masked_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        if self.dropout:
            final_attn_scores = self._dropout_layer(attention_scores, training=training)
        else:
            final_attn_scores = attention_scores

        # `context_layer` = [B, T, N, H]
        attention_output = ops.einsum(self._combine_equation, final_attn_scores, value)
        return attention_output, attention_scores


@register_keras_serializable(package='hgq')
class QMultiHeadAttentionT(QMultiHeadAttention):
    """Table-projection multi-head attention.

    The attention score path is inherited from :class:`QMultiHeadAttention`.
    Query, key, value, and output projections are owned by :class:`QEinsumDenseT`
    table layers.
    """

    def __init__(
        self,
        num_heads,
        key_dim,
        value_dim=None,
        dropout=0.0,
        use_bias=True,
        output_shape=None,
        attention_axes=None,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        seed=None,
        qkvo_iq_conf: QuantizerConfig | None = None,
        qkvo_toq_conf: QuantizerConfig | None = None,
        qkvo_oq_conf: QuantizerConfig | None = None,
        softmax_iq_conf: QuantizerConfig | None = None,
        softmax_exp_iq_conf: QuantizerConfig | None = None,
        softmax_exp_oq_conf: QuantizerConfig | None = None,
        softmax_inv_iq_conf: QuantizerConfig | None = None,
        softmax_inv_oq_conf: QuantizerConfig | None = None,
        softmax_oq_conf: QuantizerConfig | None = None,
        stable_softmax=True,
        softmax_allow_heterogeneous_table: bool = False,
        parallelization_factor=-1,
        n_hl: int = 1,
        d_hl: int = 8,
        subnn_activation='tanh',
        table_spec: int | tuple[int, int] = (6, 5),
        batch_norm: bool = False,
        bn_center: bool = True,
        bn_scale: bool = True,
        bn_momentum: float = 0.99,
        bn_epsilon: float = 0.001,
        **kwargs,
    ):
        kwargs = gather_vars_to_kwargs(
            'self|qkvo_toq_conf|n_hl|d_hl|subnn_activation|table_spec|batch_norm|bn_.+',
        )
        self._qkvo_toq_conf = qkvo_toq_conf or QuantizerConfig(place='table')
        self._table_n_hl = n_hl
        self._table_d_hl = d_hl
        self._table_subnn_activation = keras.activations.get(subnn_activation)
        self._table_spec = (table_spec, table_spec) if isinstance(table_spec, int) else table_spec
        self._table_batch_norm = batch_norm
        self._table_bn_args = {
            'center': bn_center,
            'scale': bn_scale,
            'momentum': bn_momentum,
            'epsilon': bn_epsilon,
        }
        kwargs['fuse'] = 'none'
        super().__init__(**kwargs)

    def _make_table_dense(
        self,
        equation: str,
        output_shape,
        bias_axes,
        name: str,
        *,
        enable_iq: bool,
        enable_oq: bool,
    ):
        return QEinsumDenseT(
            equation=equation,
            output_shape=output_shape,
            bias_axes=bias_axes,
            n_hl=self._table_n_hl,
            d_hl=self._table_d_hl,
            subnn_activation=self._table_subnn_activation,
            toq_conf=self._qkvo_toq_conf,
            parallelization_factor=self.parallelization_factor,
            batch_norm=self._table_batch_norm,
            bn_center=self._table_bn_args['center'],
            bn_scale=self._table_bn_args['scale'],
            bn_momentum=self._table_bn_args['momentum'],
            bn_epsilon=self._table_bn_args['epsilon'],
            table_idxs=self._table_spec,
            name=name,
            dtype=self.dtype_policy,
            enable_iq=enable_iq,
            iq_conf=self._qkvo_iq_conf,
            enable_oq=enable_oq,
            oq_conf=self._qkv_oq_conf if name in ('query', 'key', 'value') else self._qkvo_oq_conf,
            enable_ebops=self.enable_ebops,
            beta0=self._beta0.clone(),
        )

    def build(self, query_shape, value_shape=None, key_shape=None):
        value_shape = query_shape if value_shape is None else value_shape
        key_shape = value_shape if key_shape is None else key_shape

        if value_shape[1:-1] != key_shape[1:-1]:
            raise ValueError(
                'All dimensions of `value` and `key`, except the last one, '
                f'must be equal. Received: value_shape={value_shape} and '
                f'key_shape={key_shape}',
            )

        query_rank = len(query_shape)
        key_rank = len(key_shape)
        value_rank = len(value_shape)

        einsum_equation, bias_axes, output_rank = _build_proj_equation(query_rank - 1, bound_dims=1, output_dims=2)
        self._query_dense = self._make_table_dense(
            einsum_equation,
            _get_output_shape(output_rank - 1, [self._num_heads, self._key_dim], query_shape),
            bias_axes if self._use_bias else None,
            'query',
            enable_iq=self.enable_iq,
            enable_oq=True,
        )
        einsum_equation, bias_axes, output_rank = _build_proj_equation(key_rank - 1, bound_dims=1, output_dims=2)
        self._key_dense = self._make_table_dense(
            einsum_equation,
            _get_output_shape(
                output_rank - 1,
                [self._num_heads, self._key_dim],
                key_shape,
            ),
            None,
            'key',
            enable_iq=self.enable_iq,
            enable_oq=True,
        )
        einsum_equation, bias_axes, output_rank = _build_proj_equation(value_rank - 1, bound_dims=1, output_dims=2)
        self._value_dense = self._make_table_dense(
            einsum_equation,
            _get_output_shape(output_rank - 1, [self._num_heads, self._value_dim], value_shape),
            bias_axes if self._use_bias else None,
            'value',
            enable_iq=self.enable_iq,
            enable_oq=True,
        )
        self._value_dense.build(value_shape)
        self._query_dense.build(query_shape)
        self._key_dense.build(key_shape)

        self._build_attention(output_rank, (query_shape, value_shape, key_shape))

        self._output_dense = self._make_output_dense(query_shape, {}, 'attention_output')
        output_dense_input_shape = list(self._query_dense.compute_output_shape(query_shape))
        output_dense_input_shape[-1] = self._value_dim
        self._output_dense.build(tuple(output_dense_input_shape))

        if self.enable_ebops:
            self._beta = self.add_weight(name='beta', shape=(), initializer=self._beta0, trainable=False)
            self._ebops = self.add_weight(name='ebops', shape=(), initializer=Constant(0.0), trainable=False, dtype='uint32')
        else:
            self._beta = None
            self._ebops = None

        self._dot_product_ebops_equation = self._dot_product_equation.split('->', 1)[0] + '->'
        self._combine_ebops_equation = self._combine_equation.split('->', 1)[0] + '->'

        self.n_parallel = math.prod(query_shape[1:-1])
        if self.parallelization_factor < 0:
            self.parallelization_factor = self.n_parallel
        self.built = True

    def _make_output_dense(self, query_shape, common_kwargs, name=None):
        del common_kwargs
        query_rank = len(query_shape)
        if self._output_shape:
            if not isinstance(self._output_shape, Sized):
                output_shape = [self._output_shape]
            else:
                output_shape = self._output_shape
        else:
            output_shape = [query_shape[-1]]
        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            query_rank - 1, bound_dims=2, output_dims=len(output_shape)
        )
        return self._make_table_dense(
            einsum_equation,
            _get_output_shape(output_rank - 1, output_shape, query_shape),
            bias_axes if self._use_bias else None,
            name or 'attention_output',
            enable_iq=True,
            enable_oq=self.enable_oq,
        )

    def get_config(self):
        config = super().get_config()
        config.pop('qkvo_kq_conf', None)
        config.pop('qkvo_bq_conf', None)
        config.pop('fuse', None)
        config.update(
            {
                'qkvo_toq_conf': self._qkvo_toq_conf,
                'n_hl': self._table_n_hl,
                'd_hl': self._table_d_hl,
                'subnn_activation': self._table_subnn_activation,
                'table_spec': self._table_spec,
                'batch_norm': self._table_batch_norm,
                'bn_center': self._table_bn_args['center'],
                'bn_scale': self._table_bn_args['scale'],
                'bn_momentum': self._table_bn_args['momentum'],
                'bn_epsilon': self._table_bn_args['epsilon'],
            }
        )
        return config
