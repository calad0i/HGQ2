import math
from collections.abc import Sized
from typing import Literal

from keras import ops
from keras.initializers import Constant
from keras.layers import Dropout, MultiHeadAttention
from keras.src.layers.attention.multi_head_attention import _build_attention_equation, _build_proj_equation

from ..quantizer.config import QuantizerConfig
from ..utils.misc import gather_vars_to_kwargs
from .core.base import QLayerBase
from .core.einsum_dense import QEinsumDense
from .softmax import QSoftmax


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
        softmax_iq_conf: QuantizerConfig | None = None,
        softmax_exp_iq_conf: QuantizerConfig | None = None,
        softmax_exp_oq_conf: QuantizerConfig | None = None,
        softmax_inv_iq_conf: QuantizerConfig | None = None,
        softmax_inv_oq_conf: QuantizerConfig | None = None,
        softmax_oq_conf: QuantizerConfig | None = None,
        stable_softmax=True,
        softmax_allow_heterogeneous_table: bool = False,
        parallelization_factor=-1,
        **kwargs,
    ):
        kwargs = gather_vars_to_kwargs('self|.+q_conf')

        self._qkvo_iq_conf = qkvo_iq_conf or QuantizerConfig(place='datalane')
        self._qkvo_kq_conf = qkvo_kq_conf or QuantizerConfig(place='weight')
        self._qkvo_bq_conf = qkvo_bq_conf or QuantizerConfig(place='bias')
        self._qkvo_oq_conf = qkvo_oq_conf or QuantizerConfig(place='datalane')
        self._softmax_iq_conf = softmax_iq_conf or QuantizerConfig(place='datalane')
        self._softmax_exp_iq_conf = softmax_exp_iq_conf or QuantizerConfig(place='datalane')
        self._softmax_exp_oq_conf = softmax_exp_oq_conf or QuantizerConfig(place='table')
        self._softmax_inv_iq_conf = softmax_inv_iq_conf or QuantizerConfig(place='datalane')
        self._softmax_inv_oq_conf = softmax_inv_oq_conf or QuantizerConfig(place='table')
        self._softmax_oq_conf = softmax_oq_conf or QuantizerConfig(place='datalane')
        self._softmax_allow_heterogeneous_table = kwargs.pop('softmax_allow_heterogeneous_table')
        self.parallelization_factor = kwargs.pop('parallelization_factor')
        self._stable_softmax = kwargs.pop('stable_softmax')
        self._fuse = kwargs.pop('fuse', 'none').lower()

        super().__init__(**kwargs)

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

    def build(
        self,
        query_shape,
        value_shape,
        key_shape=None,
    ):
        """Builds layers and variables.

        Parameters
        ----------
        query_shape : tuple
            Shape of the `query` tensor.
        value_shape : tuple
            Shape of the `value` tensor.
        key_shape : tuple, optional
            Shape of the `key` tensor.
        """

        # Copied and modified from keras MultiHeadAttention, substituted
        # EinsumDense with QEinsumDense and added sequence length (shape) to its
        # output shape when initializing, if known.
        key_shape = value_shape if key_shape is None else key_shape

        # if query_shape[-1] != value_shape[-1]:
        #     raise ValueError(
        #         "The last dimension of `query_shape` and `value_shape` "
        #         f"must be equal, but are {query_shape[-1]}, {value_shape[-1]}. "
        #         "Received: query_shape={query_shape}, value_shape={value_shape}"
        #     )

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
            **self._get_common_kwargs_for_sublayer(),
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
            **self._get_common_kwargs_for_sublayer(),
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
            **self._get_common_kwargs_for_sublayer(),
        )
        self._value_dense.build(value_shape)

        if self._fuse == 'qkv':
            self._query_dense._iq = self._value_dense._iq
        if self._fuse in ('qkv', 'kv'):
            self._key_dense._iq = self._value_dense._iq

        self._query_dense.build(query_shape)
        self._query_dense._enable_iq = True
        self._key_dense.build(key_shape)
        self._key_dense._enable_iq = True

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

    def _build_attention(self, rank, shapes=None):
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
        else:
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
        _inverse_sqrt_key_dim = 1.0 / math.sqrt(float(self._key_dim))
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
            input_scaler=_inverse_sqrt_key_dim,
            enable_ebops=self.enable_ebops,
        )
        self._dropout_layer = Dropout(
            rate=self._dropout,
            dtype=self.dtype_policy,
            seed=self.seed,
        )
        self._inverse_sqrt_key_dim = 1.0
        # Build softmax and dropout layers if possible.
        if shapes is not None:
            q_shape, v_shape, _ = shapes
            attn_score_shape = (None, self._num_heads, *q_shape[1:-1], *v_shape[1:-1])
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
                'softmax_iq_conf': self._softmax_iq_conf,
                'softmax_exp_iq_conf': self._softmax_exp_iq_conf,
                'softmax_exp_oq_conf': self._softmax_exp_oq_conf,
                'softmax_inv_iq_conf': self._softmax_inv_iq_conf,
                'softmax_inv_oq_conf': self._softmax_inv_oq_conf,
                'softmax_oq_conf': self._softmax_oq_conf,
                'softmax_allow_heterogeneous_table': self._softmax_allow_heterogeneous_table,
                'parallelization_factor': self.parallelization_factor,
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

    def _compute_ebops(self, query_shape, value_shape, key_shape=None):
        Q_shape = (1,) + self._query_dense.full_output_shape[1:]
        K_shape = (1,) + self._key_dense.full_output_shape[1:]
        V_shape = (1,) + self._value_dense.full_output_shape[1:]
        attn_score_shape = (1, self._num_heads, *query_shape[1:-1], *value_shape[1:-1])

        # PF not supported for MHA for now.
        # if self.parallelization_factor > 0:
        #     assert len(query_shape) == 3, f'EBOPs computation with pf>0 is only supported for 3D tensors, but got {query_shape}.'
        #     b, *n, h, dk = Q_shape
        #     b, *n, h, dv = K_shape
        #     b, *n, h, dv = V_shape

        #     Q_shape = b, (1,) * len(n), h, dk
        #     K_shape = b, (1,) * len(n), h, dv
        #     V_shape = b, (1,) * len(n), h, dv
        #     attn_score_shape = b, self._num_heads, *(1,) * len(n) * 2

        bw_q = self._query_dense.oq.bits_(Q_shape)
        bw_k = self._key_dense.oq.bits_(K_shape)
        bw_v = self._value_dense.oq.bits_(V_shape)
        bw_attn = self._softmax.oq.bits_(attn_score_shape)

        ebops_qk = ops.einsum(self._dot_product_ebops_equation, bw_q, bw_k)
        ebops_av = ops.einsum(self._combine_ebops_equation, bw_attn, bw_v)
        ebops = ebops_qk + ebops_av  # type: ignore
        if self.parallelization_factor > 0:
            return ebops * self.parallelization_factor
        return ebops

    @property
    def ebops(self) -> int:
        if self._ebops is None:
            return 0
        ebops = sum(
            (  # type: ignore
                self._query_dense.ebops,
                self._key_dense.ebops,
                self._value_dense.ebops,
                self._softmax.ebops,
                self._output_dense.ebops,
                ops.convert_to_tensor(self._ebops),
            )
        )
        return round(ops.convert_to_numpy(ebops).item())  # type: ignore

    def call(
        self,
        query,
        value,
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
