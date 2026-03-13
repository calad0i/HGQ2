from math import ceil
from typing import Literal

from keras import ops

from ...quantizer.config import QuantizerConfig
from ...utils.misc import gather_vars_to_kwargs
from ..conv import QConv2D
from ..core.einsum_dense import QEinsumDense
from .linformer import QMultiHeadAttention


class QSALTAttention(QMultiHeadAttention):
    """Quantized version of SALT attention from "Spatially Aware Linear Transformer (SALT) for Particle Jet Tagging"
    (https://arxiv.org/abs/2510.23641). Adds a conv2d operation on the computed attnmask before applying it to the value tensor.
    """

    def __init__(
        self,
        num_heads,
        lin_kv_proj_dim,
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
        fuse: Literal['none'] = 'none',
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
        share_kv_proj=False,
        cluster_k_proj=False,
        cluster_v_proj=False,
        conv_size=3,
        separate_conv=False,
        **kwargs,
    ):
        if fuse != 'none':
            raise ValueError(f'Only fuse="none" can be used for QLinformerAttention, but got fuse="{fuse}".')
        kwargs = gather_vars_to_kwargs('self|lin_kv_proj_dim|share_kv_proj|conv_size|cluster_k_proj|cluster_v_proj|separate_conv')
        self._kv_proj_dim = lin_kv_proj_dim if isinstance(lin_kv_proj_dim, int) else tuple(lin_kv_proj_dim)
        self.share_kv_proj = share_kv_proj
        self.conv_size = conv_size
        self.cluster_k_proj = cluster_k_proj
        self.cluster_v_proj = cluster_v_proj
        self.separate_conv = separate_conv
        super().__init__(**kwargs)

    def build(self, query_shape, value_shape=None, key_shape=None):
        value_shape = value_shape or query_shape
        key_shape = key_shape or value_shape
        key_rank = len(key_shape)
        value_rank = len(value_shape)
        assert key_rank == value_rank == len(query_shape) == 3, (
            f'Only rank 3 inputs are supported for SALTAttn, but got query ({query_shape}), key ({key_shape}), and value ({value_shape}).'
        )
        assert key_rank == value_rank, (
            f'Key and value must have the same rank, but got key shape {key_shape} and value shape {value_shape}.'
        )
        if self.share_kv_proj:
            assert key_shape == value_shape, (
                f'When share_kv_proj is True, k and v must have the same shape. Got key ({key_shape}) and value ({value_shape}).'
            )
            assert self.cluster_k_proj == self.cluster_v_proj, (
                f'When share_kv_proj is True, cluster_k_proj and cluster_v_proj must be the same. Got cluster_k_proj={self.cluster_k_proj} and cluster_v_proj={self.cluster_v_proj}.'
            )

        if self._attention_axes is None:
            self._attention_axes = tuple(range(1, value_rank - 1))
        elif not isinstance(self._attention_axes, tuple):
            self._attention_axes = tuple(self._attention_axes)

        _value_shape_proj, _key_shape_proj = list(value_shape), list(key_shape)
        _value_shape_proj[1] = self._kv_proj_dim
        _key_shape_proj[1] = self._kv_proj_dim
        self._key_shape_proj = tuple(_key_shape_proj)
        self._value_shape_proj = tuple(_value_shape_proj)

        eq_lin_k_proj = 'bmnd,mn->bmd' if self.cluster_k_proj else 'bnd,nm->bmd'  # x is 1, rm'rd here
        eq_lin_v_proj = 'bmnd,mn->bmd' if self.cluster_v_proj else 'bnd,nm->bmd'

        # padding config
        self.n_k_pad = 0
        self.n_v_pad = 0
        if self.cluster_k_proj:
            self.n_k_pad = ceil(key_shape[1] / self._kv_proj_dim) * self._kv_proj_dim - key_shape[1]
            cluster_length = (key_shape[1] + self.n_k_pad) // self._kv_proj_dim
            key_shape = (key_shape[0], self._kv_proj_dim, cluster_length, key_shape[2])
            self._k_reshape_to = (-1,) + key_shape[1:]
        if self.cluster_v_proj:
            self.n_v_pad = ceil(value_shape[1] / self._kv_proj_dim) * self._kv_proj_dim - value_shape[1]
            cluster_length = (value_shape[1] + self.n_v_pad) // self._kv_proj_dim
            value_shape = (value_shape[0], self._kv_proj_dim, cluster_length, value_shape[2])
            self._v_reshape_to = (-1,) + value_shape[1:]

        self._lin_k_proj = QEinsumDense(
            eq_lin_k_proj, self._key_shape_proj[1:], bias_axes=None, **self._get_common_kwargs_for_sublayer()
        )
        self._lin_k_proj.build(key_shape)

        if not self.share_kv_proj:
            self._lin_v_proj = QEinsumDense(
                eq_lin_v_proj, self._value_shape_proj[1:], bias_axes=None, **self._get_common_kwargs_for_sublayer()
            )
            self._lin_v_proj.build(value_shape)
        else:
            self._lin_v_proj = self._lin_k_proj

        attn_score_shape = (query_shape[0], self.num_heads, query_shape[1], self._kv_proj_dim)
        print(attn_score_shape)
        if self.conv_size > 0:
            self.conv = QConv2D(
                filters=attn_score_shape[1],
                kernel_size=self.conv_size,
                padding='same',
                groups=self.num_heads if self.separate_conv else 1,
                **self._get_common_kwargs_for_sublayer(),
                data_format='channels_first',
            )
            self.conv.build(attn_score_shape)

        super().build(query_shape, self._value_shape_proj, key_shape=self._key_shape_proj)

    def _masked_softmax(self, attention_scores, attention_mask=None):
        if self.conv_size > 0:
            attention_scores = self.conv(attention_scores)
        attention_scores = super()._masked_softmax(attention_scores, attention_mask)
        return attention_scores

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
        assert use_causal_mask is False, 'Causal mask is not supported in QLinformerAttention.'
        value = value if value is not None else query
        key = key if key is not None else value

        if self.cluster_k_proj:
            key = ops.pad(key, [[0, 0], [0, self.n_k_pad], [0, 0]])
            key = ops.reshape(key, self._k_reshape_to)

        if self.n_v_pad > 0:
            value = ops.pad(value, [[0, 0], [0, self.n_v_pad], [0, 0]])
            value = ops.reshape(value, self._v_reshape_to)

        key = self._lin_k_proj(key, training=training)
        value = self._lin_v_proj(value, training=training)

        return super().call(
            query,
            value,
            key,
            query_mask=query_mask,
            value_mask=value_mask,
            key_mask=key_mask,
            attention_mask=attention_mask,
            return_attention_scores=return_attention_scores,
            training=training,
        )

    @property
    def ebops(self):
        if self._ebops is None:
            return ops.cast(0, 'uint32')
        ebops = sum(
            (  # type: ignore
                self._lin_k_proj.ebops,
                self._lin_v_proj.ebops,
                ops.convert_to_tensor(self._ebops),  # type: ignore
            )
        )
        if self.conv_size > 0:
            ebops += self.conv.ebops  # type: ignore
        return ebops  # type: ignore

    def _compute_ebops(self, query_shape, value_shape=None, key_shape=None):
        return super()._compute_ebops(query_shape, self._value_shape_proj, key_shape=self._key_shape_proj)

    def get_config(self):
        config = super().get_config()
        config['lin_kv_proj_dim'] = self._kv_proj_dim
        config['share_kv_proj'] = self.share_kv_proj
        config['conv_size'] = self.conv_size
        config['cluster_k_proj'] = self.cluster_k_proj
        config['cluster_v_proj'] = self.cluster_v_proj
        config['separate_conv'] = self.separate_conv
        return config
