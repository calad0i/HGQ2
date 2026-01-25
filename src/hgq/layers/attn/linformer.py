from typing import Literal

from keras import ops

from ...quantizer.config import QuantizerConfig
from ...utils.misc import gather_vars_to_kwargs
from ..core.einsum_dense import QEinsumDense
from .mha import QMultiHeadAttention


class QLinformerAttention(QMultiHeadAttention):
    __output_quantizer_handled__ = True

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
        kwargs = gather_vars_to_kwargs('self|lin_kv_proj_dim')
        self._kv_proj_dim = (lin_kv_proj_dim,) if isinstance(lin_kv_proj_dim, int) else tuple(lin_kv_proj_dim)
        super().__init__(**kwargs)

    def build(self, query_shape, value_shape, key_shape=None):
        key_shape = key_shape or value_shape
        key_rank = len(key_shape)
        value_rank = len(value_shape)
        assert key_rank == value_rank, (
            f'Key and value must have the same rank, but got key shape {key_shape} and value shape {value_shape}.'
        )
        if self.attention_axes is not None:
            attn_axes = tuple(self.attention_axes)
        else:
            attn_axes = tuple(range(1, value_rank - 1))

        assert len(attn_axes) == len(self._kv_proj_dim), (
            f'Attention axes are {attn_axes}, but kv_proj_dim is {self._kv_proj_dim}. They must match in length.'
        )

        _value_shape_proj, _key_shape_proj = list(value_shape), list(key_shape)
        for i, j in enumerate(attn_axes):
            _value_shape_proj[j] = self._kv_proj_dim[i]
            _key_shape_proj[j] = self._kv_proj_dim[i]
        self._key_shape_proj = tuple(_key_shape_proj)
        self._value_shape_proj = tuple(_value_shape_proj)

        template = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        inp_idx = template[:value_rank]
        attn_idx = ''.join(inp_idx[i] for i in attn_axes)
        new_attn_idx = template[value_rank : value_rank + len(self._kv_proj_dim)]
        ker_idx = attn_idx + new_attn_idx
        _out_idx = list(inp_idx)
        for i, j in enumerate(attn_axes):
            _out_idx[j] = new_attn_idx[i]
        out_idx = ''.join(_out_idx)
        eq_lin_kv_proj = f'{inp_idx},{ker_idx}->{out_idx}'

        self._lin_k_proj = QEinsumDense(
            eq_lin_kv_proj, self._key_shape_proj[1:], bias_axes=None, **self._get_common_kwargs_for_sublayer()
        )

        self._lin_v_proj = QEinsumDense(
            eq_lin_kv_proj, self._value_shape_proj[1:], bias_axes=None, **self._get_common_kwargs_for_sublayer()
        )

        self._lin_k_proj.build(key_shape)
        self._lin_v_proj.build(value_shape)

        super().build(query_shape, self._value_shape_proj, key_shape=self._key_shape_proj)

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
        assert use_causal_mask is False, 'Causal mask is not supported in QLinformerAttention.'
        key = key if key is not None else value
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
                self._lin_k_proj.ebops,
                self._lin_v_proj.ebops,
                ops.convert_to_tensor(self._ebops),
            )
        )
        return round(ops.convert_to_numpy(ebops).item())  # type: ignore

    def _compute_ebops(self, query_shape, value_shape, key_shape=None):
        return super()._compute_ebops(query_shape, self._value_shape_proj, key_shape=self._key_shape_proj)

    def get_config(self):
        config = super().get_config()
        config['lin_kv_proj_dim'] = self._kv_proj_dim
        return config
