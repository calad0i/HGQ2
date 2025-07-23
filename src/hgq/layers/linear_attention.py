import math
from typing import Literal
import warnings

from ..quantizer.config import QuantizerConfig
from ..utils.misc import gather_vars_to_kwargs
from .core.base import QLayerBase
from .core.einsum_dense import QEinsumDense
from .softmax import QSoftmax

import keras
from keras import ops, regularizers, constraints, initializers
from keras.initializers import Constant
from keras.layers import Dropout, EinsumDense




class LinearAttention(keras.layers.Layer):
    """
    Linformer Attention
    """
    def __init__(
        self,
        num_heads,
        key_dim,
        proj_k,
        value_dim=None,
        dropout=0.0,
        use_bias=True,
        param_sharing='none',
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.proj_k = proj_k
        self.value_dim = value_dim if value_dim is not None else key_dim
        self.dropout = dropout
        self.use_bias = use_bias
        self.param_sharing = param_sharing
        self.seed = seed
        self.supports_masking = False
        
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.embed_dim = self.key_dim * self.num_heads
        self.scaling = float(self.key_dim)**-0.5

    def build(self, query_shape, value_shape, key_shape=None):
        key_shape = value_shape if key_shape is None else key_shape

        common_kwargs = dict(
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
        )

        qkv_equation = "bsd,de->bse"
        self._query_dense = EinsumDense(qkv_equation, output_shape=(None, self.embed_dim), bias_axes="e" if self.use_bias else None, name="query", **common_kwargs)
        self._key_dense = EinsumDense(qkv_equation, output_shape=(None, self.embed_dim), bias_axes="e" if self.use_bias else None, name="key", **common_kwargs)
        self._value_dense = EinsumDense(qkv_equation, output_shape=(None, self.value_dim * self.num_heads), bias_axes="e" if self.use_bias else None, name="value", **common_kwargs)
        self._output_dense = EinsumDense(qkv_equation, output_shape=(None, self.embed_dim), bias_axes="e" if self.use_bias else None, name="output", **common_kwargs)

        linformer_equation = "bes,sk->bek"
        self._e_proj = EinsumDense(linformer_equation, output_shape=(self.embed_dim, self.proj_k), bias_axes="k" if self.use_bias else None, name='e_projection', **common_kwargs)
        if self.param_sharing == 'layerwise':
            self._f_proj = self._e_proj
        else:
            self._f_proj = EinsumDense(linformer_equation, output_shape=(self.value_dim * self.num_heads, self.proj_k), bias_axes="k" if self.use_bias else None, name='f_projection', **common_kwargs)

        if self.dropout > 0.0:
            self._dropout_layer = Dropout(self.dropout, seed=self.seed)
            
        self.built = True

    def _project_sequence(self, x, proj_layer):
        x_T = ops.transpose(x, axes=[0, 2, 1])
        x_proj_T = proj_layer(x_T)
        return ops.transpose(x_proj_T, axes=[0, 2, 1])

    def _compute_attention(self, query, key, value, training=None):
        batch_size = ops.shape(query)[0]
        tgt_len = ops.shape(query)[1]

        query = ops.transpose(ops.reshape(query, (batch_size, tgt_len, self.num_heads, self.key_dim)), axes=[0, 2, 1, 3])
        key = ops.transpose(ops.reshape(key, (batch_size, -1, self.num_heads, self.key_dim)), axes=[0, 2, 1, 3])
        value = ops.transpose(ops.reshape(value, (batch_size, -1, self.num_heads, self.value_dim)), axes=[0, 2, 1, 3])

        query *= self.scaling
        
        key_transposed = ops.transpose(key, axes=[0, 1, 3, 2])
        attention_scores = ops.matmul(query, key_transposed)
        
        attention_scores = keras.activations.softmax(attention_scores, axis=-1)
        
        if self.dropout > 0.0:
            attention_scores = self._dropout_layer(attention_scores, training=training)
        
        attention_output = ops.matmul(attention_scores, value)

        attention_output = ops.transpose(attention_output, axes=[0, 2, 1, 3])
        attention_output = ops.reshape(attention_output, (batch_size, tgt_len, self.num_heads * self.value_dim))
        return attention_output, attention_scores

    def call(self, query, value, key=None, attention_mask=None, return_attention_scores=False, training=None, use_causal_mask=False):
        if key is None: key = value
        if not self.built:
             self.build(query.shape, value.shape, None if key is value else key.shape)
             
        if attention_mask is not None or use_causal_mask:
            warnings.warn(f"{self.__class__.__name__} does not support masking.", UserWarning)

        query = self._query_dense(query)
        key = self._key_dense(key)
        value = self._value_dense(value)

        key = self._project_sequence(key, self._e_proj)
        value = self._project_sequence(value, self._f_proj)

        attention_output, attention_scores = self._compute_attention(query, key, value, training=training)
        attention_output = self._output_dense(attention_output)

        if return_attention_scores:
            return attention_output, attention_scores
        return attention_output

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "proj_k": self.proj_k,
            "value_dim": self.value_dim,
            "dropout": self.dropout,
            "use_bias": self.use_bias,
            "param_sharing": self.param_sharing,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "seed": self.seed,
        })
        return config





class QLinearAttention(LinearAttention, QLayerBase):
    __output_quantizer_handled__ = True

    def __init__(
        self,
        num_heads,
        key_dim,
        proj_k,
        value_dim=None,
        dropout=0.0,
        use_bias=True,
        param_sharing='none',
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        seed=None,
        fuse: Literal['none', 'qkv', 'kv'] = 'none',
        qkvo_iq_conf: QuantizerConfig | None = None, qkvo_kq_conf: QuantizerConfig | None = None,
        qkvo_bq_conf: QuantizerConfig | None = None, qkvo_oq_conf: QuantizerConfig | None = None,
        ef_iq_conf: QuantizerConfig | None = None, ef_kq_conf: QuantizerConfig | None = None,
        ef_bq_conf: QuantizerConfig | None = None, ef_oq_conf: QuantizerConfig | None = None,
        softmax_iq_conf: QuantizerConfig | None = None, softmax_exp_iq_conf: QuantizerConfig | None = None,
        softmax_exp_oq_conf: QuantizerConfig | None = None, softmax_inv_iq_conf: QuantizerConfig | None = None,
        softmax_inv_oq_conf: QuantizerConfig | None = None, softmax_oq_conf: QuantizerConfig | None = None,
        stable_softmax=True, softmax_allow_heterogeneous_table: bool = False,
        parallelization_factor=-1,
        **kwargs,
    ):
        super().__init__(
            num_heads=num_heads, key_dim=key_dim, proj_k=proj_k,
            value_dim=value_dim, dropout=dropout, use_bias=use_bias,
            param_sharing=param_sharing, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
            seed=seed, **kwargs
        )
        QLayerBase.__init__(self, **kwargs)

        self._qkvo_iq_conf = qkvo_iq_conf or QuantizerConfig(place='datalane')
        self._qkvo_kq_conf = qkvo_kq_conf or QuantizerConfig(place='weight')
        self._qkvo_bq_conf = qkvo_bq_conf or QuantizerConfig(place='bias')
        self._qkvo_oq_conf = qkvo_oq_conf or QuantizerConfig(place='datalane')
        self._ef_iq_conf = ef_iq_conf or QuantizerConfig(place='datalane')
        self._ef_kq_conf = ef_kq_conf or QuantizerConfig(place='weight')
        self._ef_bq_conf = ef_bq_conf or QuantizerConfig(place='bias')
        self._ef_oq_conf = ef_oq_conf or QuantizerConfig(place='datalane')
        self._softmax_iq_conf = softmax_iq_conf or QuantizerConfig(place='datalane')
        self._softmax_exp_iq_conf = softmax_exp_iq_conf or QuantizerConfig(place='datalane')
        self._softmax_exp_oq_conf = softmax_exp_oq_conf or QuantizerConfig(place='table')
        self._softmax_inv_iq_conf = softmax_inv_iq_conf or QuantizerConfig(place='datalane')
        self._softmax_inv_oq_conf = softmax_inv_oq_conf or QuantizerConfig(place='table')
        self._softmax_oq_conf = softmax_oq_conf or QuantizerConfig(place='datalane')
        self._softmax_allow_heterogeneous_table = softmax_allow_heterogeneous_table
        self.parallelization_factor = parallelization_factor
        self._stable_softmax = stable_softmax
        self._fuse = fuse.lower()

    def _get_common_dense_kwargs(self):
        return {
            "kernel_initializer": self.kernel_initializer, "bias_initializer": self.bias_initializer,
            "kernel_regularizer": self.kernel_regularizer, "bias_regularizer": self.bias_regularizer,
            "kernel_constraint": self.kernel_constraint, "bias_constraint": self.bias_constraint,
        }

    def _get_qkvo_kwargs(self):
        kwargs = {
            'iq_conf': self._qkvo_iq_conf, 'kq_conf': self._qkvo_kq_conf,
            'bq_conf': self._qkvo_bq_conf, 'oq_conf': self._qkvo_oq_conf,
            'enable_ebops': self.enable_ebops, 'beta0': self._beta0.clone(),
            'parallelization_factor': self.parallelization_factor,
        }
        kwargs.update(self._get_common_dense_kwargs())
        return kwargs

    def _get_ef_kwargs(self):
        kwargs = {
            'iq_conf': self._ef_iq_conf, 'kq_conf': self._ef_kq_conf,
            'bq_conf': self._ef_bq_conf, 'oq_conf': self._ef_oq_conf,
            'enable_ebops': self.enable_ebops, 'beta0': self._beta0.clone(),
            'parallelization_factor': self.parallelization_factor,
        }
        kwargs.update(self._get_common_dense_kwargs())
        return kwargs

    def build(self, query_shape, value_shape, key_shape=None):
        key_shape = value_shape if key_shape is None else key_shape
        
        sequence_length = key_shape[1]
        if sequence_length is None:
            raise ValueError(f"The sequence dimension of the input to {self.name} must be fixed (not None).")

        qkv_equation = "bsd,de->bse"
        self._query_dense = QEinsumDense(qkv_equation, output_shape=(None, self.embed_dim), bias_axes="e" if self.use_bias else None, name="query", **self._get_qkvo_kwargs())
        self._key_dense = QEinsumDense(qkv_equation, output_shape=(None, self.embed_dim), bias_axes="e" if self.use_bias else None, name="key", **self._get_qkvo_kwargs())
        self._value_dense = QEinsumDense(qkv_equation, output_shape=(None, self.value_dim * self.num_heads), bias_axes="e" if self.use_bias else None, name="value", **self._get_qkvo_kwargs())
        self._output_dense = QEinsumDense(qkv_equation, output_shape=(None, self.embed_dim), bias_axes="e" if self.use_bias else None, name="output", **self._get_qkvo_kwargs())

        linformer_equation = "bes,sk->bek"
        self._e_proj = QEinsumDense(linformer_equation, output_shape=(self.embed_dim, self.proj_k), bias_axes="k" if self.use_bias else None, name='e_projection', **self._get_ef_kwargs())
        if self.param_sharing == 'layerwise': self._f_proj = self._e_proj
        else: self._f_proj = QEinsumDense(linformer_equation, output_shape=(self.value_dim * self.num_heads, self.proj_k), bias_axes="k" if self.use_bias else None, name='f_projection', **self._get_ef_kwargs())

        self._query_dense.build(query_shape)
        self._key_dense.build(key_shape)
        self._value_dense.build(value_shape)
        
        key_proj_shape = (key_shape[0], key_shape[1], self.embed_dim)
        value_proj_shape = (value_shape[0], value_shape[1], self.value_dim * self.num_heads)
        
        self._e_proj.build((key_proj_shape[0], key_proj_shape[2], key_proj_shape[1]))
        if self.param_sharing != 'layerwise':
            self._f_proj.build((value_proj_shape[0], value_proj_shape[2], value_proj_shape[1]))
        
        self._build_attention(4, (query_shape, key_shape, value_shape))
        
        query_proj_shape = (query_shape[0], query_shape[1], self.embed_dim)
        output_dense_input_shape = list(query_proj_shape)
        output_dense_input_shape[-1] = self.num_heads * self.value_dim
        self._output_dense.build(tuple(output_dense_input_shape))
        
        if self.enable_ebops:
            self._beta = self.add_weight(name='beta', shape=(), initializer=self._beta0, trainable=False)
            self._ebops = self.add_weight(name='ebops', shape=(), initializer=Constant(0.0), trainable=False, dtype='uint32')
        else: self._beta, self._ebops = None, None
            
        if hasattr(self, "enable_iq") and self.enable_iq:
            self._iq = self._query_dense._iq

        self.built = True

    def _build_attention(self, rank, shapes=None):
        self._softmax = QSoftmax(
            axis=-1, dtype=self.dtype_policy, stable=self._stable_softmax,
            iq_conf=self._softmax_iq_conf, exp_iq_conf=self._softmax_exp_iq_conf,
            exp_oq_conf=self._softmax_exp_oq_conf, inv_iq_conf=self._softmax_inv_iq_conf,
            inv_oq_conf=self._softmax_inv_oq_conf, oq_conf=self._softmax_oq_conf,
            allow_heterogeneous_table=self._softmax_allow_heterogeneous_table,
            enable_ebops=self.enable_ebops,
        )
        
        if self.dropout > 0.0:
            self._dropout_layer = Dropout(rate=self.dropout, seed=self.seed, dtype=self.dtype_policy)
        
        if shapes is not None:
            query_shape, _, _ = shapes
            attn_score_shape = (query_shape[0], self.num_heads, query_shape[1], self.proj_k)
            self._softmax.build(attn_score_shape)
            if self.dropout > 0.0:
                self._dropout_layer.build(attn_score_shape)

    @property
    def ebops(self):
        if not hasattr(self, 'enable_ebops') or not self.enable_ebops: return ops.cast(0, 'uint32')
        total_ebops = sum((
            self._query_dense.ebops, self._key_dense.ebops, self._value_dense.ebops,
            self._e_proj.ebops, self._f_proj.ebops, self._softmax.ebops,
            self._output_dense.ebops, ops.convert_to_tensor(getattr(self, '_ebops', 0)),
        ))
        return round(ops.convert_to_numpy(total_ebops).item())

    def call(self, query, value, key=None, attention_mask=None, return_attention_scores=False, training=None, use_causal_mask=False):
        if key is None: key = value
        if not self.built:
            self.build(query.shape, value.shape, None if key is value else key.shape)

        if attention_mask is not None or use_causal_mask:
            warnings.warn(f"{self.__class__.__name__} does not support masking.", UserWarning)
            attention_mask = None 

        query = self._query_dense(query)
        key = self._key_dense(key)
        value = self._value_dense(value)

        key = self._project_sequence(key, self._e_proj)
        value = self._project_sequence(value, self._f_proj)

        attention_output, attention_scores = self._compute_attention(query, key, value, attention_mask=None, training=training)
        attention_output = self._output_dense(attention_output)

        if hasattr(self, 'enable_oq') and self.enable_oq and hasattr(self, 'oq'): attention_output = self.oq(attention_output, training=training)
        return (attention_output, attention_scores) if return_attention_scores else attention_output

    def _compute_attention(self, query, key, value, attention_mask=None, training=None):
        batch_size = ops.shape(query)[0]
        tgt_len = ops.shape(query)[1]

        query = ops.transpose(ops.reshape(query, (batch_size, tgt_len, self.num_heads, self.key_dim)), axes=[0, 2, 1, 3])
        key = ops.transpose(ops.reshape(key, (batch_size, -1, self.num_heads, self.key_dim)), axes=[0, 2, 1, 3])
        value = ops.transpose(ops.reshape(value, (batch_size, -1, self.num_heads, self.value_dim)), axes=[0, 2, 1, 3])

        query = ops.multiply(query, ops.cast(1.0 / math.sqrt(float(self.key_dim)), query.dtype))

        key_transposed = ops.transpose(key, axes=[0, 1, 3, 2])
        attention_scores = ops.matmul(query, key_transposed)

        if hasattr(self, '_masked_softmax'):
            attention_scores = self._masked_softmax(attention_scores, attention_mask)
        else:
            attention_scores = self._softmax(attention_scores)
            
        if self.dropout > 0.0:
            attention_scores = self._dropout_layer(attention_scores, training=training)

        attention_output = ops.matmul(attention_scores, value)
        
        attention_output = ops.transpose(attention_output, axes=[0, 2, 1, 3])
        attention_output = ops.reshape(attention_output, (batch_size, tgt_len, self.num_heads * self.value_dim))

        return attention_output, attention_scores

    def get_config(self):
        config = super().get_config()
        config.update({
            'fuse': self._fuse,
            'qkvo_iq_conf': self._qkvo_iq_conf, 'kq_conf': self._qkvo_kq_conf,
            'bq_conf': self._qkvo_bq_conf, 'oq_conf': self._qkvo_oq_conf,
            'ef_iq_conf': self._ef_iq_conf, 'ef_kq_conf': self._ef_kq_conf,
            'ef_bq_conf': self._ef_bq_conf, 'ef_oq_conf': self._ef_oq_conf,
            'softmax_iq_conf': self._softmax_iq_conf, 'softmax_exp_iq_conf': self._softmax_exp_iq_conf,
            'softmax_exp_oq_conf': self._softmax_exp_oq_conf, 'softmax_inv_iq_conf': self._softmax_inv_iq_conf,
            'softmax_inv_oq_conf': self._softmax_inv_oq_conf, 'softmax_oq_conf': self._softmax_oq_conf,
            'stable_softmax': self._stable_softmax,
            'softmax_allow_heterogeneous_table': self._softmax_allow_heterogeneous_table,
            'parallelization_factor': self.parallelization_factor,
        })
        return config
