import math
from collections.abc import Callable, Sequence
from typing import TypedDict

from keras import ops

from ...quantizer.config import QuantizerConfig
from ...utils.misc import gather_vars_to_kwargs
from ..core.base import QLayerBaseSingleInput
from ..core.einsum_dense import QEinsumDense
from .base import QLIF, csd_count


class LIFConfig(TypedDict, total=False):
    beta: float | Sequence[float]
    threshold: float | Sequence[float]
    spike_grad: Callable | None
    surrogate_disable: bool
    learn_beta: bool
    learn_threshold: bool
    reset_mechanism: str
    detach_reset: bool
    graded_spikes_factor: float
    learn_graded_spikes_factor: bool
    unroll: bool
    iq_conf: QuantizerConfig
    sq_conf: QuantizerConfig
    beta_q_conf: QuantizerConfig
    graded_spikes_factor_q_conf: QuantizerConfig  # only with learn_graded_spikes_factor
    enable_iq: bool
    enable_sq: bool
    parallelization_factor: int


class QLIFMHA(QLayerBaseSingleInput):
    """Quantized spiking self-attention in the style of Spikformer (Zhou et al., ICLR 2023).

    Parameters
    ----------
    num_heads : int
        Number of attention heads N.
    key_dim : int
        Size H of each query and key head.
    value_dim : int, optional
        Size Hv of each value head. Defaults to ``key_dim``.
    scale : float, optional
        Context scaling, must be a power of two, so that it is a shift in fixed point. Defaults to
        ``2**-floor(log2(key_dim) / 2 + 0.5)``, (pow2 of ``1 / sqrt(key_dim)``).
    use_bias : bool, optional
        Whether the projections use biases. Default True.
    lif_config : LIFConfig, optional
        Arguments for the internal ``QLIF`` layers
    qkvo_kq_conf : QuantizerConfig, optional
        Kernel quantizer config of the four projections.
    qkvo_bq_conf : QuantizerConfig, optional
        Bias quantizer config of the four projections.
    out_proj_iq_conf : QuantizerConfig, optional
        Input quantizer config of output dense.
    """

    def __init__(
        self,
        num_heads: int,
        key_dim: int,
        value_dim: int | None = None,
        scale: float | None = None,
        use_bias: bool = True,
        lif_config: LIFConfig | None = None,
        iq_conf: QuantizerConfig | None = None,
        qkvo_kq_conf: QuantizerConfig | None = None,
        qkvo_bq_conf: QuantizerConfig | None = None,
        out_proj_iq_conf: QuantizerConfig | None = None,
        oq_conf: QuantizerConfig | None = None,
        enable_iq: bool | None = None,
        enable_oq: bool | None = None,
        enable_ebops: bool | None = None,
        beta0: float | None = None,
        **kwargs,
    ):
        if scale is None:
            scale = 2.0 ** -math.floor(math.log2(key_dim) / 2 + 0.5)
        if not (scale > 0 and 2.0 ** round(math.log2(scale)) == scale):
            raise ValueError(f'scale must be a power of two to be exact in fixed point, got {scale}.')
        kwargs = gather_vars_to_kwargs('self|num_heads|key_dim|value_dim|scale|use_bias|lif_config|qkvo_.+|out_proj_iq_conf')
        self._num_heads = num_heads
        self._key_dim = key_dim
        self._value_dim = key_dim if value_dim is None else value_dim
        self._scale = scale
        self._use_bias = use_bias
        self._lif_config = {
            'iq_conf': QuantizerConfig('default', 'datalane'),
            'sq_conf': QuantizerConfig(place='datalane'),
            'beta_q_conf': QuantizerConfig('default', 'weight'),
            **(lif_config or {}),
        }
        self._qkvo_kq_conf = qkvo_kq_conf or QuantizerConfig(place='weight')
        self._qkvo_bq_conf = qkvo_bq_conf or QuantizerConfig(place='bias')
        self._out_proj_iq_conf = out_proj_iq_conf or QuantizerConfig(place='datalane')
        super().__init__(**kwargs)

        self._query_lif = self._make_lif('query_lif', num_heads * key_dim)
        self._key_lif = self._make_lif('key_lif', num_heads * key_dim)
        self._value_lif = self._make_lif('value_lif', num_heads * self._value_dim)

    @property
    def num_heads(self):
        return self._num_heads

    @property
    def key_dim(self):
        return self._key_dim

    @property
    def value_dim(self):
        return self._value_dim

    @property
    def scale(self):
        return self._scale

    def _sublayer_kwargs(self):
        return {'enable_ebops': self.enable_ebops, 'beta0': self._beta0.clone(), 'dtype': self.dtype_policy}

    def _make_projection(self, name: str, equation: str, output_shape, bias_axes: str, enable_iq: bool, iq_conf=None):
        return QEinsumDense(
            equation,
            output_shape,
            bias_axes=bias_axes if self._use_bias else None,
            iq_conf=iq_conf,
            kq_conf=self._qkvo_kq_conf,
            bq_conf=self._qkvo_bq_conf,
            enable_iq=enable_iq,
            enable_oq=False,
            name=name,
            **self._sublayer_kwargs(),
        )

    def _make_lif(self, name: str, units: int):
        return QLIF(units, return_sequences=True, enable_oq=False, name=name, **self._sublayer_kwargs(), **self._lif_config)

    def build(self, input_shape):
        super().build(input_shape)
        _, seq, features = input_shape
        heads, key_dim, value_dim = self._num_heads, self._key_dim, self._value_dim

        # The projections share the layer's input quantizer, and each QLIF's input quantizer quantizes its projection.
        self._query_dense = self._make_projection('query', 'abc,cde->abde', (seq, heads, key_dim), 'de', enable_iq=False)
        self._key_dense = self._make_projection('key', 'abc,cde->abde', (seq, heads, key_dim), 'de', enable_iq=False)
        self._value_dense = self._make_projection('value', 'abc,cde->abde', (seq, heads, value_dim), 'de', enable_iq=False)
        for dense in (self._query_dense, self._key_dense, self._value_dense):
            if self.enable_iq:
                dense._iq = self.iq
                dense._enable_iq = True
            dense.build(input_shape)

        for lif in (self._query_lif, self._key_lif, self._value_lif):
            lif.build((None, seq, lif.cell.units))

        # QLayerBase applies oq after call, so the output projection has no output quantizer of its own.
        self._output_dense = self._make_projection(
            'attention_output', 'abcd,cde->abe', (seq, features), 'e', enable_iq=True, iq_conf=self._out_proj_iq_conf
        )
        self._output_dense.build((None, seq, heads, value_dim))

    def compute_output_shape(self, input_shape):
        return input_shape

    def _spike_train(self, dense: QEinsumDense, lif: QLIF, inputs, training=None):
        """Project the input to ``(B, S, N, H)`` and fire a QLIF over S, with the heads flattened into its features."""
        projected = dense(inputs, training=training)
        _, seq, heads, dim = projected.shape
        spikes = lif(ops.reshape(projected, (-1, seq, heads * dim)), training=training)
        return ops.reshape(spikes, (-1, seq, heads, dim))

    def call(self, inputs, mask=None, training=None):
        if mask is not None:
            raise ValueError(f'{self.name}: QLIFMHA does not support masks.')
        query = self._spike_train(self._query_dense, self._query_lif, inputs, training)
        key = self._spike_train(self._key_dense, self._key_lif, inputs, training)
        value = self._spike_train(self._value_dense, self._value_lif, inputs, training)

        memory = ops.einsum('bsnk,bsnv->bnkv', key, value)
        context = ops.einsum('bsnk,bnkv->bsnv', query, memory) * self._scale  # type: ignore
        return self._output_dense(context, training=training)

    def _compute_ebops(self, shape):
        # all 1 bit pulses, scaling delayed to the end...
        seq = shape[1]
        heads, key_dim, value_dim = self._num_heads, self._key_dim, self._value_dim
        count_bits = math.ceil(math.log2(seq + 1))  # k^T @ v counts coincident pulses, at most seq
        context_bits = math.ceil(math.log2(seq * key_dim + 1))  # q @ (k^T @ v) adds key_dim counts
        cells = (self._query_lif.cell, self._key_lif.cell, self._value_lif.cell)
        if any(cell.learn_graded_spikes_factor for cell in cells):
            gain_bits = sum(cell._graded_spikes_factor_bits(()) for cell in cells)
        else:
            gain_bits = csd_count(self._scale * math.prod(cell._graded_spikes_factor_init for cell in cells)) - 1

        ebops_count = seq * heads * key_dim * value_dim  # 1b x 1b
        ebops_context = seq * heads * key_dim * value_dim * count_bits  # 1b x count products
        ebops_gain = seq * heads * value_dim * context_bits * gain_bits
        return ebops_count + ebops_context + ebops_gain  # type: ignore

    @property
    def ebops(self):
        if self._ebops is None:
            return ops.cast(0, 'uint32')
        ebops = sum(
            (  # type: ignore
                self._query_dense.ebops,
                self._key_dense.ebops,
                self._value_dense.ebops,
                self._query_lif.ebops,
                self._key_lif.ebops,
                self._value_lif.ebops,
                self._output_dense.ebops,
                ops.convert_to_tensor(self._ebops),  # type: ignore
            )
        )
        return ebops  # type: ignore

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'num_heads': self._num_heads,
                'key_dim': self._key_dim,
                'value_dim': self._value_dim,
                'scale': self._scale,
                'use_bias': self._use_bias,
                'lif_config': self._lif_config,
                'qkvo_kq_conf': self._qkvo_kq_conf,
                'qkvo_bq_conf': self._qkvo_bq_conf,
                'out_proj_iq_conf': self._out_proj_iq_conf,
            }
        )
        return config
