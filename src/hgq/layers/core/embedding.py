from keras import ops
from keras.layers import Embedding

from ...quantizer import Quantizer
from ...quantizer.config import QuantizerConfig
from ...utils.misc import gather_vars_to_kwargs
from .base import QLayerBaseSingleInput


class QEmbedding(QLayerBaseSingleInput, Embedding):
    def __init__(
        self,
        input_dim,
        output_dim,
        embeddings_initializer='uniform',
        embeddings_regularizer=None,
        activity_regularizer=None,
        embeddings_constraint=None,
        mask_zero=False,
        kq_conf: None | QuantizerConfig = None,
        **kwargs,
    ):
        kwargs = gather_vars_to_kwargs('self|kq_conf')
        super().__init__(**kwargs)

        kq_conf = kq_conf or QuantizerConfig('default', 'weight')
        self._kq = Quantizer(kq_conf, name=f'{self.name}_kq')

    @property
    def kq(self):
        return self._kq

    def build(self, input_shape=None):
        super().build(input_shape)

        self.kq.build(ops.shape(self.embeddings))

    def call(self, inputs):
        qembeddings = self.qembeddings
        out = ops.take(qembeddings, inputs, axis=0)

        return out

    def _compute_ebops(self, shape):
        bw_emb = self.kq.bits_(ops.shape(self.embeddings))

        out_size = ops.prod(shape) * self.output_dim
        size = ops.cast(out_size, self.dtype)

        ebops = ops.mean(bw_emb) * size

        return ebops

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'kq_conf': self.kq.config,
            }
        )
        return config

    @property
    def qembeddings(self):
        return self.kq(self.embeddings)