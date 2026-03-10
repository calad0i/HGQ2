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
        # Gather local variables to pass up, filtering out Q-specific ones
        kwargs = gather_vars_to_kwargs('self|kq_conf')
        super().__init__(**kwargs)

        # Using kq_conf (Kernel Quantizer) to remain consistent with your QDense config naming
        kq_conf = kq_conf or QuantizerConfig('default', 'weight')
        self._kq = Quantizer(kq_conf, name=f'{self.name}_kq')

    @property
    def kq(self):
        return self._kq

    def build(self, input_shape=None):
        super().build(input_shape)
        
        # Build the quantizer based on the shape of the Keras embedding matrix
        self.kq.build(ops.shape(self.embeddings))

    def call(self, inputs):
        # 1. Quantize the embedding lookup table
        qembeddings = self.qembeddings
        
        # 2. Perform the lookup (this perfectly mimics your HLS index mapping)
        out = ops.take(qembeddings, inputs, axis=0)
        
        return out

    def _compute_ebops(self, shape):
        # Embedding layers perform memory lookups rather than MAC operations.
        # Returning 0 ensures frameworks counting Effective Bit Operations (EBOps) don't crash.
        return 0

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
        # Apply the quantizer function to the underlying Keras embeddings matrix
        return self.kq(self.embeddings)