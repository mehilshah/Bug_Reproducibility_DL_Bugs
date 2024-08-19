import numpy as np
import tensorflow as tf
from keras import layers as tfl

class Encoder(tfl.Layer):
    def __init__(self,):
        super().__init__()
        self.embed_layer = tfl.Embedding(4500, 64, mask_zero=True)
        self.attn_layer = tfl.MultiHeadAttention(num_heads=2,
                                                 attention_axes=2,
                                                 key_dim=16)
        return

    def call(self, x):
        # Input shape: (4, 5, 20) (Batch size: 4)
        x = self.embed_layer(x)  # Output: (4, 5, 20, 64)
        x = self.attn_layer(query=x, key=x, value=x)  # Output: (4, 5, 20, 64)
        return x


eg_input = tf.constant(np.random.randint(0, 150, (4, 5, 20)))
enc = Encoder()
enc(eg_input)