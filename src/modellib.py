import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)



from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf

from tensorflow import keras
import tensorflow as tf

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0, layer_norm=True):
    # Normalization and Attention
    if layer_norm:

        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    else:
        x = inputs
    
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    if layer_norm:
        x = layers.LayerNormalization(epsilon=1e-6)(res)
    else:
        x = res
    
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def keypoint_embedding(input, units): 
    INIT_HE_UNIFORM = keras.initializers.he_uniform
    INIT_GLOROT_UNIFORM = keras.initializers.glorot_uniform
    GELU = keras.activations.gelu
    x = keras.layers.Dense(units, use_bias=False, kernel_initializer=INIT_GLOROT_UNIFORM)(input)
    x = keras.layers.Activation(GELU)(x)
    x = keras.layers.Dense(units, use_bias=False, kernel_initializer=INIT_HE_UNIFORM)(x)
    return x

# def positional_encoding(length, depth):
#   depth = depth/2

#   positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
#   depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

#   angle_rates = 1 / (10000**depths)         # (1, depth)
#   angle_rads = positions * angle_rates      # (pos, depth)

#   pos_encoding = np.concatenate(
#       [np.sin(angle_rads), np.cos(angle_rads)],
#       axis=-1) 

#   return tf.cast(pos_encoding, dtype=tf.float16)


# class PositionalEmbedding(tf.keras.layers.Layer):
#   def __init__(self, length, d_model):
#     super().__init__()
#     self.d_model = d_model
#     self.pos_encoding = positional_encoding(length=length, depth=d_model)
    
#   def call(self, x):
#     length = tf.shape(x)[1]
#     # x *= tf.math.sqrt(tf.cast(self.d_model, tf.float16))
#     x = x + self.pos_encoding[tf.newaxis, :length, :]
#     return x

# class GestureEmbedder(tf.keras.models.Model):
#     def __init__(self):
#         super(GestureEmbedder, self).__init__()
        
#     def build(self, input_shape):
#         self.gesture_embedder = keras.models.load_model('gesture_models/gesture_embedder/')
#         self.gesture_embedder.trainable = False

#     def call(self, x):
#         x = tf.reshape(x, [-1, 42, 3])
        
#         left = x[:, 0:21, :]
#         right = x[:, 21:, :]
        
#         # print(left.shape, right.shape)

#         out1 = self.gesture_embedder([left, tf.zeros((16384, 1), dtype=tf.float16), left])
#         out2 = self.gesture_embedder([right, tf.ones((16384, 1), dtype=tf.float16), right])
        
#         out1 = tf.cast(out1, dtype=tf.float16)
#         out2 = tf.cast(out2, dtype=tf.float16)
#         out = tf.concat([out1, out2], axis=-1)
#         out *= 2.0
        
#         return out 
    

# def build_model(
#     embed_dim,
#     input_shape,
#     head_size,
#     num_heads,
#     ff_dim,
#     num_transformer_blocks,
#     mlp_units,
#     dropout=0,
#     mlp_dropout=0,
#     n_classes=None,
#     layer_norm=True,
#     pos_embedding=False
# ):
#     inputs = keras.Input(shape=input_shape, name='keypoints')
#     lhand = keras.Input(shape=(64, 42, 3), name='hands')


#     hand_embedding_size = 256
#     left_embedder = GestureEmbedder()
    
#     lembedding = tf.keras.layers.TimeDistributed(left_embedder)(lhand)
#     lembedding = layers.Dense(64, activation='relu')(lembedding)
    
#     embed = keypoint_embedding(inputs, embed_dim)

#     embed = tf.concat([embed, lembedding], axis=-1)
#     embed_dim = embed_dim + 64
    
#     if pos_embedding: 
#         pos_emb_layer = PositionalEmbedding(length=input_shape[0], d_model=embed_dim)
#         x = pos_emb_layer(embed)

#     else:
#         pos_emb_layer = layers.Embedding(input_dim=input_shape[0], output_dim=embed_dim,  embeddings_initializer = tf.keras.initializers.constant(0.0))
#         positions = tf.range(start=0, limit=input_shape[0], delta=1)
#         x = embed + pos_emb_layer(positions)
        
#     for _ in range(num_transformer_blocks):
#         x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout, layer_norm)

#     x = layers.GlobalAveragePooling1D()(x)
#     x = layers.Dropout(mlp_dropout)(x)

#     for dim in mlp_units:
#         x = layers.Dense(dim, activation="relu")(x)
#         x = layers.Dropout(mlp_dropout)(x)
    
#     outputs = layers.Dense(n_classes, activation="softmax")(x)
#     return keras.Model([inputs, lhand], outputs)

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     args = parser.parse_args()



#     from tensorflow.keras import mixed_precision
#     policy = mixed_precision.Policy('mixed_float16')
#     mixed_precision.set_global_policy(policy)


#     embed_dim = 512

#     model = build_model(
#         embed_dim=embed_dim,
#         input_shape=(64, 256),
#         head_size=embed_dim,
#         num_heads=4,
#         ff_dim=embed_dim*2,
#         num_transformer_blocks=1,
#         mlp_units=[embed_dim],
#         mlp_dropout=0.4,
#         dropout=0.25,
#         n_classes=250,
#         layer_norm=True
#     )
#     model.summary()