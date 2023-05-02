import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from tensorflow.keras import layers
from tensorflow import keras


import tensorflow as tf
import numpy as np 

SHOULDERS = [500, 501]
LHAND = list(range(468, 489))
RHAND = list(range(522, 543))

class Parts:
    def __init__(self) -> None:
        pass

    def make_relevant_parts(self):
        self.LHAND = list(range(468, 489))
        self.RHAND = list(range(522, 543))
        self.POSE = list(range(489, 522))
        self.SHOULDERS = [500, 501]
        self.ELBOWS = [502, 503]
        self.WRISTS = [489+15, 489+16]
        self.HAND_POSE_POINTS = [489+x for x in [15,16,17,18,19,20,21,22]]

        self.UPPER_LIP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415]
        
        self.LOWER_LIP = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324]
        
        self.LIP = self.LOWER_LIP + self.UPPER_LIP        
        
        self.REYE = [
            145, 153,
            158, 157,
        ]

        self.LEYE = [
            374, 380,
            385, 384
        ]
        
        # self.relevant_parts = [self.LEYE, self.REYE, self.LIP, self.LHAND, self.RHAND, self.SHOULDERS, self.ELBOWS, self.HAND_POSE_POINTS]
        self.relevant_parts = [self.LEYE, self.REYE, self.LIP, self.SHOULDERS, self.ELBOWS, self.LHAND, self.RHAND]
        # self.relevant_parts = [self.LEYE, self.REYE, self.LIP, self.SHOULDERS, self.ELBOWS, self.WRISTS]

        self.relevant_indices = []
        for part in self.relevant_parts:
            self.relevant_indices += part
        
        return self

@tf.function
def preprocess(x):
    # extract 64 frames to work with
    # remove frames where both hands are NAN
    x = tf.cast(x, tf.float16)

    lna = tf.math.is_nan(x[:,LHAND[0],0])
    rna = tf.math.is_nan(x[:,RHAND[0],0])

    nlna = tf.math.reduce_sum(tf.cast(lna, tf.int32))
    nrna = tf.math.reduce_sum(tf.cast(rna, tf.int32))
    
    dominant_hand = RHAND
    dominant_hand_index = rna
    handedness = 1.0

    if nlna < nrna:
        dominant_hand = LHAND
        dominant_hand_index = lna
        handedness = 0.0

    hand_present = tf.where(~dominant_hand_index)
    hand_present = tf.reshape(hand_present, shape=(-1,))
    x = tf.gather(x, hand_present, axis=0)

    n_frames = tf.shape(x)[0]
    indices = tf.linspace(0, n_frames - 1, 64, name="linspace", axis=0)
    indices = tf.cast(indices, tf.int32)

    keypoints = tf.gather(x, indices)

    def get_normalized_hand(kps, hand_index):
        h = tf.gather(kps, hand_index, axis=1)
        z = h[:, :, 2:]
        h = h[:,:,0:2]
        
        mi = tf.math.reduce_min(h, axis=1)
        ma = tf.math.reduce_max(h, axis=1)
        h = (h - mi[:, tf.newaxis, :])/((ma[:, tf.newaxis, :] - mi[:, tf.newaxis, :]) + 0.00001)
        h = tf.concat([h, z], axis=-1)
        return h
    
    lh = get_normalized_hand(keypoints, dominant_hand)
    hands = lh
    

    # hands = tf.concat([lh, rh], axis=-2)

    # Check for NaN values in the tensor
    is_nan = tf.math.is_nan(keypoints)

    # Replace NaN values with zeros
    keypoints = tf.where(is_nan, tf.zeros_like(keypoints), keypoints)


    # normalize by shoulders    
    desired_shoulder_length = 2.0
    left_shoulder_coordinates = keypoints[:, SHOULDERS[0], :]
    right_shoulder_coordinates = keypoints[:, SHOULDERS[1], :]
    shoulder_length = tf.norm(left_shoulder_coordinates - right_shoulder_coordinates, axis=1)
    ratio = (desired_shoulder_length / shoulder_length)
    ratio = ratio[:, tf.newaxis, tf.newaxis]
    keypoints = keypoints * ratio

    # Calculate the mean of both shoulders for each frame
    mean_shoulder_coordinates = (left_shoulder_coordinates + right_shoulder_coordinates) / 2

    # Subtract the mean shoulder coordinates from all keypoints to center each frame
    centered_keypoints = keypoints - mean_shoulder_coordinates[:, tf.newaxis, :]

    parts = Parts().make_relevant_parts()
    
    part_indices = parts.relevant_indices
    x = tf.gather(centered_keypoints, part_indices, axis=1)
    x = tf.gather(x, [0,1], axis=2)
    x = tf.reshape(x, (64, 148))

    x = tf.cast(x, tf.float16)
    hands = tf.cast(hands, tf.float16)
    handedness = tf.cast(handedness, tf.float16)
    handedness = tf.repeat(handedness, repeats=64)

    return x, hands, handedness


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

def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

  return tf.cast(pos_encoding, dtype=tf.float16)


class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, length, d_model):
    super().__init__()
    self.d_model = d_model
    self.pos_encoding = positional_encoding(length=length, depth=d_model)
    
  def call(self, x):
    length = tf.shape(x)[1]
    # x *= tf.math.sqrt(tf.cast(self.d_model, tf.float16))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x

def build_model(
    embed_dim,
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
    n_classes=None,
    layer_norm=True,
    pos_embedding=False
):

    inputs = tf.keras.Input(shape=input_shape, name='keypoints', dtype=tf.float16)

    hand_embedding = tf.keras.Input(shape=(64, 128), name='hand_embedding', dtype=tf.float16)
    
    embed = keypoint_embedding(inputs, embed_dim)
    embed2 = keypoint_embedding(hand_embedding, 64)

    embed = tf.concat([embed, embed2], axis=-1)

    embed_dim = embed_dim + 64

    if pos_embedding:
        pos_emb_layer = PositionalEmbedding(length=input_shape[0], d_model=embed_dim)
        x = pos_emb_layer(embed)

    else:
        print("NOT POSEMB")
        pos_emb_layer = layers.Embedding(input_dim=input_shape[0], output_dim=embed_dim,  embeddings_initializer = tf.keras.initializers.constant(0.0))
        positions = tf.range(start=0, limit=input_shape[0], delta=1)
        x = embed + pos_emb_layer(positions)
        
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout, layer_norm)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(mlp_dropout)(x)

    # for dim in mlp_units:
    #     x = layers.Dense(dim, activation="relu")(x)
    #     x = layers.Dropout(0.1)(x)
    
    outputs = layers.Dense(n_classes, activation="softmax", kernel_initializer=keras.initializers.glorot_uniform)(x)
    
    return tf.keras.Model([inputs, hand_embedding], outputs, name="transformer")


@tf.function
def hand_embedder_map(hands, handedness, gesture_embedder):
    shape = tf.shape(hands)
    size = shape[0] * shape[1]
    
    hands = tf.reshape(hands, [size, 21, 3])
    handedness = tf.reshape(handedness, (size, 1))
    out = gesture_embedder([hands, handedness, hands], training=False)
    out = out * 2.0
    out = tf.cast(out, dtype=tf.float16)
    out = tf.reshape(out, (shape[0], shape[1], 128))
    
    return out


class TFLiteModel(tf.keras.models.Model):
    def __init__(self):
        super(TFLiteModel, self).__init__()

        # Load the feature generation and main models
        self.preprocess_layer = preprocess
        embed_dim = 512
        self.model = build_model(
                        embed_dim=embed_dim,
                        input_shape=(64, 148),
                        head_size=embed_dim,
                        num_heads=4,
                        ff_dim=embed_dim*3,
                        num_transformer_blocks=1,
                        mlp_units=[embed_dim],
                        mlp_dropout=0.3,
                        dropout=0.3,
                        n_classes=250,
                        layer_norm=True,
                        pos_embedding=False
                    )

        self.hand_embedder = tf.keras.models.load_model('gesture_models/gesture_embedder/')
        self.hand_embedder.trainable = False

    def build(self, input_shape):
        return super().build((None, None, 543, 3))

    def call(self, inputs, training=True):
        # Preprocess Data
        x, hands, handedness = tf.map_fn(self.preprocess_layer, inputs, fn_output_signature=(tf.float16, tf.float16, tf.float16))
        hand_embedding = hand_embedder_map(hands, handedness, self.hand_embedder)

        # hand_embedding = tf.map_fn(self.hand_embedder, hands, fn_output_signature=tf.float16, parallel_iterations=20, swap_memory=True, infer_shape=False)
        outputs = self.model({ 'keypoints': x, 'hand_embedding':hand_embedding }, training)
        return {'outputs': outputs}

# tf.config.run_functions_eagerly(True) 
# import numpy as np
# x = np.random.random((100, 543, 3))
# x[11] = np.nan
# x[13, LHAND] = np.nan
# x[14, RHAND] = np.nan
# x[15, RHAND] = np.nan

# kp, lh, handedness = preprocess(x)
# print(kp.shape, lh.shape, handedness)



# from tensorflow.keras import mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

# import numpy as np
# model = TFLiteModel()
# x = np.random.random((1024, 100, 543, 3))
# x = x.astype(np.float16)
# y = np.ones((1024, 250), dtype=np.float16)
# x = np.asarray(x, dtype=np.float32)
# model.compile(tf.optimizers.Adam(0.0001), 'categorical_crossentropy')
# model.build(input_shape=(256, 100, 543, 3))
# model.summary()
# y = model.fit(x, y, epochs=100, batch_size=256)
# # print(y['outputs'])

# mdl = TFLiteWrapper(tf.keras.models.load_model('test_model/model_1/'))
# mdl.save('tfliteable_model')