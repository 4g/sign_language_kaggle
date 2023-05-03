import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from tensorflow.keras import layers
from tensorflow import keras
keras.utils.set_random_seed(42)


import tensorflow as tf
import numpy as np

import mediapipe as mp
from keypoint_lib import get_keypoint_flip_mapping
mp_face_mesh = mp.solutions.face_mesh

class Parts:
    def __init__(self) -> None:
        self.make_relevant_parts()

    def make_relevant_parts(self):
        self.LHAND = list(range(468, 489))
        self.RHAND = list(range(522, 543))
        self.POSE = list(range(489, 522))
        self.SHOULDERS = [500, 501]
        self.ELBOWS = [502, 503]
        self.WRISTS = [489+15, 489+16]


        get_face_mesh_part = lambda part : list(set(list(np.reshape(np.asarray(list(part)), -1))))
        self.LIP =  get_face_mesh_part(mp_face_mesh.FACEMESH_LIPS)
        self.LEYE = get_face_mesh_part(mp_face_mesh.FACEMESH_RIGHT_EYE)
        self.REYE = get_face_mesh_part(mp_face_mesh.FACEMESH_LEFT_EYE)

        self.LIP_LEFT = self.LIP.index(78)
        self.LIP_RIGHT = self.LIP.index(308)

        self.left, self.right = get_keypoint_flip_mapping()


@tf.function
def rotate_3d(coordinates):
    # Rotation angles in degrees
    a1, a2, a3 = 30, 45, 60

    a1_rad, a2_rad, a3_rad = np.radians(a1), np.radians(a2), np.radians(a3)


    # Rotation matrices for each axis
    rot_x = np.array([
        [1, 0, 0],
        [0, np.cos(a1_rad), -np.sin(a1_rad)],
        [0, np.sin(a1_rad), np.cos(a1_rad)],
    ], dtype=np.float32)

    rot_y = np.array([
        [np.cos(a2_rad), 0, np.sin(a2_rad)],
        [0, 1, 0],
        [-np.sin(a2_rad), 0, np.cos(a2_rad)],
    ], dtype=np.float32)

    rot_z = np.array([
        [np.cos(a3_rad), -np.sin(a3_rad), 0],
        [np.sin(a3_rad), np.cos(a3_rad), 0],
        [0, 0, 1],
    ], dtype=np.float32)
        # Combined rotation matrix
    
    rot_matrix = tf.matmul(rot_z, tf.matmul(rot_y, rot_x))

    # Apply rotation to the coordinates
    rotated_coordinates = tf.matmul(coordinates, rot_matrix)
    return rotated_coordinates

@tf.function
def preprocess(x, augment=False):
    # extract 64 frames to work with
    # remove frames where both hands are NAN
    parts = Parts()

    @tf.function
    def hflip(kps):
        new_kps = kps * tf.constant([-1, 1, 1], dtype=tf.float32) + tf.constant([1, 0, 0], dtype=tf.float32)
        index_map = np.asarray(list(range(543)), dtype=np.int32)
        tmp = index_map[parts.left]
        index_map[parts.left] = index_map[parts.right]
        index_map[parts.right] = tmp
        
        new_kps = tf.gather(new_kps, index_map, axis=1)

        return new_kps
    
    @tf.function
    def center_around(kps, indices):
        mean_coordinate = tf.math.reduce_mean(tf.gather(kps, indices=indices, axis=1), axis=1, keepdims=True)
        centered_keypoints = kps - mean_coordinate
        return centered_keypoints

    @tf.function
    def resize_by_bone(kps, joint1, joint2, desired_length=2.0):
        joint1 = kps[:, joint1, 0:2]
        joint2 = kps[:, joint2, 0:2]
        bone_length = tf.norm(joint1 - joint2, axis=1)
        ratio = (desired_length / bone_length)
        ratio = ratio[:, tf.newaxis, tf.newaxis]
        kps = kps * ratio
        return kps


    lna = tf.math.is_nan(x[:,parts.LHAND[0],0])
    rna = tf.math.is_nan(x[:,parts.RHAND[0],0])

    nlna = tf.math.reduce_sum(tf.cast(lna, tf.int32))
    nrna = tf.math.reduce_sum(tf.cast(rna, tf.int32))
    
    # assume right hand dominant
    dominant_hand_index = rna

    # override if left hand has less NA
    if nlna < nrna:
        x = hflip(x)
        dominant_hand_index = lna

    # extract frames where dominant hand is present
    hand_present = tf.where(~dominant_hand_index)
    hand_present = tf.reshape(hand_present, shape=(-1,))
    x = tf.gather(x, hand_present, axis=0)

    # extract 64 equal spaced frames
    n_frames = tf.shape(x)[0]
    indices = tf.linspace(0, n_frames - 1, 64, name="linspace", axis=0)
    indices = tf.cast(indices, tf.int32)

    keypoints = tf.gather(x, indices)

    lip = tf.gather(keypoints, parts.LIP, axis=1)
    lip = center_around(lip, [parts.LIP_LEFT, parts.LIP_RIGHT])
    lip = resize_by_bone(lip, joint1=parts.LIP_LEFT, joint2=parts.LIP_RIGHT)
    lip = tf.gather(lip, [0,1], axis=2)

    hand = tf.gather(keypoints, parts.RHAND, axis=1)
    hand = center_around(hand, [0])
    hand = resize_by_bone(hand, joint1=0, joint2=12)

    if augment:
        hand = rotate_3d(hand)

    pose_parts = parts.SHOULDERS + parts.ELBOWS + [parts.LIP_LEFT, parts.LIP_RIGHT] + parts.WRISTS
    pose = tf.gather(keypoints, pose_parts, axis=1)
    pose = center_around(kps=pose, indices=[0,1])
    pose = resize_by_bone(kps=pose, joint1=0, joint2=1)
    pose = tf.gather(pose, [0,1], axis=2)


    hand = tf.reshape(hand, (64, -1))
    pose = tf.reshape(pose, (64, -1))
    lip = tf.reshape(lip, (64, -1))

    def bye_nan(t):
        is_nan = tf.math.is_nan(t)
        t = tf.where(is_nan, tf.zeros_like(t), t)
        return t

    hand = bye_nan(hand)
    pose = bye_nan(pose)
    lip = bye_nan(lip)

    return hand, lip, pose

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0, layer_norm=True):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    
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

def build_model(
    embed_dim,
    shapes,
    num_steps,
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

    hand = tf.keras.Input(shape=shapes[0], name='hand', dtype=tf.float32)
    lip = tf.keras.Input(shape=shapes[1], name='lip', dtype=tf.float32)
    pose = tf.keras.Input(shape=shapes[2], name='pose', dtype=tf.float32)
    
    hand_embed = keypoint_embedding(hand, embed_dim)
    lip_embed = keypoint_embedding(lip, embed_dim)
    pose_embed = keypoint_embedding(pose, embed_dim)
    embed = tf.concat([hand_embed, lip_embed, pose_embed], axis=-1)

    embed_dim = embed_dim * len(shapes)

    pos_emb_layer = layers.Embedding(input_dim=num_steps, output_dim=embed_dim,  embeddings_initializer = tf.keras.initializers.constant(0.0))
    positions = tf.range(start=0, limit=num_steps, delta=1)
    x = embed + pos_emb_layer(positions)
    
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout, layer_norm)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(mlp_dropout)(x)

    # for dim in mlp_units:
    #     x = layers.Dense(dim, activation="relu")(x)
    #     x = layers.Dropout(0.1)(x)
    
    outputs = layers.Dense(n_classes, activation="softmax", kernel_initializer=keras.initializers.glorot_uniform)(x)
    
    return tf.keras.Model([hand, lip, pose], outputs, name="transformer")


class TFLiteModel(tf.keras.models.Model):
    def __init__(self):
        super(TFLiteModel, self).__init__()

        # Load the feature generation and main models
        self.preprocess_layer = preprocess
        embed_dim = 256
        
        outputs = self.preprocess_layer(np.random.random((64, 543, 3)).astype(np.float32))
        shapes = [x.shape for x in outputs]
        print("preprocessed_shape", shapes)

        self.model = build_model(
                        embed_dim=embed_dim,
                        num_steps=64,
                        shapes=shapes,
                        head_size=embed_dim,
                        num_heads=4,
                        ff_dim=1536,
                        num_transformer_blocks=1,
                        mlp_units=[embed_dim],
                        mlp_dropout=0.3,
                        dropout=0.3,
                        n_classes=250,
                        layer_norm=True,
                        pos_embedding=False
                    )

    def build(self, input_shape):
        return super().build((None, None, 543, 3))

    def call(self, inputs, training=True):
        # Preprocess Data
        hand, lip, pose = tf.map_fn(self.preprocess_layer, inputs, fn_output_signature=(tf.float32, tf.float32, tf.float32), parallel_iterations=20)
        outputs = self.model({'hand': hand,
                             'lip':lip,
                             'pose': pose}, training)
        return {'outputs': outputs}

# tf.config.run_functions_eagerly(True) 
# import numpy as np
# parts = Parts()

# x = np.random.random((22, 543, 3)).astype(np.float32)
# x[11] = np.nan
# x[13, parts.LHAND] = np.nan
# x[14, parts.RHAND] = np.nan
# x[15, parts.RHAND] = np.nan

# kp = preprocess(x)
# print(kp)

# x = np.random.random((100, 543, 3)).astype(np.float32)
# x[11] = np.nan
# x[13, parts.LHAND] = np.nan
# x[14, parts.LHAND] = np.nan
# x[15, parts.RHAND] = np.nan

# kp = preprocess(x)
# print(kp)



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