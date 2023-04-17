import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)



from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


import numpy as np
import random
from tqdm import tqdm
import modellib
import generator
from tensorflow import keras
import tensorflow as tf
import datetime

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {"d_model": self.d_model.numpy(), "warmup_steps": self.warmup_steps}

    # There's actually no need to define `from_config` here, since returning
    # `cls(**config)` is the default behavior.
    @classmethod
    def from_config(cls, config):
        return cls(**config)



def train(data_dir, output):
    step_size = 64
    batch_size = 256

    train_gen = generator.BinaryGenerator(data_dir,
                                            batch_size=batch_size,
                                            step_size=step_size,
                                            shuffle=True,
                                            augment=True,
                                            oversample=True,
                                            split_start=0.1,
                                            split_end=1.0)
    
    val_gen = generator.BinaryGenerator(data_dir,
                                            batch_size=batch_size,
                                            step_size=step_size,
                                            shuffle=False,
                                            augment=False,
                                            oversample=False,
                                            split_start=0.0,
                                            split_end=0.1)

    
    METRICS = [
      keras.metrics.CategoricalAccuracy(name='accuracy'),
    ]

    # norm_layer = keras.layers.Normalization(axis=-1)
    # norm_layer.adapt(train_gen)

    embed_dim = 512
    
    model = modellib.build_model(
        embed_dim=embed_dim,
        input_shape=(step_size, train_gen.n_points),
        head_size=embed_dim,
        num_heads=4,
        ff_dim=embed_dim*2,
        num_transformer_blocks=1,
        mlp_units=[embed_dim],
        mlp_dropout=0.4,
        dropout=0.25,
        n_classes=250,
        layer_norm=True
    )
    
    optimizer = keras.optimizers.AdamW(learning_rate=0.0001, weight_decay=0.004)

    learning_rate = CustomSchedule(embed_dim)
    optimizer = tf.keras.optimizers.AdamW(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9, weight_decay=0.004)
    

    model.compile(optimizer=optimizer,
                  loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.75),
                  metrics=METRICS)

    model.summary()

    # lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler)
    # v = [scheduler(i) for i in range(1000)]
    # print(v)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    board = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                     factor=0.5,
                                                     patience=15,
                                                     min_lr=2e-5)
    
    checkpoint_filepath = f"{output}/" + "model_{epoch}/"
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath)

    # model = keras.models.load_model(f"16_04/model_55/")
    #
    # model.compile(optimizer=keras.optimizers.Adam(0.00001, weight_decay=0.0004),
    #               loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.4),
    #               metrics=METRICS)
    
    # tmp.save_weights("/tmp/weights.hdf5")

    # model.load_weights("/tmp/weights.hdf5")

    model.fit(train_gen,
             validation_data=val_gen,
             epochs=200,
             shuffle=True,
             callbacks=[reduce_lr,board, checkpoint],
              use_multiprocessing=True,
              workers=4
             )

    model.save(f"{output}")
    # model = keras.models.load_model(f"{output}")
    results = model.evaluate(val_gen, verbose=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)

    args = parser.parse_args()
    train(args.data_dir, args.output)
    