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
import datetime

def train(data_dir, output):
    step_size = 64
    batch_size = 256

    train_gen = generator.BinaryGenerator(data_dir,
                                            batch_size=batch_size,
                                            step_size=step_size,
                                            shuffle=True,
                                            augment=True,
                                            oversample=True,
                                            split_start=0,
                                            split_end=0.9)
    
    val_gen = generator.BinaryGenerator(data_dir,
                                            batch_size=batch_size,
                                            step_size=step_size,
                                            shuffle=False,
                                            augment=False,
                                            oversample=False,
                                            split_start=0.9,
                                            split_end=1.0)

    
    METRICS = [
      keras.metrics.CategoricalAccuracy(name='accuracy'),
    ]

    # norm_layer = keras.layers.Normalization(axis=-1)
    # norm_layer.adapt(train_gen)

    embed_dim = 384
    
    model = modellib.build_model(
        embed_dim=embed_dim,
        input_shape=(step_size, train_gen.n_points),
        head_size=embed_dim,
        num_heads=4,
        ff_dim=embed_dim,
        num_transformer_blocks=1,
        mlp_units=[embed_dim],
        mlp_dropout=0.4,
        dropout=0.25,
        n_classes=250
    )
 

    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=0.0001, weight_decay=0.004),
                  loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.4),
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

    # model = keras.models.load_model(f"15_04/model_198/")
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
    