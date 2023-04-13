import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import numpy as np
import random
from tqdm import tqdm
import modellib
import generator
from tensorflow import keras
import datetime

def scheduler(epoch, lr):
    if epoch < 10:
        lr = 0.003
    
    elif epoch < 60:
        lr = 0.0003
    
    elif epoch < 120:
        lr = 0.0001

    elif epoch < 1:
        lr = 0.0001
    
    elif epoch < 160:
        lr = 0.00005
    
    else:
        lr = 0.00001
    
    return lr


def train(data_dir, output):
    step_size = 64
    batch_size = 1024

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
      keras.metrics.SparseCategoricalAccuracy(name='sparse_categorical_accuracy'),
    ]

    # norm_layer = keras.layers.Normalization(axis=-1)
    # norm_layer.adapt(train_gen)

    
    model = modellib.build_model(
        embed_dim=128,
        input_shape=(step_size, train_gen.n_points),
        head_size=128,
        num_heads=4,
        ff_dim=128,
        num_transformer_blocks=1,
        mlp_units=[512],
        mlp_dropout=0.4,
        dropout=0.25,
        n_classes=250
    )
 

    model.compile(optimizer=keras.optimizers.Adam(0.0001),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=METRICS)

    model.summary()

    lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler)
    
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    board = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_sparse_categorical_accuracy',
                                                     factor=0.5,
                                                     patience=15,
                                                     min_lr=2e-4)
    
    checkpoint_filepath = f"{output}/" + "model_{epoch}/"
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath)

    # tmp = keras.models.load_model(f"model_large_2/model_199")
    # tmp.save_weights("/tmp/weights.hdf5")

    # model.load_weights("/tmp/weights.hdf5")

    model.fit(train_gen,
             validation_data=val_gen,
             epochs=100,
             shuffle=True,
             callbacks=[reduce_lr, board, checkpoint])

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
    