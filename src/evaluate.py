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

def train(data_dir, model_path):
    gen = generator.BinaryGenerator(data_dir,
                                    batch_size=128,
                                    step_size=64,
                                    shuffle=False,
                                    augment=False,
                                    oversample=False,
                                    split_start=0.0,
                                    split_end=0.1,
                                    cache=False,
                                    mode='val')

    model = keras.models.load_model(f"{model_path}")
    correct = 0
    total = 0
    
    for xb, yb in tqdm(gen):
        results = model.predict(xb, verbose=False)
        ypred = np.argmax(results, axis=1)
        ytrue = np.argmax(yb, axis=1)
        correct += sum(ypred == ytrue)
        total += len(ypred)
    print(correct, total, correct/total)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)

    args = parser.parse_args()
    train(args.data_dir, args.model)
    