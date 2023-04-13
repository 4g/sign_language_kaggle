import numpy as np
import random
from tqdm import tqdm
import modellib
import generator
from keras.optimizers import adam_v2
from tensorflow import keras

def scheduler(epoch, lr):
    return 0.0003 * (0.3 ** (epoch//30))

def train(data_dir, model_path):
    gen = generator.BinaryGenerator(data_dir,
                                    batch_size=128,
                                    step_size=64,
                                    shuffle=False,
                                    augment=False,
                                    oversample=False,
                                    split_start=0.8,
                                    split_end=1.0,
                                    cache=False)

    model = keras.models.load_model(f"{model_path}")
    correct = 0
    total = 0
    
    for xb, yb in tqdm(gen):
        results = model.predict(xb, verbose=False)
        ypred = np.argmax(results, axis=1)
        correct += sum(ypred == yb)
        total += len(ypred)
    print(correct, total, correct/total)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)

    args = parser.parse_args()
    train(args.data_dir, args.model)
    