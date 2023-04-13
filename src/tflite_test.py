import tensorflow as tf
import generator
import numpy as np
from tqdm import tqdm

def evaluate(data_dir, model):
    # Convert the model.
    model = tf.keras.models.load_model(model)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    tflite_path = 'model.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    interpreter = tf.lite.Interpreter(tflite_path)

    found_signatures = list(interpreter.get_signature_list().keys())
    print("Found signatures", found_signatures)
    
    prediction_fn = interpreter.get_signature_runner("serving_default")
    print(prediction_fn.get_input_details())
    print(prediction_fn.get_output_details())
    
    gen = generator.BinaryGenerator(data_dir,
                                batch_size=1,
                                step_size=64,
                                shuffle=False,
                                augment=False,
                                oversample=False,
                                split_start=0.8,
                                split_end=1.0,
                                cache=False)
    correct, total = 0, 0
    for xb, yb in tqdm(gen):
        frames = xb[0]
        y = yb[0]
        output = prediction_fn(input_1=frames)
        sign = np.argmax(output["dense_2"])
        if int(sign) == int(y):
            correct += 1
        total += 1
    print(correct/total)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)

    args = parser.parse_args()
    evaluate(args.data_dir, args.model)
    