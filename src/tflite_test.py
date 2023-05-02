import tensorflow as tf
import generator
import numpy as np
from tqdm import tqdm
import tflite_runtime.interpreter as tflite

class TestModel(tf.Module):
  def __init__(self, model):
    super(TestModel, self).__init__()
    self.model = model

  @tf.function(input_signature=[tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32)])
  def add(self, inputs):
    '''
    Simple method that accepts single input 'x' and returns 'x' + 4.
    '''
    # Name the output 'result' for convenience.
    inputs = tf.expand_dims(inputs, axis=0)
    inputs = tf.cast(inputs, tf.float16)
    outputs = self.model(inputs)
    output = outputs['outputs'][0,:]
    return {'outputs':output}

def evaluate(data_dir, model):
#   Convert the model.
    model = tf.keras.models.load_model(model)
    model = TestModel(model)
    # # model(np.zeros((128, 543, 3)))

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]

    tflite_model = converter.convert()

    # Save the model.
    tflite_path = 'model.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    tflite_path = 'model.tflite'
    interpreter = tf.lite.Interpreter(tflite_path)

    found_signatures = list(interpreter.get_signature_list().keys())
    print("Found signatures", found_signatures)
    
    prediction_fn = interpreter.get_signature_runner("serving_default")
    print(prediction_fn.get_input_details())
    print(prediction_fn.get_output_details())
    
    gen = generator.BinaryGenerator(data_dir,
                                batch_size=1,
                                step_size=32,
                                shuffle=False,
                                augment=False,
                                oversample=False,
                                split_start=0.0,
                                split_end=0.1,
                                cache=False)
    correct, total = 0, 0
    for xb, yb in tqdm(gen):
        frames = xb[0]
        y = yb[0]
        frames = np.float32(frames)
        output = prediction_fn(inputs=frames)
        # output = model(frames)
        output = output["outputs"]
        
        sign = np.argmax(output)
        y = np.argmax(y)

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
    