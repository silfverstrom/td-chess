import tensorflow as tf
import numpy as np
import chess
import timeit

#model = tf.keras.models.load_model("/Users/silfverstrom/Workspace/link/projects/td-chess/output/tuner_v4/model.11-1.56")
model = tf.keras.models.load_model("/Users/silfverstrom/Workspace/link/projects/td-chess/output/tuner_v7/model.38-1.63")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# test model
#converter = tf.lite.TFLiteConverter.from_saved_model("/Users/silfverstrom/Workspace/link/projects/td-chess/output/tuner_v4/model.11-1.56")

#converter = tf.lite.TFLiteConverter.from_saved_model("/Users/silfverstrom/Workspace/link/projects/td-chess/output/tuner_v4/model.11-1.56")
#converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]

#converter.representative_dataset = representative_data_gen
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#converter.inference_input_type = tf.uint8
#converter.inference_output_type = tf.uint8
tflite_quant_model = converter.convert()
# Save the model.
with open('/tmp/test.tflite', 'wb') as f:
  f.write(tflite_quant_model)


  #interpreter = tf.lite.Interpreter(model_path='/tmp/model.tflite')
interpreter = tf.lite.Interpreter(model_path='/tmp/test.tflite')


def run_tflite(x1, x2):
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']


    input_data = np.array([x1], dtype=np.float32)
    input_data2 = np.array([x2], dtype=np.float32)

    #input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.set_tensor(input_details[1]['index'], input_data2)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0][0]



def ru():
    u1 = []
    u2 = []
    for i in range(774):
        u1.append(1.0)
    for i in range(15):
        u2.append(1.0)
    run_tflite(u1, u2)

print(timeit.timeit(lambda: ru(), number=1000))
