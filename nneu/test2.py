import tensorflow as tf
if __name__ == '__main__':
    model = tf.keras.models.load_model('output/nnue_0.1.2')
    print(model.summary())
