import chess
from enum import Enum
from enum import IntFlag
import numpy as np
import tensorflow as tf
import pdb
from halfkp import get_halfkp_indeces

INPUT_SHAPE = 41024
FEATURE_TRANSFORMER_HALF_DIMENSIONS = 256
DENSE_LAYERS_WIDTH = 32
checkpoint_filepath = '/tmp/checkpoint_nneu/'

def build_model_inputs():
  return tf.keras.Input(shape=(41024,), sparse=True), tf.keras.Input(shape=(41024,), sparse=True)

def build_feature_transformer(inputs1, inputs2):
  l = tf.keras.layers
  ft_dense_layer = l.Dense(FEATURE_TRANSFORMER_HALF_DIMENSIONS, name='feature_transformer')
  clipped_relu = l.ReLU(max_value=1.0)
  transformed1 = clipped_relu(ft_dense_layer(inputs1))
  transformed2 = clipped_relu(ft_dense_layer(inputs2))
  return tf.keras.layers.Concatenate()([transformed1, transformed2])

def build_hidden_layers(inputs):
  l = tf.keras.layers
  hidden_layer_1 = l.Dense(DENSE_LAYERS_WIDTH, name='hidden_layer_1')
  hidden_layer_2 = l.Dense(DENSE_LAYERS_WIDTH, name='hidden_layer_2')
  activation_1 = l.ReLU(max_value=1.0)
  activation_2 = l.ReLU(max_value=1.0)
  return activation_2(hidden_layer_2(activation_1(hidden_layer_1(inputs))))

def build_output_layer(inputs):
  l = tf.keras.layers
  output_layer = l.Dense(1, name='output_layer')
  return output_layer(inputs)


def get_model():
    l = tf.keras.layers

    inputs1, inputs2 = build_model_inputs()
    outputs = build_output_layer(build_hidden_layers(build_feature_transformer(inputs1, inputs2)))
    #outputs = l.Dense(1)(inputs1)
    model = tf.keras.Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model
def gen(path, batch_size=256):
    f = open(path)
    while True:
        batch_x = []
        batch_y = []
        line = f.readline().strip()
        if not line:
            f.seek(0)
            break
        data = line.split(',')
        fen = data[0]
        y = float(data[1])
        eval = y / 100


        # real max min
        #mx = 11254.0
        #mi = -10479.0

        # artificial
        mx = 2000
        mi = -2000
        #normalise
        if y > mx or y < mi:
            continue

        d = mx - mi
        if d == 0:
            d = 1
        y = (y - mi) / d

        #x = get_training_data(chess.Board(fen))
        #x = np.random.rand(INPUT_SHAPE)
        board = chess.Board(fen)
        try:
            X = get_halfkp_indeces(board)
            #yield (X[0], X[1]), y
            yield (X[0], X[1]), eval
        except:
            continue

def train():
    model = get_model()
    train_dataset = tf.data.Dataset.from_generator(
        gen,
        args=['/Users/silfverstrom/Workspace/link/projects/td-chess/data/data_100k.csv'],
        output_signature=((tf.SparseTensorSpec((41024,)), tf.SparseTensorSpec((41024,))), tf.TensorSpec(()))
    )
    test_dataset = tf.data.Dataset.from_generator(
        gen,
        args=['/Users/silfverstrom/Workspace/link/projects/td-chess/data/data_100k.csv'],
        output_signature=((tf.SparseTensorSpec((41024,)), tf.SparseTensorSpec((41024,))), tf.TensorSpec(()))
    )
    train_dataset = train_dataset.shuffle(2560)
    #train_dataset = train_dataset

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',
        baseline=None, restore_best_weights=False
    )
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
    )

    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    model.fit(
        train_dataset.batch(256),
        validation_data=test_dataset.batch(100),
        epochs=10,
        steps_per_epoch=4000,
        validation_steps=400,
        callbacks = [early_stopping, model_checkpoint_callback]
    )
    pdb.set_trace()

if __name__ == '__main__':
    train()
