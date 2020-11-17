import chess
import chess.engine
import numpy as np
import chess.pgn
import os
import sys

import pdb
import tensorflow as tf
import datetime
import configparser
import random
from halfkp import get_halfkp_indeces

INPUT_SHAPE = 41024
FEATURE_TRANSFORMER_HALF_DIMENSIONS = 256
DENSE_LAYERS_WIDTH = 32

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

def get_generator(engine, config):
    def gen(path):
        f = open(path)
        while True:
            line = f.readline()

            if not line:
                break
            try:
                board = chess.Board().from_epd(line)[0]

                ev = engine.analyse(board, chess.engine.Limit(depth=config['stockfish_depth']))
                y = float(str(ev['score'].white()))
                y = y / 100
                X = get_halfkp_indeces(board)
            except Exception as e:
                continue

            if len(X) == 2:
                yield (X[0], X[1]), y
            else:
                continue
    return gen

def get_model():
    l = tf.keras.layers

    inputs1, inputs2 = build_model_inputs()
    outputs = build_output_layer(build_hidden_layers(build_feature_transformer(inputs1, inputs2)))
    #outputs = l.Dense(1)(inputs1)
    model = tf.keras.Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Must be run with args')
        sys.exit()
    config_path = sys.argv[1]


    config = configparser.ConfigParser()
    config.read(config_path)
    config = dict(config['DEFAULT'])
    print(config)

    DB_PATH = config['db_path']
    DB_TEST_PATH = config['db_test_path']
    DB_LENGTH = int(config['db_length'])
    BATCH_SIZE = int(config['batch_size'])
    checkpoint_filepath = config['checkpoint_filepath'] + 'model.{epoch:02d}-{val_loss:.2f}'
    EPOCHS = int(config['epochs'])
    log_dir = config['log_dir'] + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


    path = config['stockfish_path']
    engine = chess.engine.SimpleEngine.popen_uci(path)
    chess.engine.Limit(depth=config['stockfish_depth'])

    model = get_model()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',
        baseline=None, restore_best_weights=False
    )
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        save_freq='epoch',
        period=1,
        save_best_only=False
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    gen = get_generator(engine, config)
    train_dataset = tf.data.Dataset.from_generator(
        gen,
        args=[DB_PATH],
        output_signature=((tf.SparseTensorSpec((41024,)), tf.SparseTensorSpec((41024,))), tf.TensorSpec(()))
    )
    test_dataset = tf.data.Dataset.from_generator(
        gen,
        args=[DB_TEST_PATH],
        output_signature=((tf.SparseTensorSpec((41024,)), tf.SparseTensorSpec((41024,))), tf.TensorSpec(()))
    )
    print(model.summary())

    steps = round(DB_LENGTH / BATCH_SIZE)

    train_dataset = train_dataset.repeat(EPOCHS)

    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])


    model.fit(
        train_dataset.batch(BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=test_dataset.batch(15),
        steps_per_epoch=steps,
        validation_freq=1,
        callbacks=[early_stopping, model_checkpoint_callback, tensorboard_callback]
    )

    pdb.set_trace()
