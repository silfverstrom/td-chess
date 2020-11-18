import chess
import chess.engine
import numpy as np
import chess.pgn
import os
import sys

import pdb
import tensorflow as tf
from training_data import get_training_data
import datetime
import configparser
import random

def get_model():
    l = tf.keras.layers

    input1 = l.Input(shape=(774))
    x1 = l.Dense(2048, use_bias=False)(input1)
    x1 = l.BatchNormalization()(x1)
    x1 = l.ReLU()(x1)
    x1 = l.Dense(64, use_bias=False)(x1)
    x1 = l.BatchNormalization()(x1)
    x1 = l.ReLU()(x1)
    x1 = l.Dense(32)(x1)
    x1 = l.ReLU()(x1)

    input2 = l.Input(shape=(15))
    x2 = l.Dense(32)(input2)
    x2 = l.ReLU()(x2)

    x = l.Concatenate()([x1, x2])
    x = l.Dense(32)(x)
    x = l.ReLU()(x)
    x = l.Dropout(0.5)(x)

    output =l.Dense(1, activation='linear')(x)

    model = tf.keras.Model(inputs=[input1, input2], outputs=output)

    return model

def get_generator(engine, config):
    def gen(path):
        f = open(path)
        while True:
            line = f.readline()
            if not line:
                break

            split = line.split(',')
            epd_line = split[0]
            y = float(split[1]) / 100
            try:
                board = chess.Board().from_epd(epd_line)[0]
                x, x1 = get_training_data(board)
            except Exception as e:
                continue

            yield (x, x1), y
    return gen


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
    optimizer = tf.keras.optimizers.Adam()

    model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',
        baseline=None, restore_best_weights=False
    )
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        save_freq='epoch',
        period=1,
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    gen = get_generator(engine, config)
    train_dataset = tf.data.Dataset.from_generator(
        gen,
        args=[DB_PATH],
        output_signature=((tf.TensorSpec(774,), tf.TensorSpec(15)), tf.TensorSpec(None))
    )
    test_dataset = tf.data.Dataset.from_generator(
        gen,
        args=[DB_TEST_PATH],
        output_signature=((tf.TensorSpec(774,), tf.TensorSpec(15)), tf.TensorSpec(None))
    )
    print(model.summary())

    steps = round(DB_LENGTH / BATCH_SIZE)

    train_dataset = train_dataset.repeat(EPOCHS)


    model.fit(
        train_dataset.batch(BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=test_dataset.batch(15),
        steps_per_epoch=steps,
        validation_freq=1,
        callbacks=[early_stopping, model_checkpoint_callback, tensorboard_callback]
    )

    pdb.set_trace()
