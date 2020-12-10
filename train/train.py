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

from bin_data import NNUEBinData

from model import get_model
from data_generator import get_dataset

def logit(x):
    return - tf.math.log(1. / x - 1.) * 600
def mae_scaled(y_true, y_pred):
    # scale
    y1 = tf.map_fn(lambda x: logit(x), y_true)
    y2 = tf.map_fn(lambda x: logit(x), y_pred)
    return tf.keras.losses.MAE(y1, y2)

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

    model = get_model()
    print(model.summary())

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

    train_dataset = get_dataset(DB_PATH)
    test_dataset = get_dataset(DB_TEST_PATH)
    steps = round(DB_LENGTH / BATCH_SIZE)

    train_dataset = train_dataset.repeat(EPOCHS)

    test_dataset = test_dataset.take(10)

    #model = tf.keras.models.load_model('/tmp/test')


    model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=test_dataset,
        steps_per_epoch=steps,
        validation_freq=1,
        callbacks=[early_stopping, model_checkpoint_callback, tensorboard_callback]
    )

    pdb.set_trace()
