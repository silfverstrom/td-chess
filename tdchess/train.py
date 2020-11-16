import chess
import chess.engine
import numpy as np
import chess.pgn

import pdb
import tensorflow as tf
from training_data import get_training_data

path = "/usr/local/Cellar/stockfish/12/bin/stockfish"
engine = chess.engine.SimpleEngine.popen_uci(path)
chess.engine.Limit(depth=4)

DB_PATH = '/Users/silfverstrom/Documents/data/chess/tuner/quiet-labeled.epd'
#DB_PATH_TEST = '/home/niklas/workspace/td-chess/tdchess/data_100k_test.csv'
#DB_PATH_TEST = '/home/niklas/workspace/td-chess/tdchess/data_100k_test.csv'
DB_LENGTH = 750e3
BATCH_SIZE = 256
checkpoint_filepath = '/tmp/tdchess_checkpoint/'

def get_model():
    l = tf.keras.layers
    model = tf.keras.models.Sequential([
        l.Dense(2048, activation='relu'),
        l.BatchNormalization(),
        l.Dense(2048, activation='relu'),
        l.BatchNormalization(),
        l.Dense(2048, activation='relu'),
        l.BatchNormalization(),
        l.Dropout(0.5),
        l.Dense(1, activation='linear'),
    ])
    return model

def gen(path, batch_size=256):
    f = open(path)
    while True:
        line = f.readline()

        if not line:
            f.seek(0)

        board = chess.Board().from_epd(line)[0]

        ev = engine.analyse(board, chess.engine.Limit(depth=0))
        try:
            y = float(str(ev['score'].white()))
            y = y / 100
        except Exception as e:
            continue

        x = get_training_data(board)

        yield (x), y


if __name__ == '__main__':

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
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )

    train_dataset = tf.data.Dataset.from_generator(
        gen,
        args=[DB_PATH],
        #output_signature=((tf.TensorSpec(()), tf.SparseTensorSpec(())), tf.TensorSpec(()))
        output_signature=((tf.TensorSpec(789,)), tf.TensorSpec(None))
    )
    print("HEJ", train_dataset.take(1))

    steps = round(DB_LENGTH / BATCH_SIZE)

    model.fit(
        train_dataset.batch(BATCH_SIZE),
        epochs=1,
        #validation_data=gen(DB_PATH_TEST),
        steps_per_epoch=steps,
        validation_freq=1,
        callbacks=[early_stopping, model_checkpoint_callback]
    )

    pdb.set_trace()
