import chess
import chess.engine
import numpy as np
import chess.pgn

import pdb
import tensorflow as tf
from sklearn.model_selection import train_test_split
from training_data import get_training_data

#path = "/usr/local/Cellar/stockfish/12/bin/stockfish"
#engine = chess.engine.SimpleEngine.popen_uci(path)
#chess.engine.Limit(depth=0)

DB_PATH = '/home/niklas/workspace/td-chess/tdchess/data_100k.csv'
DB_PATH_TEST = '/home/niklas/workspace/td-chess/tdchess/data_100k_test.csv'
DB_LENGTH = 6991225
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
        batch_x = []
        batch_y = []
        for _ in range(batch_size):
            line = f.readline().strip()
            if not line:
                f.seek(0)
                break
            data = line.split(',')
            fen = data[0]
            y = float(data[1])

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

            x = get_training_data(chess.Board(fen))
            batch_x.append(x)
            batch_y.append(y)
        yield np.array(batch_x), np.array(batch_y)


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

    steps = round(DB_LENGTH / BATCH_SIZE)

    model.fit(gen(DB_PATH), epochs=100, validation_data=gen(DB_PATH_TEST), steps_per_epoch=steps, validation_freq=1, callbacks=[early_stopping, model_checkpoint_callback])

    pdb.set_trace()
