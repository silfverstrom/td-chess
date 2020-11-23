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

FEATURE_TRANSFORM = 256

def get_model():
    l = tf.keras.layers

    input_player = l.Input(shape=(40960))
    input_opp = l.Input(shape=(40960))

    x1 = l.Dense(FEATURE_TRANSFORM)(input_player)
    x1 = l.ReLU()(x1)
    x2 = l.Dense(FEATURE_TRANSFORM)(input_opp)
    x2 = l.ReLU()(x2)

    input_material = l.Input(shape=(19))
    x3 = l.Dense(64)(input_material)
    x3 = l.ReLU()(x3)

    x = l.Concatenate()([x1, x2, x3])
    x = l.Dense(64)(x)
    x = l.ReLU()(x)
    x = l.Dense(32)(x)
    x = l.ReLU()(x)

    output =l.Dense(1, activation='linear')(x)

    model = tf.keras.Model(
        inputs=[input_player, input_opp, input_material],
        outputs=output
    )

    return model

if __name__ == '__main__':
    model = get_model()
    print(model.summary())
