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
def get_large_benchmark_model():
    l = tf.keras.layers

    input1 = l.Input(shape=(774))
    x1 = l.Dense(2048, activation='relu')(input1)
    x1 = l.BatchNormalization()(x1)
    x1 = l.Dense(2048, activation='relu')(x1)
    x1 = l.BatchNormalization()(x1)
    x1 = l.Dense(2048, activation='relu')(x1)

    input2 = l.Input(shape=(15))
    x2 = l.Dense(2048, activation='relu')(input2)

    x = l.Concatenate()([x1, x2])
    x = l.Dense(2048)(x)
    x = l.Dropout(0.5)(x)

    output =l.Dense(1, activation='linear')(x)

    model = tf.keras.Model(inputs=[input1, input2], outputs=output)

    return model
