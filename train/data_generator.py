import chess
import chess.engine
import numpy as np
import chess.pgn
import os
import sys
import time
import pdb
import tensorflow as tf
from training_data import get_training_data
import datetime
import configparser
import random

from bin_data import NNUEBinData

def get_generator():
    def gen(path):
        bin_generator = NNUEBinData(path)
        for data in bin_generator:
            board = data[0]
            move = data[1]
            result = data[2]
            score = data[3]
            try:
                x, x2, x3 = get_training_data(board)
            except Exception as e:
                continue

            yield (x, x2, x3), score
    return gen

def mapper(X, y):
    x1 = X[0]
    x2 = X[1]
    x3 = X[2]

    print(y)

    y = tf.math.sigmoid((y / 600))
    #p = (score / 600.0).sigmoid()

    return (x1, x2, x3), y
def get_dataset(db_path, batch_size=200):
    gen = get_generator()
    train_dataset = tf.data.Dataset.from_generator(
        gen,
        args=[db_path],
        #output_signature=((tf.TensorSpec(768,), tf.TensorSpec(768,), tf.TensorSpec(19)), tf.TensorSpec(None))
        output_signature=((tf.TensorSpec(40960,), tf.TensorSpec(40960,), tf.TensorSpec(19)), tf.TensorSpec(None))
        #output_signature=((tf.SparseTensorSpec((774,)), tf.SparseTensorSpec((15,))), tf.TensorSpec(None))
    )

    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.map(mapper)
    train_dataset = train_dataset.prefetch(batch_size*4)

    return train_dataset
def benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        count = 0
        for sample in dataset:
            count = count + 1
            if count > 1000:
                break
            # Performing a training step
            time.sleep(0.001)
            pass
    tf.print("Execution time:", time.perf_counter() - start_time)

def logit(x):
    return - tf.math.log(1. / x - 1.)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Must be run with args')
        sys.exit()
    config_path = sys.argv[1]

    config = configparser.ConfigParser()
    config.read(config_path)
    config = dict(config['DEFAULT'])
    DB_PATH = config['db_path']
    DB_TEST_PATH = config['db_test_path']

    train_dataset = get_dataset(DB_PATH)

    for val in train_dataset.batch(1).take(100).as_numpy_iterator():
        print(val[1].mean(), val[1].std(), logit(val[1].mean())*600)
        pdb.set_trace()


    #benchmark(
    #    train_dataset
    #)
