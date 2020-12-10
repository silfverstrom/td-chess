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

    #input_material = l.Input(shape=(19))
    #x3 = l.Dense(64)(input_material)
    #x3 = l.ReLU()(x3)

    x = l.Concatenate()([x1, x2])
    x = l.Dense(32)(x)
    x = l.ReLU()(x)
    x = l.Dense(32)(x)
    x = l.ReLU()(x)

    output =l.Dense(1, activation='linear')(x)

    model = tf.keras.Model(
        #inputs=[input_player, input_opp, input_material],
        inputs=[input_player, input_opp],
        outputs=output
    )

    return model

# experimetn
def loss_fn(outcome, score, pred, lambda_):
  q = pred
  t = outcome
  p = util.cp_conversion(score)
  #print(t.size())
  #print(p.size())
  #print(pred.size())
  epsilon = 1e-12
  teacher_entropy = -(p * (p + epsilon).log() + (1.0 - p) * (1.0 - p + epsilon).log())
  outcome_entropy = -(t * (t + epsilon).log() + (1.0 - t) * (1.0 - t + epsilon).log())

  teacher_loss = -(p * F.logsigmoid(q) + (1.0 - p) * F.logsigmoid(-q))
  outcome_loss = -(t * F.logsigmoid(q) + (1.0 - t) * F.logsigmoid(-q))
  result  = lambda_ * teacher_loss    + (1.0 - lambda_) * outcome_loss
  entropy = lambda_ * teacher_entropy + (1.0 - lambda_) * outcome_entropy
  #print(result.size())
  return result.mean() - entropy.mean()
if __name__ == '__main__':
    model = get_model()
    print(model.summary())
