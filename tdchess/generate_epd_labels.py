import  chess
import chess.engine
import numpy as np
import pdb
import random

def normalise(Y):
    mi = min(Y)
    mx = max(Y)
    Y_norm = []
    for y in Y:
        d = mx - mi
        if d == 0:
            d = 1
        y_ = (y - mi) / d
        Y_norm.append(y_)
    return Y_norm

path = "/usr/local/Cellar/stockfish/12/bin/stockfish"
engine = chess.engine.SimpleEngine.popen_uci(path)
chess.engine.Limit(depth=0)

X = []
Y = []

f = open('/Users/silfverstrom/Documents/data/chess/tuner/quiet-labeled.epd', 'r')
fw = open('/Users/silfverstrom/Documents/data/chess/tuner/quiet-labeled_stockfish_depth0.epd', 'w')

i = 0
while True:
    line = f.readline().strip()
    if not line:
        break
    board = chess.Board().from_epd(line)[0]
    ev = engine.analyse(board, chess.engine.Limit(depth=0))

    try:
        y = float(str(ev['score'].white()))
        out = "{},{}\n".format(line, y)
        fw.write(out)
        i = i + 1
        if i % 1000 == 0:
            print("step ", i)
    except Exception as e:
        continue
