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
xf = open('data/lichess_x.csv', 'w')
yf = open('data/lichess_y.csv', 'w')
fens = np.load('data/ccrl/fens_lichess_v1.npy')
print('Starting', len(fens))
for i, fen in enumerate(fens):

    board = chess.Board(fen)
    ev = engine.analyse(board, chess.engine.Limit(depth=0))

    fen = board.fen()
    x = get_training_data(board)
    try:
        y = float(str(ev['score'].white()))
    except Exception as e:
        continue

    xf.write(",".join([str(xi) for xi in x])+'\n')
    yf.write(str(y)+'\n');
    #X.append(x)
    #Y.append(y)

    if i % 1000 == 0:
        print("Step {}".format(i))
