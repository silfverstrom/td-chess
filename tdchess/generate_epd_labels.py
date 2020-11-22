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
fw = open('/Users/silfverstrom/Documents/data/chess/tuner/quiet-labeled_stockfish_depth0_random1.epd', 'w')

count = 0
while True:
    line = f.readline().strip()
    if not line:
        break
    board = chess.Board().from_epd(line)[0]

    legal = list(board.legal_moves)
    random.shuffle(legal)
    mx = 4
    if len(legal) < 4:
        mx = len(legal)
    for i in range(mx):
        move = legal[i]
        # no capture moves
        if board.is_capture(move):
            continue
        board.push(move)
        ev = engine.analyse(board, chess.engine.Limit(depth=0))
        try:
            y = float(str(ev['score'].white()))
            out = "{},{}\n".format(board.epd(), y)
            fw.write(out)
        except:
            pass
        board.pop()

    ev = engine.analyse(board, chess.engine.Limit(depth=0))
    try:
        y = float(str(ev['score'].white()))
        out = "{},{}\n".format(line, y)
        fw.write(out)
        count = count + 1
        if count % 1000 == 0:
            print("step ", count)
    except Exception as e:
        continue
