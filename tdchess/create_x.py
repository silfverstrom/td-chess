from tables import pawntable, knightstable, bishopstable, rookstable, queenstable, kingstable
import chess
import chess.engine
import numpy as np
import chess.pgn
import os

import pdb
path = "/usr/games/stockfish"
engine = chess.engine.SimpleEngine.popen_uci(path)
chess.engine.Limit(depth=0)

def get_training_data(path):
    X = []
    Y = []
    data = np.load(path)
    for fen, y in zip(data['X'], data['Y']):
        board = chess.Board(fen)
        x = []
        # add meta
        tt = [1, 0] if board.turn else [0, 1]
        meta = tt
        piece_map = board.piece_map()
        empty = [0 for _ in range(12)]
        pieces = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']

        rep = [empty.copy() for _ in range(8*8)]

        for key in piece_map:
            val = str(piece_map[key])
            ind = pieces.index(val)

            label = None
            if ind <= 5 and not board.turn:  # black piece, black to move
                label = 1
            if ind <= 5 and board.turn:  # black piece, white to move
                label = -1

            if ind > 5 and board.turn:  # white piece, white to move
                label = 1
            if ind > 5 and not board.turn:  # black piece, black to move
                label = -1

            rep[key][ind] = label
        rep = np.array(rep).flatten()
        x.extend(rep)

        X.append(x)
        Y.append(y)

    return X, Y

def generate_data(stockfish=False):
    for path in os.listdir("data/"):
        X, Y = get_training_data("data/{}".format(path))
        with open('training_data/train_{}.npy'.format(path), 'wb') as f:
            np.savez(f, X=X, Y=Y)
            print("saved", path)

if __name__ == '__main__':
    generate_data()
