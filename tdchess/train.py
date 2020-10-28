from tables import pawntable
import chess
import chess.engine
import numpy as np
import chess.pgn

import pdb
import tensorflow as tf
from sklearn.model_selection import train_test_split

def evaluate(board):
    wp = len(board.pieces(chess.PAWN, chess.WHITE))
    bp = len(board.pieces(chess.PAWN, chess.BLACK))
    material = 100*(wp-bp)

    pawnsq = sum([pawntable[i] for i in board.pieces(chess.PAWN, chess.WHITE)])
    pawnsq= pawnsq + sum([-pawntable[chess.square_mirror(i)]
                                    for i in board.pieces(chess.PAWN, chess.BLACK)])

    material = material + pawnsq
    return material

def get_bitboard(board):
    white_pawns = [0 for _ in range(64)]
    for position in board.pieces(chess.PAWN, chess.WHITE):
        white_pawns[position] = 1
    black_pawns = [0 for _ in range(64)]
    for position in board.pieces(chess.PAWN, chess.BLACK):
        black_pawns[position] = 1
    pieces = white_pawns
    pieces.extend(black_pawns)

    return pieces

def get_training_data(fens):
    X = []
    Y = []
    for fen in fens:
        board = chess.Board(fen)
        y = evaluate(board)
        x = get_bitboard(board)

        X.append(x)
        Y.append(y)

    return X,Y

def get_fens():
    pgn = open("/Users/silfverstrom/Workspace/link/projects/chess_engine/data/lichess_db_standard_rated_2014-08.pgn")

    fens = []
    count = 0
    while count < 100:
        game = chess.pgn.read_game(pgn)
        board = chess.Board()
        for move in game.mainline_moves():
            board.push(move)
            fen = board.fen()
            fens.append(fen)
        count = count + 1
    return fens
def get_model():
    l = tf.keras.layers
    model = tf.keras.models.Sequential([
        l.Dense(64, activation='relu'),
        l.Dense(64, activation='relu'),
        #l.Dense(100, activation='relu'),
        #l.Dense(100, activation='relu'),
        #l.Dropout(0.2),
        l.Dense(1, activation='linear'),
    ])
    return model

if __name__ == '__main__':
    #fen = "r1b1r1k1/ppp3pp/5b2/5p2/2P5/1P2p1PP/P2NPRB1/5RK1 w - - 0 25"
    fens = get_fens()

    model = get_model()
    X,Y = get_training_data(fens)
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y)

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae'])
    model.fit(X_train,Y_train, epochs=100, validation_data=(X_test, Y_test), validation_freq=10)

    pdb.set_trace()
