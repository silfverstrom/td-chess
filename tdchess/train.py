from tdchess.tables import pawntable, knightstable, bishopstable, rookstable, queenstable, kingstable
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
    wn = len(board.pieces(chess.KNIGHT, chess.WHITE))
    bn = len(board.pieces(chess.KNIGHT, chess.BLACK))
    wb = len(board.pieces(chess.BISHOP, chess.WHITE))
    bb = len(board.pieces(chess.BISHOP, chess.BLACK))
    wr = len(board.pieces(chess.ROOK, chess.WHITE))
    br = len(board.pieces(chess.ROOK, chess.BLACK))
    wq = len(board.pieces(chess.QUEEN, chess.WHITE))
    bq = len(board.pieces(chess.QUEEN, chess.BLACK))

    #material = 100*(wp-bp)+320*(wn-bn)+330*(wb-bb)
    #material = 100*(wp-bp)+320*(wn-bn)
    #material = 100*(wp-bp)+320*(wn-bn)+500*(wr-br)
    material = 100*(wp-bp)+320*(wn-bn)+330*(wb-bb)+500*(wr-br)+900*(wq-bq)

    pawnsq = sum([pawntable[i] for i in board.pieces(chess.PAWN, chess.WHITE)])
    pawnsq= pawnsq + sum([-pawntable[chess.square_mirror(i)]
                                    for i in board.pieces(chess.PAWN, chess.BLACK)])
    knightsq = sum([knightstable[i] for i in board.pieces(chess.KNIGHT, chess.WHITE)])
    knightsq = knightsq + sum([-knightstable[chess.square_mirror(i)]
                                    for i in board.pieces(chess.KNIGHT, chess.BLACK)])
    bishopsq= sum([bishopstable[i] for i in board.pieces(chess.BISHOP, chess.WHITE)])
    bishopsq= bishopsq + sum([-bishopstable[chess.square_mirror(i)]
                                    for i in board.pieces(chess.BISHOP, chess.BLACK)])
    rooksq = sum([rookstable[i] for i in board.pieces(chess.ROOK, chess.WHITE)])
    rooksq = rooksq + sum([-rookstable[chess.square_mirror(i)]
                                    for i in board.pieces(chess.ROOK, chess.BLACK)])
    queensq = sum([queenstable[i] for i in board.pieces(chess.QUEEN, chess.WHITE)])
    queensq = queensq + sum([-queenstable[chess.square_mirror(i)]
                                    for i in board.pieces(chess.QUEEN, chess.BLACK)])
    kingsq = sum([kingstable[i] for i in board.pieces(chess.KING, chess.WHITE)])
    kingsq = kingsq + sum([-kingstable[chess.square_mirror(i)]
                                    for i in board.pieces(chess.KING, chess.BLACK)])

    material = material + pawnsq + knightsq + bishopsq+ rooksq+ queensq + kingsq
    #material = pawnsq + knightsq + bishopsq+ rooksq+ queensq + kingsq
    return material

def get_bitboard(board, piece_type):
    white_pawns = [0 for _ in range(64)]
    for position in board.pieces(piece_type, chess.WHITE):
        white_pawns[position] = 1
    black_pawns = [0 for _ in range(64)]
    for position in board.pieces(piece_type, chess.BLACK):
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

        x = []
        pawns = get_bitboard(board, chess.PAWN)
        knights = get_bitboard(board, chess.KNIGHT)
        bishops = get_bitboard(board, chess.BISHOP)
        rooks = get_bitboard(board, chess.ROOK)
        queens = get_bitboard(board, chess.QUEEN)
        kings = get_bitboard(board, chess.KING)

        x.extend(pawns)
        x.extend(knights)
        x.extend(bishops)
        x.extend(rooks)
        x.extend(queens)
        x.extend(kings)

        X.append(x)
        Y.append(y)

    return X,Y

def get_fens():
    with open('fens.npy', 'rb') as f:
        return np.load(f)
def save_fens():
    pgn = open("/Users/silfverstrom/Workspace/link/projects/chess_engine/data/lichess_db_standard_rated_2014-08.pgn")

    fens = []
    count = 0
    while count < 500:
        game = chess.pgn.read_game(pgn)
        board = chess.Board()
        for move in game.mainline_moves():
            board.push(move)
            fen = board.fen()
            fens.append(fen)
        count = count + 1

    fens = np.array(fens)
    with open('fens.npy', 'wb') as f:
        np.save(f, fens)
    print('Fens saved')
def get_model():
    l = tf.keras.layers
    model = tf.keras.models.Sequential([
        l.Dense(64),
        #l.Dense(100, activation='relu'),
        #l.Dropout(0.2),
        l.Dense(1, activation='linear'),
    ])
    return model

def generate_data():
    fens = get_fens()

    X,Y = get_training_data(fens)

    with open('data.npy', 'wb') as f:
        np.savez(f, X=X, Y=Y)
if __name__ == '__main__':
    #fen = "r1b1r1k1/ppp3pp/5b2/5p2/2P5/1P2p1PP/P2NPRB1/5RK1 w - - 0 25"
    #save_fens()


    #generate_data()
    X = None
    Y = None
    with open('data.npy', 'rb') as f:
        data = np.load(f)
        X = data['X']
        Y = data['Y']
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y)

    # Normalize values
    #pdb.set_trace()
    #Y_train = tf.keras.utils.normalize(Y_train, axis=0)[0]
    #Y_test = tf.keras.utils.normalize(Y_test, axis=0)[0]

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
        baseline=None, restore_best_weights=False
    )


    model = get_model()

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae'])
    model.fit(X_train,Y_train, epochs=150, validation_data=(X_test, Y_test),
              batch_size=32, validation_freq=1, callbacks=[early_stopping])

    pdb.set_trace()
