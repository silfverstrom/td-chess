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

def get_training_data(board):

    x = []
    # add meta
    tt = [1,0] if board.turn else [0,1]
    piece_map = board.piece_map()
    empty = [0 for _ in range(12)]
    pieces = ['p','n','b','r','q','k','P','N','B','R','Q','K']

    rep = [empty.copy() for _ in range(8*8)]

    castle_rights = [0,0,0,0]
    if board.has_kingside_castling_rights(chess.WHITE):
        castle_rights[0] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        castle_rights[1] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        castle_rights[2] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        castle_rights[3] = 1

    for key in piece_map:
        val = str(piece_map[key])
        ind = pieces.index(val)

        label = None
        if ind <= 5 and not board.turn: #black piece, black to move
            label = 1
        if ind <= 5 and board.turn: #black piece, white to move
            label = -1

        if ind > 5 and board.turn: #white piece, white to move
            label = 1
        if ind > 5 and not board.turn: #black piece, black to move
            label = -1

        rep[key][ind] = label
    rep = np.array(rep).flatten()
    x.extend(rep)
    x.extend(tt)
    x.extend(castle_rights)

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

    pieces = [wp, bp, wn, bn, wb, bb, wr, br, wq, bq]

    pieces = normalise(pieces)
    diff = [wp - bp, wn - bn, wb - bb, wr - br, wq - bq]
    diff = normalise(diff)
    x.extend(pieces)
    x.extend(diff)

    return x

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
