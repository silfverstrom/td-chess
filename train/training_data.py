import chess
import chess.engine
import numpy as np
import chess.pgn
import pdb

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


def get_training_data(board):
    ''' relative training data based on turn'''
    # add meta
    piece_map = board.piece_map()
    #empty = [0 for _ in range(12)]
    pieces = ['p','n','b','r','q','P','N','B','R','Q']

    #white = [empty.copy() for _ in range(8*8)]
    #black = [empty.copy() for _ in range(8*8)]

    white = np.zeros((64,64,10))
    black = np.zeros((64,64,10))

    wki = board.pieces(chess.KING, chess.WHITE).pop()
    bki = board.pieces(chess.KING, chess.BLACK).pop()

    for key in piece_map:
        val = str(piece_map[key])
        try:
            ind = pieces.index(val)
        except:
            continue # kings

        label = None
        if ind > 5:
            white[wki][key][ind] = 1
        elif ind <= 5:
            black[bki][key][ind] = 1

    white = white.flatten()
    black = black.flatten()

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

    if board.turn:
        pieces = [wp, bp, wn, bn, wb, bb, wr, br, wq, bq]
        diff = [wp - bp, wn - bn, wb - bb, wr - br, wq - bq]
    else:
        pieces = [bp, wp, bn, wn, bb, wb, br, wr, bq, wq]
        diff = [bp - wp, bn - wn, bb - wb, br - wr, bq - wq]

    pieces = normalise(pieces)
    diff = normalise(diff)
    meta = pieces
    meta.extend(diff)

    castle_rights = [0,0,0,0]
    if board.has_kingside_castling_rights(chess.WHITE):
        castle_rights[0] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        castle_rights[1] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        castle_rights[2] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        castle_rights[3] = 1
    #meta.extend(castle_rights)
    #meta = np.array(meta)

    if board.turn:
        #return white, black, meta
        return white, black
    else:
        #return black, white, meta
        return black, white


if __name__ == '__main__':
    out = get_training_data(chess.Board())
    print(out)
    pdb.set_trace()
