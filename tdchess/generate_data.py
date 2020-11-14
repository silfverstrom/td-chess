from tables import pawntable, knightstable, bishopstable, rookstable, queenstable, kingstable
import chess
import chess.engine
import numpy as np
import chess.pgn

import pdb
path = "/usr/games/stockfish"
engine = chess.engine.SimpleEngine.popen_uci(path)
chess.engine.Limit(depth=0)


def evaluate_stockfish(board):
    ev = engine.analyse(board, chess.engine.Limit(depth=0))
    try:
        y = float(str(ev['score'].relative))
    except BaseException:
        y = float(str(ev['score'].relative)[1:])*1000
    return y


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
    pawnsq = pawnsq + sum([-pawntable[chess.square_mirror(i)]
                           for i in board.pieces(chess.PAWN, chess.BLACK)])
    knightsq = sum([knightstable[i]
                    for i in board.pieces(chess.KNIGHT, chess.WHITE)])
    knightsq = knightsq + sum([-knightstable[chess.square_mirror(i)]
                               for i in board.pieces(
                                  chess.KNIGHT, chess.BLACK)])
    bishopsq = sum([bishopstable[i]
                    for i in board.pieces(chess.BISHOP, chess.WHITE)])
    bishopsq = bishopsq + sum([-bishopstable[chess.square_mirror(i)]
                               for i in board.pieces(
                                  chess.BISHOP, chess.BLACK)])
    rooksq = sum([rookstable[i]
                  for i in board.pieces(chess.ROOK, chess.WHITE)])
    rooksq = rooksq + sum([-rookstable[chess.square_mirror(i)]
                           for i in board.pieces(chess.ROOK, chess.BLACK)])
    queensq = sum([queenstable[i]
                   for i in board.pieces(chess.QUEEN, chess.WHITE)])
    queensq = queensq + sum([-queenstable[chess.square_mirror(i)]
                             for i in board.pieces(chess.QUEEN, chess.BLACK)])
    kingsq = sum([kingstable[i]
                  for i in board.pieces(chess.KING, chess.WHITE)])
    kingsq = kingsq + sum([-kingstable[chess.square_mirror(i)]
                           for i in board.pieces(chess.KING, chess.BLACK)])

    material = material + pawnsq + knightsq + bishopsq + rooksq + queensq + kingsq
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


def get_training_data(fens, stockfish=False):
    X = []
    Y = []
    for fen in fens:
        board = chess.Board(fen)
        if stockfish:
            y = evaluate_stockfish(board)
        else:
            y = evaluate(board)

        #pawns = get_bitboard(board, chess.PAWN)
        #knights = get_bitboard(board, chess.KNIGHT)
        #bishops = get_bitboard(board, chess.BISHOP)
        #rooks = get_bitboard(board, chess.ROOK)
        #queens = get_bitboard(board, chess.QUEEN)
        #kings = get_bitboard(board, chess.KING)

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
        # add bitboards
        # x.extend(meta)
        # x.extend(pawns)
        # x.extend(knights)
        # x.extend(bishops)
        # x.extend(rooks)
        # x.extend(queens)
        # x.extend(kings)
        x.extend(rep)

        X.append(x)
        Y.append(y)

    return X, Y


def get_fens():
    with open('fens.npy', 'rb') as f:
        return np.load(f)


def save_fens():
    pgn = open(
        "/home/niklas/lichess_db_standard_rated_2014-08.pgn")

    fens = []
    count = 0
    while count < 50000:
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




def generate_data(stockfish=False):
    fens = get_fens()
    X, Y = get_training_data(fens, stockfish)

    if stockfish:
        with open('stockfish.npy', 'wb') as f:
            np.savez(f, X=X, Y=Y)
    else:
        with open('data.npy', 'wb') as f:
            np.savez(f, X=X, Y=Y)


if __name__ == '__main__':
    #fen = "r1b1r1k1/ppp3pp/5b2/5p2/2P5/1P2p1PP/P2NPRB1/5RK1 w - - 0 25"
    #save_fens()

    stockfish = True
    generate_data(stockfish)
