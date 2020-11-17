from tdchess.tables import pawntable, knightstable, bishopstable, rookstable, queenstable, kingstable
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

def get_training_data(fen, stockfish=False):
    board = chess.Board(fen)

    x = []
    # add meta
    tt = [1,0] if board.turn else [0,1]
    meta = tt
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

    x1 = pieces
    x1.extend(diff)

    return x, x1

def test_fens():
    path = "/usr/local/Cellar/stockfish/12/bin/stockfish"
    engine = chess.engine.SimpleEngine.popen_uci(path)
    chess.engine.Limit(depth=0)

    fens = []
    Y = []
    X = []

    pgn = open("/Users/silfverstrom/Workspace/link/projects/chess_engine/data/lichess_db_standard_rated_2014-08.pgn")
    count = 0
    while count < 100000:
        game = chess.pgn.read_game(pgn)
        board = chess.Board()

        for move in game.mainline_moves():
            board.push(move)
            ev = engine.analyse(board, chess.engine.Limit(depth=0))

            fen = board.fen()
            x = get_training_data(fen)
            try:
                y = float(str(ev['score'].white()))

                fens.append(fen)
                Y.append(y)
                X.append(x)
            except:
                #y = float(str(ev['score'].white())[1:])
                pass

        if count % 5000 == 0:
            print(count)
            with open('data/stockfish_{}.npy'.format(count), 'wb') as f:
                np.savez(f, X=X, Y=Y, F=fens)
            fens = []
            Y = []
            X = []
        count = count + 1
if __name__ == '__main__':
    test_fens()
