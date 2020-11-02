from tdchess.tables import pawntable, knightstable, bishopstable, rookstable, queenstable, kingstable
import chess
import chess.engine
import numpy as np
import chess.pgn

def test_fens():
    path = "/usr/local/Cellar/stockfish/12/bin/stockfish"
    engine = chess.engine.SimpleEngine.popen_uci(path)
    chess.engine.Limit(depth=0)

    fens = []
    Y = []

    pgn = open("/Users/silfverstrom/Workspace/link/projects/chess_engine/data/lichess_db_standard_rated_2014-08.pgn")
    count = 0
    while count < 100000:
        game = chess.pgn.read_game(pgn)
        board = chess.Board()

        for move in game.mainline_moves():
            board.push(move)
            ev = engine.analyse(board, chess.engine.Limit(depth=0))
            try:
                y = float(str(ev['score'].white()))
                fen = board.fen()
                fens.append(fen)
                Y.append(y)
            except:
                #y = float(str(ev['score'].white())[1:])
                pass

        if count % 5000 == 0:
            print(count)
            with open('data/stockfish_{}.npy'.format(count), 'wb') as f:
                np.savez(f, X=fens, Y=Y)
            fens = []
            Y = []
        count = count + 1
if __name__ == '__main__':
    test_fens()
