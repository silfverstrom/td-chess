import chess
import chess.engine

path = "/usr/local/Cellar/stockfish/12/bin/stockfish"
engine = chess.engine.SimpleEngine.popen_uci(path)
chess.engine.Limit(depth=4)

if __name__ == '__main__':

    f = open('/Users/silfverstrom/Documents/data/chess/tuner/quiet-labeled.epd')
    fr = open('data/tuner_quiet.epd', 'w')
    for i in range(50):
        line = f.readline()
        board = chess.Board().from_epd(line)[0]

        ev = engine.analyse(board, chess.engine.Limit(depth=0))
        y = ev['score'].white()
        print(board.fen(), y)

    f.close()
    fr.close()
