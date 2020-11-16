import chess
from enum import Enum
from enum import IntFlag
import numpy as np
import tensorflow as tf
import pdb
import timeit
#from halfkp import get_halfkp_indeces


model = tf.keras.models.load_model('output/nnue_0.1.2')
SQUARE_NB = 64
class PieceSquare(IntFlag):
    NONE = 0,
    W_PAWN = 1,
    B_PAWN = 1 * SQUARE_NB + 1
    W_KNIGHT = 2 * SQUARE_NB + 1
    B_KNIGHT = 3 * SQUARE_NB + 1
    W_BISHOP = 4 * SQUARE_NB + 1
    B_BISHOP = 5 * SQUARE_NB + 1
    W_ROOK = 6 * SQUARE_NB + 1
    B_ROOK = 7 * SQUARE_NB + 1
    W_QUEEN = 8 * SQUARE_NB + 1
    B_QUEEN = 9 * SQUARE_NB + 1
    W_KING = 10 * SQUARE_NB + 1
    END = W_KING  # pieces without kings (pawns included)
    B_KING = 11 * SQUARE_NB + 1
    END2 = 12 * SQUARE_NB + 1

    @staticmethod
    def from_piece(p: chess.Piece, is_white_pov: bool):
        return {
            chess.WHITE: {
                chess.PAWN: PieceSquare.W_PAWN,
                chess.KNIGHT: PieceSquare.W_KNIGHT,
                chess.BISHOP: PieceSquare.W_BISHOP,
                chess.ROOK: PieceSquare.W_ROOK,
                chess.QUEEN: PieceSquare.W_QUEEN,
                chess.KING: PieceSquare.W_KING
            },
            chess.BLACK: {
                chess.PAWN: PieceSquare.B_PAWN,
                chess.KNIGHT: PieceSquare.B_KNIGHT,
                chess.BISHOP: PieceSquare.B_BISHOP,
                chess.ROOK: PieceSquare.B_ROOK,
                chess.QUEEN: PieceSquare.B_QUEEN,
                chess.KING: PieceSquare.B_KING
            }
        }[p.color == is_white_pov][p.piece_type]

def orient(is_white_pov: bool, sq: int):
  # Use this one for "flip" instead of "rotate"
  # return (chess.A8 * (not is_white_pov)) ^ sq
  return (63 * (not is_white_pov)) ^ sq

def make_halfkp_index(is_white_pov: bool, king_sq: int, sq: int, p: chess.Piece):
  return orient(is_white_pov, sq) + PieceSquare.from_piece(p, is_white_pov) + PieceSquare.END * king_sq

# Returns SparseTensors
def get_halfkp_indeces(board: chess.Board):
  result = []
  is_white_pov = board.turn
  for i, turn in enumerate([board.turn, not board.turn]):
    indices = []
    values = []
    for sq, p in board.piece_map().items():
      if p.piece_type == chess.KING:
        continue
      indices.append([0, make_halfkp_index(turn, orient(turn, board.king(turn)), sq, p)])
      values.append(1)
    result.append(tf.sparse.reorder(tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=[1, 41024])))
  return result



def predict(fen):
    res = model.predict(get_halfkp_indeces(chess.Board(fen)))[0][0]
    return res

#time = timeit.timeit(lambda: predict('r3r1k1/3R1ppp/8/p7/1b6/4B3/1PP2PPP/5RK1 w - - 0 22'), number=1000)
#print("tme took", time)
#model = tf.keras.models.load_model('output/nneu_0.1')
if __name__ == '__main__':
    res = model.predict(get_halfkp_indeces(chess.Board()))[0][0]
    print(res)
    fen = 'r1bqkbnr/pppp1ppp/8/4p3/2n1P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 0 5'
    res = model.predict(get_halfkp_indeces(chess.Board(fen)))[0][0]
    print(res)

    fen = '3b1rk1/p6R/1p2r1p1/3p1p2/3B1p2/2PB4/PP3KP1/7R w - - 1 25'
    res = model.predict(get_halfkp_indeces(chess.Board(fen)))[0][0]
    print(res)

    fen = '2kr1b1r/ppp2p1p/2nqbnp1/4p3/2P5/3PP1P1/PB1NNPBP/R2QK2R b KQ - 0 10'
    res = model.predict(get_halfkp_indeces(chess.Board(fen)))[0][0]
    print(res)

    fen = 'r3r1k1/3R1ppp/8/p7/1b6/4B3/1PP2PPP/5RK1 w - - 0 22'
    res = model.predict(get_halfkp_indeces(chess.Board(fen)))[0][0]
    print(res)

    print(predict('rnb1kbnr/ppp1pp1p/3p3p/8/3PP1q1/8/PPP2PPP/RN2KBNR w KQkq - 0 5'))


    print(model.summary())
