import chess
import mmap
import random
import os
import numpy as np
import sys

#PACKED_SFEN_VALUE_BYTES = 40
#PACKED_SFEN_VALUE_BYTES = 40*8
#PACKED_SFEN_VALUE_BYTES = 32
PACKED_SFEN_VALUE_BYTES = 40

HUFFMAN_MAP = {0b000 : chess.PAWN, 0b001 : chess.KNIGHT, 0b010 : chess.BISHOP, 0b011 : chess.ROOK, 0b100: chess.QUEEN}

''' stockfish
    //
    // Huffman Encoding
    //
    // Empty  xxxxxxx0
    // Pawn   xxxxx001 + 1 bit (Color)
    // Knight xxxxx011 + 1 bit (Color)
    // Bishop xxxxx101 + 1 bit (Color)
    // Rook   xxxxx111 + 1 bit (Color)
    // Queen   xxxx1001 + 1 bit (Color)
    //
    // Worst case:
    // - 32 empty squares    32 bits
    // - 30 pieces           150 bits
    // - 2 kings             12 bits
    // - castling rights     4 bits
    // - ep square           7 bits
    // - rule50              7 bits
    // - game ply            16 bits
    // - TOTAL               228 bits < 256 bits


    constexpr HuffmanedPiece huffman_table[] =
    {
        {0b0000,1}, // NO_PIECE
        {0b0001,4}, // PAWN
        {0b0011,4}, // KNIGHT
        {0b0101,4}, // BISHOP
        {0b0111,4}, // ROOK
        {0b1001,4}, // QUEEN
'''

def twos(v, w):
  return v - int((v << 1) & 2**w)

class BitReader():
  def __init__(self, bytes, at):
    self.bytes = bytes
    self.seek(at)

  def readBits(self, n):
    r = self.bits & ((1 << n) - 1)
    self.bits >>= n
    self.position -= n
    return r

  def refill(self):
    while self.position <= 24:
      self.bits |= self.bytes[self.at] << self.position
      self.position += 8
      self.at += 1

  def seek(self, at):
    self.at = at
    self.bits = 0
    self.position = 0
    self.refill()

def is_quiet(board, from_, to_):
  for mv in board.legal_moves:
    if mv.from_square == from_ and mv.to_square == to_:
      return not board.is_capture(mv)
  return False

class ToTensor(object):
  def __call__(self, sample):
    bd, _, outcome, score = sample
    #FIXME convert to tensor
    us = bd.turn
    them = not bd.turn
    outcome = [outcome]
    score = [score]
    # FIXME get data
    #white, black = halfkp.get_halfkp_indices(bd)
    white = black = -9999.0
    return us.float(), them.float(), white.float(), black.float(), outcome.float(), score.float()

class RandomFlip(object):
  def __call__(self, sample):
    bd, move, outcome, score = sample
    mirror = random.choice([False, True])
    if mirror:
      bd = bd.mirror()
    return bd, move, outcome, score

class NNUEBinData():
  def __init__(self, filename):
    super(NNUEBinData, self).__init__()
    self.filename = filename
    self.len = os.path.getsize(filename) // PACKED_SFEN_VALUE_BYTES
    self.file = None

  def __len__(self):
    return self.len

  def get_raw_exp(self, idx):
    if self.file is None:
      self.file = open(self.filename, 'r+b')
      self.bytes = mmap.mmap(self.file.fileno(), 0)
    base = PACKED_SFEN_VALUE_BYTES * idx
    br = BitReader(self.bytes, base)

    turn = not br.readBits(1)
    white_king_sq = br.readBits(6)
    black_king_sq = br.readBits(6)

    br.readBits(1)
    for i in range(64):
        piece = br.readBits(3)
        color = br.readBits(1)
        print('piece', piece, color)
        print('first', HUFFMAN_MAP[piece])

    #rest = br.readBits(256-17)

    print(turn, white_king_sq, black_king_sq)
  def get_raw(self, idx):
    if self.file is None:
      self.file = open(self.filename, 'r+b')
      self.bytes = mmap.mmap(self.file.fileno(), 0)

    base = PACKED_SFEN_VALUE_BYTES * idx
    br = BitReader(self.bytes, base)

    bd = chess.Board(fen=None)
    bd.turn = not br.readBits(1)
    white_king_sq = br.readBits(6)
    black_king_sq = br.readBits(6)
    bd.set_piece_at(white_king_sq, chess.Piece(chess.KING, chess.WHITE))
    bd.set_piece_at(black_king_sq, chess.Piece(chess.KING, chess.BLACK))

    assert(black_king_sq != white_king_sq)

    for rank_ in range(8)[::-1]:
      br.refill()
      for file_ in range(8):
        i = chess.square(file_, rank_)
        if white_king_sq == i or black_king_sq == i:
          continue
        if br.readBits(1):
          assert(bd.piece_at(i) == None)
          piece_index = br.readBits(3)
          piece = HUFFMAN_MAP[piece_index]
          color = br.readBits(1)
          bd.set_piece_at(i, chess.Piece(piece, not color))
          br.refill()

    br.seek(base + 32)
    score = twos(br.readBits(16), 16)
    move = br.readBits(16)
    to_ = move & 63
    from_ = (move & (63 << 6)) >> 6

    br.refill()
    ply = br.readBits(16)
    bd.fullmove_number = ply // 2

    move = chess.Move(from_square=chess.SQUARES[from_], to_square=chess.SQUARES[to_])

    # 1, 0, -1
    game_result = br.readBits(8)
    outcome = {1: 1.0, 0: 0.5, 255: 0.0}[game_result]
    return bd, move, outcome, score

  def __getitem__(self, idx):
    item = self.get_raw(idx)
    # FIXME transform
    return item
    #return self.transform(item)

  # Allows this class to be pickled (otherwise you will get file handle errors).
  def __getstate__(self):
    state = self.__dict__.copy()
    state['file'] = None
    state.pop('bytes', None)
    return state

if __name__ == '__main__':
    bin_data = NNUEBinData(sys.argv[1])
    for i in bin_data:
        print(i)
