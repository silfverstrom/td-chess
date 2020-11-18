from collections import namedtuple
import time
import numpy as np
from chess.polyglot import zobrist_hash

Entry = namedtuple('Entry', 'score depth')

def sort_moves(board, moves, transition_table):
    def score_move(move):
        if transition_table:
            board.push(move)
            hash_key = zobrist_hash(board)
            board.pop()
            entry = transition_table.get(hash_key)
            if entry:
                return entry.score

        score = int(board.is_capture(move)) * 9999
        score += int(board.gives_check(move)) * 1000
        return score

    #sorted_moves = moves.copy()
    sorted_moves = moves
    sorted_moves.sort(key=score_move, reverse=True)
    return sorted_moves

def iterative_deepening(board, depth, maximizingPlayer, evaluator, timelimit=None, transition_table=None):
    best = (-999999, None)
    legal_moves = list(board.legal_moves)
    np.random.shuffle(legal_moves)
    start = time.time()

    for d in range(depth - 1, depth if not timelimit else 100):
        dbest = (-999999, None)
        moves = sort_moves(board, legal_moves, transition_table)
        alpha = -99999
        beta = 99999
        for move in moves:
            board.push(move)

            score = alphabeta(board, d, maximizingPlayer, evaluator, transition_table, alpha, beta)

            print(move, score)
            if score > dbest[0]:
                dbest = (score, move)

            board.pop()

            if maximizingPlayer:
                alpha = max(alpha, score)
                if alpha >= beta: # Beta cutoff
                    break
            else:
                beta = min(beta, score)
                if beta <= alpha:
                    break

            if timelimit:
                dt = time.time() - start
                left = timelimit - dt
                if left < 0:
                    print('DEPTH', d - 1, dt)
                    return best

        best = dbest
        if timelimit:
            dt = time.time() - start
            left = timelimit - dt
            if left < 1:
                print('DEPTH', d, dt)
                return best

    return best

def alphabeta(board, depth, maximizingPlayer, evaluator, transition_table=None, alpha=-999999, beta=999999):
    moves = list(board.legal_moves)
    if depth == 0:
        v = quiesce(board, evaluator, maximizingPlayer, alpha, beta)
        if maximizingPlayer:
            return v
        else:
            return -v

    hash_key = None
    if transition_table is not None:
        hash_key = zobrist_hash(board)

    score = None
    if hash_key and hash_key in transition_table:
        entry = transition_table[hash_key]
        if entry.depth >= depth:
            return entry.score

    #moves = sort_moves(board, moves, transition_table)
    if maximizingPlayer:
        value = -99999
        for child in moves:
            board.push(child)
            value = max(value, alphabeta(board, depth - 1, False, evaluator, transition_table, alpha, beta))
            alpha = max(alpha, value)
            board.pop()
            if alpha >= beta: # Beta cutoff
                break
    else:
        value = 99999
        for child in moves:
            board.push(child)
            value = min(value, alphabeta(board, depth - 1, True, evaluator, transition_table, alpha, beta))
            beta = min(beta, value)
            board.pop()
            if beta <= alpha:
                break

    if hash_key:
        if hash_key in transition_table and transition_table[hash_key].depth >= depth:
            return value

        entry = Entry(score=value, depth=depth)
        transition_table[hash_key] = entry

    return value

def negamax(board, depth, evaluator, alpha, beta, color):
    if depth == 0:
        return [quiesce(board, evaluator, alpha, beta, color), None]
    _max = [-99999, None]
    moves = board.legal_moves
    for move in moves:
        board.push(move)
        score = -negamax(board, depth - 1, evaluator, -beta, -alpha, -color)[0]
        board.pop()
        if score > _max[0]:
            _max = [score, move]
        alpha = max(alpha, _max[0])
        if alpha >= beta:
            break
    return _max

def quiesce(board, evaluator, alpha, beta, color, depth=100):
    standpat = color * evaluator(board)
    if depth == 0:
        return standpat
    if standpat >= beta:
        return beta
    if alpha < standpat:
        alpha = standpat

    for child in board.legal_moves:
        if not board.is_capture(child):
            continue
        board.push(child)
        score = -quiesce(board, evaluator, -beta, -alpha, -color, depth - 1)
        board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha
