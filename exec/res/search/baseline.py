import random
import time

import search.checker as ck
import numpy

CAMPS = [(0, 3), (0, 4), (0, 5), (1, 4),
         (3, 8), (4, 8), (5, 8), (4, 7),
         (8, 3), (8, 4), (8, 5), (7, 4),
         (3, 0), (4, 0), (5, 0), (4, 1)]
CAPTURING_CAMPS = [c for c in CAMPS if c not in [(0, 4), (8, 4), (4, 0), (4, 8)]]
THRONE = [(4, 4)]
ESCAPES = [(0, 1), (0, 2), (0, 6), (0, 7),
           (8, 1), (8, 2), (8, 6), (8, 7),
           (1, 0), (2, 0), (6, 0), (7, 0),
           (1, 8), (2, 8), (6, 8), (7, 8)]


def move_baseline_pawn(move, board, print_debug):
    from_row = int(move[0][1:]) - 1
    from_col = ck.number_of(move[0][:1])
    to_row = int(move[1][1:]) - 1
    to_col = ck.number_of(move[1][:1])

    square_from = board[from_row][from_col]
    square_to = board[to_row][to_col]

    # if print_debug:
        # print("str(board)=")
        # print(str(board))
        # print("square_from: " + square_from + ", square_to: " + square_to)
        # print("from row col " + str(from_row) + " " + str(from_col) + ", to row col " + str(to_row) + " " + str(to_col))

    if (square_from != "WHITE" and square_from != "BLACK" and square_from != "KING") or square_to != "EMPTY":
        print("[ERROR] Illegal move in baseline square value. Terminating...")
        print("square_from: " + square_from + ", square_to: " + square_to)
        print("from row col " + str(from_row) + " " + str(from_col) + ", to row col " + str(to_row) + " " + str(to_col))
        exit(1)

    new_board = numpy.copy(board)

    new_board[from_row][from_col] = "EMPTY"

    if square_from == "WHITE":
        new_board[to_row][to_col] = "WHITE"
        king_move = False
    elif square_from == "KING":
        new_board[to_row][to_col] = "KING"
        king_move = True
    else:
        new_board[to_row][to_col] = "BLACK"
        king_move = False

    # if print_debug:
    #    print("str(new_board)=")
    #    print(str(new_board))

    black_pawns = []
    white_pawns = []
    empty_cells = []
    king_row, king_col = (-1, -1)
    for i, row in enumerate(new_board):
        for j, square in enumerate(row):
            if square == "BLACK":
                black_pawns.append((i, j))
            elif square == "WHITE":
                white_pawns.append((i, j))
            elif square == "EMPTY":
                empty_cells.append((i, j))
            elif square == "KING":
                king_row, king_col = (i, j)

    if (king_row, king_col) == (-1, -1):
        print("Error! The king is not in the board: " + board)
        exit(2)

    # needed to block_corridor() method
    previous_empty_cells = [cell for cell in board if cell == "EMPTY"]
    previous_black_pawns = [cell for cell in board if cell == "BLACK"]
    # The append is necessary to the correct elaboration of corridors() method
    empty_cells.append((to_row, to_col))
    return from_row, from_col, to_row, to_col, king_move, king_row, king_col, white_pawns, black_pawns, empty_cells, previous_empty_cells, previous_black_pawns


# try to capture pawn in (row, col)
def capture(to_row, to_col, row, col, attackers):
    # same row
    if (row, col - 1) in attackers and (row, col + 1) in attackers and (to_row, to_col) in [(row, col - 1), (row, col + 1)]:
        return True
    # same col
    if (row - 1, col) in attackers and (row + 1, col) in attackers and (to_row, to_col) in [(row - 1, col), (row + 1, col)]:
        return True

    return False


def capture_king(to_row, to_col, king_row, king_col, black_pawns):
    # 2 pawns capture
    king_special = [(4, 4), (3, 4), (5, 4), (4, 3), (4, 5)]
    if (king_row, king_col) not in king_special and capture(to_row, to_col, king_row, king_col, black_pawns):
        return True

    # Special case: King in throne or adjacent to throne
    # Throne
    if (king_row, king_col) == (4, 4) and (4, 3) in black_pawns and (4, 5) in black_pawns and (3, 4) in black_pawns and (5, 4) in black_pawns and (to_row, to_col) in [(4, 3), (4, 5), (3, 4), (5, 4)]:
        return True
    # Up
    elif (king_row, king_col) == (3, 4) and (3, 3) in black_pawns and (3, 5) in black_pawns and (2, 4) in black_pawns and (to_row, to_col) in [(3, 3), (3, 5), (2, 4)]:
        return True
    # Down
    elif (king_row, king_col) == (5, 4) and (5, 3) in black_pawns and (5, 5) in black_pawns and (6, 4) in black_pawns and (to_row, to_col) in [(5, 3), (5, 5), (6, 4)]:
        return True
    # Left
    elif (king_row, king_col) == (4, 3) and (3, 3) in black_pawns and (5, 3) in black_pawns and (4, 2) in black_pawns and (to_row, to_col) in [(3, 3), (5, 3), (4, 2)]:
        return True
    # Right
    elif (king_row, king_col) == (4, 5) and (3, 5) in black_pawns and (5, 5) in black_pawns and (4, 6) in black_pawns and (to_row, to_col) in [(3, 5), (5, 5), (4, 6)]:
        return True

    return False


def corridors(king_row, king_col):
    up = list(range(0, king_row))
    up_corridor = [(i, king_col) for i in up]
    down = list(range(king_row + 1, 9))
    down_corridor = [(i, king_col) for i in down]
    left = list(range(0, king_col))
    left_corridor = [(king_row, i) for i in left]
    right = list(range(king_col + 1, 9))
    right_corridor = [(king_row, i) for i in right]
    return up_corridor, down_corridor, left_corridor, right_corridor


def all_corridors(king_row, king_col):
    up_corridor, down_corridor, left_corridor, right_corridor = corridors(king_row, king_col)
    return up_corridor + down_corridor + left_corridor + right_corridor


def way_out(king_row, king_col, empty_cells):
    king_way_out = [(0, king_col), (8, king_col), (king_row, 0), (king_row, 8)]
    wasted = []
    up_corridor, down_corridor, left_corridor, right_corridor = corridors(king_row, king_col)

    for cell in up_corridor:
        if cell not in empty_cells:
            wasted.append((0, king_col))
            break
    for cell in down_corridor:
        if cell not in empty_cells:
            wasted.append((8, king_col))
            break
    for cell in left_corridor:
        if cell not in empty_cells:
            wasted.append((king_row, 0))
            break
    for cell in right_corridor:
        if cell not in empty_cells:
            wasted.append((king_row, 8))
            break

    return [cell for cell in king_way_out if cell not in wasted], up_corridor, down_corridor, left_corridor, right_corridor


def block_corridor(from_row, from_col, to_row, to_col, king_row, king_col, empty_cells):
    king_way_out, up_corridor, down_corridor, left_corridor, right_corridor = way_out(king_row, king_col, empty_cells)
    if king_way_out and (from_row, from_col) not in all_corridors(king_row, king_col):
        # Up
        if (0, king_col) in king_way_out and (to_row, to_col) in up_corridor:
            return True
        # Down
        elif (8, king_col) in king_way_out and (to_row, to_col) in down_corridor:
            return True
        # Left
        elif (king_row, 0) in king_way_out and (to_row, to_col) in left_corridor:
            return True
        # Right
        elif (king_row, 8) in king_way_out and (to_row, to_col) in right_corridor:
            return True

    return False


def adjacent(to_row, to_col, king_row, king_col):
    return (to_row, to_col) in [(king_row - 1, king_col), (king_row + 1, king_col), (king_row, king_col - 1), (king_row, king_col + 1)]


def capture_pawn(to_row, to_col, attackers, defenders):
    for def_pawn in defenders:
        if capture(to_row, to_col, def_pawn[0], def_pawn[1], attackers):
            return True
    return False


def king_out(king_move, king_row, king_col):
    return king_move and (king_row, king_col) in ESCAPES


def move_to_corridor(king_move, king_row, king_col, empty_cells):
    if not king_move:
        return False
    king_way_out = way_out(king_row, king_col, empty_cells)
    if king_way_out[0]:
        # Up
        if (0, king_col) in king_way_out[0]:
            return True
        # Down
        elif (8, king_col) in king_way_out[0]:
            return True
        # Left
        elif (king_row, 0) in king_way_out[0]:
            return True
        # Right
        elif (king_row, 8) in king_way_out[0]:
            return True
    return False


def dangerous_pawn(from_row, from_col, to_row, to_col, empty_cells):
    if from_col != to_col and from_row != to_row:
        return []
    elif from_col == to_col and from_row == to_row:
        return [(to_col, to_row)]
    elif from_row == to_row:
        res = []
        for i in range(min(from_col, to_col) + 1, max(from_col, to_col) + 1):
            if (to_row, i) not in empty_cells:
                return []
            else:
                res.append((to_row, i))
        return res
    elif from_col == to_col:
        res = []
        for i in range(min(from_row, to_row) + 1, max(from_row, to_row) + 1):
            if (i, to_col) not in empty_cells:
                return []
            else:
                res.append((to_row, i))
        return res
    else:
        print("Fatal error in dangerous_pawn method, it should never happen")
        exit(3)


def dangerous_corridors(king_row, king_col, black_pawns, empty_cells):
    corr = []
    for pawn in black_pawns:
        for to_append in dangerous_pawn(pawn[0], pawn[1], king_row, king_col, empty_cells):
            corr.append(to_append)
    return corr


def king_threaten(king_row, king_col, black_pawns, corridors):
    if (king_row - 1, king_col) in black_pawns and (king_row + 1, king_col) in corridors:
        return True
    elif (king_row + 1, king_col) in black_pawns and (king_row - 1, king_col) in corridors:
        return True
    elif (king_row, king_col - 1) in black_pawns and (king_row, king_col + 1) in corridors:
        return True
    elif (king_row, king_col + 1) in black_pawns and (king_row, king_col - 1) in corridors:
        return True
    else:
        return False


def prevent_king_capture(king_move, from_row, from_col, to_row, to_col, king_row, king_col, black_pawns, empty_cells):
    if king_move:
        adjacent_before = len([pawn for pawn in black_pawns if adjacent(pawn[0], pawn[1], from_row, from_col)])
        adjacent_after = len([pawn for pawn in black_pawns if adjacent(pawn[0], pawn[1], king_row, king_col)])
        return adjacent_after < adjacent_before
    # pawn move
    else:
        # if a white pawn goes there, it makes the king safer (NO THRONE EVALUATION, the king should feel "less fear")
        corridors = dangerous_corridors(king_row, king_col, black_pawns, empty_cells)
        return (to_row, to_col) in corridors and king_threaten(king_row, king_col, black_pawns, corridors)


def avoid_corridor(from_row, from_col, king_row, king_col):
    return adjacent(from_row, from_col, king_row, king_col)


def neighbourhood(black_pawns):
    neighbours = []
    for pawn in black_pawns:
        neighbours.append((pawn[0] + 1, pawn[1]))
        neighbours.append((pawn[0] - 1, pawn[1]))
        neighbours.append((pawn[0], pawn[1] + 1))
        neighbours.append((pawn[0], pawn[1] - 1))
    return neighbours


def can_reach(from_row, from_col, to_row, to_col, empty_cells):
    if from_col != to_col and from_row != to_row:
        return False
    elif from_col == to_col and from_row == to_row:
        return True
    elif from_row == to_row:
        for i in range(min(from_col, to_col) + 1, max(from_col, to_col) + 1):
            if (to_row, i) not in empty_cells + [(from_row, from_col)]:
                return False
        return True
    elif from_col == to_col:
        for i in range(min(from_row, to_row) + 1, max(from_row, to_row) + 1):
            if (i, to_col) not in empty_cells + [(from_row, from_col)]:
                return False
        return True
    else:
        print("Fatal error in can_reach method, it should never happen")
        exit(3)


def someone_can_reach(row, col, pawns, empty_cells):
    for pawn in pawns:
        if can_reach(pawn[0], pawn[1], row, col, empty_cells):
            return True
    return False


def avoid_king_suicide(king_move, to_row, to_col, black_pawns, empty_cells):
    if king_move:
        # adjacent up
        if (to_row - 1, to_col) in black_pawns and someone_can_reach(to_row + 1, to_col, black_pawns, empty_cells):
            return True
        # adjacent down
        elif (to_row + 1, to_col) in black_pawns and someone_can_reach(to_row - 1, to_col, black_pawns, empty_cells):
            return True
        # adjacent left
        elif (to_row, to_col - 1) in black_pawns and someone_can_reach(to_row, to_col + 1, black_pawns, empty_cells):
            return True
        # adjacent right
        elif (to_row, to_col + 1) in black_pawns and someone_can_reach(to_row, to_col - 1, black_pawns, empty_cells):
            return True
        # not adjacent
        else:
            return False
    return False


def move_to_suboptimal_corridor(king_move, to_row, to_col, black_pawns, empty_cells):
    return move_to_corridor(king_move, to_row, to_col, empty_cells) and (to_row, to_col) in neighbourhood(black_pawns)


def block_corridor_adjacent(from_row, from_col, to_row, to_col, king_row, king_col, empty_cells):
    return block_corridor(from_row, from_col, to_row, to_col, king_row, king_col, empty_cells) and adjacent(to_row, to_col, king_row, king_col)


def heuristic(from_row, from_col, to_row, to_col, turn, king_move, king_row, king_col, black_pawns, white_pawns, empty_cells, previous_empty_cells, previous_black_pawns, print_debug):
    ##########################
    # Known problems:
    # (SHOULD BE FIXED) 1) black sometimes avoid to block important corridor, blocking an already blocked corridor
    # (SHOULD BE FIXED) 2) black cannot move in camps
    # (SHOULD BE FIXED) 3) king sometimes goes next to che camp and get capture
    ##########################
    # BLACK heuristic:
    # 1)king capture
    # 2)obstacle king to win next turn (block winning corridor)
    # 3)go adjacent to the king
    # 4)pawn capture
    # 5)random
    ##########################
    if turn:
        # 1
        if capture_king(to_row, to_col, king_row, king_col, black_pawns + CAPTURING_CAMPS):
            if print_debug:
                print("100. I capture the king by the move")
            return 100
        # Special case: avoid to create corridor for the king or, in general, if a pawn is adjacent to the king then avoid to move it
        elif avoid_corridor(from_row, from_col, king_row, king_col):
            if print_debug:
                print("-25. Moving from here would result in creating a corridor or in not being adjacent to the king anymore")
            return -25
        # Special case: block a corridor and be adjacent to the king
        elif block_corridor_adjacent(from_row, from_col, to_row, to_col, king_row, king_col, [cell for cell in empty_cells if cell not in CAMPS] + [cell for cell in CAMPS if cell not in previous_black_pawns and (from_row, from_col) in CAMPS]):
            if print_debug:
                print("88. I block a winning corridor and I am adjacent to the king by the move ")
            return 88
        # 2
        elif block_corridor(from_row, from_col, to_row, to_col, king_row, king_col, [cell for cell in empty_cells if cell not in CAMPS] + [cell for cell in CAMPS if cell not in previous_black_pawns and (from_row, from_col) in CAMPS]):
            if print_debug:
                print("75. I block a winning corridor by the move")
            return 75
        # 3
        elif adjacent(to_row, to_col, king_row, king_col):
            if print_debug:
                print("50. I go next to the king by the move")
            return 50
        # 4
        elif capture_pawn(to_row, to_col, black_pawns + CAPTURING_CAMPS + THRONE, white_pawns):
            if print_debug:
                print("25. I capture a pawn by the move")
            return 25
        # 5
        else:
            if print_debug:
                print("0. I have no evident advantage by the move")
            return 0
    ##########################
    # WHITE heuristic:
    # 1)king out
    # 2)move king to a position in a corridor
    # 3)obstacle black to win next turn (avoid king be surrounded by black)
    # 4)pawn capture
    # 5)random
    ##########################
    else:
        # 1
        if king_out(king_move, king_row, king_col):
            if print_debug:
                print("100. I escape the king by the move")
            return 100
        # Special case: avoid king to go next to a black pawn in case another black pawn can capture him
        # can be improved considering special case of capture next to the throne or camps
        elif avoid_king_suicide(king_move, to_row, to_col, black_pawns + CAPTURING_CAMPS, [cell for cell in empty_cells if cell not in CAMPS]):
            if print_debug:
                print("-25. Moving the king there would allow the black to win the game")
            return -25
        # Special case: king goes to a position with a free corridor, it touches a black but black cannot win
        elif move_to_suboptimal_corridor(king_move, to_row, to_col, black_pawns, [cell for cell in empty_cells if cell not in CAMPS]):
            if print_debug:
                print("63. I move the king into a corridor, but it touches now a black (black cannot win next turn)")
            return 63
        # 2
        elif move_to_corridor(king_move, king_row, king_col, [cell for cell in empty_cells if cell not in CAMPS]):
            if print_debug:
                print("75. I move the king into a corridor by the move")
            return 75
        # 3
        # pawn avoidance OR king escape, not considered the fact that next to the throne is harder to capture the king
        # it can be improved
        elif prevent_king_capture(king_move, from_row, from_col, to_row, to_col, king_row, king_col, black_pawns + CAPTURING_CAMPS, [cell for cell in empty_cells if cell not in CAMPS]):
            if print_debug:
                print("50. I prevent the king to be captured or I just avoid the king to be in touch with a black pawn")
            return 50
        # 4
        elif capture_pawn(to_row, to_col, white_pawns + CAPTURING_CAMPS + THRONE + [(king_row, king_col)], black_pawns):
            if print_debug:
                print("25. I capture a pawn by the move")
            return 25
        # 5
        else:
            if print_debug:
                print("0. I have no evident advantage by the move")
            return 0


def generateBaselineMove(board, turn, json_turn, print_debug):
    # [(legal_from=e4, legal_to=e5), (e1, e0), (b2, b5) ... ], letters are rows and numbers are columns
    legal_moves = ck.get_legal_moves(board, turn)
    # if there are no legal moves, it is useless to proceed, a random one is returned
    if not legal_moves:
        return "c2", "c3"
    # the shuffle is useful in case the heuristics method detects equivalent moves (seed not fixed)
    random.seed(time.time())
    random.shuffle(legal_moves)
    evaluated_moves = []
    evaluations = []

    for move in legal_moves:
        from_row, from_col, to_row, to_col, king_move, king_row, king_col, white_pawns, black_pawns, empty_cells, previous_empty_cells, previous_black_pawns = move_baseline_pawn(move, board, print_debug)
        if print_debug:
            print(str(move) + ", eval:", end=' ', flush=True)
        evaluation = heuristic(from_row, from_col, to_row, to_col, turn, king_move, king_row, king_col, black_pawns, white_pawns, empty_cells, previous_empty_cells, previous_black_pawns, print_debug)
        evaluated_moves.append(move)
        evaluations.append(evaluation)
    evaluations_np = numpy.array(evaluations)
    evaluated_moves_np = numpy.array(evaluated_moves)
    if print_debug:
        print("FINAL DECISION:")
        # print("INDEX: " + str(evaluations_np.argmax(axis=0)))
        # print("(evaluations, evaluated_moves): " + str(numpy.array(list(zip(evaluated_moves_np, evaluations_np)))))
        print("move index: " + str(evaluations_np.argmax(axis=0)) +
              ", with evaluation of " + str(evaluations_np[evaluations_np.argmax(axis=0)]) +
              " that corresponds to the move " + str(evaluated_moves_np[evaluations_np.argmax(axis=0)]))

    raw_move = evaluated_moves_np[evaluations_np.argmax(axis=0)].tolist()
    return_move = {"from": raw_move[0], "to": raw_move[1], "turn": json_turn}
    if print_debug:
        print("the return is: " + str(return_move))
    return return_move


def evaluate_single_move(move, board, turn, print_debug):
    if move not in ck.get_legal_moves(board, turn):
        return -9999
    else:
        from_row, from_col, to_row, to_col, king_move, king_row, king_col, white_pawns, black_pawns, empty_cells, \
            previous_empty_cells, previous_black_pawns = move_baseline_pawn(move, board, print_debug)
        evaluation = heuristic(from_row, from_col, to_row, to_col, turn, king_move, king_row, king_col, black_pawns,
                               white_pawns, empty_cells, previous_empty_cells, previous_black_pawns, print_debug)
        return evaluation


def test():
    # clean row to cut and paste:
    # ["EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY"],

    board = [["EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY"],
             ["EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY"],
             ["EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY", "WHITE", "EMPTY", "EMPTY", "EMPTY"],
             ["EMPTY", "EMPTY", "EMPTY", "EMPTY", "BLACK", "EMPTY", "EMPTY", "EMPTY", "EMPTY"],
             ["EMPTY", "EMPTY", "EMPTY", "EMPTY", "THRONE", "KING", "EMPTY", "BLACK", "EMPTY"],
             ["EMPTY", "EMPTY", "EMPTY", "EMPTY", "WHITE", "EMPTY", "WHITE", "EMPTY", "EMPTY"],
             ["EMPTY", "EMPTY", "EMPTY", "EMPTY", "WHITE", "EMPTY", "EMPTY", "EMPTY", "EMPTY"],
             ["EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY", "BLACK", "EMPTY", "EMPTY", "EMPTY"],
             ["EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY", "BLACK", "EMPTY", "EMPTY", "EMPTY"]]

    # generateBaselineMove(board, 1, "BLACK", True)
    generateBaselineMove(board, 0, "WHITE", True)
