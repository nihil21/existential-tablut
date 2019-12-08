import numpy as np
from search.utils import Game


def get_legal_moves(raw_board, is_turn_black):

    # black turn
    if is_turn_black:
        turn = np.ones((1, 9, 9), dtype='bool')
    # white turn
    else:
        turn = np.zeros((1, 9, 9), dtype='bool')

    # not really used... but...
    random_layer = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 1, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    w_pawns_matrix = []
    b_pawns_matrix = []
    w_king_matrix = []

    for row in raw_board:
        w_pawns_row = []
        b_pawns_row = []
        w_king_row = []
        for square in row:
            if square == "EMPTY":
                # O
                w_pawns_row.append(0)
                b_pawns_row.append(0)
                w_king_row.append(0)
            elif square == "WHITE":
                # W
                w_pawns_row.append(1)
                b_pawns_row.append(0)
                w_king_row.append(0)
            elif square == "BLACK":
                # B
                w_pawns_row.append(0)
                b_pawns_row.append(1)
                w_king_row.append(0)
            elif square == "KING":
                # K
                w_pawns_row.append(0)
                b_pawns_row.append(0)
                w_king_row.append(1)
            elif square == "THRONE":
                # T
                w_pawns_row.append(0)
                b_pawns_row.append(0)
                w_king_row.append(0)
            else:
                print("[ERROR] Not recognized board square value. Terminating...")
                exit(1)
        w_pawns_matrix.append(w_pawns_row)
        b_pawns_matrix.append(b_pawns_row)
        w_king_matrix.append(w_king_row)

    # from list matrix to np.matrix
    w_pawns = np.array(w_pawns_matrix)
    b_pawns = np.array(b_pawns_matrix)
    w_king = np.array(w_king_matrix)

    # 3 lines of magic
    state = create_state(w_pawns, w_king, b_pawns, random_layer, turn)
    game = Game()
    # [(2,2,0), (2,2,1), ... ]
    actions_raw = game.legal_actions(state)
    #print("\nactions_raw = " + str(actions_raw))

    actions = []
    for action in actions_raw:
        # prepare the 'letter' (from)
        raw_number = action[1]
        letter_from = letter_of(raw_number)

        # prepare the 'number' (from)
        number_from = action[0] + 1

        # prepare 'letter' and 'number (to)
        # relative_action in 1,..,8; movement on a certain direction
        relative_action = (action[2] % 8) + 1
        # north
        if 0 <= action[2] <= 7:
            letter_to = letter_from
            number_to = number_from - relative_action
        # east
        elif 8 <= action[2] <= 15:
            letter_to = letter_of(action[1] + relative_action)
            number_to = number_from
        # south
        elif 16 <= action[2] <= 23:
            letter_to = letter_from
            number_to = number_from + relative_action
        # west
        elif 24 <= action[2] <= 31:
            letter_to = letter_of(action[1] - relative_action)
            number_to = number_from
        else:
            print("\nERROR! row action from not recognize: " + str(action[2] + '\n'))

        actions.append((letter_from + str(number_from), letter_to + str(number_to)))

    # [(from=e4,to=e5), (e1, e0) ... ], letters are rows and numbers are columns
    return actions


def letter_of(raw_number):
    if raw_number == 0:
        letter = 'a'
    elif raw_number == 1:
        letter = 'b'
    elif raw_number == 2:
        letter = 'c'
    elif raw_number == 3:
        letter = 'd'
    elif raw_number == 4:
        letter = 'e'
    elif raw_number == 5:
        letter = 'f'
    elif raw_number == 6:
        letter = 'g'
    elif raw_number == 7:
        letter = 'h'
    elif raw_number == 8:
        letter = 'i'
    else:
        print("ERROR! row action from not recognize: " + str(raw_number))
    return letter


def create_state(w_pawns, w_king, b_pawns, r_layer, turn):
        white_planes = np.tile([w_pawns, w_king], (8, 1, 1))
        black_planes = np.tile(b_pawns, (8, 1, 1))
        random_planes = np.tile(r_layer, (8, 1, 1))

        return np.concatenate(
            [white_planes, black_planes, random_planes, turn])


def get_legal_index_from(legal_moves, output_move_link):
    legal_index_from = []
    for key, value in output_move_link:
        for move in legal_moves:
            if key == move[0]:
                legal_index_from.append(value)
                break
    # remove duplicates, maybe unnecessary due to 'break'
    legal_index_from = list(dict.fromkeys(legal_index_from))
    return legal_index_from

#############################################
# THE FOLLOWING CODE IS USED FOR THE BASELINE
#############################################


def number_of(letter):
    if letter == 'a':
        number = 0
    elif letter == 'b':
        number = 1
    elif letter == 'c':
        number = 2
    elif letter == 'd':
        number = 3
    elif letter == 'e':
        number = 4
    elif letter == 'f':
        number = 5
    elif letter == 'g':
        number = 6
    elif letter == 'h':
        number = 7
    elif letter == 'i':
        number = 8
    else:
        print("ERROR! row action from not recognize: " + str(letter))
    return number
