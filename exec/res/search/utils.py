import itertools
from typing import List, Tuple

import numpy as np

LAST_WHITE_STATE_INDEX = 14
LAST_KING_STATE_INDEX = 15
LAST_BLACK_STATE_INDEX = 23
TURN_STATE_INDEX = 32

NO_CAMPS_POSITIONS = np.array(
    [[1, 1, 1, 0, 0, 0, 1, 1, 1], [1, 1, 1, 1, 0, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 0],
     [0, 0, 1, 1, 0, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1, 0],
     [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 1, 1, 1, 1],
     [1, 1, 1, 0, 0, 0, 1, 1, 1]],
    dtype='bool')
BLACK_CAMPS_POSITIONS = np.array([[(0, 3), (0, 4), (0, 5), (1, 4)],
                                  [(3, 8), (4, 8), (5, 8), (4, 7)],
                                  [(8, 3), (8, 4), (8, 5), (7, 4)],
                                  [(3, 0), (4, 0), (5, 0), (4, 1)]])
KING_CAPTURED_THRONE = np.array(
    [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0]],
    dtype='bool')
NEXT_THRONE_POSITIONS = {
    (3, 4): np.array([(2, 4), (3, 3), (3, 5)]),
    (4, 3): np.array([(3, 3), (4, 2), (5, 3)]),
    (4, 5): np.array([(3, 5), (4, 6), (5, 5)]),
    (5, 4): np.array([(5, 3), (5, 5), (6, 4)])
}

# These positions include the unreachable black camps. If memory consumption is an issue, this representation can be
# optimized
WHITE_WIN_POSITIONS = np.array(
    list(
        set(
            itertools.chain.from_iterable(
                ((0, i), (8, i), (i, 0), (i, 8)) for i in range(9)))))


class Node(object):
    """TODO"""
    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0

        return self.value_sum / self.visit_count


class Game(object):
    """TODO"""
    def __init__(self, history=None):
        self.history = history or []

    @staticmethod
    def legal_actions(state):
        actions = {}
        white_pawns, white_king, black_pawns = state[
            LAST_WHITE_STATE_INDEX], state[LAST_KING_STATE_INDEX], state[
                LAST_BLACK_STATE_INDEX]
        black_turn = np.all(state[TURN_STATE_INDEX])

        # Remove the current pawns' and king's positions
        all_legal_positions = NO_CAMPS_POSITIONS & np.array(
            1 - (white_pawns + white_king + black_pawns), dtype='bool')

        if black_turn:
            current_player_pieces = black_pawns
        else:
            current_player_pieces = white_pawns + white_king

        for r, c in zip(*np.where(current_player_pieces == 1)):
            piece_legal_positions = all_legal_positions.copy()

            if black_turn:
                camp_index = np.where(
                    np.all(BLACK_CAMPS_POSITIONS == (r, c), axis=2))[0]
                # r_ind, c_ind = BLACK_CAMPS_POSITIONS[camp_index].T
                # piece_legal_positions[r_ind, c_ind] = 1
                if camp_index.size > 0:
                    camp_positions = BLACK_CAMPS_POSITIONS[camp_index]
                    camp_positions = camp_positions.reshape(
                        camp_positions.shape[1:])
                    for camp_r, camp_c in camp_positions:
                        if black_pawns[camp_r, camp_c] == 0:
                            piece_legal_positions[camp_r, camp_c] = 1

            # Add orthogonal moves
            piece_final_positions = []
            row = piece_legal_positions[r]
            col = piece_legal_positions[:, c]
            piece_final_positions += Game.__find_final_positions(
                line=row, piece_position=(r, c))
            piece_final_positions += Game.__find_final_positions(
                line=col, piece_position=(r, c), row=False)

            if piece_final_positions:
                actions[(r, c)] = piece_final_positions

        actions = {
            k: list(map(tuple, k - np.array(v)))
            for k, v in actions.items()
        }
        actions = {
            k: np.apply_along_axis(Game.__map_actions, axis=1, arr=v)
            for k, v in actions.items()
        }
        actions = [k + (vi, ) for k, v in actions.items() for vi in v]

        return actions

    @staticmethod
    def terminal(state):
        black_turn = np.all(state[TURN_STATE_INDEX])
        black_pawns = state[LAST_BLACK_STATE_INDEX]
        white_king = state[LAST_KING_STATE_INDEX]

        if black_turn:
            king_position = np.argwhere(white_king)
            king_in_throne = (king_position == (4, 4)).all()
            king_next_throne = np.array([
                (king_position == king_pos).all()
                for king_pos in NEXT_THRONE_POSITIONS.keys()
            ]).any()

            # Check for throne capture
            if king_in_throne and black_pawns[tuple(
                    np.argwhere(KING_CAPTURED_THRONE).T)].all():
                return True

            # Check for near throne capture
            for king_pos, black_pos in NEXT_THRONE_POSITIONS.items():
                if (king_position == king_pos).all() and black_pawns[tuple(
                        black_pos.T)].all():
                    return True

            # Check for normal king capture
            horiz_black_pos = np.concatenate(
                [king_position + (0, 1), king_position + (0, -1)])
            vert_black_pos = np.concatenate(
                [king_position + (1, 0), king_position + (-1, 0)])
            if not king_in_throne and not king_next_throne and (
                    black_pawns[tuple(horiz_black_pos.T)].all()
                    or black_pawns[tuple(vert_black_pos.T)].all()):
                return True
        else:
            return np.argwhere(white_king) in WHITE_WIN_POSITIONS

        return False

    def apply(self, action):
        self.history.append(action)

    @staticmethod
    def __find_final_positions(line: np.ndarray,
                               piece_position: Tuple,
                               row: bool = True) -> List[Tuple[int, int]]:
        """Finds the final positions of """
        sep_ind = piece_position[row]

        other_pieces_indices = np.where(
            [line[i] == 0 and i != sep_ind for i in range(9)])[0]
        try:
            max_index_before = max(
                filter(lambda a: a < sep_ind, other_pieces_indices))
        except ValueError:
            max_index_before = -1
        try:
            min_index_after = min(
                filter(lambda a: a > sep_ind, other_pieces_indices))
        except ValueError:
            min_index_after = 9

        r, c = piece_position
        if row:
            return [(r, i) for i in list(range(max_index_before + 1, c)) +
                    list(range(c + 1, min_index_after))]
        else:
            return [(i, c) for i in list(range(max_index_before + 1, r)) +
                    list(range(r + 1, min_index_after))]

    @staticmethod
    def __map_actions(t):
        r, c = t
        if r > 0:
            return r - 1
        elif r < 0:
            return -r + 15
        elif c > 0:
            return c + 23
        elif c < 0:
            return -c + 7
