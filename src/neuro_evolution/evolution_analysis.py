import os
import numpy as np
import matplotlib.pyplot as plt
import sys


def extract_score(path):
    """This function reads the report file and extracts the score of each net
    :param path: path to the report file

    :returns scores_tuple: a tuple containing the list of scores of both players
    :returns wins_tuple: a tuple containing the list of numbers of matches won for both players
    :returns draws_tuple: a tuple containing the list of numbers of draws for both players
    :returns lose_tuple: a tuple containing the list of numbers of matches lost for both players
    """
    with open(path, "r") as file:
        scores = []

        #############
        # remove header and baseline
        file.readline()
        file.readline()
        #############

        line = file.readline()
        while line:
            tokens = line.split()
            # total moves are managed
            if len(tokens) == 15:
                scores.append((int(tokens[2]), int(tokens[4][:-1]),
                               int(tokens[8][:-1]), int(tokens[10][:-1]),
                               int(tokens[14][:-1])))
            elif len(tokens) > 1 and tokens[0] == "White":
                #############
                # remove baseline
                file.readline()
                #############

            line = file.readline()

        half_len = int(len(scores) / 2)
        tot_len = len(scores)
    scores_tuple = (scores[0:half_len], scores[half_len:tot_len])
    return scores_tuple


def max_median(scores_by_gen):
    # Score
    max_score_blacks = []
    max_score_whites = []
    median_score_blacks = []
    median_score_whites = []
    # Wins
    max_wins_blacks = []
    max_wins_whites = []
    median_wins_blacks = []
    median_wins_whites = []
    # Draws
    max_draws_blacks = []
    max_draws_whites = []
    median_draws_blacks = []
    median_draws_whites = []
    # Defeats
    max_defeats_blacks = []
    max_defeats_whites = []
    median_defeats_blacks = []
    median_defeats_whites = []
    # Total moves
    max_moves_blacks = []
    max_moves_whites = []
    median_moves_blacks = []
    median_moves_whites = []

    for scores_tuple in scores_by_gen:
        cur_black_scores, cur_white_scores = scores_tuple[0], scores_tuple[1]
        # The list is sorted
        median_index = int(len(cur_black_scores) / 2)
        # Score
        max_score_blacks.append(cur_black_scores[0][0])
        max_score_whites.append(cur_white_scores[0][0])
        median_score_blacks.append(cur_black_scores[median_index][0])
        median_score_whites.append(cur_white_scores[median_index][0])
        # Wins
        max_wins_blacks.append(cur_black_scores[0][1])
        max_wins_whites.append(cur_white_scores[0][1])
        median_wins_blacks.append(cur_black_scores[median_index][1])
        median_wins_whites.append(cur_white_scores[median_index][1])
        # Draws
        max_draws_blacks.append(cur_black_scores[0][2])
        max_draws_whites.append(cur_white_scores[0][2])
        median_draws_blacks.append(cur_black_scores[median_index][2])
        median_draws_whites.append(cur_white_scores[median_index][2])
        # Defeats
        max_defeats_blacks.append(cur_black_scores[0][3])
        max_defeats_whites.append(cur_white_scores[0][3])
        median_defeats_blacks.append(cur_black_scores[median_index][3])
        median_defeats_whites.append(cur_white_scores[median_index][3])
        # Total moves
        max_moves_blacks.append(cur_black_scores[0][4])
        max_moves_whites.append(cur_white_scores[0][4])
        median_moves_blacks.append(cur_black_scores[median_index][4])
        median_moves_whites.append(cur_white_scores[median_index][4])

    # Create NumPy arrays with the scores of max and median, black and white, players
    score = ((np.array(max_score_blacks), np.array(max_score_whites)),
             (np.array(median_score_blacks), np.array(median_score_whites)))
    # Create NumPy arrays with wins no. of max and median, black and white, players
    wins = ((np.array(max_wins_blacks), np.array(max_wins_whites)),
            (np.array(median_wins_blacks), np.array(median_wins_whites)))
    # Create NumPy arrays with draws no. of max and median, black and white, players
    draws = ((np.array(max_draws_blacks), np.array(max_draws_whites)),
             (np.array(median_draws_blacks), np.array(median_draws_whites)))
    # Create NumPy arrays with lo no. of max and median, black and white, players
    defeats = ((np.array(max_defeats_blacks), np.array(max_defeats_whites)),
               (np.array(median_defeats_blacks), np.array(median_defeats_whites)))
    # NumPy Moves
    moves = ((np.array(max_moves_blacks), np.array(max_moves_whites)),
             (np.array(median_moves_blacks), np.array(median_moves_whites)))

    return score, wins, draws, defeats, moves


def stem_score(title, scores, x, plt_color):
    plt.title(title)
    plt.xlabel('Generation number')
    plt.ylabel('Score')
    markerline1, stemlines1, baseline = plt.stem(x,
                                                 scores[0],
                                                 '-',
                                                 label='Black players',
                                                 use_line_collection=True)
    plt.setp(markerline1, color=plt_color[0])
    plt.setp(stemlines1, color=plt_color[0])
    plt.setp(baseline, color='#000000')
    markerline2, stemlines2, baseline = plt.stem(x,
                                                 scores[1],
                                                 '-',
                                                 label='White players',
                                                 use_line_collection=True)
    plt.setp(markerline2, color=plt_color[1])
    plt.setp(stemlines2, color=plt_color[1])
    plt.setp(baseline, color='#000000')
    plt.legend()
    plt.show()


def stacked_bar_matches(title, win_no, draw_no, defeat_no, num_gen):
    plt.title(title)
    plt.ylabel('Number of matches')
    ind = np.arange(num_gen)
    ticks = produce_ticks(num_gen)
    plt.xticks(ind, ticks[0])
    plt.yticks(ticks[1])
    width = 0.35

    p1 = plt.bar(ind, win_no, width)
    p2 = plt.bar(ind, draw_no, width, bottom=win_no)
    p3 = plt.bar(ind, defeat_no, width, bottom=(win_no + draw_no))
    plt.legend((p1[0], p2[0], p3[0]), ('Number of wins', 'Number of draws', 'Number of defeats'))
    plt.show()


def produce_ticks(num_gen):
    temp_list = []
    for i in range(0, num_gen):
        temp_list.append("Gen" + str(i))
    return tuple(temp_list), np.arange(0, int(num_players) + 1, 5)


# ---------- Main ----------
report_dir = sys.argv[1]
num_players = sys.argv[2]
scores_by_gen = []
for entry in os.scandir(report_dir):
    if entry.path.endswith(".txt"):
        scores_by_gen.append(extract_score(entry.path))

# Find max and median players
score, wins, draws, defeats, moves = max_median(scores_by_gen)

# ---------- Plots for scores ----------
x = np.linspace(0, len(scores_by_gen) - 1, len(scores_by_gen))
plt_colors = ((63 / 255, 82 / 255, 228 / 255, 0.7), (224 / 255, 36 / 255, 36 / 255, 0.7))

# ----- Max scores -----
stem_score('Maximum score w.r.t. generation number', score[0], x, plt_colors)

# ----- Median score -----
stem_score('Median score w.r.t. generation number', score[1], x, plt_colors)

# ---------- Plots for black matches ----------
# ----- Max matches -----
stacked_bar_matches('Matches results of most successful black player w.r.t. generation number',
                    wins[0][0], draws[0][0], defeats[0][0], len(scores_by_gen))
stacked_bar_matches('Matches results of most successful white player w.r.t. generation number',
                    wins[0][1], draws[0][1], defeats[0][1], len(scores_by_gen))
# ----- Median matches -----
stacked_bar_matches('Matches results of median black player w.r.t. generation number',
                    wins[1][0], draws[1][0], defeats[1][0], len(scores_by_gen))
stacked_bar_matches('Matches results of median white player w.r.t. generation number',
                    wins[1][1], draws[1][1], defeats[1][1], len(scores_by_gen))