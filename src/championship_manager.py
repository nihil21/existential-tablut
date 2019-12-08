import numpy as np
import threading


class Score:
    def __init__(self, deserved_win, draw, deserved_lose, fail_win, fail_lose):
        self.deserved_win = deserved_win
        self.fail_win = fail_win
        self.draw = draw
        self.deserved_lose = deserved_lose
        self.fail_lose = fail_lose
        self.total_moves = 0
        self.moves_in_points = 0

    # The variable result must be in {3, 2, 1, 0 , -1}.
    # The choice of these values is made according to the definition of method "points" of class Score
    def calculate(self, result, match_moves):
        self.total_moves += match_moves
        self.moves_in_points += self.get_move_point(result, match_moves)
        # Deserved Win
        if result == 3:
            self.deserved_win += 1
        # Fail Win
        elif result == 2:
            self.fail_win += 1
        # Draw
        elif result == 1:
            self.draw += 1
        # Deserved Lose
        elif result == 0:
            self.deserved_lose += 1
        # Fail Lose
        elif result == -1:
            self.fail_lose += 1
        else:
            print("result not valid: " + str(result) + " for score: " + str(self))

    def get_move_point(self, result, match_moves):
        if result in [3, 2]:
            return - match_moves
        elif result in [1, 0, -1]:
            return match_moves
        else:
            print("FATAL ERROR, invalid result of a match: " + result)

    def points(self):
        # To achieve readability all terms are expressed here
        return 450 * self.deserved_win + 2 * self.fail_win + 1 * self.draw + 0 * self.deserved_lose + -1 * self.fail_lose + 2 * self.moves_in_points

    def __repr__(self):
        return "[DW: " + str(self.deserved_win) + ", FW: " + str(self.fail_win) + ", D: " + str(self.draw) + ", DL: " + str(self.deserved_lose) + ", FL: " + str(self.fail_lose) + ", TM: " + str(self.total_moves) + "]"


class Network:
    def __init__(self, name, score_white, score_black, network_white=None, network_black=None):
        self.name = name
        self.score_white = score_white
        self.score_black = score_black
        self.lock = threading.Lock()

        # following fields are not currently used, they could be useful for future implementation
        self.network_white = network_white
        self.network_black = network_black

    # The following method is not used, but could be useful in future
    def black_compare_to(self, network):
        delta_points = self.score_black.points() - network.score_black.points()
        if delta_points != 0:
            return delta_points
        else:
            return self.score_black.win - network.score_black.win

    # The following method is not used, but could be useful in future
    def white_compare_to(self, network):
        delta_points = self.score_white.points() - network.score_white.points()
        if delta_points != 0:
            return delta_points
        else:
            return self.score_white.win - network.score_white.win

    # The variable result must be in {3, 2, 1, 0 , -1}.
    # The choice of these values is made according to the definition of method "points" of class Score
    # The variable is_white must be 1 or 0.
    def calculate(self, is_white, result, total_move):
        # Enter critical section
        with self.lock:
            # Do critical work
            if is_white == 1:
                self.score_white.calculate(result, total_move)
            elif is_white == 0:
                self.score_black.calculate(result, total_move)
            else:
                print("result not valid: " + str(result) + " for network: " + str(self))
        # Exit critical section

    def full_repr(self):
        return '[' + self.name + ', score_white: ' + str(self.score_white) + ', score_black: ' + \
               str(self.score_black) + ']'

    def __repr__(self):
        return '[' + self.name + ']'


class Match:
    def __init__(self, white, black):
        self.white = white
        self.black = black

    def __repr__(self):
        return '[white: ' + str(self.white) + ', black: ' + str(self.black) + ']'


class ChampDay:
    def __init__(self, matches):
        self.matches = matches

    def __repr__(self):
        return 'Day:\n' + str(self.matches)


def shift_left(data, add):
    temp = np.empty(data.shape[0], dtype=Network)
    for i in range(data.shape[0] - 1):
        temp[i] = data[i + 1]
    temp[data.shape[0] - 1] = add
    return temp


def shift_right(data, add):
    temp = np.empty(data.shape[0], dtype=Network)
    for i in range(data.shape[0] - 1):
        temp[i + 1] = data[i]
    temp[0] = add
    return temp


class Berger:
    def __init__(self, networks):
        self.networks = networks

    def generate_days(self):
        team_num = self.networks.shape[0]
        days_num = team_num - 1
        days = np.empty(days_num, dtype=ChampDay)
        whites = np.empty(team_num // 2, dtype=Network)
        blacks = np.empty(team_num // 2, dtype=Network)

        for i in range(team_num // 2):
            whites[i] = self.networks[i]
            blacks[i] = self.networks[team_num - 1 - i]

        for i in range(days_num):
            matches = np.empty(team_num // 2, dtype=Match)
            k = 0
            for j in range(team_num // 2):
                matches[k] = Match(whites[j], blacks[j])
                k += 1
            pivot = whites[0]
            carry_over = blacks[blacks.shape[0] - 1]
            blacks = shift_right(blacks, whites[1])
            whites = shift_left(whites, carry_over)
            whites[0] = pivot
            days[i] = ChampDay(matches)
        return days


class Championship:
    def __init__(self, networks_name):
        if len(networks_name) % 2 == 1:
            print("Error: there must be an even number of network")
            return
        self.networks = np.empty(len(networks_name), dtype=Network)
        for i in range(len(networks_name)):
            # 'white' and 'black' are not defined, but they are here to future possible implementation
            self.networks[i] = Network(networks_name[i], Score(0, 0, 0, 0, 0), Score(0, 0, 0, 0, 0))
        if len(networks_name) > 2:
            # dim = n * (n - 1) + n = n ** 2
            self.dim = self.networks.shape[0] ** 2
            berger = Berger(self.networks)
            self.days = berger.generate_days()
        else:
            matches = np.empty(1, dtype=Match)
            matches[0] = Match(self.networks[0], self.networks[1])
            self.days = np.empty(1, dtype=ChampDay)
            self.days[0] = ChampDay(matches)
        # Generate return matches
        for day in self.days:
            matches = np.empty(self.networks.shape[0] // 2, dtype=Match)
            self.days = np.append(self.days, ChampDay(matches))
            for index, match in enumerate(day.matches):
                matches[index] = Match(match.black, match.white)
        # Generate matches between a net and itself
        matches = np.empty(self.networks.shape[0], dtype=Match)
        for index in range(self.networks.shape[0]):
            matches[index] = Match(self.networks[index], self.networks[index])
        self.days = np.append(self.days, ChampDay(matches))

    # return list of all matches (class Match)
    def plain(self):
        plain_matches = np.empty(self.dim, dtype=Match)
        i = 0
        for day in self.days:
            for match in day.matches:
                plain_matches[i] = match
                i += 1
        return plain_matches

    # return list of all matches (list of tuples of str)
    def all_matches(self):
        all_matches = []
        for day in self.days:
            for match in day.matches:
                all_matches.append((match.white.name, match.black.name))
        return all_matches

    # The variable result must be in {3, 2, 1, 0 , -1}.
    # The choice of these values is made according to the definition of method "points" of class Score
    # The variable is_white must be 1 or 0.
    # net_name must be a "name" present in a network in self.networks
    def calculate(self, net_name, is_white, result, total_move):
        found = 0
        for net in self.networks:
            if net.name == net_name:
                net.calculate(is_white, result, total_move)
                found = 1
        if found == 0:
            print("cannot find in self.networks a net with name: " + str(net_name))

    def sorted_white(self):
        sorted_net = sorted(self.networks, key=lambda x: x.score_white.points(), reverse=True)
        sorted_name = []
        for net in sorted_net:
            sorted_name.append(net.name)
        return sorted_name

    def sorted_black(self):
        sorted_net = sorted(self.networks, key=lambda x: x.score_black.points(), reverse=True)
        sorted_name = []
        for net in sorted_net:
            sorted_name.append(net.name)
        return sorted_name

    def white_with_score(self):
        result_dict = {}
        for net in self.networks:
            result_dict[net.name] = net.score_white
        return result_dict

    def black_with_score(self):
        result_dict = {}
        for net in self.networks:
            result_dict[net.name] = net.score_black
        return result_dict

    def white_with_points(self, USE_BASELINE, baseline_net):
        result_dict = {}
        for net in self.networks:
            if not USE_BASELINE:
                result_dict[net.name] = net.score_white.points()
            else:
                if net.name[-len(baseline_net):] != baseline_net:
                    result_dict[net.name] = net.score_white.points()
        return result_dict

    def black_with_points(self, USE_BASELINE, baseline_net):
        result_dict = {}
        for net in self.networks:
            if not USE_BASELINE:
                result_dict[net.name] = net.score_black.points()
            else:
                if net.name[-len(baseline_net):] != baseline_net:
                    result_dict[net.name] = net.score_black.points()
        return result_dict

    def __repr__(self):
        return 'networks:\n' + str(self.networks) + '\n\ndays:\n' + str(self.days)


def test():
    # n: neural networks number
    n = 4
    # List of string
    networks_name = []
    for i in range(n):
        networks_name.append('net' + str(i))
    c = Championship(networks_name)
    # print(str(c))
    list_matches_tuple = c.all_matches()
    print("N = 4")
    print("all matches")
    print(str(list_matches_tuple))
    print("black sorted")
    print(str(c.sorted_black()))
    print("white sorted")
    print(str(c.sorted_white()))
    print("black sorted with score")
    print(str(c.black_with_score()))
    print("white sorted with score")
    print(str(c.white_with_score()))
    print("black sorted with points")
    print(str(c.black_with_points()))
    print("white sorted with points")
    print(str(c.white_with_points()))
    # -------------------------------------
    print("-------------------------------")
    c = Championship(networks_name[:2])
    list_matches_tuple = c.all_matches()
    print("N = 2")
    print("all matches")
    print(str(list_matches_tuple))
    print("-------------------------------")
    print("N = 3")
    c = Championship(networks_name[:3])


# uncomment the following line to test the module
# test()
