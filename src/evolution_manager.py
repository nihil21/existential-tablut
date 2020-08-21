import argparse
import os
import pickle
import threading
from time import sleep
from tensorflow.keras.models import load_model
from predictPlayNN import connectAndPlay
from championship_manager import Championship
from neuro_evolution.genetic_algorithm import evolve

WHITE_PORT = 5800
BLACK_PORT = 5801

CROSSOVER_RATE = 0.01
MUTATION_RATE_INDIVIDUALS = 0.5
MUTATION_RATE_NEURONS = 0.005


def load_models(folder_path, generation, players_number):
    # black_model = {"id": (from, to), ...}
    black_model = {}
    white_model = {}

    gen_str = str(generation)

    for i in range(players_number):
        black_model[gen_str + "_" + str(i)] = (
            load_model(folder_path + gen_str + "/black/modelFB_" + gen_str + "_" + str(i), compile=False),
            load_model(folder_path + gen_str + "/black/modelTB_" + gen_str + "_" + str(i), compile=False)
        )
        white_model[gen_str + "_" + str(i)] = (
            load_model(folder_path + gen_str + "/white/modelFW_" + gen_str + "_" + str(i), compile=False),
            load_model(folder_path + gen_str + "/white/modelTW_" + gen_str + "_" + str(i), compile=False)
        )

    # model.predict() is not thread safe, so we have to compute the predict function here before creating threads
    for k in black_model.keys():
        black_model[k][0].make_predict_function()
        black_model[k][1].make_predict_function()
    for k in white_model.keys():
        white_model[k][0].make_predict_function()
        white_model[k][1].make_predict_function()

    return black_model, white_model


def load_labels(folder_path):
    black_label = []
    white_label = []

    black_label.append(pickle.loads(open(folder_path + "label/labelFB", "rb").read()))
    black_label.append(pickle.loads(open(folder_path + "label/labelTB", "rb").read()))

    white_label.append(pickle.loads(open(folder_path + "label/labelFW", "rb").read()))
    white_label.append(pickle.loads(open(folder_path + "label/labelTW", "rb").read()))

    return black_label, white_label


def save_report(folder_path, champ, generation_number):
    with open(folder_path + "report/report_" + str(generation_number) + ".txt", "w") as reportFile:

        def write_report(sorted_champ, details):
            i = 1
            for net in sorted_champ:
                reportFile.write(str(i) + ") " + net[0] + " " + str(net[1]) + " " + str(details[net[0]]) + "\n")
                i += 1

        reportFile.write("Black with points:\n")
        # print also baseline scores
        black_champ = champ.black_with_points(False, "baseline_net")
        sorted_black_champ = sorted(black_champ.items(), key=lambda kv: kv[1], reverse=True)
        black_details = champ.black_with_score()
        write_report(sorted_black_champ, black_details)

        reportFile.write("\nWhite with points:\n")
        # print also baseline scores
        white_champ = champ.white_with_points(False, "baseline_net")
        sorted_white_champ = sorted(white_champ.items(), key=lambda kv: kv[1], reverse=True)
        white_details = champ.white_with_score()
        write_report(sorted_white_champ, white_details)


def wait_for_threads():
    main_thread = threading.currentThread()
    for t in threading.enumerate():
        if t is not main_thread:
            t.join()


def save_models(folder_path, generation, black_new_model, white_new_model):
    for k in black_new_model.keys():
        black_new_model[k][0].save(folder_path + str(generation) + "/black/modelFB_" + k)
        black_new_model[k][1].save(folder_path + str(generation) + "/black/modelTB_" + k)

    for k in white_new_model.keys():
        white_new_model[k][0].save(folder_path + str(generation) + "/white/modelFW_" + k)
        white_new_model[k][1].save(folder_path + str(generation) + "/white/modelTW_" + k)


def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-sgn", "--starting-generation-number", required=True,
                    help="the number of the first generation to consider")
    ap.add_argument("-nf", "--net-folder", required=True,
                    help="the folder containing the nets divided by generation number")
    ap.add_argument("-pn", "--players-number", required=True, help="the number of players (nets) for every generation")
    ap.add_argument("-gn", "--generation-number", required=True, help="how many generations have to be computed")

    ap.add_argument("-b", "--baseline", required=True, help="1: use baseline championship, 0: use network championship")
    # more arguments needed?
    args = vars(ap.parse_args())

    # folder hierarchy:
    # neuralNetworks
    #     labels
    #         labelFB
    #         labelTB
    #         labelFW
    #         labelTW
    #     0
    #         black
    #             modelFB_0_0
    #             modelTB_0_0
    #             ...
    #             modelFB_0_49
    #             modelTB_0_49
    #         white
    #             modelFW_0_0
    #             ...
    #             modelTW_0_49
    #     1
    #         ...
    #    ...
    #     n
    #         ...

    folder = args["net_folder"]
    if folder[-1] != "/":
        folder += "/"

    starting_generation_number = int(args["starting_generation_number"])
    players_number = int(args["players_number"])

    generation_number = int(args["generation_number"])

    # load neural networks from the folder provided
    print("[INFO] loading networks and label binaries...")
    black_model, white_model = load_models(folder, starting_generation_number, players_number)
    black_label, white_label = load_labels(folder)

    for g in range(starting_generation_number, starting_generation_number + generation_number):
        print("[INFO] generation number " + str(g))

        # generate championship for this generation
        championship = Championship([str(g) + "_" + str(i) for i in range(players_number)])

        # matches: [(white player, black player), ...]
        # in this case: [('1_0', '1_29'), ('1_1', '1_28') ... ]
        matches = championship.all_matches()

        ##########################################################################
        # These lines are needed to reduce number of concurrent thread
        # if 'limited_match_per_time' all threads will start to execute asap
        limited_match_per_time = False
        # len(matches) = n ** 2, match_per_time >= 1 (at least 1 match)
        # match_per_time = sqrt(len(matches))
        match_per_time = 900
        num_current_match = 1
        ##########################################################################
        ##########################################################################
        # These lines are needed to manage the baseline player, which will be the
        # last player in the list, to do not create issues in the evolution step
        use_baseline = int(args["baseline"])
        # folder that contains moves that makes baseline lose against a net
        if use_baseline and not os.path.isdir(folder + 'baseline_defeated_by/'):
            os.makedirs(folder + 'baseline_defeated_by/')
        baseline_net = '_' + str(players_number - 1)
        # remove all matches where the baseline does not play, and where both are baseline
        if use_baseline:
            matches = [match
                       for match in matches
                       if (match[0][-len(baseline_net):] == baseline_net
                           or match[1][-len(baseline_net):] == baseline_net)
                       and not (match[0][-len(baseline_net):] == baseline_net
                                and match[1][-len(baseline_net):] == baseline_net)]
            matches *= 5
        ##########################################################################

        # lock to correctly use Theano
        lock = threading.Lock()

        print("[INFO] playing " + str(len(matches)) + " matches...")

        for m in matches:
            # white player created
            white_player = m[0]
            baseline_player = False
            if use_baseline and white_player[-len(baseline_net):] == baseline_net:
                baseline_player = True
            model_from = white_model[white_player][0]
            model_to = white_model[white_player][1]

            white_thread_play = threading.Thread(target=connectAndPlay, args=(
                model_from, model_to, white_label[0], white_label[1], white_player, "W", WHITE_PORT, championship, lock,
                baseline_player, folder + 'baseline_defeated_by/', m[1]))
            white_thread_play.start()

            # black player created
            black_player = m[1]
            baseline_player = False
            if use_baseline and black_player[-len(baseline_net):] == baseline_net:
                baseline_player = True
            model_from = black_model[black_player][0]
            model_to = black_model[black_player][1]

            black_thread_play = threading.Thread(target=connectAndPlay, args=(
                model_from, model_to, black_label[0], black_label[1], black_player, "B", BLACK_PORT, championship, lock,
                baseline_player, folder + 'baseline_defeated_by/', m[0]))
            black_thread_play.start()

            if limited_match_per_time:
                print("started match number: " + str(num_current_match))
                if num_current_match % match_per_time == 0 or num_current_match == len(matches) - 1:
                    wait_for_threads()
                else:
                    sleep(0.5)
                num_current_match += 1
            else:
                sleep(0.5)

        if not limited_match_per_time:
            wait_for_threads()

        # print report
        save_report(folder, championship, g)

        # evolution of the networks
        print("[INFO] evolving networks...")

        black_next_generation = []
        if use_baseline:
            black_model = {key: value for key, value in black_model.items() if key[-len(baseline_net):] != baseline_net}
        print("[INFO] evolving " + str(len(black_model.keys())) + " nets ...")
        black_thread_evolve = threading.Thread(target=evolve, args=(
            black_model, championship.black_with_points(use_baseline, baseline_net), players_number // 10,
            CROSSOVER_RATE, MUTATION_RATE_INDIVIDUALS, MUTATION_RATE_NEURONS, black_next_generation,
            lock, use_baseline))
        black_thread_evolve.start()

        white_next_generation = []
        if use_baseline:
            white_model = {key: value for key, value in white_model.items() if key[-len(baseline_net):] != baseline_net}
        white_thread_evolve = threading.Thread(target=evolve, args=(
            white_model, championship.white_with_points(use_baseline, baseline_net), players_number // 10,
            CROSSOVER_RATE, MUTATION_RATE_INDIVIDUALS, MUTATION_RATE_NEURONS, white_next_generation,
            lock, use_baseline))
        white_thread_evolve.start()

        wait_for_threads()

        # cleaning previous loaded models
        black_model.clear()
        for i in range(players_number):
            black_model[str(g + 1) + "_" + str(i)] = (black_next_generation[i][0], black_next_generation[i][1])

        white_model.clear()
        for i in range(players_number):
            white_model[str(g + 1) + "_" + str(i)] = (white_next_generation[i][0], white_next_generation[i][1])

        # create new directories for the mutated neural networks
        os.makedirs(folder + str(g + 1) + "/black/")
        os.makedirs(folder + str(g + 1) + "/white/")

        print("[INFO] saving evolved networks...")
        save_models_thread = threading.Thread(target=save_models, args=(folder, g + 1, black_model, white_model))
        save_models_thread.start()


if __name__ == '__main__':
    main()
