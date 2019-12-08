import argparse

from keras.models import load_model
import pickle
import threading
from time import sleep
import os

from predictPlayNN import connectAndPlay
from championship_manager import Championship
from neuro_evolution.genetic_algorithm import evolve


def loadModels(folder, generation, playersNumber):
    # blackModel = {"id": (from, to), ...}
    blackModel = {}
    whiteModel = {}

    genStr = str(generation)

    for i in range(playersNumber):
        blackModel[genStr + "_" + str(i)] = (
            load_model(folder + genStr + "/black/modelFB_" + genStr + "_" + str(i), compile=False),
            load_model(folder + genStr + "/black/modelTB_" + genStr + "_" + str(i), compile=False)
        )
        whiteModel[genStr + "_" + str(i)] = (
            load_model(folder + genStr + "/white/modelFW_" + genStr + "_" + str(i), compile=False),
            load_model(folder + genStr + "/white/modelTW_" + genStr + "_" + str(i), compile=False)
        )

    # model.predict() is not thread safe, so we have to compute the predict function here before creating threads
    for k in blackModel.keys():
        blackModel[k][0]._make_predict_function()
        blackModel[k][1]._make_predict_function()
    for k in whiteModel.keys():
        whiteModel[k][0]._make_predict_function()
        whiteModel[k][1]._make_predict_function()

    return blackModel, whiteModel


def loadLabels(folder):
    blackLabel = []
    whiteLabel = []

    blackLabel.append(pickle.loads(open(folder + "label/labelFB", "rb").read()))
    blackLabel.append(pickle.loads(open(folder + "label/labelTB", "rb").read()))

    whiteLabel.append(pickle.loads(open(folder + "label/labelFW", "rb").read()))
    whiteLabel.append(pickle.loads(open(folder + "label/labelTW", "rb").read()))

    return blackLabel, whiteLabel


def saveReport(folder, championship, generationNumber):
    with open(folder + "report/report_" + str(generationNumber) + ".txt", "w") as reportFile:
        reportFile.write("Black with points:\n")
        # print also baseline scores
        champ = championship.black_with_points(False, "baseline_net")
        sortedChamp = sorted(champ.items(), key=lambda kv: kv[1], reverse=True)
        details = championship.black_with_score()
        i = 1
        for net in sortedChamp:
            reportFile.write(str(i) + ") " + net[0] + " " + str(net[1]) + " " + str(details[net[0]]) + "\n")
            i += 1

        reportFile.write("\nWhite with points:\n")
        # print also baseline scores
        champ = championship.white_with_points(False, "baseline_net")
        sortedChamp = sorted(champ.items(), key=lambda kv: kv[1], reverse=True)
        details = championship.white_with_score()
        i = 1
        for net in sortedChamp:
            reportFile.write(str(i) + ") " + net[0] + " " + str(net[1]) + " " + str(details[net[0]]) + "\n")
            i += 1


def waitForThreads():
    mainThread = threading.currentThread()
    for t in threading.enumerate():
        if t is not mainThread:
            t.join()


def saveModels(folder, generation, blackNewModel, whiteNewModel):
    for k in blackNewModel.keys():
        blackNewModel[k][0].save(folder + str(generation) + "/black/modelFB_" + k)
        blackNewModel[k][1].save(folder + str(generation) + "/black/modelTB_" + k)

    for k in whiteNewModel.keys():
        whiteNewModel[k][0].save(folder + str(generation) + "/white/modelFW_" + k)
        whiteNewModel[k][1].save(folder + str(generation) + "/white/modelTW_" + k)


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

WHITEPORT = 5800
BLACKPORT = 5801

CROSSOVERRATE = 0.01
MUTATIONRATE_INDIVIDUALS = 0.5
MUTATIONRATE_NEURONS = 0.005

folder = args["net_folder"]
if folder[-1] != "/":
    folder += "/"

startingGenerationNumber = int(args["starting_generation_number"])
playersNumber = int(args["players_number"])

generationNumber = int(args["generation_number"])

# load neural networks from the folder provided
print("[INFO] loading networks and label binarizers...")
blackModel, whiteModel = loadModels(folder, startingGenerationNumber, playersNumber)
blackLabel, whiteLabel = loadLabels(folder)

for g in range(startingGenerationNumber, startingGenerationNumber + generationNumber):
    print("[INFO] generation number " + str(g))

    # generate championship for this generation
    championship = Championship([str(g) + "_" + str(i) for i in range(playersNumber)])

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
    USE_BASELINE = int(args["baseline"])
    # folder that contains moves that makes baseline lose against a net
    if USE_BASELINE and not os.path.isdir(folder + 'baseline_defeated_by/'):
        os.makedirs(folder + 'baseline_defeated_by/')
    baseline_net = '_' + str(playersNumber - 1)
    # remove all matches where the baseline does not play, and where both are baseline
    if USE_BASELINE:
        matches = [match
                   for match in matches
                   if (match[0][-len(baseline_net):] == baseline_net or match[1][-len(baseline_net):] == baseline_net)
                   and not (match[0][-len(baseline_net):] == baseline_net
                            and match[1][-len(baseline_net):] == baseline_net)]
        matches *= 5
    ##########################################################################

    # lock to correctly use Theano
    lock = threading.Lock()

    print("[INFO] playing " + str(len(matches)) + " matches...")

    for m in matches:
        # white player created
        whitePlayer = m[0]
        baseline_player = False
        if USE_BASELINE and whitePlayer[-len(baseline_net):] == baseline_net:
            baseline_player = True
        modelFrom = whiteModel[whitePlayer][0]
        modelTo = whiteModel[whitePlayer][1]

        whiteThreadPlay = threading.Thread(target=connectAndPlay, args=(
            modelFrom, modelTo, whiteLabel[0], whiteLabel[1], whitePlayer, "W", WHITEPORT, championship, lock,
            baseline_player, folder + 'baseline_defeated_by/', m[1]))
        whiteThreadPlay.start()

        # black player created
        blackPlayer = m[1]
        baseline_player = False
        if USE_BASELINE and blackPlayer[-len(baseline_net):] == baseline_net:
            baseline_player = True
        modelFrom = blackModel[blackPlayer][0]
        modelTo = blackModel[blackPlayer][1]

        blackThreadPlay = threading.Thread(target=connectAndPlay, args=(
            modelFrom, modelTo, blackLabel[0], blackLabel[1], blackPlayer, "B", BLACKPORT, championship, lock,
            baseline_player, folder + 'baseline_defeated_by/', m[0]))
        blackThreadPlay.start()

        if limited_match_per_time:
            print("started match number: " + str(num_current_match))
            if num_current_match % match_per_time == 0 or num_current_match == len(matches) - 1:
                waitForThreads()
            else:
                sleep(0.5)
            num_current_match += 1
        else:
            sleep(0.5)

    if not limited_match_per_time:
        waitForThreads()

    # print report
    saveReport(folder, championship, g)

    # evolution of the networks
    print("[INFO] evolving networks...")

    blackNextGeneration = []
    if USE_BASELINE:
        blackModel = {key: value for key, value in blackModel.items() if key[-len(baseline_net):] != baseline_net}
    print("[INFO] evolving " + str(len(blackModel.keys())) + " nets ...")
    blackThreadEvolve = threading.Thread(target=evolve, args=(
        blackModel, championship.black_with_points(USE_BASELINE, baseline_net), playersNumber // 10, CROSSOVERRATE,
        MUTATIONRATE_INDIVIDUALS, MUTATIONRATE_NEURONS, blackNextGeneration, lock, USE_BASELINE))
    blackThreadEvolve.start()

    whiteNextGeneration = []
    if USE_BASELINE:
        whiteModel = {key: value for key, value in whiteModel.items() if key[-len(baseline_net):] != baseline_net}
    whiteThreadEvolve = threading.Thread(target=evolve, args=(
        whiteModel, championship.white_with_points(USE_BASELINE, baseline_net), playersNumber // 10, CROSSOVERRATE,
        MUTATIONRATE_INDIVIDUALS, MUTATIONRATE_NEURONS, whiteNextGeneration, lock, USE_BASELINE))
    whiteThreadEvolve.start()

    waitForThreads()

    # cleaning previous loaded models
    blackModel.clear()
    for i in range(playersNumber):
        blackModel[str(g + 1) + "_" + str(i)] = (blackNextGeneration[i][0], blackNextGeneration[i][1])

    whiteModel.clear()
    for i in range(playersNumber):
        whiteModel[str(g + 1) + "_" + str(i)] = (whiteNextGeneration[i][0], whiteNextGeneration[i][1])

    # create new directories for the mutated neural networks
    os.makedirs(folder + str(g + 1) + "/black/")
    os.makedirs(folder + str(g + 1) + "/white/")

    print("[INFO] saving evolved networks...")
    saveModelsThread = threading.Thread(target=saveModels, args=(folder, g + 1, blackModel, whiteModel))
    saveModelsThread.start()
