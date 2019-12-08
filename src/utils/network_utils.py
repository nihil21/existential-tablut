from neuro_evolution.genetic_algorithm import mutate
import numpy as np


def generate(initial_models, n):
    """
    Given an initial list of existing keras models, this function applies genetic operators and return a list of new
    models
    :param initial_models: list of existing keras models
    :param n: number of models that must be generated
    :return: new_models: list of new models, based on the initial ones but created by applying genetic operators
    """
    # List that will contain new models
    new_models = []

    # Nets received are less than the nets to be generated
    if len(initial_models) < n:
        # Copy initial models and append them to new_models
        for i in range(0, len(initial_models)):
            model_from, model_to = initial_models[i][0], initial_models[i][1]
            # Mutation with 0 rate => perfect clone
            model_from, model_to = mutate(model_from, 0, 0), mutate(model_to, 0, 0)
            new_models.append((model_from, model_to))

        # Mutation rates
        rate_i = 0.5
        rate_n = 0.005

        # Generate remaining models
        for i in range(len(initial_models), n):
            # Random index
            index = np.random.randint(low=0, high=len(initial_models))
            model_from, model_to = initial_models[index][0], initial_models[index][1]
            # Mutation
            model_from = mutate(model_from, rate_individual=rate_i, rate_neuron=rate_n)
            model_to = mutate(model_to, rate_individual=rate_i, rate_neuron=rate_n)
            new_models.append((model_from, model_to))

        return new_models


def save_models(model_list, path, role, gen_num):
    """
    Function that saves in batch a list of models
    :param model_list: list of the networks (tuples of from-to models)
    :param path: folder in which the models will be saved
    :param role: W (white) or B (black)
    :param gen_num: integer that represents the generation number (for naming only)
    """
    if role == 'W':
        path += "white/"
    elif role == 'B':
        path += "black/"

    for i in range(0, len(model_list)):
        model_tuple = model_list[i]
        model_from, model_to = model_tuple[0], model_tuple[1]
        model_from.save(path + "modelF" + role + "_" + str(gen_num) + "_" + str(i))
        model_to.save(path + "modelT" + role + "_" + str(gen_num) + "_" + str(i))
