import keras
import numpy as np
import heapq
from operator import itemgetter


def evolve(cur_gen, rank, n_kept, crossover_rate, mutation_rate_individual, mutation_rate_neuron, new_gen, lock,
           USE_BASELINE):
    """This function implements the evolution strategy
        :param cur_gen: dictionary containing the ids of the network pair and the corresponding
            tuples (model_from, model_to)
        :param rank: dictionary containing the ids of the network pair and the corresponding score
        :param n_kept: number of the best individuals that must be cloned into the new generation
        :param crossover_rate: rate of crossover, expressed as a real number between 0 and 1
        :param mutation_rate_individual: rate of mutation for the individuals,
                expressed as a real number between 0 and 1
        :param mutation_rate_neuron: rate of mutation for the neurons, expressed as a real number between 0 and 1

        :return next_gen: list of the new network pairs, obtained by neuroevolution
    """
    # Seed
    np.random.seed(42)

    # Calculate fitness
    fitness = calculate_fitness(cur_gen, rank)
    # Prepare list for new generation

    # Only the n_kept best individuals are cloned
    top_individuals = list(dict(heapq.nlargest(n_kept, fitness.items(), key=itemgetter(1))).keys())
    for i in range(0, n_kept):
        # Get id
        net_id = top_individuals[i]
        # Get the from and to models of the corresponding net (tuple)
        model_from, model_to = cur_gen[net_id][0], cur_gen[net_id][1]
        # Mutation with 0 rate => perfect clone
        clone_from = mutate(model_from, 0, 0, lock)
        clone_to = mutate(model_to, 0, 0, lock)
        # Append to new_gen
        new_gen.append((clone_from, clone_to))

    # Check if the remainder individuals are uneven
    if USE_BASELINE:
        remainder = len(cur_gen) + 1 - n_kept
    else:
        remainder = len(cur_gen) - n_kept
    if remainder % 2 != 0:
        # If they are, add a single mutated offspring and decrease the remainder number
        model_from, model_to = select(cur_gen, fitness)
        model_from = mutate(model_from, mutation_rate_individual, mutation_rate_neuron, lock)
        model_to = mutate(model_to, mutation_rate_individual, mutation_rate_neuron, lock)
        new_gen.append((model_from, model_to))
        remainder -= 1

    # Selection + crossover + mutation
    if USE_BASELINE:
        limit = len(cur_gen) + 1 - n_kept
    else:
        limit = len(cur_gen) - n_kept
    for i in range(0, limit, 2):
        # Select two parent tuples
        parent_tuple = select(cur_gen, fitness)
        parent_from1, parent_to1 = parent_tuple[0], parent_tuple[1]
        parent_tuple = select(cur_gen, fitness)
        parent_from2, parent_to2 = parent_tuple[0], parent_tuple[1]
        # Crossover for model_from and model_to
        child_from1, child_from2 = crossover(parent_from1, parent_from2, crossover_rate, lock)
        child_to1, child_to2 = crossover(parent_to1, parent_to2, crossover_rate, lock)
        # Mutation of the children
        child_from1 = mutate(child_from1, mutation_rate_individual, mutation_rate_neuron, lock)
        child_to1 = mutate(child_to1, mutation_rate_individual, mutation_rate_neuron, lock)
        child_from2 = mutate(child_from2, mutation_rate_individual, mutation_rate_neuron, lock)
        child_to2 = mutate(child_to2, mutation_rate_individual, mutation_rate_neuron, lock)
        # Append to new_gen
        new_gen.append((child_from1, child_to1))
        new_gen.append((child_from2, child_to2))

    # new_gen is returned automatically as a parameter


def calculate_fitness(cur_gen, rank):
    """This function returns a dictionary with the fitnesses of the nets
        :param cur_gen: dictionary containing the ids of the network pair and the corresponding
            tuples (model_from, model_to)
        :param rank: dictionary containing the ids of the network pair and the corresponding score

        :returns fitness: dictionary that associates a fitness to an id
    """

    # Prepare fitness dictionary
    fitness = {}
    s = 0
    for net_id in cur_gen:
        s += rank[net_id]
    # Fitness normalized
    for net_id in cur_gen:
        f = rank[net_id] / s
        fitness.update({net_id: f})
    return fitness


def select(cur_gen, fitness):
    """This function implements the selection strategy (RWM)
        :param cur_gen: dictionary containing the ids of the network pair and the corresponding
            tuples (model_from, model_to)
        :param fitness: dictionary that associates a fitness to an id

        :returns net: tuple of model_from and model_to selected from the dictionary
    """

    # List of ids
    ids = list(cur_gen.keys())

    # RWM
    index = 0
    r = np.random.rand() * 1
    while r > 0:
        r -= fitness.get(ids[index])
        index += 1
    index -= 1

    return cur_gen[ids[index]]


def mutate(model, rate_individual, rate_neuron, lock):
    """This function defines and applies a mutation function to every weight and bias of the keras model
        :param model: the Neural Network of type keras.engine.training.Model that must be mutated
        :param rate_individual: the mutation rate for the individuals, expressed as a real number between 0 and 1
        :param rate_neuron: the mutation rate for the neurons, expressed as a real number between 0 and 1

        :returns new_model: the new Neural Network of type keras.engine.training.Model
    """

    # Parameter check
    if not isinstance(model, keras.models.Model):
        raise TypeError('keras.engine.training.Model expected,', type(model), 'found')

    # Mutation function to be applied
    def mutation_fn(x):
        if np.random.rand() < rate_neuron:
            return x + np.random.randn()
        else:
            return x

    # Converting mutation_fn to Python's ufunc in order to improve performance
    mutation_ufn = np.frompyfunc(mutation_fn, 1, 1)

    # Apply mutation_fn
    with lock:
        new_model = keras.models.clone_model(model)
        dna = model.get_weights()
    new_dna = []
    if np.random.rand() < rate_individual:
        for i in range(0, len(dna), 2):
            weights, biases = dna[i], dna[i + 1]
            new_weights, new_biases = mutation_ufn(weights), mutation_ufn(biases)
            new_dna.append(new_weights)
            new_dna.append(new_biases)
        with lock:
            new_model.set_weights(new_dna)
    else:
        with lock:
            new_model.set_weights(dna)

    return new_model


def crossover(parent1, parent2, rate, lock):
    """This function performs the crossover between two keras models
        :param parent1: the first parent Neural Network of type keras.engine.training.Model
        :param parent2: the second parent Neural Network of type keras.engine.training.Model
        :param rate: the crossover rate, expressed as a real number between 0 and 1

        :returns child1: the first child Neural Network of type keras.engine.training.Model
        :returns child2: the second child Neural Network of type keras.engine.training.Model
    """

    # Parameter check
    if not isinstance(parent1, keras.models.Model) or not isinstance(parent2, keras.models.Model):
        raise TypeError('Two keras.engine.training.Model expected,', type(parent1), 'and', type(parent2), 'found')

    # Cloning of parents
    with lock:
        child1 = keras.models.clone_model(parent1)
        child2 = keras.models.clone_model(parent2)
    if np.random.rand() < rate:
        # DNA
        with lock:
            dna1 = parent1.get_weights()
            dna2 = parent2.get_weights()
        new_dna1 = []
        new_dna2 = []

        for i in range(0, len(dna1), 2):
            # First parent chromosomes (one for weights, one for biases)
            parent1_w_chromosome, parent1_b_chromosome, w_shape1 = get_chromosomes(dna1, i)
            # Second parent chromosomes (one for weights, one for biases)
            parent2_w_chromosome, parent2_b_chromosome, w_shape2 = get_chromosomes(dna2, i)

            # Random cross-points
            w_len = w_shape1[0] * w_shape1[1]
            b_len = len(parent1_b_chromosome)
            w_cross = np.random.randint(low=0, high=w_len)
            b_cross = np.random.randint(low=0, high=b_len)

            # Crossover
            child1_w_chromosome = np.concatenate((parent1_w_chromosome[0:w_cross], parent2_w_chromosome[w_cross:w_len]))
            child1_b_chromosome = np.concatenate((parent1_b_chromosome[0:b_cross], parent2_b_chromosome[b_cross:b_len]))
            child2_w_chromosome = np.concatenate((parent2_w_chromosome[0:w_cross], parent1_w_chromosome[w_cross:w_len]))
            child2_b_chromosome = np.concatenate((parent2_b_chromosome[0:b_cross], parent1_b_chromosome[b_cross:b_len]))

            # Reshaping
            child1_w_chromosome = np.reshape(child1_w_chromosome, w_shape1)
            child2_w_chromosome = np.reshape(child2_w_chromosome, w_shape2)

            # Adding to children DNA
            new_dna1.append(child1_w_chromosome)
            new_dna1.append(child1_b_chromosome)
            new_dna2.append(child2_w_chromosome)
            new_dna2.append(child2_b_chromosome)
        with lock:
            child1.set_weights(new_dna1)
            child2.set_weights(new_dna2)
    else:
        with lock:
            child1.set_weights(parent1.get_weights())
            child2.set_weights(parent2.get_weights())

    return child1, child2


def get_chromosomes(dna, index):
    weights, chromosome_b = dna[index], dna[index + 1]
    w_shape = weights.shape
    chromosome_w = np.reshape(weights, [w_shape[0] * w_shape[1]])

    return chromosome_w, chromosome_b, w_shape
