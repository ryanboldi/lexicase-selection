"""
File that holds the lexicase selection algorithm as well as other variants
"""
import numpy as np


def lexicase_selection(fitnesses, epsilon=False, elitism=False, num_to_select=None):
    """
    Lexicase selection algorithm
    :param fitnesses (np.ndarray): fitnesses of the population. This should be a
      numpy array of shape (num_individuals, num_objectives)
    :param epsilon (bool): (optional) whether or not to use epsilon lexicase
        selection. Epsilon lexicase relaxes the selection condition to instead be
        an epsilon away from the best obhjective value. Default is False
    :param elitism (bool): (optional) whether or not to use elitism. Elitism
        Ensures that the best individual (based on aggregate fitness) is always
        selected. The default is False
    :param num_to_select (int): (optional) number of individuals to select.
      Default is the same as the population size
    :return (np.ndarray): indices of selected individuals (shape (num_to_select,))
    """
    if num_to_select is None:
        num_to_select = fitnesses.shape[0]

    if epsilon:
        # do lexicase selection w/ epsilon
        x_median = np.median(fitnesses, axis=1)
        # Calculate absolute deviation from median
        dev = abs(fitnesses - x_median[:, None])
        mad = np.median(dev, axis=0)
    else:
        mad = np.zeros(fitnesses.shape[1])

    selected = np.zeros(num_to_select, dtype=int)

    for itr in range(
        # only start at 0 if not elitism
        int(elitism),
        num_to_select,
    ):
        num_features = fitnesses.shape[1]  # number of objectives
        features = np.arange(num_features)  # list of objective indices
        np.random.shuffle(features)  # Randomize the order of the objectives
        pool = np.ones(fitnesses.shape[0], dtype=bool)  # logical array if selected

        while (
            len(features) != 0 and np.sum(pool) != 1
        ):  # while we still have cases to use
            current_feature = features[0]  # this is the objective we are using
            features = features[1:]  # this is the rest of the objectives

            best = np.max(
                fitnesses[pool, current_feature]
            )  # get the best on this objective
            old_pool = pool

            # filter selected pop with this feature.
            pool = np.logical_and(
                pool,
                fitnesses[:, current_feature] >= best - mad[current_feature],
            )
            # If it filters everyone, skip this objective and continue
            if np.sum(pool) == 0:
                pool = old_pool
                continue

        selected[itr] = np.random.choice(
            np.nonzero(pool)[0]
        )  # choose a random individual from the pool

    return selected
