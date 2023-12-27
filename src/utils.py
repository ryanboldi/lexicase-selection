"""
This file defines some utility functions that are used in the rest of the code
"""
import numpy as np


def error_to_fitness(error_matrix):
    """
    Convert an error matrix to a fitness matrix
    :param error_matrix (np.ndarray): error matrix of shape (num_individuals, num_objectives)
    :return (np.ndarray): fitness matrix of shape (num_individuals, num_objectives)
    """
    return 1 / (1 + error_matrix)
