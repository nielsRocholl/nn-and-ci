import os
import time
import matplotlib.pyplot as plt
import numpy as np
# use direct imports to speed up the code
from numpy import arange, array, dot, zeros, sort, ndarray
from numpy.random import choice, normal
from pandas import DataFrame, concat, read_csv
from matplotlib.pyplot import show, subplots
# use numba for speed up
from numba import jit


# @jit(nopython=True)
def student_network(feature_vectors: np.array, weights: np.ndarray, vk=1) -> ndarray:
    """
    Calculate the weighted sum of hidden states, i.e. the linear output.
    ALL VALUES MUST BE FLOATS.
    :param vk: fixed second layer weights
    :param feature_vectors: input vector
    :param weights: weight vector
    :return: the weighted sum of the hidden states
    """
    return np.sum(np.tanh(vk * np.dot(weights, feature_vectors)))


def sgd(feature_vectors: np.ndarray, weights: np.ndarray, labels: np.ndarray):
    """
    stochastic gradient descent procedure w.r.t. the weight vectors wk, k = 1, 2, . . . K ,
    aimed at the minimization of the cost function
    :return: the updated weight vectors
    """

    def contribution(xi: np.array, weights: np.ndarray, label):
        """
        Calculate the contribution of a data point to the gradient.
        :param xi: a single data point/ feature vector
        :param weights: the weight vector
        :param label: a single label
        :return: the contribution of a data point to the gradient
        """
        return ((student_network(xi, weights) - label) ** 2) / 2

    def cost_function(feature_vectors: np.ndarray, weights: np.ndarray, labels: np.ndarray):
        """
        Calculate the cost function, i.e. the sum of the contributions of all data points.
        :param feature_vectors: the feature vectors
        :param weights: the weight vector
        :param labels: the labels
        :return: the weighted sum of the hidden states
        """
        return np.sum([contribution(xi, weights, label) for xi, label in zip(feature_vectors, labels)])

    def gradient(xi: np.array, weights: np.ndarray, label):
        """
        Calculate the gradient of the cost function w.r.t. a single weight
        :param xi: a single data point / feature vector
        :param weights: the weight vector
        :param label: a single label
        :return: the gradient of the cost function
        """
        # TODO: Geen idee of deze gradient calculation klopt
        prediction = student_network(xi, weights)
        return (prediction - label) * xi

    def gradient_via_central_difference(xi: np.array, weights: np.ndarray, label):
        """
        Calculate the gradient of the cost function w.r.t. a single weight
        :param xi: a single data point / feature vector
        :param weights: the weight vector
        :param label: a single label
        :return: the gradient of the cost function
        """
        epsilon = 1e-4
        grad = np.zeros_like(weights)
        print(grad)
        for i in range(weights.shape[0]):
            weights_plus = weights.copy()
            weights_minus = weights.copy()
            weights_plus[i] += epsilon / 2
            weights_minus[i] -= epsilon / 2
            grad[i] = (contribution(xi, weights_plus, label) - contribution(xi, weights_minus, label)) / epsilon
        return grad


    # TODO: ik weet niet of een van de twee gradienten klopt, hier moet naar gekeken worden. Ik heb de TA's gemaild
    return None


def generate_contiguous_data(p, n) -> tuple:
    """
    data set ID = {ξ, τ(ξ)} with continuous training labels τ(ξ) ∈ IR.
    :param p: number of data points
    :param n: number of dimensions
    :return: dataset of size p, where each ξ is of dimension n, with continuous training labels τ(ξ) ∈ IR
    """
    # create a vector of n independent random components, with mean 0 and variance 1
    feature_vectors = np.random.normal(0, 1, (p, n))
    # create a random continuous label of uniform distribution between -1 and 1
    labels = np.random.uniform(-1, 1, p)
    return feature_vectors, labels


def create_input_to_hidden_weights(K, n) -> np.ndarray:
    """
    create input to hidden weights as independent random vectors with |w1|^2 = 1 and |w2|^2 = 1.
    :param K: number of hidden units
    :param n: number of dimensions
    :return: weights
    """
    weights = np.random.normal(0, 1, (K, n))
    # normalize the weights
    for i in range(K):
        weights[i] = weights[i] / np.linalg.norm(weights[i])
    return weights


def main():
    variables = {
        'vk': 1,  # fixed second layer weights
        'K': 2,  # number of hidden units
        'p': 100,  # number of data points
        'n': 5,  # number of dimensions
        'alpha': 0.05,  # learning rate
    }
    feature_vectors, labels = generate_contiguous_data(variables['p'], variables['n'])
    # initialize input to hidden weights as independent random vectors with |w1|^2 = 1 and |w2|^2 = 1.
    input_to_hidden_weights = create_input_to_hidden_weights(variables['K'], variables['n'])
    # initialize hidden to output weights according to vk
    hidden_to_output_weights = np.array([variables['vk'] for _ in range(variables['K'])])

    # print(sgd(feature_vectors, input_to_hidden_weights, labels))
    # print(input_to_hidden_weights[0])



if __name__ == '__main__':
    main()
