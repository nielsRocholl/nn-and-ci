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


def sgd(feature_vectors: np.ndarray, weights: np.ndarray, labels: np.ndarray, vk=1):
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

    def delta(sigma, t, weights, feature_vector):
        """
        delta = (sigma - t) * h'(sum_{j=1}^K v_j * g(w^{(j)} · xi))
        sigma is the output of the student network, t is the target, h is the identity function and g
        is the tanh function, v_j are the weights of the hidden to output layer,
        w^{(j)} are the input to hidden weights and xi is the input.
        """
        return (sigma - t) * (1 - np.tanh(np.sum(vk * np.dot(weights, feature_vector))) ** 2)

    def gradient_of_single_input_to_hidden_weight(xi: np.array, weight: np.ndarray, label):
        """
        Calculate the gradient of the cost function w.r.t. a single weight
        delta_wm = sigma * vk * g'(w_m · xi) * xi
        :param xi: a single data point / feature vector
        :param weight: m_th weight vector
        :param label: a single label
        :return: the gradient
        """
        d = delta(student_network(xi, weights), label, weights, xi)
        g_prime = 1 - np.tanh(np.dot(weight, xi)) ** 2
        return d * vk * g_prime * xi

    def total_error():
        sigma = student_network(feature_vectors, weights)
        return np.sum([(delta(sigma, t, weights, xi) * np.tanh() ) for t, xi in zip(labels, feature_vectors)])



    print(feature_vectors[0], weights[0], labels[0])
    print("gradient: ", gradient_of_single_input_to_hidden_weight(feature_vectors[0], weights[0], labels[0]))


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

    sgd(feature_vectors, input_to_hidden_weights, labels)



if __name__ == '__main__':
    main()
