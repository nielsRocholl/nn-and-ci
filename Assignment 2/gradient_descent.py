import matplotlib.pyplot as plt
import numpy as np
# use direct imports to speed up the code
from numpy import ndarray
from numpy.random import choice
import pandas as pd


def plot_error_vs_epochs(errors: np.ndarray, epochs: np.ndarray, title: str, till=None):
    """
    Plot the error vs. epochs.
    :param errors: the errors
    :param epochs: the epochs
    :param title: the title of the plot
    """
    # if till is not None:
    #     errors = errors[:till]
    #     epochs = epochs[:till]
    plt.plot(epochs, errors)
    plt.ylim(0, 1)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.show()


def student_network(feature_vectors: np.ndarray, weights: np.ndarray, vk=1) -> ndarray:
    """
    Calculate the weighted sum of hidden states, i.e. the linear output.
    ALL VALUES MUST BE FLOATS.
    :param vk: fixed second layer weights
    :param feature_vectors: input vector
    :param weights: weight vector
    :return: the weighted sum of the hidden states
    """
    return np.sum(vk * np.tanh(np.dot(weights, feature_vectors)))


def sgd(feature_vectors_: np.ndarray, weights_: np.ndarray, labels_: np.ndarray, alpha, vk=1, t_max=10):
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
        return np.sum([contribution(xi, weights, label) for xi, label in zip(feature_vectors, labels)]) / len(
            feature_vectors)

    def delta(sigma, t):
        """
        Calculate delta = (sigma - t) * h'(sum_{j=1}^K v_j * g(w^{(j)} · xi))
        :param sigma: the output of the student network
        :param t: the target label
        h is the identity function h(x) = x, so h'(x) = 1
        g is the tanh function g(x) = tanh(x)
        v_j are the weights of the hidden to output layer, v_j = 1,
        w^{(j)} are the input to hidden weights and xi is the input.
        """
        return (sigma - t) * 1

    def gradient_of_single_input_to_hidden_weight(xi: np.ndarray, weight: np.ndarray, label):
        """
        Calculate the gradient of the cost function w.r.t. a single weight
        delta_wm = sigma * vk * g'(w_m · xi) * xi
        :param xi: a single data point / feature vector
        :param weight: m_th weight vector
        :param label: a single label
        :return: the gradient
        """
        d = delta(student_network(xi, weight), label)
        g_prime = 1 - np.tanh(np.dot(weight, xi)) ** 2
        return d * vk * g_prime * xi

    def train(feature_vectors=feature_vectors_, weights=weights_, labels=labels_, alpha=alpha, t_max=t_max):
        # keep track of the weights and errors
        weights_and_errors = []
        error = []
        for t in range(t_max):
            e = cost_function(feature_vectors, weights, labels)
            error.append(e)
            print("Epoch {}: error = {}".format(t, e))
            for _ in range(len(feature_vectors)):
                # pick a random data point
                i = np.random.randint(0, len(feature_vectors))
                xi = feature_vectors[i]
                label = labels[i]
                for k in np.random.permutation(len(weights)):
                    weights[k] -= alpha * gradient_of_single_input_to_hidden_weight(xi, weights[k], label)
            weights_and_errors.append((weights, e))
        # find the weights with the lowest error
        lowest_error = min(weights_and_errors, key=lambda x: x[1])
        index = weights_and_errors.index(lowest_error)
        print("Lowest error: {}".format(lowest_error[1]))
        plot_error_vs_epochs(np.array(error), np.arange(t_max), "Error vs. epochs", till=index)
        # return the weights with the lowest error
        return lowest_error[0]

    final_weights = train()
    return final_weights


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


def part_a(variables) -> None:
    """
    Perform part A of the assignment. Generate artificial data and train the soft committee network.
    :param variables:
    :return: None
    """
    print("Performing Part A")
    # generate data`
    feature_vectors, labels = generate_contiguous_data(variables['p'], variables['n'])
    # initialize input to hidden weights as independent random vectors with |w1|^2 = 1 and |w2|^2 = 1.
    input_to_hidden_weights = create_input_to_hidden_weights(variables['K'], variables['n'])
    # train the network
    weights = sgd(feature_vectors, input_to_hidden_weights, labels, variables['alpha'], t_max=variables['t_max'])


def part_b(variables) -> None:
    """
    :param variables:
    :return: None
    Perform part B of the assignment. Load the data and train the soft committee network.
    load the data comprise a 50 × 5000-dim. array xi corresponding to 5000 input vectors
    (dimension N = 50) and a 5000-dim. vector tau corresponding to the target values
    """
    print("Performing Part B")
    # get the first P data points
    feature_vectors = pd.read_csv('xi.csv', header=None).values.T[:variables['p']]
    labels = np.array(pd.read_csv('tau.csv', header=None).values).flatten()[:variables['p']]
    input_to_hidden_weights = create_input_to_hidden_weights(variables['K'], feature_vectors.shape[1])
    weights = sgd(feature_vectors, input_to_hidden_weights, labels, variables['alpha'], t_max=variables['t_max'])


def main():
    variables = {
        'vk': 1,  # fixed second layer weights
        'K': 2,  # number of hidden units
        'p': 100,  # number of data points
        'n': 5,  # number of dimensions
        'alpha': 0.05,  # learning rate
        't_max': 40,  # maximum number of iterations
    }
    # part_a(variables)
    part_b(variables)


if __name__ == '__main__':
    main()
