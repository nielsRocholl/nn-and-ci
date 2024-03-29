import os

import matplotlib.pyplot as plt
import numpy as np
# use direct imports to speed up the code
from numpy import ndarray
from numpy.random import choice
import pandas as pd


def plot_error_vs_epochs(errors: np.ndarray, error_generalization: np.ndarray, epochs: np.ndarray, title: str,
                         variables: dict):
    """
    Plot the error vs. epochs.
    :param variables: the variables used in the experiment
    :param error_generalization: the generalization error
    :param errors: the errors
    :param epochs: the epochs
    :param title: the title of the plot
    """
    plt.figure(figsize=(7, 5), dpi=200)
    plt.plot(epochs, errors, label="Training error")
    plt.plot(epochs, error_generalization, label="Generalization error")
    plt.ylim(0, 1)
    plt.grid()
    variables = ' '.join([f'{key}={value}' for key, value in variables.items()])
    plt.legend(fontsize='large')
    plt.title(title, fontdict={'fontsize': 18})
    plt.xlabel("Epochs", fontdict={'fontsize': 16})
    plt.ylabel("Error", fontdict={'fontsize': 16})

    plt.tick_params(labelsize=14)
    plt.savefig(f'plots/{title}_{variables}.png')
    plt.show()


def plot_error_vs_epochs_for_different_p():
    """
    Plot the error vs epoch, for different p values.
    """
    # load the data
    data = pd.read_csv('results/error.csv')
    # create a list of colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    # create a subplot for each p value
    fig, axes = plt.subplots(nrows=len(data.p.unique()), figsize=(7, 2 * len(data.p.unique())), dpi=200, sharey=True)
    # iterate over the p values and plot the data in the corresponding subplot
    for i, p in enumerate(data.p.unique()):
        # filter the data for the current p value
        data_p = data[data.p == p]
        c = colors.pop()
        axes[i].plot(data_p.epoch, data_p.error, label=f'Train Error (p={p})', color=c, linewidth=2)
        axes[i].plot(data_p.epoch, data_p.error_generalization, label=f'Test Error (p={p})', linestyle='--', color=c,
                     linewidth=2)
        axes[i].grid(linewidth=0.5)
        # axes[i].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large')
        axes[i].legend(fontsize='medium', loc='upper right')
    fig.text(0.5, 0.04, 'Epochs', ha='center', size=16)
    fig.text(0.04, 0.5, 'Error', va='center', rotation='vertical', size=16)
    fig.suptitle('Error vs. Epochs for Different p Values', size=18)
    plt.savefig('plots/error_vs_epochs_for_different_p.png')
    plt.show()


def plot_weight_vector(weights: np.ndarray) -> None:
    """
    Plot the weight vectors in a bar graph.
    :param weights: the weight vectors
    """

    def plot_bar() -> None:
        """
        Plot the weight vectors in a bar graph.
        :return: None
        """
        x = np.arange(50)
        width = 0.40
        fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
        # add whitespace around the plot
        fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
        ax.bar(x - width / 2, weights[0], width, label='First Hidden Unit', color='orange', linewidth=2)
        ax.bar(x + width / 2, weights[1], width, label='Second Hidden Unit', color='blue', linewidth=2)
        ax.set_ylabel('Weight Value', fontdict={'fontsize': 16})
        ax.set_xlabel('Weight Index', fontdict={'fontsize': 16})
        ax.set_title('Weight Vectors', fontdict={'fontsize': 18})
        plt.tick_params(labelsize=14)
        ax.legend(fontsize='large')
        plt.savefig('plots/weight_vectors_bar.png')
        plt.show()

    def pot_3d() -> None:
        """
        Plot the weight vectors in a 3D graph.
        :return: None
        """
        fig = plt.figure(figsize=(7, 7), dpi=200)
        ax = fig.add_subplot(111, projection='3d')
        x = np.arange(50)
        y1 = np.ones(50)
        y2 = np.ones(50) * 2
        ax.plot(x, y1, weights[0], label='First Hidden Unit')
        ax.plot(x, y2, weights[1], label='Second Hidden Unit')
        # add whitespace on the right
        fig.subplots_adjust(right=0.8)
        ax.set_xlabel('Weight Index', fontdict={'fontsize': 14}, labelpad=20)
        ax.set_ylabel('Hidden Unit', fontdict={'fontsize': 14}, labelpad=20)
        ax.set_zlabel('Weight Value', fontdict={'fontsize': 14}, labelpad=20)
        ax.set_title('Weight Vectors', fontdict={'fontsize': 16})
        plt.tick_params(labelsize=14, pad=10)
        # move the legend tiny bit to the top right
        ax.legend(loc='upper left', bbox_to_anchor=(0.7, 1), fontsize='large')
        plt.savefig('plots/weight_vectors_3d.png')
        plt.show()

    def plot_heatmap() -> None:
        """
        Plot the weight vectors in a heatmap.
        :return: None
        """
        fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
        plt.imshow(weights, cmap='viridis', aspect=5)
        plt.colorbar()

        # Add labels and a title
        plt.xlabel('Weight Index', fontdict={'fontsize': 16})
        plt.ylabel('Hidden Unit', fontdict={'fontsize': 16})
        plt.title('Weight Vectors', fontdict={'fontsize': 18})
        plt.tick_params(labelsize=14)
        plt.savefig('plots/weight_vectors_heatmap.png')
        plt.show()

    plot_bar()
    pot_3d()
    plot_heatmap()


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


def sgd(feature_vectors_: np.ndarray, labels_: np.ndarray,
        feature_vectors_gen_: np.ndarray, labels_gen_: np.ndarray,
        weights_: np.ndarray, alpha, vk=1, t_max=10, adaptive=False) -> tuple:
    """
    stochastic gradient descent procedure w.r.t. the weight vectors wk, k = 1, 2, . . . K ,
    aimed at the minimization of the cost function
    :return: the updated weight vectors
    """

    def contribution(xi: np.array, weights: np.ndarray, label) -> float:
        """
        Calculate the contribution of a data point to the gradient.
        :param xi: a single data point/ feature vector
        :param weights: the weight vector
        :param label: a single label
        :return: the contribution of a data point to the gradient
        """
        return ((student_network(xi, weights) - label) ** 2) / 2

    def cost_function(feature_vectors: np.ndarray, weights: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate the cost function, i.e. the sum of the contributions of all data points.
        :param feature_vectors: the feature vectors
        :param weights: the weight vector
        :param labels: the labels
        :return: the weighted sum of the hidden states
        """
        return np.sum([contribution(xi, weights, label) for xi, label in zip(feature_vectors, labels)]) / len(
            feature_vectors)

    def generalization_error(feature_vectors: np.ndarray, weights: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate the cost function, i.e. the sum of the contributions of all data points.
        :param feature_vectors: the feature vectors
        :param weights: the weight vector
        :param labels: the labels
        :return: the weighted sum of the hidden states
        """
        return np.sum([contribution(xi, weights, label) for xi, label in zip(feature_vectors, labels)]) / len(
            feature_vectors)

    def delta(sigma, t) -> float:
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

    def gradient_of_single_input_to_hidden_weight(xi: np.ndarray, weight: np.ndarray, label) -> float:
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

    def train(feature_vectors=feature_vectors_, labels=labels_,
              feature_vectors_gen=feature_vectors_gen_, labels_gen=labels_gen_,
              weights=weights_, alpha=alpha, t_max=t_max, adaptive=adaptive) -> tuple:
        # keep track of the weights and errors
        weights_and_errors = []
        error = []
        error_generalization = []
        for t in range(t_max):
            e = cost_function(feature_vectors, weights, labels)
            e_generalization = generalization_error(feature_vectors_gen, weights, labels_gen)
            error.append(e)
            error_generalization.append(e_generalization)
            for _ in range(len(feature_vectors)):
                # pick a random data point
                i = np.random.randint(0, len(feature_vectors))
                xi = feature_vectors[i]
                label = labels[i]
                for k in range(len(weights)):
                    weights[k] -= alpha * gradient_of_single_input_to_hidden_weight(xi, weights[k], label)
            weights_and_errors.append((weights, error_generalization))
            if adaptive:
                alpha *= 0.9
        # find the weights with the lowest error
        lowest_error = min(weights_and_errors, key=lambda x: x[1])
        # return the weights with the lowest error
        return lowest_error[0], error, error_generalization

    final_weights, error, error_generalization = train()
    return final_weights, error, error_generalization


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


def run_assignment(variables) -> None:
    """
    :param variables:
    :return: None
    Perform part B of the assignment. Load the data and train the soft committee network.
    load the data comprise a 50 × 5000-dim. array xi corresponding to 5000 input vectors
    (dimension N = 50) and a 5000-dim. vector tau corresponding to the target values
    """
    # get the first P data points
    feature_vectors = pd.read_csv('xi.csv', header=None).values.T[:variables['p']]
    labels = np.array(pd.read_csv('tau.csv', header=None).values).flatten()[:variables['p']]
    # get test data
    feature_vectors_test = pd.read_csv('xi.csv', header=None).values.T[
                           variables['p']:variables['p'] + int(variables['p'] * 0.3)]
    labels_test = np.array(pd.read_csv('tau.csv', header=None).values).flatten()[
                  variables['p']:variables['p'] + int(variables['p'] * 0.3)]
    e = []
    e_gen = []
    for experiment in range(variables['experiments']):
        input_to_hidden_weights = create_input_to_hidden_weights(variables['K'], feature_vectors.shape[1])
        weights, error, error_generalization = sgd(feature_vectors, labels, feature_vectors_test, labels_test,
                                                   input_to_hidden_weights, variables['alpha'],
                                                   t_max=variables['t_max'], adaptive=variables['adaptive'])
        e.append(error)
        e_gen.append(error_generalization)
        # if there is more than one hidden unit, plot the weight vectors
        if experiment == variables['experiments'] - 1 and variables['K'] > 1:
            plot_weight_vector(weights)
    # average the errors
    e = np.mean(np.array(e), axis=0)
    e_gen = np.mean(np.array(e_gen), axis=0)
    # save the errors to a file
    df = pd.DataFrame(
        {'p': variables['p'], 'error': e, 'error_generalization': e_gen, 'epoch': np.arange(variables['t_max'])})
    if os.path.exists('results/error.csv'):
        # check if p is already in the file
        if not pd.read_csv('results/error.csv').p.isin([variables['p']]).any():
            df.to_csv('results/error.csv', mode='a', header=False)
    else:
        df.to_csv('results/error.csv')
    plot_error_vs_epochs(e, e_gen, np.arange(variables['t_max']), "Error vs. epochs", variables)


def main():
    variables = {
        'vk': 1,  # fixed second layer weights
        'K': 2,  # number of hidden units
        'p': 100,  # number of data points
        'n': 5,  # number of dimensions
        'alpha': 0.05,  # learning rate
        't_max': 40,  # maximum number of iterations
        'experiments': 50,  # number of experiments
        'adaptive': False  # adaptive learning rate or not

    }
    print(f'Performing base experiment')
    format_variables = '\n '.join([f'{key}={value}' for key, value in variables.items()])
    print(f'Variables: {format_variables}')
    run_assignment(variables)
    print(f'\nPerforming bonus 1: varying p')
    plot_error_vs_epochs_for_different_p()
    print(f'\nPerforming bonus 2: adaptive learning rate')
    variables['adaptive'] = True
    run_assignment(variables)


if __name__ == '__main__':
    main()
