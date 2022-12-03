import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_artificial_data(p=1000, n=1) -> list:
    """
    Generate artificial data sets containing P randomly generated N-dimensional feature vectors and binary labels.
    :param p: number of data points
    :param n: number of dimensions
    :return: list of tuples (feature vector, label)
    """

    def get_random_component():
        # create a vector of n independent random components, with mean 0 and variance 1
        return np.random.normal(0, 1, n)

    def get_random_label():
        # create a random label with 50% chance of being 1 and 50% chance of being -1
        return np.random.choice([1, -1])

    # create n-dimensional feature vectors
    feature_vectors = np.array([get_random_component() for _ in range(p)])

    # create binary labels
    labels = np.array([get_random_label() for _ in range(p)])

    # combine feature vectors and labels into a single data set
    return [(feature_vectors[i], labels[i]) for i in range(p)]


def calculate_local_potential(weight_vector: np.array, feature_vector: np.array, label: float) -> float:
    """
    Calculate the local potential of a perceptron.
    :param weight_vector: weight vector
    :param feature_vector: feature vector
    :param label: label
    :return: local potential
    """
    return np.dot(weight_vector, feature_vector) * label


def update_weight_vector(weight_vector, feature_vector, label, n, local_potential):
    """
    Update the weight vector.
    :param weight_vector: weight vector
    :param feature_vector: feature vector
    :param label: label
    :param n: learning rate
    :param local_potential: local potential
    :return: updated weight vector
    """
    if local_potential <= 0:
        return weight_vector + (1 / n) * feature_vector * label
    else:
        return weight_vector


def train_perceptron(data: list, weights: np.array, epochs: int, n: int) -> np.array:
    """
    Train a perceptron on a given data set.
    :param data: data set
    :param weights: weight vector
    :param epochs: maximum number of epochs
    :param n: size of the data set
    :return: weights, bias, number of iterations
    """

    # sweep over data
    for e in range(epochs):
        local_potentials = []
        # iterate over samples in dataset (1 to P)
        for feature_vector, label in data:
            # calculate local potential
            local_potentials.append(calculate_local_potential(weights, feature_vector, label))
            # update weights
            weights = update_weight_vector(weights, feature_vector, label, n=n, local_potential=local_potentials[-1])
        # check if all local potentials are positive
        if all(local_potential > 0 for local_potential in local_potentials):
            # converged
            return weights, True
    # did not converge
    return weights, False


def fraction_of_successful_runs(alpha, n, nD, nmax):
    """
    Calculate the fraction of successful runs for a given set of hyperparameters.
    :param alpha: α
    :param n: total number of samples
    :param nD: number of data sets
    :param nmax: maximum number of epochs
    :return:
    """
    converged = 0
    p = int(alpha * n)
    for d in range(nD):
        # create artificial data set
        artificial_data = generate_artificial_data(p=p, n=n)
        # initialise weights
        weights = np.zeros(len(artificial_data[0][0]))
        # train perceptron
        weights, success = train_perceptron(data=artificial_data,
                                            weights=weights,
                                            epochs=nmax,
                                            n=n
                                            )
        converged = converged + 1 if success else converged
    return converged / nD


def plot_fractions_of_successful_runs(hyper_params, results):
    """
    Plot the fraction of successful runs for each combination of hyperparameters.
    :param hyper_params:
    :param results:
    :return:
    """
    fig, ax = plt.subplots()
    for n in hyper_params['N']:
        ax.plot(results[results['N'] == n]['α'], results[results['N'] == n]['fraction_of_successful_runs'],
                label=f'N={n}')
    ax.set_xlabel('α')
    ax.set_ylabel('fraction of successful runs')
    ax.legend()
    plt.show()


def main():
    # initialise hyperparameters based on the problem statement:
    hyper_params = {
        'N': np.array([20, 40]),
        'α': np.arange(0.75, 3, 0.25),
        'nD': np.array([50]),
        'nmax': np.array([100])
    }
    # generate all combinations of hyperparameters
    keys, values = zip(*hyper_params.items())
    hyper_param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    results = pd.DataFrame(columns=['N', 'α', 'nD', 'nmax', 'fraction_of_successful_runs'])
    # perform training for each combination of hyperparameters
    for n in hyper_params['N']:
        for alpha in hyper_params['α']:
            for nD in hyper_params['nD']:
                for nmax in hyper_params['nmax']:
                    # use pd. concat to append a new row to the results dataframe
                    results = pd.concat([results, pd.DataFrame({
                        'N': [n],
                        'α': [alpha],
                        'nD': [nD],
                        'nmax': [nmax],
                        'fraction_of_successful_runs': [fraction_of_successful_runs(alpha=alpha, n=n, nD=nD, nmax=nmax)]
                    })])

    # plot convergence as a using matplotlib
    plot_fractions_of_successful_runs(hyper_params=hyper_params, results=results)


if __name__ == '__main__':
    main()
