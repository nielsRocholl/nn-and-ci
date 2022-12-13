import time
# use direct imports to speed up the code
from numpy import arange, array, dot, zeros
from numpy.random import choice, normal
from pandas import DataFrame, concat
from matplotlib.pyplot import show, subplots
# use numba for speed up
from numba import jit


@jit(nopython=True)
def generate_artificial_data(p, n) -> tuple:
    """
    Generate artificial data sets containing P randomly generated N-dimensional feature vectors and binary labels.
    Use numba to speed up the code. This requires the use of numba accepted operations.
    :param p: number of data points
    :param n: number of dimensions
    :return:
    """
    feature_vectors = zeros((p, n))
    labels = zeros(p)
    classes = array([-1, 1])
    for i in range(p):
        # create a vector of n independent random components, with mean 0 and variance 1
        feature_vectors[i] = normal(0, 1, n)
        # create a random label with 50% chance of being 1 and 50% chance of being -1
        labels[i] = choice(classes)
    return feature_vectors, labels


@jit(nopython=True)
def calculate_local_potential(weight_vector: array, feature_vector: array, label: float) -> float:
    """
    Calculate the local potential of a perceptron.
    :param weight_vector: weight vector
    :param feature_vector: feature vector
    :param label: label
    :return: local potential
    """
    return dot(weight_vector, feature_vector) * label


@jit(nopython=True)
def update_weight_vector(weight_vector, feature_vector, label, n, local_potential) -> array:
    """
    Update the weight vector.
    :param weight_vector: weight vector
    :param feature_vector: feature vector
    :param label: label
    :param n:
    :param local_potential: local potential
    :return: updated weight vector
    """
    if local_potential <= 0:
        return weight_vector + (1 / n) * feature_vector * label
    else:
        return weight_vector


@jit(nopython=True)
def train_perceptron(feature_vectors: array, labels: array, weights: array, epochs: int, n: int, n_max: int) -> array:
    """
    Train a perceptron on a given data set.
    :param feature_vectors: feature vectors
    :param labels: labels
    :param weights: weight vector
    :param epochs: maximum number of epochs
    :param n: size of the data set
    :param n_max: maximum number of epochs
    :return: weights, bias, number of iterations
    """

    # sweep over data and show progress bar
    for e in range(epochs):
        # limit the number of sweeps to n_max
        if e >= n_max:
            return weights, False
        converged = True
        # iterate over samples in dataset (1 to P)
        for i in range(len(feature_vectors)):
            # extract feature vector and label
            feature_vector = feature_vectors[i]
            label = labels[i]
            # calculate local potential
            local_potential = calculate_local_potential(weights, feature_vector, label)
            # if the potential is smaller than 0, not converged
            converged = False if local_potential <= 0 else converged
            # update weights
            weights = update_weight_vector(weights, feature_vector, label, n=n, local_potential=local_potential)
        # if all local potentials are positive, return weights
        if converged:
            # converged
            return weights, True
    # did not converge
    return weights, False


@jit(nopython=True)
def fraction_of_successful_runs(alpha, n, n_d, n_max) -> float:
    """
    Calculate the fraction of successful runs for a given set of hyperparameters.
    :param alpha: α
    :param n: total number of samples
    :param n_d: number of data sets
    :param n_max: maximum number of epochs
    :return: fraction of successful runs
    """
    converged = 0
    p = int(alpha * n)
    for d in range(n_d):
        # create artificial data set
        feature_vectors, labels = generate_artificial_data(p, n)
        # initialise weights
        weights = zeros(n)
        # train perceptron
        weights, success = train_perceptron(feature_vectors=feature_vectors,
                                            labels=labels,
                                            weights=weights,
                                            epochs=n_max,
                                            n=n,
                                            n_max=n_max,
                                            )
        converged = converged + 1 if success else converged
    return converged / n_d


def plot_fractions_of_successful_runs(hyper_params, results) -> None:
    """
    Plot the fraction of successful runs for each combination of hyperparameters.
    :param hyper_params: dictionary of hyperparameters
    :param results: pandas dataframe containing results
    :return: None
    """
    fig, ax = subplots()
    for n in hyper_params['N']:
        ax.plot(results[results['N'] == n]['α'], results[results['N'] == n]['fraction_of_successful_runs'],
                label=f'N={n}')
    ax.set_xlabel('P/N', fontsize=14)
    ax.set_ylabel('Qls', fontsize=14)
    ax.set_title('Fraction of Successful Runs', fontsize=18)
    ax.grid()
    # set plot color to light grey
    ax.set_facecolor('#f0f0f0')

    ax.legend(prop={'size': 12})
    show()


def main():
    # initialise hyperparameters based on the problem statement:
    hyper_params = {
        'N': array([5]),                # dimensionality of the feature space
        'α': arange(0.5, 4.5, 0.25),    # fraction of samples
        'nD': array([10000]),           # number of independent data sets
        'n_max': array([100])           # maximum number of epochs
    }
    # pandas dataframe to store results
    results = DataFrame(columns=['N', 'α', 'nD', 'n_max', 'fraction_of_successful_runs'])
    # perform training for each combination of hyperparameters
    t1 = time.time()
    for n in hyper_params['N']:
        for alpha in hyper_params['α']:
            for nD in hyper_params['nD']:
                for n_max in hyper_params['n_max']:
                    results = concat([results, DataFrame({
                        'N': [n],
                        'α': [alpha],
                        'nD': [nD],
                        'n_max': [n_max],
                        'fraction_of_successful_runs': [
                            fraction_of_successful_runs(alpha=alpha, n=n, n_d=nD, n_max=n_max)]
                    })])
    t2 = time.time()
    print(f'Elapsed time: {t2 - t1} seconds')
    # plot convergence as a using matplotlib
    plot_fractions_of_successful_runs(hyper_params=hyper_params, results=results)


if __name__ == '__main__':
    main()
