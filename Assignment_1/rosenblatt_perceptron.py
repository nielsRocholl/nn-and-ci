import numpy as np


def generate_artificial_data(p=1000, n=1) -> list:
    """
    Generate artificial data sets containing P randomly generated N-dimensional feature vectors and binary labels.
    :param p: number of data points
    :param n: number of dimensions
    :return:
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


def train_perceptron(data: list, epochs=100) -> tuple:
    """
    Train a perceptron on a given data set.
    :param data: data set
    :param epochs: maximum number of epochs
    :return: weights, bias, number of iterations
    """
    # initialize weights and bias
    weights = np.zeros(len(data[0][0]))
    bias = 0

    # sweep over data
    for e in range(epochs):
        # iterate over samples in dataset (1 to P)
        for feature_vector, label in data:
            pass

    return weights, bias, e


def main():
    # initialise hyperparameters based on the problem statement:
    hyper_params = {
        'N': np.array([20, 40]),
        'α': np.arange(0.75, 3, 0.25),
        'nD': np.array([50]),
        'nmax': np.array([100])
    }
    # create artificial data set
    data = generate_artificial_data(p=10, n=2)
    # train perceptron
    weights, bias, i = train_perceptron(data, learning_rate=0.1, epochs=1000)
    print(f'weights: {weights}')


if __name__ == '__main__':
    main()
