import numpy as np


def generate_artificial_data(p=1000, n=1):
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


def main():
    # create artificial data set
    data = generate_artificial_data(p=10, n=2)
    print(data)


if __name__ == '__main__':
    main()
