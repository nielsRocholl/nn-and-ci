# Neural Networks and Computational Intelligence

This repository conatains a python implementation of the Rosenblatt Perceptron Algorithm, in addition to Backpropagation and Stochastic Gradient Descent for a soft committee machine. We only use Numpy for mathematical operations, and Numba for computational speed up. These are two seperate projects, this MD file will shortly cover them both.




# Rosenblatt Perceptron Algorithm

This project demonstrates the implementation of the Rosenblatt perceptron training algorithm in Python, using the Numpy library for mathematical operations. To enhance the efficiency of the code, the Numba package is utilized to dynamically compile Python functions into optimized machine code.

Our approach comprises three key parts: 

1. Artificial Data Generation
2. Perceptron Model (Training)
3. Calculation of the fraction of successful runs `Q_{l.s.}`

---

## The Number of Linearly Separable Dichotomies

The primary results generated for this experiment pertain to the fraction of successful runs, denoted by `Q_{l.s.}`. A successful run is one where the algorithm manages to converge (i.e., `E^{v} > 0` for all `v`) within a set number of epochs (100). This fraction relates to `P_{ls}` from the lecture notes (equation 3.42), which is the fraction of linearly separable dichotomies. 

The `P_{ls}` can be calculated theoretically, but our goal is to find it experimentally as `Q_{l.s.}`.

---

## Artificial Data Generation

A dataset `D` is created, containing `P` randomly generated feature vectors `ξ`, each having a dimensionality of `N`. The components of individual feature vectors are drawn from a Gaussian distribution with zero mean and unit variance. Each feature vector is assigned a binary label `S`, which has an equal probability of being either -1 or 1.

```python
def generate_artificial_data(N, P):
    feature_vectors = [numpy.random.normal(0, 1, N) for _ in range(P)]
    labels = [numpy.choice([-1, 1]) for _ in range(P)]
    return feature_vectors, labels
```

## Perceptron Model
The Rosenblatt perceptron training algorithm is a crucial component of this project. Before training, all weights are initialized to zero. The perceptron is then trained over `N_{max}` sweeps, iterating over all samples `ξ^{\mu}` in the dataset.

For each sample, the local potential `E^{\mu(t)}` is calculated. The weights are then updated according to the local potential. The training process ends when the algorithm converges `(E^{v} > 0` for all v in the dataset) or when the maximum number of sweeps `N_{max}` is completed.

## Results:

![Performance](Rosenblatt%20Perceptron%20Algorithm/plots/experiment_Large%20Experiment.png)

---
---


# Backpropagation and Stochastic Gradient Descent for a Soft Committee Machine

In this project, we consider a network where the activation functions `g` and `h` are represented by the following equations:

```math
g(x) = tanh(x)
h(x) = x
```

The hidden-to-output weights v_k are fixed at a value of 1 to mimic a soft Committee Machine. Given these settings, the student network equation becomes:

```math
\sigma(\xi) = \sum_{k=1}^K tanh(w_k \cdot \xi)
```

We've implemented the stochastic gradient descent (SGD) algorithm using Python and the Numpy library for mathematical operations. The problem was divided into three main components: error calculation, gradient calculation, and training the network.

## Error Calculation

The error is calculated using the cost function below:

```math
E = \frac{1}{P} \sum_{\mu = 1}^P \frac{1}{2} (\sigma(\xi^\mu) - \tau(\xi^\mu))^2 = \frac{1}{P} \sum_{\mu = 1}^P e^\mu
```

This function represents the quadratic deviation between the desired output labels and the produced output labels. The function contribution calculates the contribution to the cost function of the `$\mu$th` sample presented to the network. This is then used in the function cost_function to calculate the total error.

The generalization error, which represents the quadratic deviation from the target function for Q test examples, is defined as:

```math
E_{gen} = \frac{1}{Q} \sum_{\rho = P + 1}^{P + Q} \frac{1}{2} (\sigma(\xi^\rho) - \tau(\xi^\rho))^2
```

This is implemented in the function generalization_error.

## Gradient Calculation
For the gradient calculation, we first calculate the derivatives of the activation functions g and h as follows:

```math
g'(x) = 1 - tanh^2(x)
h'(x) = 1
```

We then define a delta value:

```math
\delta = \sigma - \tau
```

The gradient with respect to the mth weight vector is then calculated as:

```math
\nabla_{w_m} e = \delta \: v_m \: (1 - tanh^2(w_m \cdot \xi)) \: \xi
```

This is implemented in the function gradient_of_single_input_to_hidden_weight.

Training using Stochastic Gradient Descent
With the quadratic cost function and gradient calculations in place, we can now use these equations in the SGD algorithm. We initialize the weights w as independent random vectors such that `| w |^2 = 1`. We define a variable `t_max` indicating the number of epochs. The total number of iterations that the algorithm should run is then defined by t_max * P where P are the number of input samples. All steps of the training algorithm are brought together in the function train. This function accepts a dictionary of hyperparameters and then performs the experiment accordingly.
The function `create_input_to_hidden_weights` is used to initialize the weights. We also set a maximum number of epochs (`t_max`). The total number of iterations is defined by the product of `t_max` and the number of input samples `P`.

The main function `train` incorporates all the steps of the training algorithm. It takes in a dictionary of hyperparameters and performs the experiment based on these values. The learning rate can be static or adaptive and should be specified in the hyperparameter settings.

The hyperparameters used in this project are listed in the table below:

| Hyperparameter | Description |
| --- | --- |
| Learning rate (`η`) | The step size at each iteration while moving toward a minimum of a loss function. |
| Epochs (`t_max`) | One Epoch is when an entire dataset is passed forward and backward through the neural network only once. |
| Weights (`w`) | The strength of the different connections in the artificial neural network. |

**Note:** The hyperparameters are subject to change based on the specific requirements of your project or experiment.

By implementing the error calculation, gradient calculation, and the training process of the stochastic gradient descent algorithm, we can train the network and make predictions based on the input data.

