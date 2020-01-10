import numpy as np
import matplotlib.pyplot as plt

# Plot Data


def plot_data(x, y):
    # plt.figure()
    plt.scatter(x, y.flatten(), marker='x', c='r', zorder=2)
    plt.xlabel('Population of city in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()

# Normalize features


def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

# Compute cost


def compute_cost(X, Y, theta):
    m = len(Y)
    error = np.dot(X, theta) - Y
    const = 2 * m
    J = (1 / const) * (np.dot(error.T, error))
    return J

# Gradient descent to update parameters


def gradient_descent(X, Y, theta, alpha, num_iters, normalize=True):

    m = len(Y)
    J_history = np.zeros((num_iters, 1))
    mu = 0
    sigma = 1

    if normalize:
        X, mu, sigma = feature_normalize(X)

    # add a column of ones to x
    X = np.insert(X, 0, 1, axis=1)

    for num in range(num_iters):
        h = np.dot(X, theta)
        error = h - Y
        delta = np.dot(X.T, error) / m
        theta = theta - alpha * delta

        J_history[num] = compute_cost(X, Y, theta)
        if num % 100 == 0:
            print("Iteration ", num, ": cost ", J_history[num])

    return theta, J_history, mu, sigma

# Predict other data


def predict(test, theta, mu, sigma):

    # normalize test data
    test = (test - mu) / sigma
    # add a column of ones to x
    test = np.insert(test, 0, 1, axis=1)

    return np.dot(test, theta)


def train(X, Y, theta, alpha, num_iters, normalize=False):
    alpha = alpha
    num_iters = num_iters

    theta, J_history, mu, sigma = gradient_descent(
        X, Y, theta, alpha, num_iters, normalize)

    return theta, J_history, mu, sigma, normalize
