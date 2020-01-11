import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
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


def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g


def cost_function(theta, X, y, lambd=0):
    m, n = X.shape
    y = y.reshape(m, 1)
    assert(m == y.shape[0])
    theta = theta.reshape(n, 1)
    theta_above = theta[1:, :]
    theta_x = X.dot(theta)
    h = sigmoid(theta_x)
    mul = -1 / m
    lambd_mul = lambd / (2 * m)

    pos = np.dot(y.T, np.log(h))
    neg = np.dot((1 - y).T, np.log(1 - h))
    J = (mul * (pos + neg)) + (lambd_mul * theta_above.T.dot(theta_above))
    return J.flatten()[0]


def gradient(theta, X, y, lambd=0):
    m, n = X.shape
    y = y.reshape(m, 1)
    assert(m == y.shape[0])
    theta = theta.reshape(n, 1)
    grad = np.zeros(theta.shape)

    theta_above = theta[1:, :]
    theta_x = X.dot(theta)
    h = sigmoid(theta_x)

    grad[:, 0] = (1 / m) * (X[:, 0].T.dot(h - y))
    grad[1:, :] = (1 / m) * (X[:, 1:].T.dot(h - y)) + \
        ((lambd / m) * theta_above)
    return grad


def one_vs_all(X, y, num_labels, lambd):
    m, n = X.shape
    assert (m == y.shape[0])
    y = y.reshape(m, 1)

    all_theta = np.zeros((num_labels, n + 1))

    X = np.insert(X, 0, 1, axis=1)

    for c in range(1, num_labels + 1):

        initial_theta = np.zeros((n + 1))
        # y_val = y == c
        y_val = np.array([1 if label == c else 0 for label in y])
        y_val = np.reshape(y_val, (m, 1))

        options = {'maxiter': 100, 'disp': True}
        result = op.minimize(fun=cost_function,
                             x0=initial_theta,
                             args=(X, y_val, lambd),
                             method='TNC',
                             jac=gradient,
                             options=options)
        theta = result.x
        all_theta[c - 1, :] = theta

    return all_theta


def predict_ova(all_theta, X):
    X = np.insert(X, 0, 1, axis=1)
    m = X.shape[0]
    pred = sigmoid(X.dot(all_theta.T))
    p = np.argmax(pred, axis=1).reshape(m, 1)

    return p + 1  # added one because the label is from 1-10


def display_data(imgData):
    current = 0
    pad = 1
    display_array = -np.ones((pad + 10 * (20 + pad), pad + 10 * (20 + pad)))
    for i in range(10):
        for j in range(10):
            display_array[pad + i * (20 + pad):pad + i * (20 + pad) + 20, pad + j * (20 + pad):pad + j * (20 + pad) + 20] = (imgData[current, :].reshape(20, 20, order="F"))
            current += 1
    plt.imshow(display_array, cmap='gray')
    plt.axis('off')
    plt.show()
