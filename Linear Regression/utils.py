import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
from mpl_toolkits import mplot3d


def plot_data(x, y):
    # plt.figure()
    plt.scatter(x, y.flatten(), marker='x', c='r', zorder=2)
    plt.xlabel('Population of city in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.title('Population vs Profit')
    plt.show()


def plot_data_3d(X, Y, Z):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter3D(X, Y, Z, cmap='hsv')


def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


def compute_cost(X, Y, theta):
    m = len(Y)
    error = np.dot(X, theta) - Y
    const = 2 * m
    J = (1 / const) * (np.dot(error.T, error))
    return J


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

# Leguralized Linear Regression utils


def linearReg_cost_function(theta, X, y, lambd=0):
    m, n = X.shape
    y = y.reshape(m, 1)
    assert(m == y.shape[0])
    theta = theta.reshape(n, 1)
    theta_above = theta[1:, :]
    h = X.dot(theta)
    error = h - y
    lambd_mul = lambd / (2 * m)

    J = (error.T.dot(error)) / (2 * m) + (lambd_mul * theta_above.T.dot(theta_above))
    return J.flatten()[0]


def linearReg_gradient(theta, X, y, lambd=0):
    m, n = X.shape
    y = y.reshape(m, 1)
    assert(m == y.shape[0])
    theta = theta.reshape(n, 1)
    grad = np.zeros(theta.shape)

    theta_above = theta[1:, :]
    h = X.dot(theta)

    grad[:, 0] = (1 / m) * (X[:, 0].T.dot(h - y))
    grad[1:, :] = (1 / m) * (X[:, 1:].T.dot(h - y)) + ((lambd / m) * theta_above)
    return np.reshape(grad, (-1, 1), order='F')


def train_linearReg(X, y, lambd):
    # X = np.insert(X, 0, 1, axis=1)
    m, n = X.shape
    initial_theta = np.zeros(n)
    options = {'maxiter': 200, 'disp': False}
    result = op.minimize(fun=linearReg_cost_function,
                         x0=initial_theta,
                         args=(X, y, lambd),
                         method='TNC',
                         jac=linearReg_gradient,
                         options=options)
    theta = result.x
    return theta


def learning_curve(X, y, Xval, yval, lambd):
    m, n = X.shape
    train_error = np.zeros(m)
    val_error = np.zeros(m)

    # Compute train/cross validation errors using training examples
    for i in range(m):
        X_train = X[0:i + 1, :].reshape(i + 1, -1)
        y_train = y[0:i + 1, :].reshape(-1, 1)
        theta = train_linearReg(X_train, y_train, lambd)
        train_error[i] = linearReg_cost_function(theta, X_train, y_train, lambd)
        val_error[i] = linearReg_cost_function(theta, Xval, yval, lambd)

    return train_error, val_error


def poly_features(X, p):
    X_poly = np.zeros((X.size, p))
    for i in range(p):
        X_poly[:, i] = (X**(i + 1)).T
    return X_poly


def validation_curve(X, y, Xval, yval):
    lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    train_err = np.zeros(len(lambda_vec))
    val_err = np.zeros(len(lambda_vec))

    for i, lambd in enumerate(lambda_vec):
        theta = train_linearReg(X, y, lambd)
        train_err[i] = linearReg_cost_function(theta, X, y, lambd)
        val_err[i] = linearReg_cost_function(theta, Xval, yval, lambd)
    return lambda_vec, train_err, val_err
