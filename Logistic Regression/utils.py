import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
# Plot Data


def plot_data(features, labels, ax, label=None, legend=False):
    labels = labels.flatten()
    pos = np.where(labels == 1)
    neg = np.where(labels == 0)

    ax.scatter(features[pos, 0], features[pos, 1], c='b', marker='o', label=label[0])
    ax.scatter(features[neg, 0], features[neg, 1], c='r', marker='x', label=label[1])

    if legend:
        ax.legend(loc=0, frameon=True, shadow=True)


def map_feature(x1, x2):
    """
     MAPFEATURE Feature mapping function to polynomial features

     MAPFEATURE(X1, X2) maps the two input features
     to quadratic features used in the regularization exercise.

     Returns a new feature array with more features, comprising of
     X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..

     Inputs X1, X2 must be the same size
    """
    x1.shape = (x1.size, 1)
    x2.shape = (x2.size, 1)
    degree = 6
    out = np.ones(x1.shape)
    for i in range(1, degree + 1):
        for j in range(1 + i):
            r = (x1 ** (i - j)) * (x2 ** j)
            out = np.append(out, r, axis=1)
    return out


def plot_decision_boundary(theta, X, y, ax, label):
    plot_data(X[:, 1:3], y, ax, label[0:2])
    if X.shape[1] <= 3:
        plot_x = np.array([min(X[:, 1]) - 2, max(X[:, 1]) + 2])
        plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])
        ax.plot(plot_x, plot_y, label=label[2], color='green')
        ax.legend(loc=3, frameon=True, shadow=True)
    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros(shape=(len(u), len(v)))
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = (map_feature(np.array(u[i]), np.array(v[j])).dot(np.array(theta)))
        z = z.T  # important to transpose z before calling contour

        ax.contour(u, v, z, 0, linewidths=2)


def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g


def train(X, Y, initial_theta, options, lambd=0):
    result = op.minimize(fun=cost_function,
                         x0=initial_theta,
                         args=(X, Y, lambd),
                         method='TNC',
                         jac=gradient,
                         options=options)
    return result.x, result.fun


def predict(theta, X):
    m = X.shape[0]
    p = np.zeros((m, 1))

    for i in range(m):
        if sigmoid(X[i, :].dot(theta)) >= 0.5:
            p[i] = 1
        if sigmoid(X[i, :].dot(theta)) < 0.5:
            p[i] = 0
    return p


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
