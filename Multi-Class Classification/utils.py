import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op


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


def display_data(data, width=None):
    m, n = data.shape
    if width is None:
        width = np.round(np.sqrt(n))

    width = int(width)
    height = int(n / width)
    rows = int(np.floor(np.sqrt(m)))
    cols = int(np.ceil(m / rows))

    pad = 1
    # set up blank display
    array = -np.ones((pad + int(rows * (height + pad)), pad + int(cols * (width + pad))), order='F')
    current = 0
    for j in range(rows):
        for i in range(cols):
            if current >= m:
                break
            max_val = np.max(np.abs(data[current, :]))
            row_start = pad + j * (height + pad)
            row_end = pad + j * (height + pad) + height
            col_start = pad + i * (width + pad)
            col_end = pad + i * (width + pad) + width

            array[row_start:row_end, col_start:col_end] = np.reshape(
                data[current, :], (height, width), order='F') / max_val
            current += 1
        if current >= m:
            break
    plt.imshow(array, cmap='gray')
    plt.axis('off')
