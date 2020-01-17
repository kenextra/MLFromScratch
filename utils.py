import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
from sklearn.preprocessing import OneHotEncoder
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


def sigmoid_gradient(z):
    g = sigmoid(z) * (1 - sigmoid(z))
    return g.reshape(1, -1)


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


def cost_function_nn(nn_params, X, y, input_layer_size, hidden_layer_size, num_labels, lambd=0):
    # Reshape nn_params into theta1 and theta2
    theta1 = np.reshape(nn_params[0: hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, input_layer_size + 1), order='F')
    theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                        (num_labels, hidden_layer_size + 1), order='F')

    m, _ = X.shape

    # Feedforward pass
    a1 = np.insert(X, 0, 1, axis=1)  # add bias
    z2 = sigmoid(a1.dot(theta1.T))
    a2 = np.insert(z2, 0, 1, axis=1)  # add bias
    z3 = a2.dot(theta2.T)

    all_h = sigmoid(z3)
    # eye_matrix = np.eye(num_labels, order='F')
    # y_matrix = eye_matrix[y, :]

    a = OneHotEncoder(categories='auto', dtype='int32')
    y_matrix = a.fit_transform(y).todense()

    mul = -1/m

    y_logh = np.trace(y_matrix.T.dot(np.log(all_h)))
    y_minus_logh = np.trace((1-y_matrix).T.dot(np.log(1-all_h)))

    theta_1 = theta1[:, 1:]
    theta_1 = np.trace(theta_1.T.dot(theta_1))
    theta_2 = theta2[:, 1:]
    theta_2 = np.trace(theta_2.T.dot(theta_2))

    all_logh = y_logh + y_minus_logh

    J = mul * all_logh + (lambd/(2*m) * (theta_1 + theta_2))

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


def gradient_nn(nn_params, X, y, input_layer_size, hidden_layer_size, num_labels, lambd=0):
    # Reshape nn_params into theta1 and theta2
    theta1 = np.reshape(nn_params[0: hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, input_layer_size + 1), order='F')
    theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                        (num_labels, hidden_layer_size + 1), order='F')

    m, _ = X.shape

    theta1_grad = np.zeros(theta1.shape)
    theta2_grad = np.zeros(theta2.shape)

    # Backpropagation Algorithm
    delta1 = np.zeros(theta1.shape)
    delta2 = np.zeros(theta2.shape)

    a = OneHotEncoder(categories='auto', dtype='int32')
    y_matrix = a.fit_transform(y).todense()

    for w in range(m):
        x = X[w, :].reshape(1, -1)
        a1 = np.insert(x, 0, 1, axis=1)
        z2 = np.dot(a1, theta1.T)
        a2 = np.insert(sigmoid(z2), 0, 1, axis=1)
        z3 = np.dot(a2, theta2.T)
        a3 = sigmoid(z3)
        d3 = a3 - y_matrix[w, :].reshape(1, -1)
        z_2 = np.insert(z2, 0, 1, axis=1)
        d2 = np.dot(d3, (theta2 * sigmoid_gradient(z_2)))
        delta1 = delta1 + np.dot(d2[:, 1:].T, a1)
        delta2 = delta2 + np.dot(d3.T, a2)

    # Regularization with the cost function and gradients
    # Do not regularize the first column
    theta1_grad[:, 0] = (1/m) * delta1[:, 0].T
    theta1_grad[:, 1:] = 1/m * (delta1[:, 1:] + lambd * theta1[:, 1:])

    # Do not regularize the first column
    theta2_grad[:, 0] = 1/m * delta2[:, 0].T
    theta2_grad[:, 1:] = 1/m * (delta2[:, 1:] + lambd * theta2[:, 1:])

    # Unroll gradients
    grad1 = np.reshape(theta1_grad, (-1, 1), order='F')
    grad2 = np.reshape(theta2_grad, (-1, 1), order='F')
    grad = np.vstack((grad1, grad2))

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
    array = -np.ones((pad + int(rows*(height+pad)), pad + int(cols *(width+pad))), order='F')
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
    plt.show()


def predict_nn(Theta1, Theta2, X):
    m, _ = X.shape

    z2 = np.insert(X, 0, 1, axis=1).dot(Theta1.T)  # add bias
    a2 = sigmoid(z2)
    z3 = np.insert(a2, 0, 1, axis=1).dot(Theta2.T)  # add bias

    pred = sigmoid(z3)

    p = np.argmax(pred, axis=1).reshape(-1, 1)

    return p + 1


def rand_initialize_weights(lin, lout):
    init_epsilon = 0.12
    w = np.random.rand(lout, 1 + lin) * 2 * init_epsilon - init_epsilon
    w = np.reshape(w, (lout, 1 + lin), order='F')

    return w
