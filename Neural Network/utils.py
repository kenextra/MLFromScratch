import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import time
from sklearn.preprocessing import OneHotEncoder


def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g


def sigmoid_gradient(z):
    g = sigmoid(z) * (1 - sigmoid(z))
    return g.reshape(1, -1)


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

    mul = -1 / m

    y_logh = np.trace(y_matrix.T.dot(np.log(all_h)))
    y_minus_logh = np.trace((1 - y_matrix).T.dot(np.log(1 - all_h)))

    theta_1 = theta1[:, 1:]
    theta_1 = np.trace(theta_1.T.dot(theta_1))
    theta_2 = theta2[:, 1:]
    theta_2 = np.trace(theta_2.T.dot(theta_2))

    all_logh = y_logh + y_minus_logh

    J = mul * all_logh + (lambd / (2 * m) * (theta_1 + theta_2))

    return J.flatten()[0]


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
    theta1_grad[:, 0] = (1 / m) * delta1[:, 0].T
    theta1_grad[:, 1:] = 1 / m * (delta1[:, 1:] + lambd * theta1[:, 1:])

    # Do not regularize the first column
    theta2_grad[:, 0] = 1 / m * delta2[:, 0].T
    theta2_grad[:, 1:] = 1 / m * (delta2[:, 1:] + lambd * theta2[:, 1:])

    # Unroll gradients
    grad1 = np.reshape(theta1_grad, (-1, 1), order='F')
    grad2 = np.reshape(theta2_grad, (-1, 1), order='F')
    grad = np.vstack((grad1, grad2))

    return grad


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


def debug_initialize_weights(fanout, fanin):
    W = np.zeros((fanout, 1 + fanin))
    W = np.reshape(np.sin(range(1, W.size + 1)), W.shape, order='F') / 10
    return W


def check_nn_gradients(lambd=0):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # Generate some random test
    theta1 = debug_initialize_weights(hidden_layer_size, input_layer_size)
    theta2 = debug_initialize_weights(num_labels, hidden_layer_size)

    # Reusing debdebug_initialize_weights to generate X
    X = debug_initialize_weights(m, input_layer_size - 1)
    y = 1 + np.mod(range(1, m + 1), num_labels)
    y = np.reshape(y, (-1, 1), order='F')
    # Unroll parameters
    t1 = theta1.flatten(order='F').reshape(-1, 1)
    t2 = theta2.flatten(order='F').reshape(-1, 1)
    nn_params = np.vstack((t1, t2))

    grad = gradient_nn(nn_params, X, y, input_layer_size, hidden_layer_size, num_labels, lambd)

    numgrad = np.zeros(nn_params.shape)
    perturb = np.zeros(nn_params.shape)
    e = 1e-4

    for p in range(nn_params.size):
        # Set perturbation vector
        perturb[p] = e
        loss1 = cost_function_nn(nn_params - perturb, X, y, input_layer_size, hidden_layer_size, num_labels, lambd)
        loss2 = cost_function_nn(nn_params + perturb, X, y, input_layer_size, hidden_layer_size, num_labels, lambd)

        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2 * e)
        perturb[p] = 0

    result = np.hstack((numgrad.reshape(-1, 1), grad.reshape(-1, 1)))
    print(result)
    print("The two columns should be very similar")

    diff = np.linalg.norm(numgrad - grad, 2) / np.linalg.norm(numgrad + grad, 2)

    print(f"Diff must be less than 1e-9\nRelative Difference: {diff}")


def train(X, y, initial_nn_params, input_layer_size, hidden_layer_size, num_labels, options, lambd=0):
    start_time = time.perf_counter()

    result = op.minimize(fun=cost_function_nn,
                         x0=initial_nn_params,
                         args=(X, y, input_layer_size, hidden_layer_size, num_labels, lambd),
                         method='TNC',
                         jac=gradient_nn,
                         options=options)
    print(f"It took {time.perf_counter() - start_time:0.2f} seconds to train the Network")

    return result.x
