import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from utils import predict_nn, display_data, sigmoid, gradient_nn
from utils import sigmoid_gradient, cost_function_nn, rand_initialize_weights

data = loadmat("data/ex4data1.mat")
print(data.keys())

X = data['X']
m, n = X.shape
print(f"We have {m} hand written digits examples")
y = data['y']

# Randomly select data points to display
indices = np.random.permutation(m)
data_points = X[indices[0:100], :]

for i, da in enumerate(data_points):
    sp = plt.subplot(10, 10, i + 1, facecolor='red')
    sp.axis('Off')
    data_points_ = np.reshape(da, (20, 20), order='F')
    plt.imshow(data_points_, cmap='gray', interpolation='bicubic')

display_data(data_points)

weights = loadmat("data/ex4weights.mat")
theta1 = weights['Theta1']
print(f'theta1 shape: {theta1.shape}')
theta2 = weights['Theta2']
print(f'theta2 shape: {theta2.shape}')

# Unroll parameters
theta1 = np.reshape(theta1, (-1, 1), order='F')
theta2 = np.reshape(theta2, (-1, 1), order='F')
params = np.vstack((theta1, theta2))
print(params.shape)

"""
# Test cost and gradient
lambd = 0
input_layer_size = 400
hidden_layer_size = 25
num_labels = 10
J = cost_function_nn(params, input_layer_size,
                     hidden_layer_size, num_labels,
                     X, y, lambd)
grad = gradient_nn(params, input_layer_size,
                   hidden_layer_size, num_labels,
                   X, y, lambd)
"""


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
    y = 1 + np.mod(range(1, m + 1), num_labels).T
    y = y.reshape(-1, 1)
    # Unroll parameters
    t1 = theta1.flatten(order='F').reshape(-1, 1)
    t2 = theta2.flatten(order='F').reshape(-1, 1)
    nn_params = np.vstack((t1, t2))

    grad = gradient_nn(nn_params, input_layer_size,
                       hidden_layer_size, num_labels, X, y, lambd)

    numgrad = np.zeros(nn_params.shape)
    perturb = np.zeros(nn_params.shape)
    e = 1e-4

    for p in range(nn_params.size):
        # Set perturbation vector
        perturb[p] = e
        loss1 = cost_function_nn(
            nn_params - perturb, input_layer_size, hidden_layer_size, num_labels, X, y, lambd)
        loss2 = cost_function_nn(
            nn_params + perturb, input_layer_size, hidden_layer_size, num_labels, X, y, lambd)

        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2 * e)
        perturb[p] = 0

    result = np.hstack((numgrad.reshape(-1, 1), grad.reshape(-1, 1)))
    print(result)
    print("The two columns should be very similar")

    diff = np.linalg.norm(numgrad - grad, keepdims=True) / \
        np.linalg.norm(numgrad + grad, keepdims=True)

    print(f"Diff must be less than 1e-9\n Relative Difference: {diff}")


check_nn_gradients(lambd=3)
