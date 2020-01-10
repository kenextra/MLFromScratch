from __future__ import print_function
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from utils import plot_data, feature_normalize, predict, train
from mpl_toolkits import mplot3d

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", required=True,
                help="Path to training data")
ap.add_argument("-t", "--test_file", help="Path to test data")
ap.add_argument("-n", "--normalize", default=True,
                help="To normalize data")
args = vars(ap.parse_args())
# LOAD DATA
# data = pd.read_csv('ex1data1.txt', header=None)
data = pd.read_csv(args['file'], header=None)

# Convert pandas dataframe to numpy array
numpy_data = data.values
n_row, n_col = numpy_data.shape

X = numpy_data[:, 0:n_col - 1].reshape(n_row, n_col - 1)
Y = numpy_data[:, n_col - 1].reshape(n_row, 1)

m = len(X)

print('Number of training examples: ', m)
print('Shape of Features: ', X.shape)
print('Number of features: ', X.shape[1])

# Plot the data in 2D or 3D
if X.shape[1] == 1:
    fig = plt.figure(num=1, figsize=(10, 6))
    plot_data(X, Y)
elif X.shape[1] == 2:
    fig = plt.figure(num=1, figsize=(10, 6))
    # ax = plt.axes(projection="3d")
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter3D(X[:, 0], X[:, 1], Y, cmap='hsv')

# initilaize parameters
initial_theta = np.zeros((X.shape[1] + 1, 1))
alpha = 0.01
num_iters = 1500

theta, J_history, mu, sigma, normalize = train(X, Y, initial_theta, alpha,
                                               num_iters, normalize=args['normalize'])
print('theta: ', theta)
print('mu: ', mu)
print('sigma: ', sigma)

# Use learned parameters to make predictions
if X.shape[1] == 1:
    test = np.array([[7]])
    result = predict(test, theta, mu, sigma) * 10000
    print(result)

if X.shape[1] == 2:
    test2 = np.array([[1650, 3]])
    result2 = predict(test2, theta, mu, sigma)
    print(result2)

fig = plt.figure(num=2, figsize=(10, 6))
if normalize:
    X, _, _ = feature_normalize(X)
    plt.title('Population is normalized')
else:
    plt.title('Population is not normalized')

if X.shape[1] == 1:
    plt.plot(X, np.dot(np.insert(X, 0, 1, axis=1), theta), zorder=1)
    plot_data(X, Y)

fig = plt.figure(num=3, figsize=(10, 6))
plt.plot([i for i in range(1, len(J_history) + 1)], J_history)
plt.show()
