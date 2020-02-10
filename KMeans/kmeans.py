import numpy as np
from scipy.io import loadmat


def find_closest_centroids(X, centroids):
    K = centroids.shape[0]
    m = X.shape[0]
    idx = np.zeros(m)

    for i in range(m):
        cidx = (np.linalg.norm(X[i, :] - centroids, axis=1))**2
        idx[i] = np.argmin(cidx)
    return idx


data = loadmat('data/ex7data2.mat')

X = data['X']

K = 3
centroids = np.array([[3, 3], [6, 2], [8, 5]])

idx = find_closest_centroids(X, centroids)

print(idx[0:3])
