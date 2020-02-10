import numpy as np
from math import pi
import matplotlib.pyplot as plt


def estimate_gaussian(X):
    m, n = X.shape
    mu = np.mean(X, axis=0)
    mu = np.reshape(mu, (-1, 1), order='F')
    x = X - mu.T
    sigma = np.diag(1 / m * (x.T.dot(x)))
    sigma = np.reshape(sigma, (-1, 1), order='F')
    return mu, sigma


def multivariate_gaussian(X, mu, Sigma):
    m, n = Sigma.shape
    k = len(mu)
    if n == 1 or m == 1:
        Sigma = np.diag(Sigma.flatten())

    X = X - mu.flatten()
    const = (2 * pi)**(-k / 2)
    det = np.linalg.det(Sigma) ** (-0.5)
    pinv = np.linalg.pinv(Sigma)
    sumb = np.sum(X.dot(pinv) * X, axis=1)
    exponen = np.exp(-0.5 * sumb)
    p = const * det * exponen
    return np.reshape(p, (-1, 1), order='F')


def visualize_fit(X, mu, sigma):
    plt.figure(figsize=(12, 8))
    pts, step = np.linspace(0, 35, num=71, retstep=True)
    X1, X2 = np.meshgrid(pts, pts)
    x1 = np.reshape(X1, (-1, 1), order='F')
    x2 = np.reshape(X2, (-1, 1), order='F')
    XS = np.hstack((x1, x2))
    Z = multivariate_gaussian(XS, mu, sigma)
    Z = np.reshape(Z, X1.shape, order='F')
    plt.scatter(X[:, 0], X[:, 1], c='b', marker='x')
    if np.sum(np.isinf(Z)) == 0:
        levels = [10**level for level in range(-20, 0, 3)]
        plt.contour(X1, X2, Z, levels)
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    plt.title('Network server statistics')


def select_threshold(yval, pval):
    bestEpsilon = 0
    bestF1 = 0
    start = min(pval)
    stop = max(pval)
    step = (stop - start) / 1000
    for epsilon in np.arange(start, stop, step):
        p = pval < epsilon
        fp = np.sum((p == 1) & (yval == 0)).astype(float)
        tp = np.sum((p == 1) & (yval == 1)).astype(float)
        fn = np.sum((p == 0) & (yval == 1)).astype(float)
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        F1 = (2 * prec * rec) / (prec + rec)

        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon
    return bestEpsilon, bestF1
