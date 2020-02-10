import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
# from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
import re
import string
from sklearn.preprocessing import OneHotEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
# Plot Data


def pca(xnorm, n_components):
    m, _ = xnorm.shape

    Sigma = (1 / m) * (xnorm.T.dot(xnorm))
    U, S, _ = randomized_svd(Sigma, n_components=n_components, random_state=None)
    # U, S, _ = np.linalg.svd(Sigma, full_matrices=False)
    return U, S


def drawline(p1, p2, color_type='k-'):
    plt.plot(np.array([p1[0], p2[0]]), np.array([p1[1], p2[1]]), color_type, linewidth=2)


# Project the data onto K = 1 dimension
def project_data(x_norm, U, K):
    m, _ = x_norm.shape
    Z = np.zeros((m, K))
    for i in range(m):
        x = x_norm[i, :]
        Z[i, :] = x.dot(U[:, 0:K])
    return Z


def recover_data(Z, U, K):
    row = Z.shape[0]
    col = U.shape[0]

    X_rec = np.zeros((row, col))
    for i in range(row):
        v = Z[i, :]
        for j in range(col):
            X_rec[i, j] = v.dot(U[j, 0:K].T)
    return X_rec


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


def plot_data_points(X, idx, K):
    R = np.linspace(0, K, K + 1)
    norm = plt.Normalize(0, K + 1)
    palete = plt.cm.hsv(norm(R))
    colors = palete[idx, :]

    plt.scatter(X[:, 0], X[:, 1], facecolors='none', color=colors)

# Normalize features


def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma
