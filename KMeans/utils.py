import numpy as np
import matplotlib.pyplot as plt


def find_closest_centroids(X, centroids):
    m = X.shape[0]
    idx = np.zeros(m, dtype=int)

    for i in range(m):
        cidx = (np.linalg.norm(X[i, :] - centroids, axis=1))**2
        idx[i] = np.argmin(cidx)
    return idx


def compute_centroids(X, idx, K):
    m, n = X.shape

    centroids = np.zeros((K, n))

    for k in range(K):
        l_index = idx == k
        x = X[l_index, :]
        centroids[k, :] = sum(x) / sum(l_index)

    return centroids


def plot_data_points(X, idx, K):
    R = np.linspace(0, K, K + 1)
    norm = plt.Normalize(0, K + 1)
    palete = plt.cm.hsv(norm(R))
    colors = palete[idx, :]

    plt.scatter(X[:, 0], X[:, 1], facecolors='none', color=colors)


def draw_line(p1, p2):
    x = np.hstack((p1[0], p2[0])).reshape(1, -1)
    y = np.hstack((p1[1], p2[1])).reshape(1, -1)
    plt.plot(x, y, "->", linewidth=2.0)


def plot_progress_kmeans(X, centroids, previous_centroids, idx, K):
    R = np.linspace(0, K, K + 1)
    norm = plt.Normalize(0, K + 1)
    palete = plt.cm.hsv(norm(R))
    colors = palete[idx, :]

    plt.scatter(X[:, 0], X[:, 1], facecolors='none', color=colors)
    plt.plot(previous_centroids[:, 0], previous_centroids[:, 1], 'rx', markersize=10, linewidth=5.0)
    plt.plot(centroids[:, 0], centroids[:, 1], 'rx', markersize=10, linewidth=5.0)
    for j in range(centroids.shape[0]):
        p1 = centroids[j, :]
        p2 = previous_centroids[j, :]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "->", linewidth=2.0)
    return plt


def run_kmeans(X, initial_centroids, max_iters, plot_progress=False):
    """
    if plot_progress:
        plt.figure(figsize=(12, 8))
    """

    m, _ = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros(m)

    # Run K-means
    for i in range(max_iters):
        print(f"K-means iteration {i+1}/{max_iters} ....")

        idx = find_closest_centroids(X, centroids)
        if plot_progress:
            # plt = plot_progress_kmeans(X, centroids, previous_centroids, idx, K, i)
            # plt = plotProcessKMeans(X, centroids, previous_centroids)
            plt = plot_progress_kmeans(X, centroids, previous_centroids, idx, K)
            # plt.show()
            previous_centroids = centroids
            # time.sleep(2)

        centroids = compute_centroids(X, idx, K)

    if plot_progress:
        plt.show()

    return centroids, idx


def initialize_centroids(X, K):
    m = X.shape[0]
    rand_indices = np.random.permutation(m)
    centroids = X[rand_indices, :]
    return centroids
