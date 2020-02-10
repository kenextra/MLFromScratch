import numpy as np
from scipy.io import loadmat
from utils import display_data, one_vs_all, predict_ova

data = loadmat("data/ex3data1.mat")
X = data['X']
y = data['y']

m, n = X.shape
indices = np.random.permutation(m)
data_points = X[indices[0:100], :]

display_data(data_points)

num_labels = 10
lambd = 0.1
all_theta = one_vs_all(X, y, num_labels, lambd)

pred = predict_ova(all_theta, X)

print(f'Train Accuracy: {np.mean(pred == y) * 100:.1f}%')
