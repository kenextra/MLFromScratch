import numpy as np
import scipy.optimize as op


def cofi_cost_function(params, Y, R, num_users, num_movies, num_features, lambd):
    X = np.reshape(params[0:num_movies * num_features], (num_movies, num_features), order='F')
    # theta = params[num_movies*num_features:, :]
    theta = params[num_movies * num_features:]
    Theta = np.reshape(theta, (num_users, num_features))
    Yrated = X.dot(Theta.T)
    error = Yrated - Y
    error_factor = R * error
    lamb = lambd / 2

    J = 0.5 * (np.sum(np.sum(error_factor**2))) + (lamb * np.trace(Theta.T.dot(Theta))) + (lamb * np.trace(X.T.dot(X)))
    return J


def cofi_gradient(params, Y, R, num_users, num_movies, num_features, lambd):
    X = np.reshape(params[0:num_movies * num_features], (num_movies, num_features), order='F')
    # theta = params[num_movies*num_features:, :]
    theta = params[num_movies * num_features:]
    Theta = np.reshape(theta, (num_users, num_features))
    Yrated = X.dot(Theta.T)
    error = Yrated - Y
    error_factor = R * error
    # lamb = lambd / 2

    X_grad = error_factor.dot(Theta) + (lambd * X)
    Theta_grad = error_factor.T.dot(X) + (lambd * Theta)
    X_grad = np.reshape(X_grad, (-1, 1), order='F')
    Theta_grad = np.reshape(Theta_grad, (-1, 1), order='F')

    grad = np.vstack((X_grad, Theta_grad))
    return grad


def normalize_ratings(Y, R):
    m, n = Y.shape
    Ymean = np.zeros(m)
    Ynorm = np.zeros(Y.shape)
    for i in range(m):
        idx = np.nonzero(R[i, :] == 1)
        Ymean[i] = np.mean(Y[i, idx])
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]
    return Ynorm, Ymean.reshape(-1, 1)


def train(initial_parameters, Ynorm, R, num_users, num_movies, num_features, options, lambd=0):
    result = op.minimize(fun=cofi_cost_function,
                         x0=initial_parameters,
                         args=(Ynorm, R, num_users, num_movies, num_features, lambd),
                         method='TNC',
                         jac=cofi_gradient,
                         options=options)
    return result.x


def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma
