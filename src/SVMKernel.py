import numpy as np
import numpy.linalg as la

def linear(X1, X2):
    return X1.dot(X2.T)


def gaussian_vec(X1, X2, sigma="auto"):
    if sigma == "auto":
        sigma = 1 / X1.shape[1]

    return np.exp(- np.sqrt(np.sum(X1 ** 2, axis=1, keepdims=True) + np.sum(X2 ** 2, axis=1) - 2 * X1.dot(X2.T)) / (sigma ** 2))