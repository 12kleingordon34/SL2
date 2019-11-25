import numpy as np


def polynomial(X, Y, d):
    """
    Calculates a d-order polynomial kernel for
    data points x_i, x_j
    """
    product = np.inner(X, Y)
    return np.power(product, d)
