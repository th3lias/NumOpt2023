import numpy as np


def approximate_gradient(f, x, eps=np.power(1.1e-16, 1/3, dtype='float32')):
    grad = np.empty_like(x)

    for i in range(x.shape[0]):
        ei = np.zeros_like(x)
        ei[i] = eps
        grad[i] = (f(x + ei) - f(x - ei)) / (2 * eps)

    return grad