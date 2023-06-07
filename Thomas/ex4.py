import numpy as np
from lib import *
import math


def f26(x):
    return (5 * x[0, 0] - x[1, 0] ** 2) ** 2 + (x[1, 0] - 5) ** 2


def f26grad(x):
    return np.array([ [10 * (5 * x[0, 0] - x[1, 0] ** 2)],
            [4 * x[1, 0] ** 3 + (2 - 20 * x[0, 0]) * x[1, 0] - 10]])


def f26hessian(x):
    return np.array([
        [50, -20 * x[1, 0]],
        [-20 * x[1, 0], 12 * x[1, 0] ** 2 - 20 * x[0, 0] + 2]])


def f27(x):
    return np.array((x[0, 0] ** 2 + 2 * x[1, 0]) ** 2 + (x[0, 0] + 1) ** 2)


def f27grad(x):
    return np.array([[4 * x[0, 0] ** 3 + (8 * x[1, 0] + 2) * x[0, 0] + 2],
            [4 * (2 * x[1, 0] + x[0, 0] ** 2)]])


def f27hessian(x):
    return np.array([
        [12 * x[0, 0] ** 2 + 8 * x[1, 0] + 2, 8 * x[0, 0]],
        [8 * x[0, 0], 8]])


def f28(x):
    return (3 * x[1, 0] ** 2 - x[0, 0]) ** 2 + (x[1, 0] + 1) ** 2


def f28grad(x):
    return np.array([ [2 * x[0, 0] - 6 * x[1, 0] ** 2],
             [36 * x[1, 0] ** 3 + (2 - 12 * x[0, 0]) * x[1, 0] + 2]])


def f28hessian(x):
    return np.array([
        [2, -12 * x[1, 0]],
        [-12 * x[1, 0], 108 * x[1, 0] ** 2 - 12 * x[0, 0] + 2]])


def f29(x):
    return (2 * x[0, 0] - 2 * x[1, 0] ** 2) ** 2 + (x[1, 0] + 2) ** 2


def f29grad(x):
    return np.array([[8 * (x[0, 0] - x[1, 0] ** 2)],
            [16 * x[1, 0] ** 3 + (2 - 16 * x[0, 0]) * x[1, 0] + 4]])


def f29hessian(x):
    return np.array([
        [8, -16 * x[1, 0]],
        [-16 * x[1, 0], 48 * x[1, 0] ** 2 - 16 * x[0, 0] + 2]])


def f30(x):
    return (x[0, 0] ** 2 - 3 * x[1, 0]) ** 2 + (x[0, 0] - 3) ** 2


def f30grad(x):
    return np.array([[4 * x[0, 0] ** 3 + (2 - 12 * x[1, 0]) * x[0, 0] - 6],
            [18 * x[1, 0] - 6 * x[0, 0] ** 2]])


def f30hessian(x):
    return np.array([
        [12 * x[0, 0] ** 2 - 12 * x[1, 0] + 2, -12 * x[0, 0]],
        [-12 * x[0, 0], 18]])


if __name__ == "__main__":
    fns = [f26, f27, f28, f29, f30]
    grads = [f26grad, f27grad, f28grad, f29grad, f30grad]
    hessians = [f26hessian, f27hessian, f28hessian, f29hessian, f30hessian]

    minimizers = np.array([
        [[5.0], [5.0]],
        [[-1.0], [-0.5]],
        [[3.0], [-1.0]],
        [[4.0], [-2.0]],
        [[3.0], [3.0]]
    ])

    print("Newton Method")
    for i, f, g, h in zip(range(5), fns, grads, hessians):
        minimizer = minimizers[i, :]
        x0 = np.array([[0.0], [0.0]])

        x, n, grad, fx = newton_method(f, g, h, x0)
        print("x estimated:", x, "f(x estimated):", fx, "||x estimated - x*||:", np.linalg.norm(x - minimizer),
              "||grad||:", np.linalg.norm(grad), "Iterations:", n)
