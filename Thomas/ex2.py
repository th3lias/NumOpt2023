import numpy as np
from lib import *
import math
from scipy.interpolate import approximate_taylor_polynomial
import matplotlib.pyplot as plt
from scipy.stats import norm


def create_c(m, n, a):
    c = np.empty(shape=(n + 1, m))
    c[0, :] = 1

    for i in range(1, n+1):
        c[i, :] = np.power(a[:, 0], i)

    return c


def f(x, c, b, m):
    sum_res_squ = 0
    for j in range(m):
        sum_res_squ += np.power(residual(x, c, b, j), 2)

    return sum_res_squ / 2


def f_grad(x, c, b, m):
    sum = np.zeros(shape=(n + 1, 1))

    for j in range(m):
        cj = c[:, j].reshape(c.shape[0], 1)
        sum += (np.matmul(cj.T, x)[0, 0] - b[j, 0]) * cj

    return sum


def f_hessian(x, c, m):
    sum = np.zeros(shape=(n + 1, n + 1))

    for j in range(m):
        cj = c[:, j].reshape(c.shape[0], 1)
        sum += np.matmul(cj, cj.T)

    return sum


def residual(x, c, b, j):
    return np.matmul(c[:, j].T, x) - b[j, 0]


def plot_polynomial(coefficients, filename, title, q, fn, taylor):
    x = np.linspace(-q - 2, q + 2)
    y_sin = fn(x)
    pol = np.polynomial.Polynomial(coefficients)
    y_pol = pol(x)
    y_taylor = taylor(x)

    plt.title(title)
    line1, = plt.plot(x, y_sin, color='blue', label='g(x)')
    line2, = plt.plot(x, y_pol, color='orange', label='Optimal polynomial')
    line3, = plt.plot(x, y_taylor, color="red", label='Taylor expansion')
    plt.grid(visible=True)

    plt.legend(handles=[line1, line2, line3])

    plt.savefig(filename)


if __name__ == "__main__":
    qs = [ # interval -q to q
        # 2,
        # 2,
        # 2,
        # 2,
        10
    ]
    ms = [  # m datapoints
        # 100,
        # 100,
        # 200,
        # 200,
        150
    ]
    ns = [  # degree of polynomial
        # 3,
        # 5,
        # 3,
        # 5,
        5
    ]
    fns = [
        # np.sin,
        # np.sin,
        # np.cos,
        # np.cos,
        lambda x : norm.pdf(x, 0, 5)
    ]

    taylors = []

    for n, q, fn in zip(ns, qs, fns):
        taylors.append(approximate_taylor_polynomial(fn, 0, n, q, order=n+2))

    print("Newton Method")
    for i, q, m, n, fn, taylor in zip(range(len(qs)), qs, ms, ns, fns, taylors):
        a = np.linspace(-q, q, m).reshape(m, 1)
        b = fn(a)
        c = create_c(m, n, a)

        x0 = np.zeros(shape=(n + 1, 1), dtype='float64')
        x, n_, grad, fx = newton_method(
            f=lambda x: f(x, c, b, m),
            f_grad=lambda x: f_grad(x, c, b, m),
            f_hessian=lambda x: f_hessian(x, c, m),
            x0=x0)

        print("x estimated:", x, "||grad||:", np.linalg.norm(grad), "Iterations:", n_)
        plot_polynomial(x[:, 0], f"nm{i}", f"Newton Method q={q} m={m} n={n} iterations={n_}", q, fn, taylor)

    print("Steepest Descent")
    for i, q, m, n, fn, taylor in zip(range(len(qs)), qs, ms, ns, fns, taylors):
        a = np.linspace(-q, q, m).reshape(m, 1)
        b = fn(a)
        c = create_c(m, n, a)

        x0 = np.zeros(shape=(n + 1, 1), dtype='float64')
        x, n_, grad, fx = steepest_descent(
            f=lambda x: f(x, c, b, m),
            f_grad=lambda x: f_grad(x, c, b, m),
            x0=x0)

        print("x estimated:", x, "||grad||:", np.linalg.norm(grad), "Iterations:", n_)
        plot_polynomial(x[:, 0], f"sd{i}", f"Steepest Descent q={q} m={m} n={n} iterations={n_}", q, fn, taylor)
