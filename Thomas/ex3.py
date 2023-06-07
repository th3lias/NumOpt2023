import numpy as np
from lib import *
import math


def quadratic(x, Q, b):
    return np.matmul(np.matmul(x.T, Q), x) / 2 - np.matmul(b.T, x)


def quadratic_grad(x, Q, b):
    return np.matmul(Q, x) - b


if __name__ == "__main__":
    ns = [5, 8, 12, 20, 30]

    minimizers = []
    Qs = []
    bs = []
    x0s = []
    eigenvalues = []
    eigenvectors = []
    for n in ns:
        Qs.append(create_hilbert_matrix(n))
        bs.append(np.ones(shape=(n, 1), dtype='float32'))
        minimizers.append(np.matmul(np.linalg.inv(Qs[-1]), bs[-1]))
        x0s.append(np.zeros(shape=(n, 1), dtype='float32'))
        val, vec = np.linalg.eig(Qs[-1])
        eigenvalues.append(val)
        eigenvectors.append(vec)

    print("Steepest Descent")
    for n, x0, minimizer, Q, b, ev in zip(ns, x0s, minimizers, Qs, bs, eigenvalues):
        f = lambda x : quadratic(x, Q, b)
        f_grad = lambda x : quadratic_grad(x, Q, b)
        x, it, grad, fx, n_inequality_satisfied = steepest_descent_iii(f, f_grad, x0, ev, minimizer, alpha=1,
                                            max_iterations=200000)
        print("n:", n, "f(x estimated):", fx, "||x estimated - x*||:", np.linalg.norm(x - minimizer),
              "||grad||:", np.linalg.norm(grad), f"\nInequalities satisfied: {n_inequality_satisfied / it * 100}%",
              "Iterations:", it,
              "Minimizer", minimizer,
              "\nx estimated:", x)
