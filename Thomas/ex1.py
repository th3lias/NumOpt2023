import numpy as np
from lib import *
import math


def polynomial(x, a, b, c):
    return np.power(x, 4)/4 - np.power(x, 3) / 3 * (a + b + c) + np.power(x, 2) / 2 * (c*b + a*c + a*b) - a*b*c*x


def polynomial_grad(x, a, b, c):
    return (x - a) * (x - b) * (x - c)


def polynomial_hessian(x, a, b, c):
    return 3 * np.power(x, 2) - 2 * x * (a + b + c) + a * b + c * b + a * c


def f1(x):
    return polynomial(x, a=-5, b=3, c=10)


def f1_grad(x):
    return polynomial_grad(x, a=-5, b=3, c=10)


def f1_hessian(x):
    return polynomial_hessian(x, a=-5, b=3, c=10)


def f2(x):
    return polynomial(x, a=2, b=3, c=4)


def f2_grad(x):
    return polynomial_grad(x, a=2, b=3, c=4)


def f2_hessian(x):
    return polynomial_hessian(x, a=2, b=3, c=4)


def f3(x):
    return polynomial(x, a=0.5, b=1, c=3)


def f3_grad(x):
    return polynomial_grad(x, a=0.5, b=1, c=3)


def f3_hessian(x):
    return polynomial_hessian(x, a=0.5, b=1, c=3)


def f4(x):
    return polynomial(x, a=-2, b=0.5, c=4)


def f4_grad(x):
    return polynomial_grad(x, a=-2, b=0.5, c=4)


def f4_hessian(x):
    return polynomial_hessian(x, a=-2, b=0.5, c=4)


def f5(x):
    return polynomial(x, a=-1.5, b=3, c=5)


def f5_grad(x):
    return polynomial_grad(x, a=-1.5, b=3, c=5)


def f5_hessian(x):
    return polynomial_hessian(x, a=-1.5, b=3, c=5)


if __name__ == "__main__":
    f = [f1, f2, f3, f4, f5]
    f_grad = [f1_grad, f2_grad, f3_grad, f4_grad, f5_grad]
    f_hessian = [f1_hessian, f2_hessian, f3_hessian, f4_hessian, f5_hessian]

    minimizers = [-5, 2, 0.5, -2, -1.5]

    for f_, f_grad_, minimizer in zip(f, f_grad, minimizers):
        print("Steepest Descent")
        x0 = np.array([0.0])
        x, n, grad, fx = steepest_descent(f_, f_grad_, x0)
        print("x estimated:", x, "f(x estimated):", fx, "||x estimated - x*||:", np.linalg.norm(x - minimizer),
              "||grad||:", np.linalg.norm(grad), "Iterations:", n)

    minimizers = [-5, 2, 0.5, 4, -1.5]

    for f_, f_grad_, f_hessian, minimizer in zip(f, f_grad, f_hessian, minimizers):
        print("Newton Method")
        x0 = np.array([0.0])
        x, n, grad, fx = newton_method(f_, f_grad_, f_hessian, x0)
        print("x estimated:", x, "f(x estimated):", fx, "||x estimated - x*||:", np.linalg.norm(x - minimizer),
              "||grad||:", np.linalg.norm(grad), "Iterations:", n)
