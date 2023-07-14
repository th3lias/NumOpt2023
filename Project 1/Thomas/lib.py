import numpy as np


def backtracking_line_search(f, x, fx, grad, alpha, c, p, rho):
    # print(f, x, fx, grad, alpha, c, p, rho)
    while f(x + alpha * p) > fx + c * alpha * np.matmul(grad.T, p):
        alpha = rho * alpha

    return alpha


def steepest_descent(f, f_grad, x0, alpha=np.array([0.5]), rho=np.array([0.5]), c=np.array([0.5]), stop=1e-6,
                     backtracking=True, debug=False, debug_at_step=100, max_iterations=500000):
    n = 0

    x = x0
    fx = f(x)
    grad = f_grad(x)

    while np.linalg.norm(grad) > stop:
        p = -grad
        curr_alpha = backtracking_line_search(f, x, fx, grad, alpha, c, p, rho) if backtracking else alpha
        x += curr_alpha * p
        fx = f(x)
        grad = f_grad(x)

        n += 1

        if debug and n % debug_at_step == 0:
            print("Iteration:", n, "x:", x, "f(x):", fx, "grad:", grad, "grad mag:", np.linalg.norm(grad),
                  "alpha:", curr_alpha)

        if n == max_iterations:
            break

    return x, n, grad, fx


def steepest_descent_iii(f, f_grad, x0, eigenvalues, minimum, alpha=np.array([0.5]), rho=np.array([0.5]),
                         c=np.array([0.5]), stop=1e-6, backtracking=True, debug=False, debug_at_step=100,
                         max_iterations=500000):
    n = 0
    n_inequality_satisfied = 0
    lambda_fraction_squ = np.power((eigenvalues[0] - eigenvalues[-1]) / (eigenvalues[0] + eigenvalues[-1]), 2)
    f_min = f(minimum)

    x = x0
    fx = f(x)
    grad = f_grad(x)

    while np.linalg.norm(grad) > stop:
        p = -grad
        curr_alpha = backtracking_line_search(f, x, fx, grad, alpha, c, p, rho) if backtracking else alpha
        fx_old = np.copy(fx)
        x += curr_alpha * p
        fx = f(x)
        grad = f_grad(x)

        if 2 * (fx - f_min) <= lambda_fraction_squ * 2 * (fx_old - f_min):
            n_inequality_satisfied += 1

        n += 1

        if debug and n % debug_at_step == 0:
            print("Iteration:", n, "x:", x, "f(x):", fx, "grad:", grad, "grad mag:", np.linalg.norm(grad),
                  "alpha:", curr_alpha)

        if n == max_iterations:
            break

    return x, n, grad, fx, n_inequality_satisfied


def newton_method(f, f_grad, f_hessian, x0, alpha=np.array([1]), rho=np.array([0.5]), c=np.array([0.5]), stop=1e-6,
                  backtracking=True, debug=False, debug_at_step=100, max_iterations=500000):
    n = 0

    x = x0
    fx = f(x)
    grad = f_grad(x)
    hessian = f_hessian(x)

    while np.linalg.norm(grad) > stop:
        p = -np.array([1]) * np.matmul(np.linalg.inv(hessian) if hessian.shape[0] > 1 else hessian, grad)
        curr_alpha = backtracking_line_search(f, x, fx, grad, alpha, c, p, rho) if backtracking else alpha
        x += curr_alpha * p
        fx = f(x)
        grad = f_grad(x)
        hessian = f_hessian(x)

        n += 1

        if debug and n % debug_at_step == 0:
            print("Iteration:", n, "x:", x, "f(x):", fx, "grad:", grad, "grad mag:", np.linalg.norm(grad),
                  "alpha:", curr_alpha)

        if n == max_iterations:
            break

    return x, n, grad, fx


def create_hilbert_matrix(n):
    mat = np.empty(shape=(n, n))

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            mat[i - 1, j - 1] = 1 / (i + j - 1)

    return mat
