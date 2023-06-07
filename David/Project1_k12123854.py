import numpy as np

def steepest_descent(f, grad_f, x_0, alpha_init=0.5, rho=0.5, c=0.1, eps=1e-6, max_iter=50000, verbose=False):
    x = x_0
    n_iter = 0

    while n_iter < max_iter:
        grad = grad_f(x)
        if np.linalg.norm(grad) < eps:
            break

        alpha = alpha_init
        while f(alpha * (-grad) + x) > f(x) + c * alpha * np.dot(np.asarray(grad).flatten(), np.asarray(-grad).flatten()):
            alpha = rho * alpha

        x = x - alpha * grad

        if verbose:
            f_val = f(x)
            print("Iteration {0}: f = {1}".format(n_iter, f_val))
        n_iter += 1

    x_min = x
    f_min = f(x_min)
    grad_norm = np.dot(np.array(grad).flatten(), np.array(grad).flatten())
    return x_min, f_min, grad_norm, n_iter


def steepest_descent_iii(f, grad_f, x_0, eigenvalues, minimizer, alpha_init=0.5, rho=0.5, c=0.1, eps=1e-6, max_iter=50000, verbose=False):
    x = x_0
    n_iter = 0
    n_inequality_satisfied = 0
    lambda_fraction = np.power((eigenvalues[0] - eigenvalues[-1]) / (eigenvalues[0] + eigenvalues[-1]), 2)
    f_min = f(minimizer)

    while n_iter < max_iter:
        grad = grad_f(x)
        if np.linalg.norm(grad) < eps:
            break

        fx_old = f(x)
        alpha = alpha_init
        while f(alpha * (-grad) + x) > fx_old + c * alpha * np.dot(np.asarray(grad).flatten(), np.asarray(-grad).flatten()):
            alpha = rho * alpha

        x = x - alpha * grad

        if 2 * (f(x) - f_min) <= lambda_fraction * 2 * (fx_old - f_min):
            n_inequality_satisfied += 1

        if verbose:
            f_val = f(x)
            print("Iteration {0}: f = {1}".format(n_iter, f_val))
        n_iter += 1

    x_min = x
    f_min = f(x_min)
    grad_norm = np.dot(grad.flatten(), grad.flatten())
    return x_min, f_min, grad_norm, n_iter, n_inequality_satisfied


def newtons_method(f, grad_f, hess_f, x_0, rho=0.5, c=0.5, eps=1e-6, max_iter=10000, verbose=False):
    x = x_0
    n_iter = 0

    while n_iter < max_iter:
        grad = grad_f(x)
        if np.linalg.norm(grad) < eps:
            break

        hess = hess_f(x)

        try:
            inv_hess = np.linalg.inv(hess)
        except Exception as e:
            print(e)
            break

        grad = np.array(grad)
        alpha = 1

        p_k = np.dot(-inv_hess, grad)

        while f(x + alpha * p_k) > f(x) + c * alpha * np.dot(grad.flatten(), p_k.flatten()):
            alpha = rho * alpha

        x = x - alpha * (np.dot(inv_hess, grad))

        if verbose:
            f_val = f(x)
            print("Iteration {0}: f = {1}".format(n_iter, f_val))
        n_iter += 1

    x_min = x
    f_min = f(x_min)
    grad_norm = np.dot(np.array(grad).flatten(), np.array(grad).flatten())
    return x_min, f_min, grad_norm, n_iter


def create_hilbert_mat(n):
    mat = np.empty(shape=(n, n))

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            mat[i - 1, j - 1] = 1 / (i + j - 1)

    return mat


def quadratic_function(x, Q, b):
    return (1 / 2) * np.matmul(np.matmul(x.T, Q), x) - np.matmul(b.T, x)


def quadratic_function_grad(x, Q, b):
    return np.matmul(Q, x) - b


def poly_4d(x, a, b, c):
    return (x * (3 * (x ** 3) - 4 * (a + b + c) * (x ** 2) + 6 * (b * c + a * c + a * b) * x - 12 * a * b * c)) / 12


def grad_poly_4d(x, a, b, c):
    return (x - a) * (x - b) * (x - c)


def hess_poly_4d(x, a, b, c):
    hess = np.empty((1, 1))
    hess[0][0] = (x - b) * (x - c) + (x - a) * (x - c) * (x - a) * (x - b)
    return hess


def f1(x):
    return poly_4d(x, 1, 2, 3)


def f1_grad(x):
    return grad_poly_4d(x, 1, 2, 3)


def f1_hess(x):
    return hess_poly_4d(x, 1, 2, 3)


def f2(x):
    return poly_4d(x, 3, 3, 3)


def f2_grad(x):
    return grad_poly_4d(x, 3, 3, 3)


def f2_hess(x):
    return hess_poly_4d(x, 3, 3, 3)


def f3(x):
    return poly_4d(x, 5, 8, 1)


def f3_grad(x):
    return grad_poly_4d(x, 5, 8, 1)


def f3_hess(x):
    return hess_poly_4d(x, 5, 8, 1)


def f4(x):
    return poly_4d(x, 2, 4, 4)


def f4_grad(x):
    return grad_poly_4d(x, 2, 4, 4)


def f4_hess(x):
    return hess_poly_4d(x, 2, 4, 4)


def f5(x):
    return poly_4d(x, 5, 2, 1)


def f5_grad(x):
    return grad_poly_4d(x, 5, 2, 1)


def f5_hess(x):
    return hess_poly_4d(x, 5, 2, 1)


def f26(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def f26_grad(x):
    return [400 * x[0] ** 3 + (2 - 400 * x[1]) * x[0] - 2, 200 * (x[1] - x[0] ** 2)]


def f26_hess(x):
    return [
        [1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]],
        [-400 * x[0], 200]]


def f27(x):
    return (x[0] - 4) ** 4 + (x[1] + 2) ** 4


def f27_grad(x):
    return [4 * (x[0] - 4) ** 3, 4 * (x[1] + 2) ** 3]


def f27_hess(x):
    return [[12 * (x[0] - 4) ** 2, 0],
            [0, 12 * (x[1] + 2) ** 2]]


def f28(x):
    return (- x[0] ** 2 + 25 * x[1]) ** 2 + (5 - x[0]) ** 2


def f28_grad(x):
    return [4 * x[0] ** 3 + (2 - 100 * x[1]) * x[0] - 10, 50 * (25 * x[1] - x[0] ** 2)]


def f28_hess(x):
    return [[12 * x[0] ** 2 - 100 * x[1] + 2, -100 * x[0]],
            [-100 * x[0], 1250]]


def f29(x):
    return (-2 * x[0] ** 2 + 4 * x[1]) ** 2 + (x[0] + 2) ** 2


def f29_grad(x):
    return [16 * x[0] ** 3 + (2 - 32 * x[1]) * x[0] + 4,
            32 * x[1] - 16 * x[0] ** 2]


def f29_hess(x):
    return [[48 * x[0] ** 2 - 32 * x[0] + 2, -32 * x[0]],
            [-32 * x[0], 32]]


def f30(x):
    return (x[0] ** 2 - 2 * x[1]) ** 2 + (x[0] - 3) ** 2


def f30_grad(x):
    return [4 * x[0] ** 3 + (2 - 8 * x[1]) * x[0] - 6,
            8 * x[1] - 4 * x[0] ** 2]


def f30_hess(x):
    return [
        [12 * x[0] ** 2 - 8 * x[1] + 2, -8 * x[0]],
        [-8 * x[0], 8]]


def f31(x):
    return (x[0] ** 2 - 3 * x[1]) ** 2 + (x[0] - 3) ** 2


def f31_grad(x):
    return [4 * x[0] ** 3 + (2 - 12 * x[1]) * x[0] - 6,
            18 * x[1] - 6 * x[0] ** 2]


def f31_hess(x):
    return [
        [12 * x[0] ** 2 - 12 * x[1] + 2, -12 * x[0]],
        [-12 * x[0], 18]]


def f32(x):
    return (2 * x[0] - 2 * x[1] ** 2) ** 2 + (x[1] + 2) ** 2


def f32_grad(x):
    return [8 * (x[0] - x[1] ** 2),
            16 * x[1] ** 3 + (2 - 16 * x[0]) * x[1] + 4]


def f32_hess(x):
    return [
        [8, -16 * x[1]],
        [-16 * x[1], 48 * x[1] ** 2 - 16 * x[0] + 2]]


def f33(x):
    return (3 * x[1] ** 2 - x[0]) ** 2 + (x[1] + 1) ** 2


def f33_grad(x):
    return [2 * x[0] - 6 * x[1] ** 2,
            36 * x[1] ** 3 + (2 - 12 * x[0]) * x[1] + 2]


def f33_hess(x):
    return[
        [2, -12 * x[1]],
        [-12 * x[1], 108 * x[1] ** 2 - 12 * x[0] + 2]]


def f34(x):
    return (x[0] ** 2 + 2 * x[1]) ** 2 + (x[0] + 1) ** 2


def f34_grad(x):
    return [4 * x[0] ** 3 + (8 * x[1] + 2) * x[0] + 2,
            4 * (2 * x[1] + x[0] ** 2)]


def f34_hess(x):
    return [
        [12 * x[0] ** 2 + 8 * x[1] + 2, 8 * x[0]],
        [8 * x[0], 8]]


def f35(x):
    return (5 * x[0] - x[1] ** 2) ** 2 + (x[1] - 5) ** 2


def f35_grad(x):
    return [10 * (5 * x[0] - x[1] ** 2),
            4 * x[1] ** 3 + (2 - 20 * x[0]) * x[1] - 10]


def f35_hess(x):
    return [
        [50, -20 * x[1]],
        [-20 * x[1], 12 * x[1] ** 2 - 20 * x[0] + 2]]


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


def f_grad(x, c, b, m, n):
    sum = np.zeros(shape=(n + 1, 1))

    for j in range(m):
        cj = c[:, j].reshape(c.shape[0], 1)
        sum += (np.matmul(cj.T, x)[0, 0] - b[j, 0]) * cj

    return sum


def f_hessian(x, c, b, m, n):
    hess = np.zeros((n+1, n+1))
    for j in range(m):
        cj = c[:, j].reshape(c.shape[0], 1)
        x_cj = np.matmul(x.T, cj)[0, 0]
        hess += np.matmul(cj, cj.T) * np.power(x_cj - b[j, 0], 2)
    return hess


def residual(x, c, b, j):
    return np.matmul(c[:, j].T, x) - b[j, 0]


def problem1_SD():
    print("Problem 1 - Steepest Descent")
    print("-------------------------------------------------------------")

    print("Task 1")
    x_min, f_min, grad_norm, n_iter = steepest_descent(f1, f1_grad, 0, verbose=False)
    print("Minimum found after {0} iterations: {1}".format(n_iter, x_min))
    print("Minimum value: {0}".format(f_min))
    print("||grad f(x)||:", grad_norm)
    print("Difference to actual minimizer: ", 3 - x_min)

    print("Task 2")
    x_min, f_min, grad_norm, n_iter = steepest_descent(f2, f2_grad, 0, verbose=False)
    print("Minimum found after {0} iterations: {1}".format(n_iter, x_min))
    print("Minimum value: {0}".format(f_min))
    print("||grad f(x)||:", grad_norm)
    print("Difference to actual minimizer: ", 3 - x_min)

    print("Task 3")
    x_min, f_min, grad_norm, n_iter = steepest_descent(f3, f3_grad, 0, verbose=False)
    print("Minimum found after {0} iterations: {1}".format(n_iter, x_min))
    print("Minimum value: {0}".format(f_min))
    print("||grad f(x)||:", grad_norm)
    print("Difference to actual minimizer: ", 1 - x_min)

    print("Task 4")
    x_min, f_min, grad_norm, n_iter = steepest_descent(f4, f4_grad, 0, verbose=False)
    print("Minimum found after {0} iterations: {1}".format(n_iter, x_min))
    print("Minimum value: {0}".format(f_min))
    print("||grad f(x)||:", grad_norm)
    print("Difference to actual minimizer: ", 4 - x_min)

    print("Task 5")
    x_min, f_min, grad_norm, n_iter = steepest_descent(f5, f5_grad, 0, verbose=False)
    print("Minimum found after {0} iterations: {1}".format(n_iter, x_min))
    print("Minimum value: {0}".format(f_min))
    print("||grad f(x)||:", grad_norm)
    print("Difference to actual minimizer: ", 5 - x_min)

    print("-------------------------------------------------------------")


def problem1_NM():
    print("Problem 1 - Newton's Method")
    print("-------------------------------------------------------------")

    # Task 1
    print("Task 6 (Task 1)")
    x_min, f_min, grad_norm, n_iter = newtons_method(f1, f1_grad, f1_hess, 0, verbose=False)
    print("Minimum found after {0} iterations: {1}".format(n_iter, x_min))
    print("Minimum value: {0}".format(f_min))
    print("||grad f(x)||:", grad_norm)
    print("Difference to actual minimizer: ", 1 - x_min)

    print("Task 7 (Task 2)")
    x_min, f_min, grad_norm, n_iter = newtons_method(f2, f2_grad, f2_hess, 0, verbose=False)
    print("Minimum found after {0} iterations: {1}".format(n_iter, x_min))
    print("Minimum value: {0}".format(f_min))
    print("||grad f(x)||:", grad_norm)
    print("Difference to actual minimizer: ", 3 - x_min)

    print("Task 8 (Task 3)")
    x_min, f_min, grad_norm, n_iter = newtons_method(f3, f3_grad, f3_hess, 0, verbose=False)
    print("Minimum found after {0} iterations: {1}".format(n_iter, x_min))
    print("Minimum value: {0}".format(f_min))
    print("||grad f(x)||:", grad_norm)
    print("Difference to actual minimizer: ", 1 - x_min)

    print("Task 9 (Task 4)")
    x_min, f_min, grad_norm, n_iter = newtons_method(f4, f4_grad, f4_hess, 0, verbose=False)
    print("Minimum found after {0} iterations: {1}".format(n_iter, x_min))
    print("Minimum value: {0}".format(f_min))
    print("||grad f(x)||:", grad_norm)
    print("Difference to actual minimizer: ", 2 - x_min)

    print("Task 10 (Task 5)")
    x_min, f_min, grad_norm, n_iter = newtons_method(f5, f5_grad, f5_hess, 0, verbose=False)
    print("Minimum found after {0} iterations: {1}".format(n_iter, x_min))
    print("Minimum value: {0}".format(f_min))
    print("||grad f(x)||:", grad_norm)
    print("Difference to actual minimizer: ", 1 - x_min)

    print("-------------------------------------------------------------")


def problem2_SD():
    print("Problem 2 - Steepest Descent")
    print("-------------------------------------------------------------")

    qs = [1, 2, 2, 1.5, 0.5]  # interval -q to q
    ms = [50, 50, 100, 50, 20]  # m datapoints
    ns = [3, 3, 5, 3, 1]  # degree of polynomial


    i = 11
    for q, m, n in zip(qs, ms, ns):
        print("Task ", i)
        print("q =", q, "; m =", m, "; degree =", n)
        a = np.linspace(-q, q, num=m)
        a = a.reshape(m, 1)
        b = np.sin(a)
        c = create_c(m, n, a)

        x0 = np.zeros(shape=(n + 1, 1), dtype='float64')
        x_min, f_min, grad_norm, n_iter = steepest_descent(
            f=lambda x: f(x, c, b, m),
            grad_f=lambda x: f_grad(x, c, b, m, n),
            x_0=x0,
            verbose=False,
            alpha_init=1,
            c=0.5,
            max_iter=2000)

        print("x estimated:", x_min,
              "f(x) estimated:", f_min,
              "||grad||:", grad_norm,
              "Iterations:", n_iter)

        i += 1
    print("-------------------------------------------------------------")


def problem2_NM():
    print("Problem 2 - Newton's Method")
    print("-------------------------------------------------------------")

    qs = [1, 2, 2, 3, 5]  # interval -q to q
    ms = [50, 50, 100, 100, 100]  # m datapoints
    ns = [3, 3, 5, 9, 15]  # degree of polynomial

    i = 16
    for q, m, n in zip(qs, ms, ns):
        print("Task ", i)
        print("q =", q, "; m =", m, "; degree =", n)
        a = np.linspace(-q, q, num=m)
        a = a.reshape(m, 1)
        b = np.sin(a)
        c = create_c(m, n, a)

        x0 = np.zeros(shape=(n + 1, 1), dtype='float64')
        x_min, f_min, grad_norm, n_iter = newtons_method(
            f=lambda x: f(x, c, b, m),
            grad_f=lambda x: f_grad(x, c, b, m, n),
            hess_f=lambda x: f_hessian(x, c, b, m, n),
            x_0=x0,
            verbose=False,
            c=0.5,
            max_iter=2000)

        print("x estimated:", x_min,
              "f(x) estimated:", f_min,
              "Iterations:", n_iter)

        i += 1
    print("-------------------------------------------------------------")


def problem3():
    print("Problem 3 - Steepest Descent")
    print("-------------------------------------------------------------")

    eigenvalues = []
    eigenvectors = []
    Qs = []
    bs = []
    x_0s = []
    minimizers = []
    n_values = [5, 8, 12, 20, 30]

    for n in n_values:
        Qs.append(create_hilbert_mat(n))
        bs.append(np.ones(shape=(n, 1), dtype='float32'))
        x_0s.append(np.zeros(shape=(n, 1)))
        minimizers.append(np.matmul(np.linalg.inv(Qs[-1]), bs[-1]))

        eig_val, eig_vec = np.linalg.eig(Qs[-1])
        eigenvalues.append(eig_val)
        eigenvectors.append(eig_vec)

    i = 21
    for n, Q, b, x_0, minimizer, eig_val in zip(n_values, Qs, bs, x_0s, minimizers, eigenvalues):
        print("Task ", i)
        print("n = ", n)
        f = lambda x: quadratic_function(x, Q, b)
        f_grad = lambda x: quadratic_function_grad(x, Q, b)

        x_min, f_min, grad_norm, n_iter, n_inequalitiy = steepest_descent_iii(f, f_grad, x_0, eig_val, minimizer, max_iter=1200000)
        print("Minimum found after {0} iterations: {1}".format(n_iter, x_min))
        print("Minimum value: {0}".format(f_min))
        print("True Minimizer: ", minimizer)
        print("||grad f(x)||:", grad_norm)
        print("Difference to true minimizer: ", np.dot((minimizer - x_min).flatten(), (minimizer - x_min).flatten()))
        print("Eigenvalues: ", eig_val)
        print("(lambda_n - lambda_1) / (lambda_n + lambda_1): ", (eig_val[-1] - eig_val[0]) / (eig_val[-1] + eig_val[0]))
        print("Condition number of Q: ", eig_val[-1] / eig_val[0])
        print("Inequalities satisified: ", n_inequalitiy)
        print("")
        i = i + 1

    print("-------------------------------------------------------------")


def problem4():
    print("Problem 4 - Newton's Method")
    print("-------------------------------------------------------------")

    print("Task 26 (Rosenbrock)")
    x_min, f_min, grad_norm, n_iter = newtons_method(f26, f26_grad, f26_hess, [0, 0], verbose=False)
    print("Minimum found after {0} iterations: {1}".format(n_iter, x_min))
    print("Minimum value: {0}".format(f_min))
    print("||grad f(x)||:", grad_norm)
    print("Difference to actual minimizer: ", np.dot([1, 1] - x_min, [1, 1] - x_min))

    print("Task 27")
    x_min, f_min, grad_norm, n_iter = newtons_method(f27, f27_grad, f27_hess, [0, 0], verbose=False)
    print("Minimum found after {0} iterations: {1}".format(n_iter, x_min))
    print("Minimum value: {0}".format(f_min))
    print("||grad f(x)||:", grad_norm)
    print("Difference to actual minimizer: ", np.dot([4, -2] - x_min, [4, -2] - x_min))

    print("Task 28")
    x_min, f_min, grad_norm, n_iter = newtons_method(f28, f28_grad, f28_hess, [0, 0], verbose=False)
    print("Minimum found after {0} iterations: {1}".format(n_iter, x_min))
    print("Minimum value: {0}".format(f_min))
    print("||grad f(x)||:", grad_norm)
    print("Difference to actual minimizer: ", np.dot([5, 1] - x_min, [5, 1] - x_min))

    print("Task 29")
    x_min, f_min, grad_norm, n_iter = newtons_method(f29, f29_grad, f29_hess, [0, 0], verbose=False)
    print("Minimum found after {0} iterations: {1}".format(n_iter, x_min))
    print("Minimum value: {0}".format(f_min))
    print("||grad f(x)||:", grad_norm)
    print("Difference to actual minimizer: ", np.dot([-2, 2] - x_min, [-2, 2] - x_min))

    print("Task 30")
    x_min, f_min, grad_norm, n_iter = newtons_method(f30, f30_grad, f30_hess, [0, 0], verbose=False)
    print("Minimum found after {0} iterations: {1}".format(n_iter, x_min))
    print("Minimum value: {0}".format(f_min))
    print("||grad f(x)||:", grad_norm)
    print("Difference to actual minimizer: ", np.dot([3, 4.5] - x_min, [3, 4.5] - x_min))

    print("Task 31")
    x_min, f_min, grad_norm, n_iter = newtons_method(f31, f31_grad, f31_hess, [0, 0], verbose=False)
    print("Minimum found after {0} iterations: {1}".format(n_iter, x_min))
    print("Minimum value: {0}".format(f_min))
    print("||grad f(x)||:", grad_norm)
    print("Difference to actual minimizer: ", np.dot([3, 3] - x_min, [3, 3] - x_min))

    print("Task 32")
    x_min, f_min, grad_norm, n_iter = newtons_method(f32, f32_grad, f32_hess, [0, 0], verbose=False)
    print("Minimum found after {0} iterations: {1}".format(n_iter, x_min))
    print("Minimum value: {0}".format(f_min))
    print("||grad f(x)||:", grad_norm)
    print("Difference to actual minimizer: ", np.dot([4, -2] - x_min, [4, -2] - x_min))

    print("Task 33")
    x_min, f_min, grad_norm, n_iter = newtons_method(f33, f33_grad, f33_hess, [0, 0], verbose=False)
    print("Minimum found after {0} iterations: {1}".format(n_iter, x_min))
    print("Minimum value: {0}".format(f_min))
    print("||grad f(x)||:", grad_norm)
    print("Difference to actual minimizer: ", np.dot([3, -1] - x_min, [3, -1] - x_min))

    print("Task 34")
    x_min, f_min, grad_norm, n_iter = newtons_method(f34, f34_grad, f34_hess, [0, 0], verbose=False)
    print("Minimum found after {0} iterations: {1}".format(n_iter, x_min))
    print("Minimum value: {0}".format(f_min))
    print("||grad f(x)||:", grad_norm)
    print("Difference to actual minimizer: ", np.dot([-1, -0.5] - x_min, [-1, -0.5] - x_min))

    print("Task 35")
    x_min, f_min, grad_norm, n_iter = newtons_method(f35, f35_grad, f35_hess, [0, 0], verbose=False)
    print("Minimum found after {0} iterations: {1}".format(n_iter, x_min))
    print("Minimum value: {0}".format(f_min))
    print("||grad f(x)||:", grad_norm)
    print("Difference to actual minimizer: ", np.dot([5, 5] - x_min, [5, 5] - x_min))




#problem1_SD()
#problem1_NM()
#problem2_SD()
#problem2_NM()
#problem3()
problem4()




