import numpy as np

def conjugate_gradient(A, b, x0, tol=1e-6, max_iter=100000):
    x = x0.copy().flatten()
    r = A.dot(x) - b.copy().flatten()
    p = -r.copy()

    for n_iter in range(max_iter):
        if np.sqrt(np.dot(r, r)) <= tol:
            break

        Ap = A.dot(p)
        alpha = np.dot(r, r) / np.dot(p, Ap,)
        x = x + alpha * p
        r_new = r + alpha * Ap
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = -r_new + beta * p
        r = r_new

    f_min = A.dot(x)

    return x, n_iter, f_min


def steepest_descent(f, grad_f, x_0, eigenvalues, minimizer, Q, eps=1e-6, max_iter=50000, verbose=False):
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

        # Exact line search
        alpha = np.dot(grad.flatten(), grad.flatten()) / np.dot(np.dot(grad.flatten(), Q), grad.flatten())

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


def problem2():
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


    i = 1
    print("------------------- CG -------------------")
    for n, Q, b, x_0, minimizer, eig_val in zip(n_values, Qs, bs, x_0s, minimizers, eigenvalues):
        print("############################################")
        print("Task ", i)
        print("A = Hilbert, b = (1, 1, 1, 1, 1)")
        print("n = ", n)

        f = lambda x: quadratic_function(x, Q, b)
        f_grad = lambda x: quadratic_function_grad(x, Q, b)

        x_min, n_iter, f_min = conjugate_gradient(Q, b, x_0)
        grad_norm = np.dot(f_grad(x_min).flatten(), f_grad(x_min).flatten())
        print("Minimum found after {0} iterations: {1}".format(n_iter, x_min))
        print("Minimum value: {0}".format(f_min))
        print("||grad f(x)||:", grad_norm)
        print("Difference to true minimum: ",
              np.dot((b - f_min).flatten(), (b - f_min).flatten()))

        i = i + 1
    print("-------------------------------------------------------------")

    i = 1
    print("------------------- SD -------------------")
    for n, Q, b, x_0, minimizer, eig_val in zip(n_values, Qs, bs, x_0s, minimizers, eigenvalues):
        print("############################################")
        print("Task ", i)
        print("Q = Hilbert, b = (1, 1, 1, 1, 1)")
        print("f(x) = (1/2) * x.T * Q * x - b.T * x")
        print("n = ", n)

        f = lambda x: quadratic_function(x, Q, b)
        f_grad = lambda x: quadratic_function_grad(x, Q, b)

        x_min, f_min, grad_norm, n_iter, n_inequalitiy = steepest_descent(f, f_grad, x_0, eig_val, minimizer, Q, max_iter=1000000)
        print("Minimum found after {0} iterations: {1}".format(n_iter, x_min))
        print("Minimum value: {0}".format(f_min))
        print("||grad f(x)||:", grad_norm)
        print("Difference to true minimum: ",
              np.dot((b - f_min).flatten(), (b - f_min).flatten()))

        i = i + 1
    print("-------------------------------------------------------------")

problem2()