# non linear Conjugate gradient with Polak-Ribiere formula
import numpy as np


class PR:
    def __init__(self, f, df, x0, tol, solution):
        self.f = f
        self.df = df
        self.x0 = x0
        self.tol = tol
        self.solution = solution

    def run(self):
        # initialization
        i = 0
        x = self.x0
        d = -self.df(x)
        results = np.array([x])
        while True:
            # line search
            a = self.lineSearch(x, d)
            # update x
            x = x + a * d
            # update d
            d = -self.df(x) + self.beta(x, results[-1]) * d
            # save results
            results = np.vstack((results, x))
            # update iteration
            i += 1
            if np.linalg.norm(self.df(x)) < self.tol:
                break
        return results, i, x, np.linalg.norm(self.df(x)), np.linalg.norm(x - self.solution)

    def lineSearch(self, x, d):
        # initial step size
        a = 1
        # initial function value
        f0 = self.f(x)
        # initial gradient value
        df0 = self.df(x)
        # initial function value after step
        f1 = self.f(x + a * d)
        # initial gradient value after step
        df1 = self.df(x + a * d)
        # loop until Armijo condition is satisfied
        while f1 > f0 + 0.5 * a * df0.T @ d:
            # update step size
            a = 0.5 * a
            # update function value after step
            f1 = self.f(x + a * d)
        return a

    def beta(self, x, x_old):
        return (self.df(x).T @ (self.df(x) - self.df(x_old)) / np.linalg.norm(self.df(x_old)) ** 2)

    # main function
if __name__ == "__main__":
    # pass rosennbrock function with point (1.2, 1.2)
    pr = PR(lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2,
            lambda x: np.array([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)]),
            np.array([1.2, 1.2]), 1e-6, np.array([1, 1]))
    results, no_iter, x, grad_norm, distance_solution = pr.run()
    print("Rosennbrock function with point (1.2, 1.2)")
    print("Number of iterations: ", no_iter)
    print("Solution: ", x)
    print("Gradient norm: ", grad_norm)
    print("Distance to solution: ", distance_solution)
    print("#############################################")
    # pass rosennbrock function with point (-1.2, 1)
    pr = PR(lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2,
            lambda x: np.array([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)]),
            np.array([-1.2, 1]), 1e-6, np.array([1, 1]))
    results, no_iter, x, grad_norm, distance_solution = pr.run()
    print("Rosennbrock function with point (-1.2, 1)")
    print("Number of iterations: ", no_iter)
    print("Solution: ", x)
    print("Gradient norm: ", grad_norm)
    print("Distance to solution: ", distance_solution)
    print("#############################################")
    # pass rosennbrock function with point (0.2, 0.8)
    pr = PR(lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2,
            lambda x: np.array([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)]),
            np.array([0.2, 0.8]), 1e-6, np.array([1, 1]))
    results, no_iter, x, grad_norm, distance_solution = pr.run()
    print("Rosennbrock function with point (0.2, 0.8)")
    print("Number of iterations: ", no_iter)
    print("Solution: ", x)
    print("Gradient norm: ", grad_norm)
    print("Distance to solution: ", distance_solution)
    print("#############################################")
    # pass f(x) = 150 * (x1*x2) ** 2 + (0.5 * x1 + 2 * x2 - 2) ** 2 with point (-0.2, 1.2)
    pr = PR(lambda x: 150 * (x[0] * x[1]) ** 2 + (0.5 * x[0] + 2 * x[1] - 2) ** 2, lambda x: np.array(
        [(600 * x[0] * x[1] ** 2 + x[0] + 4 * x[1] - 4) / 2, 300 * x[0] ** 2 * x[1] + 2 * x[0] + 8 * x[1] - 8]),
            np.array([-0.2, 1.2]), 1e-6, np.array([0, 1]))
    results, no_iter, x, grad_norm, distance_solution = pr.run()
    print("f(x) = 150 * (x1*x2) ** 2 + (0.5 * x1 + 2 * 2* x2 - 2) ** 2 with point (-0.2, 1.2)")
    print("Number of iterations: ", no_iter)
    print("Solution: ", x)
    print("Gradient norm: ", grad_norm)
    print("Distance to solution: ", distance_solution)
    print("#############################################")
    # pass f(x) = 150 * (x1*x2) ** 2 + (0.5 * x1 + 2 * x2 - 2) ** 2 with point (3.8, 0.1)
    pr = PR(lambda x: 150 * (x[0] * x[1]) ** 2 + (0.5 * x[0] + 2 * x[1] - 2) ** 2, lambda x: np.array(
        [(600 * x[0] * x[1] ** 2 + x[0] + 4 * x[1] - 4) / 2, 300 * x[0] ** 2 * x[1] + 2 * x[0] + 8 * x[1] - 8]),
            np.array([3.8, 0.1]), 1e-6, np.array([4, 0]))
    results, no_iter, x, grad_norm, distance_solution = pr.run()
    print("f(x) = 150 * (x1*x2) ** 2 + (0.5 * x1 + 2 * 2* x2 - 2) ** 2 with point (3.8, 0.1)")
    print("Number of iterations: ", no_iter)
    print("Solution: ", x)
    print("Gradient norm: ", grad_norm)
    print("Distance to solution: ", distance_solution)
    print("#############################################")
    # pass f(x) = 150 * (x1*x2) ** 2 + (0.5 * x1 + 2 * x2 - 2) ** 2 with point (1.9, 0.6)
    pr = PR(lambda x: 150 * (x[0] * x[1]) ** 2 + (0.5 * x[0] + 2 * x[1] - 2) ** 2, lambda x: np.array(
        [(600 * x[0] * x[1] ** 2 + x[0] + 4 * x[1] - 4) / 2, 300 * x[0] ** 2 * x[1] + 2 * x[0] + 8 * x[1] - 8]),
            np.array([1.9, 0.6]), 1e-6, np.array([4, 0]))
    print("f(x) = 150 * (x1*x2) ** 2 + (0.5 * x1 + 2 * 2* x2 - 2) ** 2 with point (1.9, 0.6)")
    print("Number of iterations: ", no_iter)
    print("Solution: ", x)
    print("Gradient norm: ", grad_norm)
    print("Distance to solution: ", distance_solution)
    print("#############################################")