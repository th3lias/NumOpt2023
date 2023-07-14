# nonlinear CG method with Fletcher-Reeves formula
import numpy as np
from utils import approximate_gradient
class FR:
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
        d = -self.df(self.f, x)
        results = np.array([x])
        while True:
            # line search
            a = self.lineSearch(x, d)
            # update x
            x = x + a * d
            # update d
            d = -self.df(self.f, x) + self.beta(x, results[-1]) * d
            # save results
            results = np.vstack((results, x))
            # update iteration
            i += 1
            if np.linalg.norm(self.df(self.f, x)) < self.tol:
                break
        return results, i, x, np.linalg.norm(self.df(self.f, x)), np.linalg.norm(x - self.solution)


    def lineSearch(self, x, d, c=0.5):
        # initial step size
        a = 1
        # initial function value
        f0 = self.f(x)
        # initial gradient value
        df0 = self.df(self.f, x)
        # initial function value after step
        f1 = self.f(x + a * d)
        # loop until Armijo condition is satisfied
        while f1 > f0 + c * a * df0.T @ d:
            # update step size
            a = 0.5 * a
            # update function value after step
            f1 = self.f(x + a * d)
        return a

    def beta(self, x, x_old):
        return (self.df(self.f, x).T @ self.df(self.f, x)) / (self.df(self.f, x_old).T @ self.df(self.f, x_old))

# main function
if __name__ == "__main__":
    # pass rosennbrock function with point (1.2, 1.2)
    fr = FR(lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2, lambda f, x: np.array([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)]), np.array([1.2, 1.2]), 1e-6, np.array([1, 1]))
    results, no_iter, x, grad_norm, distance_solution = fr.run()
    print("Rosennbrock function with point (1.2, 1.2)")
    print("Number of iterations: ", no_iter)
    print("Solution: ", x)
    print("Gradient norm: ", grad_norm)
    print("Distance to solution: ", distance_solution)
    print("#############################################")
    # pass rosennbrock function with point (-1.2, 1)
    fr = FR(lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2, lambda f, x: np.array([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)]), np.array([-1.2, 1]), 1e-6, np.array([1, 1]))
    results, no_iter, x, grad_norm, distance_solution = fr.run()
    print("Rosennbrock function with point (-1.2, 1)")
    print("Number of iterations: ", no_iter)
    print("Solution: ", x)
    print("Gradient norm: ", grad_norm)
    print("Distance to solution: ", distance_solution)
    print("#############################################")
    # pass rosennbrock function with point (0.2, 0.8)
    fr = FR(lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2, lambda f, x: np.array([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)]), np.array([0.2, 0.8]), 1e-6, np.array([1, 1]))
    results, no_iter, x, grad_norm, distance_solution = fr.run()
    print("Rosennbrock function with point (0.2, 0.8)")
    print("Number of iterations: ", no_iter)
    print("Solution: ", x)
    print("Gradient norm: ", grad_norm)
    print("Distance to solution: ", distance_solution)
    print("#############################################")

    # pass f(x) = 150 * (x1*x2) ** 2 + (0.5 * x1 + 2 * x2 - 2) ** 2 with point (-0.2, 1.2)
    fr = FR(lambda x: 150 * (x[0] * x[1]) ** 2 + (0.5 * x[0] + 2 * x[1] - 2) ** 2, lambda f, x: np.array([(600 * x[0] * x[1] ** 2 + x[0] + 4*x[1] - 4)/2, 300 * x[0] ** 2 * x[1] + 2*x[0] + 8 * x[1] - 8]), np.array([-0.2, 1.2]), 1e-6, np.array([0, 1]))
    results, no_iter, x, grad_norm, distance_solution = fr.run()
    print("f(x) = 150 * (x1*x2) ** 2 + (0.5 * x1 + 2 * 2* x2 - 2) ** 2 with point (-0.2, 1.2)")
    print("Number of iterations: ", no_iter)
    print("Solution: ", x)
    print("Gradient norm: ", grad_norm)
    print("Distance to solution: ", distance_solution)
    print("#############################################")
    # pass f(x) = 150 * (x1*x2) ** 2 + (0.5 * x1 + 2 * x2 - 2) ** 2 with point (3.8, 0.1)
    fr = FR(lambda x: 150 * (x[0] * x[1]) ** 2 + (0.5 * x[0] + 2 * x[1] - 2) ** 2, lambda f, x: np.array([(600 * x[0] * x[1] ** 2 + x[0] + 4*x[1] - 4)/2, 300 * x[0] ** 2 * x[1] + 2*x[0] + 8 * x[1] - 8]), np.array([3.8, 0.1]), 1e-6, np.array([4, 0]))
    results, no_iter, x, grad_norm, distance_solution = fr.run()
    print("f(x) = 150 * (x1*x2) ** 2 + (0.5 * x1 + 2 * 2* x2 - 2) ** 2 with point (3.8, 0.1)")
    print("Number of iterations: ", no_iter)
    print("Solution: ", x)
    print("Gradient norm: ", grad_norm)
    print("Distance to solution: ", distance_solution)
    print("#############################################")
    # pass f(x) = 150 * (x1*x2) ** 2 + (0.5 * x1 + 2 * x2 - 2) ** 2 with point (1.9, 0.6)
    fr = FR(lambda x: 150 * (x[0] * x[1]) ** 2 + (0.5 * x[0] + 2 * x[1] - 2) ** 2, lambda f, x: np.array([(600 * x[0] * x[1] ** 2 + x[0] + 4*x[1] - 4)/2, 300 * x[0] ** 2 * x[1] + 2*x[0] + 8 * x[1] - 8]), np.array([1.9, 0.6]), 1e-6, np.array([4, 0]))
    print("f(x) = 150 * (x1*x2) ** 2 + (0.5 * x1 + 2 * 2* x2 - 2) ** 2 with point (1.9, 0.6)")
    print("Number of iterations: ", no_iter)
    print("Solution: ", x)
    print("Gradient norm: ", grad_norm)
    print("Distance to solution: ", distance_solution)
    print("#############################################")

    print("USING APPROXIMATED GRADIENT")
    print("#############################################")
    # pass rosennbrock function with point (1.2, 1.2)
    f = lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    pr = FR(f, approximate_gradient, np.array([1.2, 1.2]), 1e-6, np.array([1, 1]))
    results, no_iter, x, grad_norm, distance_solution = pr.run()
    print("Rosennbrock function with point (1.2, 1.2)")
    print("Number of iterations: ", no_iter)
    print("Solution: ", x)
    print("Gradient norm: ", grad_norm)
    print("Distance to solution: ", distance_solution)
    print("#############################################")
    # pass rosennbrock function with point (0.2, 0.8)
    pr = FR(lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2, approximate_gradient, np.array([0.2, 0.8]), 1e-6,
            np.array([1, 1]))
    results, no_iter, x, grad_norm, distance_solution = pr.run()
    print("Rosennbrock function with point (0.2, 0.8)")
    print("Number of iterations: ", no_iter)
    print("Solution: ", x)
    print("Gradient norm: ", grad_norm)
    print("Distance to solution: ", distance_solution)
    print("#############################################")
    # pass f(x) = 150 * (x1*x2) ** 2 + (0.5 * x1 + 2 * x2 - 2) ** 2 with point (-0.2, 1.2)
    pr = FR(lambda x: 150 * (x[0] * x[1]) ** 2 + (0.5 * x[0] + 2 * x[1] - 2) ** 2, approximate_gradient,
            np.array([-0.2, 1.2]), 1e-6, np.array([0, 1]))
    results, no_iter, x, grad_norm, distance_solution = pr.run()
    print("f(x) = 150 * (x1*x2) ** 2 + (0.5 * x1 + 2 * 2* x2 - 2) ** 2 with point (-0.2, 1.2)")
    print("Number of iterations: ", no_iter)
    print("Solution: ", x)
    print("Gradient norm: ", grad_norm)
    print("Distance to solution: ", distance_solution)
    print("#############################################")
    # pass f(x) = 150 * (x1*x2) ** 2 + (0.5 * x1 + 2 * x2 - 2) ** 2 with point (3.8, 0.1)
    pr = FR(lambda x: 150 * (x[0] * x[1]) ** 2 + (0.5 * x[0] + 2 * x[1] - 2) ** 2, approximate_gradient,
            np.array([3.8, 0.1]), 1e-6, np.array([4, 0]))
    results, no_iter, x, grad_norm, distance_solution = pr.run()
    print("f(x) = 150 * (x1*x2) ** 2 + (0.5 * x1 + 2 * 2* x2 - 2) ** 2 with point (3.8, 0.1)")
    print("Number of iterations: ", no_iter)
    print("Solution: ", x)
    print("Gradient norm: ", grad_norm)
    print("Distance to solution: ", distance_solution)
    print("#############################################")
    # pass f(x) = 150 * (x1*x2) ** 2 + (0.5 * x1 + 2 * 2* x2 - 2) ** 2 with point (1.9, 0.6)
    pr = FR(lambda x: 150 * (x[0] * x[1]) ** 2 + (0.5 * x[0] + 2 * x[1] - 2) ** 2, approximate_gradient,
            np.array([1.9, 0.6]), 1e-6, np.array([4, 0]))
    results, no_iter, x, grad_norm, distance_solution = pr.run()
    print("f(x) = 150 * (x1*x2) ** 2 + (0.5 * x1 + 2 * 2* x2 - 2) ** 2 with point (1.9, 0.6)")
    print("Number of iterations: ", no_iter)
    print("Solution: ", x)
    print("Gradient norm: ", grad_norm)
    print("Distance to solution: ", distance_solution)
    print("#############################################")

