import numpy as np

from scipy.linalg import hilbert
class SteepestDescent:
    def __init__(self, f, df, x0, alpha, epsilon, max_iter, count_inequality=False, solution=None, eigenvaluesInfo=None, n=None, c=1e-3):
        self.f = f
        self.df = df
        self.x0 = x0
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.count_inequality = count_inequality
        self.solution = solution
        self.eigenvaluesInfo = eigenvaluesInfo
        self.n = n
        self.c = c

    def run(self):
        alpha = self.alpha
        x = self.x0
        i = 1
        count_inequality = 0
        results = np.array([])
        results = np.append(results, x, axis=0)
        while i < self.max_iter:
            alpha = self.line_search(self.alpha, x)
            # update x
            x_new = x - alpha * self.df(x)

            # save results
            results = np.append(results, x_new)

            # update iteration
            i += 1
            if self.count_inequality:
                if ((x_new - self.solution).T @ hilbert(self.n) @ (x_new - self.solution)) <= (self.eigenvaluesInfo[2] ** 2 * ((x - self.solution).T @ hilbert(self.n) @ (x - self.solution))):
                    count_inequality += 1
            x = x_new
            # update alpha

            if np.linalg.norm(self.df(x)) <= self.epsilon:
                break
        return x,results, i, count_inequality

    def line_search(self, step, x):
        f_x = self.f(x)
        gradient_square_norm = np.linalg.norm(self.df(x)) ** 2

        # Until the sufficient decrease condition is met
        while self.f(x - step * self.df(x)) > (f_x - self.c * step * gradient_square_norm):
            # Update the stepsize (backtracking)
            step /= 2
        return step