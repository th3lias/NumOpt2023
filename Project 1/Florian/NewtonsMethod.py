import numpy as np
from scipy.optimize import newton


class NewtonsMethod:
    def __init__(self, alpha, f, df, ddf, x0, tol=1e-6, maxiter=100, c=1e-3):
        self.alpha = alpha
        self.f = f
        self.df = df
        self.ddf = ddf
        self.x0 = x0
        self.tol = tol
        self.maxiter = maxiter
        self.c = 1e-3

    def run(self):
        x = self.x0
        i = 0
        while i < self.maxiter:
            alpha = self.line_search(x)
            xnew = x - alpha * ( self.df(x) / self.ddf(x) )

            x = xnew
            if np.linalg.norm(self.df(x)) <= self.tol:
                return (xnew, i)

            i+=1
        return (x,i)

    def line_search(self,x):
        alpha = 1
        f_x = self.f(x)
        gradient_square_norm = np.linalg.norm(self.df(x)) ** 2

        # Until the sufficient decrease condition is met
        while self.f(x - alpha * self.df(x)) >= (f_x - self.c * alpha * gradient_square_norm):
            # Update the stepsize (backtracking)
            alpha /= 2
        return alpha


    def run_multivariate(self):
        x = self.x0
        i = 0
        while i <= self.maxiter:
            alpha = self.line_search_multivariate(x)
            x -= alpha * (np.linalg.inv(self.ddf(x)) @ self.df(x))
            i += 1
            if np.linalg.norm(self.df(x)) < self.tol:
                break

        return x, i

    def line_search_multivariate(self, x):
        alpha = self.alpha
        while self.f(x - alpha * np.linalg.inv(self.ddf(x)) @ self.df(x)) > self.f(x) + self.c * alpha * self.df(x).T @ np.linalg.inv(self.ddf(x)) @ self.df(x):
            alpha *= 0.5
        return alpha