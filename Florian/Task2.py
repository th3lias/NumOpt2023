import math

import numpy as np
import matplotlib.pyplot as plt

from NewtonsMethod import NewtonsMethod
from SteepestDescent import SteepestDescent


def f1(x):
    return np.sin(x)


def f1_str():
    return "sin(x)"


def df1(x):
    return np.cos(x)


def df1_str():
    return "cos(x)"


def ddf1(x):
    return -np.sin(x)


df1.__str__ = df1_str
ddf1.__str__ = df1_str

f1.__str__ = f1_str()


def f2(x):
    return np.cos(x)


def f2_str():
    return "cos(x)"


f2.__str__ = f2_str


def f3(x):
    return 3 * x ** 2 + 10 * x + 5


def f3_str():
    return "3 * x**2 + 10*x + 5"


f3.__str__ = f3_str


class Task2:
    def __init__(self, f, m, q, n):
        self.jacobian = None
        self.f = f
        self.r = None
        self.data_points = []
        self.poly = None
        self.generate_data_and_polynomials(m, q, n)

    def generate_data_and_polynomials(self, m, q, n):
        self.generate_data(m, q)
        self.poly = self.generate_polynomial(m, q, n)
        self.r = self.create_residuals(m, q, n)
        self.jacobian = self.generate_jacobian(m, q, n)

    def generate_data(self, m, q):
        # generate data points uniformly distributed in [-q, q]
        self.data_points = []
        for i in range(m):
            a = np.random.uniform(-q, q)
            b = self.f(a)
            self.data_points.append((a, b))

    def generate_jacobian(self, m, q, n):
        jacobian = np.zeros((m, n + 1))
        for i in range(len(self.data_points)):
            a, b = self.data_points[i]
            for j in range(n+1):
                jacobian[i][j] = a ** j
        return jacobian

    def generate_polynomial(self, m, q, n):
        # This is only a function, so a I can compare the results with the optimal polynomial
        a = np.array([a for a, b in self.data_points])
        b = np.array([b for a, b in self.data_points])
        return np.polyfit(a, b, n)

    def function(self, x):
        residual_values = np.array([r(x) for r in self.r])
        return 1 / 2 * np.sum(np.power(residual_values, 2))

    def create_residuals(self, m, q, n):
        residuals = []
        for i, (a, b) in enumerate(self.data_points):
            c = np.zeros((n + 1,))
            for j in range(n+1):
                c[j] = a ** j
            residuals.append(lambda x, c=c, b=b: c.T @ x - b)
        return residuals

    def gradient(self, x):
        residual_values = np.array([r(x) for r in self.r])
        return self.jacobian.T @ residual_values

    def hessian(self,x):
        return np.matmul(self.jacobian.T, self.jacobian)

    def steepest_descent(self, x0, epsilon=1e-6, max_iterations=10000):
        return SteepestDescent(self.function, self.gradient, x0, 0.5, epsilon, max_iterations)

    def newton(self, x0, epsilon=1e-6, max_iterations=10000):
        return NewtonsMethod(1,self.function, self.gradient, self.hessian, x0, epsilon, max_iterations)

def func_sin(x, n):
    sin_approx = 0
    for i in range(n):
        coef = (-1)**i
        num = x**(2*i+1)
        denom = math.factorial(2*i+1)
        sin_approx += ( coef ) * ( (num)/(denom) )
    return sin_approx
if __name__ == "__main__":
    parameters_list = [(f1, 50, 3, 5), (f1, 100, 2, 7),(f1, 150, 4, 3),(f1, 10, 3, 3),(f1, 100, 1.5, 5)]
    for i, (f, m, q, n) in enumerate(parameters_list):
        t = Task2(f1, m, q, n)

        print("Function :", f.__str__)
        print("Degree: {}".format(n))
        print("List of data points: {}".format(t.data_points))
        sd = t.steepest_descent(np.zeros((n + 1,)))
        sum_iterations_steepest_descent = 0
        result, results, iterations_steepest_descent, _ = sd.run()
        sum_iterations_steepest_descent += iterations_steepest_descent
        difference = np.linalg.norm(t.poly - np.flip(result))
        counter = 0
        while np.linalg.norm(t.gradient(result)) > 1e-6:
            sd = t.steepest_descent(result)
            result, results, iterations_steepest_descent, _ = sd.run()
            #print("gradient norm: {}".format(np.linalg.norm(t.gradient(result))))
            new_difference = np.linalg.norm(t.poly - np.flip(result))
            #print("Difference between optimal polynomial and approximated polynomial: {}".format(new_difference))
            if new_difference < difference:
                counter += 1
            else:
                counter = 0
            sum_iterations_steepest_descent += iterations_steepest_descent

            if counter >= 5 or new_difference < 1e-6:
                break
            difference = new_difference
        result = np.flip(result)
        print("Approximated parameters of polynomial from Steepest Descent: {}", result)
        print("Iterations Steepest Descent: {}".format(sum_iterations_steepest_descent))

        x = np.linspace(-q, q, 1000)
        y = np.sin(x)
        t_sin = [func_sin(angle, n) for angle in x]
        fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(10, 10))
        ax[0].plot(x, y, label="sin(x)")
        ax[0].scatter([a for a, b in t.data_points], [b for a, b in t.data_points], label="data points", color="green")

        ax[1].scatter([a for a, b in t.data_points], [b for a, b in t.data_points], label="data points", color="green")
        ax[1].plot(x, np.polyval(result, x), label="approximated polynomial steepest descent")

        ax[2].scatter([a for a, b in t.data_points], [b for a, b in t.data_points], label="data points", color="green")
        ax[2].plot(x, t_sin, label="taylor expansion")

        ax[0].set_title("sin(x)")
        ax[1].set_title("Steepest Descent")
        ax[2].set_title("Taylor Expansion")
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        ax[2].legend()
        # set ylim for subplots
        ax[0].set_ylim(-1.5, 1.5)
        ax[1].set_ylim(-1.5, 1.5)
        ax[2].set_ylim(-1.5, 1.5)
        #plt.show()

        newton = t.newton(np.zeros((n + 1,)))
        result_newton, iterations_newton = newton.run_multivariate()
        # reverse the order of the coefficients
        result_newton = np.flip(result_newton)
        print("Approximated parameters of polynomial from Newton: {}".format(result_newton))
        print("Iterations Newton: {}".format(iterations_newton))
        ax[3].scatter([a for a, b in t.data_points], [b for a, b in t.data_points], label="data points", color="green")
        ax[3].plot(x, np.polyval(result_newton, x), label="approximated polynomial newton")
        ax[3].set_title("Newton")
        ax[3].legend()
        ax[3].set_ylim(-1.5, 1.5)
        fig.suptitle("Problem {}, n= {}, m= {}, q= {}".format(i+1, n, m, q))
        fig.tight_layout()
        plt.show()
