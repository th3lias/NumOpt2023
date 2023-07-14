# Try newton's method for optimization with function of 2 variables and degree 4

import numpy as np

from NewtonsMethod import NewtonsMethod


# a1, a2, b1, b2, c1, c2, d1, d2, e1, e2, f1, f2,
def generate_f(parameters, x1, x2):
    q1 = parameters["a1"] * x1 ** 2 + parameters["b1"] * x2 ** 2 + parameters["c1"] * x1 * x2 + parameters["d1"] * x1 + parameters["e1"] * x2 + parameters["f1"]
    q2 = parameters["a2"] * x1 ** 2 + parameters["b2"] * x2 ** 2 + parameters["c2"] * x1 * x2 + parameters["d2"] * x1 + parameters["e2"] * x2 + parameters["f2"]
    return q1 ** 2 + q2 ** 2

def generate_f2(parameters, x1, x2):
    # l(x,y) = x^4 + y^4 - 4x^3 - 4y^3 + 6x^2 + 6y^2 - 8x^2y + 5
    return x1 ** 4 + x2 ** 4 - 4 * x1 ** 3 - 4 * x2 ** 3 + 6 * x1 ** 2 + 6 * x2 ** 2 - 8 * x1 ** 2 * x2 + 5

class Task4:
    def __init__(self, parameters):
        self.parameters = parameters

    def f(self, x):
        return generate_f(self.parameters, x[0], x[1])

    def grad(self, x):
        h = 1e-4
        grad = np.zeros_like(x)
        for idx in range(x.size):
            tmp_val = x[idx]
            x[idx] = float(tmp_val) + h
            fxh1 = self.f(x)
            x[idx] = tmp_val - h
            fxh2 = self.f(x)
            grad[idx] = (fxh1 - fxh2) / (2 * h)
            x[idx] = tmp_val
        return grad

    def hessian(self, x):
        h = 1e-4
        hessian = np.zeros((x.size, x.size))
        for idx in range(x.size):
            tmp_val = x[idx]
            x[idx] = float(tmp_val) + h
            fxh1 = self.grad(x)
            x[idx] = tmp_val - h
            fxh2 = self.grad(x)
            hessian[idx] = (fxh1 - fxh2) / (2 * h)
            x[idx] = tmp_val
        return hessian

    def run(self, init_x, alpha=1, step_num=100000, tolerance=1e-6):
        newton = NewtonsMethod(alpha, self.f, self.grad, self.hessian, init_x, tolerance, step_num)
        x, i = newton.run_multivariate()
        return x, i



if __name__ == "__main__":
    parameters_list = [{"a1": -10, "a2": 0, "b1": 0, "b2": 0, "c1": 0, "c2": 0, "d1": 0, "d2": -1, "e1": 10, "e2": 0, "f1": 0, "f2": 1},
                       {"a1": 0, "a2": 0, "b1": 1, "b2": 0, "c1": 0, "c2": 0, "d1": 1, "d2": 0, "e1": 0, "e2": 0, "f1": 0, "f2": 0},
                       {"a1": 1, "a2": 0, "b1": -1, "b2": 0, "c1": 0, "c2": 0, "d1": 0, "d2": 0, "e1": 0, "e2": 0, "f1": 1, "f2": 0},
                       {"a1": 1, "a2": 1, "b1": -1, "b2": 1, "c1": 0, "c2": 1, "d1": 0, "d2": 0, "e1": 0, "e2": 0, "f1": 0, "f2": 1},
                       {"a1": 1, "a2": 0, "b1": 0, "b2": 1, "c1": -1, "c2": 2, "d1": 0, "d2": 0, "e1": -1, "e2": 0, "f1": 0, "f2": 0}]
    solution_list = [np.array([1.0, 1.0]), np.array([(-i**2,i) for i in range(-10,10)]), np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0])]
    init_x_list = [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [10.0, 10.0]]
    for i, parameters in enumerate(parameters_list):
        task4 = Task4(parameters)
        print("Parameters: {}".format(parameters))
        print("Local Minima: {}".format(solution_list[i]))

        solution, iterations = (task4.run(np.array(init_x_list[i])))
        print("Solution from Newton: {}".format(solution))
        print("Gradient Value at solution: {}".format(task4.grad(solution)))
        closest_solution = solution_list[i][np.argmin([np.linalg.norm(solution - sol) for sol in solution_list[i]])]
        print("Distance from closest solution: {}".format(np.linalg.norm(solution - closest_solution)))
        print("Iterations: {}".format(iterations))

