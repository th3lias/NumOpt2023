import numpy as np
from numpy import eye

from SteepestDescent import SteepestDescent
from scipy.linalg import hilbert

class Task3:
    def __init__(self):
        self.n = [5, 8, 12, 20, 30]
        self.functions = [self.generatefunction(n) for n in [5, 8, 12, 20, 30]]
        self.gradients = [self.generateGradient(n) for n in [5, 8, 12, 20, 30]]
        self.solutions = self.calculateSolutions()
        self.eigenvaluesInfo = self.calculateEigenvaluesAndConditionNumber()

    def generatefunction(self, n):
        # function should have following form: f(x) = (1/2)*x.T*Q*x - b.T*x, where Q is the Hilbert matrix and b is the vector of ones
        return lambda x: (1/2)*np.dot(np.dot(x.T, self.generateQ(n)), x) - np.dot(np.ones(n).T, x)

    def generateQ(self, n):
        # generate Hilbert matrix
        return hilbert(n)
    def generateGradient(self, n):
        # generate gradient
        return lambda x: np.dot(self.generateQ(n), x) - np.ones(n)

    def calculateSolutions(self):
        # solve linear system of Qx = b
        return [np.linalg.solve(self.generateQ(n), np.ones(n)) for n in [5, 8, 12, 20, 30]]

    def calculateEigenvaluesAndConditionNumber(self):
        # calculate eigenvalues and condition number of Q, also calculate (lambda_n - lambda_1) / (lambda_n + lambda_1), where lambda_n and lambda_1 are the largest and smallest eigenvalues
        return [(np.linalg.eigvals(self.generateQ(n)), np.linalg.cond(self.generateQ(n)), (np.linalg.eigvals(self.generateQ(n))[-1] - np.linalg.eigvals(self.generateQ(n))[0]) / (np.linalg.eigvals(self.generateQ(n))[-1] + np.linalg.eigvals(self.generateQ(n))[0])) for n in [5, 8, 12, 20, 30]]





# main function
if __name__ == '__main__':
    task3 = Task3()
    eigenvaluesAndConditionNumber = task3.eigenvaluesInfo
    for i in range(len(task3.functions)):
        f = task3.functions[i]
        df = task3.gradients[i]
        print("Problem: Hilbert Matrix with Dimension: {}".format(task3.n[i]))
        #print(task3.generateQ(task3.n[i]))
        print("Eigenvalues of Hilbert Matrx: {}".format(eigenvaluesAndConditionNumber[i][0]))
        print("Condition Number of Hilbert Matrix: {}".format(eigenvaluesAndConditionNumber[i][1]))
        print("Number (λn − λ1)/(λn +λ1) of Hilbert Matrix: {}".format(eigenvaluesAndConditionNumber[i][2]))
        sd = SteepestDescent(f, df, np.zeros(task3.n[i]), 0.5, 1e-6, 10000, True, task3.solutions[i], eigenvaluesAndConditionNumber[i], task3.n[i])
        result,results, iterations, count_inequality = sd.run()

        print("Actual solution: {}".format(task3.solutions[i]))
        # print solution (last n elements) from results
        solution = result
        print("Approximated Solution: {}".format(solution))
        print("Gradient norm Value: {}".format(np.linalg.norm(df(solution))))
        print("Distance from solution: {}".format(np.linalg.norm(solution - task3.solutions[i])))
        # print iterations
        print("Iterations: {}".format(iterations))
        print("Number of times the Inequality 3.29 holds: {}".format(count_inequality))
        print("#############################################")