import math
import sys

import numpy
import numpy as np
from numpy.testing import assert_allclose, assert_equal, \
    assert_almost_equal, assert_raises
from scipy.optimize import linprog

from activeset import ActiveSet
class ConstrainedLS(ActiveSet):
    """
    An active set method for constrained least squares

    Example:
        A = [[0.9501, 0.7620, 0.6153, 0.4057],
             [0.2311, 0.4564, 0.7919, 0.9354],
             [0.6068, 0.0185, 0.9218, 0.9169],
             [0.4859, 0.8214, 0.7382, 0.4102],
             [0.8912, 0.4447, 0.1762, 0.8936]]
        b = [0.0578, 0.3528, 0.8131, 0.0098, 0.1388]
        Ci = [[0.2027, 0.2721, 0.7467, 0.4659],
              [0.1987, 0.1988, 0.4450, 0.4186],
              [0.6037, 0.0152, 0.9318, 0.8462]]
        di = [0.5251, 0.2026, 0.6721]
        Ce = [[3, 5, 7, 9]]
        de = [4]
        cu = [2, 2, 2, 2]
        cl = [-0.1, -0.1, -0.1, -0.1]

        cls = ConstrainedLS(atol=1e-7)
        x, scr, nit = cls(A, b, Ce, de, Ci, di, cu, cl)

        print x
        [[-0.10000], [-0.10000], [0.15991], [0.40896]]

    """

    # objective function Hessians:
    #   H = 2 * A.T * A
    #   h = 2 * A.T * b
    def _calc_Hessians(self):
        At = self.A.T
        self.H = 2 * np.dot(At, self.A)
        self.h = 2 * np.dot(At, self.b)


    # score the solution x:
    #  f(x) = || A * x - b || ** 2
    def _calc_objective(self, x):
        return np.sum((np.dot(self.A, x) - self.b)**2)


    # command requires objective target vector
    def __call__(self, A, b, Ce=[], de=[], Ci=[], di=[], cu=[], cl=[], x0=[]):
        return self.run(A=A, b=b, Ce=Ce, de=de, Ci=Ci, di=di,
                        cu=cu, cl=cl, x0=x0)

def convert_to_standard_QP(M, y, x):
    m, n = M.shape[0], 2 * M.shape[0]
    # objective function = 1/2 * x^T * G * x + c^T * x = 1/2(x^T M^T M x - 2 y^T M x (+y^T y))
    q_x = (1 / 2) * (x @ M.T @ M @ x - 2 * y.T @ M @ x + y @ y)
    G = M.T @ M
    c = 2 * M.T @ y

    # constraints = 1*x<=1
    ai = np.zeros((m, n))
    bi = np.ones(m)
    ai[:, :n // 2] = np.eye(m)
    return G, c, ai, bi

# input validation and unit testing for ActiveSet base
# class. unit tests verified against Matlab lsqlin().

def simplextableau(c, A, b):
    tol = 1e-14
    res = ''
    c = numpy.array(c)
    A = numpy.array(A)
    b = numpy.array(b)
    m, n = A.shape
    Ni = numpy.array(range(n - m))
    Bi = numpy.array(range(m)) + n - m
    x = numpy.zeros((n, 1))
    xB = numpy.array(b)
    combs = math.factorial(n) / (math.factorial(m) * math.factorial(n - m))
    for k in range(4 * m):
        l = numpy.linalg.solve(A[:, Bi], c[Bi])
        sN = c[Ni] - numpy.matmul(numpy.transpose(A[:, Ni]), l)
        sm = numpy.min(sN)
        qq = numpy.argmin(sN)
        q = Ni[qq]
        xm = numpy.min(xB)
        p = numpy.argmin(xB)
        mu = numpy.minimum(sm, xm)
        if mu >= -tol:
            res = 'solution found'
            break
        if mu == sm:
            a = A[:, q]
            p = numpy.argmax(a)
            phi = A[p, q]
            if phi <= tol:
                res = 'primal infeasible or unbounded'
                break
        else:
            sigma = A[p, Ni]
            qq = numpy.argmin(sigma)
            q = Ni[qq]
            phi = A[p, q]
            if phi >= -tol:
                res = 'duel infeasible or unbounded'
        xB[p] = xB[p] / phi
        A[p, :] = A[p, :] / phi
        oi = (range(m))
        oi.remove(p)
        xB[oi] = xB[oi] - A[oi, q] * xB[p]
        A[oi, :] = A[oi, :] - numpy.multiply.outer(A[oi, q], A[p, :])
        Ni[Ni == q] = Bi[p]
        Bi[p] = q
    x[Bi, 0] = xB
    opt = numpy.dot(c, x)
    if len(res) == 0:
        res = 'Iterations exceed maximum number, cycling may be happening.'
    return x, opt[0], res

def find_feasible_point(A, b):
    n = A.shape[1]  # Dimension of variables

    # Formulate the Phase I problem
    c_phase1 = np.zeros(n+1)  # Coefficients for the additional variable
    c_phase1[-1] = -1  # Minimize the additional variable

    A_phase1 = np.hstack((A, np.ones((A.shape[0], 1))))
    b_phase1 = b

    # Solve the Phase I problem
    result_phase1 = linprog(c_phase1, A_ub=A_phase1, b_ub=b_phase1)

    if result_phase1.success:
        feasible_point = result_phase1.x[:-1]  # Extract the feasible point
        return feasible_point
    else:
        print("No feasible point found.")
        return None
def main():
    cls = ConstrainedLS(atol=1e-7)

    for m in range(5):
        M = np.random.randn(m+1, 2 * (m + 1)) + 1
        # compute eigenvalues from M and choose as y
        y = np.linalg.eig(M.T @ M)[0][:m+1]
        i = 0
        while i <= 3:
            x = np.random.randn(2 * (m+1))
            G, c, ai, bi = convert_to_standard_QP(M, y, x)

            try:
                x = find_feasible_point(ai, bi)
                ai = list(ai)
                bi = list(bi)
                x0 = list(x)
                c = list(c)
                G = list(G)
                x, scr, nit = cls(G, c, Ci=ai, di=bi, x0=x0)
                i = i + 1
            except ValueError:
                print("Infeasible point")

            print("x(final) = \n", x)

if __name__ == "__main__":
    sys.exit(int(main() or 0))