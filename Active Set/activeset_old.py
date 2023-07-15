import math
import time

import numpy
import numpy as np


def activesetquadraticbigm(x, d, G, ce, be, ci, bi, W=[]):
    tol = 1e-8
    M = 1e0
    reas = ''
    w, v = numpy.linalg.eig(G)
    hasnegigenval = numpy.any(w < 0)
    n = len(x)
    ni = len(ci)
    ne = len(ce)
    ce = numpy.array(ce)
    ci = numpy.array(ci)
    bi = numpy.array(bi)
    be = numpy.array(be)
    if ni == 0:
        ci = numpy.zeros((0, n))
        bi = numpy.zeros((0, 1))
    else:
        bi = numpy.reshape(bi, (ni, 1))
    if ne == 0:
        ce = numpy.zeros((0, n))
        be = numpy.zeros((0, 1))
    else:
        be = numpy.reshape(be, (ne, 1))
    eq = numpy.matmul(ce, x) - be
    iq = numpy.matmul(ci, x) - bi
    if not numpy.all(numpy.abs(eq) < tol) and numpy.all(iq > -tol):
        c = numpy.concatenate((numpy.zeros(2 * n), numpy.ones(ni + 2 * ne)))
        A = numpy.concatenate((numpy.concatenate((-ci, ci, numpy.eye(ni), numpy.zeros((ni, 2 * ne))), 1),
                               numpy.concatenate((ce, -ce, numpy.zeros((ne, ni)), numpy.eye(ne), numpy.zeros((ne, ne))),
                                                 1),
                               numpy.concatenate((-ce, ce, numpy.zeros((ne, ni + ne)), numpy.eye(ne)), 1)), 0)
        b = numpy.concatenate((-bi, be, -be))
        xx, opt, res = simplextableau(c, A, b)
        if not res == 'solution found':
            return x, [], [], 'infeasiable constraints'
            print('infeasiable constraints')
    if len(W) != ni + 2 * ne + 1:
        W = numpy.zeros(ni + 2 * ne + 1, dtype=bool)

    x.append(0.)
    cie = numpy.concatenate((numpy.concatenate((ci, numpy.ones((ni, 1))), 1),
                             numpy.concatenate((ce, numpy.ones((ne, 1))), 1),
                             numpy.concatenate((-ce, numpy.ones((ne, 1))), 1),
                             numpy.concatenate((numpy.zeros((1, n)), numpy.ones((1, 1))), 1)), 0)
    bie = numpy.concatenate((bi, be, -be, numpy.zeros((1, 1))), 0)[:, 0]
    d.append(M)
    G = numpy.concatenate((numpy.concatenate((G, numpy.zeros((n, 1))), 1),
                           numpy.concatenate((numpy.zeros((1, n)), numpy.ones((1, 1))), 1)), 0)
    x[n] = 2 * numpy.max(bie - numpy.matmul(cie, x))

    for j in range(10):
        for k in range(4 * (ni + 2 * ne + 2)):
            na = numpy.sum(W)
            A = cie[W, :]
            b = bie[W]
            c = numpy.matmul(A, x) - b
            g = d + numpy.matmul(G, x)
            K = numpy.concatenate((numpy.concatenate((G, numpy.transpose(A)), 1),
                                   numpy.concatenate((A, numpy.zeros((ne + na, ne + na))), 1)), 0)
            if len(c) == 0:
                npl = numpy.linalg.solve(K, g)
            else:
                npl = numpy.linalg.solve(K, numpy.concatenate((g, c), 0))
            p = -npl[0:n + 1]

            if hasnegigenval:
                cdirnegcurv = False
                Q, R = numpy.linalg.qr(numpy.transpose(A), 'complete')
                Z = Q[:, na:]
                if not Z.shape[1] == 0:
                    w, v = numpy.linalg.eig(numpy.matmul(numpy.matmul(numpy.transpose(Z), G), Z))
                    if numpy.min(w) < 0:
                        dd = numpy.matmul(Z, v[:, numpy.argmin(w)])
                        p = -1e4 * numpy.sign(numpy.dot(g, dd)) * dd
                        cdirnegcurv = True

            if numpy.all(numpy.abs(p) < tol):
                lambdaW = npl[n + 1:n + 1 + na]
                l = numpy.zeros(ni + 2 * ne + 1)
                l[W] = lambdaW
                if numpy.all(lambdaW >= -tol):
                    break
                else:
                    mlw = numpy.sort(lambdaW[lambdaW < 0])
                    ri = mlw[0] == l
                    if 'ac' in locals() and numpy.all(ri == ac) and len(mlw) > 1:
                        ri = mlw[numpy.ceil((len(mlw) - 1) * numpy.rand())] == l
                    W[ri] = False
            else:
                cip = numpy.matmul(cie, p)
                bicixcip = (bie - numpy.array(numpy.matmul(cie, x))) / cip
                findminof = bicixcip[numpy.invert(W) & (cip < 0)]
                if len(findminof > 0):
                    alpha = numpy.min(findminof)
                else:
                    alpha = 1
                if alpha > 1:
                    alpha = 1
                if hasnegigenval and alpha == 1 and cdirnegcurv:
                    print('Problem unbounded, did not find a constraint in a negative curvature direction')
                    return x, W, [], 'Problem unbounded, did not find a constraint in a negative curvature direction'
                x = x + alpha * p
                if alpha < 1:
                    ac = (alpha == bicixcip) & numpy.invert(W) & (cip < 0)
                    W = W | ac
        if x[n] <= tol and all(lambdaW >= -tol):
            x = x[0:n]
            l = numpy.concatenate((l[0:ni], l[ni:ni + ne] - l[ni + ne:ni + 2 * ne]))
            return x, W, l, 'Solution found'
        elif x[n] > tol:
            x[n] = 0
            M = M * 10
            d[n] = M
            x[n] = 2 * numpy.max(bie - numpy.matmul(cie, x))
        else:
            x[n] = 0
            x[0:n] = numpy.random.uniform(-.5, .5, n)
            x[n] = 2 * numpy.max(bie - numpy.matmul(cie, x))
            W = numpy.zeros(ni + 2 * ne + 1, dtype=bool)
    print('Problem unbounded, did not find a constraint in a negative curvature direction')
    return x, W, l, 'Problem unbounded, did not find a constraint in a negative curvature direction'


def activesetquadraticprogramming(x, d, G, ce, be, ci, bi, tol, W=[]):
    reas = ''
    w, v = numpy.linalg.eig(G)
    hasnegigenval = numpy.any(w < 0)
    n = len(x)
    ni = len(ci)
    ne = len(ce)
    ce = numpy.array(ce)
    ci = numpy.array(ci)
    bi = numpy.array(bi)
    be = numpy.array(be)
    if ni == 0:
        ci = numpy.zeros((0, n))
    if ne == 0:
        ce = numpy.zeros((0, n))
    eq = numpy.matmul(ce, x) - be
    iq = numpy.matmul(ci, x) - bi
    if numpy.all(numpy.abs(eq) < tol) and numpy.all(iq > -tol):
        if len(W) != ni:
            W = iq < tol
    else:
        c = numpy.concatenate((numpy.zeros(2 * n), numpy.ones(ni + 2 * ne)))
        A = numpy.concatenate((numpy.concatenate((-ci, ci, numpy.eye(ni), numpy.zeros((ni, 2 * ne))), 1),
                               numpy.concatenate((ce, -ce, numpy.zeros((ne, ni)), numpy.eye(ne), numpy.zeros((ne, ne))),
                                                 1),
                               numpy.concatenate((-ce, ce, numpy.zeros((ne, ni + ne)), numpy.eye(ne)), 1)), 0)
        b = numpy.concatenate((-bi, be, -be))
        x, opt, res = simplextableau(c, A, b)
        x = x[0:n] - x[n:2 * n]
        W = ci * x - bi < tol
        if not res == 'solution found':
            return x, W, [], 'infeasiable constraints'
            print('infeasiable constraints')
    for k in range(4 * ni + 4):
        na = numpy.sum(W)
        A = numpy.concatenate((ce, ci[W, :]), 0)
        b = numpy.concatenate((be, bi[W]), 0)
        c = numpy.matmul(A, x) - b
        g = d + numpy.matmul(G, x)
        K = numpy.concatenate(
            (numpy.concatenate((G, numpy.transpose(A)), 1), numpy.concatenate((A, numpy.zeros((ne + na, ne + na))), 1)),
            0)
        npl = numpy.linalg.solve(K, numpy.concatenate((g, c), 0))
        p = -npl[0:n]

        if hasnegigenval:
            cdirnegcurv = False
            Q, R = numpy.linalg.qr(numpy.transpose(A), 'complete')
            Z = Q[:, na:]
            if not Z.shape[1] == 0:
                w, v = numpy.linalg.eig(numpy.matmul(numpy.matmul(numpy.transpose(Z), G), Z))
                if numpy.min(w) < 0:
                    dd = numpy.matmul(Z, v[:, numpy.argmin(w)])
                    p = -1e4 * numpy.sign(numpy.dot(g, dd)) * dd
                    cdirnegcurv = True
        if numpy.all(numpy.abs(p) < tol):
            lambdaW = npl[n + ne:n + ne + na]
            l = numpy.zeros(ni)
            l[W] = lambdaW
            if numpy.all(lambdaW >= -tol):
                return x, W, l, k, 'Solution found'
            else:
                mlw = numpy.sort(lambdaW[lambdaW < 0])
                ri = mlw[0] == l
                if 'ac' in locals() and numpy.all(ri == ac) and len(mlw) > 1:
                    ri = mlw[numpy.ceil((len(mlw) - 1) * numpy.rand())] == l
                W[ri] = False
        else:
            cip = numpy.matmul(ci, p)
            bicixcip = (bi - numpy.matmul(ci, x)) / cip
            findminof = bicixcip[numpy.invert(W) & (cip < 0)]
            if len(findminof > 0):
                alpha = numpy.min(findminof)
            else:
                alpha = 1
            if alpha > 1:
                alpha = 1
            if hasnegigenval and alpha == 1 and cdirnegcurv:
                print('Problem unbounded, did not find a constraint in a negative curvature direction')
                return x, W, [], k, 'Problem unbounded, did not find a constraint in a negative curvature direction'
            x = x + alpha * p
            if alpha < 1:
                ac = (alpha == bicixcip) & numpy.invert(W) & (cip < 0)
                W = W | ac
    print('Problem unbounded, did not find a constraint in a negative curvature direction')
    return x, W, l, k, 'Problem unbounded, did not find a constraint in a negative curvature direction'


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

def convert_to_standard_QP(M, y, x):
    m, n = M.shape[0], 2 * M.shape[0]
    # objective function = 1/2 * x^T * G * x + c^T * x = 1/2*(x^T M^T M x - 2 y^T M x (+y^T y))
    q_x = (1 / 2) * (x @ M.T @ M @ x - 2 * y.T @ M @ x + y @ y)
    G = M.T @ M
    c = 2 * M.T @ y

    # constraints = 1*x<=1
    ai = np.zeros((m, n))
    bi = np.ones(m)
    ai[:, :n // 2] = np.eye(m)
    return G, c, ai, bi

if __name__ == '__main__':

    tol = 1e-8
    x = [2., 0.]
    d = [-2., -5.]
    G = [[2., 0.], [0., 2.]]
    ce = []
    be = []
    ci = [[1., -2.], [-1., -2.], [-1., 2.], [1., 0.], [0., 1.]]
    bi = [-2., -6., -2., 0., 0.]
    W = [False, False, False, False, False]
    print(activesetquadraticprogramming(x, d, G, ce, be, ci, bi, tol, W))
    print(activesetquadraticbigm(x, d, G, ce, be, ci, bi, W))


    M = numpy.array([[1., 0.], [0., 1.], [-1., 2.], [1., 0.], [0., 1.]])
    y = numpy.array([-2., -6., -2., 0., 0.])
    x = np.array([2., 0.])
    convert_to_standard_QP(M, y, x)

    # Choose 5 such matrices M and 5 vectors y with m = 1, 2, 3, 4, 5 and n = 2,4,6,8,10
    for m in range(5):
        M = np.random.randn(m+1, 2 * (m + 1)) + 1
        # compute eigenvalues from M and choose as y
        y = np.linalg.eig(M.T @ M)[0][:m+1]
        for i in range(3):
            x = np.random.randn(2 * (m+1))
            G, c, ai, bi = convert_to_standard_QP(M, y, x)
            # measure running time

            start = time.time()
            x_final, noOfIterations = (activesetquadraticbigm(list(x), list(c), list(G), [], [], list(bi), list(ai), [False] * len(bi)))
            end = time.time()
            print('No of iterations: ', noOfIterations)
            print('Time taken: ', end - start)
            print('Final iterate: ', x_final)
            print('stopping criteria: ', tol)



