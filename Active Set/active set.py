# active set method for convex QPs.

import numpy as np

def active_set(Q, c, A, b, x0, max_iter=100):
    """
    Solve the following QP problem using active set method:
    min 1/2 x^T Q x + c^T x
    s.t. Ax = b, x >= 0
    """
    # Initialization
    x = x0
    m, n = A.shape
    active_set = np.arange(n)
    # Iteration
    for i in range(max_iter):
        # Solve the subproblem
        Q_active = Q[np.ix_(active_set, active_set)]
        c_active = c[active_set]
        x_active = x[active_set]
        x_new = np.linalg.solve(Q_active, -c_active)
        # Check the KKT conditions
        if np.all(x_new >= 0):
            # Check the equality constraints
            if np.linalg.norm(A.dot(x_new) - b) < 1e-6:
                return x_new
            else:
                # Find the index of the most violated constraint
                violation = A.dot(x_new) - b
                index = np.argmax(np.abs(violation))
                # Add the constraint to the active set
                active_set = np.append(active_set, index)
        else:
            # Find the index of the most violated constraint
            violation = A.dot(x_new) - b
            index = np.argmax(np.abs(violation))
            # Remove the constraint from the active set
            active_set = np.delete(active_set, index)
    return x

if __name__ == '__main__':
    # Generate data
    np.random.seed(0)
    m, n = 10, 5
    Q = np.random.randn(n, n)
    Q = Q.T.dot(Q)
    c = np.random.randn(n)
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    x0 = np.random.randn(n)
    # Solve the problem
    x = active_set(Q, c, A, b, x0)
    # Print the results
    print('The solution is: {}'.format(x))
    print('The optimal value is: {}'.format(0.5 * x.dot(Q).dot(x) + c.dot(x)))

