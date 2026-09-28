import numpy as np
from scipy import sparse
from cvxopt import matrix, spmatrix, solvers

def to_cvxopt(M):
    M = sparse.coo_matrix(M)
    return spmatrix(M.data.astype(float).tolist(), M.row.tolist(), M.col.tolist(), size=M.shape)

def quadprog(H, f, A, b, Aeq=None, beq=None, tol=1e-10, maxiters=200):
    H = sparse.csc_matrix(0.5 * (H + H.T))
    P = to_cvxopt(H)
    q = matrix(f.astype(float))

    G = to_cvxopt(A) if A is not None else None
    h = matrix(b.astype(float)) if b is not None else None

    Aeq_mat = to_cvxopt(Aeq) if Aeq is not None else None
    beq_mat = matrix(beq.astype(float)) if beq is not None else None

    solvers.options['show_progress'] = False
    solvers.options['abstol'] = tol
    solvers.options['reltol'] = tol
    solvers.options['feastol'] = tol
    solvers.options['maxiters'] = maxiters
    sol = solvers.qp(P, q, G, h, Aeq_mat, beq_mat)
    return np.array(sol['x']).flatten(), sol['status']
