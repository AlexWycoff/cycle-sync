import numpy as np
from scipy import sparse
from .cvxopt_quadprog import quadprog
from .cycle_inconsistency import compute_cycle_inconsistency

def run_irls(AdjMat, tijMat, IRAABVec, Ind_i, Ind_j, tijMat3d, IndMat, TriEdge, TriK, params):
    n = AdjMat.shape[0]
    d = tijMat.shape[0]
    m = tijMat.shape[1]

    rows = np.repeat(np.arange(m), 2)
    cols = np.stack([Ind_i, Ind_j], axis=1).flatten()
    data = np.tile([-1, 1], m)

    l_mat = sparse.coo_matrix((data, (rows, cols)), shape=(m, n)).tocsr()
    Lmat = sparse.kron(l_mat, sparse.identity(d))

    A_eq = sparse.kron(sparse.csr_matrix(np.ones((1, n))), sparse.identity(d))
    b_eq = np.zeros(d)

    u_flattened = tijMat.flatten('F')
    row_idx = np.arange(d * m)
    col_idx = np.repeat(np.arange(m), d)
    Gmat = sparse.coo_matrix((u_flattened, (row_idx, col_idx)), shape=(d * m, m))
    Mmat = sparse.hstack([Lmat, -Gmat]).tocsr()

    f = np.zeros(d * n + m)
    A_ineq = sparse.hstack([sparse.csr_matrix((m, d * n)), -sparse.identity(m)]).tocsr()
    b_ineq = -np.ones(m)

    A_eq_full = sparse.hstack([A_eq, sparse.csr_matrix((d, m))]).tocsr()

    logw = -params['tau1'] * IRAABVec
    costVec = np.zeros(params['WLSiters'])
    statusList = []

    for it in range(1, params['WLSiters'] + 1):
        wVec = np.exp(logw - np.max(logw))
        W = sparse.diags(np.kron(wVec, np.ones(d)))

        H = (Mmat.T @ W @ Mmat).tocsc()
        H = H + params['ridge'] * H.diagonal().mean() * sparse.identity(H.shape[0])

        sol, status = quadprog(H, f, A_ineq, b_ineq, A_eq_full, b_eq, params['tolQuad'], params['maxitQuad'])
        statusList.append(status)
        t_est = sol[:d * n].reshape(n, d).T
        alph = sol[d * n:]

        residual_vec = (Mmat @ sol).reshape(m, d).T
        rVec = np.sqrt(np.sum(residual_vec**2, axis=0))
        costVec[it - 1] = np.sum(wVec * rVec**2)

        sVec = compute_cycle_inconsistency(t_est, rVec, tijMat, tijMat3d, TriEdge, TriK, Ind_i, Ind_j, IndMat, params)

        lam = params['flam'](it)
        hVec = (1 - lam) * rVec + lam * sVec
        logw = -params['tau4'] * hVec - np.log(hVec + params['delt'])

    wVec = np.exp(logw - np.max(logw))

    return t_est, alph, {'wVec': wVec, 'rVec': rVec, 'sVec': sVec, 'costVec': costVec, 'statusList': statusList}
