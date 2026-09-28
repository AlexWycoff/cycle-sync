import numpy as np

def compute_cycle_inconsistency(t_est, rVec, tijMat, tijMat3d, TriEdge, TriK, Ind_i, Ind_j, IndMat, params):
    m = tijMat.shape[1]

    dVec = np.linalg.norm(t_est[:, Ind_j] - t_est[:, Ind_i], axis=0)

    i = Ind_i[TriEdge]
    j = Ind_j[TriEdge]
    ki = IndMat[i, TriK]
    kj = IndMat[j, TriK]

    t_cycle = dVec[TriEdge] * tijMat[:, TriEdge] + dVec[ki] * tijMat3d[:, i, TriK] + dVec[kj] * tijMat3d[:, TriK, j]
    d_ijk = np.linalg.norm(t_cycle, axis=0)

    expo = -params['beta'] * (rVec[ki] + rVec[kj])
    expo_max = np.full(m, -np.inf)
    np.maximum.at(expo_max, TriEdge, expo)
    w_ijk = np.exp(expo - expo_max[TriEdge])

    Z = np.zeros(m)
    num = np.zeros(m)
    np.add.at(Z, TriEdge, w_ijk)
    np.add.at(num, TriEdge, w_ijk * d_ijk)

    sVec = rVec.copy()
    has_tri = Z > 0
    sVec[has_tri] = num[has_tri] / Z[has_tri]

    return sVec
