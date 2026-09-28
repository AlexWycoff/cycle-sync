import numpy as np

def build_edge_index_matrices(AdjMat, tijMat):
    '''
    Edge l connects (Ind_i[l], Ind_j[l]) with Ind_i > Ind_j (lower triangle, row-major).
    Convention: tijMat[:, l] is the unit direction of t_j - t_i.
    tijMat3d[:, a, b] is the observed direction of t_a - t_b, for any measured pair.
    IndMat[a, b] = IndMat[b, a] = l.
    '''
    tril_mask = np.tril(np.ones_like(AdjMat), -1).astype(bool)
    Ind_i, Ind_j = np.nonzero(tril_mask * (AdjMat != 0))

    m = tijMat.shape[1]
    n = AdjMat.shape[0]
    if len(Ind_i) != m:
        raise ValueError(f"AdjMat has {len(Ind_i)} edges but tijMat has {m} columns.")

    tijMat3d = np.zeros((3, n, n))
    IndMat = np.zeros((n, n), dtype=int)

    for l in range(m):
        i = Ind_i[l]
        j = Ind_j[l]
        tijMat3d[:, j, i] = tijMat[:, l]
        tijMat3d[:, i, j] = -tijMat[:, l]
        IndMat[i, j] = l
        IndMat[j, i] = l

    return Ind_i, Ind_j, tijMat3d, IndMat

def tijmat_from_dict(AdjMat, t_dict):
    '''
    Build tijMat in the edge order Cycle_Sync expects.
    t_dict[(a, b)] must be the world-frame direction of t_b - t_a (either orientation may be given).
    '''
    tril_mask = np.tril(np.ones_like(AdjMat), -1).astype(bool)
    Ind_i, Ind_j = np.nonzero(tril_mask * (AdjMat != 0))

    tijMat = np.zeros((3, len(Ind_i)))
    for l in range(len(Ind_i)):
        i = int(Ind_i[l])
        j = int(Ind_j[l])
        if (i, j) in t_dict:
            d = np.asarray(t_dict[(i, j)], dtype=float)
        else:
            d = -np.asarray(t_dict[(j, i)], dtype=float)
        tijMat[:, l] = d / np.linalg.norm(d)

    return tijMat
