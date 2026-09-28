import numpy as np

def sample_triangles(AdjMat, tijMat3d, Ind_i, Ind_j, params, rng):
    sinmin = params['sinmin']
    nsample = params['nsample']

    A = (AdjMat != 0)
    m = len(Ind_i)

    CoDeg = (A.astype(float) @ A.astype(float)) * A
    IndCand = np.where(CoDeg[Ind_i, Ind_j] > 0)[0]

    CoIndMat = np.zeros((nsample, m), dtype=int)
    IndPos = []

    for l in IndCand:
        i = Ind_i[l]
        j = Ind_j[l]

        CoInds = np.where(A[:, i] & A[:, j])[0]
        Xkis = tijMat3d[:, i, CoInds]
        Xjks = tijMat3d[:, j, CoInds]

        cosines = np.abs(np.sum(Xkis * Xjks, axis=0))
        good = CoInds[cosines < np.sqrt(1 - sinmin**2)]

        if good.size > 0:
            CoIndMat[:, l] = rng.choice(good, size=nsample, replace=True)
            IndPos.append(l)

    return CoIndMat, np.array(IndPos, dtype=int)

def list_triangles(AdjMat, Ind_i, Ind_j, params, rng):
    A = (AdjMat != 0)
    maxtri = params['maxtri']
    TriEdge = []
    TriK = []

    for l in range(len(Ind_i)):
        ks = np.where(A[:, Ind_i[l]] & A[:, Ind_j[l]])[0]
        if ks.size > maxtri:
            ks = rng.choice(ks, size=maxtri, replace=False)
        TriEdge.append(np.full(ks.size, l))
        TriK.append(ks)

    return np.concatenate(TriEdge).astype(int), np.concatenate(TriK).astype(int)
