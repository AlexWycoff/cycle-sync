import numpy as np

def reweight_aab(IRAABVec, SAABMat0, CoIndMat, Ind_i, Ind_j, IndMat, IndPos, params):
    m = SAABMat0.shape[1]

    tau = np.pi
    tau_rate = 2
    tau_max = 20 * np.pi

    unverified = np.ones(m, dtype=bool)
    unverified[IndPos] = False
    if IndPos.size == 0:
        return IRAABVec

    IRAABVec = IRAABVec.copy()
    IRAABVec[unverified] = np.median(IRAABVec[IndPos])

    ks = CoIndMat[:, IndPos]
    Ski_ind = IndMat[Ind_i[IndPos][None, :], ks]
    Sjk_ind = IndMat[Ind_j[IndPos][None, :], ks]

    for _ in range(params['aab_iters']):
        tau = min(tau * tau_rate, tau_max)

        Smax = IRAABVec[Ski_ind] + IRAABVec[Sjk_ind]
        W = np.exp(-tau * (Smax - np.min(Smax, axis=0)))
        W /= np.sum(W, axis=0)

        IRAABVec[IndPos] = np.sum(W * SAABMat0[:, IndPos], axis=0)
        IRAABVec[unverified] = np.median(IRAABVec[IndPos])

    return IRAABVec
