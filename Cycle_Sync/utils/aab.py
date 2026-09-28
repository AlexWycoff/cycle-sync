import numpy as np

def compute_naive_aab(tijMat, tijMat3d, CoIndMat, Ind_i, Ind_j, IndPos):
    nsample = CoIndMat.shape[0]
    m = tijMat.shape[1]

    SAABMat0 = np.zeros((nsample, m))
    if IndPos.size == 0:
        return np.mean(SAABMat0, axis=0), SAABMat0

    ks = CoIndMat[:, IndPos]
    Xki = tijMat3d[:, Ind_i[IndPos][None, :], ks]
    Xjk = tijMat3d[:, ks, Ind_j[IndPos][None, :]]
    g = tijMat[:, IndPos][:, None, :]

    X = np.sum(Xki * g, axis=0)
    Y = np.sum(Xjk * g, axis=0)
    Z = np.sum(Xki * Xjk, axis=0)

    denominator = 1 - Z**2
    nondegenerate = denominator > 1e-12
    denominator = np.where(nondegenerate, denominator, 1.0)
    c1 = (X - Y * Z) / denominator
    c2 = (Y - X * Z) / denominator
    S = nondegenerate & (c1 < 0) & (c2 < 0)

    proj = c1 * Xki + c2 * Xjk
    angle_in = np.arctan2(np.linalg.norm(g - proj, axis=0), np.linalg.norm(proj, axis=0))

    angle_ki = np.arctan2(np.linalg.norm(np.cross(g, Xki, axis=0), axis=0), -X)
    angle_jk = np.arctan2(np.linalg.norm(np.cross(g, Xjk, axis=0), axis=0), -Y)
    angle_out = np.minimum(angle_ki, angle_jk)

    SAABMat0[:, IndPos] = np.where(S, angle_in, angle_out) / np.pi
    IRAABVec = np.mean(SAABMat0, axis=0)

    return IRAABVec, SAABMat0
