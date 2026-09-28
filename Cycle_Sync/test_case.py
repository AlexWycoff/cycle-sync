import numpy as np

t_true = np.array([
    [0, 1, 1, 0],
    [0, 0, 1, 1],
    [1, 0, 1, 1]
], dtype=float)

def find_adjacency():
    n = t_true.shape[1]
    AdjMat = np.ones((n, n), dtype=int) - np.eye(n)
    tijMat = []

    for i in range(n):
        for j in range(i):
            d = t_true[:, j] - t_true[:, i]
            norm_val = np.linalg.norm(d)
            if norm_val > 1e-10:
                d = d / norm_val
            tijMat.append(d)

    tijMat = np.array(tijMat).T
    return AdjMat, tijMat

def align_procrustes(X_est, X_true, rotation=False):
    '''
    Align X_est to X_true by scale and translation. Location synchronization only has a
    scale + translation gauge (rotations are already fixed), so rotation=False is the
    correct evaluation; rotation=True is kept for comparison and can hide errors.
    '''
    mu_est = X_est.mean(axis=1, keepdims=True)
    mu_true = X_true.mean(axis=1, keepdims=True)
    Xc_est = X_est - mu_est
    Xc_true = X_true - mu_true

    R = np.eye(3)
    if rotation:
        U, _, Vt = np.linalg.svd(Xc_true @ Xc_est.T)
        if np.linalg.det(U @ Vt) < 0:
            U[:, -1] *= -1
        R = U @ Vt

    scale = np.trace(R @ Xc_est @ Xc_true.T) / np.trace(Xc_est @ Xc_est.T)

    X_aligned = scale * R @ Xc_est + mu_true
    err = np.sqrt(np.mean(np.sum((X_aligned - X_true)**2, axis=0)))

    return X_aligned, R, scale, mu_true, err
