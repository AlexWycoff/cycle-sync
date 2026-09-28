import numpy as np
from cycle_sync import Cycle_Sync
from test_case import align_procrustes

def uniform_corruption(n, p, q, sigma, seed):
    '''
    Erdos-Renyi graph; each edge direction is corrupted to a random unit vector with
    probability q, otherwise perturbed by noise of size sigma. Same conventions as Cycle_Sync.
    '''
    rng = np.random.default_rng(seed)
    t_true = rng.standard_normal((3, n))
    AdjMat = np.tril(rng.random((n, n)) < p, -1).astype(float)
    AdjMat = AdjMat + AdjMat.T
    Ind_i, Ind_j = np.nonzero(np.tril(AdjMat, -1))

    tijMat = t_true[:, Ind_j] - t_true[:, Ind_i]
    tijMat /= np.linalg.norm(tijMat, axis=0)
    noise = rng.standard_normal(tijMat.shape)
    tijMat += sigma * noise / np.linalg.norm(noise, axis=0)
    corrupted = rng.random(tijMat.shape[1]) < q
    tijMat[:, corrupted] = rng.standard_normal((3, corrupted.sum()))
    tijMat /= np.linalg.norm(tijMat, axis=0)

    return AdjMat, tijMat, t_true

if __name__ == '__main__':
    for q in [0.0, 0.2, 0.4, 0.6]:
        AdjMat, tijMat, t_true = uniform_corruption(50, 0.5, q, 0.0, seed=1)
        t_est, out = Cycle_Sync(AdjMat, tijMat, {'seed': 0})
        X_aligned, _, _, _, _ = align_procrustes(t_est, t_true)
        errors = np.linalg.norm(X_aligned - t_true, axis=0)
        print(f"q = {q:.1f}: median error {np.median(errors):.2e}, mean {np.mean(errors):.2e} ({out.TotalTime:.1f} s)")
        if q <= 0.4:
            assert np.max(errors) < 1e-4, "noiseless data should be recovered exactly at this corruption level"
    print("All synthetic checks passed.")
