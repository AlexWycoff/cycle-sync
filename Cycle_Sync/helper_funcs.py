import numpy as np

def findMask(F, pts0, pts1):
    F = F / np.linalg.norm(F.flatten())
    x0h = np.vstack((pts0.T, np.ones((1, pts0.shape[0]))))
    x1h = np.vstack((pts1.T, np.ones((1, pts1.shape[0]))))

    X = np.tile(x1h, (3, 1)) * np.repeat(x0h, 3, axis=0)

    dist1 = np.abs(F.flatten(order='F').reshape(1, -1) @ X)
    eta = np.median(dist1)

    mask = (dist1 < eta)[0]
    return mask.astype(bool)

def normalize_calibration(pts, K):
    '''
    Normalize keypoints using the calibration data.
    pts: (2 by N) or (3 by N) data matrix
    '''
    if pts.shape[0] > 3:
        pts = pts.T

    N = pts.shape[1]

    Cx = K[0, 2]
    Cy = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    pts = pts - np.tile(np.array([[Cx], [Cy]]), (1, N))
    pts[0:2, :] = pts[0:2, :] / np.tile(np.array([[fx], [fy]]), (1, N))
    pts = np.vstack((pts, np.ones((1, N))))  # 3 x N

    return pts

def normalize_points(xu):
    '''
    Input: xu: 2 by N matrix
    Output: xu1: normalized points
    T1: Projection matrix
    '''

    mean_u1 = np.mean(xu[0, :])
    mean_u2 = np.mean(xu[1, :])
    var_u1 = np.var(xu[0, :] - mean_u1)
    var_u2 = np.var(xu[1, :] - mean_u2)

    T1 = np.array([
        [1 / np.sqrt(var_u1), 0, 0],
        [0, 1 / np.sqrt(var_u2), 0],
        [0, 0, 1]
    ]) @ np.array([
        [1, 0, -mean_u1],
        [0, 1, -mean_u2],
        [0, 0, 1]
    ])

    if xu.shape[0] == 2:
        xuu = np.vstack((xu, np.ones((1, xu.shape[1]))))
    else:
        xuu = xu
        
    return T1 @ xuu, T1

def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    sorted_errors = errors[sort_idx]
    recall = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

    errors_with_zero = np.concatenate(([0], sorted_errors))
    recall_with_zero = np.concatenate(([0], recall))

    aucs = np.zeros_like(thresholds, dtype=float)

    for i, t in enumerate(thresholds):
        last_indices = np.where(errors_with_zero <= t)[0]
        if len(last_indices) == 0:
            aucs[i] = 0
        else:
            last_index = last_indices[-1]
            if last_index == len(errors_with_zero) - 1:
                r = recall_with_zero
                e = errors_with_zero
            else:
                r = np.concatenate((recall_with_zero[:last_index + 1], [recall_with_zero[last_index]]))
                e = np.concatenate((errors_with_zero[:last_index + 1], [t]))
            aucs[i] = np.trapz(r, e) / t
    return aucs

def skewmat(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def relpose(F, x1, x2, K1, K2, mask=None, max_points=200):
    '''
    Relative pose from a fundamental matrix, disambiguated by cheirality.
    x1, x2: matched pixel coordinates, (N, 2) or (2, N).
    mask: optional inlier mask (e.g. from cv2.findFundamentalMat); only inliers vote.
    Returns Ruv, tuv with x_cam2 = Ruv @ x_cam1 + tuv and ||tuv|| = 1,
    i.e. Ruv = R2 @ R1.T and tuv is proportional to R2 @ (c1 - c2).
    '''
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if x1.shape[0] != 2:
        x1 = x1.T
        x2 = x2.T
    if mask is not None:
        keep = np.asarray(mask).ravel().astype(bool)
        x1 = x1[:, keep]
        x2 = x2[:, keep]
    if x1.shape[1] > max_points:
        idx = np.linspace(0, x1.shape[1] - 1, max_points).astype(int)
        x1 = x1[:, idx]
        x2 = x2[:, idx]

    y1 = np.linalg.inv(K1) @ np.vstack((x1, np.ones((1, x1.shape[1]))))
    y2 = np.linalg.inv(K2) @ np.vstack((x2, np.ones((1, x2.shape[1]))))

    E = K2.T @ F @ K1
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    candidates = [(U @ W @ Vt, U[:, 2]), (U @ W @ Vt, -U[:, 2]),
                  (U @ W.T @ Vt, U[:, 2]), (U @ W.T @ Vt, -U[:, 2])]

    best, best_count = candidates[0], -1
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    for R, t in candidates:
        P2 = np.hstack((R, t.reshape(3, 1)))
        count = 0
        for p in range(y1.shape[1]):
            A = np.vstack([
                skewmat(y1[:, p]) @ P1,
                skewmat(y2[:, p]) @ P2
            ])
            _, _, vh = np.linalg.svd(A)
            X = vh[-1]
            if abs(X[3]) < 1e-12:
                continue
            X = X[:3] / X[3]
            if X[2] > 0 and (R @ X + t)[2] > 0:
                count += 1
        if count > best_count:
            best, best_count = (R, t), count

    Ruv, tuv = best
    return Ruv, tuv / np.linalg.norm(tuv)

def world_direction(R2, tuv):
    '''
    World-frame unit direction of c2 - c1, from relpose's tuv and camera 2's
    cam_from_world rotation R2 (ground truth or from rotation synchronization).
    '''
    d = -R2.T @ tuv
    return d / np.linalg.norm(d)
