import numpy as np
from geom.dlt import dlt_homography, reprojection_errors

def ransac_homography(pts1, pts2, iters=2000, thresh=3.0, seed=0):
    """
    RANSAC to robustly estimate homography.

    Input
    -----
    pts1, pts2 : (N,2) arrays, N>=4
    iters      : number of RANSAC iterations
    thresh     : inlier threshold (pixels) on symmetric transfer error
    seed       : RNG seed

    Output
    ------
    H_best     : (3,3) homography refit on inliers (or None)
    inliers    : (N,) bool mask of inliers w.r.t. H_best

    ------------
    - Randomly sample minimal set (4), fit model, count inliers, keep best,
      and refit on consensus set. (lec02 RANSAC)
    """
    assert pts1.shape == pts2.shape and pts1.shape[0] >= 4
    rng = np.random.default_rng(seed)
    N = pts1.shape[0]
    best_H = None; best_inliers = None; best_score = -1

    #############################
    ######### Implement here ####
    # Hint:
    # - Loop 'iters' times:
    #   * Randomly choose 4 unique indices.
    #   * Fit H via DLT on the 4 points.
    #   * Compute symmetric transfer errors on all pairs.
    #   * Mark inliers by 'thresh'; if better than best, store H & mask.
    # - If no valid model, return (None, zeros(N)).
    # - Else refit H on best inliers (DLT) and return.
    for _ in range(iters):
        idx = rng.choice(N, size=4, replace=False)
        sam1, sam2 = pts1[idx], pts2[idx]
        H = dlt_homography(sam1, sam2)
        if H is None or not np.all(np.isfinite(H)):
            continue
        err = reprojection_errors(H, pts1, pts2)
        if err is None or err.shape[0] != N:
            continue
        if not np.isfinite(err).any():
            continue

        inliers = np.logical_and(err < thresh, np.isfinite(err))
        score = int(inliers.sum())
        if score > best_score:
            best_score = score
            best_inliers = inliers
            best_H = H

    if best_H is None or best_inliers is None or best_score < 4:
        return None, np.zeros(N, dtype=bool)
    H_refit = dlt_homography(pts1[best_inliers], pts2[best_inliers])
    if H_refit is None or not np.all(np.isfinite(H_refit)):
        H_refit = best_H

    err_final = reprojection_errors(H_refit, pts1, pts2)
    if err_final is not None and err_final.shape[0] == N and np.any(np.isfinite(err_final)):
        final_inliers = (err_final < thresh) & np.isfinite(err_final)
    else:
        final_inliers = best_inliers

    return H_refit, final_inliers
    #############################
