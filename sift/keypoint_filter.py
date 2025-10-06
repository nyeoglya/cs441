import numpy as np

# --------------------------------------------
# keypoint_filter.py  (Slides: edge response test via Hessian ratio)
# --------------------------------------------

def hessian_edge_reject(D: np.ndarray, y: int, x: int, edge_th: float = 10.0) -> bool:
    """
    Hessian-based edge rejection (Tr^2 / Det test).

    DEPENDS-ON: none
    USED-BY   : filter_keypoints

    Return
    ------
    keep : bool (True if NOT edge-like)
    """
    #############################
    ######### Implement here ####
    # Hints:
    # - Estimate a 2×2 Hessian at (y,x) from local finite differences.
    # - Compute trace^2 / det, check against ((r+1)^2)/r and det>0.
    Dxx = float(D[y, x+1]) - 2.0*float(D[y, x]) + float(D[y, x-1])
    Dyy = float(D[y+1, x]) - 2.0*float(D[y, x]) + float(D[y-1, x])
    Dxy = 0.25 * (float(D[y+1, x+1]) - float(D[y+1, x-1]) - float(D[y-1, x+1]) + float(D[y-1, x-1]))

    tr = Dxx + Dyy
    det = Dxx*Dyy - Dxy**2

    if det <= 0:
        return False

    gamma = (edge_th+1)**2/edge_th
    return tr ** 2 / det < gamma
    #############################


def filter_keypoints(dog_pyr, kpts, edge_th: float = 10.0):
    """
    Apply edge rejection to 3D-NMS candidates.

    DEPENDS-ON: hessian_edge_reject
    USED-BY   : SIFT pipeline
    """
    out = []
    for (o, s, y, x) in kpts:
        D = dog_pyr[o][s]
        if 1 <= x < D.shape[1] - 1 and 1 <= y < D.shape[0] - 1:
            if hessian_edge_reject(D, y, x, edge_th=edge_th):
                out.append((o, s, y, x))
    return out
