import numpy as np
from .gaussian import image_gradients

# --------------------------------------------
# descriptor.py  (Slides: 4x4 cells × 8 bins = 128D, aligned to dominant angle)
# --------------------------------------------

def sift_descriptor(gauss_pyr, kpt, theta: float,
                    cell: int = 4, bins: int = 8, scale: int = 8):
    """
    128-D SIFT-like descriptor (4x4 cells * 8 bins), aligned by theta.

    DEPENDS-ON: gaussian.image_gradients
    USED-BY   : matching

    Return
    ------
    desc : (128,) float32 or None
    """
    o, s, y, x = kpt
    G = gauss_pyr[o][s]
    H, W = G.shape

    Ix, Iy = image_gradients(G)
    mag = np.sqrt(Ix**2 + Iy**2)
    ang = (np.arctan2(Iy, Ix) - theta) % (2*np.pi)

    win = cell * scale
    y0, y1 = max(0, y - win//2), min(H, y + win//2)
    x0, x1 = max(0, x - win//2), min(W, x + win//2)
    if y1 - y0 <= 1 or x1 - x0 <= 1:
        return None

    m = mag[y0:y1, x0:x1]
    a = ang[y0:y1, x0:x1]

    # Split into a (cell x cell) grid
    hstep = max(1, (y1 - y0) // cell)
    wstep = max(1, (x1 - x0) // cell)
    desc = np.zeros((cell, cell, bins), dtype=np.float64)

    #############################
    ######### Implement here ####
    # Hints:
    # - For each cell, aggregate a simple orientation histogram from its pixels.
    # - Hard-assign angles to bins; weights from gradient magnitudes.
    bin_width = 2*np.pi / bins
    for i in range(cell):
        for j in range(cell):
            # 셀마다 패치 생성
            m_patch = m[i*hstep:(i+1)*hstep, j*wstep:(j+1)*wstep]
            a_patch = a[i*hstep:(i+1)*hstep, j*wstep:(j+1)*wstep]

            # 히스토그램 만들기
            for a_data, m_data in zip(a_patch.ravel(), m_patch.ravel()):
                idx = int(a_data // bin_width) % bins
                desc[i, j, idx] += m_data
    #############################

    #############################
    ######### Implement here ####
    # Hints:
    # - Flatten to 128-D and L2-normalize.
    # - (Optional) small clipping before renormalization if you want textbook SIFT.
    vec = desc.ravel().astype(np.float32) # flatten
    norm = np.linalg.norm(vec) + 1e-7 # 정규화용 값 + 0 방지
    return vec / norm
    #############################
