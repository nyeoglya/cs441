import numpy as np
from .gaussian import image_gradients, gaussian_kernel_1d, separable_conv2d

# --------------------------------------------
# orientation.py  (Slides: gradient-based orientation histogram, multi-peak)
# --------------------------------------------

def orientation_assignment(gauss_pyr, kpt, num_bins: int = 36, peak_rel: float = 0.8):
    """
    Assign dominant orientations to a keypoint.

    DEPENDS-ON: gaussian.image_gradients
                (Optional weighting) gaussian.gaussian_kernel_1d, gaussian.separable_conv2d
    USED-BY   : descriptor.sift_descriptor

    Output
    ------
    orientations : list[float] (radians)
    """
    o, s, y, x = kpt
    G = gauss_pyr[o][s]
    Ix, Iy = image_gradients(G)
    mag = np.sqrt(Ix**2 + Iy**2)
    ang = np.arctan2(Iy, Ix)

    # Local window near (y,x)
    win = 8
    y0, y1 = max(0, y - win), min(G.shape[0], y + win + 1)
    x0, x1 = max(0, x - win), min(G.shape[1], x + win + 1)
    m = mag[y0:y1, x0:x1].copy()
    a = ang[y0:y1, x0:x1].copy()

    k = gaussian_kernel_1d(1.5)
    m = separable_conv2d(m, k)

    #############################
    ######### Implement here ####
    # Hints:
    # - Build a histogram over [0, 2π) with 'num_bins' bins using window magnitudes.
    # - Select all bin centers whose height ≥ peak_rel * (global max).
    hist = np.zeros(num_bins, dtype=float)
    bin_width = 2*np.pi / num_bins
    for ai, mi in zip(a.ravel(), m.ravel()):
        idx = int(ai // bin_width) % num_bins
        hist[idx] += mi

    thres = hist.max() * peak_rel
    orientations = []
    for i, hval in enumerate(hist):
        if hval >= thres:
            theta = (i + 0.5) * bin_width
            orientations.append(theta % (2*np.pi))

    return orientations
    #############################
