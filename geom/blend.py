import numpy as np
from sift.gaussian import gaussian_blur  # uses Gaussian from slides

def feather_blend(imgA, maskA, imgB, maskB, sigma=15.0):
    """
    Feather blending via smoothed weights.

    --------------------
    - Gaussian smoothing/padding are in slides; this uses them only to make
      smooth alpha weights. The "blend" step itself is conceptually required
      for panorama. (Homography/warping from slides; blending is an application.)
    """
    wA = gaussian_blur(maskA.astype(np.float64), sigma)
    wB = gaussian_blur(maskB.astype(np.float64), sigma)
    W = wA + wB + 1e-12
    wA /= W; wB /= W
    return wA[...,None]*imgA + wB[...,None]*imgB
