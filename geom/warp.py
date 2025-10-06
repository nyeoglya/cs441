import numpy as np

def warp_inverse_map(src, H, out_shape, fill=0.0):
    """
    Warp source image to target canvas using inverse mapping with bilinear sampling.

    Input
    -----
    src       : (H,W) or (H,W,3) float array in [0,1]
    H         : (3,3) homography mapping src -> dst
    out_shape : (H_out, W_out)
    fill      : background fill value

    Output
    ------
    out : warped image shaped out_shape (and same channels as src)

    ------------
    - Homogeneous coordinates + projective transform (lec02/lec03).
    - Inverse mapping is standard for resampling.
    """
    Hh, Hw = out_shape
    if src.ndim == 2: C = 1; src_c = src[...,None]
    else: C = src.shape[2]; src_c = src
    out = np.full((Hh, Hw, C), fill, dtype=np.float64)

    #############################
    ######### Implement here ####
    # Hint:
    # - Create a grid of destination coords; convert to homogeneous.
    # - Map by H^{-1} to source coords; divide by w.
    # - Bilinear sample (check 4 neighbors; skip out-of-bounds).
    # - Write to 'out' per channel. Return gray if C==1.
    H_inv = np.linalg.inv(H)

    ys, xs = np.meshgrid(np.arange(Hh), np.arange(Hw), indexing='ij')
    dest = np.stack([xs, ys, np.ones_like(xs, dtype=np.float64)], axis=-1)

    src_h, src_w = src_c.shape[:2]

    for i in range(Hw):
        for j in range(Hh):
            pt = dest[j, i]
            ipt = H_inv @ pt

            w = ipt[2]
            if w == 0 or not np.isfinite(w):
                continue

            x_src = ipt[0] / w
            y_src = ipt[1] / w

            x0 = int(np.floor(x_src))
            y0 = int(np.floor(y_src))
            x1 = x0 + 1
            y1 = y0 + 1

            if x0 < 0 or y0 < 0 or x1 >= src_w or y1 >= src_h:
                continue

            wx = x_src - x0
            wy = y_src - y0
            w00 = (1 - wx) * (1 - wy)
            w10 = wx * (1 - wy)
            w01 = (1 - wx) * wy
            w11 = wx * wy

            v00 = src_c[y0, x0]
            v10 = src_c[y0, x1]
            v01 = src_c[y1, x0]
            v11 = src_c[y1, x1]
            out[j, i] = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11

    if C == 1:
        out = out[..., 0]
    return out
    #############################


def compose_bounds(img, H):
    """
    Compute bounding box after warping 'img' by H (for panorama canvas sizing).

    Output
    ------
    x_min, y_min, x_max, y_max : ints (inclusive/exclusive as you prefer)
    """
    H0, W0 = img.shape[:2]
    corners = np.array([[0,0,1],[W0,0,1],[0,H0,1],[W0,H0,1]], dtype=np.float64).T
    warped = (H @ corners); warped = (warped[:2,:]/warped[2:,:]).T
    xs = np.concatenate([warped[:,0],[0,W0]]); ys = np.concatenate([warped[:,1],[0,H0]])
    x_min, x_max = int(np.floor(xs.min())), int(np.ceil(xs.max()))
    y_min, y_max = int(np.floor(ys.min())), int(np.ceil(ys.max()))
    return x_min, y_min, x_max, y_max
