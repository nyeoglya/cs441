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
    src_h, src_w = src.shape[:2]

    #############################
    ######### Implement here ####
    # Hint:
    # - Create a grid of destination coords; convert to homogeneous.
    # - Map by H^{-1} to source coords; divide by w.
    # - Bilinear sample (check 4 neighbors; skip out-of-bounds).
    # - Write to 'out' per channel. Return gray if C==1.
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        if C == 1: return out[...,0]
        return out

    # 도착지 그리드 만들기
    y_out, x_out = np.indices(out_shape)
    dest_coords_h = np.stack((x_out.ravel(), y_out.ravel(), np.ones(x_out.size)))

    # 원본 좌표로 매핑
    src_coords_h = H_inv @ dest_coords_h
    w = src_coords_h[2]
    w[w == 0] = 1e-9 # 혹시모를 0으로 나누기 문제 제거

    src_x = (src_coords_h[0] / w).reshape(out_shape)
    src_y = (src_coords_h[1] / w).reshape(out_shape)

    # 유효한 좌표 찾기
    mask = (src_x >= 0) & (src_x < src_w - 1) & (src_y >= 0) & (src_y < src_h - 1)

    # 샘플링 위치
    x0 = np.floor(src_x).astype(int)
    y0 = np.floor(src_y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    # 가중치 계산
    wx = src_x - x0
    wy = src_y - y0

    # 실제 샘플링
    v00 = src_c[y0[mask], x0[mask]]
    v10 = src_c[y0[mask], x1[mask]]
    v01 = src_c[y1[mask], x0[mask]]
    v11 = src_c[y1[mask], x1[mask]]

    # 가중치 곱해서 보간
    interp_val = (v00.T * (1 - wx[mask]) * (1 - wy[mask]) +
                  v10.T * wx[mask] * (1 - wy[mask]) +
                  v01.T * (1 - wx[mask]) * wy[mask] +
                  v11.T * wx[mask] * wy[mask]).T
    
    out[mask] = interp_val

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
