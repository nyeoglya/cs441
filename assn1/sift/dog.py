import numpy as np
from .gaussian import gaussian_blur

# --------------------------------------------
# dog.py  (Slides: scale space, k=2^(1/S), DoG, 3×3×3 NMS, contrast)
# --------------------------------------------

def build_gaussian_pyramid(gray: np.ndarray, sigma0: float = 1.6,
                           octaves: int = 4, scales: int = 3):
    """
    Build SIFT-style Gaussian pyramid with (scales+3) levels per octave.

    DEPENDS-ON: gaussian_blur
    USED-BY   : build_dog_pyramid, SIFT keypoint detection

    Output
    ------
    pyr : list[list[np.ndarray]]  (octaves × (scales+3))
    """
    #############################
    ######### Implement here ####
    # Hint:
    # - Use k = 2^(1/scales).
    # - First level in each octave has absolute sigma0.
    # - Next levels use incremental blur (level-to-level).
    # - Next octave base = previous octave's mid-level downsampled by 2.
    # 다운샘플링 함수
    def downsample(img):
        h, w = img.shape
        img = img[:h - h % 2, :w - w % 2]
        return img.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))

    k = 2**(1/scales)
    g_base = gaussian_blur(gray, sigma0) # 블러한 걸 시작 값으로
    g_pyramid = []
    for _ in range(octaves): # 각 octave마다
        octave_imgs = [g_base]
        for i in range(1, scales+3): # scales+3 크기의 gaussian pyramid 만들기
            sigma_prev = sigma0 * (k ** (i - 1))
            sigma_total = sigma0 * (k ** i)
            sigma_diff = (sigma_total**2 - sigma_prev**2) ** 0.5
            next_img = gaussian_blur(octave_imgs[-1], sigma_diff) # 다음 이미지에 블러 적용
            octave_imgs.append(next_img)

        g_base = downsample(octave_imgs[len(octave_imgs)//2].copy()) # 직전 octave의 절반 정도 위치에서 새 base 가져오기
        g_pyramid.append(octave_imgs)
    return g_pyramid
    #############################


def build_dog_pyramid(gauss_pyr):
    """
    DoG pyramid: D[o][s] = G[o][s+1] - G[o][s]

    DEPENDS-ON: none
    USED-BY   : nms_3d, keypoint_filter
    """
    dog = []
    for levels in gauss_pyr:
        d = [levels[i+1] - levels[i] for i in range(len(levels) - 1)]
        dog.append(d)
    return dog


def nms_3d(dog_pyr, contrast_th: float = 0.03):
    """
    3×3×3 NMS in (x,y,scale) to propose keypoints; apply contrast threshold.

    DEPENDS-ON: none
    USED-BY   : keypoint_filter.filter_keypoints

    Output
    ------
    kpts : list[(o, s, y, x)]
    """
    kpts = []
    for o, dlevels in enumerate(dog_pyr):
        for s in range(1, len(dlevels) - 1):
            Dm, D0, Dp = dlevels[s-1], dlevels[s], dlevels[s+1]
            H, W = D0.shape
            for y in range(1, H-1):
                for x in range(1, W-1):
                    v = D0[y, x]
                    if abs(v) < contrast_th:
                        continue
                    patch = np.array([
                        Dm[y-1:y+2, x-1:x+2],
                        D0[y-1:y+2, x-1:x+2],
                        Dp[y-1:y+2, x-1:x+2]])
                    if (v > 0 and v >= patch.max()) or (v < 0 and v <= patch.min()):
                        kpts.append((o, s, y, x))
    return kpts
