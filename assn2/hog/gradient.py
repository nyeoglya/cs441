"""
Part 1: HOG Feature Extraction - Gradient Computation
Implement TODO 1 and TODO 2.
"""

import numpy as np


def compute_gradient(image):
    """
    Compute the x and y gradients of the image.
    
    Args:
        image: (H, W, 3) color image
    
    Returns:
        gx: (H, W) gradient in x direction
        gy: (H, W) gradient in y direction
    
    Hint:
        - For color images, compute gradients for each channel (R, G, B) and choose the maximum gradient
        - Use central difference method: [-1, 0, 1] kernel
        - You can use `np.pad()` with mode='edge' for boundary handling
    """
    
    image = image.astype(float)
    height, width, n_channels = image.shape
    
    # Initialize arrays to store the final gradients
    gx = np.zeros((height, width))
    gy = np.zeros((height, width))

    # ========================================
    # TODO 1: Implement gradient computation
    # ========================================

    # 채널별 패딩
    padded_x = np.pad(image, ((0, 0), (1, 1), (0, 0)), 'edge')
    padded_y = np.pad(image, ((1, 1), (0, 0), (0, 0)), 'edge')

    # 채널별 gradient 계산
    gx_channels = (padded_x[:, 2:, :] - padded_x[:, :-2, :]) / 2.0
    gy_channels = (padded_y[2:, :, :] - padded_y[:-2, :, :]) / 2.0

    # 채널별 magnitude 제곱값 계산
    mag_sq_channels = np.square(gx_channels) + np.square(gy_channels)
    max_mag_indices = np.argmax(mag_sq_channels, axis=2) # 픽셀별 최댓값 계산
    h_indices, w_indices = np.ogrid[:height, :width]

    # gradient 선택
    gx = gx_channels[h_indices, w_indices, max_mag_indices]
    gy = gy_channels[h_indices, w_indices, max_mag_indices]
    
    # ========================================
    
    return gx, gy


def compute_magnitude_angle(gx, gy):
    """
    Compute the magnitude and angle of gradients.
    
    Args:
        gx: (H, W) gradient in x direction
        gy: (H, W) gradient in y direction
    
    Returns:
        magnitude: (H, W) gradient magnitude
        angle: (H, W) gradient direction (0~180 degrees, unsigned)
    
    Hint:
        - You can use `np.degrees()` to convert angle from radians to degrees
        - For unsigned gradient, add 180 degree to negative angles
    """
    
    # ========================================
    # TODO 2: Implement magnitude and angle computation
    # ========================================
    
    magnitude = np.sqrt(gx**2+gy**2) # magnitude 계산
    angle = np.degrees(np.arctan2(gy,gx)) # 각도 계산
    angle[angle < 0] += 180 # 음수 보정
    
    # ========================================
    
    return magnitude, angle
