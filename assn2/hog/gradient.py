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

    max_mag_sq = np.zeros((height, width))
    for i in range(n_channels):
        channel_img = image[:,:,i]
        # 상하좌우 각각 1픽셀씩 패딩해서 추가
        padded_x = np.pad(channel_img, ((0, 0), (1, 1)), 'edge')
        padded_y = np.pad(channel_img, ((1, 1), (0, 0)), 'edge')
        
        # 중앙 차분으로 차이 계산
        gx_channel = (padded_x[:, 2:] - padded_x[:, :-2]) / 2.0
        gy_channel = (padded_y[2:, :] - padded_y[:-2, :]) / 2.0

        # 최댓값 계산
        channel_mag_sq = np.square(gx_channel) + np.square(gy_channel)
        
        update_mask = channel_mag_sq > max_mag_sq
        gx[update_mask] = gx_channel[update_mask]
        gy[update_mask] = gy_channel[update_mask]
        max_mag_sq[update_mask] = channel_mag_sq[update_mask]
    
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
