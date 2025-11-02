"""
Part 3: Multi-Scale Sliding Window Detection - Image Pyramid Generation
Implement TODO 9.
"""

import numpy as np


def generate_pyramid(image, scale_factor=1.2, min_size=(64, 128), max_scale=1.5):
    """
    Generate an image pyramid for multi-scale detection.

    Args:
        image: (H, W, 3) input color image
        scale_factor: ratio between consecutive pyramid levels (default: 1.2)
        min_size: (min_width, min_height) minimum size to generate (default: 64x128)
        max_scale: maximum upscaling factor (default: 2.0)
                   Limits how much we upscale to avoid excessive computation
                   Common range: 1.5-2.5x for pedestrian detection

    Returns:
        pyramid: list of tuples (scaled_image, scale)
                 where scale is the scaling factor applied to the original image

    Hint:
        - For upscaling: stop when scale would exceed max_scale
        - For downscaling: stop when dimensions would be smaller than min_size
        - min_size is typically set as the window size
        - Use `resize_image` function below
    """

    # ========================================
    # TODO 9: Implement image pyramid generation
    # ========================================

    H, W, _ = image.shape
    pyramid = [(image, 1.0)]

    # downscale
    current_scale = 1.0
    while True:
        next_scale = current_scale / scale_factor # 다음 단계 스케일 계산
        new_W = int(W * next_scale)
        new_H = int(H * next_scale)

        if new_W < min_size[0] or new_H < min_size[1]:
            break
        
        current_scale = next_scale
        new_size = (new_H, new_W)
        resized_img = resize_image(image, new_size)
        pyramid.insert(0, (resized_img, current_scale))

    # upscale
    current_scale = 1.0
    while True:
        next_scale = current_scale * scale_factor # 다음 단계 스케일 계산
        if next_scale > max_scale:
            break
        current_scale = next_scale
        new_size = (int(H * current_scale), int(W * current_scale))
        resized_img = resize_image(image, new_size)
        pyramid.append((resized_img, current_scale))

    # ========================================

    return pyramid



def resize_image(image, new_size):
    """
    Resizes an image using bilinear interpolation, using only NumPy.

    Args:
        image: (H, W, C) input NumPy array.
        new_size: (new_height, new_width) target size tuple.

    Returns:
        resized_image: (new_height, new_width, C) resized image.
    """
    
    old_height, old_width, n_channels = image.shape
    new_height, new_width = int(new_size[0]), int(new_size[1])
    
    resized = np.zeros((new_height, new_width, n_channels), dtype=image.dtype)
    
    # Ensure float division for ratios
    row_ratio = float(old_height) / float(new_height)
    col_ratio = float(old_width) / float(new_width)

    # Use float64 for precise calculations
    image_float = image.astype(np.float64)

    for i in range(new_height):
        for j in range(new_width):
            
            # Map coordinates from new image to original
            src_row = i * row_ratio
            src_col = j * col_ratio
            
            row_floor = int(np.floor(src_row))
            col_floor = int(np.floor(src_col))

            # Find the 4 neighboring pixels (P11, P12, P21, P22)
            # and clip indices to stay within image boundaries
            r1 = min(row_floor,     old_height - 1)
            c1 = min(col_floor,     old_width - 1)
            r2 = min(row_floor + 1, old_height - 1)
            c2 = min(col_floor + 1, old_width - 1)

            p11 = image_float[r1, c1] # Top-Left
            p12 = image_float[r1, c2] # Top-Right
            p21 = image_float[r2, c1] # Bottom-Left
            p22 = image_float[r2, c2] # Bottom-Right

            # Calculate fractional parts for weights
            row_frac = src_row - row_floor
            col_frac = src_col - col_floor

            # Calculate bilinear interpolation weights
            w1 = (1 - row_frac) * (1 - col_frac)
            w2 = (1 - row_frac) * col_frac
            w3 = row_frac * (1 - col_frac)
            w4 = row_frac * col_frac
            
            # Compute weighted sum for all channels
            interpolated_pixel = w1 * p11 + w2 * p12 + w3 * p21 + w4 * p22
            
            resized[i, j] = interpolated_pixel.astype(image.dtype)
            
    return resized