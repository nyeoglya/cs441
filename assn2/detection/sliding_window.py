"""
Part 3: Multi-Scale Sliding Window Detection - Sliding Window
Implement TODO 10.
"""

import numpy as np


def sliding_window(image, window_size=(64, 128), step_size=8):
    """
    Generate sliding windows over the image.
    
    Args:
        image: (H, W, 3) input image
        window_size: (width, height) size of the detection window (default: 64x128)
        step_size: step size in pixels for sliding (default: 8)
    
    Yields:
        (x, y, window): tuple of (x coordinate, y coordinate, cropped window image)
                        where window is (window_height, window_width, 3)
    
    Hint:
        - Slide the window from top-left to bottom-right
        - Only yield windows that fit completely within the image
    """
    
    # ========================================
    # TODO 10: Implement sliding window
    # ========================================
    
    win_width, win_height = window_size
    img_height, img_width = image.shape[:2]

    for y in range(0, img_height - win_height + 1, step_size):
        for x in range(0, img_width - win_width + 1, step_size):
            window = image[
                y : y + win_height,
                x : x + win_width,
                :
            ]
    # ========================================

            # Yield position and window - DO NOT delete this!
            yield (x, y, window)
    