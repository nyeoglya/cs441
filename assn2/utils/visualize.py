"""
Visualization utilities for object detection.
"""

import numpy as np


##### bounding boxes #####
def draw_bounding_boxes(image, detections, color=(0, 255, 0), thickness=2):
    # Create a copy to avoid modifying the original image
    image_with_boxes = image.copy()

    for detection in detections:
        if len(detection) == 4:
            x, y, width, height = detection
            score = None
        elif len(detection) == 5:
            x, y, width, height, score = detection
        else:
            raise ValueError(f"Invalid detection format: {detection}. "
                           "Expected (x, y, width, height) or (x, y, width, height, score)")

        # Convert to integers
        x, y, width, height = int(x), int(y), int(width), int(height)

        # Calculate bottom-right corner
        x2 = x + width
        y2 = y + height

        # Draw rectangle
        _draw_rectangle(image_with_boxes, (x, y), (x2, y2), color, thickness)

    return image_with_boxes


def draw_detection_comparison(image, ground_truth, predictions,
                              gt_color=(0, 255, 0), pred_color=(255, 0, 0)):
    """
    Draw both ground truth and predicted bounding boxes on the same image.

    Args:
        image: (H, W, 3) input color image
        ground_truth: list of ground truth boxes (x, y, width, height)
        predictions: list of predicted boxes (x, y, width, height) or with scores
        gt_color: color for ground truth boxes (default: green)
        pred_color: color for prediction boxes (default: red)

    Returns:
        image_with_boxes: (H, W, 3) image with both types of boxes
    """
    # Draw ground truth boxes first (green)
    image_with_boxes = draw_bounding_boxes(
        image, ground_truth, color=gt_color
    )

    # Draw prediction boxes on top (red)
    image_with_boxes = draw_bounding_boxes(
        image_with_boxes, predictions, color=pred_color
    )

    return image_with_boxes


def _draw_rectangle(image, pt1, pt2, color, thickness):
    """
    Draw a rectangle on an image using only NumPy.

    Args:
        image: (H, W, 3) image array
        pt1: (x1, y1) top-left corner
        pt2: (x2, y2) bottom-right corner
        color: (B, G, R) color tuple
        thickness: line thickness in pixels
    """
    x1, y1 = pt1
    x2, y2 = pt2
    h, w = image.shape[:2]

    # Clip coordinates to image boundaries
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    # Ensure valid rectangle
    if x2 <= x1 or y2 <= y1:
        return

    # Draw four lines for the rectangle
    # Top horizontal line
    for t in range(thickness):
        if y1 + t < h:
            image[y1 + t, x1:x2] = color

    # Bottom horizontal line
    for t in range(thickness):
        if y2 - t >= 0:
            image[y2 - t, x1:x2] = color

    # Left vertical line
    for t in range(thickness):
        if x1 + t < w:
            image[y1:y2, x1 + t] = color

    # Right vertical line
    for t in range(thickness):
        if x2 - t >= 0:
            image[y1:y2, x2 - t] = color


def visualize_detections(image, detections):
    """
    Visualize detections by drawing bounding boxes on the image.

    Args:
        image: (H, W, 3) input color image
        detections: list of detection tuples
        title: optional title (not used, kept for backward compatibility)
        save_path: optional path (not used, kept for backward compatibility)

    Returns:
        image_with_boxes: (H, W, 3) image with bounding boxes drawn
    """
    image_with_boxes = draw_bounding_boxes(image, detections)
    return image_with_boxes



##### HOG features #####
def visualize_hog(image, cell_histograms, cell_size=8, nbins=9):
    """
    Visualize HOG features by drawing oriented gradient histograms for each cell.

    Args:
        image: (H, W, 3) original color image (for size reference)
        cell_histograms: (n_cells_y, n_cells_x, nbins) histogram for each cell
        cell_size: size of each cell in pixels (default: 8)
        nbins: number of bins in gradient orientation histogram (default: 9)

    Returns:
        hog_image: (H, W) grayscale visualization of HOG features

    Example:
        from hog.gradient import compute_gradient, compute_magnitude_angle
        from hog.histogram import compute_cell_histogram

        gx, gy = compute_gradient(image)
        magnitude, angle = compute_magnitude_angle(gx, gy)
        cell_histograms = compute_cell_histogram(magnitude, angle, cell_size=8, nbins=9)
        hog_viz = visualize_hog(image, cell_histograms, cell_size=8, nbins=9)
    """
    height, width = image.shape[:2]
    n_cells_y, n_cells_x = cell_histograms.shape[:2]

    # Create blank canvas for visualization
    hog_image = np.zeros((height, width), dtype=np.float64)

    # Angle of each bin (in radians for drawing)
    # HOG uses unsigned gradients (0-180 degrees = 0-pi radians)
    bin_angles = np.arange(nbins) * np.pi / nbins

    # Visualize each cell
    for i in range(n_cells_y):
        for j in range(n_cells_x):
            # Center of the current cell
            cy = int((i + 0.5) * cell_size)
            cx = int((j + 0.5) * cell_size)

            # Histogram of the current cell
            hist = cell_histograms[i, j, :]

            # Normalize histogram for consistent visualization
            hist_max = hist.max()
            if hist_max == 0:
                continue

            # Draw line for each bin
            for bin_idx in range(nbins):
                # Angle and magnitude for this bin
                theta = bin_angles[bin_idx]
                magnitude = hist[bin_idx]

                # Scale magnitude for visualization
                # Make lines proportional to histogram value
                radius = (magnitude / hist_max) * (cell_size / 2.0)

                # Calculate line endpoints (draw in both directions from center)
                dx = radius * np.cos(theta)
                dy = radius * np.sin(theta)

                # Start and end points
                x1 = int(cx - dx)
                y1 = int(cy - dy)
                x2 = int(cx + dx)
                y2 = int(cy + dy)

                # Clip to image boundaries
                x1 = np.clip(x1, 0, width - 1)
                x2 = np.clip(x2, 0, width - 1)
                y1 = np.clip(y1, 0, height - 1)
                y2 = np.clip(y2, 0, height - 1)

                # Draw line with intensity proportional to magnitude
                _draw_line_grayscale(hog_image, y1, x1, y2, x2, magnitude)

    # Normalize to 0-255 range for display
    if hog_image.max() > 0:
        hog_image = (hog_image / hog_image.max() * 255).astype(np.uint8)
    else:
        hog_image = hog_image.astype(np.uint8)

    return hog_image


def _draw_line_grayscale(image, y1, x1, y2, x2, intensity):
    """
    Draw a line on a grayscale image using simple interpolation.

    Args:
        image: (H, W) grayscale image array to draw on
        y1, x1: starting point coordinates
        y2, x2: ending point coordinates
        intensity: line intensity value
    """
    # Number of points to interpolate
    distance = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    num_points = int(distance) + 1

    if num_points == 0:
        return

    # Interpolate points along the line
    y_coords = np.linspace(y1, y2, num_points).astype(int)
    x_coords = np.linspace(x1, x2, num_points).astype(int)

    # Set pixel values (accumulate intensity for overlapping lines)
    for y, x in zip(y_coords, x_coords):
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            image[y, x] = max(image[y, x], intensity)
