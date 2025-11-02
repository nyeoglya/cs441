"""
Part 1: HOG Feature Extraction - Cell Histogram Computation
Implement TODO 3.
"""

import numpy as np


def compute_cell_histogram(magnitude, angle, cell_size=8, nbins=9):
    """
    Compute gradient orientation histogram for each cell.
    
    Args:
        magnitude: (H, W) gradient magnitude
        angle: (H, W) gradient direction (0~180 degrees)
        cell_size: size of each cell (default: 8x8)
        nbins: number of histogram bins (default: 9)
    
    Returns:
        cell_histograms: (n_cells_y, n_cells_x, nbins) histogram for each cell
    
    Hint:
        - Divide the image into cell_size x cell_size cells
        - Create orientation histogram for pixels within each cell
        - Use bilinear interpolation to distribute magnitude to adjacent bins
    """
    
    height, width = magnitude.shape
    n_cells_y = height // cell_size
    n_cells_x = width // cell_size
    
    # Array to store results
    cell_histograms = np.zeros((n_cells_y, n_cells_x, nbins))
    
    # ========================================
    # TODO 3: Implement cell histogram computation
    # ========================================
    
    bin_size = 180.0 / nbins
    
    # 픽셀별 index 계산
    y_coords, x_coords = np.mgrid[:height, :width]
    cell_y_idx = y_coords // cell_size
    cell_x_idx = x_coords // cell_size
    
    # 픽셀별 weight 계산
    bin_float = (angle / bin_size) - 0.5
    bin_idx_1 = np.floor(bin_float).astype(int)
    bin_idx_2 = np.ceil(bin_float).astype(int)
    
    weight_2 = bin_float - bin_idx_1
    weight_1 = 1.0 - weight_2
    
    # index 계산
    bin_idx_1 = (bin_idx_1 + nbins) % nbins
    bin_idx_2 = (bin_idx_2 + nbins) % nbins
    
    # magnitude 갱신
    np.add.at(cell_histograms, (cell_y_idx, cell_x_idx, bin_idx_1), weight_1 * magnitude)
    np.add.at(cell_histograms, (cell_y_idx, cell_x_idx, bin_idx_2), weight_2 * magnitude)

    # ========================================
    
    return cell_histograms