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
    
    bin_size = 180.0 / nbins # 각 bin 각도 기준
    for r in range(height):
        for c in range(width):
            mag = magnitude[r, c]
            ang = angle[r, c]
            cell_y = r // cell_size
            cell_x = c // cell_size
            bin_float = (ang / bin_size) - 0.5 # 빈 위치 계산          
            bin_idx_1 = int(np.floor(bin_float))
            bin_idx_2 = int(np.ceil(bin_float))
            
            # 선형보간 가중치
            weight_2 = bin_float - bin_idx_1
            weight_1 = 1.0 - weight_2
            
            # 마지막 빈이랑 처음 빈 연결
            bin_idx_1 = (bin_idx_1 + nbins) % nbins
            bin_idx_2 = (bin_idx_2 + nbins) % nbins

            cell_histograms[cell_y, cell_x, bin_idx_1] += weight_1 * mag
            cell_histograms[cell_y, cell_x, bin_idx_2] += weight_2 * mag

    # ========================================
    
    return cell_histograms