"""
Part 1: HOG Feature Extraction

This module provides functionality for extracting HOG (Histogram of Oriented Gradients) features.

Main Components:
    - gradient.py: Image gradient computation (TODO 1, 2)
    - histogram.py: Cell-wise histogram computation (TODO 3)
    - descriptor.py: HOG descriptor and block normalization (TODO 4, 5)

For standard INRIA Person Dataset (64x128 images):
    - cells: 8x16 = 128 cells
    - blocks: 7x15 = 105 blocks
    - feature dimension: 105 * 4 * 9 = 3780
"""

from .descriptor import HOGDescriptor
from .gradient import compute_gradient, compute_magnitude_angle
from .histogram import compute_cell_histogram

__all__ = [
    'HOGDescriptor',
    'compute_gradient',
    'compute_magnitude_angle',
    'compute_cell_histogram'
]