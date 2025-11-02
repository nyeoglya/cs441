"""
Part 3: Sliding Window Detection

This module provides functionality for multi-scale pedestrian detection using sliding windows.

Main Components:
    - pyramid.py: Image pyramid generation for multi-scale detection (TODO 9)
    - sliding_window.py: Sliding window over image (TODO 10)
    - nms.py: Non-Maximum Suppression to remove overlapping detections (TODO 11)
    - detector.py: Object Detector pipeline (TODO 12)

Detection Pipeline:
    1. Generate image pyramid (multiple scales)
    2. For each scale:
       - Slide detection window
       - Extract HOG features
       - Classify with SVM
       - Record detections with score > threshold
    3. Apply NMS to merge overlapping detections
"""

from .pyramid import generate_pyramid
from .sliding_window import sliding_window
from .nms import non_maximum_suppression, compute_iou
from .detector import ObjectDetector

__all__ = [
    'generate_pyramid',
    'sliding_window',
    'non_maximum_suppression',
    'compute_iou',
    'ObjectDetector'
]