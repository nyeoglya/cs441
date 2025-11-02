"""
Part 2: SVM Training

This module provides functionality for training a linear SVM classifier.

Main Components:
    - loss.py: Hinge loss computation (TODO 6)
    - optimizer.py: Gradient computation for SGD (TODO 7)
    - classifier.py: Linear SVM classifier with training loop (TODO 8)

SVM Training Details:
    - Loss function: Hinge loss with L2 regularization
    - Optimization: Mini-batch Stochastic Gradient Descent (SGD)
    - Labels: +1 for positive (pedestrian), -1 for negative (background)
    - C parameter: Controls trade-off between margin and misclassification
        - Larger C: Smaller margin, fewer misclassifications (may overfit)
        - Smaller C: Larger margin, more tolerance for misclassifications

Hyperparameters:
    - C: Regularization parameter (try: 0.01, 0.1, 1.0, 10.0)
    - learning_rate: Step size for SGD (try: 0.0001, 0.001, 0.01)
    - epochs: Number of passes through training data (try: 50, 100, 200)
    - batch_size: Mini-batch size (try: 32, 64, 128)
"""

from .classifier import LinearSVM
from .loss import compute_hinge_loss
from .optimizer import compute_gradient

__all__ = [
    'LinearSVM',
    'compute_hinge_loss',
    'compute_gradient'
]