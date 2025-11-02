"""
Part 2: SVM Training - Hinge Loss Computation
Implement TODO 6.
"""

import numpy as np


def compute_hinge_loss(X, y, w, b, C=1.0):
    """
    Compute the hinge loss for linear SVM.
    
    Args:
        X: (N, D) feature matrix, N samples with D dimensions
        y: (N,) labels, +1 for positive class, -1 for negative class
        w: (D,) weight vector
        b: scalar bias term
        C: regularization parameter (default: 1.0)
    
    Returns:
        loss: scalar hinge loss value
    
    Hint:
        - If margin >= 1, no loss for that sample
        - If margin < 1, loss = 1 - margin
        - Don't forget to add a regularization term
    """
    
    N = X.shape[0]

    # ========================================
    # TODO 6: Implement hinge loss computation
    # ========================================
    
    margins = y * (np.dot(X, w) + b)
    hinge_losses = np.maximum(0, 1 - margins)
    loss = C * np.sum(hinge_losses) + 0.5 * np.sum(w**2)
    
    # ========================================
    
    return loss