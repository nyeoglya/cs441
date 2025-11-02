"""
Part 2: SVM Training - Gradient Computation
Implement TODO 7.
"""

import numpy as np


def compute_gradient(X, y, w, b, C=1.0):
    """
    Compute the gradient of hinge loss with respect to w and b.
    
    Args:
        X: (N, D) feature matrix
        y: (N,) labels (+1 or -1)
        w: (D,) weight vector
        b: scalar bias term
        C: regularization parameter (default: 1.0)
    
    Returns:
        grad_w: (D,) gradient with respect to w
        grad_b: scalar gradient with respect to b
    
    Hint:
        - For samples with margin >= 1: no gradient contribution
        - Don't forget to add regularization gradient: w (for grad_w)
    """
    
    N, D = X.shape

    # ========================================
    # TODO 7: Implement gradient computation
    # ========================================

    y_pred = np.dot(X, w) + b
    margin = y * y_pred
    upos_mask = margin < 1
    
    coeff = np.zeros(N)
    coeff[upos_mask] = -C * y[upos_mask]
    
    grad_w = w + np.dot(X.T, coeff)
    grad_b = -C * np.sum(y[upos_mask])

    # ========================================
    
    return grad_w, grad_b