"""
Part 2: SVM Training - Linear SVM Classifier
Implement TODO 8.
"""

import numpy as np
from .loss import compute_hinge_loss
from .optimizer import compute_gradient

from tqdm import tqdm


class LinearSVM:
    """
    Linear SVM classifier using hinge loss and SGD optimization.
    
    Parameters:
        C: regularization parameter (default: 1.0)
        random_state: random seed for reproducibility (default: 42)
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.w = None
        self.b = None
        
    def train(self, X, y, C=1.0, learning_rate=0.001, epochs=100, batch_size=64, verbose=True):
        """
        Train the SVM using Stochastic Gradient Descent.
        
        Args:
            X: (N, D) training feature matrix
            y: (N,) training labels (+1 or -1)
            learning_rate: learning rate for SGD (default: 0.001)
            epochs: number of training epochs (default: 100)
            batch_size: mini-batch size for SGD (default: 64)
            verbose: whether to print training progress (default: True)
        
        Hint:
            - Use mini-batch SGD to train the SVM
        """
        
        np.random.seed(self.random_state)
        
        N, D = X.shape
        
        # Initialize weights and bias
        self.w = np.random.randn(D) * 0.01
        self.b = 0.0

        # Training loop
        for epoch in tqdm(range(epochs), desc=f"Epoch"):

            # ========================================
            # TODO 8: Implement SVM training with SGD
            # ========================================

            rand_ind = np.arange(N)
            np.random.shuffle(rand_ind)
            for i in range(0, N, batch_size): # 데이터를 배치 사이즈로 순회
                b_ind = rand_ind[i:min(i + batch_size, N)] # 데이터 자르기
                grad_w, grad_b = compute_gradient(X[b_ind], y[b_ind], self.w, self.b, C) # 가중치 계산                
                self.w -= grad_w * learning_rate # 가중치 업데이트
                self.b -= grad_b * learning_rate
            
            # ========================================

            # Print loss every 10 epochs
            if verbose and (epoch + 1) % 10 == 0:
                loss = compute_hinge_loss(X, y, self.w, self.b, C)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
        
        if verbose:
            final_loss = compute_hinge_loss(X, y, self.w, self.b, C)
            print(f"Training completed. Final loss: {final_loss:.4f}")        

    
    def predict(self, X):
        """
        Predict class labels or decision scores.
        
        Args:
            X: (N, D) feature matrix
        
        Returns:
            scores: (N,) decision function values (w^T x + b)
                    Positive scores indicate positive class
                    Negative scores indicate negative class
        
        Note:
            To get binary predictions: np.sign(scores)
        """
        if self.w is None or self.b is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        scores = np.dot(X, self.w) + self.b
        return scores
    
    def evaluate(self, X, y):
        """
        Evaluate the classifier on given data.
        
        Args:
            X: (N, D) feature matrix
            y: (N,) true labels (+1 or -1)
        
        Returns:
            accuracy: classification accuracy
        """
        scores = self.predict(X)
        # predictions = np.sign(scores)
        predictions = np.where(scores >= 0, 1, -1)
        accuracy = np.mean(predictions == y)
        return accuracy