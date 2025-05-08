import torch
import torch.nn as nn

class LogarithmicDistance(nn.Module):
    """
    Loss function for measuring distance in the Lie algebra.
    Applies weighted MSE in the logarithmic coordinates.
    
    Args:
        weights: Tensor of weights for each dimension in the Lie algebra
    """
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights
        
    def forward(self, pred, target):
        """
        Compute weighted MSE loss between predicted and target velocities in Lie algebra.
        
        Args:
            pred: Predicted velocities in Lie algebra
            target: Target velocities in Lie algebra
            
        Returns:
            Scalar loss value
        """
        if self.weights is not None:
            # Apply weights to the squared difference
            loss = torch.sum(self.weights * (pred - target) ** 2, dim=1).mean()
        else:
            # Regular MSE loss
            loss = torch.sum((pred - target) ** 2, dim=1).mean()
            
        return loss 