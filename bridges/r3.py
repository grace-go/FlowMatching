import torch
import numpy as np

class R3:
    """
    Trivial Lie group R^3 implementation.
    This is just the abelian group of 3D vectors with addition.
    """
    def __init__(self):
        self.dim = 3  # Dimension of the Lie algebra (same as group)
        self.matrix_dim = 3  # For compatibility with matrix groups
        
        # Identity basis elements for the Lie algebra
        e1 = torch.eye(3)[0]
        e2 = torch.eye(3)[1]
        e3 = torch.eye(3)[2]
        
        self.lie_algebra_basis = torch.stack([e1, e2, e3])
    
    def sample_uniform(self, batch_size, device='cpu'):
        """
        Sample uniformly from R^3 in a reasonable range.
        
        Args:
            batch_size: Number of samples to generate
            device: Device to create tensors on
            
        Returns:
            Batch of 3D vectors
        """
        return torch.rand(batch_size, 3, device=device) * 4 - 2  # Range [-2, 2]
    
    def sample_normal(self, batch_size, sigma=1.0, device='cpu'):
        """
        Sample from a normal distribution centered at the origin.
        
        Args:
            batch_size: Number of samples to generate
            sigma: Standard deviation parameter
            device: Device to create tensors on
            
        Returns:
            Batch of 3D vectors
        """
        return torch.randn(batch_size, 3, device=device) * sigma
    
    def exp(self, v):
        """
        Exponential map from Lie algebra to Lie group.
        For R^3, this is the identity map.
        
        Args:
            v: Batch of vectors in the Lie algebra
            
        Returns:
            Same vectors (identity map)
        """
        return v
    
    def log(self, x):
        """
        Logarithm map from Lie group to Lie algebra.
        For R^3, this is the identity map.
        
        Args:
            x: Batch of vectors in the group
            
        Returns:
            Same vectors (identity map)
        """
        return x
    
    def L(self, g, h):
        """
        Left action: g · h
        For R^3, this is just vector addition.
        
        Args:
            g: Batch of vectors
            h: Batch of vectors
            
        Returns:
            g + h
        """
        return g + h
    
    def L_inv(self, g, h):
        """
        Compute g^{-1} · h
        For R^3, this is h - g.
        
        Args:
            g: Batch of vectors
            h: Batch of vectors
            
        Returns:
            h - g
        """
        return h - g
    
    def matrix_to_algebra(self, X):
        """
        Convert a 3D vector to a vector in the Lie algebra.
        For R^3, this is the identity map.
        
        Args:
            X: Batch of 3D vectors
            
        Returns:
            Same vectors
        """
        return X
    
    def algebra_to_matrix(self, v):
        """
        Convert a vector in the Lie algebra to a 3D vector.
        For R^3, this is the identity map.
        
        Args:
            v: Batch of vectors in the Lie algebra
            
        Returns:
            Same vectors
        """
        return v
    
    def identity(self, batch_size=1, device='cpu'):
        """
        Return the identity element of R^3 (zero vector).
        
        Args:
            batch_size: Batch size
            device: Device to create tensor on
            
        Returns:
            Batch of zero vectors
        """
        return torch.zeros(batch_size, 3, device=device) 