import torch
import numpy as np
from so3_utils import hat_map, vee_map, algebra_to_matrix, matrix_to_algebra, exp_map, log_map

class SO3:
    """
    Special Orthogonal Group SO(3) implementation.
    """
    def __init__(self):
        self.dim = 3  # Dimension of the Lie algebra
        self.matrix_dim = 3  # Dimension of the matrix representation
        
        # Precompute the Lie algebra basis
        e1 = torch.zeros(3, 3)
        e1[0, 1], e1[1, 0] = -1, 1
        
        e2 = torch.zeros(3, 3)
        e2[0, 2], e2[2, 0] = 1, -1
        
        e3 = torch.zeros(3, 3)
        e3[1, 2], e3[2, 1] = -1, 1
        
        self.lie_algebra_basis = torch.stack([e1, e2, e3])
    
    def sample_uniform(self, batch_size, device='cpu'):
        """
        Sample uniformly from SO(3) using the quaternion method.
        
        Args:
            batch_size: Number of samples to generate
            device: Device to create tensors on
            
        Returns:
            Batch of rotation matrices
        """
        # Sample quaternions uniformly
        u1 = torch.rand(batch_size, device=device)
        u2 = torch.rand(batch_size, device=device) * 2 * np.pi
        u3 = torch.rand(batch_size, device=device) * 2 * np.pi
        
        # Convert to quaternions
        q0 = torch.sqrt(1 - u1) * torch.sin(u2)
        q1 = torch.sqrt(1 - u1) * torch.cos(u2)
        q2 = torch.sqrt(u1) * torch.sin(u3)
        q3 = torch.sqrt(u1) * torch.cos(u3)
        
        # Convert quaternions to rotation matrices
        R = torch.zeros(batch_size, 3, 3, device=device)
        
        R[:, 0, 0] = 1 - 2 * (q2**2 + q3**2)
        R[:, 0, 1] = 2 * (q1 * q2 - q0 * q3)
        R[:, 0, 2] = 2 * (q1 * q3 + q0 * q2)
        
        R[:, 1, 0] = 2 * (q1 * q2 + q0 * q3)
        R[:, 1, 1] = 1 - 2 * (q1**2 + q3**2)
        R[:, 1, 2] = 2 * (q2 * q3 - q0 * q1)
        
        R[:, 2, 0] = 2 * (q1 * q3 - q0 * q2)
        R[:, 2, 1] = 2 * (q2 * q3 + q0 * q1)
        R[:, 2, 2] = 1 - 2 * (q1**2 + q2**2)
        
        return R
    
    def sample_normal(self, batch_size, sigma=1.0, device='cpu'):
        """
        Sample from a normal distribution centered at the identity.
        
        Args:
            batch_size: Number of samples to generate
            sigma: Standard deviation parameter
            device: Device to create tensors on
            
        Returns:
            Batch of rotation matrices
        """
        # Sample tangent vectors from normal distribution in R^3
        v = torch.randn(batch_size, 3, device=device) * sigma
        
        # Convert to rotation matrices using the exponential map
        return self.exp(v)
    
    def exp(self, v):
        """
        Exponential map from Lie algebra to Lie group.
        
        Args:
            v: Batch of vectors in the Lie algebra
            
        Returns:
            Batch of rotation matrices
        """
        return exp_map(v)
    
    def log(self, R):
        """
        Logarithm map from Lie group to Lie algebra.
        
        Args:
            R: Batch of rotation matrices
            
        Returns:
            Batch of vectors in the Lie algebra
        """
        return log_map(R)
    
    def L(self, g, h):
        """
        Left action: g · h
        
        Args:
            g: Batch of rotation matrices
            h: Batch of rotation matrices
            
        Returns:
            Batch of rotation matrices representing g · h
        """
        return torch.matmul(g, h)
    
    def L_inv(self, g, h):
        """
        Compute g^{-1} · h
        
        Args:
            g: Batch of rotation matrices
            h: Batch of rotation matrices
            
        Returns:
            Batch of rotation matrices representing g^{-1} · h
        """
        # For SO(3), inverse is transpose
        g_inv = g.transpose(-1, -2)
        return torch.matmul(g_inv, h)
    
    def matrix_to_algebra(self, X):
        """
        Convert a skew-symmetric matrix to a vector in the Lie algebra.
        
        Args:
            X: Batch of skew-symmetric matrices
            
        Returns:
            Batch of vectors in the Lie algebra
        """
        return matrix_to_algebra(X)
    
    def algebra_to_matrix(self, v):
        """
        Convert a vector in the Lie algebra to a skew-symmetric matrix.
        
        Args:
            v: Batch of vectors in the Lie algebra
            
        Returns:
            Batch of skew-symmetric matrices
        """
        return algebra_to_matrix(v)
    
    def identity(self, batch_size=1, device='cpu'):
        """
        Return the identity element of SO(3).
        
        Args:
            batch_size: Batch size
            device: Device to create tensor on
            
        Returns:
            Batch of identity matrices
        """
        return torch.eye(3, device=device).expand(batch_size, 3, 3) 