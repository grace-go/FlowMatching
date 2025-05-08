import torch
import numpy as np

def hat_map(v):
    """
    Convert a 3D vector to a skew-symmetric matrix (hat map).
    
    Args:
        v: 3D vector [v1, v2, v3]
        
    Returns:
        3x3 skew-symmetric matrix
    """
    v = v.view(-1, 3)
    
    zero = torch.zeros_like(v[:, 0])
    
    X = torch.stack([
        torch.stack([zero, -v[:, 2], v[:, 1]], dim=1),
        torch.stack([v[:, 2], zero, -v[:, 0]], dim=1),
        torch.stack([-v[:, 1], v[:, 0], zero], dim=1)
    ], dim=1)
    
    return X.view(*v.shape[:-1], 3, 3)

def vee_map(X):
    """
    Convert a skew-symmetric matrix to a 3D vector (vee map).
    
    Args:
        X: 3x3 skew-symmetric matrix
        
    Returns:
        3D vector [v1, v2, v3]
    """
    return torch.stack([X[..., 2, 1], X[..., 0, 2], X[..., 1, 0]], dim=-1)

def algebra_to_matrix(v):
    """
    Convert a vector in the Lie algebra so(3) to a matrix representation.
    
    Args:
        v: Vector in so(3) Lie algebra
        
    Returns:
        3x3 skew-symmetric matrix
    """
    return hat_map(v)

def matrix_to_algebra(X):
    """
    Convert a matrix in the Lie algebra so(3) to a vector representation.
    
    Args:
        X: 3x3 skew-symmetric matrix
        
    Returns:
        Vector in so(3) Lie algebra
    """
    return vee_map(X)

def exp_map(v):
    """
    Exponential map from so(3) to SO(3).
    
    Args:
        v: Vector in so(3) Lie algebra
        
    Returns:
        Rotation matrix in SO(3)
    """
    v_hat = hat_map(v)
    theta = torch.norm(v, dim=-1, keepdim=True)
    
    # Handle small angles
    I = torch.eye(3, device=v.device).expand(*v.shape[:-1], 3, 3)
    
    # Rodrigues' formula
    sin_term = torch.sin(theta) / theta
    sin_term = torch.where(theta < 1e-8, 1.0, sin_term)
    
    cos_term = (1 - torch.cos(theta)) / (theta**2)
    cos_term = torch.where(theta < 1e-8, 0.5, cos_term)
    
    return I + sin_term.unsqueeze(-1) * v_hat + cos_term.unsqueeze(-1) * torch.matmul(v_hat, v_hat)

def log_map(R):
    """
    Logarithm map from SO(3) to so(3).
    
    Args:
        R: Rotation matrix in SO(3)
        
    Returns:
        Vector in so(3) Lie algebra
    """
    # Extract the angle of rotation
    trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # Numerical stability
    theta = torch.acos(cos_theta)
    
    # Handle small angles and pi rotations
    skew = 0.5 * (R - R.transpose(-1, -2))
    
    # For rotations close to identity
    small_angle_mask = theta < 1e-4
    # For rotations close to pi
    pi_angle_mask = (torch.abs(theta - np.pi) < 1e-4) & ~small_angle_mask
    # Normal case
    normal_mask = ~small_angle_mask & ~pi_angle_mask
    
    v = torch.zeros(*R.shape[:-2], 3, device=R.device)
    
    # Handle normal case
    if normal_mask.any():
        scale = theta[normal_mask] / (2 * torch.sin(theta[normal_mask]))
        v_normal = scale.unsqueeze(-1) * vee_map(skew[normal_mask])
        v[normal_mask] = v_normal
    
    # Handle small angles
    if small_angle_mask.any():
        v_small = vee_map(skew[small_angle_mask])
        v[small_angle_mask] = v_small
    
    # Handle pi rotations
    if pi_angle_mask.any():
        # For pi rotations, we use the fact that (R+I) has rank 1
        # and its column space is the rotation axis
        R_pi = R[pi_angle_mask]
        R_plus_I = R_pi + torch.eye(3, device=R.device).unsqueeze(0)
        
        # Find the column with the largest norm
        norms = torch.norm(R_plus_I, dim=1)
        max_col = torch.argmax(norms, dim=1)
        
        # Extract the rotation axis
        axes = torch.zeros_like(v[pi_angle_mask])
        for i, col_idx in enumerate(max_col):
            axis = R_plus_I[i, :, col_idx]
            axis = axis / torch.norm(axis)
            axes[i] = axis * np.pi
        
        v[pi_angle_mask] = axes
    
    return v 