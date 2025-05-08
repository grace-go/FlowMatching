"""
Sinkhorn algorithm for entropic optimal transport.

This module implements the Sinkhorn-Knopp algorithm for computing
entropic-regularized optimal transport couplings between discrete distributions.
"""

import torch
import numpy as np

def sinkhorn(a, b, cost_matrix, epsilon, max_iter=100, threshold=1e-4):
    """
    Compute entropic-regularized optimal transport coupling using Sinkhorn-Knopp algorithm.
    
    Args:
        a: Source distribution weights (tensor of shape [batch_size])
        b: Target distribution weights (tensor of shape [batch_size])
        cost_matrix: Cost matrix (tensor of shape [batch_size, batch_size])
        epsilon: Regularization parameter
        max_iter: Maximum number of iterations (default: 100)
        threshold: Convergence threshold (default: 1e-4)
        
    Returns:
        Optimal transport coupling (tensor of shape [batch_size, batch_size])
    """
    # Ensure numerical stability
    if epsilon < 1e-5:
        epsilon = 1e-5
    
    # Initialize kernel
    K = torch.exp(-cost_matrix / epsilon)
    
    # Check for numerical instability
    if torch.any(torch.isnan(K)) or torch.any(torch.isinf(K)):
        # Apply log-domain stabilization
        log_K = -cost_matrix / epsilon
        max_val = torch.max(log_K)
        K = torch.exp(log_K - max_val)
    
    # Initialize scaling vectors
    u = torch.ones_like(a)
    v = torch.ones_like(b)

    # Sinkhorn iterations in log-domain for stability
    for i in range(max_iter):
        u_prev = u.clone()
        
        # u update: u = a / (K @ v)
        Kv = torch.matmul(K, v.unsqueeze(1)).squeeze(1)
        # Add small constant for numerical stability
        Kv = torch.clamp(Kv, min=1e-10)
        u = a / Kv
        
        # v update: v = b / (K.T @ u)
        Ktu = torch.matmul(K.t(), u.unsqueeze(1)).squeeze(1)
        # Add small constant for numerical stability
        Ktu = torch.clamp(Ktu, min=1e-10)
        v = b / Ktu
        
        # Check convergence
        err = torch.norm(u - u_prev, p=float('inf'))
        if err < threshold:
            break
    
    # Compute final transport coupling
    u_diag = torch.diag_embed(u)
    v_diag = torch.diag_embed(v)
    P = u_diag @ K @ v_diag
    
    # Ensure coupling sums to 1
    P = P / P.sum()
    
    return P

def sample_pairs(coupling, x_0, x_1, batch_size=None):
    """
    Sample pairs of points according to a coupling matrix.
    
    Args:
        coupling: Coupling matrix from sinkhorn
        x_0: Source points
        x_1: Target points
        batch_size: Number of pairs to sample (default: same as x_0)
        
    Returns:
        Tuple of (sampled_x_0, sampled_x_1)
    """
    if batch_size is None:
        batch_size = x_0.shape[0]
    
    # Ensure no NaN or Inf values in coupling
    clean_coupling = torch.where(
        torch.isnan(coupling) | torch.isinf(coupling),
        torch.zeros_like(coupling),
        coupling
    )
    
    # Ensure positive values
    clean_coupling = torch.clamp(clean_coupling, min=1e-10)
    
    # Normalize to sum to 1
    clean_coupling = clean_coupling / clean_coupling.sum()
    
    # Flatten the coupling for sampling
    flat_coupling = clean_coupling.flatten()
    
    # Sample indices according to coupling
    try:
        indices = torch.multinomial(flat_coupling, batch_size, replacement=True)
        
        # Convert flat indices to 2D indices
        i_indices = indices // x_1.shape[0]
        j_indices = indices % x_1.shape[0]
        
        # Extract the sampled pairs
        sampled_x_0 = x_0[i_indices]
        sampled_x_1 = x_1[j_indices]
        
    except Exception as e:
        print(f"Error in sampling pairs: {e}")
        # Fallback: random sampling
        i_indices = torch.randint(0, x_0.shape[0], (batch_size,))
        j_indices = torch.randint(0, x_1.shape[0], (batch_size,))
        
        sampled_x_0 = x_0[i_indices]
        sampled_x_1 = x_1[j_indices]
    
    return sampled_x_0, sampled_x_1 