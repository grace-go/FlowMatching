"""
Schrödinger Bridge Conditional Flow Matching (SB-CFM)

This module implements SB-CFM for R^n spaces, using the architecture
described in the paper: "Conditional Flow Matching with Schrödinger Bridge".
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from .sinkhorn import sinkhorn, sample_pairs
from lieflow.groups import Rn

class SBCFM(nn.Module):
    """
    Schrödinger Bridge Conditional Flow Matching (SB-CFM) for R^n spaces.
    
    This model extends vanilla flow matching with:
    1. Entropic-regularized optimal transport coupling
    2. Brownian bridge path sampling
    3. Corrected drift term accounting for stochasticity
    
    Args:
        dim: Dimension of the space (n in R^n)
        H: Width of the network (number of hidden units)
        L: Depth of the network (number of hidden layers)
        sigma: Noise level parameter
    """
    
    def __init__(self, dim=3, H=64, L=2, sigma=0.1):
        super().__init__()
        self.dim = dim
        self.G = Rn(dim)  # Use the Rn class from lieflow
        self.sigma = sigma
        
        # Define the neural network architecture
        layers = []
        # Input: vector in R^n + time
        input_dim = dim + 1  # +1 for time
        output_dim = dim     # Output is a vector field in R^n
        
        # First layer
        layers.append(nn.Linear(input_dim, H))
        layers.append(nn.SiLU())
        
        # Hidden layers
        for _ in range(L):
            layers.append(nn.Linear(H, H))
            layers.append(nn.SiLU())
        
        # Output layer
        layers.append(nn.Linear(H, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x, t):
        """
        Forward pass of the neural network.
        
        Args:
            x: Batch of vectors in R^n
            t: Batch of time points in [0, 1]
            
        Returns:
            Predicted vector field (drift)
        """
        # Ensure time has the right shape [B, 1]
        if len(t.shape) == 1:
            t = t.unsqueeze(1)
        
        # Concatenate input vector and time
        x_t = torch.cat([x, t], dim=1)
        
        # Pass through the network
        return self.network(x_t)
    
    def _compute_cost_matrix(self, x_0, x_1):
        """Compute squared Euclidean distance cost matrix."""
        batch_size_0 = x_0.shape[0]
        batch_size_1 = x_1.shape[0]
        
        # Reshape for broadcasting
        x_0_expanded = x_0.unsqueeze(1)  # [batch_size_0, 1, dim]
        x_1_expanded = x_1.unsqueeze(0)  # [1, batch_size_1, dim]
        
        # Compute pairwise squared distances
        costs = ((x_0_expanded - x_1_expanded) ** 2).sum(dim=2)
        
        return costs
    
    def _get_sinkhorn_coupling(self, x_0, x_1):
        """Compute the entropic regularized optimal transport coupling."""
        # Compute cost matrix
        cost_matrix = self._compute_cost_matrix(x_0, x_1)
        
        # Set regularization parameter based on sigma
        reg_param = 2 * self.sigma**2
        
        # Uniform weights for both distributions
        a = torch.ones(x_0.shape[0], device=x_0.device) / x_0.shape[0]
        b = torch.ones(x_1.shape[0], device=x_1.device) / x_1.shape[0]
        
        # Compute Sinkhorn coupling
        coupling = sinkhorn(a, b, cost_matrix, reg_param)
        
        return coupling
    
    def _sample_brownian_bridge(self, x_0, x_1, t):
        """Sample from Brownian bridge path in R^n."""
        batch_size = x_0.shape[0]
        
        # Deterministic path - linear interpolation
        x_det = x_0 + t * (x_1 - x_0)
        
        # Add scaled Brownian noise
        noise = torch.randn_like(x_0) * self.sigma
        noise_scale = torch.sqrt(t * (1 - t))
        scaled_noise = noise * noise_scale
        
        # Apply noise to deterministic point
        x_t = x_det + scaled_noise
        
        return x_t, x_det, scaled_noise
    
    def _compute_drift(self, x_t, x_det, x_1, t, noise):
        """Compute the drift for SB-CFM, combining deterministic drift and score term."""
        # Deterministic drift term (x_1 - x_t) / (1-t)
        t_inv = torch.clamp(1 - t, min=1e-5)
        drift_forward = (x_1 - x_t) / t_inv
        
        # Score correction term for Brownian bridge
        t_term = torch.clamp(t * (1 - t), min=1e-5)
        score_correction = noise / t_term
        
        # Combined drift
        drift = drift_forward - 0.5 * self.sigma**2 * score_correction
        
        return drift
    
    def sb_loss(self, x_0_batch, x_1_batch):
        """
        Compute the SB-CFM loss for a batch of source and target points.
        
        Args:
            x_0_batch: Batch of source points
            x_1_batch: Batch of target points
            
        Returns:
            Scalar loss value
        """
        batch_size = x_0_batch.shape[0]
        device = x_0_batch.device
        
        try:
            # Compute Sinkhorn coupling
            coupling = self._get_sinkhorn_coupling(x_0_batch, x_1_batch)
            x_0, x_1 = sample_pairs(coupling, x_0_batch, x_1_batch, batch_size)
        except Exception as e:
            print(f"Sinkhorn sampling failed: {e}. Using direct pairing instead.")
            # Fallback: Use the provided batches directly or random pairing
            if x_0_batch.shape[0] != x_1_batch.shape[0]:
                indices = torch.randperm(min(x_0_batch.shape[0], x_1_batch.shape[0]))[:batch_size]
                x_0 = x_0_batch[indices]
                x_1 = x_1_batch[indices]
            else:
                x_0, x_1 = x_0_batch, x_1_batch
        
        # Sample time points
        t = torch.rand(batch_size, 1, device=device)
        
        # Sample Brownian bridge points
        x_t, x_det, noise = self._sample_brownian_bridge(x_0, x_1, t)
        
        # Compute ideal drift
        ideal_drift = self._compute_drift(x_t, x_det, x_1, t, noise)
        
        # Predict drift using neural network
        predicted_drift = self(x_t, t)
        
        # Compute loss (MSE)
        loss = ((predicted_drift - ideal_drift) ** 2).mean()
        
        return loss
    
    def sample(self, x_0, num_steps=100):
        """
        Generate samples by solving the ODE.
        
        Args:
            x_0: Initial points
            num_steps: Number of discretization steps
            
        Returns:
            Final points after ODE integration
        """
        device = x_0.device
        batch_size = x_0.shape[0]
        
        # Initialize time steps
        times = torch.linspace(0, 1, num_steps+1, device=device)
        dt = times[1] - times[0]
        
        # Initialize current state
        current_x = x_0.clone()
        
        # Solve ODE using Euler method
        for i in range(num_steps):
            t = times[i].view(1, 1).expand(batch_size, 1)
            
            # Get the vector field at current point
            v_t = self(current_x, t)
            
            # Update using Euler step
            current_x = current_x + dt * v_t
        
        return current_x
    
    def train_network(self, device, train_loader, optimizer, epochs=100, log_interval=10):
        """Train the network on the given data."""
        self.train()
        losses = []
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for x_0, x_1 in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                x_0, x_1 = x_0.to(device), x_1.to(device)
                
                optimizer.zero_grad()
                loss = self.sb_loss(x_0, x_1)
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            
            if (epoch + 1) % log_interval == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        return losses
    
    @property
    def parameter_count(self):
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters()) 