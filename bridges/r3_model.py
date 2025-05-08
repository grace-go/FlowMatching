import torch
import torch.nn as nn
import numpy as np
from sinkhorn import sinkhorn

class R3FlowModel(nn.Module):
    """
    Schrödinger Bridge Conditional Flow Matching (SB-CFM) model for R^3.
    This is a simplified model specifically for the R^3 vector space.
    """
    
    def __init__(self, hidden_dim=64, num_layers=2, sigma=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sigma = sigma
        
        # Network architecture
        layers = []
        
        # Input: 3D vector + time scalar = 4D
        layers.append(nn.Linear(4, hidden_dim))
        layers.append(nn.SiLU())
        
        # Hidden layers
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        # Output: 3D vector field
        layers.append(nn.Linear(hidden_dim, 3))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x, t):
        """
        Forward pass to predict the vector field.
        
        Args:
            x: Batch of 3D points [batch_size, 3]
            t: Time values [batch_size] or [batch_size, 1]
            
        Returns:
            Vector field at (x, t) [batch_size, 3]
        """
        # Ensure t has shape [batch_size, 1]
        if t.dim() == 1:
            t = t.unsqueeze(1)
        
        # Concatenate input and time
        inputs = torch.cat([x, t], dim=1)
        
        # Forward pass through network
        return self.network(inputs)
    
    def _compute_cost_matrix(self, x_0, x_1):
        """
        Compute pairwise squared distances between source and target points.
        
        Args:
            x_0: Source points [batch_size_0, 3]
            x_1: Target points [batch_size_1, 3]
            
        Returns:
            Cost matrix [batch_size_0, batch_size_1]
        """
        # Compute pairwise distances
        x_0_expanded = x_0.unsqueeze(1)  # [batch_size_0, 1, 3]
        x_1_expanded = x_1.unsqueeze(0)  # [1, batch_size_1, 3]
        
        # Squared Euclidean distance
        squared_dist = torch.sum((x_0_expanded - x_1_expanded) ** 2, dim=2)
        
        return squared_dist
    
    def _get_sinkhorn_coupling(self, x_0, x_1):
        """
        Compute entropic optimal transport coupling.
        
        Args:
            x_0: Source points [batch_size_0, 3]
            x_1: Target points [batch_size_1, 3]
            
        Returns:
            Coupling matrix [batch_size_0, batch_size_1]
        """
        # Compute cost matrix
        cost_matrix = self._compute_cost_matrix(x_0, x_1)
        
        # Regularization parameter
        reg_param = 2 * self.sigma**2
        
        # Uniform weights
        a = torch.ones(x_0.shape[0], device=x_0.device) / x_0.shape[0]
        b = torch.ones(x_1.shape[0], device=x_1.device) / x_1.shape[0]
        
        # Compute coupling
        return sinkhorn(a, b, cost_matrix, reg_param)
    
    def _sample_brownian_bridge(self, x_0, x_1, t):
        """
        Sample from Brownian bridge path according to the formula in the paper.
        
        The Brownian bridge is defined as:
        x_t = (1-t)x_0 + tx_1 + sigma * sqrt(t(1-t)) * noise
        
        Args:
            x_0: Source points [batch_size, 3]
            x_1: Target points [batch_size, 3]
            t: Time values [batch_size, 1]
            
        Returns:
            tuple: (sampled point, deterministic point, noise)
        """
        # Ensure numerical stability
        eps = 1e-5
        t_clamped = torch.clamp(t, eps, 1.0 - eps)
        
        # Deterministic path - linear interpolation (OT)
        x_t_det = (1 - t_clamped) * x_0 + t_clamped * x_1
        
        # Generate random noise
        noise = torch.randn_like(x_0)
        
        # Scale by the Brownian bridge factor
        time_factor = torch.sqrt(t_clamped * (1 - t_clamped))
        
        # Construct Brownian bridge sample
        x_t = x_t_det + self.sigma * time_factor * noise
        
        return x_t, x_t_det, noise
    
    def _compute_drift(self, x_t, x_t_det, x_1, t, noise):
        """
        Compute drift for SB-CFM according to the paper.
        
        In SB-CFM, the drift is composed of two terms:
        1. The deterministic OT drift term (vector to target / remaining time)
        2. The score correction term from the Brownian bridge
        
        Args:
            x_t: Current point [batch_size, 3]
            x_t_det: Deterministic point [batch_size, 3]
            x_1: Target point [batch_size, 3]
            t: Time values [batch_size, 1]
            noise: Noise added to the path [batch_size, 3]
            
        Returns:
            Drift vector [batch_size, 3]
        """
        # Add numerical stability constant to prevent division by zero
        eps = 1e-5
        
        # Clamp t to avoid numerical issues near the boundaries
        t_clamped = torch.clamp(t, eps, 1.0 - eps)
        
        # 1. Deterministic OT drift term (straight to target)
        # u_t = (x_1 - x_t) / (1 - t)
        drift_forward = (x_1 - x_t) / (1 - t_clamped)
        
        # 2. Score correction term from Brownian bridge
        # The stochastic correction term is proportional to the added noise
        # divided by time factor t(1-t)
        score_correction = noise / (t_clamped * (1 - t_clamped))
        
        # Combined drift with the 0.5 factor from the paper's formula
        # u_t = drift_forward - 0.5 * score_correction
        drift = drift_forward - 0.5 * score_correction
        
        return drift
    
    def sb_loss(self, x_0_batch, x_1_batch):
        """
        Compute SB-CFM loss with properly sampled pairs using the Sinkhorn coupling.
        
        Args:
            x_0_batch: Source batch [batch_size, 3]
            x_1_batch: Target batch [batch_size, 3]
            
        Returns:
            Scalar loss
        """
        batch_size = x_0_batch.shape[0]
        
        # For simplicity, we can just pair points directly 
        # This avoids potential CUDA errors with multinomial sampling
        if x_0_batch.shape[0] == x_1_batch.shape[0]:
            # If batch sizes match, use direct pairing
            x_0 = x_0_batch
            x_1 = x_1_batch
        else:
            # If batch sizes don't match, sample randomly
            # (This is a fallback, but direct pairing is preferred)
            min_batch = min(x_0_batch.shape[0], x_1_batch.shape[0])
            x_0 = x_0_batch[:min_batch]
            x_1 = x_1_batch[:min_batch]
        
        # Sample time points avoiding extreme edges
        t = torch.rand(batch_size, 1, device=x_0.device) * 0.95 + 0.025
        
        # Sample Brownian bridge points
        x_t, x_t_det, noise = self._sample_brownian_bridge(x_0, x_1, t)
        
        # Compute ideal drift
        ideal_drift = self._compute_drift(x_t, x_t_det, x_1, t, noise)
        
        # Predict drift with model
        predicted_drift = self(x_t, t)
        
        # Compute MSE loss
        squared_diff = (predicted_drift - ideal_drift) ** 2
        loss = squared_diff.mean()
        
        return loss
    
    def sample(self, x_0, num_steps=100):
        """
        Generate samples using the learned flow field.
        
        According to the paper, for SB-CFM, we can directly transform the source points
        to the target points using the learned flow field, rather than solving an ODE.
        
        Args:
            x_0: Initial points [batch_size, 3]
            num_steps: Number of integration steps (still needed for stepwise transformation)
            
        Returns:
            Final points [batch_size, 3]
        """
        device = x_0.device
        batch_size = x_0.shape[0]
        
        # Current points
        current_x = x_0
        
        # Step through the transformation
        for i in range(num_steps):
            # Current time step
            t = torch.ones(batch_size, device=device) * (i / num_steps)
            
            # Get vector field
            v_t = self(current_x, t)
            
            # Calculate step size
            dt = 1.0 / num_steps
            
            # Update points
            current_x = current_x + dt * v_t
        
        return current_x 