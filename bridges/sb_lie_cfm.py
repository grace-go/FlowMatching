import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from .sinkhorn import sinkhorn

class SBLieFlowFieldGroup(nn.Module):
    """
    Schrödinger Bridge Conditional Flow Matching (SB-CFM) for Lie groups.
    
    This model extends vanilla flow matching with:
    1. Entropic-regularized optimal transport coupling
    2. Brownian bridge path sampling
    3. Corrected drift term accounting for stochasticity
    
    Args:
        `G`: group (like SO3).
      Optional:
        `H`: width of the network: number of channels. Defaults to 64.
        `L`: depth of the network: number of layers - 2. Defaults to 2.
        `sigma`: temperature parameter for SB-CFM. Defaults to 0.1.
    """
    
    def __init__(self, G, H=64, L=2, sigma=0.1):
        super().__init__()
        self.G = G
        self.sigma = sigma
        self.H = H
        
        # Define the neural network architecture
        layers = []
        # Input: group element (flattened) + time
        # For matrix groups like SO(3), this is matrix_dim x matrix_dim (e.g., 3x3=9)
        # For R^3, it's just 3
        input_dim = self.G.matrix_dim**2 + 1  # +1 for time
        output_dim = self.G.dim  # Output in the Lie algebra
        
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
    
    def forward(self, g, t):
        """
        Forward pass of the neural network.
        
        Args:
            g: Batch of group elements (matrices or vectors)
            t: Batch of time points in [0, 1]
            
        Returns:
            Vector field in the Lie algebra
        """
        batch_size = g.shape[0]
        
        # Handle different input types (matrices vs vectors)
        if len(g.shape) == 3 and g.shape[1] == g.shape[2] == self.G.matrix_dim:
            # Matrix input (e.g., SO(3)): flatten from [B, 3, 3] to [B, 9]
            g_flat = g.reshape(batch_size, -1)
        else:
            # Vector input (e.g., R^3): already in right format [B, 3]
            g_flat = g
            
        # Ensure time has the right shape [B, 1]
        if len(t.shape) == 1:
            t = t.unsqueeze(1)
        
        # Concatenate group element and time
        x = torch.cat([g_flat, t], dim=1)
        
        # Pass through the network
        return self.network(x)
    
    def _compute_cost_matrix(self, g_0, g_1):
        """Compute squared geodesic distance cost matrix."""
        batch_size_0 = g_0.shape[0]
        batch_size_1 = g_1.shape[0]
        
        # We compute pairwise distances between all elements in the batches
        costs = torch.zeros((batch_size_0, batch_size_1), device=g_0.device)
        
        for i in range(batch_size_0):
            # Compute log(g_0[i]^{-1} g_1[j]) for all j
            log_diffs = torch.stack([
                self.G.log(self.G.L_inv(g_0[i:i+1], g_1[j:j+1])).squeeze()
                for j in range(batch_size_1)
            ])
            
            # Compute squared norm
            costs[i] = torch.sum(log_diffs**2, dim=1)
            
        return costs
    
    def _get_sinkhorn_coupling(self, g_0, g_1):
        """Compute the entropic regularized optimal transport coupling."""
        # Compute cost matrix
        cost_matrix = self._compute_cost_matrix(g_0, g_1)
        
        # Set regularization parameter based on sigma
        reg_param = 2 * self.sigma**2
        
        # Uniform weights for both distributions
        a = torch.ones(g_0.shape[0], device=g_0.device) / g_0.shape[0]
        b = torch.ones(g_1.shape[0], device=g_1.device) / g_1.shape[0]
        
        # Compute Sinkhorn coupling using our implementation
        coupling = sinkhorn(a, b, cost_matrix, reg_param)
        
        return coupling
    
    def _sample_brownian_bridge(self, g_0, g_1, t):
        """Sample from Brownian bridge path on the Lie group."""
        batch_size = g_0.shape[0]
        
        # Deterministic path - exponential curve from g_0 to g_1
        A_t = torch.stack([
            self.G.log(self.G.L_inv(g_0[i:i+1], g_1[i:i+1])).squeeze()
            for i in range(batch_size)
        ])
        
        # Compute deterministic midpoint at time t
        g_det = torch.stack([
            self.G.L(g_0[i], self.G.exp(t[i] * A_t[i]))
            for i in range(batch_size)
        ])
        
        # Add scaled Brownian noise
        # Sample random tangent vector in the Lie algebra
        noise = torch.randn(batch_size, self.G.dim, device=g_0.device)
        
        # Scale by sigma * sqrt(t(1-t))
        noise_scale = self.sigma * torch.sqrt(t * (1 - t))
        scaled_noise = noise * noise_scale
        
        # Map to group and apply to deterministic point
        g_t = torch.stack([
            self.G.L(g_det[i], self.G.exp(scaled_noise[i]))
            for i in range(batch_size)
        ])
        
        return g_t, g_det, scaled_noise
    
    def _compute_drift(self, g_t, g_det, g_1, t, noise):
        """Compute the drift for SB-CFM, combining deterministic drift and score term."""
        batch_size = g_t.shape[0]
        
        # Initialize drift
        drift = torch.zeros(batch_size, self.G.dim, device=g_t.device)
        
        for i in range(batch_size):
            # Deterministic drift term (log(g_t^{-1} g_1) / (1-t))
            A_t1 = self.G.log(self.G.L_inv(g_t[i:i+1], g_1[i:i+1])).squeeze()
            drift_forward = A_t1 / (1 - t[i])
            
            # Score correction term
            # This is the correction for a Brownian bridge
            score_correction = noise[i] / (t[i] * (1 - t[i]))
            
            # Combined drift
            drift[i] = drift_forward - 0.5 * score_correction
        
        return drift
    
    def sb_loss(self, g_0_batch, g_1_batch):
        """
        Compute the SB-CFM loss for a batch of source and target group elements.
        
        Args:
            g_0_batch: Batch of source group elements
            g_1_batch: Batch of target group elements
            
        Returns:
            Scalar loss value
        """
        batch_size = g_0_batch.shape[0]
        
        # Compute Sinkhorn coupling if batch sizes are different
        if g_0_batch.shape[0] != g_1_batch.shape[0]:
            coupling = self._get_sinkhorn_coupling(g_0_batch, g_1_batch)
            
            # Sample (g_0, g_1) pairs according to coupling
            sampled_pairs = []
            for _ in range(batch_size):
                flat_idx = torch.multinomial(coupling.flatten(), 1).item()
                i_idx, j_idx = flat_idx // g_1_batch.shape[0], flat_idx % g_1_batch.shape[0]
                sampled_pairs.append((g_0_batch[i_idx], g_1_batch[j_idx]))
            
            g_0 = torch.stack([p[0] for p in sampled_pairs])
            g_1 = torch.stack([p[1] for p in sampled_pairs])
        else:
            # Use the provided batches directly
            g_0 = g_0_batch
            g_1 = g_1_batch
        
        # Sample time points
        t = torch.rand(batch_size, 1, device=g_0.device)
        
        # Sample Brownian bridge points
        g_t, g_det, noise = self._sample_brownian_bridge(g_0, g_1, t)
        
        # Compute ideal drift
        ideal_drift = self._compute_drift(g_t, g_det, g_1, t, noise)
        
        # Predict drift using neural network
        predicted_drift = self(g_t, t)
        
        # Compute loss (MSE in the Lie algebra)
        loss = ((predicted_drift - ideal_drift) ** 2).mean()
        
        return loss
    
    def sample(self, g_0, num_steps=100):
        """
        Generate samples by solving the ODE.
        
        Args:
            g_0: Initial group elements
            num_steps: Number of discretization steps
            
        Returns:
            Final group elements after ODE integration
        """
        device = g_0.device
        batch_size = g_0.shape[0]
        
        # Initialize time steps
        times = torch.linspace(0, 1, num_steps+1, device=device)
        dt = times[1] - times[0]
        
        # Initialize current state
        current_g = g_0
        
        # Solve ODE using Euler method
        for i in range(num_steps):
            t = times[i].view(1).expand(batch_size)
            
            # Get the vector field at current point
            v_t = self(current_g, t)
            
            # The step depends on the group type
            if len(current_g.shape) == 3:  # Matrix group (e.g., SO(3))
                # Convert to Lie algebra matrix form if needed and step forward
                current_g = torch.stack([
                    self.G.L(current_g[j], self.G.exp(dt * self.G.algebra_to_matrix(v_t[j])))
                    for j in range(batch_size)
                ])
            else:  # Vector group (e.g., R^3)
                # For R^3, we just take a step in the direction of the vector field
                current_g = current_g + dt * v_t
        
        return current_g

    @property
    def parameter_count(self):
        """Return the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 