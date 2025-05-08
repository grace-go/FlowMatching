"""
Schrödinger Bridge Conditional Flow Matching (SB-CFM) for SO(3)

This module implements a specialized version of SB-CFM for the SO(3) group
with improved numerical stability and proper handling of the manifold structure.
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from .sinkhorn import sinkhorn, sample_pairs
from lieflow.groups import SO3

class SBCFMSO3(nn.Module):
    """
    Schrödinger Bridge Conditional Flow Matching for SO(3) with improved stability.
    
    This model extends vanilla flow matching with:
    1. Entropic-regularized optimal transport coupling
    2. Brownian bridge path sampling on SO(3)
    3. Corrected drift term accounting for stochasticity
    
    Args:
        H: Width of the network (number of hidden units)
        L: Depth of the network (number of hidden layers)
        sigma: Noise level parameter
        epsilon_stab: Stability parameter for matrix logarithm
    """
    
    def __init__(self, H=128, L=2, sigma=0.1, epsilon_stab=0.01):
        super().__init__()
        self.G = SO3()
        self.sigma = sigma
        self.epsilon_stab = epsilon_stab  # Stabilization parameter
        
        # Input: 9 (flattened 3x3 matrix) + 1 (time)
        input_dim = 10
        # Output: 3 (Lie algebra dimension for SO(3))
        output_dim = 3
        
        # Define the neural network architecture
        layers = []
        
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
            g: Batch of SO(3) elements as 3x3 matrices
            t: Batch of time points in [0, 1]
            
        Returns:
            Vector field in the Lie algebra of SO(3)
        """
        batch_size = g.shape[0]
        
        # Flatten the matrices from [B, 3, 3] to [B, 9]
        g_flat = g.reshape(batch_size, -1)
            
        # Ensure time has the right shape [B, 1]
        if len(t.shape) == 1:
            t = t.unsqueeze(1)
        
        # Concatenate group element and time
        x = torch.cat([g_flat, t], dim=1)
        
        # Pass through the network to get components in the Lie algebra
        return self.network(x)
    
    def _compute_cost_matrix(self, g_0, g_1):
        """
        Compute squared geodesic distance cost matrix with improved stability.
        
        For SO(3), the geodesic distance is related to the angle of rotation
        in the logarithm of g_0^{-1} * g_1.
        """
        batch_size_0 = g_0.shape[0]
        batch_size_1 = g_1.shape[0]
        
        # We compute pairwise distances between all elements in the batches
        costs = torch.zeros((batch_size_0, batch_size_1), device=g_0.device)
        
        for i in range(batch_size_0):
            # Compute g_0[i]^{-1} * g_1 for all g_1
            g_i_inv = g_0[i].transpose(-2, -1)  # Transpose for SO(3) inverse
            product = g_i_inv @ g_1  # [batch_size_1, 3, 3]
            
            # Compute the angle from the trace
            # For SO(3): cos(theta) = (tr(R) - 1) / 2
            trace = torch.diagonal(product, dim1=-2, dim2=-1).sum(-1)  # [batch_size_1]
            
            # Clamp trace to valid range for numerical stability
            trace_clamped = torch.clamp(trace, -1.0 + self.epsilon_stab, 3.0 - self.epsilon_stab)
            
            # Compute angle and squared distance
            theta = torch.acos((trace_clamped - 1) / 2)
            costs[i] = theta**2
            
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
        
        # Compute Sinkhorn coupling
        coupling = sinkhorn(a, b, cost_matrix, reg_param)
        
        return coupling
    
    def _skew_symmetric(self, v):
        """
        Convert a 3D vector to a skew-symmetric matrix.
        
        Args:
            v: Batch of 3D vectors [B, 3]
            
        Returns:
            Batch of skew-symmetric matrices [B, 3, 3]
        """
        batch_size = v.shape[0]
        skew = torch.zeros(batch_size, 3, 3, device=v.device)
        
        skew[:, 0, 1] = -v[:, 2]
        skew[:, 0, 2] = v[:, 1]
        skew[:, 1, 0] = v[:, 2]
        skew[:, 1, 2] = -v[:, 0]
        skew[:, 2, 0] = -v[:, 1]
        skew[:, 2, 1] = v[:, 0]
        
        return skew
    
    def _liegroup_geodesic(self, R1, R2, t):
        """
        Compute the geodesic path between two rotation matrices at time t.
        
        Args:
            R1: Starting rotation matrices [B, 3, 3]
            R2: Ending rotation matrices [B, 3, 3]
            t: Time point between 0 and 1 [B, 1]
            
        Returns:
            Rotation matrices at time t [B, 3, 3]
        """
        batch_size = R1.shape[0]
        device = R1.device
        
        try:
            # Get relative rotation from R1 to R2
            R1_inv = R1.transpose(-2, -1)
            rel_R = R1_inv @ R2
            
            # Get the logarithm (rotation vector in the Lie algebra)
            log_R = self.G.log(rel_R, ε_stab=self.epsilon_stab)
            
            # Scale by t
            t_flat = t.view(-1)  # Flatten to [B]
            scaled_log_R = torch.zeros_like(log_R)
            
            # Apply t to each rotation matrix individually
            for i in range(batch_size):
                scaled_log_R[i] = log_R[i] * t_flat[i]
            
            # Exponentiate back to SO(3)
            exp_scaled = torch.matrix_exp(scaled_log_R)
            
            # Apply to starting point
            result = R1 @ exp_scaled
            
            return result
        except Exception as e:
            print(f"Error in geodesic: {e}")
            # Fallback: Linear interpolation in ambient space + projection to SO(3)
            result = torch.zeros_like(R1)
            
            # Apply t to each rotation matrix individually
            t_flat = t.view(-1)  # Flatten to [B]
            for i in range(batch_size):
                # Linear interpolation
                ti = t_flat[i]
                interp = (1 - ti) * R1[i] + ti * R2[i]
                
                # Project to SO(3) using SVD
                U, _, V = torch.linalg.svd(interp)
                result[i] = U @ V
            
            return result
    
    def _sample_brownian_bridge(self, g_0, g_1, t):
        """
        Sample from Brownian bridge path on SO(3) with improved stability.
        
        Args:
            g_0: Starting SO(3) elements [B, 3, 3]
            g_1: Ending SO(3) elements [B, 3, 3]
            t: Time points [B, 1]
            
        Returns:
            g_t: Sampled SO(3) elements at time t
            g_det: Deterministic path points at time t
            scaled_noise_vec: Noise vectors used for sampling
        """
        batch_size = g_0.shape[0]
        device = g_0.device
        
        # 1. Compute deterministic geodesic path using custom geodesic function
        g_det = self._liegroup_geodesic(g_0, g_1, t)
        
        # 2. Add scaled noise in the Lie algebra
        # Sample random tangent vector
        noise_vec = torch.randn(batch_size, 3, device=device)
        
        # Scale by sigma * sqrt(t(1-t))
        t_flat = t.view(-1)
        noise_scale = torch.zeros(batch_size, device=device)
        for i in range(batch_size):
            noise_scale[i] = self.sigma * torch.sqrt(t_flat[i] * (1 - t_flat[i]))
        
        # Apply scale to each vector individually
        scaled_noise_vec = torch.zeros_like(noise_vec)
        for i in range(batch_size):
            scaled_noise_vec[i] = noise_vec[i] * noise_scale[i]
        
        # Convert to skew-symmetric matrices
        scaled_noise_skew = self._skew_symmetric(scaled_noise_vec)
        
        # Map to group and apply to deterministic point
        g_t = torch.zeros_like(g_0)
        for i in range(batch_size):
            # Exponentiate noise
            noise_rotation = torch.matrix_exp(scaled_noise_skew[i])
            
            # Right multiplication for stability
            perturbed = g_det[i] @ noise_rotation
            
            # Project to SO(3)
            U, _, V = torch.linalg.svd(perturbed)
            g_t[i] = U @ V
        
        return g_t, g_det, scaled_noise_vec
    
    def _compute_drift(self, g_t, g_det, g_1, t, noise_vec):
        """
        Compute the drift for SB-CFM on SO(3), combining deterministic drift and score term.
        
        Args:
            g_t: Sampled SO(3) elements at time t [B, 3, 3]
            g_det: Deterministic path points at time t [B, 3, 3]
            g_1: Target SO(3) elements [B, 3, 3]
            t: Time points [B, 1]
            noise_vec: Noise vectors in Lie algebra [B, 3]
            
        Returns:
            Drift vectors in the Lie algebra [B, 3]
        """
        batch_size = g_t.shape[0]
        device = g_t.device
        
        # Process each sample individually to avoid broadcast issues
        drift = torch.zeros(batch_size, 3, device=device)
        t_flat = t.view(-1)
        
        for i in range(batch_size):
            # 1. Compute the forward drift: log(g_t^{-1} * g_1) / (1-t)
            g_t_inv = g_t[i].transpose(-2, -1)
            rel_rotation = g_t_inv @ g_1[i]
            
            try:
                # Get the logarithm with stabilization
                log_rel = self.G.log(rel_rotation, ε_stab=self.epsilon_stab)
                
                # Extract the axis-angle components
                log_rel_vec = torch.zeros(3, device=device)
                log_rel_vec[0] = log_rel[2, 1]
                log_rel_vec[1] = log_rel[0, 2]
                log_rel_vec[2] = log_rel[1, 0]
                
                # Scale by 1/(1-t) for the forward drift
                t_i = t_flat[i]
                t_inv = max(1 - t_i, 1e-5)
                drift_forward = log_rel_vec / t_inv
                
                # Compute the score correction term
                t_term = max(t_i * (1 - t_i), 1e-5)
                score_correction = noise_vec[i] / t_term
                
                # Combined drift
                sample_drift = drift_forward - 0.5 * self.sigma**2 * score_correction
                
                # Apply gradient clipping to avoid exploding gradients
                drift_norm = torch.norm(sample_drift)
                max_norm = 10.0
                if drift_norm > max_norm:
                    sample_drift = sample_drift * max_norm / drift_norm
                
                drift[i] = sample_drift
                
            except Exception as e:
                print(f"Error in drift computation for sample {i}: {e}")
                # Safe fallback
                drift[i] = torch.zeros(3, device=device)
        
        return drift
    
    def sb_loss(self, g_0_batch, g_1_batch):
        """
        Compute the SB-CFM loss for a batch of source and target SO(3) elements.
        
        Args:
            g_0_batch: Batch of source SO(3) elements [B, 3, 3]
            g_1_batch: Batch of target SO(3) elements [B, 3, 3]
            
        Returns:
            Scalar loss value
        """
        batch_size = g_0_batch.shape[0]
        device = g_0_batch.device
        
        # Compute Sinkhorn coupling with exception handling
        try:
            coupling = self._get_sinkhorn_coupling(g_0_batch, g_1_batch)
            g_0, g_1 = sample_pairs(coupling, g_0_batch, g_1_batch, batch_size)
        except Exception as e:
            print(f"Sinkhorn sampling failed: {e}. Using direct pairing instead.")
            # Fallback to direct pairing
            indices = torch.randperm(min(g_0_batch.shape[0], g_1_batch.shape[0]))[:batch_size]
            g_0 = g_0_batch[indices[:batch_size]]
            g_1 = g_1_batch[indices[:batch_size]]
        
        # Sample time points
        t = torch.rand(batch_size, 1, device=device)
        
        # Sample Brownian bridge points
        g_t, g_det, noise_vec = self._sample_brownian_bridge(g_0, g_1, t)
        
        # Compute ideal drift
        ideal_drift = self._compute_drift(g_t, g_det, g_1, t, noise_vec)
        
        # Predict drift using neural network
        predicted_drift = self(g_t, t)
        
        # Compute loss (MSE in the Lie algebra)
        loss = ((predicted_drift - ideal_drift) ** 2).mean()
        
        return loss
    
    def sample(self, g_0, num_steps=100):
        """
        Generate samples by solving the ODE with improved stability.
        
        Args:
            g_0: Initial SO(3) elements [B, 3, 3]
            num_steps: Number of discretization steps
            
        Returns:
            Final SO(3) elements after ODE integration
        """
        device = g_0.device
        batch_size = g_0.shape[0]
        
        # Initialize time steps
        times = torch.linspace(0, 1, num_steps+1, device=device)
        dt = times[1] - times[0]
        
        # Initialize current state
        current_g = g_0.clone()
        
        # Solve ODE using Euler method with projection to SO(3)
        for i in range(num_steps):
            # Process each sample individually
            t_i = times[i].item()
            t_batch = torch.full((batch_size, 1), t_i, device=device)
            
            # Get the vector field at current point
            v_t = self(current_g, t_batch)  # [B, 3]
            
            # Update each rotation matrix individually
            for j in range(batch_size):
                # Convert to skew-symmetric matrix
                v_skew = torch.zeros(3, 3, device=device)
                v_skew[0, 1] = -v_t[j, 2]
                v_skew[0, 2] = v_t[j, 1]
                v_skew[1, 0] = v_t[j, 2]
                v_skew[1, 2] = -v_t[j, 0]
                v_skew[2, 0] = -v_t[j, 1]
                v_skew[2, 1] = v_t[j, 0]
                
                # Scale by dt and exponentiate
                step = torch.matrix_exp(dt * v_skew)
                
                # Update using right multiplication
                updated = current_g[j] @ step
                
                # Project back to SO(3) for numerical stability
                U, _, V = torch.linalg.svd(updated)
                current_g[j] = U @ V
        
        return current_g
    
    def train_network(self, device, train_loader, optimizer, epochs=100, log_interval=10):
        """Train the network on the given data."""
        self.train()
        losses = []
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for g_0, g_1 in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                g_0, g_1 = g_0.to(device), g_1.to(device)
                
                optimizer.zero_grad()
                loss = self.sb_loss(g_0, g_1)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
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