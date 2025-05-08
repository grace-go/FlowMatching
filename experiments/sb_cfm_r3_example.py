"""
Schrödinger Bridge Conditional Flow Matching (SB-CFM) Example for R^3

This script demonstrates SB-CFM on R^3, showing how to train and sample from learned distributions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import os

from lieflow.groups import Rn
from bridges.sb_cfm import SBCFM

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create output directory
os.makedirs('output', exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def create_rn_datasets(n_samples=1000):
    """Create source and target distributions in R^3."""
    # Source: Gaussian centered at origin
    source = torch.randn(n_samples, 3) * 0.5
    
    # Target: Mixture of Gaussians
    target = torch.zeros(n_samples, 3)
    
    # Create mixture of 3 Gaussians
    centers = torch.tensor([
        [2.0, 0.0, 0.0],
        [-1.0, 1.7, 0.0],
        [-1.0, -1.7, 0.0]
    ])
    
    # Assign each sample to one of the centers
    indices = torch.randint(0, 3, (n_samples,))
    for i in range(n_samples):
        center = centers[indices[i]]
        target[i] = center + torch.randn(3) * 0.3
    
    return source, target

def train_rn_model(source, target, batch_size=64, epochs=100):
    """Train SB-CFM model on R^3 data."""
    # Create data loaders
    dataset = TensorDataset(source, target)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = SBCFM(dim=3, H=128, L=3, sigma=0.1).to(device)
    print(f"R^3 model has {model.parameter_count:,} parameters")
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Train the model
    losses = model.train_network(
        device=device,
        train_loader=dataloader,
        optimizer=optimizer,
        epochs=epochs,
        log_interval=10
    )
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('SB-CFM Training Loss (R^3)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.savefig('output/rn_sb_cfm_loss.png')
    
    return model, losses

def generate_and_visualize_rn_samples(model, source_data, target_data, n_samples=500):
    """Generate and visualize samples from the R^3 model."""
    # Sample from source distribution
    source_samples = source_data[:n_samples].to(device)
    
    # Generate samples using the model
    with torch.no_grad():
        generated_samples = model.sample(source_samples, num_steps=100).cpu()
    
    # Visualize in 3D
    fig = plt.figure(figsize=(18, 6))
    
    # Source distribution
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(source_samples.cpu()[:, 0], source_samples.cpu()[:, 1], source_samples.cpu()[:, 2], 
                c='blue', alpha=0.5, s=10)
    ax1.set_title('Source Distribution')
    ax1.set_xlim([-3, 3])
    ax1.set_ylim([-3, 3])
    ax1.set_zlim([-3, 3])
    
    # Target distribution
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(target_data[:n_samples].cpu()[:, 0], target_data[:n_samples].cpu()[:, 1], 
                target_data[:n_samples].cpu()[:, 2], c='green', alpha=0.5, s=10)
    ax2.set_title('Target Distribution')
    ax2.set_xlim([-3, 3])
    ax2.set_ylim([-3, 3])
    ax2.set_zlim([-3, 3])
    
    # Generated distribution
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(generated_samples[:, 0], generated_samples[:, 1], generated_samples[:, 2], 
                c='red', alpha=0.5, s=10)
    ax3.set_title('Generated Distribution')
    ax3.set_xlim([-3, 3])
    ax3.set_ylim([-3, 3])
    ax3.set_zlim([-3, 3])
    
    plt.tight_layout()
    plt.savefig('output/rn_distribution_comparison.png')
    # Adjust figure layout to prevent cutoff
    plt.subplots_adjust(top=0.9)  # Increase space at top
    plt.savefig('output/rn_distribution_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()  # Close the figure to free memory
    return generated_samples

def main():
    # Parameters
    n_samples = 1000
    batch_size = 64
    epochs = 50  # Reduced for quicker execution
    
    print("=== R^3 Experiment ===")
    # Create R^3 datasets
    source_rn, target_rn = create_rn_datasets(n_samples)
    print(f"Created R^3 datasets: {source_rn.shape}, {target_rn.shape}")
    
    # Train R^3 model
    model_rn, losses_rn = train_rn_model(
        source_rn, target_rn, 
        batch_size=batch_size, 
        epochs=epochs
    )
    
    # Generate and visualize R^3 samples
    generated_rn = generate_and_visualize_rn_samples(
        model_rn, 
        source_rn, 
        target_rn
    )
    
    print("Experiment completed. Results saved in output/ directory.")

if __name__ == "__main__":
    main() 