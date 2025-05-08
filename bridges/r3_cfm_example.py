import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Import our custom R3 model
from r3_model import R3FlowModel
from metrics import LogarithmicDistance

# Set random seed for reproducibility
torch.manual_seed(42)

# Define training parameters
N = 2**12       # Number of training samples
BATCH_SIZE = 2**8
EPOCHS = 20
WEIGHT_DECAY = 1e-5  # Small weight decay to prevent overfitting
LEARNING_RATE = 5e-4  # Reduced further for better stability
CLIP_VALUE = 0.5      # Tighter gradient clipping
H = 64  # Width of the network
L = 3   # Number of layers - 2
SIGMA = 0.05  # Reduced temperature parameter for more stable paths
device = "cuda" if torch.cuda.is_available() else "cpu"

# Generate training data
print("Generating training data...")

# Target distribution: mixture of Gaussians in R^3
def generate_mixture_of_gaussians(n_samples, centers, sigmas, weights, device):
    """Generate mixture of Gaussians."""
    n_components = len(centers)
    # Normalize weights
    weights = torch.tensor(weights, device=device) / sum(weights)
    
    # Sample component indices
    component_indices = torch.multinomial(weights, n_samples, replacement=True)
    
    # Generate samples
    samples = torch.zeros(n_samples, 3, device=device)
    for i, (center, sigma) in enumerate(zip(centers, sigmas)):
        mask = (component_indices == i)
        n_comp_samples = mask.sum().item()
        if n_comp_samples > 0:
            center_tensor = torch.tensor(center, device=device)
            sigma_tensor = torch.tensor(sigma, device=device)
            samples[mask] = center_tensor + torch.randn(n_comp_samples, 3, device=device) * sigma_tensor
    
    return samples

# Define mixture components for target distribution
centers = [
    [1.0, 1.0, 1.0],
    [-1.0, -1.0, -1.0],
    [1.0, -1.0, 1.0]
]
sigmas = [
    [0.2, 0.2, 0.2],
    [0.2, 0.2, 0.2],
    [0.2, 0.2, 0.2]
]
weights = [0.4, 0.3, 0.3]

# Generate target samples
target_samples = generate_mixture_of_gaussians(N, centers, sigmas, weights, device)

# Source distribution: normal distribution centered at origin
source_samples = torch.randn(N, 3, device=device)

# Create the SB-CFM model for R^3
print("Creating R3 SB-CFM model...")
model = R3FlowModel(
    hidden_dim=H,
    num_layers=L,
    sigma=SIGMA
)
model = model.to(device)

# Setup optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Training loop
print("Starting training...")
losses = []

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    n_batches = N // BATCH_SIZE
    
    with tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:
        for i in pbar:
            # Get batch
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, N)
            
            source_batch = source_samples[start_idx:end_idx]
            target_batch = target_samples[start_idx:end_idx]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Compute loss
            loss = model.sb_loss(source_batch, target_batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_VALUE)
            
            # Update parameters
            optimizer.step()
            
            # Update progress
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
    
    # Record average loss for this epoch
    avg_epoch_loss = epoch_loss / n_batches
    losses.append(avg_epoch_loss)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_epoch_loss:.4f}")

# Plot the loss curve
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title("SB-CFM Training Loss on R^3")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("r3_sb_cfm_loss.png")
print("Training complete! Loss curve saved to r3_sb_cfm_loss.png")

# After training
# Generate samples using the trained model
print("Generating samples from the model...")
n_test = 1000
test_source = torch.randn(n_test, 3, device=device)

# Generate samples from the reference target distribution for comparison
test_target = generate_mixture_of_gaussians(n_test, centers, sigmas, weights, device)

with torch.no_grad():
    # Generate samples by solving the ODE
    generated_samples = model.sample(test_source)

# Compute statistics to evaluate distribution match
def compute_stats(samples):
    """Compute basic statistics of the distribution."""
    mean = samples.mean(dim=0)
    std = samples.std(dim=0)
    return mean, std

# Compute statistics for target and generated distributions
target_mean, target_std = compute_stats(test_target)
gen_mean, gen_std = compute_stats(generated_samples)

print("Target mean:", target_mean.cpu().numpy())
print("Generated mean:", gen_mean.cpu().numpy())
print("Target std:", target_std.cpu().numpy())
print("Generated std:", gen_std.cpu().numpy())

# Plot the training loss with a log scale for better visualization
plt.figure(figsize=(10, 6))
plt.semilogy(losses)  # Log scale for y-axis
plt.title("SB-CFM Training Loss on R^3 (log scale)")
plt.xlabel("Epoch")
plt.ylabel("Loss (log)")
plt.grid(True)
plt.savefig("r3_sb_cfm_loss_log.png")

# Visualize the results
def plot_3d_scatter(samples, title, filename):
    """Plot a 3D scatter of points."""
    samples_np = samples.cpu().numpy()
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(samples_np[:, 0], samples_np[:, 1], samples_np[:, 2], alpha=0.7, s=10)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set equal aspect ratio
    max_range = np.array([
        samples_np[:, 0].max() - samples_np[:, 0].min(),
        samples_np[:, 1].max() - samples_np[:, 1].min(),
        samples_np[:, 2].max() - samples_np[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (samples_np[:, 0].max() + samples_np[:, 0].min()) * 0.5
    mid_y = (samples_np[:, 1].max() + samples_np[:, 1].min()) * 0.5
    mid_z = (samples_np[:, 2].max() + samples_np[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.savefig(filename)
    plt.close()

def plot_3d_scatter_comparison(samples1, samples2, title, filename, 
                               label1="Target", label2="Generated", 
                               alpha=0.7, s=10):
    """Plot two 3D scatter plots for comparison with different colors."""
    samples1_np = samples1.cpu().numpy()
    samples2_np = samples2.cpu().numpy()
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot both distributions with different colors
    ax.scatter(samples1_np[:, 0], samples1_np[:, 1], samples1_np[:, 2], 
              alpha=alpha, s=s, c='blue', label=label1)
    ax.scatter(samples2_np[:, 0], samples2_np[:, 1], samples2_np[:, 2], 
              alpha=alpha, s=s, c='red', label=label2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    # Set equal aspect ratio
    all_samples = torch.cat([samples1, samples2], dim=0).cpu().numpy()
    max_range = np.array([
        all_samples[:, 0].max() - all_samples[:, 0].min(),
        all_samples[:, 1].max() - all_samples[:, 1].min(),
        all_samples[:, 2].max() - all_samples[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (all_samples[:, 0].max() + all_samples[:, 0].min()) * 0.5
    mid_y = (all_samples[:, 1].max() + all_samples[:, 1].min()) * 0.5
    mid_z = (all_samples[:, 2].max() + all_samples[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.savefig(filename)
    plt.close()

# Plot individual distributions
plot_3d_scatter(test_target, "Target Distribution (Mixture of Gaussians)", "r3_target_distribution.png")
plot_3d_scatter(source_samples[:n_test], "Source Distribution (Normal)", "r3_source_distribution.png")
plot_3d_scatter(generated_samples, "Generated Distribution", "r3_generated_distribution.png")

# Plot side-by-side comparison
plot_3d_scatter_comparison(test_target, generated_samples, 
                          "Target vs Generated Distribution Comparison", 
                          "r3_distribution_comparison.png")

print("Visualization complete! Check r3_target_distribution.png, r3_source_distribution.png, and r3_generated_distribution.png")

print("Done!") 