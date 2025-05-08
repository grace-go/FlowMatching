import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Import our custom SO3 class and metrics
from so3 import SO3
from metrics import LogarithmicDistance

# Import our SB-CFM implementation
from sb_lie_cfm import SBLieFlowFieldGroup

# Set random seed for reproducibility
torch.manual_seed(42)

# Define training parameters
EPSILON = 0.01  # Noise level for source distribution
N = 2**12       # Number of training samples (reducing from 2**13 for testing)
BATCH_SIZE = 2**8  # Reducing from 2**9 for testing
EPOCHS = 20     # Reducing from 50 for testing
WEIGHT_DECAY = 0.0
LEARNING_RATE = 1e-2
H = 64  # Width of the network
L = 3   # Number of layers - 2
SIGMA = 0.1  # Temperature parameter for SB-CFM
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize SO(3) group
so3 = SO3()

# Generate training data
def generate_concentrated_rotations(n_samples, center, concentration=10.0, device=device):
    """Generate rotations concentrated around a specific rotation matrix."""
    # Generate tangent vectors from normal distribution
    tangent_vecs = torch.randn(n_samples, 3, device=device) / concentration
    
    # Map to rotation matrices
    perturbations = so3.exp(tangent_vecs)
    
    # Apply perturbations to center
    return so3.L(center.expand(n_samples, 3, 3), perturbations)

# Create two concentrated distributions for training
print("Generating training data...")
# Target distribution: concentrated around a specific rotation
target_center = so3.exp(torch.tensor([1.5, 0.8, 0.3], device=device))
target_samples = generate_concentrated_rotations(N, target_center, concentration=5.0)

# Source distribution: uniform random rotations
source_samples = so3.sample_uniform(N, device=device)

# Create the SB-CFM model
print("Creating SB-CFM model...")
model = SBLieFlowFieldGroup(
    G=so3,
    H=H,
    L=L,
    sigma=SIGMA,
)
model = model.to(device)

# Setup optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Setup loss function
loss_fn = LogarithmicDistance()

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
plt.title("SB-CFM Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("sb_cfm_loss.png")
print("Training complete! Loss curve saved to sb_cfm_loss.png")

# Generate samples using the trained model
print("Generating samples from the model...")
n_test = 1000
test_source = so3.sample_uniform(n_test, device=device)

with torch.no_grad():
    # Generate samples by solving the ODE
    generated_samples = model.sample(test_source)

# Visualize the generated samples
def plot_so3_projection(rotations, title, filename):
    """Plot a 3D scatter of rotations projected to their Euler angles."""
    # Convert rotation matrices to Euler angles
    angles = []
    for R in rotations.cpu().numpy():
        # Extract Euler angles (note: this is one of many conventions)
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
            
        angles.append([x, y, z])
    
    angles = np.array(angles)
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(angles[:, 0], angles[:, 1], angles[:, 2], alpha=0.7)
    ax.set_xlabel('X angle')
    ax.set_ylabel('Y angle')
    ax.set_zlabel('Z angle')
    ax.set_title(title)
    plt.savefig(filename)

# Plot the results
plot_so3_projection(target_samples[:n_test], "Target Distribution", "target_distribution.png")
plot_so3_projection(generated_samples, "Generated Distribution", "generated_distribution.png")
print("Visualization complete! Check target_distribution.png and generated_distribution.png")

print("Done!") 