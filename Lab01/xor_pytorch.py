"""
Task 4: Implementation of XOR Problem using PyTorch Neural Network
XOR is not linearly separable, so it requires a multi-layer neural network
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# NEURAL NETWORK DEFINITION
# ============================================================================

class XORNet(nn.Module):
    """
    Simple Neural Network for XOR problem
    Architecture: 2 input neurons -> 4 hidden neurons -> 1 output neuron
    """
    def __init__(self, input_size=2, hidden_size=4, output_size=1):
        super(XORNet, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """Forward pass through the network"""
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x


# ============================================================================
# DATA PREPARATION
# ============================================================================

print("="*70)
print("XOR PROBLEM USING PYTORCH NEURAL NETWORK")
print("="*70)

# XOR truth table
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=np.float32)

y = np.array([[0],
              [1],
              [1],
              [0]], dtype=np.float32)

print("\nXOR Gate Truth Table:")
print("Input 1 | Input 2 | Output")
print("-" * 28)
for i in range(len(X)):
    print(f"  {int(X[i][0])}     |   {int(X[i][1])}     |   {int(y[i][0])}")

# Convert to PyTorch tensors
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

print(f"\nDataset shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# ============================================================================
# MODEL TRAINING
# ============================================================================

print("\n" + "="*70)
print("TRAINING NEURAL NETWORK")
print("="*70)

# Initialize model
model = XORNet(input_size=2, hidden_size=4, output_size=1)

# Loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Training parameters
epochs = 5000
print_every = 500

# Lists to store training history
losses = []
accuracies = []

print(f"\nTraining for {epochs} epochs...")
print(f"Architecture: 2 -> 4 -> 1")
print(f"Activation: Sigmoid")
print(f"Loss: Binary Cross Entropy")
print(f"Optimizer: Adam (lr=0.1)")
print("\n" + "-"*70)

# Training loop
for epoch in range(epochs):
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Calculate accuracy
    predicted = (outputs > 0.5).float()
    accuracy = (predicted == y_tensor).float().mean() * 100
    
    # Store metrics
    losses.append(loss.item())
    accuracies.append(accuracy.item())
    
    # Print progress
    if (epoch + 1) % print_every == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {loss.item():.4f} | Accuracy: {accuracy.item():.2f}%")

print("-"*70)

# ============================================================================
# EVALUATION
# ============================================================================

print("\n" + "="*70)
print("EVALUATION RESULTS")
print("="*70)

# Set model to evaluation mode
model.eval()

with torch.no_grad():
    predictions = model(X_tensor)
    predicted_classes = (predictions > 0.5).float()
    final_accuracy = (predicted_classes == y_tensor).float().mean() * 100

print("\nFinal Predictions:")
print("Input 1 | Input 2 | Expected | Predicted | Probability")
print("-" * 60)
for i in range(len(X)):
    prob = predictions[i].item()
    pred_class = int(predicted_classes[i].item())
    expected = int(y[i][0])
    print(f"  {int(X[i][0])}     |   {int(X[i][1])}     |    {expected}     |     {pred_class}     |   {prob:.4f}")

print(f"\nFinal Accuracy: {final_accuracy:.2f}%")
print(f"Final Loss: {losses[-1]:.4f}")

# Print learned weights
print("\n" + "="*70)
print("LEARNED PARAMETERS")
print("="*70)

print("\nHidden Layer Weights:")
print(model.hidden.weight.data.numpy())
print("\nHidden Layer Bias:")
print(model.hidden.bias.data.numpy())

print("\nOutput Layer Weights:")
print(model.output.weight.data.numpy())
print("\nOutput Layer Bias:")
print(model.output.bias.data.numpy())

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

# Plot 1: Training Loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses, 'b-', linewidth=1.5, alpha=0.7)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Plot 2: Training Accuracy
plt.subplot(1, 2, 2)
plt.plot(accuracies, 'g-', linewidth=1.5, alpha=0.7)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Training Accuracy Over Time', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ylim([0, 105])

plt.tight_layout()
plt.savefig('/home/runner/work/Deep-Learning/Deep-Learning/Lab01/xor_training_history.png', dpi=150, bbox_inches='tight')
print("\n✓ Training history saved as 'xor_training_history.png'")

# Plot 3: Decision Boundary
plt.figure(figsize=(10, 8))

# Create a mesh grid
x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5
h = 0.01

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict for all points in the mesh
grid_points = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
with torch.no_grad():
    Z = model(grid_points)
Z = Z.numpy().reshape(xx.shape)

# Plot decision boundary
plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.6)
plt.colorbar(label='Probability')

# Plot data points
for i in range(len(X)):
    if y[i] == 0:
        plt.scatter(X[i, 0], X[i, 1], c='red', marker='o', s=200, 
                   edgecolors='black', linewidth=2, label='Class 0' if i == 0 else '')
    else:
        plt.scatter(X[i, 0], X[i, 1], c='blue', marker='s', s=200, 
                   edgecolors='black', linewidth=2, label='Class 1' if i == 1 else '')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel('Input 1', fontsize=12)
plt.ylabel('Input 2', fontsize=12)
plt.title('XOR Problem - Neural Network Decision Boundary', fontsize=14, fontweight='bold')
plt.legend(fontsize=10, loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/runner/work/Deep-Learning/Deep-Learning/Lab01/xor_decision_boundary.png', dpi=150, bbox_inches='tight')
print("✓ Decision boundary saved as 'xor_decision_boundary.png'")

# Plot 4: Network Architecture Visualization
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

# Layer positions
input_layer_x = 0.1
hidden_layer_x = 0.5
output_layer_x = 0.9

# Draw neurons
def draw_neuron(ax, x, y, label=''):
    circle = plt.Circle((x, y), 0.05, color='lightblue', ec='black', linewidth=2)
    ax.add_patch(circle)
    if label:
        ax.text(x, y-0.15, label, ha='center', fontsize=10, fontweight='bold')

# Input layer
input_neurons = [(input_layer_x, 0.3), (input_layer_x, 0.7)]
for i, (x, y) in enumerate(input_neurons):
    draw_neuron(ax, x, y, f'x{i+1}')

# Hidden layer
hidden_neurons = [(hidden_layer_x, y) for y in np.linspace(0.2, 0.8, 4)]
for i, (x, y) in enumerate(hidden_neurons):
    draw_neuron(ax, x, y, f'h{i+1}')

# Output layer
output_neuron = (output_layer_x, 0.5)
draw_neuron(ax, output_neuron[0], output_neuron[1], 'output')

# Draw connections
for inp in input_neurons:
    for hid in hidden_neurons:
        ax.plot([inp[0], hid[0]], [inp[1], hid[1]], 'gray', alpha=0.5, linewidth=1)

for hid in hidden_neurons:
    ax.plot([hid[0], output_neuron[0]], [hid[1], output_neuron[1]], 'gray', alpha=0.5, linewidth=1)

# Add layer labels
ax.text(input_layer_x, 0.05, 'Input Layer\n(2 neurons)', ha='center', fontsize=11, fontweight='bold')
ax.text(hidden_layer_x, 0.05, 'Hidden Layer\n(4 neurons)', ha='center', fontsize=11, fontweight='bold')
ax.text(output_layer_x, 0.05, 'Output Layer\n(1 neuron)', ha='center', fontsize=11, fontweight='bold')

plt.title('XOR Neural Network Architecture', fontsize=14, fontweight='bold', pad=20)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('/home/runner/work/Deep-Learning/Deep-Learning/Lab01/xor_architecture.png', dpi=150, bbox_inches='tight')
print("✓ Network architecture saved as 'xor_architecture.png'")

# ============================================================================
# SAVE MODEL
# ============================================================================

torch.save(model.state_dict(), '/home/runner/work/Deep-Learning/Deep-Learning/Lab01/xor_model.pth')
print("✓ Model weights saved as 'xor_model.pth'")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"✓ XOR problem solved successfully with {final_accuracy:.2f}% accuracy")
print("✓ XOR is NOT linearly separable, requiring a multi-layer network")
print("✓ The hidden layer learns a non-linear representation")
print("✓ The network successfully learned the XOR function")
print("="*70)

# Close all plots
plt.close('all')
