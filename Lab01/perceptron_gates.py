"""
Task 3: Implementation of AND and OR Gates using Perceptron
A perceptron is a simple binary classifier that can learn linearly separable patterns
"""

import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    """
    Simple Perceptron implementation for binary classification
    """
    def __init__(self, learning_rate=0.1, epochs=100):
        """
        Initialize the perceptron
        
        Args:
            learning_rate: Learning rate for weight updates
            epochs: Number of training iterations
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.errors = []
        
    def activation(self, x):
        """
        Step activation function
        Returns 1 if x >= 0, else 0
        """
        return 1 if x >= 0 else 0
    
    def predict(self, X):
        """
        Make predictions for input X
        
        Args:
            X: Input features (n_samples, n_features)
        
        Returns:
            Predictions array
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return np.array([self.activation(x) for x in linear_output])
    
    def fit(self, X, y):
        """
        Train the perceptron
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Training loop
        for epoch in range(self.epochs):
            epoch_errors = 0
            
            for idx, x_i in enumerate(X):
                # Calculate prediction
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation(linear_output)
                
                # Update weights and bias if prediction is wrong
                error = y[idx] - y_predicted
                if error != 0:
                    epoch_errors += 1
                    self.weights += self.learning_rate * error * x_i
                    self.bias += self.learning_rate * error
            
            self.errors.append(epoch_errors)
            
            # Early stopping if no errors
            if epoch_errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                break
    
    def get_params(self):
        """Return learned weights and bias"""
        return self.weights, self.bias


def plot_decision_boundary(X, y, perceptron, title):
    """
    Plot the decision boundary of the perceptron
    
    Args:
        X: Input features
        y: True labels
        perceptron: Trained perceptron model
        title: Plot title
    """
    plt.figure(figsize=(8, 6))
    
    # Plot data points
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', marker='o', s=100, label='Class 0', edgecolors='k')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='s', s=100, label='Class 1', edgecolors='k')
    
    # Plot decision boundary
    w = perceptron.weights
    b = perceptron.bias
    
    x_min, x_max = -0.5, 1.5
    x_line = np.linspace(x_min, x_max, 100)
    
    # Decision boundary: w1*x1 + w2*x2 + b = 0
    # Solving for x2: x2 = -(w1*x1 + b) / w2
    if w[1] != 0:
        y_line = -(w[0] * x_line + b) / w[1]
        plt.plot(x_line, y_line, 'g--', linewidth=2, label='Decision Boundary')
    
    plt.xlim(x_min, x_max)
    plt.ylim(-0.5, 1.5)
    plt.xlabel('Input 1', fontsize=12)
    plt.ylabel('Input 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_training_errors(errors, title):
    """
    Plot training errors over epochs
    
    Args:
        errors: List of errors per epoch
        title: Plot title
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(errors) + 1), errors, 'b-', linewidth=2, marker='o')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Number of Errors', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


# ============================================================================
# AND GATE IMPLEMENTATION
# ============================================================================
print("="*70)
print("AND GATE USING PERCEPTRON")
print("="*70)

# AND gate truth table
X_and = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
y_and = np.array([0, 0, 0, 1])

print("\nAND Gate Truth Table:")
print("Input 1 | Input 2 | Output")
print("-" * 28)
for i in range(len(X_and)):
    print(f"  {X_and[i][0]}     |   {X_and[i][1]}     |   {y_and[i]}")

# Train perceptron for AND gate
perceptron_and = Perceptron(learning_rate=0.1, epochs=100)
print("\nTraining AND gate perceptron...")
perceptron_and.fit(X_and, y_and)

# Get predictions
predictions_and = perceptron_and.predict(X_and)
weights_and, bias_and = perceptron_and.get_params()

print(f"\nLearned Parameters:")
print(f"Weights: {weights_and}")
print(f"Bias: {bias_and}")

print("\nPredictions:")
print("Input 1 | Input 2 | Expected | Predicted")
print("-" * 45)
for i in range(len(X_and)):
    print(f"  {X_and[i][0]}     |   {X_and[i][1]}     |    {y_and[i]}     |     {predictions_and[i]}")

# Calculate accuracy
accuracy_and = np.mean(predictions_and == y_and) * 100
print(f"\nAccuracy: {accuracy_and:.2f}%")

# ============================================================================
# OR GATE IMPLEMENTATION
# ============================================================================
print("\n" + "="*70)
print("OR GATE USING PERCEPTRON")
print("="*70)

# OR gate truth table
X_or = np.array([[0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]])
y_or = np.array([0, 1, 1, 1])

print("\nOR Gate Truth Table:")
print("Input 1 | Input 2 | Output")
print("-" * 28)
for i in range(len(X_or)):
    print(f"  {X_or[i][0]}     |   {X_or[i][1]}     |   {y_or[i]}")

# Train perceptron for OR gate
perceptron_or = Perceptron(learning_rate=0.1, epochs=100)
print("\nTraining OR gate perceptron...")
perceptron_or.fit(X_or, y_or)

# Get predictions
predictions_or = perceptron_or.predict(X_or)
weights_or, bias_or = perceptron_or.get_params()

print(f"\nLearned Parameters:")
print(f"Weights: {weights_or}")
print(f"Bias: {bias_or}")

print("\nPredictions:")
print("Input 1 | Input 2 | Expected | Predicted")
print("-" * 45)
for i in range(len(X_or)):
    print(f"  {X_or[i][0]}     |   {X_or[i][1]}     |    {y_or[i]}     |     {predictions_or[i]}")

# Calculate accuracy
accuracy_or = np.mean(predictions_or == y_or) * 100
print(f"\nAccuracy: {accuracy_or:.2f}%")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

# Plot AND gate
plot_decision_boundary(X_and, y_and, perceptron_and, 'AND Gate - Perceptron Decision Boundary')
plt.savefig('/home/runner/work/Deep-Learning/Deep-Learning/Lab01/and_gate_boundary.png', dpi=150, bbox_inches='tight')
print("\n✓ AND gate decision boundary saved as 'and_gate_boundary.png'")

plot_training_errors(perceptron_and.errors, 'AND Gate - Training Errors')
plt.savefig('/home/runner/work/Deep-Learning/Deep-Learning/Lab01/and_gate_errors.png', dpi=150, bbox_inches='tight')
print("✓ AND gate training errors saved as 'and_gate_errors.png'")

# Plot OR gate
plot_decision_boundary(X_or, y_or, perceptron_or, 'OR Gate - Perceptron Decision Boundary')
plt.savefig('/home/runner/work/Deep-Learning/Deep-Learning/Lab01/or_gate_boundary.png', dpi=150, bbox_inches='tight')
print("✓ OR gate decision boundary saved as 'or_gate_boundary.png'")

plot_training_errors(perceptron_or.errors, 'OR Gate - Training Errors')
plt.savefig('/home/runner/work/Deep-Learning/Deep-Learning/Lab01/or_gate_errors.png', dpi=150, bbox_inches='tight')
print("✓ OR gate training errors saved as 'or_gate_errors.png'")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"AND Gate Accuracy: {accuracy_and:.2f}%")
print(f"OR Gate Accuracy: {accuracy_or:.2f}%")
print("\nBoth AND and OR gates are linearly separable and can be")
print("learned perfectly by a single perceptron.")
print("="*70)

# Close all plots
plt.close('all')
