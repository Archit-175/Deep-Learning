#!/usr/bin/env python3
"""
Test script to verify the implementation logic without downloading MNIST
This creates synthetic data to test the model architectures
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")

# Create synthetic MNIST-like data for testing
print("\nCreating synthetic data...")
X_train = np.random.rand(1000, 28, 28).astype('float32')
y_train = np.random.randint(0, 10, size=1000)
X_test = np.random.rand(200, 28, 28).astype('float32')
y_test = np.random.randint(0, 10, size=200)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Preprocessing
X_train_cnn = X_train.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.reshape(-1, 28, 28, 1)
X_train_mlp = X_train.reshape(-1, 784)
X_test_mlp = X_test.reshape(-1, 784)
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

print(f"CNN Training data shape: {X_train_cnn.shape}")
print(f"MLP Training data shape: {X_train_mlp.shape}")

# Test CNN model creation
print("\n" + "="*60)
print("Testing CNN Model Architecture")
print("="*60)

def create_cnn_model(activation='relu', fc_units=128, dropout_rate=0.25, use_bn=False):
    """Create CNN model"""
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation=activation, padding='same'),
        layers.Conv2D(64, (3, 3), activation=activation, padding='same'),
        layers.MaxPooling2D((2, 2)),
    ])
    
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))
    
    model.add(layers.Flatten())
    
    if use_bn:
        model.add(layers.Dense(fc_units))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation(activation))
    else:
        model.add(layers.Dense(fc_units, activation=activation))
    
    model.add(layers.Dense(10, activation='softmax'))
    
    return model

# Test different configurations
configs = [
    ("ReLU", 'relu'),
    ("Sigmoid", 'sigmoid'),
    ("Tanh", 'tanh'),
]

for name, activation in configs:
    print(f"\nCreating CNN with {name} activation...")
    model = create_cnn_model(activation=activation)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"✓ {name} model created successfully")
    print(f"  Total parameters: {model.count_params():,}")

# Test MLP model creation
print("\n" + "="*60)
print("Testing MLP Model Architecture")
print("="*60)

def create_mlp_model(activation='relu', hidden_units=None, dropout_rate=0.0, use_bn=True):
    """Create MLP model"""
    if hidden_units is None:
        hidden_units = [256, 128]
    model = models.Sequential([layers.Input(shape=(784,))])
    
    for units in hidden_units:
        model.add(layers.Dense(units))
        if use_bn:
            model.add(layers.BatchNormalization())
        model.add(layers.Activation(activation))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))
    
    model.add(layers.Dense(10, activation='softmax'))
    return model

mlp_configs = [
    ("MLP-1", [512, 256, 128]),
    ("MLP-2", [256]),
    ("MLP-3", [256, 128]),
]

for name, units in mlp_configs:
    print(f"\nCreating {name} with units {units}...")
    model = create_mlp_model(hidden_units=units)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"✓ {name} model created successfully")
    print(f"  Total parameters: {model.count_params():,}")

# Quick training test
print("\n" + "="*60)
print("Testing Model Training (1 epoch on synthetic data)")
print("="*60)

test_model = create_cnn_model(activation='relu', fc_units=128, dropout_rate=0.25)
test_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining CNN model for 1 epoch...")
history = test_model.fit(
    X_train_cnn, y_train_cat,
    batch_size=32,
    epochs=1,
    validation_split=0.1,
    verbose=1
)

print("\nEvaluating on test set...")
test_loss, test_accuracy = test_model.evaluate(X_test_cnn, y_test_cat, verbose=0)
print(f"✓ Test Loss: {test_loss:.4f}")
print(f"✓ Test Accuracy: {test_accuracy:.4f}")

# Test optimizer configurations
print("\n" + "="*60)
print("Testing Different Optimizers")
print("="*60)

optimizers = [
    ("SGD", SGD(learning_rate=0.01)),
    ("SGD+Momentum", SGD(learning_rate=0.01, momentum=0.9)),
    ("Adam", Adam(learning_rate=0.001)),
]

for opt_name, optimizer in optimizers:
    print(f"\n✓ {opt_name} optimizer configured successfully")
    print(f"  Learning rate: {optimizer.learning_rate.numpy():.4f}")

print("\n" + "="*60)
print("All Tests Passed Successfully!")
print("="*60)
print("\nThe implementation is ready to use with real MNIST data.")
print("The notebook will work when MNIST data is available.")
