#!/usr/bin/env python3
"""
Lab 02 - MNIST Handwritten Digit Recognition
Complete implementation of all three tasks from DL_Practical-2(1).pdf

This script can be run as a standalone Python program or used within Jupyter notebook.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD, Adam
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("="*80)
print("LAB 02 - MNIST HANDWRITTEN DIGIT RECOGNITION")
print("="*80)
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")


def load_and_preprocess_data():
    """Load and preprocess MNIST dataset"""
    print("\n" + "="*80)
    print("LOADING AND PREPROCESSING DATA")
    print("="*80)
    
    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    # Preprocessing for CNN
    X_train_cnn = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test_cnn = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # Preprocessing for MLP
    X_train_mlp = X_train.reshape(-1, 784).astype('float32') / 255.0
    X_test_mlp = X_test.reshape(-1, 784).astype('float32') / 255.0
    
    # One-hot encode labels
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)
    
    return (X_train_cnn, X_train_mlp, y_train_cat), (X_test_cnn, X_test_mlp, y_test_cat), (X_train, y_train), (X_test, y_test)


def create_cnn_model(activation='relu', fc_units=128, dropout_rate=0.25, use_bn=False):
    """
    Create CNN model based on the base architecture:
    - Conv2D Layer 1: 32 filters, 3x3 kernel
    - Conv2D Layer 2: 64 filters, 3x3 kernel
    - Max Pooling: 2x2 kernel
    - Dropout
    - Dense Layer: fc_units neurons
    - Output Layer: 10 neurons (Softmax)
    """
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


def create_mlp_model(activation='relu', hidden_units=[256, 128], dropout_rate=0.0, use_bn=True):
    """
    Create MLP model based on the base architecture:
    - Flatten (784)
    - Dense layers with BatchNormalization (optional)
    - Output Layer: 10 neurons (Softmax)
    """
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


def train_and_evaluate(model, X_train, y_train, X_test, y_test, optimizer, epochs, batch_size=128, verbose=1):
    """Train and evaluate a model"""
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        verbose=verbose
    )
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    return history, test_accuracy


def plot_history(histories, labels, title, save_name):
    """Plot training history for multiple experiments"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training loss
    for history, label in zip(histories, labels):
        axes[0].plot(history.history['loss'], label=f'{label} (train)', linewidth=2)
        axes[0].plot(history.history['val_loss'], label=f'{label} (val)', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot training accuracy
    for history, label in zip(histories, labels):
        axes[1].plot(history.history['accuracy'], label=f'{label} (train)', linewidth=2)
        axes[1].plot(history.history['val_accuracy'], label=f'{label} (val)', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_name, dpi=100, bbox_inches='tight')
    print(f"✓ Plot saved: {save_name}")
    plt.close()


def task1_activation_comparison(X_train_cnn, y_train_cat, X_test_cnn, y_test_cat):
    """Task 1: Compare Sigmoid, Tanh, and ReLU activation functions"""
    print("\n" + "="*80)
    print("TASK 1: ACTIVATION FUNCTION COMPARISON")
    print("="*80)
    
    activation_functions = ['sigmoid', 'tanh', 'relu']
    results = []
    histories = []
    
    for activation in activation_functions:
        print(f"\n{'='*60}")
        print(f"Training CNN with {activation.upper()} activation...")
        print(f"{'='*60}")
        
        model = create_cnn_model(activation=activation, fc_units=128, dropout_rate=0.25)
        optimizer = Adam(learning_rate=0.001)
        
        history, test_acc = train_and_evaluate(
            model, X_train_cnn, y_train_cat, X_test_cnn, y_test_cat,
            optimizer=optimizer, epochs=10, verbose=2
        )
        
        histories.append(history)
        results.append({
            'Experiment': f'CNN-{activation}',
            'Activation': activation,
            'Optimizer': 'Adam',
            'Epochs': 10,
            'Final Test Accuracy': f"{test_acc:.4f}"
        })
        
        print(f"✓ {activation.upper()}: Test Accuracy = {test_acc:.4f}")
    
    # Plot comparison
    plot_history(
        histories,
        activation_functions,
        'Task 1: Activation Function Comparison',
        'task1_activation_comparison.png'
    )
    
    # Display results table
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("TASK 1 RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    
    return df


def task2_optimizer_comparison(X_train_cnn, y_train_cat, X_test_cnn, y_test_cat):
    """Task 2: Compare SGD, SGD+Momentum, and Adam optimizers"""
    print("\n" + "="*80)
    print("TASK 2: OPTIMIZER COMPARISON (with ReLU)")
    print("="*80)
    
    optimizers_config = [
        ('SGD', SGD(learning_rate=0.01)),
        ('SGD+Momentum', SGD(learning_rate=0.01, momentum=0.9)),
        ('Adam', Adam(learning_rate=0.001))
    ]
    
    results = []
    histories = []
    
    for opt_name, optimizer in optimizers_config:
        print(f"\n{'='*60}")
        print(f"Training CNN with {opt_name} optimizer...")
        print(f"{'='*60}")
        
        model = create_cnn_model(activation='relu', fc_units=128, dropout_rate=0.25)
        
        history, test_acc = train_and_evaluate(
            model, X_train_cnn, y_train_cat, X_test_cnn, y_test_cat,
            optimizer=optimizer, epochs=10, verbose=2
        )
        
        histories.append(history)
        results.append({
            'Experiment': f'CNN-{opt_name}',
            'Activation': 'ReLU',
            'Optimizer': opt_name,
            'Epochs': 10,
            'Final Test Accuracy': f"{test_acc:.4f}"
        })
        
        print(f"✓ {opt_name}: Test Accuracy = {test_acc:.4f}")
    
    # Plot comparison
    plot_history(
        histories,
        [opt[0] for opt in optimizers_config],
        'Task 2: Optimizer Comparison (ReLU Activation)',
        'task2_optimizer_comparison.png'
    )
    
    # Display results table
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("TASK 2 RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    
    return df


def task3_bn_dropout_comparison(X_train_cnn, y_train_cat, X_test_cnn, y_test_cat):
    """Task 3: Batch Normalization and Dropout experiments"""
    print("\n" + "="*80)
    print("TASK 3: BATCH NORMALIZATION AND DROPOUT EXPERIMENTS")
    print("="*80)
    
    configs = [
        ('No BN, No Dropout', False, 0.0),
        ('No BN, Dropout=0.1', False, 0.1),
        ('With BN, Dropout=0.25', True, 0.25)
    ]
    
    results = []
    histories = []
    
    for config_name, use_bn, dropout_rate in configs:
        print(f"\n{'='*60}")
        print(f"Training CNN with {config_name}...")
        print(f"{'='*60}")
        
        model = create_cnn_model(
            activation='relu',
            fc_units=128,
            dropout_rate=dropout_rate,
            use_bn=use_bn
        )
        optimizer = Adam(learning_rate=0.001)
        
        history, test_acc = train_and_evaluate(
            model, X_train_cnn, y_train_cat, X_test_cnn, y_test_cat,
            optimizer=optimizer, epochs=10, verbose=2
        )
        
        histories.append(history)
        results.append({
            'Experiment': config_name,
            'Batch Normalization': 'Yes' if use_bn else 'No',
            'Dropout Rate': dropout_rate,
            'Epochs': 10,
            'Final Test Accuracy': f"{test_acc:.4f}"
        })
        
        print(f"✓ {config_name}: Test Accuracy = {test_acc:.4f}")
    
    # Plot comparison
    plot_history(
        histories,
        [config[0] for config in configs],
        'Task 3: Batch Normalization and Dropout Comparison',
        'task3_bn_dropout_comparison.png'
    )
    
    # Display results table
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("TASK 3 RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    
    return df


def additional_experiments(X_train_cnn, y_train_cat, X_test_cnn, y_test_cat,
                          X_train_mlp, X_test_mlp):
    """Additional experiments as per assignment table"""
    print("\n" + "="*80)
    print("ADDITIONAL EXPERIMENTS (Assignment Configurations)")
    print("="*80)
    
    results = []
    
    # CNN-1: 128 FC, Adam, 10 epochs
    print("\n" + "="*60)
    print("Training CNN-1 (FC=128, Adam, 10 epochs)...")
    print("="*60)
    model_cnn1 = create_cnn_model(activation='relu', fc_units=128, dropout_rate=0.25, use_bn=True)
    history_cnn1, acc_cnn1 = train_and_evaluate(
        model_cnn1, X_train_cnn, y_train_cat, X_test_cnn, y_test_cat,
        optimizer=Adam(learning_rate=0.001), epochs=10, verbose=2
    )
    results.append({
        'Model': 'CNN-1',
        'FC Layer': '128',
        'Optimizer': 'Adam',
        'Epochs': 10,
        'Test Accuracy': f"{acc_cnn1:.4f}"
    })
    print(f"✓ CNN-1: Test Accuracy = {acc_cnn1:.4f}")
    
    # MLP-1: 512-256-128, SGD, 20 epochs
    print("\n" + "="*60)
    print("Training MLP-1 (512-256-128, SGD, 20 epochs)...")
    print("="*60)
    model_mlp1 = create_mlp_model(activation='relu', hidden_units=[512, 256, 128], dropout_rate=0.0, use_bn=True)
    history_mlp1, acc_mlp1 = train_and_evaluate(
        model_mlp1, X_train_mlp, y_train_cat, X_test_mlp, y_test_cat,
        optimizer=SGD(learning_rate=0.01, momentum=0.9), epochs=20, verbose=2
    )
    results.append({
        'Model': 'MLP-1',
        'FC Layer': '512-256-128',
        'Optimizer': 'SGD',
        'Epochs': 20,
        'Test Accuracy': f"{acc_mlp1:.4f}"
    })
    print(f"✓ MLP-1: Test Accuracy = {acc_mlp1:.4f}")
    
    # MLP-2: 256, Adam, 15 epochs
    print("\n" + "="*60)
    print("Training MLP-2 (256, Adam, 15 epochs)...")
    print("="*60)
    model_mlp2 = create_mlp_model(activation='relu', hidden_units=[256], dropout_rate=0.0, use_bn=True)
    history_mlp2, acc_mlp2 = train_and_evaluate(
        model_mlp2, X_train_mlp, y_train_cat, X_test_mlp, y_test_cat,
        optimizer=Adam(learning_rate=0.001), epochs=15, verbose=2
    )
    results.append({
        'Model': 'MLP-2',
        'FC Layer': '256',
        'Optimizer': 'Adam',
        'Epochs': 15,
        'Test Accuracy': f"{acc_mlp2:.4f}"
    })
    print(f"✓ MLP-2: Test Accuracy = {acc_mlp2:.4f}")
    
    # Display results table
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("ADDITIONAL EXPERIMENTS RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    
    return df


def main():
    """Main execution function"""
    try:
        # Load and preprocess data
        (X_train_cnn, X_train_mlp, y_train_cat), (X_test_cnn, X_test_mlp, y_test_cat), _, _ = load_and_preprocess_data()
        
        # Run all tasks
        task1_results = task1_activation_comparison(X_train_cnn, y_train_cat, X_test_cnn, y_test_cat)
        task2_results = task2_optimizer_comparison(X_train_cnn, y_train_cat, X_test_cnn, y_test_cat)
        task3_results = task3_bn_dropout_comparison(X_train_cnn, y_train_cat, X_test_cnn, y_test_cat)
        additional_results = additional_experiments(X_train_cnn, y_train_cat, X_test_cnn, y_test_cat,
                                                   X_train_mlp, X_test_mlp)
        
        # Print comprehensive summary
        print("\n" + "="*80)
        print("COMPREHENSIVE RESULTS SUMMARY")
        print("="*80)
        
        print("\n" + "-"*80)
        print("TASK 1: Activation Function Comparison")
        print("-"*80)
        print(task1_results.to_string(index=False))
        
        print("\n" + "-"*80)
        print("TASK 2: Optimizer Comparison")
        print("-"*80)
        print(task2_results.to_string(index=False))
        
        print("\n" + "-"*80)
        print("TASK 3: Batch Normalization and Dropout")
        print("-"*80)
        print(task3_results.to_string(index=False))
        
        print("\n" + "-"*80)
        print("Additional Experiments (Assignment Configurations)")
        print("-"*80)
        print(additional_results.to_string(index=False))
        
        print("\n" + "="*80)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nGenerated files:")
        print("  - task1_activation_comparison.png")
        print("  - task2_optimizer_comparison.png")
        print("  - task3_bn_dropout_comparison.png")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("\nIf MNIST download failed due to network restrictions,")
        print("you can run this script in an environment with internet access.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
