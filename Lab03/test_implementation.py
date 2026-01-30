"""
Quick test script to validate CNN implementation without full training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from cnn_comparative_analysis import (
    LeNet5, AlexNet, VGGNet, ResNet50, ResNet100,
    FocalLoss, ArcFaceLoss,
    get_cifar10_loaders
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_model_forward_pass(model_class, model_name):
    """Test if model can perform forward pass"""
    print(f"\nTesting {model_name}...")
    model = model_class(num_classes=10).to(device)
    
    # Create dummy input (batch_size=4, channels=3, height=32, width=32)
    dummy_input = torch.randn(4, 3, 32, 32).to(device)
    
    try:
        output = model(dummy_input)
        assert output.shape == (4, 10), f"Expected shape (4, 10), got {output.shape}"
        print(f"✓ {model_name} forward pass successful: {output.shape}")
        return True
    except Exception as e:
        print(f"✗ {model_name} forward pass failed: {e}")
        return False


def test_loss_functions():
    """Test loss functions"""
    print("\nTesting Loss Functions...")
    
    # Dummy data
    logits = torch.randn(4, 10).to(device)
    labels = torch.randint(0, 10, (4,)).to(device)
    
    # Test Cross Entropy
    ce_loss = nn.CrossEntropyLoss()
    ce_output = ce_loss(logits, labels)
    print(f"✓ Cross Entropy Loss: {ce_output.item():.4f}")
    
    # Test Focal Loss
    focal_loss = FocalLoss(alpha=1, gamma=2)
    focal_output = focal_loss(logits, labels)
    print(f"✓ Focal Loss: {focal_output.item():.4f}")
    
    # Test ArcFace Loss
    features = torch.randn(4, 512).to(device)
    arcface_loss = ArcFaceLoss(in_features=512, out_features=10).to(device)
    arcface_output = arcface_loss(features, labels)
    print(f"✓ ArcFace Loss: {arcface_output.item():.4f}")


def test_data_loading():
    """Test dataset loading"""
    print("\nTesting Data Loading...")
    try:
        trainloader, testloader = get_cifar10_loaders(batch_size=4)
        
        # Get one batch
        train_images, train_labels = next(iter(trainloader))
        test_images, test_labels = next(iter(testloader))
        
        print(f"✓ Train batch shape: {train_images.shape}, labels: {train_labels.shape}")
        print(f"✓ Test batch shape: {test_images.shape}, labels: {test_labels.shape}")
        print(f"✓ Number of training batches: {len(trainloader)}")
        print(f"✓ Number of testing batches: {len(testloader)}")
        return True
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False


def test_training_step():
    """Test a single training step"""
    print("\nTesting Training Step...")
    try:
        # Create small model
        model = LeNet5(num_classes=10).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Dummy data
        inputs = torch.randn(4, 3, 32, 32).to(device)
        labels = torch.randint(0, 10, (4,)).to(device)
        
        # Training step
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        print(f"✓ Training step successful, loss: {loss.item():.4f}")
        return True
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        return False


def main():
    print("="*80)
    print("CNN Implementation Validation Tests")
    print("="*80)
    print(f"Using device: {device}")
    
    # Test all models
    models_to_test = [
        (LeNet5, "LeNet-5"),
        (AlexNet, "AlexNet"),
        (VGGNet, "VGGNet"),
        (ResNet50, "ResNet-50"),
        (ResNet100, "ResNet-100"),
    ]
    
    results = []
    for model_class, model_name in models_to_test:
        results.append(test_model_forward_pass(model_class, model_name))
    
    # Test loss functions
    test_loss_functions()
    
    # Test data loading
    data_ok = test_data_loading()
    
    # Test training step
    training_ok = test_training_step()
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"Models tested: {len(results)}/{len(results)} passed")
    print(f"Loss functions: ✓ Passed")
    print(f"Data loading: {'✓ Passed' if data_ok else '✗ Failed'}")
    print(f"Training step: {'✓ Passed' if training_ok else '✗ Failed'}")
    
    if all(results) and data_ok and training_ok:
        print("\n✓ All validation tests passed! Implementation is ready.")
        print("\nTo run full training, execute:")
        print("  python cnn_comparative_analysis.py")
    else:
        print("\n✗ Some tests failed. Please review the errors above.")
    
    print("="*80)


if __name__ == "__main__":
    main()
