"""
Lab Practical 3: Comparative Analysis of Different CNN Architectures
Deep Learning (AI302)

This script implements and compares multiple CNN architectures on benchmark datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ========================== Part 1: CNN Architectures ==========================

class LeNet5(nn.Module):
    """LeNet-5 Architecture"""
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AlexNet(nn.Module):
    """AlexNet Architecture (adapted for CIFAR-10)"""
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGGNet(nn.Module):
    """VGG-like Network (adapted for CIFAR-10)"""
    def __init__(self, num_classes=10):
        super(VGGNet, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    """Residual Block for ResNet"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet Architecture"""
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x, return_features=False):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        features = x.view(x.size(0), -1)
        x = self.fc(features)
        if return_features:
            return x, features
        return x


def ResNet50(num_classes=10):
    """ResNet-50"""
    return ResNet(ResidualBlock, [3, 4, 6, 3], num_classes)


def ResNet100(num_classes=10):
    """ResNet-100"""
    return ResNet(ResidualBlock, [3, 13, 23, 3], num_classes)


# ========================== Part 2: Loss Functions ==========================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class ArcFaceLoss(nn.Module):
    """ArcFace Loss for feature learning"""
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, features, labels):
        # Normalize features and weights
        features = F.normalize(features, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cosine = F.linear(features, weight)
        
        # Convert to angle
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        
        # Add margin
        one_hot = F.one_hot(labels, self.out_features).float()
        theta_m = theta + self.m * one_hot
        
        # Convert back to cosine
        cosine_m = torch.cos(theta_m)
        
        # Scale
        output = cosine_m * self.s
        
        return F.cross_entropy(output, labels)


# ========================== Dataset Loading ==========================

def get_cifar10_loaders(batch_size=128):
    """Load CIFAR-10 dataset"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader


def get_mnist_loaders(batch_size=128):
    """Load MNIST dataset"""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(3),  # Convert to 3 channels
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                         download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader


# ========================== Training Functions ==========================

def train_model(model, trainloader, criterion, optimizer, epochs, device, model_name="Model"):
    """Train a model"""
    model.to(device)
    model.train()
    
    train_losses = []
    train_accuracies = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(trainloader, desc=f'{model_name} Epoch {epoch+1}/{epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass - handle ArcFace differently
            if isinstance(criterion, ArcFaceLoss):
                if hasattr(model, 'fc'):
                    # For ResNet-like models, get features before final layer
                    features = model(inputs, return_features=True)[1] if hasattr(model.forward, '__code__') and 'return_features' in model.forward.__code__.co_varnames else model(inputs)
                    if features.dim() == 1:
                        features = features.unsqueeze(0)
                else:
                    features = model(inputs)
                loss = criterion(features, labels)
                
                # Get outputs for accuracy calculation
                with torch.no_grad():
                    outputs = model(inputs)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                            'acc': f'{100.*correct/total:.2f}%'})
        
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100. * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        print(f'{model_name} Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    
    return train_losses, train_accuracies


def evaluate_model(model, testloader, device):
    """Evaluate a model"""
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


# ========================== Part 3: Visualization ==========================

def extract_features(model, dataloader, device, num_samples=1000):
    """Extract features for visualization"""
    model.eval()
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            if len(features_list) * inputs.size(0) >= num_samples:
                break
            
            inputs = inputs.to(device)
            
            # Get features before final layer
            if hasattr(model, 'fc'):
                # For ResNet-like models
                if 'return_features' in model.forward.__code__.co_varnames:
                    _, features = model(inputs, return_features=True)
                else:
                    # Manually extract features
                    x = inputs
                    if hasattr(model, 'conv1'):
                        x = F.relu(model.bn1(model.conv1(x)))
                        x = model.layer1(x)
                        x = model.layer2(x)
                        x = model.layer3(x)
                        x = model.layer4(x)
                        x = model.avg_pool(x)
                        features = x.view(x.size(0), -1)
                    else:
                        features = model(inputs)
            elif hasattr(model, 'features'):
                # For VGG/AlexNet-like models
                features = model.features(inputs)
                features = features.view(features.size(0), -1)
            else:
                features = model(inputs)
            
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
    
    features = np.concatenate(features_list, axis=0)[:num_samples]
    labels = np.concatenate(labels_list, axis=0)[:num_samples]
    
    return features, labels


def plot_tsne(features, labels, title, save_path):
    """Plot t-SNE visualization"""
    print(f"Computing t-SNE for {title}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6, s=20)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE plot to {save_path}")


# ========================== Main Execution ==========================

def main():
    print("="*80)
    print("Lab Practical 3: Comparative Analysis of CNN Architectures")
    print("="*80)
    
    # Configuration
    batch_size = 128
    results = []
    
    # Load CIFAR-10 dataset
    print("\nLoading CIFAR-10 dataset...")
    trainloader, testloader = get_cifar10_loaders(batch_size)
    
    # ========================== Part 2: Loss Function Comparison ==========================
    
    print("\n" + "="*80)
    print("Part 2: Loss Function and Optimizer Comparison")
    print("="*80)
    
    experiments = [
        {
            'model': VGGNet(num_classes=10),
            'model_name': 'VGGNet',
            'optimizer_name': 'Adam',
            'loss_name': 'BCE',
            'epochs': 10
        },
        {
            'model': AlexNet(num_classes=10),
            'model_name': 'AlexNet',
            'loss_name': 'Focal Loss',
            'optimizer_name': 'SGD',
            'epochs': 20
        },
        {
            'model': ResNet50(num_classes=10),
            'model_name': 'ResNet',
            'loss_name': 'ArcFace',
            'optimizer_name': 'Adam',
            'epochs': 15
        }
    ]
    
    trained_models = {}
    
    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Training {exp['model_name']} with {exp['optimizer_name']} and {exp['loss_name']}")
        print(f"{'='*60}")
        
        model = exp['model']
        
        # Setup loss function
        if exp['loss_name'] == 'BCE':
            criterion = nn.CrossEntropyLoss()  # For multi-class, CE is more appropriate than BCE
        elif exp['loss_name'] == 'Focal Loss':
            criterion = FocalLoss(alpha=1, gamma=2)
        elif exp['loss_name'] == 'ArcFace':
            criterion = ArcFaceLoss(in_features=512, out_features=10, s=30.0, m=0.50)
        
        # Setup optimizer
        if exp['optimizer_name'] == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=0.001)
        elif exp['optimizer_name'] == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        # Train model
        train_losses, train_accuracies = train_model(
            model, trainloader, criterion, optimizer, 
            exp['epochs'], device, exp['model_name']
        )
        
        # Evaluate model
        test_accuracy = evaluate_model(model, testloader, device)
        
        result = {
            'Model': exp['model_name'],
            'Optimizer': exp['optimizer_name'],
            'Epochs': exp['epochs'],
            'Loss Function': exp['loss_name'],
            'Training Accuracy': f"{train_accuracies[-1]:.2f}%",
            'Testing Accuracy': f"{test_accuracy:.2f}%"
        }
        results.append(result)
        trained_models[f"{exp['model_name']}_{exp['loss_name']}"] = model
        
        print(f"\n{exp['model_name']} Results:")
        print(f"  Training Accuracy: {train_accuracies[-1]:.2f}%")
        print(f"  Testing Accuracy: {test_accuracy:.2f}%")
    
    # ========================== Results Table ==========================
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Model':<12} {'Optimizer':<12} {'Epochs':<8} {'Loss Function':<15} {'Train Acc':<12} {'Test Acc':<12}")
    print("-"*80)
    for result in results:
        print(f"{result['Model']:<12} {result['Optimizer']:<12} {result['Epochs']:<8} {result['Loss Function']:<15} {result['Training Accuracy']:<12} {result['Testing Accuracy']:<12}")
    print("="*80)
    
    # ========================== Part 3: t-SNE Visualization ==========================
    
    print("\n" + "="*80)
    print("Part 3: Feature Visualization with t-SNE")
    print("="*80)
    
    # Visualize features from different loss functions
    for model_key, model in trained_models.items():
        print(f"\nExtracting features from {model_key}...")
        features, labels = extract_features(model, testloader, device, num_samples=1000)
        
        plot_tsne(
            features, labels, 
            f"t-SNE Visualization: {model_key}", 
            f"tsne_{model_key.replace(' ', '_').lower()}.png"
        )
    
    print("\n" + "="*80)
    print("Lab Practical 3 Completed Successfully!")
    print("="*80)
    print("\nGenerated Files:")
    print("  - tsne_vggnet_bce.png")
    print("  - tsne_alexnet_focal_loss.png")
    print("  - tsne_resnet_arcface.png")


if __name__ == "__main__":
    main()
