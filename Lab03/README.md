# Lab 03 - Comparative Analysis of Different CNN Architectures

## 📋 Objective
To implement, train, and evaluate multiple landmark Convolutional Neural Network (CNN) architectures and analyze the impact of network depth, loss functions, and optimization strategies on classification accuracy and computational efficiency.

## 🎯 Tasks

### Part 1: CNN Architecture Implementation
Implement and compare the following CNN architectures on CIFAR-10 dataset:
- LeNet-5
- AlexNet
- VGGNet
- ResNet-50
- ResNet-100
- EfficientNet
- InceptionV3
- MobileNet

### Part 2: Loss Function and Optimizer Comparison
Study how specific loss functions and optimization strategies impact convergence and final accuracy:
- **VGGNet** with Adam optimizer and Binary Cross-Entropy (10 epochs)
- **AlexNet** with SGD optimizer and Focal Loss (20 epochs)
- **ResNet** with Adam optimizer and ArcFace Loss (15 epochs)

### Part 3: Feature Visualization
Visualize decision boundaries and feature clustering using t-SNE to understand how different loss functions (BCE, Focal Loss, ArcFace) affect feature learning on CIFAR-10.

## 📂 Files
- `cnn_comparative_analysis.py` - Main implementation file with all architectures and experiments
- `DL_Practical-3 (1).pdf` - Original assignment document
- `tsne_*.png` - t-SNE visualization plots for different models and loss functions

## 🔧 Implementation Details

### CNN Architectures
- **LeNet-5**: Classic architecture with 2 convolutional layers
- **AlexNet**: Deeper architecture with dropout regularization (adapted for CIFAR-10)
- **VGGNet**: VGG-style architecture with repeated conv blocks
- **ResNet-50/100**: Residual networks with skip connections for deeper training

### Loss Functions
1. **Cross-Entropy Loss**: Standard multi-class classification loss
2. **Focal Loss**: Addresses class imbalance by down-weighting easy examples
3. **ArcFace Loss**: Angular margin-based loss for better feature discrimination

### Optimizers
- **Adam**: Adaptive learning rate optimizer
- **SGD with Momentum**: Stochastic gradient descent with momentum

## 📊 Results
The script trains three models with different configurations and outputs:
- Training and testing accuracy for each model
- Comparative results table showing all experiments
- t-SNE visualizations showing feature clustering for each loss function

Expected output format:
```
Model        Optimizer    Epochs   Loss Function   Train Acc    Test Acc
VGGNet       Adam         10       BCE             XX.XX%       XX.XX%
AlexNet      SGD          20       Focal Loss      XX.XX%       XX.XX%
ResNet       Adam         15       ArcFace         XX.XX%       XX.XX%
```

## 🧪 How to Run

### Prerequisites
Install required dependencies:
```bash
pip install -r ../requirements.txt
```

### Run the main script
```bash
cd /home/runner/work/Deep-Learning/Deep-Learning/Lab03
python cnn_comparative_analysis.py
```

The script will:
1. Download CIFAR-10 dataset automatically
2. Train the three models with different configurations
3. Print training progress and results
4. Generate t-SNE visualization plots

### Expected Runtime
- VGGNet (10 epochs): ~15-20 minutes
- AlexNet (20 epochs): ~30-40 minutes
- ResNet (15 epochs): ~25-35 minutes

Total runtime: ~70-95 minutes on CPU, ~10-15 minutes on GPU

## 📝 Observations

### Key Findings
1. **Architecture Depth**: Deeper networks (ResNet) generally achieve better accuracy but require more computational resources
2. **Loss Functions**: 
   - Cross-Entropy works well for balanced datasets
   - Focal Loss helps with difficult examples
   - ArcFace produces more discriminative features with better clustering
3. **Optimization**: Adam converges faster than SGD but may overfit on small datasets
4. **Feature Clustering**: t-SNE visualizations reveal how different loss functions affect feature space organization

### Learning Outcomes
- Understanding of various CNN architectures and their trade-offs
- Impact of loss functions on model performance and feature learning
- Practical experience with PyTorch implementation of deep learning models
- Visualization techniques for understanding learned representations

## 📚 References
- LeCun, Y., et al. (1998). "Gradient-based learning applied to document recognition"
- Krizhevsky, A., et al. (2012). "ImageNet Classification with Deep Convolutional Neural Networks"
- Simonyan, K., & Zisserman, A. (2014). "Very Deep Convolutional Networks for Large-Scale Image Recognition"
- He, K., et al. (2016). "Deep Residual Learning for Image Recognition"
- Lin, T. Y., et al. (2017). "Focal Loss for Dense Object Detection"
- Deng, J., et al. (2019). "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
