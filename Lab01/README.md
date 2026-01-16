# Lab 01 - Deep Learning Fundamentals

## 📋 Objective
This lab focuses on understanding the fundamentals of deep learning, including tensor operations, perceptron learning, and neural network implementations for both classification and regression problems.

## 🎯 Tasks
Based on DL_Practical-1.pdf, the following tasks are implemented:

1. **Introduction to PyTorch Tensors and Basic Operations**
   - Understand PyTorch tensors, initialization methods, and data types
   - Perform tensor operations: arithmetic, broadcasting, indexing, and reshaping
   - Explore automatic differentiation using PyTorch's Autograd system

2. **TensorFlow Linear Algebra Operations**
   - Matrix creation and manipulation
   - Matrix multiplication, transpose, inverse, determinant
   - Eigenvalues, eigenvectors, SVD, QR decomposition
   - Solving linear systems and various matrix operations

3. **AND/OR Gates using Perceptron**
   - Implement single-layer perceptron from scratch
   - Train perceptron to learn AND gate logic
   - Train perceptron to learn OR gate logic
   - Visualize decision boundaries

4. **XOR Problem using PyTorch Neural Network**
   - Implement multi-layer neural network for XOR problem
   - Demonstrate that XOR requires hidden layers (non-linear separation)
   - Train and evaluate the network
   - Visualize decision boundaries and training progress

5. **Simple Neural Network for Regression**
   - Implement regression neural network using PyTorch
   - Use house price dataset for training
   - Evaluate model performance with various metrics (MSE, RMSE, MAE, R²)
   - Visualize predictions and residuals

## 📂 Files
- `DL_Practical-1.pdf` - Assignment description PDF
- `Pytorch_Fundamental.ipynb` - Task 1: PyTorch tensors and basic operations
- `tensorflow_linear_algebra.py` - Task 2: TensorFlow linear algebra operations
- `perceptron_gates.py` - Task 3: AND/OR gates using perceptron
- `xor_pytorch.py` - Task 4: XOR problem with neural network
- `regression_pytorch.py` - Task 5: Regression neural network
- `house_price_full+(2) - house_price_full+(2).csv` - Dataset for regression task
- Generated visualizations and saved models

## 🔧 Implementation Details

### Task 1: PyTorch Tensors
- Covered in Jupyter notebook with comprehensive examples
- Includes tensor initialization, operations, indexing, reshaping, and autograd

### Task 2: TensorFlow Linear Algebra
- Comprehensive implementation of 20+ linear algebra operations
- All operations use TensorFlow's native functions
- Includes verification steps for complex operations

### Task 3: Perceptron for Logic Gates
- Custom Perceptron class implementation from scratch
- Uses step activation function
- Trains using the perceptron learning algorithm
- Both AND and OR gates achieve 100% accuracy
- Generates decision boundary visualizations

### Task 4: XOR Neural Network
- Architecture: 2 → 4 → 1 (input → hidden → output)
- Activation: Sigmoid
- Loss: Binary Cross Entropy
- Optimizer: Adam
- Demonstrates the necessity of hidden layers for non-linearly separable problems

### Task 5: Regression Network
- Architecture: input → 64 → 32 → 1 with ReLU activations
- Loss: Mean Squared Error
- Data preprocessing with StandardScaler
- Comprehensive evaluation with multiple metrics

## 📊 Results

All implementations run successfully and produce:
- **Perceptron (AND/OR)**: 100% accuracy on both gates
- **XOR Neural Network**: Successfully learns XOR function with >95% accuracy
- **Regression Network**: Achieves good R² score on house price prediction
- All visualizations saved as PNG files
- Model weights saved for later use

## 🧪 How to Run

### Task 1: PyTorch Fundamentals
```bash
jupyter notebook Pytorch_Fundamental.ipynb
```

### Task 2: TensorFlow Linear Algebra
```bash
python tensorflow_linear_algebra.py
```

### Task 3: Perceptron for AND/OR Gates
```bash
python perceptron_gates.py
```

### Task 4: XOR Neural Network
```bash
python xor_pytorch.py
```

### Task 5: Regression Neural Network
```bash
python regression_pytorch.py
```

## 📝 Observations

1. **Perceptron Limitations**: Single-layer perceptrons can only learn linearly separable functions (AND, OR) but cannot learn XOR.

2. **Hidden Layers Importance**: XOR problem requires at least one hidden layer to create non-linear decision boundaries.

3. **Activation Functions**: Sigmoid works well for binary classification, while ReLU is preferred for hidden layers in regression.

4. **Normalization Impact**: Feature scaling significantly improves neural network training convergence and performance.

5. **PyTorch vs TensorFlow**: Both frameworks provide powerful tools for deep learning, with slightly different APIs but similar capabilities.

## 📚 References
- https://www.tensorflow.org/api_docs/python/tf
- https://www.tensorflow.org/api_docs/python/tf/math
- https://www.tensorflow.org/api_docs/python/tf/linalg
- https://pytorch.org/docs/stable/index.html
- https://pytorch.org/tutorials/
