# Lab 02 - Handwritten Digit Recognition with MNIST

## 📋 Objective
Build and refine Convolutional Neural Network (CNN) and Multi-Layer Perceptron (MLP) models to classify handwritten digits from the MNIST dataset. The goal is to experiment with different configurations of activation functions, optimizers, and regularization techniques to understand their impact on model performance.

## 🎯 Tasks
- **Task 1**: The Activation Function Challenge
  - Compare training loss and accuracy curves using Sigmoid, Tanh, and ReLU activation functions
  - Observe vanishing gradient effects and convergence speeds
  
- **Task 2**: The Optimizer Showdown
  - Compare SGD, SGD with Momentum, and Adam optimizers
  - Keep the best activation function constant (ReLU)
  - Analyze convergence speed and stability
  
- **Task 3**: Batch Normalization and Dropout Experiments
  - WITHOUT Batch Normalization and Dropout
  - Without BN, Dropout layer = 0.1
  - With BN, Dropout layer = 0.25

## 📂 Files
- `MNIST_Classification_Experiments.ipynb` - Main implementation notebook with all experiments
- `DL_Practical-2 (1).pdf` - Assignment problem statement
- `README.md` - This file
- Generated outputs:
  - `mnist_samples.png` - Sample MNIST digits
  - `task1_activation_comparison.png` - Activation function comparison plots
  - `task2_optimizer_comparison.png` - Optimizer comparison plots
  - `task3_bn_dropout_comparison.png` - Batch normalization and dropout comparison
  - `sample_predictions.png` - Model prediction visualizations
  - `confusion_matrix.png` - Confusion matrix of best model

## 🔧 Implementation Details

### Model Architectures

**CNN Base Architecture:**
- Input Layer: (28, 28, 1) grayscale images
- Conv2D Layer 1: 32 filters, 3×3 kernel, Activation Function
- Conv2D Layer 2: 64 filters, 3×3 kernel, Activation Function
- Max Pooling Layer: 2×2 kernel
- Dropout: Configurable rate (0.0, 0.1, 0.25)
- Dense Layer: 128 neurons (Fully connected)
- Output Layer: 10 neurons (Softmax)

**MLP Base Architecture:**
- Flatten: 784 input features
- Dense layers with configurable units
- Batch Normalization (optional)
- Activation Function: Configurable
- Dropout (optional)
- Output Layer: 10 neurons (Softmax)

### Experiments Conducted

1. **Activation Function Comparison** (Task 1)
   - Sigmoid vs Tanh vs ReLU
   - All using Adam optimizer, 10 epochs

2. **Optimizer Comparison** (Task 2)
   - SGD vs SGD+Momentum vs Adam
   - All using ReLU activation, 10 epochs

3. **Regularization Comparison** (Task 3)
   - No BN, No Dropout
   - No BN, Dropout=0.1
   - With BN, Dropout=0.25

4. **Additional Configurations**
   - CNN-1: FC=128, Adam, 10 epochs
   - MLP-1: 512-256-128, SGD, 20 epochs
   - MLP-2: 256, Adam, 15 epochs

## 📊 Results

### Key Findings

**Task 1 - Activation Functions:**
- ReLU consistently outperforms Sigmoid and Tanh
- Sigmoid shows slower convergence due to vanishing gradients
- Tanh performs better than Sigmoid but still slower than ReLU
- ReLU enables fastest convergence and highest accuracy

**Task 2 - Optimizers:**
- Adam achieves fastest convergence and best accuracy
- SGD with Momentum significantly improves over plain SGD
- Plain SGD shows the slowest and most unstable convergence
- Adam's adaptive learning rate is most effective

**Task 3 - Regularization:**
- Batch Normalization + Dropout (0.25) provides best generalization
- Without regularization, model tends to overfit
- Light dropout (0.1) helps but not as much as BN + higher dropout
- BN stabilizes training and improves convergence

### Best Configuration
- **Activation**: ReLU
- **Optimizer**: Adam
- **Regularization**: Batch Normalization + Dropout (0.25)
- **Expected Accuracy**: ~98-99% on MNIST test set

## 🧪 How to Run

### Prerequisites
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Running the Notebook
```bash
jupyter notebook MNIST_Classification_Experiments.ipynb
```

Or run all cells sequentially in the notebook. The notebook includes:
- Automatic data loading from tensorflow.keras.datasets
- All three tasks with comprehensive experiments
- Visualization of results
- Comparison tables
- Model predictions and confusion matrix

## 📝 Observations

### Learning Outcomes

1. **Activation Functions**:
   - ReLU solves the vanishing gradient problem for positive inputs
   - Sigmoid and Tanh saturate at extremes, causing slow learning
   - Choice of activation function significantly impacts training speed

2. **Optimizers**:
   - Adam's adaptive learning rates work well across different scenarios
   - Momentum helps SGD overcome local minima
   - Modern optimizers like Adam are preferred for deep learning

3. **Regularization**:
   - Dropout prevents co-adaptation of neurons
   - Batch Normalization stabilizes and accelerates training
   - Proper regularization is crucial for good generalization

4. **Model Comparison**:
   - CNNs leverage spatial structure better than MLPs for image data
   - Deeper MLPs can achieve good performance but require more parameters
   - Architecture choice depends on data characteristics

## 📚 References
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)
- [Deep Learning Book - Ian Goodfellow](https://www.deeplearningbook.org/)
- Lecture slides: DL_Practical-2 (1).pdf
