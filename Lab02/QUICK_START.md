# Lab 02 - Quick Start Guide

## 🚀 Quick Start

This lab implements the complete assignment from DL_Practical-2(1).pdf for MNIST Handwritten Digit Recognition.

### Option 1: Using Jupyter Notebook (Recommended)

```bash
# Navigate to Lab02 directory
cd Lab02

# Start Jupyter Notebook
jupyter notebook MNIST_Classification_Experiments.ipynb

# Run all cells: Cell > Run All
```

### Option 2: Using Python Script

```bash
# Navigate to Lab02 directory
cd Lab02

# Run the complete experiment
python3 mnist_experiments.py

# Or make it executable and run directly
chmod +x mnist_experiments.py
./mnist_experiments.py
```

### Option 3: Test Without MNIST Data

If you cannot download MNIST data (network restrictions), test the implementation:

```bash
cd Lab02
python3 test_implementation.py
```

This will validate all model architectures using synthetic data.

## 📦 Prerequisites

Install required packages:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn jupyter
```

Or use the repository's requirements file:

```bash
pip install -r ../requirements.txt
```

## 📊 What Gets Generated

When you run the complete experiments, the following files are generated:

### Visualization Files
- `mnist_samples.png` - Sample images from MNIST dataset
- `task1_activation_comparison.png` - Activation function comparison plots
- `task2_optimizer_comparison.png` - Optimizer comparison plots  
- `task3_bn_dropout_comparison.png` - Batch normalization & dropout comparison
- `sample_predictions.png` - Model predictions on test samples
- `confusion_matrix.png` - Confusion matrix of best model

### Results
All results are displayed in the notebook/console:
- Comparison tables for each task
- Training and validation metrics
- Test accuracies for all experiments

## ⏱️ Expected Runtime

Running all experiments (on CPU):
- **Task 1**: ~5-10 minutes (3 models × 10 epochs)
- **Task 2**: ~5-10 minutes (3 models × 10 epochs)
- **Task 3**: ~5-10 minutes (3 models × 10 epochs)
- **Additional**: ~10-15 minutes (3 models × varying epochs)
- **Total**: ~30-45 minutes

With GPU acceleration: ~5-10 minutes total

## 📝 Assignment Tasks Implemented

### ✅ Task 1: Activation Function Challenge
Compare Sigmoid, Tanh, and ReLU activation functions with:
- Training and validation loss curves
- Training and validation accuracy curves
- Final test accuracy comparison

### ✅ Task 2: Optimizer Showdown
Compare SGD, SGD+Momentum, and Adam optimizers with:
- Best activation function (ReLU) kept constant
- Loss stability analysis
- Convergence speed comparison

### ✅ Task 3: Batch Normalization and Dropout
Three experimental scenarios:
1. No BN, No Dropout
2. No BN, Dropout=0.1
3. With BN, Dropout=0.25

### ✅ Additional Models
As specified in assignment table:
- **CNN-1**: FC=128, Adam, 10 epochs
- **MLP-1**: 512-256-128, SGD, 20 epochs
- **MLP-2**: 256, Adam, 15 epochs

## 🎯 Expected Results

With real MNIST data, expect:
- **Best CNN**: ~98-99% accuracy
- **Best MLP**: ~97-98% accuracy
- **ReLU** outperforms Sigmoid and Tanh
- **Adam** converges fastest among optimizers
- **BN + Dropout** provides best generalization

## 🔍 Files in This Directory

| File | Description |
|------|-------------|
| `MNIST_Classification_Experiments.ipynb` | Main Jupyter notebook with all tasks |
| `mnist_experiments.py` | Standalone Python script version |
| `test_implementation.py` | Test script with synthetic data |
| `README.md` | Complete documentation |
| `EXECUTION_SUMMARY.md` | Implementation details and observations |
| `QUICK_START.md` | This file |
| `DL_Practical-2 (1).pdf` | Original assignment PDF |

## 💡 Tips

1. **Memory Issues**: If you run out of memory, reduce batch size in the code (default: 128)
2. **Faster Execution**: Use Google Colab with GPU for faster training
3. **Save Time**: Test with fewer epochs first, then run full experiments
4. **Visualization**: All plots are automatically saved as PNG files

## 🐛 Troubleshooting

### MNIST Download Fails
```
Error: URL fetch failure on https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz: 403 -- Forbidden
```

**Solution**: Run in an environment with unrestricted internet access, or use Google Colab.

### TensorFlow Not Found
```
ModuleNotFoundError: No module named 'tensorflow'
```

**Solution**: 
```bash
pip install tensorflow
```

### Out of Memory
```
ResourceExhaustedError: OOM when allocating tensor
```

**Solution**: Reduce batch size or use fewer epochs.

## 📚 Learn More

- See `README.md` for detailed documentation
- See `EXECUTION_SUMMARY.md` for implementation notes
- Check the assignment PDF: `DL_Practical-2 (1).pdf`

## ✅ Verification

To verify everything is set up correctly:

```bash
python3 test_implementation.py
```

This should output:
```
============================================================
All Tests Passed Successfully!
============================================================
```

---

**Ready to start?** Open the Jupyter notebook and run all cells! 🚀
