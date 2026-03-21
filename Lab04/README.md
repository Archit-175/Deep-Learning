# Lab 04 – CNN Architectures for Imbalanced Image Classification

## 📋 Objective

Design and implement CNN architectures for multi-class image classification on
imbalanced datasets.  All seven problem statements from
**DL_Practical-4_Updated.pdf** are addressed in a single, self-contained script.

---

## 🎯 Problem Statements Covered

| # | Topic | Key Technique |
|---|-------|---------------|
| PS1 | Architecture Design | Custom CNN with BatchNorm, Dropout, L2 (weight decay) |
| PS2 | Imbalanced Data Handling | Class weighting · Oversampling · Augmentation |
| PS3 | Comparative Architecture Analysis | CustomCNN vs ResNet18 vs DenseNet121 |
| PS4 | Loss Function & Optimiser Study | Cross-Entropy · Weighted-CE · Focal Loss · Label Smoothing × SGD / Adam / AdamW / RMSProp |
| PS5 | Feature Visualisation | t-SNE · PCA · Grad-CAM |
| PS6 | Transfer Learning | ImageNet pre-trained ResNet18 vs from-scratch |
| PS7 | Error Analysis | Per-class accuracy · Confusion patterns · Improvement proposals |

---

## 📂 Files

| File | Description |
|------|-------------|
| `cnn_imbalanced_classification.py` | Main implementation (all 7 PS) |
| `DL_Practical-4_Updated.pdf` | Original assignment specification |
| `results/` | Auto-generated plots (confusion matrices, t-SNE, PCA, Grad-CAM, training curves) |

---

## 🔧 Implementation Details

### Dataset
CIFAR-10 is used with a **synthetically induced long-tailed distribution**
(imbalance ratio ≈ 100:1).  Samples per class for training:

```
airplane: 5000 | automobile: 4000 | bird: 3000 | cat: 2000 | deer: 1500
dog: 1000 | frog: 750 | horse: 500 | ship: 200 | truck: 50
```

### Custom CNN (PS1)
Three convolutional blocks (Conv→BN→ReLU×2 + MaxPool + Dropout2d) followed
by two fully-connected layers with BatchNorm1d and Dropout.  L2 regularisation
is applied via `weight_decay` in AdamW.

### Imbalance Strategies (PS2)
- **Baseline** – plain cross-entropy, no handling
- **Class Weighting** – inverse-frequency weights in loss
- **Oversampling** – `WeightedRandomSampler` at data-loader level
- **Augmentation + Weighting** – strong augmentation (ColorJitter, RandomRotation) for all classes plus weighted loss

### Architectures (PS3)
| Architecture | Params | Notes |
|---|---|---|
| CustomCNN | ~3.4 M | Designed for CIFAR-10 (32×32) |
| ResNet18 | ~11.2 M | torchvision, final FC replaced |
| DenseNet121 | ~7.0 M | torchvision, classifier replaced |

### Loss Functions (PS4)
- Cross-Entropy (baseline)
- Weighted Cross-Entropy
- Focal Loss (gamma in {0.5, 1, 2, 5})
- Label Smoothing Cross-Entropy (ε = 0.1)

---

## 🧪 How to Run

### Full run (downloads CIFAR-10, ~10 epochs)
```bash
python cnn_imbalanced_classification.py
```

### Custom epochs / learning rate
```bash
python cnn_imbalanced_classification.py --epochs 20 --lr 3e-4 --batch-size 64
```

### Offline / no-download smoke-test (synthetic FakeData, 1 epoch)
```bash
python cnn_imbalanced_classification.py --quick
# or
python cnn_imbalanced_classification.py --offline --epochs 5
```

---

## 📊 Results

Generated files in `results/`:

| File | Content |
|------|---------|
| `ps1_confusion_matrix.png` | PS1 CustomCNN confusion matrix |
| `ps2_training_curves.png` | Loss & accuracy for 4 imbalance strategies |
| `ps3_training_curves.png` | Architecture comparison training curves |
| `ps3_cm_*.png` | Per-architecture confusion matrices |
| `ps4_training_curves.png` | Loss × optimiser grid curves |
| `ps5_tsne.png` | t-SNE feature space |
| `ps5_pca.png` | PCA feature space |
| `ps5_gradcam.png` | Grad-CAM attention maps |
| `ps6_training_curves.png` | Transfer learning comparison |
| `ps7_confusion_matrix.png` | Best-model error analysis |

---

## 📝 Observations

1. **Class imbalance severely degrades minority-class recall** when using
   plain cross-entropy – balanced accuracy drops close to chance level.
2. **Weighted cross-entropy and oversampling** offer complementary benefits;
   combining them with augmentation yields the best minority-class F1.
3. **Focal Loss (γ = 2)** converges more stably than high-γ variants on
   heavily imbalanced splits.
4. **ResNet18** outperforms the custom CNN in both accuracy and balanced
   accuracy, demonstrating the advantage of residual connections for
   imbalanced settings with limited training data.
5. **Pre-trained ImageNet weights** (PS6) accelerate convergence and boost
   balanced accuracy significantly with fewer epochs.
6. **t-SNE / PCA** reveal that minority classes cluster poorly in the feature
   space – a key motivation for oversampling and augmentation.

---

## 📚 References

1. He et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
2. Huang et al. (2017). *Densely Connected Convolutional Networks.* CVPR.
3. Lin et al. (2017). *Focal Loss for Dense Object Detection.* ICCV.
4. Cui et al. (2019). *Class-Balanced Loss Based on Effective Number of Samples.* CVPR.
5. CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
