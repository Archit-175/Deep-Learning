# Lab 05 - Recurrent Models for Handwritten Character Recognition

## 📋 Objective
Design and implement multiple recurrent neural architectures for MNIST/EMNIST handwritten character recognition, analyze vanishing gradients, visualize LSTM gates, and compare hybrid CNN + LSTM designs.

## 🎯 Tasks
- Implement vanilla RNN, LSTM, GRU, and BiLSTM sequence classifiers with row/column scanning.
- Build two CNN + LSTM hybrids: feature-extractor LSTM and ConvLSTM cell.
- Log gradient magnitudes to study vanishing/exploding behavior.
- Visualize LSTM gate activations on sample batches.
- Compare models on accuracy, parameters, training time, and inference speed.

## 📂 Files
- `rnn_sequence_models.py` — Main experiment runner with all architectures and utilities.
- `DL_Practical-5_RNN.pdf` — Assignment brief.

## 🔧 Implementation Details
- Images are reshaped into sequences (28 steps of 28 features) with optional row/column scanning.
- Models supported: Vanilla RNN, LSTM, GRU, BiLSTM, CNN+LSTM (feature map rows as sequence), and ConvLSTM.
- Training utilities support gradient clipping, optimizer choice (Adam/SGD/RMSprop), and gate visualization for single-layer LSTMs.
- Fast demo mode caps dataset size for quick smoke tests.

## 🧪 How to Run

Install dependencies (from repo root if needed):
```bash
pip install -r requirements.txt
```

Run a quick demo on a small MNIST subset (CPU-friendly):
```bash
cd Lab05
python rnn_sequence_models.py --model lstm --demo --epochs 1 --log-gates
```
If internet access is blocked for dataset downloads, append `--offline` to use synthetic FakeData.

Train a BiLSTM on the full MNIST dataset:
```bash
python rnn_sequence_models.py --model bilstm --epochs 5 --batch-size 128 --hidden-size 128 --optimizer adam
```

Compare row vs column scanning for vanilla RNN:
```bash
python rnn_sequence_models.py --model rnn --scan column --epochs 3
```

Experiment with CNN + LSTM hybrid:
```bash
python rnn_sequence_models.py --model cnn-lstm --epochs 3
```

## 📊 Results
- Script reports training loss, validation loss/accuracy, gradient norms, parameter counts, and runtime.
- Gate logging prints mean input/forget/output activations for LSTM variants.
- Use demo mode for quick checks; full EMNIST runs require longer training time.

## 📝 Observations
- Row/column scanning helps study orientation sensitivity.
- Gradient clipping plus LSTM/GRU mitigates vanishing gradients compared to vanilla RNN.
- Hybrid CNN + LSTM improves feature extraction before sequence modeling.

## 📚 References
- DL_Practical-5_RNN.pdf (lab brief)
- Hochreiter & Schmidhuber (1997) — LSTM
- Cho et al. (2014) — GRU
