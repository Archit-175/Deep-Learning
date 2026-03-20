"""
Lab Practical 5: Recurrent Models for Handwritten Character Recognition
-----------------------------------------------------------------------
Implements vanilla RNN, LSTM, GRU, BiLSTM, and hybrid CNN+LSTM variants
for MNIST/EMNIST classification. Includes utilities for gradient analysis,
gate visualization, and quick comparative runs.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

# --------------------------------------------------------------------------- #
# Device setup
# --------------------------------------------------------------------------- #

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Data utilities
# --------------------------------------------------------------------------- #

DATA_ROOT = Path(__file__).resolve().parent / "data"


def default_worker_count() -> int:
    return min(2, os.cpu_count() or 1)


def _normalize() -> transforms.Compose:
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )


def _select_dataset(
    name: str,
    train: bool,
    use_fake_data: bool = False,
    size: Optional[int] = None,
    num_classes: int = 10,
):
    name_lower = name.lower()
    if use_fake_data:
        return datasets.FakeData(
            size=size or (2048 if train else 512),
            image_size=(1, 28, 28),
            num_classes=num_classes,
            transform=_normalize(),
        )
    try:
        if name_lower == "mnist":
            return datasets.MNIST(root=DATA_ROOT, train=train, download=True, transform=_normalize())
        if name_lower.startswith("emnist-"):
            split = name_lower.split("-", maxsplit=1)[1]
            return datasets.EMNIST(
                root=DATA_ROOT,
                split=split,
                train=train,
                download=True,
                transform=_normalize(),
            )
    except Exception as exc:  # pragma: no cover - informative fallback
        print(
            f"[error] Unable to load dataset {name_lower}: {exc}. "
            "If running without internet, retry with --offline to use synthetic data."
        )
        raise
    raise ValueError(f"Unsupported dataset: {name}")


DATASET_CLASSES: Dict[str, int] = {
    "mnist": 10,
    "emnist-letters": 26,
    "emnist-balanced": 47,
    "emnist-byclass": 62,
}


def make_loaders(
    dataset: str = "mnist",
    batch_size: int = 128,
    limit_train: Optional[int] = None,
    limit_test: Optional[int] = None,
    use_fake_data: bool = False,
    num_workers: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    num_classes = DATASET_CLASSES[dataset.lower()]
    train_set = _select_dataset(
        dataset,
        train=True,
        use_fake_data=use_fake_data,
        size=limit_train,
        num_classes=num_classes,
    )
    test_set = _select_dataset(
        dataset,
        train=False,
        use_fake_data=use_fake_data,
        size=limit_test,
        num_classes=num_classes,
    )

    if limit_train:
        train_set = Subset(train_set, list(range(min(len(train_set), limit_train))))
    if limit_test:
        test_set = Subset(test_set, list(range(min(len(test_set), limit_test))))

    worker_count = num_workers if num_workers is not None else default_worker_count()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=worker_count)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=worker_count)
    return train_loader, test_loader


def to_sequence(images: torch.Tensor, scan: str = "row") -> torch.Tensor:
    """
    Convert images of shape (B, 1, 28, 28) to sequences (B, 28, 28).
    scan: 'row' keeps original ordering; 'column' transposes H/W.
    """
    if images.dim() != 4:
        raise ValueError("Expected images with shape [B, C, H, W]")
    seq = images.squeeze(1)  # (B, H, W)
    if scan == "column":
        seq = seq.transpose(1, 2)
    return seq


# --------------------------------------------------------------------------- #
# Model definitions
# --------------------------------------------------------------------------- #

class SequenceClassifier(nn.Module):
    """Generic wrapper around RNN/LSTM/GRU layers for sequence classification."""

    def __init__(
        self,
        rnn_type: str = "rnn",
        input_size: int = 28,
        hidden_size: int = 128,
        num_layers: int = 1,
        num_classes: int = 10,
        dropout: float = 0.2,
        bidirectional: bool = False,
        scan: str = "row",
    ):
        super().__init__()
        self.scan = scan
        self.rnn_type = rnn_type.lower()
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[self.rnn_type]
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        multiplier = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * multiplier, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = to_sequence(x, scan=self.scan)
        outputs, _ = self.rnn(seq)
        outputs = self.dropout(outputs[:, -1, :])
        return self.fc(outputs)


class CNNFeatureExtractor(nn.Module):
    """Small CNN feature extractor to pair with LSTM."""

    def __init__(self, in_channels: int = 1, hidden_channels: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x7
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def infer_sequence_feature_dim(encoder: nn.Module, input_shape: Tuple[int, ...] = (1, 1, 28, 28)) -> int:
    """Infer flattened feature dimension after CNN encoder for sequence modeling."""
    with torch.no_grad():
        dummy = torch.zeros(*input_shape)
        feats = encoder(dummy)
    return feats.size(1) * feats.size(3)


class CNNLSTMClassifier(nn.Module):
    """
    Hybrid model: CNN feature extractor followed by LSTM over spatial rows.
    Treats CNN feature map rows as sequence steps.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 1,
        num_classes: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        sequence_feature_dim = infer_sequence_feature_dim(self.cnn)
        self.lstm = nn.LSTM(
            input_size=sequence_feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.cnn(x)  # (B, C=64, H=7, W=7)
        seq = feats.permute(0, 2, 1, 3).contiguous()  # (B, H, C, W)
        seq = seq.view(seq.size(0), seq.size(1), -1)  # (B, 7, 64*7)
        outputs, _ = self.lstm(seq)
        outputs = self.dropout(outputs[:, -1, :])
        return self.fc(outputs)


class ConvLSTMCell(nn.Module):
    """Minimal ConvLSTM cell for 2D inputs."""

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(
        self, x: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTMClassifier(nn.Module):
    """
    ConvLSTM that treats each image row as a time-step containing a 1x28 strip.
    """

    def __init__(self, hidden_channels: int = 32, num_classes: int = 10, dropout: float = 0.1):
        super().__init__()
        self.cell = ConvLSTMCell(input_channels=1, hidden_channels=hidden_channels)
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        batch_size = x.size(0)
        h = x.new_zeros(batch_size, self.cell.hidden_channels, 1, x.size(-1))
        c = x.new_zeros(batch_size, self.cell.hidden_channels, 1, x.size(-1))
        for t in range(x.size(2)):
            row = x[:, :, t : t + 1, :]  # (B, 1, 1, 28)
            h, c = self.cell(row, h, c)
        h = self.dropout(h)
        return self.classifier(h)


# --------------------------------------------------------------------------- #
# Training utilities
# --------------------------------------------------------------------------- #

@dataclass
class TrainConfig:
    model_name: str
    dataset: str = "mnist"
    epochs: int = 1
    batch_size: int = 128
    learning_rate: float = 1e-3
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.2
    grad_clip: Optional[float] = None
    scan: str = "row"
    limit_train: Optional[int] = None
    limit_test: Optional[int] = None
    optimizer: str = "adam"
    bidirectional: bool = False
    log_gates: bool = False
    use_fake_data: bool = False
    num_workers: int = default_worker_count()


def build_model(cfg: TrainConfig, num_classes: int) -> nn.Module:
    name = cfg.model_name.lower()
    if name == "rnn":
        return SequenceClassifier(
            rnn_type="rnn",
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            num_classes=num_classes,
            dropout=cfg.dropout,
            bidirectional=cfg.bidirectional,
            scan=cfg.scan,
        )
    if name == "lstm":
        return SequenceClassifier(
            rnn_type="lstm",
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            num_classes=num_classes,
            dropout=cfg.dropout,
            bidirectional=cfg.bidirectional,
            scan=cfg.scan,
        )
    if name == "gru":
        return SequenceClassifier(
            rnn_type="gru",
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            num_classes=num_classes,
            dropout=cfg.dropout,
            bidirectional=cfg.bidirectional,
            scan=cfg.scan,
        )
    if name == "bilstm":
        return SequenceClassifier(
            rnn_type="lstm",
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            num_classes=num_classes,
            dropout=cfg.dropout,
            bidirectional=True,
            scan=cfg.scan,
        )
    if name == "cnn-lstm":
        return CNNLSTMClassifier(
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            num_classes=num_classes,
            dropout=cfg.dropout,
        )
    if name == "convlstm":
        return ConvLSTMClassifier(hidden_channels=cfg.hidden_size, num_classes=num_classes, dropout=cfg.dropout)
    raise ValueError(f"Unknown model: {cfg.model_name}")


def gradient_norms(model: nn.Module) -> float:
    norms: List[float] = []
    for param in model.parameters():
        if param.grad is not None:
            norms.append(param.grad.data.norm(2).item())
    return float(torch.tensor(norms).mean()) if norms else 0.0


def evaluate(model: nn.Module, loader: DataLoader) -> Tuple[float, float]:
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return loss_sum / total, correct / total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    grad_clip: Optional[float],
) -> Tuple[float, float]:
    model.train()
    if len(loader) == 0:
        raise ValueError(
            f"Training loader is empty. dataset_size={len(getattr(loader, 'dataset', []))}, "
            f"batch_size={getattr(loader, 'batch_size', 'unknown')}. "
            "Verify dataset availability, offline/demo size limits, and batching configuration."
        )
    epoch_loss = 0.0
    grad_logs: List[float] = []
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        grad_logs.append(gradient_norms(model))
        optimizer.step()
        epoch_loss += loss.item() * labels.size(0)
    if not grad_logs:
        raise RuntimeError("No gradients were recorded during training. Check dataloader and model output.")
    avg_grad = sum(grad_logs) / len(grad_logs)
    return epoch_loss / len(loader.dataset), avg_grad


def extract_lstm_gate_stats(model: SequenceClassifier, sample: torch.Tensor) -> Dict[str, float]:
    """
    Estimate gate activations for a single-layer LSTM using its learned weights.
    Returns mean activation values for input/forget/output gates across the batch.
    For non-LSTM or multi-layer/bidirectional configurations, an empty dictionary
    is returned.
    """
    if model.rnn_type != "lstm":
        return {}
    lstm: nn.LSTM = model.rnn  # type: ignore
    if lstm.num_layers != 1:
        return {}

    try:
        weight_ih = lstm.weight_ih_l0
        weight_hh = lstm.weight_hh_l0
        bias_ih = lstm.bias_ih_l0
        bias_hh = lstm.bias_hh_l0
    except AttributeError as exc:
        raise RuntimeError(
            "LSTM parameters not accessible for gate extraction; verify PyTorch version compatibility."
        ) from exc

    x = to_sequence(sample, scan=model.scan).to(weight_ih.device)
    h = torch.zeros(sample.size(0), lstm.hidden_size, device=weight_ih.device)
    c = torch.zeros_like(h)

    stats = {"input_gate": [], "forget_gate": [], "output_gate": []}
    for t in range(x.size(1)):
        gates = F.linear(x[:, t, :], weight_ih, bias_ih) + F.linear(h, weight_hh, bias_hh)
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c = f * c + i * g
        h = o * torch.tanh(c)
        stats["input_gate"].append(i.mean().item())
        stats["forget_gate"].append(f.mean().item())
        stats["output_gate"].append(o.mean().item())
    # Return mean across time
    return {k: float(torch.tensor(v).mean()) for k, v in stats.items()}


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def run_experiment(cfg: TrainConfig) -> Dict[str, float]:
    num_classes = DATASET_CLASSES[cfg.dataset.lower()]
    train_loader, test_loader = make_loaders(
        dataset=cfg.dataset,
        batch_size=cfg.batch_size,
        limit_train=cfg.limit_train,
        limit_test=cfg.limit_test,
        use_fake_data=cfg.use_fake_data,
        num_workers=cfg.num_workers,
    )
    model = build_model(cfg, num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    if cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=0.9)
    elif cfg.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    history = {"train_loss": [], "train_grad_norm": [], "val_loss": [], "val_acc": []}
    start = time.time()
    for _ in range(cfg.epochs):
        train_loss, grad_norm = train_one_epoch(
            model, train_loader, optimizer, criterion, grad_clip=cfg.grad_clip
        )
        val_loss, val_acc = evaluate(model, test_loader)
        history["train_loss"].append(train_loss)
        history["train_grad_norm"].append(grad_norm)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
    duration = time.time() - start

    gate_stats = {}
    if cfg.log_gates and isinstance(model, SequenceClassifier) and cfg.model_name.lower() in {"lstm", "bilstm"}:
        sample_images, _ = next(iter(test_loader))
        gate_stats = extract_lstm_gate_stats(model, sample_images[: min(16, len(sample_images))].to(DEVICE))

    params = parameter_count(model)
    return {
        "final_val_acc": history["val_acc"][-1],
        "final_val_loss": history["val_loss"][-1],
        "train_loss": history["train_loss"][-1],
        "train_grad_norm": history["train_grad_norm"][-1],
        "params": float(params),
        "duration_sec": float(duration),
        **{f"gate_{k}": v for k, v in gate_stats.items()},
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Lab05: RNN/LSTM/GRU/BiLSTM/CNN-LSTM experiments.")
    parser.add_argument(
        "--model",
        type=str,
        default="lstm",
        choices=["rnn", "lstm", "gru", "bilstm", "cnn-lstm", "convlstm"],
        help="Model architecture to train",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=sorted(DATASET_CLASSES.keys()),
        help="Dataset to use",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--scan", type=str, default="row", help="row or column scanning for sequence models")
    parser.add_argument("--optimizer", type=str, default="adam", help="adam|sgd|rmsprop")
    parser.add_argument("--bidirectional", action="store_true", help="Use bidirectional processing (for LSTM/GRU/RNN)")
    parser.add_argument("--limit-train", type=int, default=None, help="Optional cap on training samples for quick runs")
    parser.add_argument("--limit-test", type=int, default=None, help="Optional cap on test samples for quick runs")
    parser.add_argument("--log-gates", action="store_true", help="Log mean LSTM gate activations on a sample batch")
    parser.add_argument("--demo", action="store_true", help="Run a fast demo on a small subset for smoke testing")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Dataloader worker processes (defaults to a safe value based on available CPU cores)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use synthetic FakeData instead of downloading datasets (also enabled automatically with --demo)",
    )
    args = parser.parse_args()

    use_fake_data = args.offline or args.demo
    if args.demo:
        args.epochs = min(args.epochs, 1)
        args.limit_train = args.limit_train or 1024
        args.limit_test = args.limit_test or 256
        args.batch_size = min(args.batch_size, 64)

    worker_count = args.num_workers if args.num_workers is not None else default_worker_count()

    return TrainConfig(
        model_name=args.model,
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        grad_clip=args.grad_clip,
        scan=args.scan,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
        optimizer=args.optimizer,
        bidirectional=args.bidirectional,
        log_gates=args.log_gates,
        use_fake_data=use_fake_data,
        num_workers=worker_count,
    )


def main():
    cfg = parse_args()
    print(f"Using device: {DEVICE}")
    print(f"Config: {cfg}")
    metrics = run_experiment(cfg)
    print("\n===== Experiment Summary =====")
    for k, v in metrics.items():
        print(f"{k:20s}: {v}")


if __name__ == "__main__":
    main()
