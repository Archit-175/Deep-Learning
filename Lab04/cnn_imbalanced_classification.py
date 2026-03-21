"""
Lab Practical 4: CNN Architectures for Imbalanced Image Classification
-----------------------------------------------------------------------
Implements all 7 problem statements from DL_Practical-4_Updated.pdf:

  PS1 - Custom CNN with regularisation (Dropout, BN, L2)
  PS2 - Imbalanced-data handling (class-weighting, oversampling, augmentation)
  PS3 - Comparative architecture analysis (CustomCNN vs ResNet18 vs DenseNet121)
  PS4 - Loss function & optimiser experimentation
  PS5 - Feature visualisation (t-SNE, PCA, Grad-CAM)
  PS6 - Transfer learning (ImageNet pre-trained → imbalanced CIFAR-10)
  PS7 - Error analysis & improvement proposals

Dataset: CIFAR-10 with a synthetically induced long-tailed distribution
         (imbalance ratio ~100:1 from the majority to the rarest class).
         Use --offline / --quick for a fast, network-free demo run.

Usage examples
--------------
  python cnn_imbalanced_classification.py                  # full run (downloads CIFAR-10)
  python cnn_imbalanced_classification.py --quick          # 1-epoch offline smoke-test
  python cnn_imbalanced_classification.py --epochs 5 --offline
"""

from __future__ import annotations

import argparse
import os
import random
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")          # non-interactive backend – safe in all environments
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, models

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Long-tailed imbalance: samples per class for TRAINING set
# class 0 (airplane) → 5000, class 9 (truck) → 50  (ratio ≈ 100:1)
IMBALANCE_SAMPLES = [5000, 4000, 3000, 2000, 1500, 1000, 750, 500, 200, 50]

# t-SNE hyperparameters
TSNE_DEFAULT_PERPLEXITY = 30
TSNE_MAX_ITERATIONS = 300

# SGD uses a higher effective LR than the shared --lr default
SGD_LR_MULTIPLIER = 10

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lab04 – CNN Imbalanced Classification")
    p.add_argument("--epochs", type=int, default=10,
                   help="Training epochs per experiment (default: 10)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--offline", action="store_true",
                   help="Use synthetic FakeData – no network access needed")
    p.add_argument("--quick", action="store_true",
                   help="1-epoch smoke-test with synthetic data and tiny dataset")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------

class ImbalancedCIFAR10(Dataset):
    """CIFAR-10 with a synthetically induced long-tailed class distribution."""

    def __init__(self, root: Path, train: bool, samples_per_class: List[int],
                 transform=None):
        base = datasets.CIFAR10(root=root, train=train, download=True,
                                transform=None)
        self.transform = transform
        self.data: List[Tuple[np.ndarray, int]] = []

        label_to_indices: Dict[int, List[int]] = {c: [] for c in range(10)}
        for idx, (_, lbl) in enumerate(base):
            label_to_indices[lbl].append(idx)

        for cls, n in enumerate(samples_per_class):
            indices = label_to_indices[cls]
            chosen = indices[:n] if n <= len(indices) else indices
            for i in chosen:
                img, lbl = base[i]
                self.data.append((np.array(img), lbl))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        img_np, lbl = self.data[idx]
        from PIL import Image
        img = Image.fromarray(img_np)
        if self.transform:
            img = self.transform(img)
        return img, lbl


def class_weights_from_dataset(dataset: Dataset) -> torch.Tensor:
    """Compute inverse-frequency class weights."""
    labels = [dataset[i][1] for i in range(len(dataset))]
    counts = Counter(labels)
    n_classes = max(counts) + 1
    freqs = torch.tensor([counts.get(c, 1) for c in range(n_classes)], dtype=torch.float)
    weights = 1.0 / freqs
    weights = weights / weights.sum() * n_classes
    return weights


def make_weighted_sampler(dataset: Dataset) -> WeightedRandomSampler:
    """Return a sampler that up-samples minority classes."""
    labels = [dataset[i][1] for i in range(len(dataset))]
    counts = Counter(labels)
    sample_weights = [1.0 / counts[lbl] for lbl in labels]
    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )


def get_transforms(train: bool, augment_minority: bool = False) -> transforms.Compose:
    norm = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
    if train and augment_minority:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            norm,
        ])
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([transforms.ToTensor(), norm])


def make_loaders(
    batch_size: int = 128,
    use_fake: bool = False,
    quick: bool = False,
    use_sampler: bool = False,
    augment: bool = False,
) -> Tuple[DataLoader, DataLoader, Optional[torch.Tensor]]:
    """Return (train_loader, test_loader, class_weights_or_None)."""
    if use_fake or quick:
        n_train, n_test = (256, 64) if quick else (2048, 512)
        fake_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_ds = datasets.FakeData(n_train, (3, 32, 32), 10, fake_transform)
        test_ds = datasets.FakeData(n_test, (3, 32, 32), 10, fake_transform)
        # Artificially skew label distribution for fake data
        class_weights = torch.ones(10)
        sampler = None
    else:
        samples = [max(s // 10, 10) for s in IMBALANCE_SAMPLES] if quick else IMBALANCE_SAMPLES
        train_ds = ImbalancedCIFAR10(DATA_ROOT, train=True,
                                     samples_per_class=samples,
                                     transform=get_transforms(True, augment))
        test_ds = ImbalancedCIFAR10(DATA_ROOT, train=False,
                                    samples_per_class=[1000] * 10,
                                    transform=get_transforms(False))
        class_weights = class_weights_from_dataset(train_ds)
        sampler = make_weighted_sampler(train_ds) if use_sampler else None

    nw = min(2, os.cpu_count() or 1)
    shuffle_train = sampler is None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train,
                              sampler=sampler, num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=nw, pin_memory=True)
    return train_loader, test_loader, class_weights

# ---------------------------------------------------------------------------
# PS1 – Custom CNN architecture
# ---------------------------------------------------------------------------

class CustomCNN(nn.Module):
    """
    Custom CNN for CIFAR-10.
    Three convolutional blocks + two FC layers with Dropout and BatchNorm.
    L2 regularisation is applied via weight_decay in the optimiser.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.4):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate feature vector (before final linear)."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = nn.Flatten()(x)
        x = self.classifier[1](x)   # Linear 256*4*4 (4096) -> 512
        x = self.classifier[2](x)   # BN
        x = self.classifier[3](x)   # ReLU
        x = self.classifier[5](x)   # Linear 512 → 256
        return x

# ---------------------------------------------------------------------------
# PS3 – Pre-built architectures (ResNet18, DenseNet121) adapted for CIFAR-10
# ---------------------------------------------------------------------------

def make_resnet18(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    m = models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def make_densenet121(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    m = models.densenet121(weights=weights)
    m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    return m

# ---------------------------------------------------------------------------
# PS4 – Loss functions
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017)."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class LabelSmoothingCELoss(nn.Module):
    """Cross-entropy with label smoothing."""

    def __init__(self, smoothing: float = 0.1, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(1)
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(smooth_targets * log_probs).sum(dim=1)
        return loss.mean()


def build_loss(name: str, class_weights: Optional[torch.Tensor] = None,
               gamma: float = 2.0) -> nn.Module:
    w = class_weights.to(DEVICE) if class_weights is not None else None
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    if name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=w)
    if name == "focal":
        return FocalLoss(gamma=gamma, weight=w)
    if name == "label_smoothing":
        return LabelSmoothingCELoss(smoothing=0.1, weight=w)
    raise ValueError(f"Unknown loss: {name}")


def build_optimizer(name: str, params, lr: float = 1e-3) -> optim.Optimizer:
    if name == "sgd":
        return optim.SGD(params, lr=lr * SGD_LR_MULTIPLIER, momentum=0.9, weight_decay=5e-4)
    if name == "adam":
        return optim.Adam(params, lr=lr)
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=1e-2)
    if name == "rmsprop":
        return optim.RMSprop(params, lr=lr)
    raise ValueError(f"Unknown optimizer: {name}")

# ---------------------------------------------------------------------------
# Training & evaluation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
) -> Tuple[float, float]:
    model.train()
    total_loss = correct = total = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == lbls).sum().item()
        total += imgs.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Returns (accuracy, all_preds, all_labels)."""
    model.eval()
    preds_list, lbls_list = [], []
    for imgs, lbls in loader:
        imgs = imgs.to(DEVICE)
        out = model(imgs)
        preds_list.append(out.argmax(1).cpu().numpy())
        lbls_list.append(lbls.numpy())
    all_preds = np.concatenate(preds_list)
    all_labels = np.concatenate(lbls_list)
    acc = 100.0 * (all_preds == all_labels).mean()
    return acc, all_preds, all_labels


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epochs: int,
    tag: str = "",
    scheduler=None,
) -> Dict:
    model.to(DEVICE)
    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    for ep in range(1, epochs + 1):
        t0 = time.time()
        loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_acc, _, _ = evaluate(model, test_loader)
        if scheduler:
            scheduler.step()
        history["train_loss"].append(loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        elapsed = time.time() - t0
        print(f"  [{tag}] Epoch {ep:>3}/{epochs}  loss={loss:.4f}  "
              f"train_acc={tr_acc:.1f}%  val_acc={val_acc:.1f}%  ({elapsed:.1f}s)")
    return history

# ---------------------------------------------------------------------------
# PS5 – Feature extraction & visualisation
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader: DataLoader,
    layer_name: str = "penultimate",
    max_samples: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features from the penultimate layer of a CustomCNN or torchvision model."""
    model.eval()
    features, labels_out = [], []
    collected = 0

    # Register a hook on the last pooling / adaptive avg pool output
    activation: Dict[str, torch.Tensor] = {}

    def hook_fn(module, inp, out):
        activation["feat"] = out.detach()

    # For CustomCNN use get_features; for torchvision use avgpool
    use_custom = isinstance(model, CustomCNN)
    if not use_custom:
        # Works for ResNet and DenseNet (both have .avgpool or .features)
        target_layer = None
        if hasattr(model, "avgpool"):
            target_layer = model.avgpool
        elif hasattr(model, "features"):
            target_layer = model.features
        if target_layer is not None:
            handle = target_layer.register_forward_hook(hook_fn)

    for imgs, lbls in loader:
        if collected >= max_samples:
            break
        imgs = imgs.to(DEVICE)
        if use_custom:
            feat = model.get_features(imgs).cpu().numpy()
        else:
            model(imgs)
            feat_t = activation.get("feat", None)
            if feat_t is None:
                feat = np.zeros((imgs.size(0), 1))
            else:
                feat = feat_t.flatten(1).cpu().numpy()
        n = min(imgs.size(0), max_samples - collected)
        features.append(feat[:n])
        labels_out.append(lbls.numpy()[:n])
        collected += n

    if not use_custom and target_layer is not None:
        handle.remove()

    return np.concatenate(features), np.concatenate(labels_out)


def plot_tsne(features: np.ndarray, labels: np.ndarray, title: str, save_path: Path):
    perp = min(TSNE_DEFAULT_PERPLEXITY, len(features) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=TSNE_MAX_ITERATIONS)
    emb = tsne.fit_transform(features)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="tab10", s=8, alpha=0.7)
    plt.colorbar(scatter, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("t-SNE dim-1"); ax.set_ylabel("t-SNE dim-2")
    plt.tight_layout(); plt.savefig(save_path, dpi=100); plt.close()
    print(f"  Saved: {save_path.name}")


def plot_pca(features: np.ndarray, labels: np.ndarray, title: str, save_path: Path):
    pca = PCA(n_components=2, random_state=42)
    emb = pca.fit_transform(features)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="tab10", s=8, alpha=0.7)
    plt.colorbar(scatter, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.tight_layout(); plt.savefig(save_path, dpi=100); plt.close()
    print(f"  Saved: {save_path.name}")


def plot_gradcam(model: nn.Module, loader: DataLoader, save_path: Path,
                 n_images: int = 4):
    """Simple Grad-CAM for the last conv layer of CustomCNN."""
    if not isinstance(model, CustomCNN):
        print("  Grad-CAM: skipped (only implemented for CustomCNN)")
        return
    model.eval()
    target_layer = model.block3[0]   # first Conv2d in block3

    gradients: List[torch.Tensor] = []
    activations_list: List[torch.Tensor] = []

    def save_grad(grad):
        gradients.append(grad)

    def forward_hook(module, inp, out):
        activations_list.clear()
        activations_list.append(out)
        out.register_hook(save_grad)

    handle = target_layer.register_forward_hook(forward_hook)

    imgs, lbls = next(iter(loader))
    imgs = imgs[:n_images].to(DEVICE)
    lbls = lbls[:n_images].to(DEVICE)
    imgs.requires_grad_(False)

    out = model(imgs)
    class_idx = out.argmax(1)

    cams = []
    for i in range(n_images):
        model.zero_grad()
        score = out[i, class_idx[i]]
        score.backward(retain_graph=True)

        grad = gradients[-1][i]           # (C, H, W)
        act = activations_list[0][i]      # (C, H, W)
        weights = grad.mean(dim=[1, 2])   # (C,)
        cam = (weights[:, None, None] * act).sum(0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        cams.append(cam.detach().cpu().numpy())

    handle.remove()

    inv_norm = transforms.Normalize(
        mean=[-0.4914 / 0.2023, -0.4822 / 0.1994, -0.4465 / 0.2010],
        std=[1 / 0.2023, 1 / 0.1994, 1 / 0.2010],
    )
    fig, axes = plt.subplots(2, n_images, figsize=(3 * n_images, 6))
    for i in range(n_images):
        img_show = inv_norm(imgs[i].cpu()).permute(1, 2, 0).clamp(0, 1).numpy()
        axes[0, i].imshow(img_show)
        axes[0, i].set_title(f"Label: {CIFAR10_CLASSES[lbls[i]]}", fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].imshow(img_show)
        cam_resized = torch.tensor(cams[i]).unsqueeze(0).unsqueeze(0)
        cam_up = F.interpolate(cam_resized, size=(32, 32), mode="bilinear",
                               align_corners=False).squeeze().numpy()
        axes[1, i].imshow(cam_up, cmap="jet", alpha=0.5)
        axes[1, i].set_title(f"Grad-CAM", fontsize=8)
        axes[1, i].axis("off")
    plt.suptitle("Grad-CAM Visualisation (CustomCNN)")
    plt.tight_layout(); plt.savefig(save_path, dpi=100); plt.close()
    print(f"  Saved: {save_path.name}")

# ---------------------------------------------------------------------------
# Plotting helpers for training curves / confusion matrices
# ---------------------------------------------------------------------------

def plot_training_curves(histories: Dict[str, Dict], title: str, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, h in histories.items():
        axes[0].plot(h["train_loss"], label=name)
        axes[1].plot(h["val_acc"], label=name)
    axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss"); axes[0].legend(fontsize=7)
    axes[1].set_title("Validation Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)"); axes[1].legend(fontsize=7)
    plt.suptitle(title)
    plt.tight_layout(); plt.savefig(save_path, dpi=100); plt.close()
    print(f"  Saved: {save_path.name}")


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str],
                          title: str, save_path: Path):
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(xticks=range(n), yticks=range(n),
           xticklabels=class_names, yticklabels=class_names,
           xlabel="Predicted", ylabel="True", title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6,
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout(); plt.savefig(save_path, dpi=100); plt.close()
    print(f"  Saved: {save_path.name}")

# ---------------------------------------------------------------------------
# Evaluation report helper
# ---------------------------------------------------------------------------

def full_report(model: nn.Module, loader: DataLoader,
                class_names: List[str], tag: str):
    acc, preds, labels = evaluate(model, loader)
    cm = confusion_matrix(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    print(f"\n{'─'*60}")
    print(f"  {tag}  |  Accuracy: {acc:.1f}%  |  Balanced Acc: {bal_acc:.3f}"
          f"  |  Macro-F1: {macro_f1:.3f}")
    print(f"{'─'*60}")
    print(classification_report(labels, preds, target_names=class_names,
                                zero_division=0))
    return acc, preds, labels, cm, bal_acc, macro_f1

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    use_fake = args.offline or args.quick
    epochs = 1 if args.quick else args.epochs

    class_names = CIFAR10_CLASSES

    print("=" * 70)
    print("Lab Practical 4 – CNN Architectures for Imbalanced Classification")
    print(f"Device: {DEVICE}  |  Epochs: {epochs}  |  "
          f"Offline/Quick: {use_fake}")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # PS1 – Architecture Design: train CustomCNN with class-weighting     #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("PS1: Custom CNN Architecture with Regularisation")
    print("=" * 70)

    train_loader, test_loader, class_weights = make_loaders(
        batch_size=args.batch_size, use_fake=use_fake, quick=args.quick,
        use_sampler=False, augment=False,
    )
    print(f"  Train batches: {len(train_loader)}  | Test batches: {len(test_loader)}")

    custom_cnn = CustomCNN(num_classes=10, dropout=0.4)
    crit_ps1 = build_loss("weighted_ce", class_weights)
    opt_ps1 = optim.AdamW(custom_cnn.parameters(), lr=args.lr, weight_decay=1e-2)
    sched_ps1 = optim.lr_scheduler.CosineAnnealingLR(opt_ps1, T_max=epochs)

    hist_ps1 = train_model(custom_cnn, train_loader, test_loader,
                           crit_ps1, opt_ps1, epochs, "CustomCNN", sched_ps1)

    acc_ps1, preds_ps1, lbls_ps1, cm_ps1, bal_ps1, f1_ps1 = full_report(
        custom_cnn, test_loader, class_names, "PS1 – CustomCNN"
    )
    plot_confusion_matrix(cm_ps1, class_names,
                          "PS1: CustomCNN Confusion Matrix",
                          RESULTS_DIR / "ps1_confusion_matrix.png")

    # ------------------------------------------------------------------ #
    # PS2 – Imbalanced Dataset Handling: compare strategies               #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("PS2: Imbalanced Dataset Handling Strategies")
    print("=" * 70)

    strategies = {
        "Baseline (no handling)": {"use_sampler": False, "augment": False, "loss": "cross_entropy"},
        "Class Weighting":        {"use_sampler": False, "augment": False, "loss": "weighted_ce"},
        "Oversampling":           {"use_sampler": True,  "augment": False, "loss": "cross_entropy"},
        "Augmentation+Weighting": {"use_sampler": False, "augment": True,  "loss": "weighted_ce"},
    }

    ps2_results = {}
    ps2_histories = {}
    for strat_name, cfg in strategies.items():
        print(f"\n  Strategy: {strat_name}")
        tr_l, te_l, cw = make_loaders(
            batch_size=args.batch_size, use_fake=use_fake, quick=args.quick,
            use_sampler=cfg["use_sampler"], augment=cfg["augment"],
        )
        model = CustomCNN(num_classes=10).to(DEVICE)
        crit = build_loss(cfg["loss"], cw if cfg["loss"] != "cross_entropy" else None)
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
        hist = train_model(model, tr_l, te_l, crit, opt, epochs, strat_name)
        acc, _, _, _, bal_acc, f1 = full_report(model, te_l, class_names,
                                                 f"PS2 – {strat_name}")
        ps2_results[strat_name] = {"acc": acc, "bal_acc": bal_acc, "f1": f1}
        ps2_histories[strat_name] = hist

    plot_training_curves(ps2_histories, "PS2: Imbalance Handling Strategies",
                         RESULTS_DIR / "ps2_training_curves.png")

    # Summary table
    print("\n  PS2 Strategy Comparison:")
    print(f"  {'Strategy':<30} {'Acc':>6} {'Bal-Acc':>9} {'Macro-F1':>10}")
    print("  " + "-" * 60)
    for k, v in ps2_results.items():
        print(f"  {k:<30} {v['acc']:>5.1f}% {v['bal_acc']:>9.3f} {v['f1']:>10.3f}")

    # ------------------------------------------------------------------ #
    # PS3 – Comparative Architecture Analysis                             #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("PS3: Comparative Architecture Analysis")
    print("=" * 70)

    arch_map = {
        "CustomCNN":   CustomCNN(num_classes=10),
        "ResNet18":    make_resnet18(num_classes=10, pretrained=False),
        "DenseNet121": make_densenet121(num_classes=10, pretrained=False),
    }

    train_loader, test_loader, class_weights = make_loaders(
        batch_size=args.batch_size, use_fake=use_fake, quick=args.quick,
        use_sampler=True, augment=True,
    )

    ps3_histories: Dict[str, Dict] = {}
    ps3_results: Dict[str, Dict] = {}
    for arch_name, arch_model in arch_map.items():
        print(f"\n  Architecture: {arch_name}")
        n_params = sum(p.numel() for p in arch_model.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {n_params:,}")
        crit = build_loss("weighted_ce", class_weights)
        opt = optim.AdamW(arch_model.parameters(), lr=args.lr, weight_decay=1e-2)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        t0 = time.time()
        hist = train_model(arch_model, train_loader, test_loader,
                           crit, opt, epochs, arch_name, sched)
        elapsed = time.time() - t0
        acc, preds, lbls, cm, bal_acc, f1 = full_report(
            arch_model, test_loader, class_names, f"PS3 – {arch_name}"
        )
        ps3_histories[arch_name] = hist
        ps3_results[arch_name] = {
            "acc": acc, "bal_acc": bal_acc, "f1": f1,
            "params": n_params, "time_s": elapsed,
            "model": arch_model, "preds": preds, "lbls": lbls, "cm": cm,
        }
        plot_confusion_matrix(cm, class_names,
                              f"PS3: {arch_name} Confusion Matrix",
                              RESULTS_DIR / f"ps3_cm_{arch_name.lower()}.png")

    plot_training_curves(ps3_histories, "PS3: Architecture Comparison",
                         RESULTS_DIR / "ps3_training_curves.png")

    print("\n  PS3 Architecture Comparison:")
    print(f"  {'Arch':<14} {'Params':>10} {'Time(s)':>9} {'Acc':>6} {'Bal-Acc':>9} {'Macro-F1':>10}")
    print("  " + "-" * 65)
    for k, v in ps3_results.items():
        print(f"  {k:<14} {v['params']:>10,} {v['time_s']:>9.1f} "
              f"{v['acc']:>5.1f}% {v['bal_acc']:>9.3f} {v['f1']:>10.3f}")

    # ------------------------------------------------------------------ #
    # PS4 – Loss Function & Optimiser Experiment                          #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("PS4: Loss Function & Optimiser Experiment")
    print("=" * 70)

    loss_configs = [
        ("cross_entropy", 2.0), ("weighted_ce", 2.0),
        ("focal",         0.5), ("focal", 2.0), ("focal", 5.0),
        ("label_smoothing", 2.0),
    ]
    optimizer_names = ["sgd", "adam", "adamw", "rmsprop"]

    train_loader, test_loader, class_weights = make_loaders(
        batch_size=args.batch_size, use_fake=use_fake, quick=args.quick,
        use_sampler=False, augment=False,
    )

    ps4_results = []
    ps4_histories: Dict[str, Dict] = {}

    for loss_name, gamma in loss_configs:
        for opt_name in optimizer_names:
            tag = f"{loss_name}(gamma={gamma})+{opt_name}"
            print(f"\n  {tag}")
            model = CustomCNN(num_classes=10).to(DEVICE)
            crit = build_loss(loss_name, class_weights, gamma=gamma)
            opt = build_optimizer(opt_name, model.parameters(), lr=args.lr)
            sched = optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 2), gamma=0.5)
            hist = train_model(model, train_loader, test_loader,
                               crit, opt, epochs, tag, sched)
            acc, _, _, _, bal_acc, f1 = full_report(model, test_loader,
                                                     class_names, f"PS4 – {tag}")
            ps4_results.append({
                "loss": f"{loss_name}(gamma={gamma})", "optimizer": opt_name,
                "acc": acc, "bal_acc": bal_acc, "f1": f1,
            })
            ps4_histories[tag] = hist

    plot_training_curves(ps4_histories, "PS4: Loss × Optimiser Comparison",
                         RESULTS_DIR / "ps4_training_curves.png")

    print("\n  PS4 Results Summary:")
    print(f"  {'Loss Function':<30} {'Optimizer':<10} {'Acc':>6} {'Bal-Acc':>9} {'Macro-F1':>10}")
    print("  " + "-" * 72)
    for r in ps4_results:
        print(f"  {r['loss']:<30} {r['optimizer']:<10} {r['acc']:>5.1f}% "
              f"{r['bal_acc']:>9.3f} {r['f1']:>10.3f}")

    # ------------------------------------------------------------------ #
    # PS5 – Feature Visualisation                                         #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("PS5: Feature Representation & Visualisation")
    print("=" * 70)

    train_loader, test_loader, class_weights = make_loaders(
        batch_size=args.batch_size, use_fake=use_fake, quick=args.quick,
        use_sampler=False, augment=False,
    )

    # Reuse CustomCNN from PS1 (already trained)
    custom_cnn.eval()
    print("  Extracting features from CustomCNN …")
    feats, feat_labels = extract_features(custom_cnn, test_loader, max_samples=500)
    print(f"  Feature shape: {feats.shape}")

    plot_tsne(feats, feat_labels, "PS5: t-SNE – CustomCNN Features",
              RESULTS_DIR / "ps5_tsne.png")
    plot_pca(feats, feat_labels, "PS5: PCA – CustomCNN Features",
             RESULTS_DIR / "ps5_pca.png")
    plot_gradcam(custom_cnn, test_loader,
                 RESULTS_DIR / "ps5_gradcam.png", n_images=4)

    # ------------------------------------------------------------------ #
    # PS6 – Transfer Learning                                             #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("PS6: Transfer Learning – Pre-trained ResNet18 vs From-Scratch")
    print("=" * 70)

    train_loader, test_loader, class_weights = make_loaders(
        batch_size=args.batch_size, use_fake=use_fake, quick=args.quick,
        use_sampler=True, augment=True,
    )

    tl_results: Dict[str, Dict] = {}
    tl_histories: Dict[str, Dict] = {}
    for pretrained, tag in [(False, "ResNet18 (scratch)"),
                            (True,  "ResNet18 (ImageNet)")]:
        # Pre-trained weights need internet; skip silently in offline mode
        if pretrained and use_fake:
            print(f"  Skipping {tag} – offline/quick mode, no pre-trained weights.")
            continue
        print(f"\n  {tag}")
        try:
            model = make_resnet18(num_classes=10, pretrained=pretrained).to(DEVICE)
        except Exception as exc:
            print(f"  Could not load weights for {tag}: {exc}. Skipping.")
            continue
        crit = build_loss("weighted_ce", class_weights)
        # Fine-tuning: lower LR for pre-trained backbone
        lr_tl = args.lr / 10 if pretrained else args.lr
        opt = optim.AdamW(model.parameters(), lr=lr_tl, weight_decay=1e-2)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        hist = train_model(model, train_loader, test_loader,
                           crit, opt, epochs, tag, sched)
        acc, _, _, _, bal_acc, f1 = full_report(model, test_loader, class_names,
                                                 f"PS6 – {tag}")
        tl_results[tag] = {"acc": acc, "bal_acc": bal_acc, "f1": f1}
        tl_histories[tag] = hist

    if tl_histories:
        plot_training_curves(tl_histories, "PS6: Transfer Learning Comparison",
                             RESULTS_DIR / "ps6_training_curves.png")
    print("\n  PS6 Transfer Learning Results:")
    for k, v in tl_results.items():
        print(f"  {k:<30} Acc={v['acc']:.1f}%  Bal-Acc={v['bal_acc']:.3f}"
              f"  Macro-F1={v['f1']:.3f}")

    # ------------------------------------------------------------------ #
    # PS7 – Error Analysis                                                #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("PS7: Error Analysis & Improvement Proposals")
    print("=" * 70)

    # Use the best PS3 model (CustomCNN by default) for error analysis
    best_arch = max(ps3_results, key=lambda k: ps3_results[k]["bal_acc"])
    print(f"  Using best PS3 model for error analysis: {best_arch}")
    best_model = ps3_results[best_arch]["model"]
    best_preds = ps3_results[best_arch]["preds"]
    best_lbls = ps3_results[best_arch]["lbls"]
    best_cm = ps3_results[best_arch]["cm"]

    plot_confusion_matrix(best_cm, class_names,
                          f"PS7: {best_arch} Error Analysis Confusion Matrix",
                          RESULTS_DIR / "ps7_confusion_matrix.png")

    # Per-class accuracy
    per_class_acc = best_cm.diagonal() / best_cm.sum(axis=1).clip(min=1)
    print("\n  Per-class accuracy:")
    for i, (cname, pc_acc) in enumerate(zip(class_names, per_class_acc)):
        bar = "█" * int(pc_acc * 20)
        print(f"    {cname:>12}: {pc_acc*100:5.1f}%  {bar}")

    # Most confused pairs
    cm_no_diag = best_cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    top_confused = []
    for _ in range(5):
        idx = np.unravel_index(cm_no_diag.argmax(), cm_no_diag.shape)
        count = cm_no_diag[idx]
        if count == 0:
            break
        top_confused.append((class_names[idx[0]], class_names[idx[1]], count))
        cm_no_diag[idx] = 0

    print("\n  Top confused class pairs (true → predicted):")
    for true_cls, pred_cls, cnt in top_confused:
        print(f"    {true_cls:>12} → {pred_cls:<12}  count={cnt}")

    # Improvement proposals
    print("\n  Improvement Proposals:")
    proposals = [
        "1. Apply stronger augmentation (MixUp, CutMix) for minority classes.",
        "2. Use CBLoss (Class-Balanced Loss) with effective number of samples.",
        "3. Progressive re-sampling: increase minority weight over epochs.",
        "4. Ensemble diverse architectures (CustomCNN + ResNet18 + DenseNet).",
        "5. Self-supervised pre-training (SimCLR) on the full unlabelled set.",
        "6. Knowledge distillation: compress a large pre-trained model.",
        "7. Threshold calibration post-training for minority class recall.",
    ]
    for prop in proposals:
        print(f"  {prop}")

    # ------------------------------------------------------------------ #
    # Final Summary                                                        #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n  PS1 – CustomCNN:           Acc={acc_ps1:.1f}%  "
          f"Bal-Acc={bal_ps1:.3f}  Macro-F1={f1_ps1:.3f}")

    print("\n  PS2 – Imbalance Strategy Comparison:")
    for k, v in ps2_results.items():
        print(f"    {k:<30} Acc={v['acc']:.1f}%  Bal-Acc={v['bal_acc']:.3f}")

    print("\n  PS3 – Architecture Comparison:")
    for k, v in ps3_results.items():
        print(f"    {k:<14}  Acc={v['acc']:.1f}%  Bal-Acc={v['bal_acc']:.3f}"
              f"  Params={v['params']:,}")

    print("\n  PS4 – Best Loss+Optimizer:")
    if ps4_results:
        best_ps4 = max(ps4_results, key=lambda r: r["bal_acc"])
        print(f"    {best_ps4['loss']} + {best_ps4['optimizer']}  "
              f"Acc={best_ps4['acc']:.1f}%  Bal-Acc={best_ps4['bal_acc']:.3f}")

    print("\n  PS5 – Visualisations saved:")
    for f in ["ps5_tsne.png", "ps5_pca.png", "ps5_gradcam.png"]:
        fp = RESULTS_DIR / f
        print(f"    {'✓' if fp.exists() else '✗'} {f}")

    if tl_results:
        print("\n  PS6 – Transfer Learning:")
        for k, v in tl_results.items():
            print(f"    {k}  Acc={v['acc']:.1f}%  Bal-Acc={v['bal_acc']:.3f}")

    print("\n  PS7 – Best model for error analysis:", best_arch)
    print(f"    Balanced Accuracy: {ps3_results[best_arch]['bal_acc']:.3f}")

    print("\n" + "=" * 70)
    print("Lab Practical 4 Completed Successfully!")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
