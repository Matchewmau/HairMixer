"""
5-Fold Cross-Validation for ResNet50 Face Shape Classifier
===========================================================
Evaluates the trained ResNet50 model using stratified 5-fold cross-validation
on the training dataset.

Usage:
    python cross_validation_5fold.py
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for image export
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from torchvision.models import resnet50
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
from collections import Counter
import warnings
from PIL import ImageFile

warnings.filterwarnings("ignore")

# Allow loading truncated images instead of crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True


class SafeImageFolder(datasets.ImageFolder):
    """ImageFolder that skips corrupt / unreadable images."""

    def __getitem__(self, index):
        while True:
            try:
                return super().__getitem__(index)
            except (OSError, IOError, SyntaxError) as e:
                print(f"    [SKIP] Corrupt image at index {index}: {self.imgs[index][0]}")
                index = (index + 1) % len(self)


# ─────────────────────────  Configuration  ──────────────────────────
MODEL_PATH = r"d:\CODING\Python\HairMixer\backend\hairmixer_app\ml\models\resnet50_best_model.pth"
DATASET_DIR = r"D:\Documents\Thesis\FaceShape_Dataset\testing_set"
NUM_CLASSES = 5
NUM_FOLDS = 5
BATCH_SIZE = 16
IMAGE_SIZE = 224
RANDOM_SEED = 42

# Class mapping (must match model.py)
CLASS_NAMES = {0: "heart", 1: "oblong", 2: "oval", 3: "round", 4: "square"}

# ─────────────────────────  Model Definition  ──────────────────────────
class ResNet50FaceShapeClassifier(nn.Module):
    """Exact replica of the model architecture from resnet_classifier.py"""

    def __init__(self, num_classes: int = 5, pretrained: bool = False):
        super().__init__()
        base_model = resnet50(weights=None)

        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool

        in_features = base_model.fc.in_features  # 2048
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 2048),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(2048),
            nn.Dropout(p=0.4),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ─────────────────────────  Helpers  ──────────────────────────
def load_pretrained_weights(model, path, device):
    """Load checkpoint weights (handles dict / raw state_dict / DataParallel)."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        state = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint
        )
    else:
        state = checkpoint

    # Strip optional DataParallel prefix
    if any(k.startswith("module.") for k in state.keys()):
        state = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state.items()
        }

    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        print(f"WARNING: State-dict mismatch! missing={missing[:5]}  unexpected={unexpected[:5]}")
    return model


def get_transforms():
    """ImageNet-normalised evaluation transform (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def print_separator(char="=", width=70):
    print(char * width)


def print_header(text, width=70):
    print_separator()
    print(f"  {text}")
    print_separator()


# ─────────────────────────  Main  ──────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_header("5-FOLD CROSS-VALIDATION  -  ResNet50 Face Shape Classifier")
    print(f"  Device        : {device}")
    print(f"  Model path    : {MODEL_PATH}")
    print(f"  Dataset       : {DATASET_DIR}")
    print(f"  Folds         : {NUM_FOLDS}")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Image size    : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Random seed   : {RANDOM_SEED}")
    print()

    # ── 1. Load full dataset (no augmentation) ──
    full_dataset = SafeImageFolder(
        root=DATASET_DIR,
        transform=get_transforms(),
    )
    targets = np.array(full_dataset.targets)
    class_to_idx = full_dataset.class_to_idx  # e.g. {'Heart': 0, ...}
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    print(f"  Total samples : {len(full_dataset)}")
    print(f"  Classes       : {class_to_idx}")
    dist = Counter(targets)
    for idx in sorted(dist):
        print(f"    [{idx}] {idx_to_class[idx]:>8s} : {dist[idx]} images")
    print()

    # ── 2. Stratified K-Fold ──
    skf = StratifiedKFold(
        n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED
    )

    fold_results = []
    all_true = []
    all_pred = []
    all_probs = []

    total_start = time.perf_counter()

    for fold_idx, (train_indices, val_indices) in enumerate(
        skf.split(np.zeros(len(targets)), targets), start=1
    ):
        print_separator("-")
        print(f"  FOLD {fold_idx}/{NUM_FOLDS}")
        print_separator("-")

        val_subset = Subset(full_dataset, val_indices)
        val_loader = DataLoader(
            val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )

        # Show fold distribution
        val_targets = targets[val_indices]
        val_dist = Counter(val_targets)
        print(f"  Val samples: {len(val_indices)}  |  Train samples: {len(train_indices)}")
        for cidx in sorted(val_dist):
            print(f"    [{cidx}] {idx_to_class[cidx]:>8s} : {val_dist[cidx]}")

        # ── Load fresh model with pretrained weights ──
        model = ResNet50FaceShapeClassifier(num_classes=NUM_CLASSES)
        model = load_pretrained_weights(model, MODEL_PATH, device)
        model.to(device)
        model.eval()

        # ── Evaluate ──
        fold_true = []
        fold_pred = []
        fold_probs = []

        t0 = time.perf_counter()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                fold_true.extend(labels.numpy())
                fold_pred.extend(preds)
                fold_probs.extend(probs)

        elapsed = time.perf_counter() - t0

        fold_true = np.array(fold_true)
        fold_pred = np.array(fold_pred)

        fold_acc = accuracy_score(fold_true, fold_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            fold_true, fold_pred, average="weighted", zero_division=0
        )

        fold_results.append({
            "fold": fold_idx,
            "accuracy": fold_acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "n_val": len(val_indices),
            "elapsed_s": elapsed,
        })

        all_true.extend(fold_true)
        all_pred.extend(fold_pred)
        all_probs.extend(fold_probs)

        print(f"\n  Accuracy  : {fold_acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1 Score  : {f1:.4f}")
        print(f"  Time      : {elapsed:.1f}s")

        # Per-class report for this fold
        print(f"\n  Per-class (Fold {fold_idx}):")
        cr = classification_report(
            fold_true,
            fold_pred,
            target_names=[idx_to_class[i] for i in range(NUM_CLASSES)],
            digits=4,
            zero_division=0,
        )
        for line in cr.split("\n"):
            print(f"    {line}")
        print()

    # ── 3. Aggregate Results ──
    total_elapsed = time.perf_counter() - total_start
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    print_header("AGGREGATE RESULTS  (all folds combined)")

    accs = [r["accuracy"] for r in fold_results]
    precs = [r["precision"] for r in fold_results]
    recs = [r["recall"] for r in fold_results]
    f1s = [r["f1"] for r in fold_results]

    print(f"\n  {'Fold':<6} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for r in fold_results:
        print(
            f"  {r['fold']:<6} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
            f"{r['recall']:>10.4f} {r['f1']:>10.4f}"
        )
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    print(
        f"  {'Mean':<6} {np.mean(accs):>10.4f} {np.mean(precs):>10.4f} "
        f"{np.mean(recs):>10.4f} {np.mean(f1s):>10.4f}"
    )
    print(
        f"  {'Std':<6} {np.std(accs):>10.4f} {np.std(precs):>10.4f} "
        f"{np.std(recs):>10.4f} {np.std(f1s):>10.4f}"
    )
    print(
        f"  {'Min':<6} {np.min(accs):>10.4f} {np.min(precs):>10.4f} "
        f"{np.min(recs):>10.4f} {np.min(f1s):>10.4f}"
    )
    print(
        f"  {'Max':<6} {np.max(accs):>10.4f} {np.max(precs):>10.4f} "
        f"{np.max(recs):>10.4f} {np.max(f1s):>10.4f}"
    )

    # ── Overall classification report ──
    print("\n  Overall Classification Report (all predictions pooled):")
    cr_all = classification_report(
        all_true,
        all_pred,
        target_names=[idx_to_class[i] for i in range(NUM_CLASSES)],
        digits=4,
        zero_division=0,
    )
    for line in cr_all.split("\n"):
        print(f"    {line}")

    # ── Confusion Matrix ──
    cm = confusion_matrix(all_true, all_pred)
    labels = [idx_to_class[i] for i in range(NUM_CLASSES)]

    print("\n  Confusion Matrix:")
    header_str = "          " + "".join(f"{l:>9s}" for l in labels)
    print(f"    {header_str}")
    print(f"    {'-' * (10 + 9 * NUM_CLASSES)}")
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:>9d}" for v in row)
        print(f"    {labels[i]:>8s} |{row_str}")

    # ── 95% Confidence Interval for mean accuracy ──
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    ci_95 = 1.96 * std_acc / np.sqrt(NUM_FOLDS)

    print(f"\n  +---------------------------------------------+")
    print(f"  |  Mean Accuracy : {mean_acc:.4f} +/- {std_acc:.4f}         |")
    print(f"  |  95% CI        : [{mean_acc - ci_95:.4f}, {mean_acc + ci_95:.4f}]      |")
    print(f"  |  Mean F1       : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}         |")
    print(f"  |  Total Time    : {total_elapsed:.1f}s                     |")
    print(f"  +---------------------------------------------+")
    print()

    # ── 4. Export results as images ──
    export_results_as_images(
        fold_results, all_true, all_pred, cm, labels,
        accs, precs, recs, f1s, mean_acc, std_acc, ci_95,
    )


# ─────────────────────  Image Export  ──────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "cv_results")


def export_results_as_images(
    fold_results, all_true, all_pred, cm, labels,
    accs, precs, recs, f1s, mean_acc, std_acc, ci_95,
):
    """Generate and save publication-quality result images."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Shared style
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
    })

    _plot_confusion_matrix(cm, labels)
    _plot_fold_accuracy(fold_results, mean_acc, std_acc)
    _plot_per_class_metrics(all_true, all_pred, labels)
    _plot_summary_dashboard(fold_results, cm, labels, mean_acc, std_acc, ci_95, f1s)

    print(f"  >> All images saved to: {os.path.abspath(OUTPUT_DIR)}")


def _plot_confusion_matrix(cm, labels):
    """Heatmap confusion matrix with counts and percentages."""
    fig, ax = plt.subplots(figsize=(8, 6.5))

    # Normalise rows to percentages
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    im = ax.imshow(cm_pct, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall (%)", fontsize=11)

    # Annotate cells with count + percentage
    thresh = cm_pct.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_pct[i, j] > thresh else "black"
            ax.text(
                j, i,
                f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                ha="center", va="center", fontsize=9,
                color=color, fontweight="bold" if i == j else "normal",
            )

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix  -  5-Fold Cross-Validation", pad=12)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  >> Saved: {path}")


def _plot_fold_accuracy(fold_results, mean_acc, std_acc):
    """Bar chart showing accuracy per fold + mean line."""
    fig, ax = plt.subplots(figsize=(8, 5))

    folds = [r["fold"] for r in fold_results]
    accs = [r["accuracy"] * 100 for r in fold_results]
    mean_pct = mean_acc * 100
    std_pct = std_acc * 100

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    bars = ax.bar(folds, accs, color=colors, width=0.6, edgecolor="white", linewidth=1.2)

    # Mean line
    ax.axhline(y=mean_pct, color="#D62728", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.fill_between(
        [0.4, len(folds) + 0.6],
        mean_pct - std_pct, mean_pct + std_pct,
        color="#D62728", alpha=0.08,
    )
    ax.text(
        len(folds) + 0.35, mean_pct,
        f"Mean: {mean_pct:.2f}%\n(+/-{std_pct:.2f}%)",
        va="center", fontsize=9, color="#D62728", fontweight="bold",
    )

    # Value labels on bars
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{acc:.2f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_xlabel("Fold", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Per-Fold Accuracy  -  5-Fold Cross-Validation", pad=12)
    ax.set_xticks(folds)
    ax.set_xticklabels([f"Fold {f}" for f in folds])

    # Y-axis: zoom in around results
    y_min = min(accs) - 1.5
    y_max = max(accs) + 1.5
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fold_accuracy.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  >> Saved: {path}")


def _plot_per_class_metrics(all_true, all_pred, labels):
    """Grouped bar chart: precision, recall, F1 per class."""
    from sklearn.metrics import precision_recall_fscore_support

    prec, rec, f1, sup = precision_recall_fscore_support(
        all_true, all_pred, average=None, zero_division=0,
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(labels))
    w = 0.22

    b1 = ax.bar(x - w, prec * 100, w, label="Precision", color="#4C72B0", edgecolor="white")
    b2 = ax.bar(x, rec * 100, w, label="Recall", color="#55A868", edgecolor="white")
    b3 = ax.bar(x + w, f1 * 100, w, label="F1 Score", color="#C44E52", edgecolor="white")

    # Value labels
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.15,
                f"{h:.1f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xlabel("Face Shape Class", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
    ax.set_title("Per-Class Metrics  -  5-Fold Cross-Validation", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)

    y_min = min(min(prec), min(rec), min(f1)) * 100 - 2
    ax.set_ylim(y_min, 100.5)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "per_class_metrics.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  >> Saved: {path}")


def _plot_summary_dashboard(fold_results, cm, labels, mean_acc, std_acc, ci_95, f1s):
    """Single combined dashboard image with all key results."""
    from sklearn.metrics import precision_recall_fscore_support

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "5-Fold Cross-Validation Results  -  ResNet50 Face Shape Classifier",
        fontsize=16, fontweight="bold", y=0.98,
    )

    # ── Top-left: Summary table ──
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.axis("off")

    mean_f1 = np.mean(f1s)
    std_f1 = np.std(f1s)
    table_data = [
        ["Total Samples", "7,363"],
        ["Classes", "5"],
        ["Folds", "5"],
        ["Mean Accuracy", f"{mean_acc*100:.2f}% +/- {std_acc*100:.2f}%"],
        ["95% CI", f"[{(mean_acc-ci_95)*100:.2f}%, {(mean_acc+ci_95)*100:.2f}%]"],
        ["Mean F1 Score", f"{mean_f1*100:.2f}% +/- {std_f1*100:.2f}%"],
    ]
    table = ax1.table(
        cellText=table_data,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)
    # Style header
    for j in range(2):
        table[0, j].set_facecolor("#4C72B0")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(table_data) + 1):
        for j in range(2):
            table[i, j].set_facecolor("#f0f4fa" if i % 2 == 0 else "white")
    ax1.set_title("Summary", fontsize=13, fontweight="bold", pad=20)

    # ── Top-right: Fold accuracy bars ──
    ax2 = fig.add_subplot(2, 2, 2)
    folds = [r["fold"] for r in fold_results]
    fold_accs = [r["accuracy"] * 100 for r in fold_results]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    bars = ax2.bar(folds, fold_accs, color=colors, width=0.6, edgecolor="white")
    ax2.axhline(y=mean_acc*100, color="#D62728", linestyle="--", linewidth=1.5, alpha=0.7)
    for bar, acc in zip(bars, fold_accs):
        ax2.text(
            bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
            f"{acc:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_xticks(folds)
    ax2.set_xticklabels([f"Fold {f}" for f in folds])
    ax2.set_ylim(min(fold_accs) - 1.5, max(fold_accs) + 1.2)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_title("Per-Fold Accuracy", fontsize=13, fontweight="bold")

    # ── Bottom-left: Confusion matrix ──
    ax3 = fig.add_subplot(2, 2, 3)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    im = ax3.imshow(cm_pct, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
    thresh = cm_pct.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_pct[i, j] > thresh else "black"
            ax3.text(
                j, i, f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)",
                ha="center", va="center", fontsize=8, color=color,
                fontweight="bold" if i == j else "normal",
            )
    ax3.set_xticks(range(len(labels)))
    ax3.set_yticks(range(len(labels)))
    ax3.set_xticklabels(labels, fontsize=9)
    ax3.set_yticklabels(labels, fontsize=9)
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("True")
    ax3.set_title("Confusion Matrix", fontsize=13, fontweight="bold")

    # ── Bottom-right: Per-class F1, precision, recall ──
    ax4 = fig.add_subplot(2, 2, 4)
    all_true_arr = np.array([])
    all_pred_arr = np.array([])
    # Reconstruct from cm
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            all_true_arr = np.concatenate([all_true_arr, np.full(cm[i, j], i)])
            all_pred_arr = np.concatenate([all_pred_arr, np.full(cm[i, j], j)])

    prec_c, rec_c, f1_c, _ = precision_recall_fscore_support(
        all_true_arr, all_pred_arr, average=None, zero_division=0,
    )
    x = np.arange(len(labels))
    w = 0.22
    ax4.bar(x - w, prec_c * 100, w, label="Precision", color="#4C72B0", edgecolor="white")
    ax4.bar(x, rec_c * 100, w, label="Recall", color="#55A868", edgecolor="white")
    ax4.bar(x + w, f1_c * 100, w, label="F1", color="#C44E52", edgecolor="white")
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, fontsize=9)
    ax4.set_ylabel("Score (%)")
    ax4.set_ylim(min(min(prec_c), min(rec_c), min(f1_c)) * 100 - 2, 100.5)
    ax4.legend(loc="lower right", fontsize=9)
    ax4.grid(axis="y", alpha=0.3)
    ax4.set_title("Per-Class Metrics", fontsize=13, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, "summary_dashboard.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  >> Saved: {path}")


if __name__ == "__main__":
    main()
