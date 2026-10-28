"""
Deep Learning (CNN) Benchmark on PatchCamelyon (PCam): ResNet-18
Course Project: ES 335 - Machine Learning (IIT Gandhinagar)
Authors: Devarsh Patel (24110245) & Dhrumil Bhutaiya (24110075)

This script trains and evaluates a Convolutional Neural Network (ResNet-18)
directly on the raw histopathology patches of the PatchCamelyon dataset.

Objectives:
  1. Break through the handcrafted feature representation ceiling (~79.8% test accuracy).
  2. Leverage pathology-specific data augmentations (rotational invariance: 0, 90, 180, 270 deg).
  3. Conduct sealed one-shot evaluation on the untouched 5,000 held-out test cohort.
  4. Save model predictions for forensic consensus error comparison.

Usage:
  python src/train_cnn.py
  python src/train_cnn.py --epochs 10 --batch-size 128 --lr 2e-4
  python src/train_cnn.py --eval-only --checkpoint outputs/cnn/best_resnet18.pt
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    auc,
    brier_score_loss,
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "cnn"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "plots"


# -----------------------------------------------------------------------------
# 1. Dataset & Rotational Histopathology Augmentations
# -----------------------------------------------------------------------------

class PCamDataset(Dataset):
    """
    Fast PyTorch Dataset for PatchCamelyon histopathology patches.
    Pre-caches all file paths in memory during initialization to eliminate
    expensive disk existence checks during batch collation.
    """
    def __init__(self, split_dir, labels_csv, transform=None):
        self.split_dir = str(split_dir)
        self.df = pd.read_csv(labels_csv)
        self.transform = transform

        # Pre-cache image paths in memory as strings
        self.img_paths = [
            os.path.join(self.split_dir, f) if f.startswith("images")
            else os.path.join(self.split_dir, "images", f)
            for f in self.df["filename"].values
        ]
        self.labels = self.df["label"].values.astype(np.float32)
        self.ids = [str(x) for x in self.df.get("id", range(len(self.df)))]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx], self.ids[idx], img_path


def build_transforms():
    """
    Builds data augmentation pipeline for histopathology images.
    In optical microscopy, tissue slides have no inherent gravitational orientation;
    therefore, 8-fold dihedral symmetry (horizontal/vertical flips + 90-degree rotations)
    ensures strict rotational invariance.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomChoice([
            transforms.RandomRotation((0, 0)),
            transforms.RandomRotation((90, 90)),
            transforms.RandomRotation((180, 180)),
            transforms.RandomRotation((270, 270)),
        ]),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        normalize,
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, eval_transform


# -----------------------------------------------------------------------------
# 2. Model Architecture
# -----------------------------------------------------------------------------

def build_resnet18(pretrained=True):
    """
    Builds ResNet-18 initialized with ImageNet transfer learning weights.
    Replaces the 1000-class linear classification head with a single logit output.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model


# -----------------------------------------------------------------------------
# 3. Training & Validation Passes
# -----------------------------------------------------------------------------

def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device):
    """Executes one training epoch with Automatic Mixed Precision (AMP)."""
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for images, targets, _, _ in train_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.float().unsqueeze(1).to(device, non_blocking=True)
        optimizer.zero_grad()

        if scaler:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(outputs).detach().cpu().numpy().flatten()
        all_preds.extend(probs)
        all_targets.extend(targets.cpu().numpy().flatten())

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_auc = roc_auc_score(all_targets, all_preds)
    return epoch_loss, epoch_auc


def validate(model, val_loader, criterion, device):
    """Executes validation evaluation without gradient computation."""
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, targets, _, _ in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.float().unsqueeze(1).to(device, non_blocking=True)

            if device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_preds.extend(probs)
            all_targets.extend(targets.cpu().numpy().flatten())

    val_loss = running_loss / len(val_loader.dataset)
    binary_preds = (np.array(all_preds) >= 0.5).astype(int)
    val_acc = accuracy_score(all_targets, binary_preds)
    val_auc = roc_auc_score(all_targets, all_preds)
    return val_loss, val_acc, val_auc


# -----------------------------------------------------------------------------
# 4. Sealed Test Evaluation & Metrics
# -----------------------------------------------------------------------------

def evaluate_test_cohort(model, test_loader, device, output_csv=None):
    """
    Evaluates the trained model strictly one-shot on the 5,000 held-out test cohort.
    Calculates diagnostic metrics: Accuracy, ROC-AUC, PR-AUC, Sensitivity, Specificity,
    Precision, F1-Score, Brier Score, and full Confusion Matrix.
    """
    model.eval()
    t_preds, t_targets, t_ids, t_paths = [], [], [], []

    with torch.no_grad():
        for images, targets, p_ids, p_paths in test_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            t_preds.extend(probs)
            t_targets.extend(targets.numpy().flatten())
            t_ids.extend(p_ids)
            t_paths.extend(p_paths)

    t_preds = np.array(t_preds)
    t_targets = np.array(t_targets).astype(int)
    t_binary = (t_preds >= 0.5).astype(int)

    test_acc = accuracy_score(t_targets, t_binary)
    test_auc = roc_auc_score(t_targets, t_preds)
    prec_c, rec_c, _ = precision_recall_curve(t_targets, t_preds)
    test_pr = auc(rec_c, prec_c)

    cm = confusion_matrix(t_targets, t_binary)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0
    brier = brier_score_loss(t_targets, t_preds)

    metrics = {
        "model": "ResNet-18 (Deep CNN)",
        "test_accuracy": test_acc,
        "test_roc_auc": test_auc,
        "test_pr_auc": test_pr,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1_score": f1,
        "brier_score": brier,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }

    print("\n" + "=" * 65)
    print(" FINAL SEALED PCAM TEST COHORT EVALUATION (RESNET-18)")
    print("=" * 65)
    print(f" Test Accuracy:     {test_acc * 100:.2f}%")
    print(f" Test ROC-AUC:      {test_auc:.4f}")
    print(f" Test PR-AUC:       {test_pr:.4f}")
    print(f" Sensitivity:       {sens * 100:.2f}%")
    print(f" Specificity:       {spec * 100:.2f}%")
    print(f" Precision:         {prec * 100:.2f}%")
    print(f" F1-Score:          {f1:.4f}")
    print(f" Brier Score:       {brier:.4f}")
    print("-" * 65)
    print(f" Confusion Matrix:  TN={tn:,} | FP={fp:,} | FN={fn:,} | TP={tp:,}")
    print("=" * 65)

    if output_csv:
        test_df = pd.DataFrame({
            "id": t_ids,
            "filename": t_paths,
            "y_true": t_targets,
            "y_pred": t_binary,
            "prob_tumor": np.round(t_preds, 5),
        })
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        test_df.to_csv(output_csv, index=False)
        print(f"[Saved] Test cohort predictions exported to: {output_csv}")

    return metrics


# -----------------------------------------------------------------------------
# 5. Main Execution Entry Point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train ResNet-18 on PCam Histopathology Benchmark")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for DataLoaders (default: 128)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for AdamW (default: 2e-4)")
    parser.add_argument("--data-dir", type=str, default=None, help="Root dataset directory containing train/val/test")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUTS_DIR), help="Output directory for checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to pre-trained checkpoint for evaluation")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and run test evaluation only")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using computing device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Auto-detect dataset directory
    data_dir = args.data_dir
    if data_dir is None:
        candidates = [
            PROJECT_ROOT / "pcam_tumor_detection" / "data",
            PROJECT_ROOT / "data",
            PROJECT_ROOT.parent / "data",
            Path("/kaggle/input/datasets/pateldevarshh28/pcam-data"),
            Path("/kaggle/input/pcam-data"),
        ]
        for c in candidates:
            if (c / "train").exists():
                data_dir = str(c)
                break

    if data_dir is None or not os.path.exists(data_dir):
        raise FileNotFoundError(
            "Could not locate dataset folder containing train/validation/test! "
            "Please specify via --data-dir <path>."
        )

    print(f"Data directory: {data_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_transform, eval_transform = build_transforms()

    test_dataset = PCamDataset(
        os.path.join(data_dir, "test"),
        os.path.join(data_dir, "test", "test_labels.csv"),
        transform=eval_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_resnet18(pretrained=True).to(device)
    checkpoint_path = args.checkpoint or os.path.join(args.output_dir, "best_resnet18.pt")

    if not args.eval_only:
        train_dataset = PCamDataset(
            os.path.join(data_dir, "train"),
            os.path.join(data_dir, "train", "train_labels.csv"),
            transform=train_transform
        )
        val_dataset = PCamDataset(
            os.path.join(data_dir, "validation"),
            os.path.join(data_dir, "validation", "validation_labels.csv"),
            transform=eval_transform
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        print(f"Dataset splits: Train={len(train_dataset):,} | Val={len(val_dataset):,} | Test={len(test_dataset):,}")

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

        best_val_auc = 0.0
        print("=" * 70)
        print(f"STARTING RESNET-18 TRAINING ({args.epochs} EPOCHS, BATCH {args.batch_size})")
        print("=" * 70)

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            tr_loss, tr_auc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
            scheduler.step()
            vl_loss, vl_acc, vl_auc = validate(model, val_loader, criterion, device)
            dur = time.time() - t0

            print(
                f"Epoch [{epoch:02d}/{args.epochs:02d}] ({dur:.1f}s) "
                f"Train Loss: {tr_loss:.4f} | Train AUC: {tr_auc:.4f} || "
                f"Val Acc: {vl_acc * 100:6.2f}% | Val AUC: {vl_auc:.4f}"
            )

            if vl_auc > best_val_auc:
                best_val_auc = vl_auc
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  * Saved new best model checkpoint (Val AUC: {vl_auc:.4f})")

        print(f"\nTraining completed. Peak Validation AUC: {best_val_auc:.4f}")

    # Sealed Test Evaluation
    if os.path.exists(checkpoint_path):
        print(f"\nLoading best checkpoint for test evaluation: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    pred_csv_path = os.path.join(args.output_dir, "test_predictions_resnet18.csv")
    metrics = evaluate_test_cohort(model, test_loader, device, output_csv=pred_csv_path)

    # Save summary metrics to results folder
    metrics_csv = RESULTS_DIR / "cnn_benchmark_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_csv, index=False)
    print(f"[Saved] CNN Benchmark summary exported to: {metrics_csv}")


if __name__ == "__main__":
    main()

