"""
PCam Model Evaluation and Diagnostic Metrics Module.
Course Project: ES 335 (IIT Gandhinagar)

Provides shared evaluation utilities for binary tumor patch classification:
  1. compute_classification_metrics: Standardized metrics suite (ROC-AUC, PR-AUC, F1, Recall, Precision, Specificity)
  2. evaluate_model: Wrapper to evaluate any scikit-learn model on (X, y)
  3. plot_roc_curves: Publication-grade comparative ROC curve visualization
  4. plot_confusion_matrix: Annotated 2x2 diagnostic confusion matrix
  5. save_metrics_table: Centralized CSV persistence for model benchmarking
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    model_name: str = "Model",
    dataset_name: str = "Val"
) -> Dict[str, Any]:
    """
    Computes a comprehensive suite of binary classification metrics.
    
    Args:
        y_true: Binary ground truth array (0 = Normal, 1 = Tumor).
        y_pred: Predicted binary class array.
        y_prob: Optional predicted tumor probabilities (required for ROC-AUC & PR-AUC).
        model_name: Descriptive name of the model.
        dataset_name: Name of the evaluation split (e.g., 'Validation', 'Test').
        
    Returns:
        dict: Standardized metric dictionary.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    
    # Confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # Specificity = True Negative Rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    metrics = {
        "model": model_name,
        "dataset": dataset_name,
        "total_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if y_prob is not None else None,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if y_prob is not None else None,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }
    
    return metrics


def evaluate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = "Model",
    dataset_name: str = "Val"
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Evaluates a trained classifier on feature matrix X and labels y.
    
    Returns:
        metrics_dict: Dictionary of calculated metrics.
        y_pred: Predicted class labels.
        y_prob: Predicted positive-class probabilities.
    """
    y_pred = model.predict(X)
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        # Map decision function to pseudo-probabilities via sigmoid
        scores = model.decision_function(X)
        y_prob = 1.0 / (1.0 + np.exp(-scores))
    else:
        y_prob = y_pred.astype(float)
        
    metrics = compute_classification_metrics(
        y_true=y,
        y_pred=y_pred,
        y_prob=y_prob,
        model_name=model_name,
        dataset_name=dataset_name
    )
    
    return metrics, y_pred, y_prob


def plot_roc_curves(
    curves_data: List[Dict[str, Any]],
    output_path: Path,
    title: str = "ROC Curves Comparison - PCam Benchmark"
):
    """
    Plots publication-ready comparative ROC curves.
    
    Args:
        curves_data: List of dicts, each with keys:
            - 'y_true': Ground truth labels
            - 'y_prob': Predicted probabilities
            - 'label': Curve legend label (e.g., 'Track A (Full 96) [AUC = 0.842]')
            - 'color': Hex color string
            - 'linestyle': Optional linestyle (default: '-')
        output_path: Filepath where plot is saved.
        title: Plot title.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(8.5, 7.5), dpi=300)
    
    # Plot chance diagonal
    plt.plot([0, 1], [0, 1], color="#7f8c8d", linestyle="--", linewidth=1.5, label="Random Chance (AUC = 0.5000)")
    
    for item in curves_data:
        y_true = item["y_true"]
        y_prob = item["y_prob"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        
        label_str = item.get("label", f"Model (AUC = {auc_val:.4f})")
        color = item.get("color", "#2980b9")
        linestyle = item.get("linestyle", "-")
        linewidth = item.get("linewidth", 2.2)
        
        plt.plot(fpr, tpr, color=color, linestyle=linestyle, linewidth=linewidth, label=label_str)
        
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.03])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="semibold")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12, fontweight="semibold")
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.grid(True, linestyle="--", alpha=0.55)
    plt.legend(loc="lower right", fontsize=10.5, frameon=True, facecolor="#f8f9fa", edgecolor="#bdc3c7")
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved ROC curve comparison to: {output_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_path: Path,
    class_names: Tuple[str, str] = ("Normal (0)", "Tumor (1)")
):
    """
    Plots an annotated 2x2 confusion matrix heatmap with both raw counts and percentages.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    total = np.sum(cm)
    
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Tick marks
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, fontsize=11, fontweight="semibold")
    ax.set_yticklabels(class_names, fontsize=11, fontweight="semibold")
    
    # Annotations
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = (count / total) * 100
            text_color = "white" if count > thresh else "black"
            ax.text(
                j, i, f"{count:,}\n({pct:.1f}%)",
                ha="center", va="center",
                color=text_color, fontsize=12, fontweight="bold"
            )
            
    ax.set_ylabel("True Pathological Label", fontsize=12, fontweight="semibold")
    ax.set_xlabel("Model Predicted Label", fontsize=12, fontweight="semibold")
    ax.set_title(f"Confusion Matrix: {model_name}\nTotal Validation Patches: {total:,}", fontsize=13, fontweight="bold", pad=12)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved confusion matrix to: {output_path}")


def save_metrics_table(metrics_list: List[Dict[str, Any]], output_path: Path):
    """
    Persists a list of metrics dictionaries to a standardized CSV benchmark table.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(metrics_list)
    df.to_csv(output_path, index=False)
    print(f"[OK] Benchmark table saved to: {output_path} ({len(df)} models recorded)")

