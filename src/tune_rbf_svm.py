"""
Phase 6: Systematic Hyperparameter Tuning for RBF Kernel SVM on Full 40,000 Training Patches.
Course Project: ES 335 (IIT Gandhinagar)

Sweeps across:
  - Soft-margin penalty C in [0.5, 1.0, 2.0, 3.0, 4.0]
  - Gaussian kernel bandwidth gamma in [0.010, 'scale' (~0.0169), 0.025]
  - Cost-sensitive class weighting for clinical recall improvement
  
Evaluates strictly on the 5,000 unseen validation patches to prevent data leakage.
Exports:
  - outputs/results/rbf_svm_tuning_grid.csv
  - outputs/plots/rbf_svm_tuning_heatmap.png
  - outputs/plots/best_rbf_svm_cm_full96.png
  - outputs/models/Best_RBFSVM_full96.pkl
"""

import os
import sys
import time
from pathlib import Path
import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score,
    precision_score, f1_score, balanced_accuracy_score,
    confusion_matrix
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import plot_confusion_matrix

FEATURES_DIR = PROJECT_ROOT / "outputs" / "features"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"


def fit_and_eval(C, gamma, class_weight, X_train, y_train, X_val, y_val, combo_id, total_combos):
    weight_str = "None" if class_weight is None else str(class_weight)
    gamma_str = "scale" if gamma == "scale" else f"{gamma:.4f}"
    
    t0 = time.time()
    clf = SVC(
        C=C,
        kernel="rbf",
        gamma=gamma,
        class_weight=class_weight,
        cache_size=800,
        random_state=42
    )
    clf.fit(X_train, y_train)
    fit_time = time.time() - t0
    
    # Validation evaluation
    y_pred = clf.predict(X_val)
    scores = clf.decision_function(X_val)
    
    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, scores)
    rec = recall_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    bal_acc = balanced_accuracy_score(y_val, y_pred)
    n_sv = len(clf.support_)
    
    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"  [Done {combo_id:>2}/{total_combos}] C={C:<4} gamma={gamma_str:<7} weight={weight_str:<10} | Acc: {acc*100:.2f}% | AUC: {auc:.4f} | Sens: {rec*100:.2f}% | Prec: {prec*100:.2f}% | F1: {f1:.4f} | SV: {n_sv:,} ({fit_time:.1f}s)")
    
    return {
        "combo_id": combo_id,
        "C": C,
        "gamma": gamma_str,
        "class_weight": weight_str,
        "accuracy": acc,
        "roc_auc": auc,
        "recall_sensitivity": rec,
        "precision": prec,
        "f1_score": f1,
        "balanced_accuracy": bal_acc,
        "support_vectors": n_sv,
        "true_positives": tp,
        "false_negatives": fn,
        "true_negatives": tn,
        "false_positives": fp,
        "fit_time_sec": round(fit_time, 2)
    }


def main():
    print("=" * 95)
    print("PHASE 6: SYSTEMATIC HYPERPARAMETER TUNING FOR RBF KERNEL SVM (VALIDATION SET ONLY)")
    print("=" * 95)
    
    # 1. Load Data
    print("\n[1/5] Loading Track A standardized features (N=40,000 train, 5,000 val)...")
    train_full = pd.read_csv(FEATURES_DIR / "full96_train.csv")
    val_full = pd.read_csv(FEATURES_DIR / "full96_val.csv")
    
    feature_cols = [c for c in train_full.columns if c != "label"]
    X_train = train_full[feature_cols].values
    y_train = train_full["label"].values.astype(int)
    X_val = val_full[feature_cols].values
    y_val = val_full["label"].values.astype(int)
    
    # 2. Define Parameter Grid
    grid = [
        # (C, gamma, class_weight)
        (0.5, "scale", None),
        (1.0, 0.010, None),
        (1.0, "scale", None),      # Baseline reference (Phase 5)
        (1.0, 0.025, None),
        (2.0, 0.010, None),
        (2.0, "scale", None),
        (2.0, 0.025, None),
        (3.0, "scale", None),
        (4.0, "scale", None),
        (2.0, "scale", {0: 1.0, 1: 1.10}),   # Mild recall boost
        (2.0, "scale", "balanced")           # Full balance
    ]
    
    total_combos = len(grid)
    print(f"\n[2/5] Launching parallel grid search ({total_combos} combinations across 4 worker cores)...")
    print(f"      Grid includes C in [0.5, 1.0, 2.0, 3.0, 4.0], gamma in [0.010, 'scale', 0.025], and cost-weighting.")
    
    t_start = time.time()
    results = Parallel(n_jobs=4)(
        delayed(fit_and_eval)(
            c, g, w, X_train, y_train, X_val, y_val, i + 1, total_combos
        )
        for i, (c, g, w) in enumerate(grid)
    )
    total_time = time.time() - t_start
    print(f"\n[OK] Grid search completed in {total_time:.2f}s ({total_time / 60:.1f} min).")
    
    df_results = pd.DataFrame(results)
    
    # 3. Save Results Table
    print("\n[3/5] Saving grid results to outputs/results/rbf_svm_tuning_grid.csv...")
    csv_path = RESULTS_DIR / "rbf_svm_tuning_grid.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"[OK] Full tuning grid saved to: {csv_path}")
    
    # 4. Identify Best Configuration
    best_row_auc = df_results.sort_values(by="roc_auc", ascending=False).iloc[0]
    best_row_acc = df_results.sort_values(by="accuracy", ascending=False).iloc[0]
    
    print("\n" + "=" * 95)
    print("TOP 5 HYPERPARAMETER CONFIGURATIONS (Ranked by Validation ROC-AUC):")
    print("=" * 95)
    disp_cols = ["C", "gamma", "class_weight", "accuracy", "roc_auc", "recall_sensitivity", "precision", "f1_score", "support_vectors", "fit_time_sec"]
    top_df = df_results.sort_values(by="roc_auc", ascending=False)[disp_cols].copy()
    top_df["accuracy"] = (top_df["accuracy"] * 100).round(2).astype(str) + "%"
    top_df["recall_sensitivity"] = (top_df["recall_sensitivity"] * 100).round(2).astype(str) + "%"
    top_df["precision"] = (top_df["precision"] * 100).round(2).astype(str) + "%"
    top_df["roc_auc"] = top_df["roc_auc"].round(4)
    top_df["f1_score"] = top_df["f1_score"].round(4)
    print(top_df.head(5).to_string(index=False))
    print("=" * 95)
    
    # 5. Retrain & Save Final Frozen Best Model
    print(f"\n[4/5] Training final frozen Best Model (C={best_row_auc['C']}, gamma={best_row_auc['gamma']}, weight={best_row_auc['class_weight']})...")
    best_C = float(best_row_auc["C"])
    best_gamma = "scale" if best_row_auc["gamma"] == "scale" else float(best_row_auc["gamma"])
    best_weight = None if best_row_auc["class_weight"] == "None" else (
        "balanced" if best_row_auc["class_weight"] == "balanced" else {0: 1.0, 1: 1.10}
    )
    
    best_clf = SVC(
        C=best_C,
        kernel="rbf",
        gamma=best_gamma,
        class_weight=best_weight,
        cache_size=2000,
        random_state=42
    )
    best_clf.fit(X_train, y_train)
    
    model_path = MODELS_DIR / "Best_RBFSVM_full96.pkl"
    joblib.dump(best_clf, model_path)
    print(f"[OK] Best RBF SVM model binary saved to: {model_path}")
    
    # Save Confusion Matrix
    y_pred_best = best_clf.predict(X_val)
    cm_path = PLOTS_DIR / "best_rbf_svm_cm_full96.png"
    plot_confusion_matrix(
        y_val, y_pred_best,
        model_name=f"Tuned Best RBF Kernel SVM (C={best_C}, gamma={best_row_auc['gamma']})",
        output_path=cm_path
    )
    
    # 6. Generate 2D Tuning Heatmaps / Plots
    print("\n[5/5] Generating 2D Hyperparameter Tuning Heatmaps...")
    # Filter unweighted standard grid for clean 2D surface
    unweighted = df_results[df_results["class_weight"] == "None"].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Pivot for Accuracy
    pivot_acc = unweighted.pivot(index="C", columns="gamma", values="accuracy") * 100
    sns.heatmap(
        pivot_acc, annot=True, fmt=".2f", cmap="YlGnBu",
        ax=axes[0], cbar_kws={"label": "Validation Accuracy (%)"},
        linewidths=1.2, annot_kws={"fontsize": 11, "fontweight": "bold"}
    )
    axes[0].set_title("Validation Accuracy (%) across (C, gamma)", fontsize=13, fontweight="bold", pad=10)
    axes[0].set_xlabel("Kernel Bandwidth (gamma)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Regularization Penalty (C)", fontsize=11, fontweight="bold")
    
    # Pivot for ROC-AUC
    pivot_auc = unweighted.pivot(index="C", columns="gamma", values="roc_auc")
    sns.heatmap(
        pivot_auc, annot=True, fmt=".4f", cmap="magma",
        ax=axes[1], cbar_kws={"label": "Validation ROC-AUC"},
        linewidths=1.2, annot_kws={"fontsize": 11, "fontweight": "bold"}
    )
    axes[1].set_title("Validation ROC-AUC across (C, gamma)", fontsize=13, fontweight="bold", pad=10)
    axes[1].set_xlabel("Kernel Bandwidth (gamma)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Regularization Penalty (C)", fontsize=11, fontweight="bold")
    
    plt.tight_layout()
    heatmap_path = PLOTS_DIR / "rbf_svm_tuning_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Hyperparameter tuning heatmaps saved to: {heatmap_path}")
    
    print("\n" + "=" * 95)
    print("PHASE 6 RBF KERNEL SVM TUNING SUMMARY:")
    print(f"  Baseline (Default C=1.0, gamma='scale'): Acc = 82.14% | ROC-AUC = 0.9038 | Sens = 79.36% | Prec = 84.03%")
    print(f"  Tuned Champion (C={best_row_auc['C']}, gamma={best_row_auc['gamma']}, weight={best_row_auc['class_weight']}):")
    print(f"    -> Accuracy:    {best_row_auc['accuracy']*100:.2f}%  ({best_row_auc['accuracy']*100 - 82.14:+.2f}% gain)")
    print(f"    -> ROC-AUC:     {best_row_auc['roc_auc']:.4f}  ({best_row_auc['roc_auc'] - 0.9038:+.4f} gain)")
    print(f"    -> Sensitivity: {best_row_auc['recall_sensitivity']*100:.2f}%")
    print(f"    -> Precision:   {best_row_auc['precision']*100:.2f}%")
    print(f"    -> F1-Score:    {best_row_auc['f1_score']:.4f}")
    print("=" * 95)


if __name__ == "__main__":
    main()

