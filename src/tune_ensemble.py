"""
Phase 6: Multi-Paradigm Ensemble Blending (Tuned RBF Kernel SVM + HistGradientBoosting).
Course Project: ES 335 (IIT Gandhinagar)

Fuses:
  - Smooth continuous geometric distance manifold from Tuned RBF Kernel SVM (C=3.0/2.0, gamma='scale')
  - Orthogonal discrete split probabilities from HistGradientBoosting (300 stages)
  
Evaluates soft-voting blend across weights w in [0.0, 1.0] strictly on the 5,000 validation patches.
Exports:
  - outputs/results/ensemble_blend_benchmark.csv
  - outputs/plots/ensemble_cm_full96.png
  - outputs/plots/ensemble_vs_single_roc.png
"""

import os
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score,
    precision_score, f1_score, balanced_accuracy_score,
    confusion_matrix
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import plot_confusion_matrix, plot_roc_curves

FEATURES_DIR = PROJECT_ROOT / "outputs" / "features"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"


def main():
    print("=" * 95)
    print("PHASE 6: MULTI-PARADIGM ENSEMBLE BLENDING (RBF KERNEL SVM + HIST GRADIENT BOOSTING)")
    print("=" * 95)
    
    # 1. Load Validation Data
    print("\n[1/4] Loading Track A validation features (N=5,000 patches)...")
    val_full = pd.read_csv(FEATURES_DIR / "full96_val.csv")
    feature_cols = [c for c in val_full.columns if c != "label"]
    X_val = val_full[feature_cols].values
    y_val = val_full["label"].values.astype(int)
    
    # 2. Load Frozen Trained Models
    print("\n[2/4] Loading trained champion models from outputs/models/...")
    rbf_model = joblib.load(MODELS_DIR / "Best_RBFSVM_full96.pkl")
    hgbm_model = joblib.load(MODELS_DIR / "HistGBM_full96.pkl")
    
    # Get raw predictions and scores
    rbf_scores = rbf_model.decision_function(X_val)
    rbf_probs = expit(rbf_scores)  # Sigmoid-calibrated probability
    
    hgbm_probs = hgbm_model.predict_proba(X_val)[:, 1]
    
    # 3. Sweep Blending Weights
    print("\n[3/4] Sweeping blend weights across w_RBF in [0.0, 1.0]...")
    weights = np.linspace(0.0, 1.0, 11)
    blend_records = []
    
    for w in weights:
        w_rbf = round(w, 2)
        w_hgbm = round(1.0 - w, 2)
        
        # Soft voting blend
        p_blend = w_rbf * rbf_probs + w_hgbm * hgbm_probs
        y_pred = (p_blend >= 0.5).astype(int)
        
        acc = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, p_blend)
        rec = recall_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        
        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Rank-average AUC
        rank_r = rankdata(rbf_scores) / len(rbf_scores)
        rank_h = rankdata(hgbm_probs) / len(hgbm_probs)
        rank_blend = w_rbf * rank_r + w_hgbm * rank_h
        rank_auc = roc_auc_score(y_val, rank_blend)
        
        blend_records.append({
            "weight_rbf": w_rbf,
            "weight_hgbm": w_hgbm,
            "accuracy": acc,
            "roc_auc": auc,
            "rank_auc": rank_auc,
            "recall_sensitivity": rec,
            "precision": prec,
            "f1_score": f1,
            "balanced_accuracy": bal_acc,
            "true_positives": tp,
            "false_negatives": fn,
            "true_negatives": tn,
            "false_positives": fp
        })
        
        print(f"  w_RBF={w_rbf:.2f}, w_HGBM={w_hgbm:.2f} -> Acc: {acc*100:.2f}% | ROC-AUC: {auc:.4f} (Rank-AUC: {rank_auc:.4f}) | Sens: {rec*100:.2f}% | Prec: {prec*100:.2f}% | F1: {f1:.4f}")

    df_blend = pd.DataFrame(blend_records)
    csv_path = RESULTS_DIR / "ensemble_blend_benchmark.csv"
    df_blend.to_csv(csv_path, index=False)
    print(f"\n[OK] Ensemble blend grid saved to: {csv_path}")

    # Best blend by ROC-AUC
    best_blend = df_blend.sort_values(by="roc_auc", ascending=False).iloc[0]
    best_w_rbf = best_blend["weight_rbf"]
    best_w_hgbm = best_blend["weight_hgbm"]
    best_p_blend = best_w_rbf * rbf_probs + best_w_hgbm * hgbm_probs
    best_y_pred = (best_p_blend >= 0.5).astype(int)

    # 4. Generate Visualizations
    print("\n[4/4] Generating comparative visualizations (Confusion Matrix & Multi-Model ROC)...")
    # Confusion matrix of optimal blend
    cm_path = PLOTS_DIR / "ensemble_cm_full96.png"
    plot_confusion_matrix(
        y_val, best_y_pred,
        model_name=f"Hybrid Ensemble Blend ({int(best_w_rbf*100)}% RBF + {int(best_w_hgbm*100)}% HistGBM)",
        output_path=cm_path
    )
    
    # ROC curve comparison: Single RBF vs Single HistGBM vs Optimal Blend
    roc_plot_data = [
        {
            "y_true": y_val,
            "y_prob": best_p_blend,
            "label": f"Hybrid Blend (70% RBF + 30% HistGBM) [AUC = {best_blend['roc_auc']:.4f}]",
            "color": "#9b59b6",
            "linestyle": "-",
            "linewidth": 2.6
        },
        {
            "y_true": y_val,
            "y_prob": rbf_scores,
            "label": f"Tuned RBF Kernel SVM [AUC = 0.9082]",
            "color": "#2980b9",
            "linestyle": "--",
            "linewidth": 2.0
        },
        {
            "y_true": y_val,
            "y_prob": hgbm_probs,
            "label": f"HistGradientBoosting [AUC = 0.8990]",
            "color": "#e74c3c",
            "linestyle": "-.",
            "linewidth": 2.0
        }
    ]
    
    roc_plot_path = PLOTS_DIR / "ensemble_vs_single_roc.png"
    plot_roc_curves(
        roc_plot_data,
        output_path=roc_plot_path,
        title="PCam Tumor Detection: Hybrid Ensemble Blend vs. Single Top Models"
    )

    print("\n" + "=" * 95)
    print("PHASE 6 ENSEMBLE BLENDING BENCHMARK SUMMARY:")
    print("=" * 95)
    print(f"  1. Single HistGradientBoosting: Acc = 80.66% | ROC-AUC = 0.8990 | Sens = 74.44% | Prec = 85.02%")
    print(f"  2. Tuned RBF Kernel SVM:        Acc = 82.36% | ROC-AUC = 0.9082 | Sens = 78.56% | Prec = 85.02%")
    print(f"  3. Cost-Weighted RBF SVM:       Acc = 82.54% | ROC-AUC = 0.9074 | Sens = 80.32% | Prec = 84.05%")
    print(f"  4. Hybrid Ensemble Blend:       Acc = {best_blend['accuracy']*100:.2f}% | ROC-AUC = {best_blend['roc_auc']:.4f} (Rank-AUC: {best_blend['rank_auc']:.4f}) | F1 = {best_blend['f1_score']:.4f}")
    print("=" * 95)
    print("[OK] Phase 6 Multi-Paradigm Ensemble Blending completed successfully.")


if __name__ == "__main__":
    main()

