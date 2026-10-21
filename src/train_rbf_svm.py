"""
Non-Linear RBF Kernel SVM on Full 40,000 Training Patches.
Course Project: ES 335 (IIT Gandhinagar)

Trains Support Vector Classifier with Gaussian Radial Basis Function (RBF) kernel:
  K(x, x') = exp(-gamma * ||x - x'||^2)
On ALL 40,000 training patches with a 2GB kernel RAM cache (cache_size=2000).

Evaluates on 5,000 unseen validation patches across:
  - Track A (Full 96x96 Context, 59 features)
  - Track B (Crop 32x32 Core, 59 features)

Compares directly against Linear SVM and Logistic Regression.
"""

import os
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import evaluate_model, plot_roc_curves, plot_confusion_matrix

FEATURES_DIR = PROJECT_ROOT / "outputs" / "features"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"


def main():
    print("=" * 85)
    print("PHASE 5: CANDIDATE MODEL 2 --- NON-LINEAR RBF KERNEL SVM (FULL 40,000 PATCHES)")
    print("=" * 85)
    
    # 1. Load Data
    print("\n[1/4] Loading standardized features from outputs/features/...")
    train_full = pd.read_csv(FEATURES_DIR / "full96_train.csv")
    val_full = pd.read_csv(FEATURES_DIR / "full96_val.csv")
    train_crop = pd.read_csv(FEATURES_DIR / "crop32_train.csv")
    val_crop = pd.read_csv(FEATURES_DIR / "crop32_val.csv")
    
    feature_cols = [c for c in train_full.columns if c != "label"]
    
    print(f"  Track A (Full 96): Train={train_full.shape}, Val={val_full.shape}")
    print(f"  Track B (Crop 32): Train={train_crop.shape}, Val={val_crop.shape}")
    
    tracks = [
        ("Track A (Full 96)", train_full, val_full, "full96", "#2980b9"),
        ("Track B (Crop 32)", train_crop, val_crop, "crop32", "#e67e22")
    ]
    
    rbf_results = []
    roc_plot_data = []
    
    # 2. Train RBF SVM on Track A and Track B
    print("\n[2/4] Training Non-Linear RBF Kernel SVM on ALL 40,000 patches (C=1.0, gamma='scale', cache=2GB)...")
    
    for track_display, df_tr, df_v, track_key, color in tracks:
        X_train = df_tr[feature_cols].values
        y_train = df_tr["label"].values.astype(int)
        X_val = df_v[feature_cols].values
        y_val = df_v["label"].values.astype(int)
        
        print(f"\n  Fitting RBF Kernel on {track_display} (N=40,000, 59 features)...")
        t0 = time.time()
        clf = SVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            cache_size=2000,
            tol=1e-3,
            random_state=42
        )
        clf.fit(X_train, y_train)
        fit_time = time.time() - t0
        
        n_support = int(np.sum(clf.n_support_))
        pct_support = (n_support / len(X_train)) * 100
        print(f"  -> Converged in {fit_time:.2f}s ({fit_time/60:.2f} min). Support vectors: {n_support:,} ({pct_support:.1f}% of train)")
        
        # Save model
        model_save_path = MODELS_DIR / f"RBFSVM_{track_key}.pkl"
        joblib.dump(clf, model_save_path)
        
        # Evaluate
        model_name = f"RBF Kernel SVM ({track_display})"
        metrics, y_pred, y_prob = evaluate_model(
            clf, X_val, y_val,
            model_name=model_name,
            dataset_name="Validation (5,000)"
        )
        metrics["train_time_sec"] = round(fit_time, 2)
        metrics["feature_count"] = len(feature_cols)
        metrics["support_vectors"] = n_support
        rbf_results.append(metrics)
        
        # Confusion Matrix
        cm_path = PLOTS_DIR / f"rbf_svm_cm_{track_key}.png"
        plot_confusion_matrix(y_val, y_pred, model_name=model_name, output_path=cm_path)
        
        # ROC plot data
        roc_plot_data.append({
            "y_true": y_val,
            "y_prob": y_prob,
            "label": f"RBF SVM - {track_display} (AUC = {metrics['roc_auc']:.4f})",
            "color": color,
            "linestyle": "-",
            "linewidth": 2.4
        })
        
        print(f"  [Trained] {model_name:<32} | Acc: {metrics['accuracy']*100:.2f}% | ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1_score']:.4f}")

    # 3. Load Linear SVM & Logistic Regression to build 3-Way Benchmark
    print("\n[3/4] Building 3-Way Benchmark Comparison Table (LogReg vs. Linear SVM vs. RBF SVM)...")
    comp_csv_path = RESULTS_DIR / "linear_svm_vs_logreg_comparison.csv"
    df_prev = pd.read_csv(comp_csv_path)
    
    # Also load Linear SVM & LogReg probabilities for ROC plot overlay
    model_lin_a = joblib.load(MODELS_DIR / "LinearSVM_full96.pkl")
    probs_lin_a = model_lin_a.decision_function(val_full[feature_cols].values)
    roc_plot_data.append({
        "y_true": val_full["label"].values,
        "y_prob": probs_lin_a,
        "label": f"Linear SVM - Track A (AUC = 0.8684)",
        "color": "#8e44ad",
        "linestyle": "--",
        "linewidth": 1.8
    })
    
    model_lr_a = joblib.load(MODELS_DIR / "LogReg_full96_fused.pkl")
    probs_lr_a = model_lr_a.predict_proba(val_full[feature_cols].values)[:, 1]
    roc_plot_data.append({
        "y_true": val_full["label"].values,
        "y_prob": probs_lr_a,
        "label": f"LogReg Baseline - Track A (AUC = 0.8652)",
        "color": "#7f8c8d",
        "linestyle": ":",
        "linewidth": 1.8
    })

    # Combine records
    all_candidate_records = df_prev.to_dict(orient="records") + rbf_results
    df_all_cand = pd.DataFrame(all_candidate_records)
    
    # Save combined candidate metrics
    all_cand_path = RESULTS_DIR / "candidate_models_benchmark.csv"
    df_all_cand.to_csv(all_cand_path, index=False)
    print(f"[OK] Full candidate benchmark table saved to: {all_cand_path}")

    # 4. Plot 3-Way Comparative ROC Curves
    print("\n[4/4] Generating publication-grade comparative ROC curves...")
    roc_out_path = PLOTS_DIR / "rbf_vs_linear_svm_roc.png"
    plot_roc_curves(
        roc_plot_data,
        output_path=roc_out_path,
        title="Kernel Benchmark: Non-Linear RBF SVM vs. Linear Separators"
    )

    # Print Final Summary Leaderboard
    print("\n" + "=" * 95)
    print("3-WAY COMPARATIVE LEADERBOARD (LOGREG vs. LINEAR SVM vs. RBF KERNEL SVM):")
    print("=" * 95)
    disp_cols = ["model", "accuracy", "roc_auc", "recall_sensitivity", "precision", "f1_score", "train_time_sec"]
    fmt_df = df_all_cand[disp_cols].copy()
    fmt_df["accuracy"] = (fmt_df["accuracy"] * 100).round(2).astype(str) + "%"
    fmt_df["recall_sensitivity"] = (fmt_df["recall_sensitivity"] * 100).round(2).astype(str) + "%"
    fmt_df["precision"] = (fmt_df["precision"] * 100).round(2).astype(str) + "%"
    fmt_df["roc_auc"] = fmt_df["roc_auc"].round(4)
    fmt_df["f1_score"] = fmt_df["f1_score"].round(4)
    fmt_df["train_time_sec"] = fmt_df["train_time_sec"].astype(str) + "s"
    
    print(fmt_df.sort_values(by="roc_auc", ascending=False).to_string(index=False))
    print("=" * 95)
    print("[OK] Full-scale RBF Kernel SVM execution completed successfully.")


if __name__ == "__main__":
    main()

