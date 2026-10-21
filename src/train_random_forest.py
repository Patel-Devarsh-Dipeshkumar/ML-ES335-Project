"""
Random Forest Candidate Model (Bagging Ensemble) on Full 40,000 Training Patches.
Course Project: ES 335 (IIT Gandhinagar)

Trains an ensemble of 200 de-correlated decision trees:
  - Bootstrap aggregation (Bagging) across 40,000 training patches
  - Random feature subsetting (max_features='sqrt' ~ 8 features per split)
  - Anti-overfitting regularization (min_samples_leaf=4)
  - Parallel execution across all 14 CPU cores (n_jobs=-1)

Evaluates across:
  - Track A (Full 96x96 Context, 59 features)
  - Track B (Crop 32x32 Core, 59 features)
On the identical 5,000 unseen validation patches.

Compares against Logistic Regression, Linear SVM, and Non-Linear RBF Kernel SVM.
"""

import os
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

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
    print("PHASE 5: CANDIDATE MODEL 3 --- RANDOM FOREST (BAGGING ENSEMBLE)")
    print("=" * 85)
    
    # 1. Load Data
    print("\n[1/4] Loading standardized features from outputs/features/...")
    train_full = pd.read_csv(FEATURES_DIR / "full96_train.csv")
    val_full = pd.read_csv(FEATURES_DIR / "full96_val.csv")
    train_crop = pd.read_csv(FEATURES_DIR / "crop32_train.csv")
    val_crop = pd.read_csv(FEATURES_DIR / "crop32_val.csv")
    
    feature_cols = [c for c in train_full.columns if c != "label"]
    
    tracks = [
        ("Track A (Full 96)", train_full, val_full, "full96", "#27ae60"),
        ("Track B (Crop 32)", train_crop, val_crop, "crop32", "#d35400")
    ]
    
    rf_results = []
    roc_plot_data = []
    
    # 2. Train Random Forest on Track A and Track B
    print("\n[2/4] Training Random Forest (200 trees, max_features='sqrt', min_samples_leaf=4, 14 cores)...")
    
    for track_display, df_tr, df_v, track_key, color in tracks:
        X_train = df_tr[feature_cols].values
        y_train = df_tr["label"].values.astype(int)
        X_val = df_v[feature_cols].values
        y_val = df_v["label"].values.astype(int)
        
        print(f"\n  Growing 200 trees on {track_display} (N=40,000, 59 features)...")
        t0 = time.time()
        clf = RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            min_samples_leaf=4,
            max_depth=18,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        fit_time = time.time() - t0
        
        print(f"  -> Converged in {fit_time:.2f}s across all CPU cores.")
        
        # Save model
        model_save_path = MODELS_DIR / f"RandomForest_{track_key}.pkl"
        joblib.dump(clf, model_save_path)
        
        # Evaluate
        model_name = f"Random Forest ({track_display})"
        metrics, y_pred, y_prob = evaluate_model(
            clf, X_val, y_val,
            model_name=model_name,
            dataset_name="Validation (5,000)"
        )
        metrics["train_time_sec"] = round(fit_time, 2)
        metrics["feature_count"] = len(feature_cols)
        rf_results.append(metrics)
        
        # Confusion matrix
        cm_path = PLOTS_DIR / f"rf_cm_{track_key}.png"
        plot_confusion_matrix(y_val, y_pred, model_name=model_name, output_path=cm_path)
        
        # Store ROC curve
        roc_plot_data.append({
            "y_true": y_val,
            "y_prob": y_prob,
            "label": f"Random Forest - {track_display} (AUC = {metrics['roc_auc']:.4f})",
            "color": color,
            "linestyle": "-",
            "linewidth": 2.4
        })
        
        print(f"  [Trained] {model_name:<32} | Acc: {metrics['accuracy']*100:.2f}% | ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1_score']:.4f}")

    # 3. Load Existing Benchmark Models (LogReg, Linear SVM, RBF SVM)
    print("\n[3/4] Building 4-Way Comparative Leaderboard across all candidate models...")
    cand_csv_path = RESULTS_DIR / "candidate_models_benchmark.csv"
    df_existing = pd.read_csv(cand_csv_path) if cand_csv_path.exists() else pd.DataFrame()
    
    # Overwrite or append RF results
    all_records = df_existing.to_dict(orient="records") if not df_existing.empty else []
    # Avoid duplicate RF rows if re-run
    all_records = [r for r in all_records if "Random Forest" not in r["model"]] + rf_results
    df_all_candidates = pd.DataFrame(all_records)
    
    # Save unified candidate benchmark table
    df_all_candidates.to_csv(cand_csv_path, index=False)
    print(f"[OK] Candidate benchmark table saved to: {cand_csv_path}")

    # Load prior models for complete multi-model ROC plot
    val_X_full = val_full[feature_cols].values
    val_y_full = val_full["label"].values
    
    # RBF SVM Track A
    rbf_model = joblib.load(MODELS_DIR / "RBFSVM_full96.pkl")
    probs_rbf = rbf_model.decision_function(val_X_full)
    roc_plot_data.append({
        "y_true": val_y_full,
        "y_prob": probs_rbf,
        "label": f"RBF Kernel SVM - Track A (AUC = 0.9038)",
        "color": "#2980b9",
        "linestyle": "--",
        "linewidth": 2.0
    })
    
    # Linear SVM Track A
    lin_model = joblib.load(MODELS_DIR / "LinearSVM_full96.pkl")
    probs_lin = lin_model.decision_function(val_X_full)
    roc_plot_data.append({
        "y_true": val_y_full,
        "y_prob": probs_lin,
        "label": f"Linear SVM - Track A (AUC = 0.8684)",
        "color": "#8e44ad",
        "linestyle": ":",
        "linewidth": 1.7
    })
    
    # Logistic Regression Track A
    lr_model = joblib.load(MODELS_DIR / "LogReg_full96_fused.pkl")
    probs_lr = lr_model.predict_proba(val_X_full)[:, 1]
    roc_plot_data.append({
        "y_true": val_y_full,
        "y_prob": probs_lr,
        "label": f"LogReg Baseline - Track A (AUC = 0.8652)",
        "color": "#7f8c8d",
        "linestyle": "-.",
        "linewidth": 1.7
    })

    # 4. Generate Publication-Grade Multi-Model ROC Plot
    print("\n[4/4] Generating comparative ROC curves across all candidate architectures...")
    roc_plot_path = PLOTS_DIR / "candidate_models_roc.png"
    plot_roc_curves(
        roc_plot_data,
        output_path=roc_plot_path,
        title="Candidate Models Multi-Architecture ROC Curves (N=40,000)"
    )

    # Print Final Benchmark Leaderboard
    print("\n" + "=" * 95)
    print("OFFICIAL CANDIDATE MODELS BENCHMARK LEADERBOARD (Validation Set: 5,000 Patches):")
    print("=" * 95)
    disp_cols = ["model", "accuracy", "roc_auc", "recall_sensitivity", "precision", "f1_score", "train_time_sec"]
    fmt_df = df_all_candidates[disp_cols].copy()
    fmt_df["accuracy"] = (fmt_df["accuracy"] * 100).round(2).astype(str) + "%"
    fmt_df["recall_sensitivity"] = (fmt_df["recall_sensitivity"] * 100).round(2).astype(str) + "%"
    fmt_df["precision"] = (fmt_df["precision"] * 100).round(2).astype(str) + "%"
    fmt_df["roc_auc"] = fmt_df["roc_auc"].round(4)
    fmt_df["f1_score"] = fmt_df["f1_score"].round(4)
    fmt_df["train_time_sec"] = fmt_df["train_time_sec"].astype(str) + "s"
    
    print(fmt_df.sort_values(by="roc_auc", ascending=False).to_string(index=False))
    print("=" * 95)
    print("[OK] Random Forest training and benchmark evaluation completed successfully.")


if __name__ == "__main__":
    main()

