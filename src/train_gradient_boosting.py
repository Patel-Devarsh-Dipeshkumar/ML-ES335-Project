"""
HistGradientBoosting Candidate Model (Sequential Boosting Ensemble) on Full 40,000 Training Patches.
Course Project: ES 335 (IIT Gandhinagar)

Trains a histogram-based gradient boosted decision tree ensemble:
  - Sequential gradient boosting optimizing binary cross-entropy (log_loss)
  - Histogram binning (256 bins) for fast split finding and implicit noise regularization
  - Tree-depth and leaf constraints: max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20
  - L2 leaf weight regularization (l2_regularization=1.0) to prevent overfitting
  - Early stopping with internal validation monitoring

Evaluates across:
  - Track A (Full 96x96 Context, 59 features)
  - Track B (Crop 32x32 Core, 59 features)
On the identical 5,000 unseen validation patches.

Compares against Logistic Regression, Linear SVM, Non-Linear RBF Kernel SVM, and Random Forest.
"""

import os
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

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
    print("PHASE 5: CANDIDATE MODEL 4 --- HIST GRADIENT BOOSTING (SEQUENTIAL BOOSTING ENSEMBLE)")
    print("=" * 85)
    
    # 1. Load Data
    print("\n[1/5] Loading standardized features from outputs/features/...")
    train_full = pd.read_csv(FEATURES_DIR / "full96_train.csv")
    val_full = pd.read_csv(FEATURES_DIR / "full96_val.csv")
    train_crop = pd.read_csv(FEATURES_DIR / "crop32_train.csv")
    val_crop = pd.read_csv(FEATURES_DIR / "crop32_val.csv")
    
    feature_cols = [c for c in train_full.columns if c != "label"]
    
    tracks = [
        ("Track A (Full 96)", train_full, val_full, "full96", "#e74c3c"),
        ("Track B (Crop 32)", train_crop, val_crop, "crop32", "#e67e22")
    ]
    
    hgbm_results = []
    roc_plot_data = []
    
    # 2. Train HistGradientBoosting on Track A and Track B
    print("\n[2/5] Training HistGradientBoosting (300 stages, lr=0.05, max_leaves=31, min_leaf=20, l2=1.0)...")
    
    for track_display, df_tr, df_v, track_key, color in tracks:
        X_train = df_tr[feature_cols].values
        y_train = df_tr["label"].values.astype(int)
        X_val = df_v[feature_cols].values
        y_val = df_v["label"].values.astype(int)
        
        print(f"\n  Boosting 300 sequential trees on {track_display} (N=40,000, 59 features)...")
        t0 = time.time()
        clf = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=True,
            n_iter_no_change=15,
            validation_fraction=0.1,
            random_state=42
        )
        clf.fit(X_train, y_train)
        fit_time = time.time() - t0
        
        print(f"  -> Converged in {fit_time:.2f}s over {clf.n_iter_} boosting iterations.")
        
        # Save model
        model_save_path = MODELS_DIR / f"HistGBM_{track_key}.pkl"
        joblib.dump(clf, model_save_path)
        
        # Evaluate
        model_name = f"HistGradientBoosting ({track_display})"
        metrics, y_pred, y_prob = evaluate_model(
            clf, X_val, y_val,
            model_name=model_name,
            dataset_name="Validation (5,000)"
        )
        metrics["train_time_sec"] = round(fit_time, 2)
        metrics["feature_count"] = len(feature_cols)
        hgbm_results.append(metrics)
        
        # Confusion matrix
        cm_path = PLOTS_DIR / f"hgbm_cm_{track_key}.png"
        plot_confusion_matrix(y_val, y_pred, model_name=model_name, output_path=cm_path)
        
        # Store ROC curve
        roc_plot_data.append({
            "y_true": y_val,
            "y_prob": y_prob,
            "label": f"HistGBM - {track_display} (AUC = {metrics['roc_auc']:.4f})",
            "color": color,
            "linestyle": "-",
            "linewidth": 2.5
        })
        
        print(f"  [Trained] {model_name:<36} | Acc: {metrics['accuracy']*100:.2f}% | ROC-AUC: {metrics['roc_auc']:.4f} | Sens: {metrics['recall_sensitivity']*100:.2f}% | Prec: {metrics['precision']*100:.2f}%")

    # 3. Permutation Feature Importance for Track A
    print("\n[3/5] Computing Permutation Feature Importance for HistGradientBoosting (Track A)...")
    clf_full = joblib.load(MODELS_DIR / "HistGBM_full96.pkl")
    X_val_full = val_full[feature_cols].values
    y_val_full = val_full["label"].values.astype(int)
    
    # Evaluate on 2,500 validation patches for quick, robust estimation
    perm_result = permutation_importance(
        clf_full, X_val_full[:2500], y_val_full[:2500],
        n_repeats=5, random_state=42, n_jobs=-1
    )
    
    df_perm = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": perm_result.importances_mean,
        "importance_std": perm_result.importances_std
    }).sort_values(by="importance_mean", ascending=False)
    
    perm_csv_path = RESULTS_DIR / "histgbm_feature_importance.csv"
    df_perm.to_csv(perm_csv_path, index=False)
    print(f"[OK] Permutation feature importances saved to: {perm_csv_path}")
    print("\n  Top 10 Most Influential Features in HistGradientBoosting:")
    for i, row in df_perm.head(10).reset_index(drop=True).iterrows():
        print(f"    {i+1:2d}. {row['feature']:<25} Mean Importance Drop: {row['importance_mean']:.4f} (+/- {row['importance_std']:.4f})")

    # 4. Update Unified Candidate Models Benchmark Table
    print("\n[4/5] Updating Unified Candidate Models Benchmark Table...")
    cand_csv_path = RESULTS_DIR / "candidate_models_benchmark.csv"
    df_existing = pd.read_csv(cand_csv_path) if cand_csv_path.exists() else pd.DataFrame()
    
    all_records = df_existing.to_dict(orient="records") if not df_existing.empty else []
    # Avoid duplicate HGBM rows if re-run
    all_records = [r for r in all_records if "HistGradientBoosting" not in r["model"]] + hgbm_results
    df_all_candidates = pd.DataFrame(all_records)
    df_all_candidates.to_csv(cand_csv_path, index=False)
    print(f"[OK] Candidate benchmark table updated: {cand_csv_path}")

    # 5. Comparative Multi-Architecture ROC Curves Plot
    print("\n[5/5] Generating Comprehensive 5-Architecture ROC Curves Plot...")
    
    # Load comparison models for Track A
    # Random Forest Track A
    if (MODELS_DIR / "RandomForest_full96.pkl").exists():
        rf_model = joblib.load(MODELS_DIR / "RandomForest_full96.pkl")
        probs_rf = rf_model.predict_proba(X_val_full)[:, 1]
        roc_plot_data.append({
            "y_true": y_val_full,
            "y_prob": probs_rf,
            "label": f"Random Forest - Track A (AUC = 0.8878)",
            "color": "#27ae60",
            "linestyle": "--",
            "linewidth": 2.0
        })
    
    # RBF Kernel SVM Track A
    if (MODELS_DIR / "RBFSVM_full96.pkl").exists():
        rbf_model = joblib.load(MODELS_DIR / "RBFSVM_full96.pkl")
        probs_rbf = rbf_model.decision_function(X_val_full)
        roc_plot_data.append({
            "y_true": y_val_full,
            "y_prob": probs_rbf,
            "label": f"RBF Kernel SVM - Track A (AUC = 0.9038)",
            "color": "#2980b9",
            "linestyle": "-",
            "linewidth": 2.2
        })
    
    # Linear SVM Track A
    if (MODELS_DIR / "LinearSVM_full96.pkl").exists():
        lin_model = joblib.load(MODELS_DIR / "LinearSVM_full96.pkl")
        probs_lin = lin_model.decision_function(X_val_full)
        roc_plot_data.append({
            "y_true": y_val_full,
            "y_prob": probs_lin,
            "label": f"Linear SVM - Track A (AUC = 0.8684)",
            "color": "#8e44ad",
            "linestyle": ":",
            "linewidth": 1.8
        })
    
    # Logistic Regression Track A
    if (MODELS_DIR / "LogReg_full96_fused.pkl").exists():
        lr_model = joblib.load(MODELS_DIR / "LogReg_full96_fused.pkl")
        probs_lr = lr_model.predict_proba(X_val_full)[:, 1]
        roc_plot_data.append({
            "y_true": y_val_full,
            "y_prob": probs_lr,
            "label": f"LogReg Baseline - Track A (AUC = 0.8652)",
            "color": "#7f8c8d",
            "linestyle": "-.",
            "linewidth": 1.8
        })

    roc_plot_path = PLOTS_DIR / "candidate_models_roc.png"
    plot_roc_curves(
        roc_plot_data,
        output_path=roc_plot_path,
        title="PCam Tumor Detection: Candidate Model Ladder Multi-Architecture ROC Curves"
    )

    # Print Official Benchmark Leaderboard
    print("\n" + "=" * 105)
    print("OFFICIAL CANDIDATE MODELS BENCHMARK LEADERBOARD (Validation Set: 5,000 Patches):")
    print("=" * 105)
    disp_cols = ["model", "accuracy", "roc_auc", "recall_sensitivity", "precision", "f1_score", "train_time_sec"]
    fmt_df = df_all_candidates[disp_cols].copy()
    fmt_df["accuracy"] = (fmt_df["accuracy"] * 100).round(2).astype(str) + "%"
    fmt_df["recall_sensitivity"] = (fmt_df["recall_sensitivity"] * 100).round(2).astype(str) + "%"
    fmt_df["precision"] = (fmt_df["precision"] * 100).round(2).astype(str) + "%"
    fmt_df["roc_auc"] = fmt_df["roc_auc"].round(4)
    fmt_df["f1_score"] = fmt_df["f1_score"].round(4)
    fmt_df["train_time_sec"] = fmt_df["train_time_sec"].astype(str) + "s"
    
    print(fmt_df.sort_values(by="roc_auc", ascending=False).to_string(index=False))
    print("=" * 105)
    print("[OK] Candidate Model 4 (HistGradientBoosting) training and benchmarking completed successfully.")


if __name__ == "__main__":
    main()

