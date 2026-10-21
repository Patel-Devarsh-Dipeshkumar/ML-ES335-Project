"""
Phase 4: Logistic Regression Baseline & Ablation Benchmark.
Course Project: ES 335 (IIT Gandhinagar)

Trains and evaluates Logistic Regression models via Maximum Likelihood Estimation (MLE)
with L-BFGS solver across:
  - Track A (Full 96x96 Context)
  - Track B (Crop 32x32 Core)
Across 3 feature representations per track (Ablation Benchmark):
  1. Fused (All 59 features: Handcrafted + PCA)
  2. Handcrafted / Biological Only (39 features)
  3. PCA Only (20 features)

Generates:
  - outputs/results/baseline_metrics.csv
  - outputs/plots/baseline_roc_curves.png
  - outputs/plots/baseline_cm_full96.png
  - outputs/plots/baseline_cm_crop32.png
  - outputs/models/logreg_*.pkl
"""

import os
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Set up project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import (
    evaluate_model,
    plot_roc_curves,
    plot_confusion_matrix,
    save_metrics_table
)

FEATURES_DIR = PROJECT_ROOT / "outputs" / "features"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def get_feature_subsets(all_features):
    """Partition feature columns into subsets for ablation."""
    pca_cols = [c for c in all_features if c.startswith("pca_")]
    hc_cols = [c for c in all_features if not c.startswith("pca_")]
    return {
        "Fused (59)": all_features,
        "Biological Only (39)": hc_cols,
        "PCA Only (20)": pca_cols
    }


def inspect_learned_weights(model, feature_names, track_name, top_n=5):
    """Display top positive (pro-tumor) and top negative (pro-normal) weights."""
    coefs = model.coef_[0]
    weight_df = pd.DataFrame({
        "feature": feature_names,
        "weight": coefs
    }).sort_values(by="weight", ascending=False)
    
    print(f"\n  Top {top_n} PRO-TUMOR Features (Highest Positive Weights - {track_name}):")
    for _, r in weight_df.head(top_n).iterrows():
        print(f"    + {r['feature']:<22}: w = +{r['weight']:.4f}")
        
    print(f"\n  Top {top_n} PRO-NORMAL Features (Most Negative Weights - {track_name}):")
    for _, r in weight_df.tail(top_n).iloc[::-1].iterrows():
        print(f"    - {r['feature']:<22}: w = {r['weight']:.4f}")
        
    return weight_df


def main():
    print("=" * 80)
    print("PHASE 4: LOGISTIC REGRESSION BASELINE & ABLATION STUDY")
    print("=" * 80)
    
    # 1. Load Data
    print("\n[1/4] Loading standardized feature CSVs from outputs/features/...")
    train_full = pd.read_csv(FEATURES_DIR / "full96_train.csv")
    val_full = pd.read_csv(FEATURES_DIR / "full96_val.csv")
    train_crop = pd.read_csv(FEATURES_DIR / "crop32_train.csv")
    val_crop = pd.read_csv(FEATURES_DIR / "crop32_val.csv")
    
    print(f"  Track A (Full 96): Train={train_full.shape}, Val={val_full.shape}")
    print(f"  Track B (Crop 32): Train={train_crop.shape}, Val={val_crop.shape}")
    
    tracks_config = [
        ("Track A (Full 96)", train_full, val_full, "full96", "#2980b9"),
        ("Track B (Crop 32)", train_crop, val_crop, "crop32", "#e67e22")
    ]
    
    metrics_records = []
    roc_plot_data = []
    trained_models = {}
    
    # 2. Train & Evaluate Models
    print("\n[2/4] Training Logistic Regression models via Maximum Likelihood Estimation (MLE)...")
    
    for track_display, df_train, df_val, track_key, track_color in tracks_config:
        all_features = [c for c in df_train.columns if c != "label"]
        y_train = df_train["label"].values.astype(int)
        y_val = df_val["label"].values.astype(int)
        
        subsets = get_feature_subsets(all_features)
        
        for subset_name, feat_cols in subsets.items():
            model_id = f"LogReg_{track_key}_{subset_name.split()[0].lower()}"
            display_name = f"{track_display} - {subset_name}"
            
            X_train = df_train[feat_cols].values
            X_val = df_val[feat_cols].values
            
            t0 = time.time()
            clf = LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                C=1.0,
                random_state=42
            )
            clf.fit(X_train, y_train)
            train_time = time.time() - t0
            
            # Evaluate on validation split
            metrics, y_pred, y_prob = evaluate_model(
                clf, X_val, y_val,
                model_name=display_name,
                dataset_name="Validation (5,000)"
            )
            metrics["train_time_sec"] = round(train_time, 3)
            metrics["feature_count"] = len(feat_cols)
            metrics_records.append(metrics)
            
            # Save trained model
            model_save_path = MODELS_DIR / f"{model_id}.pkl"
            joblib.dump(clf, model_save_path)
            
            # Store ROC curve data
            linestyle = "-" if "Fused" in subset_name else ("--" if "Biological" in subset_name else ":")
            linewidth = 2.4 if "Fused" in subset_name else 1.8
            roc_plot_data.append({
                "y_true": y_val,
                "y_prob": y_prob,
                "label": f"{display_name} (AUC = {metrics['roc_auc']:.4f})",
                "color": track_color if "Fused" in subset_name else ("#27ae60" if "Biological" in subset_name else "#8e44ad"),
                "linestyle": linestyle,
                "linewidth": linewidth
            })
            
            # Print status line
            print(f"  [Trained] {display_name:<36} | Time: {train_time:.2f}s | Acc: {metrics['accuracy']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1_score']:.4f}")
            
            # If Fused, generate confusion matrix and inspect weights
            if "Fused" in subset_name:
                trained_models[track_key] = (clf, feat_cols, y_val, y_pred)
                cm_path = PLOTS_DIR / f"baseline_cm_{track_key}.png"
                plot_confusion_matrix(y_val, y_pred, model_name=display_name, output_path=cm_path)
                inspect_learned_weights(clf, feat_cols, track_display)

    # 3. Save Summary Table
    print("\n[3/4] Saving results to outputs/results/baseline_metrics.csv...")
    results_csv_path = RESULTS_DIR / "baseline_metrics.csv"
    save_metrics_table(metrics_records, results_csv_path)

    # 4. Generate Comparative ROC Plot
    print("\n[4/4] Generating publication-grade comparative ROC curves...")
    roc_plot_path = PLOTS_DIR / "baseline_roc_curves.png"
    plot_roc_curves(
        roc_plot_data,
        output_path=roc_plot_path,
        title="Baseline Logistic Regression ROC Curves: Track A vs. Track B Ablation"
    )

    # Print Final Benchmark Leaderboard
    print("\n" + "=" * 80)
    print("PHASE 4 BASELINE LEADERBOARD (Validation Set: 5,000 Patches):")
    print("=" * 80)
    df_res = pd.DataFrame(metrics_records)[
        ["model", "feature_count", "accuracy", "roc_auc", "recall_sensitivity", "precision", "f1_score", "train_time_sec"]
    ].sort_values(by="roc_auc", ascending=False)
    
    print(df_res.to_string(index=False))
    print("=" * 80)
    print("[OK] Phase 4 Baseline Model execution completed successfully.")


if __name__ == "__main__":
    main()

