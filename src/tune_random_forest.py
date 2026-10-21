"""
Random Forest Hyperparameter Tuning & Performance Optimization.
Course Project: ES 335 (IIT Gandhinagar)

Addresses the 3 key bottlenecks of default Random Forest:
  1. max_features: Expands feature sampling from 'sqrt' (~8 feats) to 0.25, 0.40, and 0.60
     to ensure top biological markers (lbp_bin_5, color_R_std, edge_sobel_mean) are chosen.
  2. min_samples_leaf: Tests 1, 2, and 4 to allow finer cellular boundary resolution.
  3. criterion: Compares Gini impurity vs. Shannon Entropy (Information Gain).
  4. class_weight: Tests 'balanced' to directly resolve the low sensitivity (71.1%) bottleneck.
  5. max_depth: Unconstrained vs. depth 24.

Evaluates across 40,000 training patches on 14 CPU cores on the 5,000 validation split.
"""

import os
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import evaluate_model, plot_confusion_matrix

FEATURES_DIR = PROJECT_ROOT / "outputs" / "features"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"


def main():
    print("=" * 90)
    print("PHASE 5: RANDOM FOREST HYPERPARAMETER TUNING & OPTIMIZATION")
    print("=" * 90)
    
    # 1. Load Data
    print("\n[1/4] Loading Track A standardized features (40k Train, 5k Val)...")
    train_full = pd.read_csv(FEATURES_DIR / "full96_train.csv")
    val_full = pd.read_csv(FEATURES_DIR / "full96_val.csv")
    
    feature_cols = [c for c in train_full.columns if c != "label"]
    X_train = train_full[feature_cols].values
    y_train = train_full["label"].values.astype(int)
    X_val = val_full[feature_cols].values
    y_val = val_full["label"].values.astype(int)
    
    print(f"  Feature count: {len(feature_cols)} | Train N={len(X_train):,}, Val N={len(X_val):,}")
    
    # 2. Targeted Tuning Configurations
    # We systematically test combinations targeting the feature bottleneck and sensitivity
    configs = [
        # (ID, n_trees, max_feat, min_leaf, depth, criterion, class_weight)
        ("CFG_0_Default", 200, "sqrt", 4, 18, "gini", None),
        ("CFG_1_MoreFeatures", 250, 0.25, 4, 22, "gini", None),
        ("CFG_2_DeeperTrees", 250, 0.30, 2, 25, "gini", None),
        ("CFG_3_FullResolution", 250, 0.35, 1, None, "gini", None),
        ("CFG_4_Entropy_Split", 250, 0.35, 2, 25, "entropy", None),
        ("CFG_5_Balanced_Sensitivity", 250, 0.30, 2, 25, "gini", "balanced"),
        ("CFG_6_BroadFeatures", 250, 0.45, 2, 25, "gini", None),
        ("CFG_7_Deep_Balanced", 250, 0.35, 1, None, "entropy", "balanced"),
    ]
    
    results = []
    best_clf = None
    best_acc = 0.0
    best_cfg_name = ""
    
    print("\n[2/4] Executing Tuning Grid on 14 CPU cores...")
    print("-" * 90)
    print(f"{'Config ID':<24} {'Feat Frac':<10} {'Leaf':<6} {'Depth':<8} {'Criterion':<10} {'Val Acc':<10} {'ROC-AUC':<10} {'Recall':<9} {'F1':<8} {'Time'}")
    print("-" * 90)
    
    for cfg_id, n_trees, max_f, min_leaf, depth, crit, cw in configs:
        t0 = time.time()
        clf = RandomForestClassifier(
            n_estimators=n_trees,
            max_features=max_f,
            min_samples_leaf=min_leaf,
            max_depth=depth,
            criterion=crit,
            class_weight=cw,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        fit_time = time.time() - t0
        
        # Evaluate
        metrics, y_pred, y_prob = evaluate_model(
            clf, X_val, y_val,
            model_name=f"RF_{cfg_id}",
            dataset_name="Validation (5,000)"
        )
        
        feat_str = str(max_f)
        depth_str = str(depth) if depth is not None else "None"
        acc = metrics["accuracy"]
        auc = metrics["roc_auc"]
        rec = metrics["recall_sensitivity"]
        f1 = metrics["f1_score"]
        
        results.append({
            "config_id": cfg_id,
            "n_estimators": n_trees,
            "max_features": feat_str,
            "min_samples_leaf": min_leaf,
            "max_depth": depth_str,
            "criterion": crit,
            "class_weight": str(cw),
            "val_accuracy": round(acc * 100, 2),
            "val_roc_auc": round(auc, 4),
            "val_recall": round(rec * 100, 2),
            "val_precision": round(metrics["precision"] * 100, 2),
            "val_f1": round(f1, 4),
            "fit_time_sec": round(fit_time, 2)
        })
        
        print(f"{cfg_id:<24} {feat_str:<10} {min_leaf:<6} {depth_str:<8} {crit:<10} {acc*100:<9.2f}% {auc:<10.4f} {rec*100:<8.2f}% {f1:<8.4f} {fit_time:.1f}s")
        
        # Track winning model
        if acc > best_acc:
            best_acc = acc
            best_clf = clf
            best_cfg_name = cfg_id
            
    # 3. Save Tuning CSV
    print("\n[3/4] Saving tuning comparison table...")
    df_tune = pd.DataFrame(results)
    tuning_csv_path = RESULTS_DIR / "random_forest_tuning_grid.csv"
    df_tune.to_csv(tuning_csv_path, index=False)
    print(f"[OK] Saved tuning grid table to: {tuning_csv_path}")
    
    # 4. Save and Update Best Tuned Model
    print(f"\n[4/4] Best Configuration Found: {best_cfg_name} (Val Accuracy: {best_acc*100:.2f}%)")
    best_model_path = MODELS_DIR / "RandomForest_full96_tuned.pkl"
    joblib.dump(best_clf, best_model_path)
    print(f"[OK] Saved tuned Random Forest model to: {best_model_path}")
    
    # Confusion matrix for tuned model
    cm_path = PLOTS_DIR / "rf_tuned_cm_full96.png"
    _, y_pred_best, _ = evaluate_model(best_clf, X_val, y_val, model_name=f"Tuned Random Forest ({best_cfg_name})")
    plot_confusion_matrix(y_val, y_pred_best, model_name=f"Tuned Random Forest ({best_cfg_name})", output_path=cm_path)
    
    # Update candidate benchmark table with Tuned RF
    cand_csv_path = RESULTS_DIR / "candidate_models_benchmark.csv"
    if cand_csv_path.exists():
        df_cand = pd.read_csv(cand_csv_path)
        # Add tuned model row
        best_row = df_tune[df_tune["config_id"] == best_cfg_name].iloc[0]
        tuned_record = {
            "model": "Tuned Random Forest (Track A)",
            "accuracy": best_row["val_accuracy"] / 100.0,
            "roc_auc": best_row["val_roc_auc"],
            "recall_sensitivity": best_row["val_recall"] / 100.0,
            "precision": best_row["val_precision"] / 100.0,
            "f1_score": best_row["val_f1"],
            "train_time_sec": best_row["fit_time_sec"],
            "feature_count": len(feature_cols)
        }
        # Avoid duplicate
        cand_records = [r for r in df_cand.to_dict(orient="records") if r["model"] != "Tuned Random Forest (Track A)"]
        cand_records.append(tuned_record)
        pd.DataFrame(cand_records).to_csv(cand_csv_path, index=False)
        print(f"[OK] Updated candidate benchmark table with Tuned Random Forest.")
        
    # Print Comparison Table
    print("\n" + "=" * 90)
    print("TUNING GAINS SUMMARY (Default RF vs. Tuned RF):")
    print("=" * 90)
    def_row = df_tune[df_tune["config_id"] == "CFG_0_Default"].iloc[0]
    win_row = df_tune[df_tune["config_id"] == best_cfg_name].iloc[0]
    
    print(f"  Default Setting (max_features='sqrt', leaf=4) : Acc = {def_row['val_accuracy']}% | ROC-AUC = {def_row['val_roc_auc']} | Recall = {def_row['val_recall']}%")
    print(f"  Tuned Setting   ({best_cfg_name})             : Acc = {win_row['val_accuracy']}% | ROC-AUC = {win_row['val_roc_auc']} | Recall = {win_row['val_recall']}%")
    delta_acc = win_row['val_accuracy'] - def_row['val_accuracy']
    delta_auc = win_row['val_roc_auc'] - def_row['val_roc_auc']
    delta_rec = win_row['val_recall'] - def_row['val_recall']
    print(f"  Net Tuning Improvement                        : Accuracy: {delta_acc:+.2f}% | ROC-AUC: {delta_auc:+.4f} | Recall: {delta_rec:+.2f}%")
    print("=" * 90)


if __name__ == "__main__":
    main()

