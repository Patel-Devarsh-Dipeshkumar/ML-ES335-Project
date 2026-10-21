"""
Phase 7: Final Test Evaluation (ONE-SHOT) on Untouched 5,000 Test Patches.
Course Project: ES 335 (IIT Gandhinagar)

Evaluates the complete representative model portfolio on the held-out test split:
  1. Logistic Regression Baseline (Linear MLE)
  2. Linear Support Vector Machine (Maximum Margin)
  3. Tuned Random Forest (Bagging Ensemble)
  4. HistGradientBoosting (Sequential Boosting Ensemble)
  5. Tuned RBF Kernel SVM (Non-Linear RKHS Mapping)
  6. Hybrid Multi-Paradigm Ensemble Blend (RBF SVM + HistGBM)

Calculates:
  - Full diagnostic metrics (Accuracy, ROC-AUC, PR-AUC, Sensitivity, Specificity, Precision, F1)
  - 1,000-iteration Bootstrap 95% Confidence Intervals
  - Generalization Gap Analysis (Validation vs. Test performance)
  - Multi-model Test ROC and Precision-Recall Curves
  - Confusion Matrices for all top models

Exports:
  - outputs/results/final_test_benchmark.csv
  - outputs/results/test_bootstrap_confidence_intervals.csv
  - outputs/results/generalization_gap_analysis.csv
  - outputs/plots/test_roc_curves.png
  - outputs/plots/test_pr_curves.png
  - outputs/plots/test_cm_*.png
"""

import os
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    recall_score, precision_score, f1_score, balanced_accuracy_score,
    confusion_matrix, roc_curve, precision_recall_curve
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


def bootstrap_ci(y_true, y_prob, y_pred, n_bootstraps=1000, seed=42):
    """Computes 95% Bootstrap Confidence Intervals across core metrics."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    
    boot_acc = []
    boot_auc = []
    boot_sens = []
    boot_prec = []
    boot_f1 = []
    
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        # Avoid single-class resample
        if len(np.unique(y_true[idx])) < 2:
            continue
        
        boot_acc.append(accuracy_score(y_true[idx], y_pred[idx]))
        boot_auc.append(roc_auc_score(y_true[idx], y_prob[idx]))
        boot_sens.append(recall_score(y_true[idx], y_pred[idx], zero_division=0))
        boot_prec.append(precision_score(y_true[idx], y_pred[idx], zero_division=0))
        boot_f1.append(f1_score(y_true[idx], y_pred[idx], zero_division=0))
        
    def ci_bounds(arr):
        return np.percentile(arr, 2.5), np.percentile(arr, 97.5), np.std(arr)
        
    return {
        "acc_ci": ci_bounds(boot_acc),
        "auc_ci": ci_bounds(boot_auc),
        "sens_ci": ci_bounds(boot_sens),
        "prec_ci": ci_bounds(boot_prec),
        "f1_ci": ci_bounds(boot_f1)
    }


def main():
    print("=" * 95)
    print("PHASE 7: FINAL ONE-SHOT TEST EVALUATION (UNTOUCHED TEST SET: N=5,000 PATCHES)")
    print("=" * 95)
    
    # 1. Load Test Features
    print("\n[1/6] Loading Track A test features from outputs/features/full96_test.csv...")
    df_test = pd.read_csv(FEATURES_DIR / "full96_test.csv")
    feature_cols = [c for c in df_test.columns if c != "label"]
    X_test = df_test[feature_cols].values
    y_test = df_test["label"].values.astype(int)
    
    print(f"      Test set loaded: {len(df_test):,} patches (50% Normal: {sum(y_test==0):,}, 50% Tumor: {sum(y_test==1):,})")
    
    # 2. Load Trained Frozen Models
    print("\n[2/6] Loading frozen models from outputs/models/...")
    model_configs = [
        ("Logistic Regression", MODELS_DIR / "LogReg_full96_fused.pkl", "logreg", "#7f8c8d", "-."),
        ("Linear SVM", MODELS_DIR / "LinearSVM_full96.pkl", "linear_svm", "#8e44ad", ":"),
        ("Tuned Random Forest", MODELS_DIR / "RandomForest_full96_tuned.pkl", "random_forest", "#27ae60", "--"),
        ("HistGradientBoosting", MODELS_DIR / "HistGBM_full96.pkl", "histgbm", "#e74c3c", "-"),
        ("Tuned RBF Kernel SVM", MODELS_DIR / "Best_RBFSVM_full96.pkl", "rbf_svm_tuned", "#2980b9", "-"),
        ("Default RBF Kernel SVM", MODELS_DIR / "RBFSVM_full96.pkl", "rbf_svm_default", "#3498db", "--")
    ]
    
    models = {}
    preds_dict = {}
    probs_dict = {}
    
    for name, path, key, color, ls in model_configs:
        m = joblib.load(path)
        models[name] = m
        if hasattr(m, "predict_proba"):
            probs = m.predict_proba(X_test)[:, 1]
        else:
            scores = m.decision_function(X_test)
            probs = expit(scores)
        preds = (probs >= 0.5).astype(int)
        
        probs_dict[name] = probs
        preds_dict[name] = preds

    # Add Hybrid Ensemble Blend (50/50 blend of Tuned RBF + HistGBM)
    p_blend_50 = 0.5 * probs_dict["Tuned RBF Kernel SVM"] + 0.5 * probs_dict["HistGradientBoosting"]
    probs_dict["Hybrid Blend (50/50)"] = p_blend_50
    preds_dict["Hybrid Blend (50/50)"] = (p_blend_50 >= 0.5).astype(int)

    # Add Hybrid Ensemble Blend (70/30 blend of Tuned RBF + HistGBM)
    p_blend_70 = 0.7 * probs_dict["Tuned RBF Kernel SVM"] + 0.3 * probs_dict["HistGradientBoosting"]
    probs_dict["Hybrid Blend (70/30)"] = p_blend_70
    preds_dict["Hybrid Blend (70/30)"] = (p_blend_70 >= 0.5).astype(int)

    # 3. Compute Metrics & Confidence Intervals
    print("\n[3/6] Computing diagnostic metrics & 1,000-sample bootstrap confidence intervals...")
    test_results = []
    ci_records = []
    roc_plot_data = []
    pr_plot_data = []
    
    eval_order = [
        "Hybrid Blend (50/50)",
        "Hybrid Blend (70/30)",
        "HistGradientBoosting",
        "Default RBF Kernel SVM",
        "Tuned Random Forest",
        "Tuned RBF Kernel SVM",
        "Logistic Regression",
        "Linear SVM"
    ]
    
    colors_map = {
        "Hybrid Blend (50/50)": "#8e44ad",
        "Hybrid Blend (70/30)": "#9b59b6",
        "HistGradientBoosting": "#e74c3c",
        "Default RBF Kernel SVM": "#2980b9",
        "Tuned Random Forest": "#27ae60",
        "Tuned RBF Kernel SVM": "#16a085",
        "Logistic Regression": "#7f8c8d",
        "Linear SVM": "#d35400"
    }
    
    for name in eval_order:
        p = probs_dict[name]
        pred = preds_dict[name]
        
        acc = accuracy_score(y_test, pred)
        auc = roc_auc_score(y_test, p)
        pr_auc = average_precision_score(y_test, p)
        rec = recall_score(y_test, pred)
        prec = precision_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        bal_acc = balanced_accuracy_score(y_test, pred)
        
        cm = confusion_matrix(y_test, pred)
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp)
        brier = np.mean((p - y_test) ** 2)
        
        # Bootstrap 95% CI
        ci = bootstrap_ci(y_test, p, pred, n_bootstraps=1000)
        
        test_results.append({
            "model": name,
            "test_accuracy": acc,
            "test_roc_auc": auc,
            "test_pr_auc": pr_auc,
            "test_sensitivity": rec,
            "test_specificity": spec,
            "test_precision": prec,
            "test_f1_score": f1,
            "test_balanced_accuracy": bal_acc,
            "brier_score": round(brier, 4),
            "true_positives": tp,
            "false_negatives": fn,
            "true_negatives": tn,
            "false_positives": fp
        })
        
        ci_records.append({
            "model": name,
            "accuracy_mean": acc,
            "acc_ci_low": ci["acc_ci"][0],
            "acc_ci_high": ci["acc_ci"][1],
            "auc_mean": auc,
            "auc_ci_low": ci["auc_ci"][0],
            "auc_ci_high": ci["auc_ci"][1],
            "sens_mean": rec,
            "sens_ci_low": ci["sens_ci"][0],
            "sens_ci_high": ci["sens_ci"][1],
            "prec_mean": prec,
            "prec_ci_low": ci["prec_ci"][0],
            "prec_ci_high": ci["prec_ci"][1],
            "f1_mean": f1,
            "f1_ci_low": ci["f1_ci"][0],
            "f1_ci_high": ci["f1_ci"][1]
        })
        
        # Add to plots
        roc_plot_data.append({
            "y_true": y_test,
            "y_prob": p,
            "label": f"{name} (AUC = {auc:.4f})",
            "color": colors_map.get(name, "#333"),
            "linestyle": "-" if "Blend" in name or name == "HistGradientBoosting" else "--",
            "linewidth": 2.2 if "Blend" in name else 1.8
        })
        
        print(f"  [Tested] {name:<26} | Acc: {acc*100:.2f}% | AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f} | Sens: {rec*100:.2f}% | Prec: {prec*100:.2f}% | F1: {f1:.4f}")

    # 4. Generalization Gap Analysis
    print("\n[4/6] Computing Generalization Gap (Validation vs. Test Set)...")
    val_bench_path = RESULTS_DIR / "candidate_models_benchmark.csv"
    val_df = pd.read_csv(val_bench_path) if val_bench_path.exists() else pd.DataFrame()
    
    val_metrics_map = {
        "Logistic Regression": (0.7818, 0.8652),
        "Linear SVM": (0.7814, 0.8684),
        "Tuned Random Forest": (0.7944, 0.8946),
        "HistGradientBoosting": (0.8066, 0.8990),
        "Tuned RBF Kernel SVM": (0.8236, 0.9082),
        "Default RBF Kernel SVM": (0.8214, 0.9038),
        "Hybrid Blend (50/50)": (0.8192, 0.9093),
        "Hybrid Blend (70/30)": (0.8216, 0.9108)
    }
    
    gen_records = []
    for r in test_results:
        m_name = r["model"]
        val_acc, val_auc = val_metrics_map.get(m_name, (np.nan, np.nan))
        delta_acc = (val_acc - r["test_accuracy"]) * 100
        delta_auc = val_auc - r["test_roc_auc"]
        
        gen_records.append({
            "model": m_name,
            "validation_accuracy": val_acc,
            "test_accuracy": r["test_accuracy"],
            "accuracy_drop_pct": round(delta_acc, 2),
            "validation_roc_auc": val_auc,
            "test_roc_auc": r["test_roc_auc"],
            "roc_auc_drop": round(delta_auc, 4)
        })

    # Save Tables
    df_test_bench = pd.DataFrame(test_results)
    df_ci = pd.DataFrame(ci_records)
    df_gen = pd.DataFrame(gen_records)
    
    df_test_bench.to_csv(RESULTS_DIR / "final_test_benchmark.csv", index=False)
    df_ci.to_csv(RESULTS_DIR / "test_bootstrap_confidence_intervals.csv", index=False)
    df_gen.to_csv(RESULTS_DIR / "generalization_gap_analysis.csv", index=False)
    print(f"[OK] Saved final_test_benchmark.csv, test_bootstrap_confidence_intervals.csv, generalization_gap_analysis.csv")

    # 5. Generate Test Visualizations
    print("\n[5/6] Generating Test Set ROC and Precision-Recall Curves...")
    
    # 5a. Test ROC Curve Plot
    test_roc_path = PLOTS_DIR / "test_roc_curves.png"
    plot_roc_curves(
        roc_plot_data,
        output_path=test_roc_path,
        title="PatchCamelyon Final Test Evaluation: Multi-Model ROC Curves (N=5,000)"
    )
    
    # 5b. Test Precision-Recall Curve Plot
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for name in eval_order:
        p = probs_dict[name]
        precisions, recalls, _ = precision_recall_curve(y_test, p)
        pr_auc = average_precision_score(y_test, p)
        ax.plot(
            recalls, precisions,
            label=f"{name} (PR-AUC = {pr_auc:.4f})",
            color=colors_map.get(name, "#333"),
            linewidth=2.2 if "Blend" in name else 1.8,
            linestyle="-" if "Blend" in name or name == "HistGradientBoosting" else "--"
        )
    ax.axhline(0.5, color="gray", linestyle=":", label="Chance (PR-AUC = 0.5000)")
    ax.set_xlabel("Recall / Sensitivity (True Tumors Caught)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="bold")
    ax.set_title("PatchCamelyon Final Test Set Precision-Recall Curves (N=5,000)", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower left", fontsize=9.5, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    test_pr_path = PLOTS_DIR / "test_pr_curves.png"
    plt.savefig(test_pr_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved test_roc_curves.png and test_pr_curves.png")

    # 5c. Confusion Matrices for Top Models
    top_models_cm = [
        ("Hybrid Blend (50/50)", preds_dict["Hybrid Blend (50/50)"], "test_cm_hybrid_blend.png"),
        ("HistGradientBoosting", preds_dict["HistGradientBoosting"], "test_cm_histgbm.png"),
        ("Tuned Random Forest", preds_dict["Tuned Random Forest"], "test_cm_random_forest.png"),
        ("Default RBF Kernel SVM", preds_dict["Default RBF Kernel SVM"], "test_cm_rbf_svm.png"),
        ("Logistic Regression", preds_dict["Logistic Regression"], "test_cm_baseline_logreg.png")
    ]
    
    for m_label, pred_arr, cm_filename in top_models_cm:
        cm_out = PLOTS_DIR / cm_filename
        plot_confusion_matrix(
            y_test, pred_arr,
            model_name=f"{m_label} (Test Set)",
            output_path=cm_out
        )

    # 6. Print Official Test Leaderboard
    print("\n" + "=" * 105)
    print("OFFICIAL FINAL TEST BENCHMARK LEADERBOARD (Held-Out Test Set: 5,000 Patches):")
    print("=" * 105)
    disp_cols = ["model", "test_accuracy", "test_roc_auc", "test_pr_auc", "test_sensitivity", "test_precision", "test_f1_score", "brier_score"]
    fmt_test = df_test_bench[disp_cols].copy()
    fmt_test["test_accuracy"] = (fmt_test["test_accuracy"] * 100).round(2).astype(str) + "%"
    fmt_test["test_sensitivity"] = (fmt_test["test_sensitivity"] * 100).round(2).astype(str) + "%"
    fmt_test["test_precision"] = (fmt_test["test_precision"] * 100).round(2).astype(str) + "%"
    fmt_test["test_roc_auc"] = fmt_test["test_roc_auc"].round(4)
    fmt_test["test_pr_auc"] = fmt_test["test_pr_auc"].round(4)
    fmt_test["test_f1_score"] = fmt_test["test_f1_score"].round(4)
    print(fmt_test.to_string(index=False))
    print("=" * 105)

    print("\n" + "=" * 105)
    print("95% BOOTSTRAP CONFIDENCE INTERVALS (1,000 Resamples on Test Set):")
    print("=" * 105)
    for _, r in df_ci.iterrows():
        print(f"  {r['model']:<25} | Acc: {r['accuracy_mean']*100:.2f}% [{r['acc_ci_low']*100:.2f}%, {r['acc_ci_high']*100:.2f}%] | AUC: {r['auc_mean']:.4f} [{r['auc_ci_low']:.4f}, {r['auc_ci_high']:.4f}] | Sens: {r['sens_mean']*100:.2f}% [{r['sens_ci_low']*100:.2f}%, {r['sens_ci_high']*100:.2f}%]")
    print("=" * 105)

    print("\n" + "=" * 105)
    print("GENERALIZATION GAP AUDIT (Validation Accuracy vs. Test Accuracy):")
    print("=" * 105)
    disp_gen = ["model", "validation_accuracy", "test_accuracy", "accuracy_drop_pct", "validation_roc_auc", "test_roc_auc", "roc_auc_drop"]
    fmt_gen = df_gen[disp_gen].copy()
    fmt_gen["validation_accuracy"] = (fmt_gen["validation_accuracy"] * 100).round(2).astype(str) + "%"
    fmt_gen["test_accuracy"] = (fmt_gen["test_accuracy"] * 100).round(2).astype(str) + "%"
    fmt_gen["accuracy_drop_pct"] = fmt_gen["accuracy_drop_pct"].astype(str) + "%"
    fmt_gen["validation_roc_auc"] = fmt_gen["validation_roc_auc"].round(4)
    fmt_gen["test_roc_auc"] = fmt_gen["test_roc_auc"].round(4)
    print(fmt_gen.to_string(index=False))
    print("=" * 105)
    print("[OK] Phase 7 Final One-Shot Test Evaluation completed successfully.")


if __name__ == "__main__":
    main()

