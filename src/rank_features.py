"""
Statistical Feature Ranking & Audit Script for PCam Tumor Detection
Course Project: ES 335 (IIT Gandhinagar)

Evaluates all 59 engineered features across Track A (Full 96x96) and
Track B (Crop 32x32) using:
  1. ANOVA F-statistic (Univariate linear signal separation)
  2. Random Forest Gini Importance (Multivariate tree-based interaction)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier

# Setup paths relative to script location
BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = BASE_DIR / "outputs" / "features"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"

# Feature family categorization
FAMILY_COLORS = {
    "Color Moments": "#8e44ad",      # Purple (H&E staining/pigments)
    "GLCM Texture": "#2980b9",       # Blue (spatial regularity)
    "Edge Sharpness": "#27ae60",     # Green (cell borders/nuclei density)
    "LBP Micro-texture": "#e67e22",  # Orange (fine micro-patterns)
    "PCA Components": "#7f8c8d"      # Gray (raw pixel eigenvectors)
}

def get_feature_family(name: str) -> str:
    """Categorize a feature name into its biological / mathematical family."""
    if any(name.startswith(p) for p in ["rgb_", "hsv_", "lab_", "color_"]):
        return "Color Moments"
    elif name.startswith("glcm_"):
        return "GLCM Texture"
    elif name.startswith("lbp_"):
        return "LBP Micro-texture"
    elif any(name.startswith(p) for p in ["edge_", "sobel_", "laplacian_"]):
        return "Edge Sharpness"
    elif name.startswith("pca_"):
        return "PCA Components"
    else:
        return "Other"

def compute_rankings(df: pd.DataFrame, track_name: str) -> pd.DataFrame:
    """Compute ANOVA F-scores and Random Forest Gini Importances on training set."""
    # Exclude metadata and target label
    exclude_cols = {"patch_id", "label"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].values
    y = df["label"].values.astype(int)
    
    print(f"[{track_name}] Computing ANOVA F-scores across {len(feature_cols)} features...")
    f_stats, p_vals = f_classif(X, y)
    
    print(f"[{track_name}] Fitting Random Forest (100 trees, all cores)...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y)
    rf_importances = rf.feature_importances_
    
    # Assemble results dataframe
    results = []
    for i, col in enumerate(feature_cols):
        family = get_feature_family(col)
        results.append({
            "track": track_name,
            "feature": col,
            "family": family,
            "anova_f_stat": float(f_stats[i]),
            "anova_p_val": float(p_vals[i]),
            "rf_importance": float(rf_importances[i])
        })
    
    res_df = pd.DataFrame(results)
    res_df["rank_anova"] = res_df["anova_f_stat"].rank(ascending=False).astype(int)
    res_df["rank_rf"] = res_df["rf_importance"].rank(ascending=False).astype(int)
    
    return res_df

def plot_feature_importance(df_full: pd.DataFrame, df_crop: pd.DataFrame, output_path: Path):
    """Generate publication-ready 2x2 comparison bar plot."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=300)
    plt.subplots_adjust(hspace=0.28, wspace=0.35)
    
    configs = [
        # (Row, Col, Dataframe, Metric, Metric Label, Title, Top N)
        (0, 0, df_full, "anova_f_stat", "ANOVA F-Statistic", "Track A (Full 96x96): Top 15 by ANOVA F-Score", 15),
        (0, 1, df_full, "rf_importance", "Gini Importance", "Track A (Full 96x96): Top 15 by Random Forest Importance", 15),
        (1, 0, df_crop, "anova_f_stat", "ANOVA F-Statistic", "Track B (Crop 32x32): Top 15 by ANOVA F-Score", 15),
        (1, 1, df_crop, "rf_importance", "Gini Importance", "Track B (Crop 32x32): Top 15 by Random Forest Importance", 15),
    ]
    
    for row, col, data, metric, xlabel, title, top_n in configs:
        ax = axes[row, col]
        top_df = data.sort_values(by=metric, ascending=False).head(top_n).copy()
        top_df = top_df.iloc[::-1]  # Invert so highest is at top
        
        colors = [FAMILY_COLORS.get(fam, "#95a5a6") for fam in top_df["family"]]
        bars = ax.barh(top_df["feature"], top_df[metric], color=colors, edgecolor="black", linewidth=0.7, alpha=0.88)
        
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel(xlabel, fontsize=11, fontweight="semibold")
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        ax.tick_params(axis="both", labelsize=10)
        
        # Add value labels on the bars
        max_val = top_df[metric].max()
        for bar in bars:
            val = bar.get_width()
            val_text = f"{val:.1f}" if val >= 10 else f"{val:.4f}"
            ax.text(
                val + 0.015 * max_val,
                bar.get_y() + bar.get_height() / 2,
                val_text,
                va="center",
                ha="left",
                fontsize=8.5,
                fontweight="semibold",
                color="#2c3e50"
            )
        ax.set_xlim(0, max_val * 1.18)

    # Add unified legend at the top
    handles = [
        plt.Rectangle((0,0),1,1, color=FAMILY_COLORS[fam], ec="black", lw=0.7)
        for fam in FAMILY_COLORS
    ]
    fig.legend(
        handles,
        FAMILY_COLORS.keys(),
        loc="upper center",
        ncol=5,
        fontsize=11,
        frameon=True,
        facecolor="#f8f9fa",
        edgecolor="#bdc3c7",
        bbox_to_anchor=(0.5, 0.985)
    )
    
    fig.suptitle(
        "Feature Importance & Discriminative Power Audit\nPatchCamelyon (PCam) Dual-Track Benchmark",
        fontsize=16,
        fontweight="bold",
        y=1.03
    )
    
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved feature importance chart to: {output_path}")

def main():
    print("=" * 70)
    print("STEP 3.3: STATISTICAL FEATURE RANKING & AUDIT")
    print("=" * 70)
    
    train_full_path = FEATURES_DIR / "full96_train.csv"
    train_crop_path = FEATURES_DIR / "crop32_train.csv"
    
    if not train_full_path.exists() or not train_crop_path.exists():
        raise FileNotFoundError("Feature CSV files not found in outputs/features/. Run feature_pipeline.py first.")
        
    print(f"Loading Track A (Full 96x96): {train_full_path.name}...")
    df_full_raw = pd.read_csv(train_full_path)
    print(f"Loading Track B (Crop 32x32): {train_crop_path.name}...")
    df_crop_raw = pd.read_csv(train_crop_path)
    
    # Compute rankings
    res_full = compute_rankings(df_full_raw, "full96")
    res_crop = compute_rankings(df_crop_raw, "crop32")
    
    # Combine and save
    combined_rankings = pd.concat([res_full, res_crop], ignore_index=True)
    rankings_csv_path = FEATURES_DIR / "feature_rankings.csv"
    combined_rankings.to_csv(rankings_csv_path, index=False)
    print(f"[OK] Saved full feature rankings table to: {rankings_csv_path}")
    
    # Plot Top 15 visual comparison
    plot_path = PLOTS_DIR / "feature_importance.png"
    plot_feature_importance(res_full, res_crop, plot_path)
    
    # Print console summary of Top 10 for Full96
    print("\n" + "=" * 70)
    print("TOP 10 MOST DISCRIMINATIVE FEATURES (Track A: Full 96x96):")
    print("=" * 70)
    top_full_anova = res_full.sort_values(by="anova_f_stat", ascending=False).head(10)
    print(f"{'Rank':<5} {'Feature':<22} {'Family':<18} {'ANOVA F-stat':<15} {'p-value'}")
    print("-" * 70)
    for _, row in top_full_anova.iterrows():
        p_str = "< 1e-100" if row['anova_p_val'] < 1e-100 else f"{row['anova_p_val']:.2e}"
        print(f"#{row['rank_anova']:<4} {row['feature']:<22} {row['family']:<18} {row['anova_f_stat']:<15.2f} {p_str}")
        
    print("\n" + "=" * 70)
    print("TOP 10 FEATURES BY RANDOM FOREST (Track A: Full 96x96):")
    print("=" * 70)
    top_full_rf = res_full.sort_values(by="rf_importance", ascending=False).head(10)
    print(f"{'Rank':<5} {'Feature':<22} {'Family':<18} {'RF Importance':<15}")
    print("-" * 70)
    for _, row in top_full_rf.iterrows():
        print(f"#{row['rank_rf']:<4} {row['feature']:<22} {row['family']:<18} {row['rf_importance']:<15.4f}")
        
    print("\n[OK] Step 3.3 feature audit complete.")

if __name__ == "__main__":
    main()
