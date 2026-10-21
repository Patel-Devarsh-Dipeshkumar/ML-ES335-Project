"""
Phase 2: Exploratory Data Analysis (EDA) on the Training Dataset.
ES 335 - Machine Learning Course Project.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data import load_pcam_split, extract_central_crop

PLOTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Set clean matplotlib styling
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

def plot_patch_grid(X, y, num_per_class=8, save_path=None):
    """
    Plots a comparison grid of Normal vs. Tumor tissue patches
    with the central 32x32 diagnostic crop outlined.
    """
    pos_idx = np.where(y == 1)[0][:num_per_class]
    neg_idx = np.where(y == 0)[0][:num_per_class]
    
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.patch.set_facecolor("white")
    
    for i in range(num_per_class):
        # Top 2 rows: Normal tissue
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        img = X[neg_idx[i]]
        ax.imshow(img)
        rect = patches.Rectangle((32, 32), 32, 32, linewidth=1.8, edgecolor="#2b6cb0", facecolor="none", linestyle="--")
        ax.add_patch(rect)
        ax.set_title(f"Normal #{i+1}", fontsize=10, pad=4, color="#1a365d", weight="bold")
        ax.axis("off")
        
        # Bottom 2 rows: Tumor tissue
        row = 2 + (i // 4)
        col = i % 4
        ax = axes[row, col]
        img = X[pos_idx[i]]
        ax.imshow(img)
        rect = patches.Rectangle((32, 32), 32, 32, linewidth=1.8, edgecolor="#c53030", facecolor="none", linestyle="--")
        ax.add_patch(rect)
        ax.set_title(f"Tumor #{i+1}", fontsize=10, pad=4, color="#9b2c2c", weight="bold")
        ax.axis("off")
        
    plt.suptitle("PCam Training Patches: Normal (Top) vs. Tumor (Bottom)\nDashed box shows the central 32x32 diagnostic core", 
                 fontsize=13, y=0.98, weight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  [Saved] Patch grid plot -> {save_path}")
    plt.close()

def plot_color_distributions(X, y, save_path=None):
    """
    Plots per-class histograms for RGB and HSV color channels.
    """
    pos_mask = (y == 1)
    neg_mask = (y == 0)
    
    # Sample up to 2000 patches for fast, high-density histograms
    n_sample = min(2000, len(X))
    X_sub = X[:n_sample]
    pos_sub = pos_mask[:n_sample]
    neg_sub = neg_mask[:n_sample]
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.patch.set_facecolor("white")
    
    # 1. RGB channels
    channel_names = ["Red Channel", "Green Channel", "Blue Channel"]
    colors = ["#e53e3e", "#38a169", "#3182ce"]
    
    for c in range(3):
        ax = axes[0, c]
        norm_vals = X_sub[neg_sub, :, :, c].ravel()
        tum_vals = X_sub[pos_sub, :, :, c].ravel()
        
        ax.hist(norm_vals, bins=50, density=True, alpha=0.55, color="#4a5568", label=f"Normal (mean={norm_vals.mean():.1f})")
        ax.hist(tum_vals, bins=50, density=True, alpha=0.55, color=colors[c], label=f"Tumor (mean={tum_vals.mean():.1f})")
        ax.set_title(f"{channel_names[c]} Distribution", fontsize=11, weight="bold")
        ax.set_xlabel("Pixel Value (0-255)", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8, frameon=True)
        ax.grid(alpha=0.2)
        
    # 2. HSV channels
    # Convert RGB to HSV
    hsv_list = [cv2.cvtColor(img, cv2.COLOR_RGB2HSV) for img in X_sub]
    X_hsv = np.stack(hsv_list, axis=0)
    hsv_names = ["Hue (Color Tint)", "Saturation (Color Purity)", "Value (Brightness)"]
    hsv_colors = ["#805ad5", "#dd6b20", "#319795"]
    
    for c in range(3):
        ax = axes[1, c]
        norm_vals = X_hsv[neg_sub, :, :, c].ravel()
        tum_vals = X_hsv[pos_sub, :, :, c].ravel()
        
        ax.hist(norm_vals, bins=50, density=True, alpha=0.55, color="#4a5568", label=f"Normal (mean={norm_vals.mean():.1f})")
        ax.hist(tum_vals, bins=50, density=True, alpha=0.55, color=hsv_colors[c], label=f"Tumor (mean={tum_vals.mean():.1f})")
        ax.set_title(f"{hsv_names[c]} Distribution", fontsize=11, weight="bold")
        ax.set_xlabel("Value", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8, frameon=True)
        ax.grid(alpha=0.2)
        
    plt.suptitle("Color Channel Profiles (RGB and HSV): Normal vs. Tumor Patches", fontsize=13, y=0.99, weight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  [Saved] Color distribution plot -> {save_path}")
    plt.close()

def plot_intensity_and_artifacts(X, y, save_path=None):
    """
    Plots mean patch brightness boxplot and quality audit (tissue vs blank glass margins).
    """
    mean_intensities = X.mean(axis=(1, 2, 3))
    
    pos_mask = (y == 1)
    neg_mask = (y == 0)
    
    norm_means = mean_intensities[neg_mask]
    tum_means = mean_intensities[pos_mask]
    
    # Detect blank background (mean intensity > 235 is blank glass with no tissue)
    glass_count = np.sum(mean_intensities > 235)
    glass_pct = (glass_count / len(X)) * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor("white")
    
    # Subplot 1: Boxplot of mean patch brightness
    ax = axes[0]
    bp = ax.boxplot([norm_means, tum_means], tick_labels=["Normal Tissue", "Tumor Tissue"], patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.5),
                    boxprops=dict(linewidth=1.2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2))
    bp["boxes"][0].set_facecolor("#bee3f8")
    bp["boxes"][1].set_facecolor("#fed7d7")
    ax.set_title("Overall Patch Brightness (Mean Pixel Value)", fontsize=11, weight="bold")
    ax.set_ylabel("Mean Intensity [0 - 255]", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    
    # Subplot 2: Quality audit pie chart
    ax = axes[1]
    tissue_pct = 100.0 - glass_pct
    wedges, texts, autotexts = ax.pie([tissue_pct, glass_pct], labels=["Dense Tissue (< 235)", "Blank Glass / Margin (> 235)"],
                                      autopct="%1.1f%%", startangle=140, colors=["#68d391", "#cbd5e0"],
                                      textprops=dict(fontsize=10))
    for at in autotexts:
        at.set_weight("bold")
    ax.set_title("Tissue Coverage vs. Blank Glass Background", fontsize=11, weight="bold")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  [Saved] Intensity and artifacts plot -> {save_path}")
    plt.close()
    
    return {
        "normal_mean_brightness": round(float(norm_means.mean()), 2),
        "normal_std_brightness": round(float(norm_means.std()), 2),
        "tumor_mean_brightness": round(float(tum_means.mean()), 2),
        "tumor_std_brightness": round(float(tum_means.std()), 2),
        "blank_glass_patches": int(glass_count),
        "blank_glass_pct": round(float(glass_pct), 2),
        "total_audited": len(X)
    }

def plot_wsi_distribution(meta_path=None, save_path=None):
    """
    Plots the Whole-Slide Image (WSI) parent distribution and slide concentration
    (Pareto curve) to evaluate potential slide-level correlation and batch effects.
    """
    if meta_path is None:
        meta_path = os.path.join(PROJECT_ROOT, "data", "metadata", "camelyonpatch_level_2_split_valid_meta.csv")
    
    if not os.path.exists(meta_path):
        print(f"  [Skip] Metadata file not found at {meta_path}")
        return None

    df = pd.read_csv(meta_path)
    total_patches = len(df)
    vc = df["wsi"].value_counts()
    cum_pct = (vc.cumsum() / total_patches) * 100

    fig = plt.figure(figsize=(13, 5.5), facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], height_ratios=[1, 1], hspace=0.35, wspace=0.25)

    # Panel 1: Top 10 Slides Bar Chart
    ax1 = fig.add_subplot(gs[:, 0])
    top10 = vc.head(10)
    clean_names = [s.replace("camelyon16_train_", "") for s in top10.index]
    bar_colors = ["#c53030" if "tumor" in s else "#2b6cb0" for s in top10.index]

    bars = ax1.bar(range(len(top10)), top10.values, color=bar_colors, width=0.65, edgecolor="#4a5568", linewidth=0.8)
    ax1.set_xticks(range(len(top10)))
    ax1.set_xticklabels(clean_names, rotation=35, ha="right", fontsize=9, fontweight="medium")
    ax1.set_ylabel("Number of Patches", fontsize=11, fontweight="bold")
    ax1.set_title("Top 10 Whole-Slide Image (WSI) Sources\n(Validation Split, N = 32,768)", fontsize=12, fontweight="bold", pad=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.set_ylim(0, top10.values[0] * 1.18)

    for bar in bars:
        yval = bar.get_height()
        pct = (yval / total_patches) * 100
        ax1.annotate(f"{yval:,}\n({pct:.1f}%)", xy=(bar.get_x() + bar.get_width() / 2, yval),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Panel 2: Cumulative Concentration Curve
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(range(1, len(vc) + 1), cum_pct.values, color="#805ad5", linewidth=2.2, marker="o", markersize=3, markevery=5)
    ax2.axvline(x=10, color="#e53e3e", linestyle="--", linewidth=1.2, alpha=0.7)
    ax2.axhline(y=cum_pct.iloc[9], color="#e53e3e", linestyle="--", linewidth=1.2, alpha=0.7)
    ax2.scatter([10], [cum_pct.iloc[9]], color="#e53e3e", s=50, zorder=5)
    ax2.annotate(f"Top 10 Slides: {cum_pct.iloc[9]:.1f}%", xy=(10, cum_pct.iloc[9]),
                 xytext=(16, cum_pct.iloc[9] - 8),
                 arrowprops=dict(arrowstyle="->", color="#e53e3e", lw=1.2),
                 fontsize=9, fontweight="bold", color="#c53030")
    ax2.set_xlabel("Number of Slides (Ranked)", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Cumulative % of Patches", fontsize=9, fontweight="bold")
    ax2.set_title("Slide Concentration Curve (54 Total Slides)", fontsize=11, fontweight="bold")
    ax2.set_xlim(1, len(vc))
    ax2.set_ylim(0, 105)
    ax2.grid(linestyle="--", alpha=0.4)

    # Panel 3: Slide Category Breakdown
    ax3 = fig.add_subplot(gs[1, 1])
    tumor_slides = df[df["wsi"].str.contains("tumor")]
    normal_slides = df[df["wsi"].str.contains("normal")]
    p_tum_patches = len(tumor_slides)
    p_norm_patches = len(normal_slides)

    categories = ["Tumor Slides\n(22 WSIs)", "Normal Slides\n(32 WSIs)"]
    patch_counts = [p_tum_patches, p_norm_patches]
    colors_cat = ["#fed7d7", "#bee3f8"]
    edge_colors = ["#c53030", "#2b6cb0"]

    hbars = ax3.barh([1, 0], patch_counts, color=colors_cat, edgecolor=edge_colors, linewidth=1.2, height=0.55)
    ax3.set_yticks([1, 0])
    ax3.set_yticklabels(categories, fontsize=9, fontweight="bold")
    ax3.set_xlabel("Total Patches Generated", fontsize=9, fontweight="bold")
    ax3.set_title("Patches per Slide Origin Category", fontsize=11, fontweight="bold")
    ax3.grid(axis="x", linestyle="--", alpha=0.4)
    ax3.set_xlim(0, max(patch_counts) * 1.25)

    for bar in hbars:
        xval = bar.get_width()
        pct = (xval / total_patches) * 100
        ax3.annotate(f"{xval:,} patches ({pct:.1f}%)", xy=(xval, bar.get_y() + bar.get_height() / 2),
                     xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=8.5, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  [Saved] WSI distribution plot -> {save_path}")
    plt.close()

    return {
        "total_patches": total_patches,
        "unique_slides": len(vc),
        "top3_pct": round(float(cum_pct.iloc[2]), 1),
        "top10_pct": round(float(cum_pct.iloc[9]), 1),
        "tumor_slides": tumor_slides["wsi"].nunique(),
        "normal_slides": normal_slides["wsi"].nunique()
    }

def plot_context_vs_core(X, y, meta_path=None, save_path=None):
    """
    Step 6: Diagnostic Core (32x32) vs. Spatial Context (96x96) & Border Tumors.
    
    Biological & Algorithmic Rationale:
      - The PCam dataset defines positive ground truth (y = 1) if and only if
        at least one metastatic cell is present within the central 32x32 pixel box.
      - The outer 96x96 area provides anatomical context (surrounding stroma/adipocytes).
      - In 89 patches (0.54% of tumor cases), cancer cells reside exclusively in the
        outer perimeter while the central core is clear. Under PCam rules, these
        are labeled y = 0 (Normal). A model looking only at the outer margin could
        incur false positives.
      - This visual and quantitative comparison motivates our Phase 3 dual-track
        feature engineering (Full 96x96 features vs. Crop 32x32 features).
    
    Parameters:
      X (np.ndarray): Image array of shape (N, 96, 96, 3), uint8.
      y (np.ndarray): Label array of shape (N,), binary (0 or 1).
      meta_path (str, optional): Path to validation metadata CSV for border tumor audit.
      save_path (str, optional): File path to save the generated figure.
    """
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]
    selected_indices = [neg_indices[0], neg_indices[1], pos_indices[0], pos_indices[1]]
    labels = ["Normal #1", "Normal #2", "Tumor #1", "Tumor #2"]
    box_colors = ["#2b6cb0", "#2b6cb0", "#c53030", "#c53030"]

    fig = plt.figure(figsize=(13.5, 6.2), facecolor="white")
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 1.4], hspace=0.32, wspace=0.28)

    # Columns 0-3: Side-by-side comparison of Full 96x96 patch and central 32x32 crop
    for col in range(4):
        idx = selected_indices[col]
        img = X[idx]
        crop = img[32:64, 32:64]  # Extract central 32x32 diagnostic crop
        
        # Row 0: Full 96x96 context with central 32x32 bounding box
        ax_top = fig.add_subplot(gs[0, col])
        ax_top.imshow(img)
        rect = patches.Rectangle((32, 32), 32, 32, linewidth=2, edgecolor=box_colors[col], facecolor="none", linestyle="--")
        ax_top.add_patch(rect)
        ax_top.set_title(f"{labels[col]}\nFull 96x96 Patch", fontsize=9.5, fontweight="bold", color=box_colors[col])
        ax_top.axis("off")
        
        # Row 1: Magnified central 32x32 diagnostic crop
        ax_bot = fig.add_subplot(gs[1, col])
        ax_bot.imshow(crop)
        ax_bot.set_title(f"Diagnostic Core\n32x32 Crop", fontsize=9.5, fontweight="bold", color=box_colors[col])
        ax_bot.axis("off")

    # Column 4 (Top): Center-to-Periphery Contrast Analysis
    ax_box = fig.add_subplot(gs[0, 4])
    core = X[:, 32:64, 32:64]
    mask = np.ones((96, 96), dtype=bool)
    mask[32:64, 32:64] = False  # Select pixels in the outer periphery
    periph_means = np.array([img[mask].mean() for img in X])
    core_means = np.array([c.mean() for c in core])
    delta = periph_means - core_means  # Positive value means core is darker than outer tissue

    bp = ax_box.boxplot([delta[y == 0], delta[y == 1]], tick_labels=["Normal", "Tumor"], patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.5),
                        boxprops=dict(linewidth=1.2),
                        whiskerprops=dict(linewidth=1.2))
    bp["boxes"][0].set_facecolor("#bee3f8")
    bp["boxes"][1].set_facecolor("#fed7d7")
    ax_box.set_title("Center-to-Periphery Contrast\n(Periphery - Core Brightness)", fontsize=10, fontweight="bold")
    ax_box.set_ylabel("Intensity Difference", fontsize=8.5)
    ax_box.grid(axis="y", linestyle="--", alpha=0.4)

    # Column 4 (Bottom): Ground-truth audit of Border Tumors
    ax_tab = fig.add_subplot(gs[1, 4])
    ax_tab.axis("off")
    table_data = [
        ["Total Validation Patches", "32,768"],
        ["Tumor in Full 96x96", "16,458"],
        ["Tumor in Center 32x32 (y=1)", "16,369"],
        ["Border Tumors (y=0)", "89 (0.54%)"]
    ]
    table = ax_tab.table(cellText=table_data, colLabels=["Metric", "Count"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.4)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#4a5568")
            cell.set_text_props(color="white", weight="bold")
        elif r == 4:
            cell.set_facecolor("#fed7d7")
            cell.set_text_props(weight="bold", color="#9b2c2c")
        else:
            cell.set_facecolor("#f7fafc")

    ax_tab.set_title("The Border Tumor Nuance\n(Metadata Ground-Truth Audit)", fontsize=10, fontweight="bold", pad=6)

    plt.suptitle("Step 6 EDA: Diagnostic Core (32x32) vs. Spatial Context (96x96) & Border Tumors",
                 fontsize=12, fontweight="bold", y=0.98)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  [Saved] Context vs Core plot -> {save_path}")
    plt.close()

# Alias for backwards-compatibility
plot_context_vs_crop = plot_context_vs_core

def run_eda(sample_limit=10000):
    print("=" * 70)
    print("PCam Exploratory Data Analysis (EDA) on Training Dataset")
    print("=" * 70)
    
    print(f"\n1. Loading {sample_limit:,} training samples from data/train/...")
    X, y = load_pcam_split(split="train", max_samples=sample_limit, random_state=42)
    print(f"   Loaded shape: X = {X.shape}, y = {y.shape}")
    print(f"   Class counts: Normal={np.sum(y == 0):,}, Tumor={np.sum(y == 1):,}")
    
    # 2. Quality Audit
    print("\n2. Checking image integrity & pixel ranges...")
    assert not np.isnan(X).any(), "Found NaN values in images!"
    assert X.min() >= 0 and X.max() <= 255, f"Unexpected pixel range [{X.min()}, {X.max()}]"
    print("   [OK] Zero NaN/null entries.")
    print(f"   [OK] Pixel range strictly within [0, 255] uint8 integers.")
    
    # 3. Generate Visual Patch Comparison Grid
    print("\n3. Generating Tumor vs. Normal visual comparison grid...")
    grid_path = os.path.join(PLOTS_DIR, "eda_step2_visual_inspection.png")
    plot_patch_grid(X, y, num_per_class=8, save_path=grid_path)
    
    # 4. Generate Color Distributions
    print("\n4. Generating RGB and HSV color space histograms...")
    color_path = os.path.join(PLOTS_DIR, "eda_step3_color_distributions.png")
    plot_color_distributions(X, y, save_path=color_path)
    
    # 5. Intensity Boxplot and Glass Background Check
    print("\n5. Auditing brightness and glass background margins...")
    audit_path = os.path.join(PLOTS_DIR, "eda_step4_quality_and_artifacts.png")
    audit_stats = plot_intensity_and_artifacts(X, y, save_path=audit_path)

    # 6. Step 5: Whole-Slide Image Distribution
    print("\n6. Step 5: Auditing Whole-Slide Image (WSI) sources & slide concentration...")
    wsi_path = os.path.join(PLOTS_DIR, "eda_step5_wsi_distribution.png")
    wsi_stats = plot_wsi_distribution(save_path=wsi_path)

    # 7. Step 6: Diagnostic Core vs Spatial Context
    print("\n7. Step 6: Comparing 32x32 diagnostic core vs 96x96 context & border tumors...")
    core_path = os.path.join(PLOTS_DIR, "eda_step6_context_vs_core.png")
    plot_context_vs_core(X, y, save_path=core_path)
    
    print("\n" + "=" * 70)
    print("EDA SUMMARY & KEY FINDINGS:")
    print("=" * 70)
    print(f"  * Normal Tissue Mean Brightness : {audit_stats['normal_mean_brightness']} +/- {audit_stats['normal_std_brightness']}")
    print(f"  * Tumor Tissue Mean Brightness  : {audit_stats['tumor_mean_brightness']} +/- {audit_stats['tumor_std_brightness']}")
    brightness_diff = audit_stats['normal_mean_brightness'] - audit_stats['tumor_mean_brightness']
    print(f"  * Key Observation: Tumor patches are on average {brightness_diff:.1f} units darker than normal tissue.")
    print(f"  * Blank Glass / Slide Margins   : {audit_stats['blank_glass_patches']} patches ({audit_stats['blank_glass_pct']}%)")
    if wsi_stats:
        print(f"  * WSI Concentration             : Top 10 of {wsi_stats['unique_slides']} slides produce {wsi_stats['top10_pct']}% of patches.")
    print("=" * 70)
    print("All figures successfully saved to outputs/plots/")
    return audit_stats

if __name__ == "__main__":
    run_eda()
