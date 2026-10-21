"""
Phase 3: Dual-Track Feature Extraction Pipeline.
ES 335 - Machine Learning Course Project.

This script executes the complete end-to-end feature pipeline:
  1. Dual-Track Extraction:
       - Track A ('full96'): Complete 96x96 tissue context
       - Track B ('crop32'): Central 32x32 diagnostic focus region
  2. Multi-threaded parallel extraction using joblib across CPU cores.
  3. Representation fusion:
       - 39 Handcrafted Pathology Features (Color moments, GLCM, LBP, Edges)
       - 20 Raw Pixel PCA Features
       - Total: 59 engineered dimensions
  4. Strict Data Leakage Prevention:
       - StandardScaler fitted SOLELY on the Training set.
       - PixelPCAExtractor fitted SOLELY on the Training set.
       - Exact training transformations applied to Validation and Test splits.
  5. Persistence:
       - Saves clean tabular datasets to outputs/features/
       - Saves fitted scalers and PCA models to outputs/features/
"""

import os
import sys
import time
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data import load_pcam_split, extract_central_crop
from src.features import (
    extract_handcrafted_features,
    PixelPCAExtractor,
    downsample_patch
)

FEATURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "features")
os.makedirs(FEATURES_DIR, exist_ok=True)


def extract_features_single(img_rgb):
    """
    Worker function to extract handcrafted features from a single image patch.
    """
    feats, names = extract_handcrafted_features(img_rgb)
    return feats


def process_split_features(X, track="full96", n_jobs=12):
    """
    Extracts handcrafted features for an array of images in parallel.
    
    Args:
      X (np.ndarray): (N, 96, 96, 3) or (N, 32, 32, 3) uint8 RGB array.
      track (str): 'full96' or 'crop32'.
      n_jobs (int): Number of parallel CPU workers.
      
    Returns:
      df_handcrafted (pd.DataFrame): (N, 39) dataframe with feature names.
    """
    # 1. Apply central crop if track is crop32
    if track == "crop32" and X.shape[1] == 96:
        X_proc = extract_central_crop(X)
    else:
        X_proc = X

    # 2. Get feature column names from one sample
    _, feature_names = extract_handcrafted_features(X_proc[0])
    
    # 3. Parallel extraction across images
    print(f"  -> Extracting {len(feature_names)} handcrafted features for {len(X_proc):,} [{track}] patches using {n_jobs} workers...")
    t0 = time.time()
    results = joblib.Parallel(n_jobs=n_jobs, batch_size=64)(
        joblib.delayed(extract_features_single)(img) for img in X_proc
    )
    t1 = time.time()
    print(f"  -> Handcrafted extraction finished in {t1 - t0:.2f}s ({((t1 - t0)/len(X_proc))*1000:.1f} ms/patch).")
    
    df_handcrafted = pd.DataFrame(results, columns=feature_names)
    return df_handcrafted, X_proc


def run_dual_track_pipeline(
    train_samples=10000,
    val_samples=2500,
    test_samples=2500,
    n_jobs=12,
    random_state=42
):
    """
    Executes the dual-track feature extraction across Train, Val, and Test splits.
    """
    print("=" * 80)
    print("PCam Dual-Track Feature Extraction Pipeline (Phase 3)")
    print(f"Sample Budgets: Train={train_samples:,} | Val={val_samples:,} | Test={test_samples:,}")
    print(f"Parallel Workers: {n_jobs} cores | Random State: {random_state}")
    print("=" * 80)
    
    # --------------------------------------------------------------------------
    # Step 1: Load Raw Image Splits
    # --------------------------------------------------------------------------
    print("\n[1/4] Loading image splits from data/...")
    t_start = time.time()
    
    X_train, y_train = load_pcam_split("train", max_samples=train_samples, random_state=random_state)
    print(f"  Train split loaded: X={X_train.shape}, y={y_train.shape} (Normal={np.sum(y_train==0):,}, Tumor={np.sum(y_train==1):,})")
    
    X_val, y_val = load_pcam_split("validation", max_samples=val_samples, random_state=random_state)
    print(f"  Validation split loaded: X={X_val.shape}, y={y_val.shape} (Normal={np.sum(y_val==0):,}, Tumor={np.sum(y_val==1):,})")
    
    X_test, y_test = load_pcam_split("test", max_samples=test_samples, random_state=random_state)
    print(f"  Test split loaded: X={X_test.shape}, y={y_test.shape} (Normal={np.sum(y_test==0):,}, Tumor={np.sum(y_test==1):,})")

    tracks = ["full96", "crop32"]
    
    # --------------------------------------------------------------------------
    # Step 2: Process Track A ('full96') and Track B ('crop32')
    # --------------------------------------------------------------------------
    for track in tracks:
        print("\n" + "-" * 80)
        print(f"PROCESSING TRACK: [{track.upper()}]")
        print("-" * 80)
        
        # (a) Handcrafted features for Train, Val, Test
        print(f"\n[A] Handcrafted Feature Extraction ({track}):")
        df_train_hc, X_train_proc = process_split_features(X_train, track=track, n_jobs=n_jobs)
        df_val_hc, X_val_proc = process_split_features(X_val, track=track, n_jobs=n_jobs)
        df_test_hc, X_test_proc = process_split_features(X_test, track=track, n_jobs=n_jobs)
        
        # (b) Pixel PCA Extractor (Representation A: 20 components)
        print(f"\n[B] Pixel PCA Subspace Learning ({track}):")
        # For crop32, downsample size is (16, 16); for full96, (24, 24)
        target_size = (16, 16) if track == "crop32" else (24, 24)
        pca_extractor = PixelPCAExtractor(n_components=20, target_size=target_size, random_state=random_state)
        
        # Fit PCA STRICTLY on training images (Zero Leakage!)
        pca_extractor.fit(X_train_proc)
        
        # Transform Train, Val, Test
        print("  -> Projecting Train, Val, Test onto learned PCA basis...")
        pca_train = pca_extractor.transform(X_train_proc)
        pca_val = pca_extractor.transform(X_val_proc)
        pca_test = pca_extractor.transform(X_test_proc)
        
        df_train_pca = pd.DataFrame(pca_train, columns=pca_extractor.feature_names)
        df_val_pca = pd.DataFrame(pca_val, columns=pca_extractor.feature_names)
        df_test_pca = pd.DataFrame(pca_test, columns=pca_extractor.feature_names)
        
        # (c) Merge Handcrafted + PCA (Total 59 features)
        df_train_raw = pd.concat([df_train_hc, df_train_pca], axis=1)
        df_val_raw = pd.concat([df_val_hc, df_val_pca], axis=1)
        df_test_raw = pd.concat([df_test_hc, df_test_pca], axis=1)
        
        feature_cols = df_train_raw.columns.tolist()
        print(f"  -> Total fused feature dimensions: {len(feature_cols)} (39 Handcrafted + 20 PCA)")
        
        # (d) Standard Scaling (Z-score normalization)
        # Fit scaler STRICTLY on Train (Zero Leakage!)
        print(f"\n[C] Fitting StandardScaler strictly on [{track}] training split...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(df_train_raw)
        X_val_scaled = scaler.transform(df_val_raw)
        X_test_scaled = scaler.transform(df_test_raw)
        
        # Convert back to clean DataFrames
        df_train_final = pd.DataFrame(X_train_scaled, columns=feature_cols)
        df_train_final["label"] = y_train
        
        df_val_final = pd.DataFrame(X_val_scaled, columns=feature_cols)
        df_val_final["label"] = y_val
        
        df_test_final = pd.DataFrame(X_test_scaled, columns=feature_cols)
        df_test_final["label"] = y_test
        
        # (e) Verify Integrity
        assert not df_train_final.isna().any().any(), f"NaN detected in {track} train!"
        assert not df_val_final.isna().any().any(), f"NaN detected in {track} val!"
        assert not df_test_final.isna().any().any(), f"NaN detected in {track} test!"
        
        # (f) Save to Disk in outputs/features/
        print(f"\n[D] Saving [{track}] datasets and models to outputs/features/...")
        train_path = os.path.join(FEATURES_DIR, f"{track}_train.csv")
        val_path = os.path.join(FEATURES_DIR, f"{track}_val.csv")
        test_path = os.path.join(FEATURES_DIR, f"{track}_test.csv")
        
        df_train_final.to_csv(train_path, index=False)
        df_val_final.to_csv(val_path, index=False)
        df_test_final.to_csv(test_path, index=False)
        
        # Also save fitted scaler and PCA models for downstream inference
        scaler_path = os.path.join(FEATURES_DIR, f"scaler_{track}.pkl")
        pca_path = os.path.join(FEATURES_DIR, f"pca_{track}.pkl")
        joblib.dump(scaler, scaler_path)
        joblib.dump(pca_extractor, pca_path)
        
        print(f"  [Saved] {train_path} ({df_train_final.shape})")
        print(f"  [Saved] {val_path} ({df_val_final.shape})")
        print(f"  [Saved] {test_path} ({df_test_final.shape})")
        print(f"  [Saved] Scaler -> {scaler_path}")
        print(f"  [Saved] PCA Model -> {pca_path}")

    t_total = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"DUAL-TRACK PIPELINE COMPLETE! Total Execution Time: {t_total:.1f}s ({t_total/60:.2f} minutes)")
    print("Saved files in outputs/features/:")
    for f in sorted(os.listdir(FEATURES_DIR)):
        fpath = os.path.join(FEATURES_DIR, f)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  - {f:25s} ({size_mb:.2f} MB)")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCam Dual-Track Feature Extraction Pipeline")
    parser.add_argument("--train_samples", type=int, default=10000, help="Number of training samples (default: 10,000)")
    parser.add_argument("--val_samples", type=int, default=2500, help="Number of validation samples (default: 2,500)")
    parser.add_argument("--test_samples", type=int, default=2500, help="Number of test samples (default: 2,500)")
    parser.add_argument("--n_jobs", type=int, default=12, help="Parallel CPU workers (default: 12)")
    args = parser.parse_args()
    
    run_dual_track_pipeline(
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples,
        n_jobs=args.n_jobs
    )

