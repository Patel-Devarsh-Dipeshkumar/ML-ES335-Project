"""
Data Preprocessing Pipeline: Converts raw parquet shards into standard
image files (.png) and transparent, human-readable CSV label tables.

ES 335 - Machine Learning Course Project.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import cv2
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_RAW_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "shared_data"))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def find_raw_dir(custom_path=None):
    candidates = [
        custom_path,
        DEFAULT_RAW_DIR,
        os.path.join(PROJECT_ROOT, "data", "raw_parquet"),
        os.path.join(PROJECT_ROOT, "shared_data")
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            # check if parquet files exist
            if any(f.endswith(".parquet") for f in os.listdir(c)):
                return os.path.abspath(c)
    raise FileNotFoundError(f"Could not locate raw parquet directory. Checked: {candidates}")

def get_split_parquet_files(raw_dir, split="train"):
    norm_split = "valid" if split.startswith("val") else split
    prefix = f"{norm_split}-"
    files = sorted([os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.startswith(prefix) and f.endswith(".parquet")])
    if not files:
        raise FileNotFoundError(f"No parquet shards found for split '{split}' (prefix '{prefix}') in {raw_dir}")
    return files

def extract_split(raw_dir, split_name, target_data_dir, max_samples=None, random_state=42):
    """
    Extracts images and builds a clean CSV table for a specific split.
    Guarantees balanced classes (50% tumor, 50% normal).
    """
    split_dir = os.path.join(target_data_dir, split_name)
    img_dir = os.path.join(split_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    csv_path = os.path.join(split_dir, f"{split_name}_labels.csv")
    parquet_files = get_split_parquet_files(raw_dir, split_name)
    
    print(f"\n[Preprocessing] Extracting '{split_name}' from {len(parquet_files)} parquet shards...")
    
    # Calculate target counts per class
    target_pos = (max_samples // 2) if max_samples is not None else None
    target_neg = (max_samples - target_pos) if max_samples is not None else None
    
    pos_count = 0
    neg_count = 0
    records = []
    
    total_needed = max_samples if max_samples is not None else float("inf")
    pbar = tqdm(total=max_samples if max_samples is not None else 32768, desc=f"  Extracting {split_name}")
    
    for pfile in parquet_files:
        table = pq.read_table(pfile)
        n_rows = len(table)
        img_col = table["image"]
        lbl_col = table["label"]
        
        for i in range(n_rows):
            lbl = int(lbl_col[i].as_py())
            
            # Check balance limits
            if target_pos is not None and lbl == 1 and pos_count >= target_pos:
                continue
            if target_neg is not None and lbl == 0 and neg_count >= target_neg:
                continue
                
            img_bytes = img_col[i]["bytes"].as_py()
            img_arr = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            
            sample_idx = len(records)
            sample_id = f"{split_name}_{sample_idx:06d}"
            rel_img_path = os.path.join("images", f"{sample_id}.png").replace("\\", "/")
            abs_img_path = os.path.join(img_dir, f"{sample_id}.png")
            
            # Save standard PNG
            cv2.imwrite(abs_img_path, img_arr)
            
            records.append({
                "id": sample_id,
                "filename": rel_img_path,
                "label": lbl,
                "class_name": "Tumor" if lbl == 1 else "Normal",
                "width": img_arr.shape[1],
                "height": img_arr.shape[0],
                "channels": img_arr.shape[2]
            })
            
            if lbl == 1:
                pos_count += 1
            else:
                neg_count += 1
                
            pbar.update(1)
            
            if len(records) >= total_needed:
                break
                
        if len(records) >= total_needed:
            break
            
    pbar.close()
    
    df = pd.DataFrame(records)
    # Shuffle deterministically to mix classes
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    
    print(f"  [OK] Saved {len(df):,} images to: {img_dir}")
    print(f"  [OK] Saved readable CSV to: {csv_path}")
    print(f"       Class distribution: {df['class_name'].value_counts().to_dict()}")
    return df

def run_preprocessing(raw_dir=None, target_data_dir=DEFAULT_DATA_DIR, n_train=10000, n_val=2500, n_test=2500):
    raw_dir = find_raw_dir(raw_dir)
    print("=" * 70)
    print("PCam Tumor Detection - Data Preprocessing & Formatting Pipeline")
    print("=" * 70)
    print(f"Source Parquet Root : {raw_dir}")
    print(f"Target Formatted Dir: {target_data_dir}")
    print(f"Export Plan:")
    print(f"  * Train     : {n_train:,} images + train_labels.csv")
    print(f"  * Validation: {n_val:,} images + validation_labels.csv")
    print(f"  * Test      : {n_test:,} images + test_labels.csv")
    print("=" * 70)
    
    df_train = extract_split(raw_dir, "train", target_data_dir, max_samples=n_train)
    df_val = extract_split(raw_dir, "validation", target_data_dir, max_samples=n_val)
    df_test = extract_split(raw_dir, "test", target_data_dir, max_samples=n_test)
    
    print("\n" + "=" * 70)
    print("DATA PREPROCESSING COMPLETE")
    print(f"Total images saved across splits: {len(df_train) + len(df_val) + len(df_test):,}")
    print("All images are standard .png and all metadata is human-readable .csv")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCam Data Preprocessing and PNG/CSV Extraction")
    parser.add_argument("--raw_dir", type=str, default=None, help="Directory containing raw parquet files")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Target data directory")
    parser.add_argument("--n_train", type=int, default=10000, help="Number of training samples")
    parser.add_argument("--n_val", type=int, default=2500, help="Number of validation samples")
    parser.add_argument("--n_test", type=int, default=2500, help="Number of test samples")
    args = parser.parse_args()
    
    run_preprocessing(
        raw_dir=args.raw_dir,
        target_data_dir=args.data_dir,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test
    )
