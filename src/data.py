"""
Data loading and preprocessing utilities for the PatchCamelyon (PCam) dataset.
Supports both fast loading from preprocessed PNG/CSV splits and raw parquet streams.
ES 335 - Machine Learning Course Project.
"""

import os
import io
import cv2
import numpy as np
import pandas as pd
from PIL import Image

REPO_ID = "1aurent/PatchCamelyon"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_RAW_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "shared_data"))

PUBLISHED_SPLIT_SIZES = {
    "train": 262144,
    "valid": 32768,
    "validation": 32768,
    "test": 32768
}

def normalize_split_name(split):
    split_lower = split.lower()
    if split_lower in ["valid", "validation", "val"]:
        return "validation"
    return split_lower

def decode_image_bytes(img_bytes):
    """
    Decodes raw image bytes to a (96, 96, 3) uint8 RGB array using OpenCV (with PIL fallback).
    """
    arr_bgr = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr_bgr is None:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return np.array(img, dtype=np.uint8)
    return cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)

def extract_central_crop(images):
    """
    Extracts the central 32x32 diagnostic focus region from 96x96 image patches.
    Tumor ground truth is determined by whether the central 32x32 contains >=1 metastatic pixel.
    
    Args:
        images: np.ndarray of shape (N, 96, 96, 3) or (96, 96, 3)
    Returns:
        np.ndarray of shape (N, 32, 32, 3) or (32, 32, 3)
    """
    if images.ndim == 4:
        return images[:, 32:64, 32:64, :]
    elif images.ndim == 3:
        return images[32:64, 32:64, :]
    else:
        raise ValueError(f"Expected 3D or 4D array, got shape {images.shape}")

def load_pcam_split(split="validation", max_samples=None, random_state=42, shuffle=False):
    """
    Loads images and labels for a specific split.
    Prioritizes loading from preprocessed PNG images and CSV label tables.
    
    Args:
        split: 'train', 'validation' (or 'valid'), 'test'
        max_samples: Optional int, limits number of loaded samples
        random_state: Int, random seed for reproducible subsampling
        shuffle: Whether to shuffle before returning
        
    Returns:
        X: np.ndarray of shape (N, 96, 96, 3), dtype uint8 RGB
        y: np.ndarray of shape (N,), dtype uint8 (0 or 1)
    """
    norm_split = normalize_split_name(split)
    split_dir = os.path.join(DEFAULT_DATA_DIR, norm_split)
    csv_path = os.path.join(split_dir, f"{norm_split}_labels.csv")
    
    # 1. Load from formatted CSV & PNG images if available
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        if shuffle:
            df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        if max_samples is not None:
            df = df.iloc[:max_samples]
            
        images_list = []
        for _, row in df.iterrows():
            img_rel = row["filename"]
            img_full = os.path.join(split_dir, img_rel)
            bgr = cv2.imread(img_full)
            if bgr is None:
                raise FileNotFoundError(f"Image not found: {img_full}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            images_list.append(rgb)
            
        X = np.stack(images_list, axis=0)
        y = df["label"].to_numpy(dtype=np.uint8)
        return X, y

    # 2. Fallback to raw parquet shards in shared_data
    import pyarrow.parquet as pq
    from src.preprocess_data import get_split_parquet_files, find_raw_dir
    raw_dir = find_raw_dir()
    pfiles = get_split_parquet_files(raw_dir, norm_split)
    
    images_list = []
    labels_list = []
    total_loaded = 0
    
    for pfile in pfiles:
        table = pq.read_table(pfile)
        img_col = table["image"]
        lbl_col = table["label"]
        for i in range(len(table)):
            img_bytes = img_col[i]["bytes"].as_py()
            img_arr = decode_image_bytes(img_bytes)
            lbl = int(lbl_col[i].as_py())
            images_list.append(img_arr)
            labels_list.append(lbl)
            total_loaded += 1
            if max_samples is not None and not shuffle and total_loaded >= max_samples:
                break
        if max_samples is not None and not shuffle and total_loaded >= max_samples:
            break
            
    X = np.stack(images_list, axis=0)
    y = np.array(labels_list, dtype=np.uint8)
    
    if shuffle or (max_samples is not None and max_samples < len(X)):
        rng = np.random.default_rng(random_state)
        indices = np.arange(len(X))
        rng.shuffle(indices)
        if max_samples is not None:
            indices = indices[:max_samples]
        X = X[indices]
        y = y[indices]
        
    return X, y

def get_split_stats(split="validation"):
    """
    Returns sample count and class balance for a split.
    """
    norm_split = normalize_split_name(split)
    csv_path = os.path.join(DEFAULT_DATA_DIR, norm_split, f"{norm_split}_labels.csv")
    
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path, usecols=["label"])
        total = len(df)
        positives = int((df["label"] == 1).sum())
        negatives = int((df["label"] == 0).sum())
        pos_pct = (positives / total) * 100 if total > 0 else 0
        return {
            "split": norm_split,
            "total_samples": total,
            "positives": positives,
            "negatives": negatives,
            "positive_rate_pct": round(pos_pct, 2)
        }
        
    # Parquet fallback
    import pyarrow.parquet as pq
    from src.preprocess_data import get_split_parquet_files, find_raw_dir
    raw_dir = find_raw_dir()
    pfiles = get_split_parquet_files(raw_dir, norm_split)
    labels = []
    for f in pfiles:
        table = pq.read_table(f, columns=["label"])
        labels.extend([int(v.as_py()) for v in table["label"]])
    labels = np.array(labels, dtype=np.uint8)
    total = len(labels)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    pos_pct = (positives / total) * 100 if total > 0 else 0
    return {
        "split": norm_split,
        "total_samples": total,
        "positives": positives,
        "negatives": negatives,
        "positive_rate_pct": round(pos_pct, 2)
    }
