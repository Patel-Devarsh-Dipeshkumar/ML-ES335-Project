"""
Phase 3: Feature Engineering and Extraction Module.
ES 335 - Machine Learning Course Project.

This module provides modular, vectorized feature extractors for histopathological patches:
  1. Color Moments: RGB, HSV, and CIE-Lab channel means and standard deviations (18 features).
  2. GLCM Texture Moments: Haralick contrast, dissimilarity, homogeneity, energy, correlation, and ASM (6 features).
  3. LBP Micro-Texture: Local Binary Patterns uniform histogram bins and Shannon entropy (11 features).
  4. Edge & Sharpness: Sobel gradient magnitude moments, edge density, and Laplacian variance (4 features).
  5. Raw Pixel PCA: Unsupervised dimensionality reduction on downsampled raw pixels (20 features).
  
Total Handcrafted Features: 39
Total Combined with PCA: 59
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from scipy.stats import entropy
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.decomposition import PCA


# ==============================================================================
# BLOCK 1: COLOR STATISTICS & STAIN UPTAKE (18 Features)
# ==============================================================================
def extract_color_features(img_rgb):
    """
    Extracts 18 color moments across RGB, HSV, and CIE-Lab color spaces.
    
    Biological Rationale:
      - Hematoxylin binds to acidic nucleic acids in cell nuclei, turning them purple
        and strongly absorbing green light (EDA Step 3 showed delta G = -18.2).
      - Eosin stains cytoplasmic proteins light pink.
      - Measuring channel means and standard deviations captures stain uptake
        and color variance mathematically.
        
    Args:
      img_rgb (np.ndarray): Image array of shape (H, W, 3), uint8 [0, 255].
      
    Returns:
      features (list of float): 18 color statistics.
      feature_names (list of str): Names for the 18 features.
    """
    # 1. Convert to HSV and CIE-Lab color spaces
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
    
    features = []
    feature_names = []
    
    # 2. RGB Channel Moments (6 features)
    for i, ch_name in enumerate(["R", "G", "B"]):
        ch = img_rgb[:, :, i]
        features.append(float(np.mean(ch)))
        features.append(float(np.std(ch)))
        feature_names.extend([f"color_{ch_name}_mean", f"color_{ch_name}_std"])
        
    # 3. HSV Channel Moments (6 features)
    for i, ch_name in enumerate(["H", "S", "V"]):
        ch = img_hsv[:, :, i]
        features.append(float(np.mean(ch)))
        features.append(float(np.std(ch)))
        feature_names.extend([f"color_{ch_name}_mean", f"color_{ch_name}_std"])
        
    # 4. CIE-Lab Channel Moments (6 features)
    for i, ch_name in enumerate(["L", "a", "b"]):
        ch = img_lab[:, :, i]
        features.append(float(np.mean(ch)))
        features.append(float(np.std(ch)))
        feature_names.extend([f"color_{ch_name}_mean", f"color_{ch_name}_std"])
        
    return features, feature_names


# ==============================================================================
# BLOCK 2: GLCM TEXTURE MOMENTS (6 Features)
# ==============================================================================
def extract_glcm_features(img_rgb):
    """
    Extracts 6 Haralick texture moments using Gray-Level Co-occurrence Matrix (GLCM).
    
    Biological Rationale:
      - Normal lymph node tissue has smooth connective collagen bands (high homogeneity).
      - Tumor tissue contains chaotic, crowded malignant cells with rapid brightness
        fluctuations (high contrast and dissimilarity).
        
    Args:
      img_rgb (np.ndarray): Image array of shape (H, W, 3), uint8.
      
    Returns:
      features (list of float): 6 Haralick texture moments.
      feature_names (list of str): Names for the 6 features.
    """
    # 1. Convert to grayscale (texture is color-blind)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # 2. Compute GLCM across 4 standard directions (0, 45, 90, 135 degrees)
    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True
    )
    
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    features = []
    feature_names = []
    
    # 3. Average across all 4 angles for rotation-invariant texture
    for prop in props:
        val = float(graycoprops(glcm, prop).mean())
        features.append(val)
        feature_names.append(f"glcm_{prop.lower()}")
        
    return features, feature_names


# ==============================================================================
# BLOCK 3: LBP MICRO-TEXTURE & NUCLEAR CROWDING (11 Features)
# ==============================================================================
def extract_lbp_features(img_rgb, radius=1, n_points=8):
    """
    Extracts 11 Local Binary Pattern (LBP) micro-texture features.
    
    Biological Rationale:
      - Compares each pixel with its 8 immediate neighbors to detect local micro-shapes
        (flat regions, curves, edges, and corners).
      - Bins 0-8 capture orderly geometric shapes.
      - Bin 9 captures non-uniform chaotic patterns (nuclear crowding).
      - Shannon entropy measures the overall spatial disorder of the cell architecture.
      
    Args:
      img_rgb (np.ndarray): Image array of shape (H, W, 3), uint8.
      radius (int): Pixel radius for neighborhood comparison (default = 1).
      n_points (int): Number of neighbor points on the circle (default = 8).
      
    Returns:
      features (list of float): 10 normalized histogram bins + 1 Shannon entropy value.
      feature_names (list of str): Names for the 11 features.
    """
    # 1. Convert to grayscale
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # 2. Compute uniform LBP
    lbp = local_binary_pattern(gray, P=n_points, R=radius, method="uniform")
    
    # 3. Compute normalized histogram across 10 pattern buckets
    n_bins = n_points + 2  # 10 bins for P=8 uniform
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    features = [float(v) for v in hist]
    feature_names = [f"lbp_bin_{i}" for i in range(n_bins)]
    
    # 4. Compute Shannon Entropy
    shannon_entropy = float(entropy(hist + 1e-7))
    features.append(shannon_entropy)
    feature_names.append("lbp_entropy")
    
    return features, feature_names


# ==============================================================================
# BLOCK 4: EDGE STRENGTH & SHARPNESS (4 Features)
# ==============================================================================
def extract_edge_features(img_rgb):
    """
    Extracts 4 structural edge magnitude, density, and sharpness features.
    
    Biological Rationale:
      - Normal tissue has soft collagen gradients and clear, smooth fat vacuoles.
      - Tumor tissue has thousands of packed nuclei, each forming a high-contrast boundary.
      - Sobel magnitude measures overall edge power; Laplacian variance measures structural sharpness.
      
    Args:
      img_rgb (np.ndarray): Image array of shape (H, W, 3), uint8.
      
    Returns:
      features (list of float): 4 edge metrics.
      feature_names (list of str): Names for the 4 features.
    """
    # 1. Convert to grayscale
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # 2. Compute Sobel gradients in X and Y
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # 3. Compute Laplacian for edge sharpness
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # 4. Extract metrics
    sobel_mean = float(np.mean(magnitude))
    sobel_std = float(np.std(magnitude))
    edge_density = float(np.mean(magnitude > 50))  # Fraction of pixels with strong edge
    laplacian_var = float(np.var(laplacian))        # Structural sharpness
    
    features = [sobel_mean, sobel_std, edge_density, laplacian_var]
    feature_names = ["edge_sobel_mean", "edge_sobel_std", "edge_density", "edge_laplacian_var"]
    
    return features, feature_names


# ==============================================================================
# MASTER HANDCRAFTED EXTRACTOR (Blocks 1 to 4 -> 39 Features)
# ==============================================================================
def extract_handcrafted_features(img_rgb):
    """
    Extracts all 39 handcrafted biological features for a single patch:
      - 18 Color Moments (RGB, HSV, CIE-Lab)
      - 6 GLCM Texture Moments
      - 11 LBP Micro-Texture Features
      - 4 Edge & Sharpness Features
      
    Args:
      img_rgb (np.ndarray): Image array of shape (H, W, 3), uint8.
      
    Returns:
      features (list of float): 39 feature values.
      feature_names (list of str): 39 feature names.
    """
    color_feats, color_names = extract_color_features(img_rgb)
    glcm_feats, glcm_names = extract_glcm_features(img_rgb)
    lbp_feats, lbp_names = extract_lbp_features(img_rgb)
    edge_feats, edge_names = extract_edge_features(img_rgb)
    
    all_feats = color_feats + glcm_feats + lbp_feats + edge_feats
    all_names = color_names + glcm_names + lbp_names + edge_names
    
    return all_feats, all_names


# ==============================================================================
# BLOCK 5: RAW PIXEL DOWNSAMPLING & PCA EXTRACTOR (20 Features)
# ==============================================================================
def downsample_patch(img_rgb, target_size=(24, 24)):
    """
    Downsamples an image patch to target_size (e.g. 24x24) using bilinear interpolation.
    Averages out individual pixel sensor noise while preserving macroscopic tissue structure.
    
    Args:
      img_rgb (np.ndarray): Image array of shape (H, W, 3), uint8.
      target_size (tuple): (width, height) for downsampling.
      
    Returns:
      flat_vector (np.ndarray): 1D float32 array of shape (target_size[0] * target_size[1] * 3,).
    """
    resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32).ravel() / 255.0


class PixelPCAExtractor:
    """
    Unsupervised raw pixel PCA extractor (Representation A).
    Learns 20 principal components strictly from training patches.
    """
    def __init__(self, n_components=20, target_size=(24, 24), random_state=42):
        self.n_components = n_components
        self.target_size = target_size
        self.random_state = random_state
        self.pca = PCA(n_components=n_components, svd_solver="randomized", random_state=random_state)
        self.is_fitted = False
        self.feature_names = [f"pca_pixel_pc_{i+1}" for i in range(n_components)]
        
    def fit(self, X_train_images):
        """
        Fits PCA strictly on training patches.
        """
        print(f"Fitting Pixel PCA on {len(X_train_images):,} training patches (downsampled to {self.target_size})...")
        X_flat = np.array([downsample_patch(img, self.target_size) for img in X_train_images], dtype=np.float32)
        self.pca.fit(X_flat)
        self.is_fitted = True
        explained_var = np.sum(self.pca.explained_variance_ratio_) * 100
        print(f"[OK] Pixel PCA successfully fitted! Top {self.n_components} components explain {explained_var:.1f}% variance.")
        return self
        
    def transform(self, images):
        """
        Projects images onto the learned 20 PCA components.
        """
        assert self.is_fitted, "PixelPCAExtractor must be fitted on training data before transforming!"
        X_flat = np.array([downsample_patch(img, self.target_size) for img in images], dtype=np.float32)
        return self.pca.transform(X_flat)
