# Comparing Machine Learning Approaches for Tumour Detection in Tissue Patches

[![Course: ES 335](https://img.shields.io/badge/Course-ES%20335%20(Machine%20Learning)-blue.svg)](https://www.iitgn.ac.in/)
[![Institute: IIT Gandhinagar](https://img.shields.io/badge/IIT%20Gandhinagar-Palaj%2C%20Gujarat-orange.svg)](https://www.iitgn.ac.in/)
[![Dataset: PatchCamelyon](https://img.shields.io/badge/Dataset-PatchCamelyon%20(PCam)-green.svg)](https://github.com/basveeling/pcam)
[![Python: 3.13+](https://img.shields.io/badge/Python-3.13%2B-brightgreen.svg)](https://www.python.org/)

Course project for **ES 335: Machine Learning** at the **Indian Institute of Technology Gandhinagar (IITGN)**.

* **Devarsh Patel** (Roll No: 24110245) — `devarsh.patel@iitgn.ac.in`
* **Dhrumil Bhutaiya** (Roll No: 24110075) — `dhrumil.bhutaiya@iitgn.ac.in`

---

## Problem & Clinical Motivation

When breast cancer metastasizes, malignant cells spread through lymphatic channels and lodge into the axillary sentinel lymph nodes. Pathologists examine these lymph nodes under optical microscopes to determine whether cancer has spread, which dictates whether a patient needs chemotherapy and lymph node clearance surgery.

However, searching through gigapixel Whole-Slide Images (WSIs) by hand is exhausting. Small focal micro-metastases (containing as few as 10 to 20 cancer cells) can easily be overlooked during manual screening.

The goal of this project is to build and evaluate **classical statistical machine learning and ensemble approaches** on the **PatchCamelyon (PCam)** histopathology benchmark, testing whether biologically engineered color, texture, and edge representations can reliably detect metastatic tissue.

---

## The Central 32×32 Diagnostic Ground Truth Rule

Every tissue patch in the dataset is 96×96 pixels in size (scanned at 10× magnification, 0.97 µm/px).

However, the ground truth clinical label is governed **strictly by the central 32×32 window**:
* **Label = 1 (Tumour):** Contains at least 1 pixel of metastatic carcinoma inside the central 32×32 box.
* **Label = 0 (Normal):** Contains zero tumor pixels inside the central 32×32 box (even if tumor tissue is present in the outer surrounding border).

```
   ┌───────────────────────────────────┐
   │ 96 x 96 Histological Context      │
   │                                   │
   │        ┌─────────────────┐        │
   │        │ Central 32 x 32 │        │
   │        │ Diagnostic Field│        │
   │        │ (Ground Truth)  │        │
   │        └─────────────────┘        │
   │                                   │
   └───────────────────────────────────┘
```

We evaluate two distinct pipelines:
* **Track A (Full 96×96):** Uses full spatial context to capture macroscopic tissue architecture.
* **Track B (Crop 32×32):** Crops strictly to the ground-truth region to avoid border noise.

---

## 59-Dimensional Multimodal Feature Pipeline

Rather than treating images as raw pixel vectors, we engineered **59 biologically grounded histological features**:

| Feature Family | Dimension | Biological Purpose & Target Morphology |
| :--- | :---: | :--- |
| **Color Moments** | 18 | Mean and standard deviation across RGB, HSV, and CIE-Lab color channels. Measures dark purple basophilic Hematoxylin staining (DNA/nuclei) vs. pink Eosin staining (stroma). |
| **GLCM Texture** | 6 | Contrast, dissimilarity, homogeneity, energy, correlation, and ASM from Gray-Level Co-occurrence Matrices. Measures spatial chromatin disorder. |
| **Local Binary Patterns (LBP)** | 11 | Uniform rotation-invariant micro-patterns (P=8, R=1). Captures fine cellular borders and membrane atypicality. |
| **Edge Gradients** | 4 | Sobel gradient mean, Sobel standard deviation, edge density, and Laplacian variance. Measures nuclear crowding and cell packing density. |
| **Bilinear Pixel PCA** | 20 | Dominant orthogonal luminance eigenvectors capturing macro-level spatial architecture. |

---

## Final Test Benchmark Leaderboard (5,000 Held-Out Patches)

All models were tuned strictly on training and validation splits. The held-out test cohort (N=5,000, exactly 2,500 normal and 2,500 tumor patches) was evaluated **strictly once in a sealed one-shot pass**:

| Rank | Model Architecture | Test Accuracy | ROC-AUC | PR-AUC | Sensitivity | Specificity | Precision | F1-Score | Brier Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **ResNet-18 (Deep CNN Benchmark)** | **84.94%** | **0.9392** | **0.9455** | **75.04%** | **94.84%** | **93.57%** | **0.8329** | **0.1168** |
| 2 | **Hybrid Multi-Paradigm Blend (50/50 Champion)** | 79.76% | 0.8853 | 0.8767 | 73.08% | 86.44% | 84.35% | 0.7831 | 0.1426 |
| 3 | Hybrid Blend (70/30 RBF/HistGBM) | 79.56% | 0.8804 | 0.8740 | 72.32% | 86.80% | 84.57% | 0.7796 | 0.1453 |
| 4 | HistGradientBoosting | 78.82% | 0.8845 | 0.8724 | 71.84% | 85.80% | 83.50% | 0.7723 | 0.1460 |
| 5 | Default RBF Kernel SVM | 78.80% | 0.8746 | 0.8681 | 73.28% | 84.32% | 82.37% | 0.7756 | 0.1493 |
| 6 | Tuned Random Forest (150 Trees) | 78.46% | 0.8863 | 0.8714 | 68.76% | 88.16% | 85.31% | 0.7615 | 0.1467 |
| 7 | Tuned RBF Kernel SVM (C=3.0) | 77.92% | 0.8637 | 0.8626 | 70.88% | 84.96% | 82.50% | 0.7625 | 0.1535 |
| 8 | Logistic Regression Baseline | 75.84% | 0.8373 | 0.8284 | 70.96% | 80.72% | 78.63% | 0.7460 | 0.1679 |
| 9 | Linear SVM (Dual C=1.0) | 75.46% | 0.8292 | 0.8226 | 71.12% | 79.80% | 77.88% | 0.7435 | 0.1850 |

*Statistical significance verified using 1,000-resample bootstrap 95% confidence intervals (p < 10^-7).*

---

## Key Empirical Discoveries

1. **The Linear Capacity Ceiling (~75.8%):** Both Logistic Regression and Linear SVM saturated at ~75.8% test accuracy (matching each other within 2 patches), proving that flat hyperplanes cannot untangle complex histological tissues.
2. **Generalization Gap Stability:** Tree ensembles generalized with remarkable stability:
   * **Random Forest:** dropped by only **0.98%** between validation and test splits (79.44% → 78.46%).
   * **HistGradientBoosting:** dropped by only **1.84%** (80.66% → 78.82%).
   * **High-C RBF SVM:** suffered from validation sample variance, dropping by **4.44%** (82.36% → 77.92%).
3. **The Multi-Paradigm Champion (79.76%):** Blending the continuous distance geometry of RBF SVM with the orthogonal partitioning of HistGradientBoosting achieved the lowest calibration error (Brier score 0.1426) and #1 classical test accuracy.
4. **Breaking the Classical Feature Ceiling (ResNet-18: 84.94% Test Accuracy, 0.9392 ROC-AUC):** Fine-tuning ResNet-18 with 8-fold dihedral rotational invariance augmentations broke decisively past the 79.8% classical ceiling. Crucially, false positive alarms dropped by 62% (from 339 to 129), driving precision to 93.57% and validating that hierarchical convolutional feature maps capture sub-cellular cytologic details that global summary statistics omit.

---

## The 4-Tier Pathological Failure Taxonomy

Across the 5,000 test patches, our champion ensemble made 1,009 misclassifications. Crucially, **613 of these (12.3% of the entire test cohort, >60.7% of all ensemble errors) were consensus failures shared across all model families simultaneously**.

This proves that errors are not caused by algorithmic failure, but by the **empirical representation limit of hand-engineered global features**:

| Failure Tier | Error Type | Percentage | Biological Mechanism |
| :--- | :---: | :---: | :--- |
| **Tier 1: Dense Benign Lymphoid Germinal Centers** | False Positive | 30.7% | Reactive lymphoid follicles contain tightly packed proliferating lymphocytes. This produces high edge density (d=+0.83) and intense hematoxylin dye absorption (d=+0.84) that global features cannot differentiate from invasive cancer sheets. |
| **Tier 2: Stroma-Dense Micrometastases** | False Negative | 38.7% | Infiltrating cancer incites desmoplasia, laying down collagenous connective tissue. When only 5–15 cancer cells occupy the central window, their localized biological signal is diluted by surrounding benign stroma during global feature pooling. |
| **Tier 3: Coagulative Tissue Necrosis & Ghost Cells** | False Negative | 15.9% | Rapidly growing tumors outgrow their blood supply, causing ischemic necrosis. Karyorrhexis and karyolysis dissolve cell nuclei into amorphous ghost cells, collapsing recognizable LBP texture codes. |
| **Tier 4: Adipose Vacuolation & Boundary Artifacts** | Both | 14.7% | Perinodal fat contains large white lipid vacuoles (>70% unstained patch area) and optical refractive edges that skew GLCM texture statistics. |

---

## Why Deep Learning (CNNs) is the Next Frontier

Our 613 consensus failures mathematically explain why Convolutional Neural Networks (CNNs) and Vision Transformers are necessary to surpass 80% accuracy:
* **Global Pooling vs. Local Receptive Fields:** Handcrafted features pool pixel statistics globally, so packed normal lymphocytes look identical to cancer cells.
* **Sub-Cellular Atypicality:** Human pathologists differentiate germinal centers from carcinoma under 40× oil immersion by inspecting nuclear pleomorphism (irregular nuclear shapes, clefts, and mitotic figures). Resolving these sub-cellular characteristics requires localized convolutional filters rather than global summary statistics.

---

## Repository Structure

```text
Github/
├── src/                          # Modular Python source code
│   ├── data.py                   # Split verification and batch loading
│   ├── eda.py                    # Exploratory data analysis scripts
│   ├── features.py               # 59-dim multimodal feature extraction
│   ├── feature_pipeline.py       # Unified feature engineering pipeline
│   ├── baseline_logreg.py        # Baseline Logistic Regression benchmark
│   ├── train_linear_svm.py       # Linear SVM benchmark
│   ├── train_random_forest.py    # Random Forest training
│   ├── train_gradient_boosting.py# HistGradientBoosting training
│   ├── train_rbf_svm.py          # RBF Kernel SVM training
│   ├── tune_ensemble.py          # Multi-paradigm ensemble blending
│   ├── train_cnn.py              # Deep Learning benchmark (ResNet-18)
│   ├── evaluate_test.py          # One-shot test evaluation & bootstrap CIs
│   ├── error_analysis.py         # Forensic autopsy & Cohen's d calculation
│   └── tracker.py                # Milestone progress PDF generator
├── docs/                         # Interactive diagnostic website (GitHub Pages ready)
│   ├── index.html                # Single-page diagnostic portal
│   ├── style.css                 # Academic stylesheet (light/dark mode)
│   ├── app.js                    # Client-side threshold simulator & studio
│   └── data/                     # Curated patch metadata & 5,000 test predictions
├── notebooks/                    # Interactive Jupyter notebooks
│   ├── 02_eda.ipynb              # Visual data exploration & distributions
│   ├── 03_features.ipynb         # Feature extraction & statistical ranking
│   ├── 04_baseline.ipynb         # Logistic Regression regularized sweeps
│   ├── 05_candidates.ipynb       # Candidate model exploration
│   ├── 06_tuning.ipynb           # Grid search & ensemble blending
│   ├── 07_test_evaluation.ipynb  # Sealed test evaluation & bootstrap CIs
│   ├── 08_error_analysis.ipynb   # Error cases autopsy & taxonomy
│   ├── 09_interactive_showcase.ipynb # Diagnostic simulator & web interface
│   └── 10_cnn_benchmark.ipynb    # Deep learning benchmark & training loop
├── results/                      # Experimental CSV benchmarks & metric tables
├── plots/                        # High-resolution publication figures & curves
├── report/                       # 37-page academic technical report (PDF & TeX)
├── requirements.txt              # Pinned Python package dependencies
├── .gitignore                    # Protection against large caches and data
└── README.md                     # Project documentation
```

---

## Reproduction & Setup

### 1. Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Feature Extraction
To run the 59-dimensional multimodal feature extraction:
```bash
python src/features.py
```

### 3. Model Training & Validation Tuning
Train candidate models and run the validation grid search:
```bash
python src/baseline_logreg.py
python src/train_linear_svm.py
python src/train_random_forest.py
python src/train_gradient_boosting.py
python src/train_rbf_svm.py
python src/tune_ensemble.py
```

### 4. Sealed Test Evaluation
Evaluate all models on the held-out test split with bootstrap confidence intervals:
```bash
python src/evaluate_test.py
```

### 5. Error Analysis & Pathology Taxonomy
Run the forensic error analysis:
```bash
python src/error_analysis.py
```

### 6. Deep Learning Benchmark (Milestone 10)
Train and evaluate ResNet-18 on raw histology patches:
```bash
python src/train_cnn.py
```
Or run `notebooks/10_cnn_benchmark.ipynb` directly on GPU.

---

## Authors & Contributions

* **Devarsh Patel** (24110245) &mdash; Multimodal Feature Pipeline Architecture, Candidate Model Optimization (RBF SVM, HistGBM), Multi-Paradigm Ensemble Blending, Cohen's d Feature Discrepancy & Pathological Taxonomy.
* **Dhrumil Bhutaiya** (24110075) &mdash; Exploratory Data Analysis, Baseline Linear Capacity Ceiling, Sealed Test Evaluation Harness, Bootstrap Confidence Intervals, Progress Tracking & Technical Report.
