# Methodology
## G19 — Facial Emotion Recognition | CSE276

---

## 1. Data Collection & Splitting
- Dataset: **FER-2013** (publicly available on Kaggle).
- Download instructions: See `README.md`.
- The original FER-2013 `Usage` splits are preserved exactly:
  - `Training` → model training
  - `PublicTest` → validation, hyperparameter selection, and early stopping
  - `PrivateTest` → final held-out testing **only** (never used for tuning or early stopping)

## 2. Exploratory Data Analysis (EDA)
- Visualise class distributions to identify imbalance.
- Inspect sample images per class.
- Analyse pixel intensity distributions.
- Document findings in `notebooks/02_dataset_exploration.ipynb`.

## 3. Preprocessing Pipeline
All preprocessing steps are implemented in `src/preprocessing.py`:

| Step | Technique | Mode | Reason |
|---|---|---|---|
| Colour → Greyscale | `cv2.cvtColor` | Baseline & Experiment | FER images are greyscale; RGB wastes computation |
| Resize | `cv2.resize` to 48 × 48 | Baseline & Experiment | Standardise input dimensions to `(48, 48, 1)` |
| Contrast enhancement | CLAHE (`use_clahe=True`) | **Optional experiment only** | Baseline trains **without** CLAHE so both can be compared fairly |
| Normalisation | Divide by 255 → `[0, 1]` | Baseline & Experiment | Stabilises gradient descent |

## 4. Data Augmentation
Applied **online** via Keras preprocessing layers **only during training** (`training=True`), never to validation (`PublicTest`) or test (`PrivateTest`) data:

| Augmentation | Range | Motivation |
|---|---|---|
| Horizontal flip | 50% probability | Faces are approximately symmetric |
| Random rotation | ±10° (`factor=10/360`) | Account for head tilt |
| Random zoom | ±10% | Varied camera distances |
| Brightness jitter | ±15% (`value_range=(0.0, 1.0)`) | Varied lighting on `[0, 1]` normalised inputs |

## 5. Model Architecture — CNN
See `src/model.py` for the full architecture.

```
Input (48 × 48 × 1)
  → [Data Augmentation]
  → Block 1: Conv(32) → BN → ReLU → Conv(32) → BN → ReLU → MaxPool → Dropout(0.25)
  → Block 2: Conv(64) → BN → ReLU → Conv(64) → BN → ReLU → MaxPool → Dropout(0.25)
  → Block 3: Conv(128) → BN → ReLU → Conv(128) → BN → ReLU → MaxPool → Dropout(0.25)
  → GlobalAveragePooling2D
  → Dense(256) → ReLU → Dropout(0.5)
  → Dense(7) → Softmax
```

## 6. Training Strategy
| Hyperparameter | Value | Justification |
|---|---|---|
| Optimiser | Adam | Adaptive learning rate, widely effective |
| Initial LR | 1e-3 | Standard Adam default |
| Loss | Sparse Categorical Cross-Entropy | Multi-class classification |
| Batch size | 64 | Balance of GPU memory and gradient stability |
| Max epochs | 40 | With `EarlyStopping(patience=10, restore_best_weights=True)` |
| Early stopping | patience=10 | Prevent overfitting |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=5) | Fine-tune convergence |

## 7. Evaluation Metrics
| Metric | Use |
|---|---|
| Accuracy | Overall performance (top-line) |
| Precision / Recall / F1 | Per-class, especially for imbalanced classes |
| Confusion Matrix | Understand which emotions are confused with which |
| Training curves | Diagnose overfitting / underfitting |

## 8. Inference Pipeline
1. Capture / upload image.
2. Detect face region (optional: OpenCV Haar cascade).
3. Apply `preprocess_image(image, use_clahe=False)` from `src/preprocessing.py` (matching baseline training).
4. Pass through loaded model.
5. Return top prediction and softmax probabilities.

---

## 9. Baseline Experimental Results (Actual Run — No CLAHE)

### Training & Validation (`Training` = 28,709 | `PublicTest` = 3,589)
| Setting / Metric | Observed Value |
|---|---|
| Architecture | 3-block CNN (`src/model.py`) |
| Preprocessing | Greyscale 48×48, `[0, 1]` normalisation, **no CLAHE** |
| Batch size | 64 |
| Epochs completed | 40 / 40 |
| Best epoch (by `val_loss`) | 40 |
| Training accuracy | **58.68%** |
| Validation accuracy (`PublicTest`) | **58.46%** |
| Training loss | **1.0505** |
| Validation loss (`PublicTest`) | **1.0914** |

> Training and validation metrics remain closely aligned (0.22% accuracy gap at epoch 40), indicating no severe overfitting under training-only augmentation and dropout regularisation.

### Final Held-Out Evaluation (`PrivateTest` = 3,589)
Saved in `results/private_test_classification_report.csv`:

| Metric | Value |
|---|---|
| **Test Accuracy** | **60.96%** |
| Macro Precision | 0.5897 |
| Macro Recall | 0.5905 |
| Macro F1 | 0.5771 |
| Weighted F1 | 0.5948 |

#### Per-Class Breakdown (`PrivateTest`)
| Emotion | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Angry | 0.6196 | 0.4644 | 0.5308 | 491 |
| Disgust | 0.5789 | 0.6000 | 0.5893 | 55 |
| Fear | 0.4348 | 0.2462 | 0.3144 | 528 |
| Happy | 0.8499 | 0.8373 | 0.8436 | 879 |
| Sad | 0.4901 | 0.3754 | 0.4252 | 594 |
| Surprise | 0.6754 | 0.8101 | 0.7366 | 416 |
| Neutral | 0.4794 | 0.8003 | 0.5996 | 626 |
