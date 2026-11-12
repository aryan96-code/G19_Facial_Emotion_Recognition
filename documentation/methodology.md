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
| Max epochs | 100 | With early stopping |
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
3. Apply `preprocess_image()` from `src/preprocessing.py`.
4. Pass through loaded model.
5. Return top prediction and softmax probabilities.
