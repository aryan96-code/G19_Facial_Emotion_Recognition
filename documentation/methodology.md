# Methodology & Experimental Design
## G19 — Facial Emotion Recognition | CSE276
### Submission Deadline: 25 November 2026

---

## 1. Dataset Collection & Strict Partition Isolation

### FER-2013 Benchmark Structure
The project utilizes the **FER-2013** dataset (Kaggle / ICML 2013 Challenges in Representation Learning). The dataset consists of 35,887 grayscale images of size $48 \times 48$ pixels, annotated across seven emotion classes:
0 = Angry, 1 = Disgust, 2 = Fear, 3 = Happy, 4 = Sad, 5 = Surprise, 6 = Neutral.

The official `Usage` column partitions are maintained strictly:
- **`Training` (28,709 images):** Used exclusively for parameter learning and backpropagation.
- **`PublicTest` (3,589 images):** Used exclusively for validation, learning rate scheduling, early stopping, and hyperparameter selection during ablation studies.
- **`PrivateTest` (3,589 images):** Held out strictly as an untouched test set for final benchmark reporting.

### Strict Data Isolation Protocol
In Experiment 02, data leakage was prevented by designing a dedicated data-loading function:
`load_fer2013_train_val(csv_path)` in `src/data_loader.py`.
- Reads only rows where `Usage == 'Training'` and `Usage == 'PublicTest'`.
- Never loads, parses, or accesses rows where `Usage == 'PrivateTest'`.
- Class weights are computed **strictly from the training partition** ($y_{train}$) to prevent validation distribution leakage.
- `PrivateTest` was evaluated only once per experiment after model training and hyperparameter selection were frozen.

---

## 2. Preprocessing & Online Augmentation

### Image Normalization
All inputs are formatted as follows in `src/preprocessing.py`:
- Converted to single-channel greyscale ($48 \times 48 \times 1$).
- Scaled from integer $[0, 255]$ to floating-point $[0.0, 1.0]$ (`np.float32`) by dividing by 255.0.
- **CLAHE Policy:** Contrast Limited Adaptive Histogram Equalization (CLAHE) was evaluated in exploratory experiments, but both the Baseline (Experiment 01) and the winning Improved Model (Experiment 02-E) were trained **without CLAHE**. CLAHE is kept as an optional user toggle in the Streamlit application for research comparison only.

### Online Data Augmentation
Augmentation layers are integrated directly into the Keras model graph. They execute on GPU during training (`training=True`) and automatically deactivate during validation and inference:

| Augmentation Layer | Hyperparameters | Purpose |
|:---|:---|:---|
| `RandomFlip` | `"horizontal"` (50% probability) | Natural left-right facial symmetry |
| `RandomRotation` | `factor = 10 / 360` ($\pm 10^\circ$) | Real-world head tilt |
| `RandomTranslation` | `height_factor = 0.08`, `width_factor = 0.08` ($\pm 8\%$) | Offset face positioning |
| `RandomZoom` | `height_factor = 0.10`, `width_factor = 0.10` ($\pm 10\%$) | Variations in camera distance |
| `RandomBrightness` | `factor = 0.15`, `value_range = (0.0, 1.0)` | Varying ambient lighting conditions |
| `RandomContrast` | `factor = 0.15`, `value_range = (0.0, 1.0)` | Sensor dynamic range variations |

---

## 3. Baseline Model (Experiment 01)

The baseline model provides an anchor for measuring improvements.

### Architecture (Experiment 01)
- **Input:** $(48, 48, 1)$
- **Block 1:** $2 \times \text{Conv2D}(32, 3\times3) \rightarrow \text{BatchNorm} \rightarrow \text{ReLU} \rightarrow \text{MaxPool2D}(2\times2) \rightarrow \text{Dropout}(0.25)$
- **Block 2:** $2 \times \text{Conv2D}(64, 3\times3) \rightarrow \text{BatchNorm} \rightarrow \text{ReLU} \rightarrow \text{MaxPool2D}(2\times2) \rightarrow \text{Dropout}(0.25)$
- **Block 3:** $2 \times \text{Conv2D}(128, 3\times3) \rightarrow \text{BatchNorm} \rightarrow \text{ReLU} \rightarrow \text{MaxPool2D}(2\times2) \rightarrow \text{Dropout}(0.25)$
- **Head:** $\text{GlobalAveragePooling2D} \rightarrow \text{Dense}(256) \rightarrow \text{ReLU} \rightarrow \text{Dropout}(0.50) \rightarrow \text{Dense}(7, \text{softmax})$
- **Total Parameters:** 431,655

### Baseline Results
- **Training Epochs:** 40 (best epoch: 40)
- **Training Loss / Accuracy:** 1.0505 / 58.68%
- **PublicTest Loss / Accuracy:** 1.0914 / 58.46%
- **PrivateTest Accuracy:** **60.96%**
- **PrivateTest Macro F1:** **57.71%**
- **PrivateTest Weighted F1:** **59.48%**

---

## 4. Improved CNN Architecture (Experiment 02)

To advance beyond the baseline, Experiment 02 expanded the model depth to four convolutional blocks and engineered a dual-pooling classification head.

### Four Convolutional Blocks
1. **Block 1:** $2 \times \text{Conv2D}(32, 3\times3, \text{padding='same'}) \rightarrow \text{BatchNorm} \rightarrow \text{ReLU} \rightarrow \text{MaxPool2D}(2\times2) \rightarrow \text{Dropout}(0.15)$
2. **Block 2:** $2 \times \text{Conv2D}(64, 3\times3, \text{padding='same'}) \rightarrow \text{BatchNorm} \rightarrow \text{ReLU} \rightarrow \text{MaxPool2D}(2\times2) \rightarrow \text{Dropout}(0.20)$
3. **Block 3:** $2 \times \text{Conv2D}(128, 3\times3, \text{padding='same'}) \rightarrow \text{BatchNorm} \rightarrow \text{ReLU} \rightarrow \text{MaxPool2D}(2\times2) \rightarrow \text{Dropout}(0.25)$
4. **Block 4:** $2 \times \text{Conv2D}(256, 3\times3, \text{padding='same'}) \rightarrow \text{BatchNorm} \rightarrow \text{ReLU} \rightarrow \text{MaxPool2D}(2\times2) \rightarrow \text{Dropout}(0.30)$

Spatial dimensions transition: $48\times48 \rightarrow 24\times24 \rightarrow 12\times12 \rightarrow 6\times6 \rightarrow 3\times3$ with 256 feature channels.

### Classification Head Comparison: Head A vs. Head B

#### Head A (Standard GAP)
$$\text{Output of Block 4} \rightarrow \text{GlobalAveragePooling2D} \rightarrow \text{Dense}(256) \rightarrow \text{BatchNorm} \rightarrow \text{ReLU} \rightarrow \text{Dropout}(0.50) \rightarrow \text{Dense}(7, \text{softmax})$$
- Generates a 256-dimensional feature vector.
- Total parameters: 1,243,815.

#### Head B (Dual Pooling — Selected)
$$\text{Output of Block 4} \rightarrow \begin{cases} \text{GlobalAveragePooling2D} \\ \text{GlobalMaxPooling2D} \end{cases} \rightarrow \text{Concatenate} \rightarrow \text{Dense}(256) \rightarrow \text{BatchNorm} \rightarrow \text{ReLU} \rightarrow \text{Dropout}(0.50) \rightarrow \text{Dense}(7, \text{softmax})$$
- Generates a 512-dimensional combined feature vector.
- Total parameters: 1,309,671.

#### Theoretical Justification for Head B
- **Global Average Pooling (GAP):** Computes the spatial mean across each feature map, capturing global spatial information across the receptive field.
- **Global Max Pooling (GMP):** Extracts the maximum activation across each feature map, capturing the strongest spatial activations present in the representation.
- **Concatenation:** Fusing GAP and GMP combines global spatial information with peak activation responses into a unified 512-dimensional representation without incurring the parameter growth of flattening.

---

## 5. Experiment 02 Ablation Study (`PublicTest` Validation)

Five variants were trained on Google Colab (T4 GPU) with Adam optimizer ($\text{lr} = 1\times 10^{-3}$), batch size 64, early stopping (patience 10), and ReduceLROnPlateau (factor 0.5, patience 5).

### PublicTest Validation Results

| Experiment Variant | Augmentation Strategy | Head Type | Class Weighting | Validation Accuracy (`PublicTest`) | Validation Macro F1 (`PublicTest`) | Selection Outcome |
|:---|:---|:---:|:---|:---:|:---:|:---|
| **02-A** | Baseline (Flip, Rot, Zoom, Bright) | Head A | Full CW ($N / (C \cdot N_c)$) | 64.86% | 63.97% | Baseline reference |
| **02-B** | Refined (Flip, Rot, Trans, Zoom, Bright, Contrast) | Head A | Full CW ($N / (C \cdot N_c)$) | 62.94% | 60.27% | Ablated (lower Macro F1) |
| **02-C** | Baseline (Flip, Rot, Zoom, Bright) | Head A | Smoothed CW ($w_c^{0.5}$) | 65.34% | 63.95% | Strong baseline |
| **02-D** | Baseline (Flip, Rot, Zoom, Bright) | Head A | None (No class weights) | 66.12% | 62.64% | Rejected (lower Macro F1) |
| **02-E (Head B)** | **Baseline (Flip, Rot, Zoom, Bright)** | **Head B** | **Full CW** | **65.45%** | **64.49%** | **WINNER (Selected)** |

### Model Selection Criterion
In multi-class facial expression recognition with severe class imbalance (e.g., FER-2013 has only 436 training samples for Disgust versus 7,215 for Happy), overall accuracy can be inflated by simply favoring majority classes.

Therefore, **PublicTest Macro F1-score was established *a priori* as the sole decision criterion**:
- Variant 02-D achieved a higher validation accuracy (66.12%) when trained without class weights, but suffered a lower Macro F1 (62.64%), indicating prediction bias toward dominant classes.
- **Variant 02-E (Head B) achieved the highest Macro F1 (64.49%)**, demonstrating superior balance across all emotion categories under full class weighting. Experiment 02-E Head B was consequently crowned the winning model.

---

## 6. Final Held-Out Evaluation on `PrivateTest`

Following hyperparameter freeze, Experiment 02-E Head B was evaluated on the unseen `PrivateTest` partition (3,589 samples).

### Comparison: Baseline (Exp 01) vs. Improved CNN (Exp 02-E Head B)

| Metric | Baseline Experiment 01 | Selected Experiment 02-E Head B | Absolute Improvement | Relative Improvement |
|:---|:---:|:---:|:---:|:---:|
| **Test Samples** | 3,589 | 3,589 | — | — |
| **Test Accuracy** | 60.96% | **67.12%** | **+6.16%** | **+10.10%** |
| **Macro Precision** | 58.97% | **67.51%** | **+8.54%** | **+14.48%** |
| **Macro Recall** | 59.05% | **66.01%** | **+6.96%** | **+11.79%** |
| **Macro F1-score** | 57.71% | **66.14%** | **+8.43%** | **+14.61%** |
| **Weighted F1-score** | 59.48% | **66.86%** | **+7.38%** | **+12.41%** |

### Clarification: Validation vs. Final Evaluation
- **PublicTest (Validation):** Used during development to select the model architecture. Experiment 02-E achieved **65.45% accuracy** and **64.49% Macro F1**.
- **PrivateTest (Final Evaluation):** Evaluated strictly after training completion. Experiment 02-E achieved **67.12% accuracy** and **66.14% Macro F1**.

### Results Transparency Note
The project reports strictly genuine, verified experimental metrics. Detailed per-class classification reports, confusion matrix figures, and history curves for Experiment 02 are archived in `FER_Final_Results.zip` and are unpacked into `results/` for grading. The baseline report is preserved in `results/private_test_classification_report.csv`. No per-class numbers or matrices are fabricated.

---

## 7. Real-Time Inference & Face Detection Pipeline

In `app/streamlit_app.py`, the inference pipeline handles arbitrary webcam frames and user-uploaded images:

```
[Uploaded Image or Webcam Frame]
              ↓
[src/face_detector.py]
  ├── Convert to Greyscale
  ├── OpenCV Haar Cascade: haarcascade_frontalface_default.xml
  ├── Detect all bounding boxes (x, y, w, h)
  ├── Halt with warning if len(faces) == 0
  ├── Select Primary Face: max(faces, key = width * height)
  ├── Compute 10% Margin Padding (clipped to [0, W] and [0, H])
  └── Annotate Image: Primary Face = Green box, Secondary = Blue box
              ↓
[Cropped Primary Face Sub-image]
              ↓
[src/preprocessing.py: preprocess_image(cropped_face, use_clahe=False)]
  ├── Greyscale conversion
  ├── Bilinear resize to 48 × 48
  ├── Normalize to float32 in [0.0, 1.0]
  └── Reshape to (1, 48, 48, 1)
              ↓
[models/exp02_02E_headB_baseline_full_cw.keras]
              ↓
[Softmax Output (7 Probabilities)] → Predicted Emotion + Confidence Score
```
