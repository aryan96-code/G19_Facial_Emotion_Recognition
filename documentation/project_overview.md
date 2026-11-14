# Project Overview
## G19 — Facial Emotion Recognition using Deep Learning
### CSE276 | Submission Deadline: 25 November 2026

---

## 👥 Team & Contributions
| Name | Primary Role & Contributions |
|---|---|
| **Aryan Priyadarshi** | CNN model architectures (Baseline & Improved 4-block CNN), Head A/B design, Colab GPU training pipeline, loss functions, learning rate scheduling |
| **Yuvraj** | FER-2013 data exploration, data-loader with strict test partition isolation, online Keras augmentation pipelines, normalisation logic |
| **Mashwin Sharma** | Evaluation harness (confusion matrices, classification reports), Haar Cascade face detection integration, Streamlit web demo, academic documentation |

---

## 🎯 Problem Statement
Human facial expressions serve as a fundamental medium of non-verbal communication, social signaling, and interpersonal interaction. Automated facial emotion recognition (FER) offers broad utility across:
- **Affective computing:** Developing user-centric software responsive to user sentiment.
- **Human–computer interaction (HCI):** Intelligent tutors, interactive kiosks, and assistive technologies.
- **Driver safety:** Detecting drowsiness, extreme distress, or road-rage agitation.
- **Mental health research:** Quantifying facial expressiveness and emotional blunting under clinical protocols.

This project delivers a complete, academically grounded deep learning pipeline that takes raw images or webcam streams, isolates facial regions using automated computer vision, and classifies visible expressions into seven emotion categories.

---

## 📊 Dataset: FER-2013
| Property | Value / Specification |
|---|---|
| **Source** | Kaggle (originally ICML 2013 Challenges in Representation Learning) |
| **Total Samples** | 35,887 images |
| **Resolution & Format** | $48 \times 48$ single-channel greyscale |
| **Emotion Classes (7)** | 0: Angry, 1: Disgust, 2: Fear, 3: Happy, 4: Sad, 5: Surprise, 6: Neutral |
| **Partitions** | Official `Usage` split: `Training` (28,709), `PublicTest` (3,589), `PrivateTest` (3,589) |
| **Imbalance** | Extreme: *Disgust* (436 training samples) vs. *Happy* (7,215 training samples) |

---

## 🏗️ System Architecture & Workflow

### End-to-End System Diagram
```
[Raw User Image / Live Webcam Capture]
                  │
                  ▼
   [OpenCV Haar Cascade Face Detector]
   ├── `haarcascade_frontalface_default.xml`
   ├── Detects all faces in frame
   ├── Identifies Primary Face (largest bounding box by area: w × h)
   └── Applies ~10% margin padding (clipped to image bounds)
                  │
                  ▼
       [Cropped Primary Face]
                  │
                  ▼
         [Image Preprocessing]
   ├── Greyscale conversion
   ├── Resized to 48 × 48 pixels
   ├── Normalized to float32 in [0.0, 1.0] (div by 255.0)
   └── (No CLAHE applied by default — matches training)
                  │
                  ▼
 [Selected Improved CNN Model (Experiment 02-E)]
   ├── Four Convolutional Blocks (32, 64, 128, 256 filters)
   ├── Batch Normalization + ReLU + Max Pooling + Progressive Dropout
   └── Head B: Dual Pooling (GAP + GMP Concatenation → 512 features)
                  │
                  ▼
    [Dense(256) → ReLU → Dropout(0.50)]
                  │
                  ▼
      [Dense(7, activation="softmax")]
                  │
                  ▼
[Predicted Emotion Label + Probability Distribution]
```

### Key Architectural Highlight: Head B Dual Pooling
Unlike conventional CNNs that rely solely on Global Average Pooling (GAP), the selected **Head B** combines:
- **Global Average Pooling (GAP):** Captures global spatial information across the receptive field.
- **Global Max Pooling (GMP):** Captures the strongest spatial activations across each feature map.

Their outputs are concatenated into a unified 512-dimensional representation before dense classification, combining overall spatial context with peak activation responses without the parameter growth of flattening.

---

## 🚀 Sequential Development Workflow

The project was developed methodically across six dedicated notebooks and modular source files:

1. **Notebook 01 — `01_environment_setup.ipynb`**
   - Verifies hardware (Google Colab T4 GPU, CUDA), checks library versions, and establishes global reproducibility via `set_random_seeds(42)`.
2. **Notebook 02 — `02_dataset_exploration.ipynb`**
   - Explores class frequency distributions, analyzes pixel intensity histograms, and visualizes representative samples per class.
3. **Notebook 03 — `03_preprocessing.ipynb`**
   - Validates greyscale resizing, $[0, 1]$ floating-point normalization, and verifies online Keras data augmentation layers.
4. **Notebook 04 — `04_cnn_training.ipynb`**
   - Constructs and trains the 3-block Baseline CNN (Experiment 01) using class weights computed exclusively from `Training` labels.
5. **Notebook 05 — `05_model_evaluation.ipynb`**
   - Conducts the baseline evaluation on the held-out `PrivateTest` partition, generating confusion matrices and classification reports.
6. **Notebook 06 — `06_exp02_improved_cnn.ipynb`**
   - Implements the 4-block improved CNN architecture, conducts systematic ablation studies (02-A through 02-E) using strict data isolation with `load_fer2013_train_val()`, compares Head A vs. Head B, and logs checkpoints and histories.
7. **Interactive Application — `app/streamlit_app.py`**
   - Production inference interface featuring OpenCV Haar Cascade face detection, primary face selection, real-time bounding box annotations, and probability bar charts.

---

## 📦 Submission Deliverables

- [x] **Trained Model Weights:** Primary model `models/exp02_02E_headB_baseline_full_cw.keras` (15.9 MB, 1,309,671 parameters) and baseline fallback `models/best_model.keras`.
- [x] **Automated Test Suite:** 60 automated unit and regression tests passing with 100% green status across preprocessing, architecture, data loaders, and face detector modules.
- [x] **Streamlit Web Demo:** Interactive application with real-time face detection and emotion classification from uploaded images or live webcam feeds.
- [x] **Project Documentation:** Complete academic documentation covering architecture overview, methodology, experimental results, and limitations/ethics.
- [x] **Version Control:** Clean Git repository with reproducible structure, pinned `requirements.txt`, and isolated dataset handling.
