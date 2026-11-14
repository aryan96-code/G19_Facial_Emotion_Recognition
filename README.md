# 😊 Facial Emotion Recognition using Deep Learning
### CSE276 — Group 19 | Submission Deadline: 25 November 2026

---

## 👥 Team Members
| Name | Role |
|---|---|
| Aryan Priyadarshi | Model Architecture & Training |
| Yuvraj | Data Preprocessing & Augmentation |
| Mashwin Sharma | Evaluation, App & Documentation |

---

## 🎯 Project Objective
Build an end-to-end image-classification system that predicts one of **seven facial expression categories** from face images using Convolutional Neural Networks (CNNs) trained on the **FER-2013** benchmark dataset.

The system incorporates:
1. A rigorously validated **Baseline CNN (Experiment 01)**.
2. An **Improved Four-Block CNN with Dual Pooling (Experiment 02-E Head B)** delivering state-of-the-art gains on held-out test data.
3. An automated **face detection and cropping pipeline** using OpenCV Haar Cascades for real-world webcam and uploaded images.
4. An interactive local **Streamlit application** with real-time face detection, primary face selection, and probability distribution visualization.

### Emotion Classes
| ID | Emotion | Visual Description |
|:--:|:-------:|:-------------------|
| 0 | Angry | Furrowed brows, tightened lips |
| 1 | Disgust | Wrinkled nose, raised upper lip |
| 2 | Fear | Widened eyes, open mouth |
| 3 | Happy | Raised mouth corners, smiling |
| 4 | Sad | Downturned mouth corners, drooping eyelids |
| 5 | Surprise | Arched eyebrows, wide open eyes and jaw |
| 6 | Neutral | Relaxed resting facial configuration |

---

## 📁 Project Structure
```
G19_Facial_Emotion_Recognition/
├── README.md                              ← Complete project documentation
├── requirements.txt                       ← Pinned dependencies (opencv-python>=4.8.0,<5.0.0, tensorflow, etc.)
├── .gitignore                             ← Version control ignore rules
│
├── .streamlit/                            ← Project-level Streamlit configuration
│   ├── config.toml                        ← Headless mode, port 8501, usage stats disabled
│   └── credentials.toml                   ← Blank activation credentials
│
├── notebooks/                             ← Sequential Google Colab notebooks
│   ├── 01_environment_setup.ipynb         ← Environment, GPU verification, seed setting
│   ├── 02_dataset_exploration.ipynb       ← Class distributions, pixel intensity analysis
│   ├── 03_preprocessing.ipynb             ← Greyscale normalization, augmentation visualization
│   ├── 04_cnn_training.ipynb              ← Baseline CNN training (Experiment 01)
│   ├── 05_model_evaluation.ipynb          ← Baseline evaluation on PrivateTest
│   └── 06_exp02_improved_cnn.ipynb        ← Experiment 02 (Four-block CNN, ablation, Head A vs B)
│
├── src/                                   ← Modular, reusable Python source code
│   ├── data_loader.py                     ← FER-2013 loader with strict PrivateTest isolation
│   ├── preprocessing.py                   ← Image resizing, normalization, online Keras augmentation
│   ├── model.py                           ← Baseline 3-block CNN & Improved 4-block CNN (Head A & B)
│   ├── face_detector.py                   ← Haar Cascade face detector, padding & primary face cropping
│   ├── evaluation.py                      ← Metrics computation, confusion matrices, classification reports
│   └── utils.py                           ← Seed setting, model serialization, safe model loaders
│
├── app/
│   └── streamlit_app.py                   ← Interactive web demo with Haar Cascade face detection
│
├── models/                                ← Saved model weights
│   ├── exp02_02E_headB_baseline_full_cw.keras ← Selected primary model (Experiment 02-E Head B)
│   └── best_model.keras                   ← Baseline model (Experiment 01 fallback)
│
├── dataset/                               ← FER-2013 data directory (excluded from Git)
│   └── fer2013.csv                        ← Kaggle FER-2013 CSV (35,887 samples)
│
├── results/                               ← Experiment outputs, plots, and reports
│   ├── class_distribution_train.png       ← Training set class imbalance plot
│   ├── sample_images.png                  ← Visual samples per emotion class
│   └── private_test_classification_report.csv ← Baseline Experiment 01 classification report
│
├── documentation/                         ← In-depth academic project documentation
│   ├── project_overview.md                ← High-level architecture, pipeline, and deliverables
│   ├── methodology.md                     ← Dataset splits, architecture details, ablation, results
│   └── limitations_and_ethics.md          ← Dataset, model, and ethical considerations
│
└── tests/                                 ← Automated unit and integration test suite (60 tests)
    ├── test_preprocessing.py              ← Image shape, normalisation, and dtype verification
    ├── test_model.py                      ← Baseline architecture, compilation, and serialization tests
    ├── test_exp02.py                      ← Experiment 02 4-block CNN, Head A/B, and isolation tests
    └── test_face_detector.py              ← Haar Cascade, boundary clipping, and crop pipeline tests
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/aryan96-code/G19_Facial_Emotion_Recognition.git
cd G19_Facial_Emotion_Recognition
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
> **Note on OpenCV:** The project requires `opencv-python>=4.8.0,<5.0.0` to ensure complete compatibility with `cv2.CascadeClassifier` and Haar cascade XML distribution files.

### 4. Setup the dataset
- Download `fer2013.csv` from Kaggle (e.g., [deadskull7/fer2013](https://www.kaggle.com/datasets/deadskull7/fer2013)).
- Place `fer2013.csv` directly inside the `dataset/` directory:
  ```
  dataset/fer2013.csv
  ```
- The dataset preserves the original official splits:
  - `Training`: 28,709 images
  - `PublicTest`: 3,589 images
  - `PrivateTest`: 3,589 images

### 5. Obtain Model Weights
The selected production model checkpoint (Experiment 02-E Head B, 1,309,671 parameters) is included directly in the repository at:
```
models/exp02_02E_headB_baseline_full_cw.keras
```
Users cloning the repository receive this verified model file intact and can launch the Streamlit app or run inference immediately. If cloning without weights or retraining, notebook `06_exp02_improved_cnn.ipynb` can regenerate the model.

### 6. Run automated tests
Verify environment integrity, model architectures, data isolators, and face detector modules:
```bash
pytest tests/ -v
```
All **60 automated tests** should pass.

---

## 🧠 Model Architecture & Selected Model Details

Our selected production model is **Experiment 02-E (Head B)**:
- **Architecture Name:** `FER_Improved_CNN_HeadB`
- **Total Parameters:** 1,309,671
- **Checkpoint File:** `models/exp02_02E_headB_baseline_full_cw.keras`

### Architectural Overview
```
Input: (48, 48, 1) Float32 Normalised [0, 1]
  │
  ├── [Online Augmentation Layer] (Active only during training)
  │     RandomFlip("horizontal")
  │     RandomRotation(factor=10/360)
  │     RandomTranslation(height_factor=0.08, width_factor=0.08)
  │     RandomZoom(height_factor=0.10, width_factor=0.10)
  │     RandomBrightness(factor=0.15, value_range=(0.0, 1.0))
  │     RandomContrast(factor=0.15, value_range=(0.0, 1.0))
  │
  ├── [Block 1]: 2 × Conv2D(32, 3×3) → BatchNorm → ReLU → MaxPool2D(2×2) → Dropout(0.15)
  ├── [Block 2]: 2 × Conv2D(64, 3×3) → BatchNorm → ReLU → MaxPool2D(2×2) → Dropout(0.20)
  ├── [Block 3]: 2 × Conv2D(128, 3×3) → BatchNorm → ReLU → MaxPool2D(2×2) → Dropout(0.25)
  ├── [Block 4]: 2 × Conv2D(256, 3×3) → BatchNorm → ReLU → MaxPool2D(2×2) → Dropout(0.30)
  │
  └── [Head B — Dual Pooling]:
        ├── GlobalAveragePooling2D ──┐
        └── GlobalMaxPooling2D     ──┴─→ Concatenate (512-dim feature vector)
                                          ↓
                                      Dense(256) → BatchNorm → ReLU → Dropout(0.50)
                                          ↓
                                      Dense(7, activation="softmax")
```

### Why Head B?
While standard architectures use only Global Average Pooling (GAP), Head B combines **Global Average Pooling** (which captures global spatial information) with **Global Max Pooling** (which captures the strongest spatial activations). Their outputs are concatenated into a unified 512-dimensional representation before classification, retaining both overall spatial distribution and peak feature responses without the parameter explosion of flattening.

---

## 📊 Experimental Results & Evaluation

### 1. Experiment 02 Ablation Study (`PublicTest` Validation)
During Experiment 02, five systematic variants were trained under identical conditions on the 28,709 training samples and validated on `PublicTest` (3,589 samples).

In accordance with rigorous academic methodology, **PublicTest Macro F1-score** was used as the sole decision criterion to choose the winning model for held-out evaluation:

| Experiment | Configuration | Val Accuracy | Val Macro F1 | Selection Status |
|:---|:---|:---:|:---:|:---:|
| **02-A** | Baseline Augmentation + Head A + Full Class Weights | 64.86% | 63.97% | Baseline reference |
| **02-B** | Refined Augmentation + Head A + Full Class Weights | 62.94% | 60.27% | Ablated (lower Macro F1) |
| **02-C** | Baseline Augmentation + Head A + Smoothed Class Weights | 65.34% | 63.95% | Strong baseline |
| **02-D** | Baseline Augmentation + Head A + No Class Weights | 66.12% | 62.64% | Highest Acc, lower Macro F1 |
| **02-E (Head B)** | **Baseline Augmentation + Head B + Full Class Weights** | **65.45%** | **64.49%** | **WINNER (Selected)** |

> **Key Decision:** Experiment 02-E Head B was selected because it achieved the **highest Macro F1-score (64.49%)**, ensuring balanced performance across both majority and minority emotion classes rather than biasing predictions toward majority classes.

---

### 2. Final Held-Out Evaluation on `PrivateTest` (3,589 Samples)

The final evaluation was conducted strictly on the unseen `PrivateTest` partition. Neither the model weights nor hyperparameters had prior exposure to these samples:

| Metric | Baseline (Experiment 01) | Improved CNN (Experiment 02-E Head B) | Absolute Gain |
|:---|:---:|:---:|:---:|
| **Test Accuracy** | 60.96% | **67.12%** | **+6.16%** |
| **Macro Precision** | 58.97% | **67.51%** | **+8.54%** |
| **Macro Recall** | 59.05% | **66.01%** | **+6.96%** |
| **Macro F1-score** | 57.71% | **66.14%** | **+8.43%** |
| **Weighted F1-score** | 59.48% | **66.86%** | **+7.38%** |

*Note on Results Transparency:* The verified full per-class evaluation artifacts (detailed classification CSV and metrics JSON) are archived in `FER_Final_Results.zip` and are unpacked into `results/` for grading. Baseline Experiment 01 metrics are preserved in [`results/private_test_classification_report.csv`](results/private_test_classification_report.csv).

---

## 🖥️ Streamlit Web Application

An interactive local demo is provided in `app/streamlit_app.py`.

### Launching the Application
Run from the repository root:
```bash
streamlit run app/streamlit_app.py
```
Access the application at: **`http://localhost:8501`**

### Key Application Features
- **Primary Model Priority:** Automatically loads `models/exp02_02E_headB_baseline_full_cw.keras`. If unavailable, gracefully falls back to `models/best_model.keras`.
- **Dual Input Modes:**
  - 📂 **Image Upload:** Upload JPG, JPEG, or PNG images.
  - 📷 **Webcam Capture:** Capture live photos directly through your webcam.
- **Automated Face Detection & Cropping:**
  - Uses an OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`).
  - Automatically identifies all faces in the frame.
  - Highlights the **primary face in green** (selected by largest bounding box area $w \times h$) and secondary faces in blue.
  - Adds a proportional **10% margin padding** clipped to image boundaries.
  - Displays the annotated original image alongside the cropped face thumbnail.
  - If **no face is detected**, displays a clear warning and halts inference without attempting to classify arbitrary background textures.
- **Emotion Probability Distribution:**
  - Preprocesses the cropped face into $(1, 48, 48, 1)$ float32 normalized to $[0.0, 1.0]$.
  - Displays the top predicted emotion, confidence percentage, and horizontal probability progress bars for all seven classes.

---

## ⚠️ Limitations & Ethical Considerations

1. **Construct Validity:** A CNN trained on FER-2013 classifies visible facial muscle movements (surface expressions), **not** internal emotional feelings, psychological states, or intent.
2. **Dataset Noise & Imbalance:** FER-2013 suffers from severe class imbalance (*Disgust* has only ~55 samples in test; *Happy* has ~879) and low resolution (48×48 greyscale).
3. **Haar Cascade Limits:** Haar Cascades require clear frontal or near-frontal faces. Severe head yaw/pitch, heavy occlusions (masks, sunglasses), or uneven lighting can cause missed detections.
4. **Prohibited Applications:** This academic project is designed exclusively for educational research. **It must not be used for surveillance, employment screening, mental health diagnosis, credit scoring, or any automated decision-making impacting individuals.**

See [`documentation/limitations_and_ethics.md`](documentation/limitations_and_ethics.md) for full analysis.

---

## 🛠 Technology Stack
| Tool | Purpose |
|---|---|
| Python 3.10+ | Programming language |
| TensorFlow / Keras | Deep learning modeling, online augmentation, inference |
| OpenCV (`<5.0.0`) | Face detection (Haar Cascade), image decoding, resizing |
| NumPy & Pandas | Array manipulation and tabular metrics processing |
| Matplotlib & Seaborn | Training curves and confusion matrix plotting |
| Streamlit | Local interactive web interface |
| Pytest | Automated test suite (unit and regression testing) |

---

## 🚫 Honest-Science Pledge
> Group 19 adheres to strict scientific integrity. All reported metrics reflect **actual, verified training runs** on Google Colab GPU. No accuracy figures, confusion matrices, or test results are fabricated or altered.
