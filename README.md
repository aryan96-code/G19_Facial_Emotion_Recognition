# 😊 Facial Emotion Recognition using Deep Learning
### CSE276 — Group 19 | Submission Deadline: 24 November 2026

---

## 👥 Team Members
| Name | Role |
|---|---|
| Aryan Priyadarshi | Model Architecture & Training |
| Yuvraj | Data Preprocessing & Augmentation |
| Mashwin Sharma | Evaluation, App & Documentation |

---

## 🎯 Project Objective
Build an end-to-end image-classification system that predicts one of **seven facial expression categories** from a grayscale face image using a Convolutional Neural Network (CNN) trained on the **FER-2013** dataset.

### Emotion Classes
| ID | Emotion |
|----|---------|
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Sad |
| 5 | Surprise |
| 6 | Neutral |

---

## 📁 Project Structure
```
G19_Facial_Emotion_Recognition/
├── README.md                     ← You are here
├── requirements.txt              ← All Python dependencies
├── .gitignore                    ← Files/folders excluded from Git
│
├── notebooks/                    ← Google Colab notebooks (run in order)
│   ├── 01_environment_setup.ipynb
│   ├── 02_dataset_exploration.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_cnn_training.ipynb
│   └── 05_model_evaluation.ipynb
│
├── src/                          ← Core Python modules (reusable logic)
│   ├── data_loader.py            ← Load and split the FER-2013 dataset
│   ├── preprocessing.py          ← Image normalisation and augmentation
│   ├── model.py                  ← CNN architecture definition
│   ├── evaluation.py             ← Metrics, confusion matrix, plots
│   └── utils.py                  ← Shared helper utilities
│
├── app/
│   └── streamlit_app.py          ← Interactive web demo (run locally)
│
├── models/                       ← Saved .h5 / .keras model weights
├── dataset/                      ← Place FER-2013 CSVs here (not committed)
├── results/                      ← Plots, confusion matrices, logs
│
├── documentation/
│   ├── project_overview.md
│   ├── methodology.md
│   └── limitations_and_ethics.md
│
└── tests/
    ├── test_preprocessing.py
    └── test_model.py
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd G19_Facial_Emotion_Recognition
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
- Download the **FER-2013** CSV (`fer2013.csv` containing `emotion`, `pixels`, and `Usage` columns) from Kaggle (e.g., [deadskull7/fer2013](https://www.kaggle.com/datasets/deadskull7/fer2013)).
- Place `fer2013.csv` inside `dataset/fer2013.csv` so the exact original splits (`Training`: 28,709, `PublicTest`: 3,589, `PrivateTest`: 3,589) are preserved.
- The `dataset/` folder is listed in `.gitignore` — **do not commit the data**.

### 5. Run the Colab notebooks (training)
Open each notebook in `notebooks/` sequentially in Google Colab (T4 GPU).  
After training, download `best_model.keras` (or `best_model_drive.keras`) from Google Drive into `models/best_model.keras`.

### 6. Run the Streamlit app (local demo)
```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Baseline Performance (`PrivateTest`, 3,589 images)

Trained on Google Colab using `Training` (28,709) and `PublicTest` (3,589) with **CLAHE disabled**, Adam (`lr=1e-3`), batch size `64`, `40` epochs (best epoch: `40`), and training-only class weights:

| Metric | Value |
|---|---|
| Training Accuracy / Loss | 58.68% / 1.0505 |
| Validation Accuracy / Loss (`PublicTest`) | 58.46% / 1.0914 |
| **Test Accuracy (`PrivateTest`)** | **60.96%** |
| Macro Precision / Recall / F1 | 0.5897 / 0.5905 / 0.5771 |
| Weighted F1 | 0.5948 |

Full per-class metrics are stored in [`results/private_test_classification_report.csv`](results/private_test_classification_report.csv).

### Reproducibility Details
- **Random seed**: `42` (`set_random_seeds(42)` in `src/utils.py`)
- **Dataset splits**: Original FER-2013 `Usage` column (`Training` / `PublicTest` / `PrivateTest`)
- **Class weights**: Balanced weights computed strictly from `Training` labels (`y_train`)
- **Baseline preprocessing**: Greyscale 48×48, `[0, 1]` normalisation, `USE_CLAHE = False`
- **Unit tests**: `pytest tests/ -v` (verified on Python 3.12 with `requirements.txt`)

---

## 🛠 Technology Stack
| Tool | Purpose |
|---|---|
| Python 3.10+ | Primary language |
| TensorFlow / Keras | Model training and inference |
| NumPy / Pandas | Numerical and tabular operations |
| Matplotlib / Seaborn | Visualisation |
| OpenCV | Image capture and preprocessing |
| Streamlit | Interactive web application |
| Git | Version control |

---

## 🚫 Honest-Science Pledge
> This project reports **only real experimental results** obtained from actual training runs.  
> No accuracy numbers, confusion matrices, or screenshots are fabricated.

---

## 📄 License
Academic project — CSE276. Not for commercial use.
