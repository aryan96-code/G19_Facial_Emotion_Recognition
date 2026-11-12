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
- Download **FER-2013** from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013).
- Place `fer2013.csv` (or the extracted `train/` and `test/` folders) inside the `dataset/` directory.
- The `dataset/` folder is listed in `.gitignore` — **do not commit the data**.

### 5. Run the Colab notebooks (training)
Open each notebook in `notebooks/` sequentially in Google Colab.  
After training, download the saved model file to `models/`.

### 6. Run the Streamlit app (local demo)
```bash
streamlit run app/streamlit_app.py
```

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
