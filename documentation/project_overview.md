# Project Overview
## G19 — Facial Emotion Recognition using Deep Learning
### CSE276 | Submission: 24 November 2026

---

## Team
| Name | Contribution |
|---|---|
| Aryan Priyadarshi | Model architecture, training pipeline |
| Yuvraj | Data preprocessing, augmentation |
| Mashwin Sharma | Model evaluation, Streamlit app, documentation |

---

## Problem Statement
Human facial expressions are a primary channel for non-verbal communication.
Automatic recognition of these expressions has applications in:
- Mental health monitoring
- Driver drowsiness detection
- Human–computer interaction
- Affective computing

This project builds an end-to-end system that accepts a face image and outputs one of seven emotion categories.

---

## Dataset: FER-2013
| Property | Value |
|---|---|
| Source | Kaggle (originally from ICML 2013 Workshop) |
| Total samples | ~35,887 images |
| Image format | 48 × 48 pixel greyscale |
| Classes | 7 (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) |
| Format | CSV (pixel strings) or folder structure |

**Known imbalance:** The *Disgust* class has significantly fewer samples (~600) compared to *Happy* (~9,000). This will be addressed via class-weighted loss or oversampling.

---

## System Architecture
```
[Raw Image]
    ↓
[Preprocessing] — grayscale, resize 48×48, CLAHE
    ↓
[CNN Model]     — 3 conv blocks + dense head
    ↓
[Softmax]       — 7-class probability distribution
    ↓
[Predicted Emotion Label]
```

---

## Development Workflow
1. **Notebook 01** — Verify environment (GPU, library versions)
2. **Notebook 02** — Explore dataset (class distribution, sample visualisation)
3. **Notebook 03** — Preprocess images and prepare data generators
4. **Notebook 04** — Train CNN on Colab GPU; save best weights
5. **Notebook 05** — Evaluate: classification report, confusion matrix, curves
6. **App** — Run Streamlit demo locally using saved weights

---

## Deliverables
- [ ] Working Streamlit demo
- [ ] Trained model weights (`models/best_model.keras`)
- [ ] Evaluation plots in `results/`
- [ ] Complete documentation
- [ ] GitHub repository with clean commit history
- [ ] Final project report
