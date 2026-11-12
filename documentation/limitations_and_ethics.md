# Limitations and Ethics
## G19 — Facial Emotion Recognition | CSE276

---

## ⚠️ Limitations

### 1. Dataset Limitations
- **Class imbalance**: The *Disgust* class (~600 samples) is severely under-represented compared to *Happy* (~9,000 samples). The model may exhibit lower recall for minority classes.
- **Greyscale only**: FER-2013 images are 48 × 48 greyscale. Colour and texture cues that humans use to read emotions are not available.
- **Low resolution**: The small image size limits the fine-grained feature detail the model can capture.
- **Wild images**: FER-2013 was collected from internet images with variable pose, occlusion, and illumination — this is realistic but noisy.
- **Age and demographic distribution**: The dataset does not document its demographic composition, so the model's performance across different ethnicities, ages, and genders is unknown.

### 2. Model Limitations
- **Static images only**: Our model predicts from a single frame; it ignores temporal dynamics of expression that humans rely on.
- **Single-face assumption**: The pipeline expects one face per image. Multi-face scenes are not handled.
- **Generalisation**: A model trained on FER-2013 may not generalise well to real-world camera feeds with different resolutions, compression artefacts, or lighting conditions.
- **Ambiguous emotions**: Many real-world expressions are blends of multiple emotions. A single-label prediction does not capture this nuance.
- **No face-detection fallback**: If the uploaded image does not contain a clear frontal face, predictions will be unreliable.

### 3. Performance Limitations
- **Accuracy ceiling**: State-of-the-art on FER-2013 is ~73–75%. Human inter-rater agreement on this dataset is also imperfect (~65%).
- **Reported results are experiment-specific**: Numbers reported in this project reflect actual training runs and should not be extrapolated to other contexts.

---

## ⚖️ Ethical Considerations

### Informed Consent
Emotion recognition systems should **never** be deployed on individuals without their knowledge and explicit consent. This demo is strictly an academic tool and must not be used for covert surveillance.

### Bias and Fairness
Given the unknown demographic composition of FER-2013:
- Performance gaps across racial groups, genders, or age groups are likely.
- Deployment in high-stakes contexts (hiring, law enforcement, healthcare) would be inappropriate without rigorous bias auditing.

### Misuse Potential
Facial emotion recognition has documented potential for misuse:
- **Workplace surveillance**: Monitoring employee emotional states.
- **Discriminatory profiling**: Using expressions as proxies for protected characteristics.
- **Political suppression**: Identifying emotional reactions at protests.

This project does **not** endorse any such use cases.

### Data Privacy
- No images used in this project are collected from users without consent.
- The Streamlit demo processes images **locally** — no image is uploaded to any external server.
- Uploaded images are not stored or logged.

### Academic Honesty
All results in this project are obtained from actual training experiments.
No accuracy numbers, plots, or screenshots are fabricated or borrowed from other projects.

---

## 📌 Responsible Use Statement
> This system is built for educational purposes only.
> It is not production-ready and should not be used for any real-world decision-making
> without extensive validation, bias auditing, and appropriate ethical oversight.
