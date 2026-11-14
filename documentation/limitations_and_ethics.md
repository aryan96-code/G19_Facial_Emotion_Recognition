# Limitations and Ethical Considerations
## G19 — Facial Emotion Recognition | CSE276
### Submission Deadline: 25 November 2026

---

## ⚠️ 1. Dataset Limitations (FER-2013)

### Severe Class Imbalance
The FER-2013 dataset exhibits severe class imbalance across its 35,887 samples:
- In the training set (28,709 images), **Happy** accounts for 7,215 samples (25.1%), while **Disgust** accounts for only 436 samples (1.5%) — a ratio exceeding $16:1$.
- In the private test set (3,589 images), *Disgust* contains only 55 samples compared to 879 for *Happy*.
- Although balanced class weighting was incorporated during model training ($N / (C \cdot N_c)$), minority classes naturally possess fewer distinct feature representations, leading to higher performance variance.

### Resolution & Information Loss
- All images are constrained to $48 \times 48$ single-channel greyscale.
- Critical physiological cues that humans rely on to assess emotion — such as flushing, blushing, skin pallor, micro-expression dynamics, and fine eye-wrinkle textures — are completely absent in 48-pixel greyscale representations.

### Noise in Web-Scraped Data
- FER-2013 was assembled via automated Google Image searches using emotion-related query keywords.
- Consequently, the dataset contains notable label noise, cartoon drawings, partial occlusions, watermarks, non-face artifacts, and ambiguous expressions where human labeler agreement is estimated at only ~65%.

### Demographic Representation & Fairness
- FER-2013 lacks demographic metadata regarding ethnicity, age, cultural background, or gender.
- Machine learning models trained on uncurated internet datasets historically risk disproportionately lower accuracy on under-represented ethnic groups, older populations, and individuals with physical facial atypicalities.

---

## 📷 2. Benchmark vs. Real-World Webcam Discrepancy

There is a substantial domain shift between controlled benchmark evaluation and real-world webcam deployment:

| Dimension | FER-2013 Benchmark Environment | Real-World Webcam / Upload Feed |
|:---|:---|:---|
| **Cropping** | Pre-cropped, centered, tight face boundaries | Unconstrained scenes with arbitrary head positions and sizes |
| **Head Pose** | Primarily frontal or near-frontal | Frequent pitch, yaw, and roll (head tilt and turning) |
| **Illumination** | Post-processed, balanced contrast | Extreme backlighting, harsh shadows, low-light webcam sensors |
| **Motion & Noise** | Clean static images | Motion blur, sensor noise, webcam compression artifacts |
| **Aspect Ratio** | Fixed square $48 \times 48$ | Variable camera aspect ratios (16:9, 4:3) requiring dynamic scaling |

---

## 🔍 3. Face Detection Pipeline Limitations (Haar Cascade)

The integrated real-time detection module (`src/face_detector.py`) utilizes OpenCV's classical frontal face Haar Cascade (`haarcascade_frontalface_default.xml`):

### Operating Constraints
- **Frontal Bias:** The Haar Cascade relies on handcrafted rectangular features (light-dark intensity transitions around the bridge of the nose and eye sockets). It performs reliably only on frontal or near-frontal faces ($\pm 15^\circ$ rotation).
- **Occlusions:** Masks, thick-framed glasses, hands covering the chin or mouth, or heavy bangs can degrade detection rates significantly.
- **Lighting Sensitivity:** Low-contrast or unevenly lit faces (e.g., strong side lighting) may prevent the cascade from identifying face candidate rectangles.

### Safety Safeguards Implemented
- The application **strictly refuses to run emotion inference** if no face is detected (`detected == False`).
- Rather than silently classifying random background patterns or furniture textures, the application halts execution with a user-facing warning:
  `⚠️ No face detected in the image. Please upload or capture a clearer, well-lit frontal face photo.`
- When multiple faces are present, the system explicitly selects the **primary face** by maximum bounding box area ($w \times h$), highlights it in green (with secondary faces in blue), and notifies the user.

---

## 🧠 4. Fundamental Construct Validity: Expressions vs. Internal States

> ### **CRITICAL SCIENTIFIC PRINCIPLE**
> **Facial expressions cannot reliably establish internal emotional states.**

A Convolutional Neural Network trained on FER-2013 classifies **visible surface facial muscle movements (facial configurations)**, *not* a person's genuine internal feelings, intentions, subjective psychological state, or character.

### Psychological & Social Realities
1. **Emotional Concealment & Social Display Rules:**
   Humans routinely mask, suppress, exaggerate, or simulate expressions in response to professional norms, cultural conventions, or politeness (e.g., smiling when anxious or frustrated).
2. **Context Dependency:**
   Identical facial movements can communicate vastly different mental states depending on context (e.g., tears during joy versus grief; widened eyes in excitement versus mortal terror).
3. **Neurodivergence & Physical Variance:**
   Individuals with Parkinson's disease, Bell's palsy, stroke recovery, autism, or natural resting facial traits may present facial configurations that commercial or academic FER algorithms misclassify as "Angry", "Sad", or "Neutral".
4. **No Direct Physiological Mapping:**
   Contemporary affective science (e.g., Barrett et al., 2019) conclusively demonstrates that there is no one-to-one biological correspondence between six or seven discrete emotional categories and specific facial muscle movements.

---

## ⚖️ 5. Ethical Guidelines & Prohibited Use Cases

### Explicitly Prohibited Applications
The model and code in this repository were developed strictly as an academic research project for CSE276. **They must never be deployed or utilized in any of the following high-stakes contexts:**

- ❌ **Mental Health Diagnosis:** Must not be used to diagnose depression, anxiety, trauma, or emotional blunting.
- ❌ **Workplace & Employee Surveillance:** Must not be used to monitor worker attention, satisfaction, morale, or productivity.
- ❌ **Employment & Hiring Decisions:** Must not be used to evaluate video interviews, candidate suitability, or honesty.
- ❌ **Law Enforcement & Border Control:** Must not be used for lie detection, interrogation, threat assessment, or suspicious behavior scoring.
- ❌ **Credit, Insurance, or Housing Decisions:** Must not be used to assess creditworthiness, risk profiles, or tenant selection.
- ❌ **Classroom Proctoring & Grading:** Must not be used to penalize students based on detected "boredom" or "inattention."

### Privacy & Data Protection
- **Local Execution:** The Streamlit demo operates completely locally. No webcam captures or uploaded photos are transmitted to external servers, cloud storage, or third-party APIs.
- **No Persistent Storage:** Uploaded and webcam images are held temporarily in volatile memory for inference and are discarded immediately.
- **Informed Consent:** Automated facial recognition should never be performed on individuals without their explicit, informed consent.

---

## 📌 Academic Integrity & Responsible Use Statement

> This project is designed exclusively for educational and research demonstration purposes. It highlights both the technical possibilities of deep convolutional networks and the profound limitations of automated expression classification. Group 19 disclaims any liability for unauthorized or unethical application of this software.
