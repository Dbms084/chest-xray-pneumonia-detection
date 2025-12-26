# Chest X-ray Pneumonia Detection using Deep Learning

This project focuses on detecting **pneumonia from chest X-ray images** using **Convolutional Neural Networks (CNNs)**.  
The emphasis of this work is not just achieving high accuracy, but **understanding model behavior, handling medical data carefully, and improving generalization through experimentation**.

---

## 📌 Project Motivation

- Pneumonia is a serious respiratory condition that can be identified from chest X-ray images.
- Medical imaging datasets are often:
  - Imbalanced
  - Limited in size
  - Sensitive to false negatives
- This project aims to build a **screening-oriented deep learning model** that prioritizes high sensitivity while improving specificity through regularization.

---

## 🧠 Key Concepts Used

- Convolutional Neural Networks (CNNs)
- Image preprocessing and normalization
- Train / validation / test split (no random leakage)
- Overfitting and regularization
- Evaluation using confusion matrix, precision, recall, and F1-score
- Medical interpretation of results (false positives vs false negatives)

---

## 📂 Dataset

**Chest X-Ray Images (Pneumonia)** – Kaggle  
- Two classes: `NORMAL`, `PNEUMONIA`
- Predefined folders:
chest_xray/
├── train/
├── val/
└── test/

Why this matters:
- Avoids overly optimistic validation results
- Ensures realistic generalization evaluation

---

## 🛠️ Data Exploration & Preprocessing

- Images were inspected using **PIL** to understand:
- Size variation
- Grayscale nature
- Visual differences between classes
- Images were:
- Resized to `180 × 180`
- Converted to RGB automatically by TensorFlow
- **Normalized to [0,1]** using rescaling

> Standardization was not used, as CNNs operate effectively on normalized pixel intensities.

---

## 🏗️ Baseline CNN Architecture

The initial model was built from scratch using TensorFlow/Keras:
Input → Rescaling
Conv2D → MaxPooling
Conv2D → MaxPooling
Conv2D → MaxPooling
Flatten
Dense (128)
Dense (2 classes)


### Why 3 convolution blocks?
- Block 1: edge and contrast detection
- Block 2: texture and pattern learning
- Block 3: higher-level structural features
- Balanced depth for limited medical data

---

## ⚙️ Training Configuration

- Optimizer: **Adam**
- Loss function: **SparseCategoricalCrossentropy (from_logits=True)**
- Metric: Accuracy (used only as a basic indicator)
- Training performed on **Google Colab** due to hardware constraints

---

## 📊 Baseline Model Results (Test Set)

Confusion Matrix:
[[ 64 170]
[ 2 388]]


Observations:
- Pneumonia recall ≈ **0.99**
- Normal recall ≈ **0.27**
- Model strongly biased toward predicting pneumonia
- Accuracy (~72%) was misleading due to class imbalance

---

## 🔧 Model Improvement (Regularization)

To improve generalization and reduce bias:

### Techniques applied:
- **Data Augmentation**
  - Random horizontal flip
  - Rotation
  - Zoom
- **Dropout (0.5)** after Dense layer

These changes intentionally made training harder to reduce overfitting.

---

## 📈 Improved Model Results (Test Set)

Confusion Matrix:
[[153 81]
[ 6 384]]


### Performance Summary:
| Metric | Baseline | Improved |
|------|---------|----------|
| Accuracy | 0.72 | **0.86** |
| Normal Recall | 0.27 | **0.65** |
| Pneumonia Recall | 0.99 | 0.98 |
| Macro F1-score | 0.62 | **0.84** |

### Medical Interpretation:
- Significant reduction in false positives
- Slight increase in false negatives (still very low)
- Much better balance between sensitivity and specificity
- More suitable for screening-oriented applications

---

## 🧪 Key Learnings

- Validation accuracy alone is unreliable for medical datasets
- Test set evaluation is essential
- Regularization can reduce training accuracy but improve real-world performance
- High sensitivity is critical, but specificity must also be improved
- Model decisions must be justified, not just implemented

---


---

## 🚀 Next Steps

- Add **one transfer learning model (MobileNetV2)**
- Compare baseline CNN vs transfer learning
- Analyze improvements in generalization
- Add Grad-CAM visualization for interpretability

---

## 📝 Author Note

This project was developed with a **learn-by-building approach**, focusing on understanding each architectural and training decision rather than copying end-to-end solutions.

---

## ⚠️ Disclaimer

This project is intended for **academic and learning purposes only** and should not be used for real clinical diagnosis.




