# Learning Notes – Chest X-ray Pneumonia Detection (Deep Learning)

This document records **what I learned, why decisions were made, and how the model evolved** during the development of this project.  
The goal was not just to build a working model, but to **understand each step deeply**.

---

## 1. Problem Understanding

- Pneumonia detection from chest X-ray images is a **binary classification** problem.
- Medical image datasets are often:
  - Imbalanced
  - Small
  - Sensitive to false negatives
- Accuracy alone is **not sufficient** for evaluation in medical applications.

---

## 2. Why CNN instead of ANN

I learned that:
- Images have **spatial structure** (nearby pixels are related).
- ANN flattens images and loses spatial information.
- CNN:
  - Preserves spatial relationships
  - Learns local features using filters
  - Uses parameter sharing → fewer parameters
  - Builds hierarchical features (edges → textures → regions)

Hence, CNN is fundamentally more suitable than ANN for chest X-ray analysis.

---

## 3. Dataset Exploration (Using PIL)

Before training, I manually inspected the dataset using PIL:
- Images are **grayscale** originally
- Image sizes vary
- Pneumonia images often show cloudy / opaque lung regions
- Dataset is **imbalanced** (more pneumonia than normal)

This justified:
- Resizing images to a fixed size
- Normalizing pixel values
- Careful evaluation beyond accuracy

---

## 4. Data Preprocessing Decisions

### Normalization (not standardization)
- Pixel values were scaled from `[0, 255]` → `[0, 1]`
- Standardization was not used because:
  - CNNs work well with normalized pixel intensities
  - Standardization is more suitable for tabular data

### Dataset loading
- Used `image_dataset_from_directory`
- Used predefined `train / val / test` folders
- Avoided `validation_split` to prevent overly optimistic validation results

---

## 5. Baseline CNN Architecture

Baseline CNN structure:
Input → Rescaling
Conv2D → MaxPooling
Conv2D → MaxPooling
Conv2D → MaxPooling
Flatten
Dense (128)
Output (2 classes)


### Why 3 convolution blocks?
- Block 1: learns edges and basic contrast
- Block 2: learns textures and patterns
- Block 3: learns higher-level structures
- Three blocks provide a balance between:
  - Feature extraction
  - Overfitting risk
  - Dataset size

---

## 6. Training Configuration

- Optimizer: Adam  
  (adaptive learning rate, stable convergence)
- Loss: SparseCategoricalCrossentropy (from_logits=True)  
  (labels are integer encoded, output is logits)
- Metric: Accuracy (used only as a basic indicator)

---

## 7. Baseline Model Results (Test Set)

Confusion Matrix:
[[ 64 170]
[ 2 388]]


Key observations:
- Pneumonia recall ≈ **0.99** (very high sensitivity)
- Normal recall ≈ **0.27** (poor specificity)
- Model was **biased toward pneumonia**

Medical interpretation:
- Very few pneumonia cases were missed
- Many healthy cases were falsely flagged
- Accuracy (~72%) was misleading due to class imbalance

---

## 8. Regularization Techniques Applied

To improve generalization:
- **Data Augmentation**
  - Random flip
  - Rotation
  - Zoom
- **Dropout (0.5)** after Dense layer

Effects observed:
- Training became harder (expected)
- Validation curves became noisier (small validation set)
- Overfitting reduced

---

## 9. Improved Model Results (Test Set)

Confusion Matrix:
[[153 81]
[ 6 384]]


Performance improvements:
- Normal recall improved from **0.27 → 0.65**
- Pneumonia recall remained high (**0.98**)
- Accuracy improved from **0.72 → 0.86**
- Macro F1-score improved significantly

Interpretation:
- Much better balance between sensitivity and specificity
- Fewer false positives
- Still very few false negatives
- More suitable for screening-oriented medical use

---

## 10. Key Learnings

- Validation accuracy alone can be misleading in medical datasets
- Test set evaluation is critical
- Regularization can reduce training accuracy but **improve real-world performance**
- Trade-offs between false positives and false negatives must be considered explicitly
- Simple models with good reasoning are better than complex models without understanding

---

## 11. Next Step

- Add **one transfer learning model** (MobileNetV2)
- Compare baseline CNN vs transfer learning fairly
- Analyze whether pretrained features improve generalization further

---

## 12. Reflection

This project helped me move from:
> “Running deep learning code”  
to  
> “Understanding and defending deep learning decisions”

Each model change was guided by **observations, not guesses**.
