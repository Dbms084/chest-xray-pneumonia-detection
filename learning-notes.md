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

## Learnings after using transfer learning and fine tuning 
---
# 📘 Learning Notes – Chest Pneumonia Detection Project

This document records my **learning journey, decisions, mistakes, and insights** while building a deep learning system to detect pneumonia from chest X-ray images.  
The focus is on **understanding**, not just results.

---

## 1️⃣ Understanding the Problem (Medical Context)

- Pneumonia detection is a **binary classification** problem:
  - NORMAL
  - PNEUMONIA
- In medical AI:
  - **False negatives (missing pneumonia)** are more dangerous than false positives
  - Therefore, **recall for pneumonia** is a critical metric
- Accuracy alone is **not sufficient** for evaluation

📌 This changed how I evaluated models later.

---

## 2️⃣ Baseline CNN – First Attempt

### What I did
- Built a simple CNN using:
  - Conv → Pool → Conv → Pool → Dense
- No regularization
- Trained and evaluated on test set

### What I observed
- Very high pneumonia recall (~0.99)
- Very poor normal recall (~0.25)
- Model predicted pneumonia for most images

### What I learned
- The model was **biased toward the majority / easier class**
- High accuracy does not mean a good medical model
- Baseline models help **reveal bias**, not solve the problem

---

## 3️⃣ Importance of Proper Evaluation

I learned to evaluate models using:
- Confusion matrix
- Precision, recall, F1-score
- Class-wise analysis

Key realization:
> Two models with similar accuracy can behave very differently in real-world usage.

---

## 4️⃣ Regularized CNN – Major Improvement

### What I added
- Data augmentation:
  - RandomFlip
  - RandomRotation
  - RandomZoom
- Batch Normalization
- Dropout (0.5)

### Why
- Reduce overfitting
- Improve generalization
- Balance predictions between classes

### What happened
- NORMAL recall increased significantly (~0.65)
- Pneumonia recall remained high (~0.98)
- Best macro F1-score among all models

### Key learning
> A well-regularized, task-specific CNN can outperform more complex pretrained models.

This model became the **final best model**.

---

## 5️⃣ Transfer Learning – MobileNetV2

### Frozen Feature Extractor
- Used pretrained ImageNet weights
- Trained only the classifier head

**Observation:**
- Stable learning
- Moderate improvement over baseline
- Limited adaptation to medical images

### Fine-Tuned MobileNetV2
- Unfroze top layers
- Used very small learning rate (1e-5)

**Observation:**
- Slight performance improvement
- Pneumonia recall remained excellent
- Still did not outperform regularized CNN

### Key learning
> Transfer learning improves stability, but pretrained features from natural images may not be optimal for medical imaging tasks.

---

## 6️⃣ Transfer Learning – DenseNet121

### Why DenseNet121
- Popular in medical imaging literature
- Dense feature reuse

### Result
- Frozen DenseNet121 showed:
  - Poor normal recall
  - Similar behavior to baseline CNN
- Did not improve specificity

### Learning
> Not all pretrained models generalize well to chest X-rays, even if they are popular in research.

---

## 7️⃣ Learning About Training Variance

- Re-training the same model sometimes produced different results
- Caused by:
  - Random weight initialization
  - Data shuffling
  - Stochastic optimization

Important realization:
> In deep learning, we report the **best validated run**, not every run.

---

## 8️⃣ TensorFlow Best Practices Learned

- Keep data augmentation **inside the model**
- Do not augment validation or test data
- Use `model.evaluate()` for final reporting
- Always evaluate on a **separate test set**
- Save models to avoid retraining after runtime resets

---

## 9️⃣ Model Saving & Reproducibility

Learned to save models using:

```python
model.save("model_name")

