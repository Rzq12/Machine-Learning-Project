# 🚦 German Traffic Sign Recognition Benchmark (GTSRB) — CNN Classification

## Dataset

[GTSRB - German Traffic Sign Recognition Benchmark by meowmeowmeowmeow](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

## 📑 Table of Contents

- [Project Description](#Project-Description)
- [Objective](#Objective)
- [Approach](#Approach)
- [Methods and Models](#Methods-and-Models)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Model Architecture](#2-model-architecture)
  - [3. Callbacks & Regularization](#3-callbacks--regularization)
  - [4. Evaluation Metrics](#4-evaluation-metrics)
  - [5. Tools and Libraries](#5-tools-and-libraries)

- [Model Evaluation Results](#Model-Evaluation-Results)
- [Conclusion](#conclusion)

---

## Project Description

Traffic sign recognition is a critical component in autonomous driving and advanced driver-assistance systems (ADAS). This project builds a **multi-class image classification model** using a custom **Convolutional Neural Network (CNN)** trained on the **GTSRB dataset**, which contains over 50,000 labeled images of **43 distinct German traffic sign categories**.

The model learns to extract and classify visual features from low-resolution traffic sign images (30×30 pixels), handling real-world challenges such as varying lighting, blur, and perspective distortion.

---

## Objective

The primary objectives of this project are to:

- Develop a **multi-class CNN classifier** capable of identifying 43 types of German traffic signs.
- Build a robust preprocessing pipeline that handles raw image loading and normalization.
- Apply **callbacks** to automate training optimization and prevent overfitting.
- Export the trained model in multiple formats: **SavedModel**, **TF-Lite**, and **TensorFlow.js**.

---

## Approach

- Build a custom **Sequential CNN** from scratch without transfer learning.
- Load images directly from class-indexed folders and normalize pixel values to [0, 1].
- Split training data into 80% train / 20% validation using **stratified sampling** to maintain class balance.
- Use **One-Hot Encoding** for multi-class label representation.
- Evaluate model performance using classification report, confusion matrix, and visual prediction samples.

---

## Methods and Models

### 1. Data Preprocessing

- **Image Loading:**
  - Training images loaded from class-indexed folders (`Train/0` to `Train/42`).
  - Test images loaded using file paths from `Test.csv`.

- **Image Resizing:**
  - All images resized to **30×30 pixels** using OpenCV.

- **Normalization:**
  - Pixel values divided by 255.0 to scale to the range [0, 1].

- **Dataset Split:**
  - 80% training / 20% validation, stratified by class using `train_test_split`.

- **Label Encoding:**
  - Integer labels converted to **one-hot vectors** for categorical cross-entropy loss.

### 2. Model Architecture

- **Custom Sequential CNN:**
  - `Conv2D(32, 5×5, relu)` → `Conv2D(64, 5×5, relu)` → `MaxPool2D(2×2)` → `Dropout(0.15)`
  - `Conv2D(128, 3×3, relu)` → `Conv2D(256, 3×3, relu)` → `MaxPool2D(2×2)` → `Dropout(0.20)`
  - `Flatten()` → `Dense(512, relu)` → `Dropout(0.25)` → `Dense(43, softmax)`

```python
Model Summary:
Conv2D(32) ➜ Conv2D(64) ➜ MaxPool ➜ Dropout ➜ Conv2D(128) ➜ Conv2D(256) ➜ MaxPool ➜ Dropout ➜ Flatten ➜ Dense(512) ➜ Dropout ➜ Dense(43, softmax)
```

- **Optimizer:** Adam
- **Loss Function:** Categorical Cross-Entropy
- **Output:** 43-class Softmax probabilities

### 3. Callbacks & Regularization

- **ModelCheckpoint**: Saves the best model based on validation accuracy (`best_model.keras`).
- **EarlyStopping**: Halts training if validation accuracy doesn't improve for 7 consecutive epochs, then restores the best weights.
- **ReduceLROnPlateau**: Reduces learning rate by a factor of 0.5 when validation loss plateaus for 3 epochs (minimum LR: 1e-7).

### 4. Evaluation Metrics

- Accuracy (training, validation & test)
- Categorical Cross-Entropy Loss
- Per-class Precision, Recall, and F1-Score (Classification Report)
- Confusion Matrix (43×43 heatmap)
- Confidence scores on visual prediction samples

### 5. Tools and Libraries

- `TensorFlow`, `Keras`
- `NumPy`, `Pandas`, `Matplotlib`, `Seaborn`
- `OpenCV (cv2)` for image loading and resizing
- `scikit-learn` for train/test split, classification report, and confusion matrix
- `tensorflowjs` for TFJS model export

---

## Model Evaluation Results

Training was performed using the custom Sequential CNN on the GTSRB dataset with 43 traffic sign classes. The model was trained for up to 50 epochs with early stopping enabled.

### 📊 Dataset Split

| Split          | Images     |
| -------------- | ---------- |
| Training Set   | ~31,367    |
| Validation Set | ~7,842     |
| Test Set       | 12,630     |
| **Total**      | **51,839** |

### 📉 Training Callbacks Summary

| Callback          | Monitor      | Configuration                       |
| ----------------- | ------------ | ----------------------------------- |
| ModelCheckpoint   | val_accuracy | Save best model only                |
| EarlyStopping     | val_accuracy | Patience=7, restore best weights    |
| ReduceLROnPlateau | val_loss     | Factor=0.5, Patience=3, min_lr=1e-7 |

### 🏆 Final Model Performance

| Dataset        | Loss | Accuracy |
| -------------- | ---- | -------- |
| Training Set   | —    | ~99%+    |
| Validation Set | —    | ~98%+    |
| Test Set       | —    | ~97%+    |

### 📦 Exported Model Formats

| Format         | Path                                  | Description                     |
| -------------- | ------------------------------------- | ------------------------------- |
| Keras (.keras) | `/kaggle/working/best_model.keras`    | Best checkpoint during training |
| SavedModel     | `/kaggle/working/saved_model/`        | TensorFlow SavedModel format    |
| TF-Lite        | `/kaggle/working/tflite/model.tflite` | Quantized for mobile/edge       |
| TensorFlow.js  | `/kaggle/working/tfjs_model/`         | For browser-based inference     |

---

## Conclusion

- A **custom CNN from scratch** successfully classifies 43 German traffic sign categories with high accuracy.
- **Stratified train/val split** ensured balanced class representation across all 43 categories.
- **Callbacks** (EarlyStopping, ReduceLROnPlateau) effectively prevented overfitting and automated learning rate scheduling.
- The model was exported in **three deployment formats** (SavedModel, TF-Lite, TFJS), making it suitable for server, mobile, and browser deployment.
- **SavedModel inference validation** confirmed consistent predictions matching the Keras model outputs.

---

> 💡 Future Work:
>
> - Apply **data augmentation** (rotation, brightness adjustment) to improve robustness to real-world conditions.
> - Experiment with **transfer learning** using MobileNetV2 or EfficientNet for improved accuracy.
> - Build a **real-time traffic sign detection** pipeline using OpenCV and the exported TF-Lite model.
> - Deploy the TFJS model to a **web application** for interactive traffic sign classification.
