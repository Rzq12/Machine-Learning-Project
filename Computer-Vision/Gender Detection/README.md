# 👨‍🦱👩 Gender Classification Using Deep Learning (ResNet50)

## Dataset

[Gender-Dataset by Ashutosh Chauhan](https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset)

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

Gender classification using facial images has wide applications in security systems, demographic data collection, and human-computer interaction. Leveraging the power of **deep learning** and **transfer learning**, this project builds an accurate gender classification model using the **ResNet50** architecture pretrained on ImageNet.

The dataset consists of labeled facial images categorized into male and female groups. By training on this dataset, the model learns to extract and generalize visual features that distinguish gender.

---

## Objective

The primary objectives of this project are to:

- Develop a **binary classification model** that distinguishes between male and female faces.
- Use **ResNet50 with transfer learning** to leverage pretrained features.
- Provide a pipeline that allows **custom image prediction** with visualization and confidence scores.

---

## Approach

- Utilize **ResNet50** pretrained on ImageNet as a feature extractor.
- Freeze most of the base layers to retain learned features while fine-tuning the last few.
- Apply **data augmentation** to improve model generalization.
- Integrate **callbacks** such as early stopping, learning rate reduction, and model checkpointing.

---

## Methods and Models

### 1. Data Preprocessing

- **Image Augmentation (Training set):**

  - Rotation, translation, zoom, shear, and horizontal flip.

- **Preprocessing Function:**

  - ResNet50's `preprocess_input()` used to normalize input.

- **Image Size:**

  - All images resized to **224x224 pixels**.

### 2. Model Architecture

- **Base Model:**

  - `ResNet50(weights='imagenet', include_top=False)`
  - Last 20 layers unfrozen for fine-tuning.

- **Custom Head:**

  - `GlobalAveragePooling2D`
  - Dense layer with 128 units + Dropout
  - Dense layer with 64 units + Dropout
  - Output layer with 1 unit and `sigmoid` activation for binary classification.

```python
Model Summary:
ResNet50 (Frozen except last 20 layers) ➜ GAP ➜ Dropout ➜ Dense(128) ➜ Dropout ➜ Dense(64) ➜ Dense(1, sigmoid)
```

### 3. Callbacks & Regularization

- **EarlyStopping**: Stops training if validation loss doesn't improve for 5 epochs.
- **ReduceLROnPlateau**: Reduces learning rate when a plateau is detected.
- **ModelCheckpoint**: Saves the best-performing model based on validation accuracy.

### 4. Evaluation Metrics

- Accuracy (training & validation)
- Binary Crossentropy Loss
- Confidence scores per prediction
- Real-time visualization of predictions using `matplotlib`

### 5. Tools and Libraries

- `TensorFlow`, `Keras`
- `NumPy`, `Matplotlib`
- `ImageDataGenerator` for preprocessing
- `ResNet50` from `keras.applications`

---

## Model Evaluation Results

Training was performed using **ResNet50** transferred from the pretrained ImageNet model and adapted for the gender classification task (male vs female). The dataset was split into training and validation sets. The model was evaluated over 20 epochs but stopped early due to no significant improvement in validation accuracy.

### 📉 Training Results

| Epoch          | Train Accuracy | Train Loss | Val Accuracy | Val Loss   | Learning Rate                                          |
| -------------- | -------------- | ---------- | ------------ | ---------- | ------------------------------------------------------ |
| 1              | 74.84%         | 0.4955     | 98.89%       | 0.0279     | 1.0e-4                                                 |
| 2              | 91.35%         | 0.2386     | 98.06%       | 0.0685     | 1.0e-4                                                 |
| 3              | 93.84%         | 0.1672     | 97.92%       | 0.0485     | 1.0e-4                                                 |
| 4              | 94.08%         | 0.1714     | 98.61%       | 0.0439     | 1.0e-4 ⭢ 2.0e-5 (ReduceLROnPlateau)                    |
| 5              | 93.93%         | 0.1599     | 98.89%       | 0.0361     | 2.0e-5                                                 |
| 6              | 95.69%         | 0.1225     | **99.17%**   | 0.0216     | 2.0e-5 ✅ (Best Model Saved)                           |
| 7              | 95.21%         | 0.1296     | 98.89%       | 0.0316     | 2.0e-5                                                 |
| 8              | 95.22%         | 0.1391     | 99.17%       | 0.0272     | 2.0e-5                                                 |
| 9              | 95.23%         | 0.1270     | **99.31%**   | 0.0243     | 2.0e-5 ⭢ 4.0e-6 (ReduceLROnPlateau) ✅ (Model Updated) |
| 10             | 96.86%         | 0.0938     | 99.31%       | 0.0237     | 4.0e-6                                                 |
| 11             | 95.57%         | 0.1278     | 99.31%       | 0.0264     | 4.0e-6                                                 |
| ...            | —              | —          | —            | —          | —                                                      |
| **Best Epoch** | **6**          | **95.69%** | **99.17%**   | **0.0216** | —                                                      |

![Plot-Training-History](images/Screenshot%202025-05-20%20103144.png)

## Conclusion

- Using **ResNet50 with transfer learning** enabled efficient training even with a relatively small dataset.
- **Data augmentation** and **regularization** were essential to prevent overfitting and improve generalization.
- The model performed well in distinguishing between male and female faces with high confidence and visual interpretability.
- A prediction interface allows testing with custom images, supporting real-world applicability.

---

## Demo: Predict Gender from Custom Image

To test the model on your own images, use the provided widget to input a file path and get a prediction with visual feedback:

![Predict-Gender](images/Screenshot%202025-05-20%20112648.png)

---

> 💡 Future Work:
>
> - Expand the dataset for improved diversity and robustness.
> - Explore lightweight architectures like MobileNet for faster inference.
> - Integrate face detection and alignment preprocessing.
