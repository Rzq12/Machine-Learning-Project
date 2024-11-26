# Table of Contents

1. [Project Description](#Project-Description)
2. [Methods and Models](#Methods-and-Models)
3. [Model Evaluation Results](#Model-Evaluation-Results)

# Project Description

Obesity is a significant health issue that, while often underestimated, has far-reaching consequences as a major risk factor for chronic diseases. Globally, obesity has become a pressing health concern, with prevalence increasing steadily each year. By 2022, there were **890 million obesity cases worldwide**, with adolescents contributing the most to this figure. The **World Health Organization (WHO)** has classified obesity as the **5th leading risk factor for global mortality**.

In Indonesia, obesity prevalence reached **28.7% in 2018**, with the highest mortality rate associated with obesity recorded at **80.46% in 2020**. The sharp rise in obesity cases underscores the urgent need for early preventive measures to mitigate the risks of obesity-related complications, particularly cardiovascular diseases.

### Objective

This project aims to leverage **machine learning** to classify different types of obesity based on factors influencing its development. By using predictive modeling, the project seeks to provide insights into obesity risk categories, which could serve as a foundation for early intervention strategies.

### Approach

To achieve the objectives, this project explores various machine learning methods to classify obesity types. The analysis involves:

1. **Comparing Raw and Augmented Data**:
   - Raw data: Original data without any adjustments.
   - SMOTE data: Augmented data generated using the Synthetic Minority Oversampling Technique (SMOTE) to address class imbalances.
2. **Algorithms Used**:
   - **Classical Methods**:
     - **Decision Tree**: A rule-based model for interpretable classification.
     - **Gradient Boosting**: An ensemble learning technique that improves prediction accuracy by combining weak learners.
   - **Advanced Methods**:
     - **Neural Networks**: A deep learning approach for handling complex patterns and relationships in data.

### Significance

By harnessing the power of machine learning, this project aims to:

- Provide a more accurate and systematic approach to detecting obesity risk.
- Highlight the impact of data preprocessing (e.g., SMOTE) on model performance.
- Identify the most effective algorithms for classifying obesity types.

This research not only demonstrates the application of machine learning in tackling global health issues but also provides a scalable framework for future studies on chronic disease risk detection.

---

# Methods and Models

This project explores machine learning algorithms to classify different types of obesity based on various contributing factors. The methodology focuses on leveraging both classical and advanced approaches, enhanced by hyperparameter tuning using Grid Search, to assess performance on raw and augmented datasets.

---

### 1. Data Preprocessing

To ensure the data is ready for modeling and that the results are meaningful, the following preprocessing steps were applied:

- **Data Cleaning**:

  - Handling missing values.
  - Encoding categorical variables using techniques such as one-hot encoding or label encoding.

- **Feature Scaling**:  
  Standardizing numerical features to improve model performance, especially for distance-based algorithms.

- **Handling Class Imbalance**:  
  The dataset underwent augmentation using **SMOTE (Synthetic Minority Oversampling Technique)** to generate synthetic samples for underrepresented classes, ensuring balanced training data.

---

### 2. Algorithms Used

#### Classical Methods:

1. **Decision Tree**:

   - A simple, interpretable algorithm that creates a tree-like structure for classification.
   - Strengths: Easy to understand and implement, handles non-linear relationships well.
   - Weaknesses: Prone to overfitting on raw data.

2. **Gradient Boosting**:
   - An ensemble technique that builds multiple weak learners (decision trees) in sequence, focusing on correcting the errors of previous trees.
   - Strengths: High accuracy and resilience to overfitting.
   - Weaknesses: Computationally expensive, sensitive to hyperparameters.

#### Advanced Methods:

1. **Neural Networks**:
   - A deep learning approach capable of capturing complex, non-linear patterns in data.
   - Architecture:
     - **Input Layer**: Receives the preprocessed features.
     - **Hidden Layers**: Includes fully connected layers with activation functions Selu.
     - **Output Layer**: Uses a softmax activation function for multi-class classification.
   - Strengths: Handles complex relationships, works well with large datasets.
   - Weaknesses: Requires significant computational resources and longer training times.

---

### 3. Hyperparameter Tuning: Grid Search

To optimize the model performance, **Grid Search** was employed to find the best hyperparameters for each algorithm. The following parameters were tuned:

- **Decision Tree**:

  - `max_depth`: Controls the maximum depth of the tree to prevent overfitting.
  - `min_samples_split`: Specifies the minimum number of samples required to split a node.

- **Gradient Boosting**:
  - `n_estimators`: Number of boosting stages.
  - `learning_rate`: Determines the contribution of each tree.
  - `max_depth`: Limits the depth of individual trees.

**Implementation**:  
Grid Search was implemented using **Scikit-learn's `GridSearchCV`** for classical models and manual looping for hyperparameter tuning in the neural network.

**Outcome**:  
Grid Search provided the optimal set of parameters for each model, leading to improved accuracy and reduced overfitting.

---

### 4. Evaluation Metrics

The models were evaluated on their ability to classify obesity types using the following metrics:

- **Accuracy**: The proportion of correctly classified samples.
- **Precision, Recall, and F1-Score**: Class-specific metrics to ensure balanced performance across all classes.
- **Confusion Matrix**: Visualizes true positives, false positives, and false negatives for each class.

---

### 5. Comparison: Raw vs. SMOTE Data

The project also compares the performance of models trained on raw data versus augmented (SMOTE) data:

- **Raw Data**: May result in biased predictions due to class imbalances.
- **SMOTE Data**: Helps the models better learn minority class patterns, improving overall and class-specific performance.

---

### 6. Tools and Libraries

The following tools and libraries were used:

- **Scikit-learn**: For classical models like Decision Tree, Gradient Boosting, and Grid Search.
- **TensorFlow/Keras**: For building, training, and tuning the neural network model.

By incorporating Grid Search for hyperparameter optimization, the project ensures that each model achieves its best possible performance, providing a robust framework for obesity risk classification.

---

# Model Evaluation Results

The table below summarizes the performance of different machine learning models on both **raw data** and **SMOTE data** for the obesity risk classification task. The evaluation metrics include **CV Mean Accuracy**, **Test Accuracy**, **Precision**, **Recall**, and **F1-Score**.

### 1. Results on Raw Data

| Method                | CV Mean Accuracy | Test Accuracy | Precision | Recall | F1-Score |
| --------------------- | ---------------- | ------------- | --------- | ------ | -------- |
| **Decision Tree**     | 0.7486           | 0.7600        | 0.7285    | 0.7600 | 0.7256   |
| **Gradient Boosting** | 0.8342           | 0.7800        | 0.7594    | 0.7800 | 0.7556   |
| **Neural Network**    | 0.4878           | 0.6900        | 0.4044    | 0.5843 | 0.4649   |

### 2. Results on SMOTE Data

| Method                | CV Mean Accuracy | Test Accuracy | Precision | Recall | F1-Score |
| --------------------- | ---------------- | ------------- | --------- | ------ | -------- |
| **Decision Tree**     | 0.9295           | 0.9409        | 0.9417    | 0.9409 | 0.9411   |
| **Gradient Boosting** | 0.9621           | 0.9622        | 0.9624    | 0.9622 | 0.9621   |
| **Neural Network**    | 0.7141           | 0.9031        | 0.7852    | 0.7863 | 0.7801   |

---

### Key Insights:

1. **Raw Data**:

   - Gradient Boosting outperformed other methods in terms of both **CV Mean Accuracy** (0.8342) and **Test Accuracy** (0.7800).
   - Decision Tree achieved decent performance but lagged behind Gradient Boosting.
   - Neural Network struggled with raw data, achieving the lowest metrics due to imbalanced class distribution.

2. **SMOTE Data**:
   - Gradient Boosting demonstrated the best overall performance, with the highest **Test Accuracy** (0.9622), **Precision**, **Recall**, and **F1-Score**.
   - Decision Tree also performed well on SMOTE data, achieving comparable metrics to Gradient Boosting.
   - Neural Network showed significant improvement compared to raw data, indicating the benefits of addressing class imbalance using SMOTE.

### Conclusion:

Using SMOTE data significantly improved the performance of all models, especially for the Neural Network and Decision Tree. Gradient Boosting consistently outperformed other models on both raw and SMOTE data, making it the most reliable method for this classification task.
