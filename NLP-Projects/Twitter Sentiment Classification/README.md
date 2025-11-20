# Table of Contents

1. [Project Description](#Project-Description)
2. [Methods and Models](#Methods-and-Models)
3. [Model Evaluation Results](#Model-Evaluation-Results)

# Project Description

Social media has become a vast and diverse source of data, enabling comprehensive analysis of public dynamics and opinions. During the Indonesian presidential campaign season, the intensity of activity on platforms like **X (formerly Twitter)** increases significantly. These platforms serve as vital arenas for communication, mass mobilization, and ideological debates. The data generated during these campaigns provide valuable insights into voter behavior, public sentiment, and the dynamics of the elections.

This project focuses on addressing the challenges of processing and analyzing unstructured, high-volume **User-Generated Content (UGC)** from X. Specifically, it aims to classify tweets related to the **2024 Indonesian Presidential Election** into eight classes derived from the **Astagatra Framework**—a set of components representing national resilience:

1. **Ideology**: Fundamental values, principles, and worldviews guiding the nation.
2. **Politics**: Government systems, policies, and political processes.
3. **Economy**: Resource and financial management for societal prosperity.
4. **Socio-Culture**: Social values, cultural norms, and aspects of societal life influencing national integrity.
5. **Defense and Security**: National defense and internal security against threats or disturbances.
6. **Natural Resources**: Management and utilization of natural resources for national development and security.
7. **Geography**: Physical conditions, location, and environment influencing policies and national life.
8. **Demography**: Population structure, growth, and dynamics impacting policies and development.

The project leverages machine learning to build an accurate **multiclass text classification model** capable of categorizing tweets into these eight classes. This requires handling challenges such as:

- Ambiguity in tweet content.
- Sarcasm and implicit meanings.
- Variations in language usage and typos common in UGC.

The dataset provided for this project contains complete tweet texts along with their respective labels. To ensure fairness, participants are restricted to using this dataset without incorporating external data.

### Objectives

- **Voter Behavior Analysis**: Understanding motivations and concerns influencing voter participation.
- **Campaign Optimization**: Identifying key topics and tailoring strategies to improve engagement.
- **Election Outcome Prediction**: Developing predictive models based on sentiment and discussion patterns.
- **Disinformation Detection**: Analyzing patterns to identify and mitigate misinformation campaigns.
- **Polarization Mapping**: Understanding polarization dynamics through network and sentiment analysis.

The solutions developed in this project aim to address these challenges creatively and innovatively, contributing to a deeper understanding of electoral dynamics in Indonesia.

# Methods and Models

This project utilizes advanced machine learning methods and embedding techniques to classify tweets related to the **2024 Indonesian Presidential Election** into categories based on the **Astagatra Framework**. The combination of embedding techniques and robust classification models ensures high performance in handling complex and unstructured data like tweets.

### 1. Embedding Method: **BERT (Bidirectional Encoder Representations from Transformers)**

To capture the semantic meaning and contextual nuances of the tweets, the **BERT** embedding model is used.

- **Why BERT?**

  - BERT is a state-of-the-art transformer-based language model that generates high-quality embeddings by understanding the context of words in a bidirectional manner.
  - It handles challenges such as:
    - **Ambiguity**: Deriving meaning from context.
    - **Sarkasme**: Detecting implied meanings.
    - **Noise in Text**: Addressing typos and colloquial expressions common in social media.

- **Process**:  
  Each tweet is tokenized and passed through the BERT model to produce dense vector representations. These embeddings serve as input features for the classification models.

---

### 2. Classification Models

Two machine learning models were implemented to classify the BERT embeddings into the eight Astagatra classes:

#### a. **Logistic Regression**

- A simple yet effective linear model that predicts class probabilities based on the input features.
- **Advantages**:
  - Fast and computationally efficient.
  - Suitable for handling linearly separable classes.

#### b. **Random Forest**

- An ensemble learning method that builds multiple decision trees and combines their predictions to improve accuracy and robustness.
- **Advantages**:
  - Handles non-linear relationships in data.
  - Resistant to overfitting due to averaging across multiple trees.

---

### 3. Model Comparison

- Logistic Regression and Random Forest models were trained and evaluated using metrics such as accuracy, precision, recall, and F1-score.
- Random Forest typically performs better in handling complex patterns in the data, while Logistic Regression provides a baseline for comparison.

---

### 4. Workflow Summary

1. **Data Preprocessing**: Tweets are cleaned and tokenized.
2. **Feature Extraction**: BERT embeddings are generated for each tweet.
3. **Model Training**: Both Logistic Regression and Random Forest are trained using the extracted embeddings.
4. **Evaluation**: Models are evaluated using labeled data to measure classification performance.

This combination of methods and models ensures a robust approach to tackling the challenges of classifying tweets into multiple categories effectively.

# Model Evaluation Results

The models were evaluated on their ability to classify tweets into the eight classes of the **Astagatra Framework**. The evaluation was based on the **accuracy metric**, which measures the proportion of correct predictions made by the model.

### Performance Summary

| **Model**           | **Accuracy** |
| ------------------- | ------------ |
| Logistic Regression | 0.711        |
| Random Forest       | 0.736        |

### Insights

1. **Logistic Regression**:

   - Achieved an accuracy of **71.1%**, demonstrating its effectiveness as a baseline model.
   - While efficient and fast, its performance was slightly limited when handling the complex patterns in the dataset.

2. **Random Forest**:
   - Outperformed Logistic Regression with an accuracy of **73.6%**.
   - The ensemble nature of Random Forest allowed it to better capture the non-linear relationships in the data, making it more robust for this task.

### Conclusion

The evaluation results indicate that Random Forest is a more suitable model for this classification task, given its ability to handle complex patterns and higher accuracy. However, Logistic Regression remains a strong baseline for comparison due to its simplicity and computational efficiency.

Further improvements could be explored by:

- Fine-tuning the BERT embeddings for this specific dataset.
- Experimenting with more advanced models like gradient boosting or deep learning classifiers.
