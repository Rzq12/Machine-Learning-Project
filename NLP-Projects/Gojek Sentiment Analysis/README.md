# 🛵 Gojek App Review Sentiment Analysis (Google Play Store)

## Dataset

Scraped from the **Google Play Store** using [`google-play-scraper`](https://pypi.org/project/google-play-scraper/).  
App: **Gojek** (`com.gojek.app`) — ≥ 11,000 Indonesian user reviews.

## 📑 Table of Contents

- [Project Description](#Project-Description)
- [Objective](#Objective)
- [Approach](#Approach)
- [Methods and Models](#Methods-and-Models)
  - [1. Data Collection (Scraping)](#1-data-collection-scraping)
  - [2. Data Preprocessing](#2-data-preprocessing)
  - [3. Feature Extraction](#3-feature-extraction)
  - [4. Model Schemes](#4-model-schemes)
  - [5. Tools and Libraries](#5-tools-and-libraries)

- [Model Evaluation Results](#Model-Evaluation-Results)
- [Conclusion](#conclusion)

---

## Project Description

User reviews on the **Google Play Store** provide rich, real-world feedback on app quality and user experience. This project builds a **binary sentiment classification system** for Indonesian-language reviews of the **Gojek** super-app by scraping review data directly from the Play Store and training three different machine learning schemes.

Reviews are labeled based on their star rating:

- **Rating 1–3** → `negatif`
- **Rating 4–5** → `positif`

The project benchmarks three approaches with different feature extraction methods and model architectures: a classic TF-IDF + SVM pipeline, a Word2Vec + Random Forest approach, and a transformer-based **IndoBERT** deep learning model.

---

## Objective

The primary objectives of this project are to:

- Scrape and build a **custom Indonesian review dataset** from the Gojek Google Play Store page.
- Build and compare **three sentiment classification schemes** using different feature extraction and modeling strategies.
- Achieve a **test accuracy of ≥ 92%** using the IndoBERT transformer model.
- Demonstrate end-to-end inference capability across all three trained models.

---

## Approach

- Collect ≥ 11,000 Gojek reviews via `google-play-scraper` using both newest and most-relevant sort orders, with deduplication by `reviewId`.
- Apply **Indonesian-specific text preprocessing** using Sastrawi for stemming and stopword removal.
- Split data into **80% train / 20% test** using stratified sampling to maintain class balance.
- Train and evaluate three distinct schemes, then compare their performance side-by-side.

---

## Methods and Models

### 1. Data Collection (Scraping)

- **Tool:** `google-play-scraper` library
- **App ID:** `com.gojek.app` (Language: `id`, Country: `id`)
- **Collection Strategy:** Two sort methods — `Sort.NEWEST` and `Sort.MOST_RELEVANT` — to maximize diversity.
- **Deduplication:** Reviews filtered by unique `reviewId` to remove duplicates.
- **Labeling:** Star rating mapped to binary sentiment — ratings 1–3 → `negatif`, ratings 4–5 → `positif`.
- **Output:** CSV with columns: `review_id`, `user_name`, `review`, `rating`, `thumbs_up`, `date`, `sentiment`.

### 2. Data Preprocessing

- **Tokenization:** NLTK `word_tokenize`
- **Stopword Removal:** NLTK English stopwords + Sastrawi Indonesian stopwords
- **Stemming:** Sastrawi `StemmerFactory` for Indonesian morphological stemming
- **Cleaning:** Removal of URLs, punctuation, numbers, and excess whitespace
- **BERT Variant:** For IndoBERT, text is cleaned **without stemming** to preserve contextual meaning

### 3. Feature Extraction

| Scheme   | Method             | Details                                                                                |
| -------- | ------------------ | -------------------------------------------------------------------------------------- |
| Scheme 1 | **TF-IDF**         | `max_features=30,000`, unigram + bigram, `sublinear_tf=True`, `min_df=3`               |
| Scheme 2 | **Word2Vec**       | `vector_size=100`, `window=5`, `min_count=2`, 10 epochs; averaged into sentence vector |
| Scheme 3 | **BERT Tokenizer** | `indobenchmark/indobert-base-p1`, `max_length=128`                                     |

### 4. Model Schemes

#### Scheme 1: SVM + TF-IDF

- **Algorithm:** `LinearSVC` (C=1.0, `max_iter=3000`)
- **Feature:** TF-IDF sparse matrix
- **Split:** 80/20 stratified

#### Scheme 2: Random Forest + Word2Vec

- **Algorithm:** `RandomForestClassifier` (`n_estimators=200`, `n_jobs=-1`)
- **Feature:** Word2Vec averaged sentence embeddings (100-dim)
- **Split:** 80/20 stratified

#### Scheme 3: IndoBERT Transformer (Deep Learning)

- **Model:** `indobenchmark/indobert-base-p1`
- **Framework:** HuggingFace `Trainer` API with `AutoModelForSequenceClassification`
- **Split:** 80/20 stratified (with separate `bert_text` column, no stemming)
- **Evaluation Metric:** Accuracy per epoch via `EvalPrediction`

```python
Model Pipeline:
Raw Review → Preprocessing → Feature Extraction → Classifier → Positif / Negatif
  [TF-IDF → LinearSVC]
  [Word2Vec → Random Forest]
  [IndoBERT Tokenizer → IndoBERT + Classification Head]
```

### 5. Tools and Libraries

- `google-play-scraper` for data collection
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `wordcloud` for EDA and visualization
- `nltk`, `Sastrawi` for Indonesian text preprocessing
- `scikit-learn` for TF-IDF, SVM, Random Forest, and evaluation metrics
- `gensim` for Word2Vec embeddings
- `transformers`, `torch`, `datasets`, `accelerate` for IndoBERT fine-tuning
- `tqdm` for progress tracking

---

## Model Evaluation Results

All three schemes are trained on the same 80/20 stratified split of the Gojek review dataset. The target accuracy is **≥ 92%** on the test set.

### 🔖 Training Schemes Summary

| #   | Scheme                   | Feature Extraction           | Algorithm              |
| --- | ------------------------ | ---------------------------- | ---------------------- |
| 1   | SVM + TF-IDF             | TF-IDF (unigram + bigram)    | LinearSVC              |
| 2   | Random Forest + Word2Vec | Word2Vec (avg, 100-dim)      | RandomForestClassifier |
| 3   | IndoBERT Transformer     | BERT Tokenizer (max_len=128) | indobert-base-p1       |

### 🏆 Model Performance Comparison

| Scheme        | Feature Extraction | Test Accuracy |
| ------------- | ------------------ | :-----------: |
| SVM           | TF-IDF             |     ~92%+     |
| Random Forest | Word2Vec           |     ~85%+     |
| **IndoBERT**  | **BERT Tokenizer** | **~95%+** ✅  |

> IndoBERT is the best-performing scheme, achieving the target accuracy of ≥ 92% by leveraging contextual transformer embeddings pretrained on Indonesian text.

### 📊 Dataset Statistics

| Split        | Samples      |
| ------------ | ------------ |
| Training Set | ~8,800+      |
| Test Set     | ~2,200+      |
| **Total**    | **≥ 11,000** |

---

## Conclusion

- A **custom dataset** of ≥ 11,000 Gojek Google Play reviews was successfully collected and labeled using an automated scraping pipeline.
- **Three schemes** were benchmarked, demonstrating the trade-off between classical ML (fast, interpretable) and transformer-based deep learning (slower, higher accuracy).
- **IndoBERT** achieved the best test accuracy, confirming the advantage of domain-specific pretrained Indonesian language models for sentiment classification.
- The **inference pipeline** supports real-time predictions from all three models, allowing direct comparison on new review texts.

---

> 💡 Future Work:
>
> - Expand to **multi-class sentiment** (e.g., very negative, neutral, very positive) using 5-star rating buckets.
> - Fine-tune **IndoBERT** further with a larger dataset for improved robustness.
> - Build a **Streamlit web app** for live Gojek review sentiment prediction.
> - Compare with other Indonesian transformer models such as `IndoRoBERTa` or `mBERT`.
