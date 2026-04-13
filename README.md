# 🧠 ML Emotion & Trust Classifier

> Multi-label emotion classification, sentiment analysis, and trust detection from text — powered by GoEmotions + Explainable AI.

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## Overview

This project analyzes and categorizes human emotions expressed in text using the **GoEmotions** dataset — 211,225 labeled Reddit comments. It implements three classification tasks:

- **27-class emotion detection** (multi-label)
- **3-class sentiment analysis** (Positive / Negative / Neutral)
- **Custom Trust metric** (High / Medium / Low Trust)

Explainability is a core component: SHAP and LIME are used to surface which words drive each prediction.

---

## Key Results

| Task | Accuracy |
|---|---|
| Sentiment Classification | ~61.2% |
| Trust Classification | ~62.1% |

Evaluated on a held-out 20% test split.

---

## Dataset

**GoEmotions** — 211,225 human-annotated Reddit comments across 28 labels (27 emotions + Neutral).

### Sentiment Mapping

| Sentiment | Emotions |
|---|---|
| **Positive** | Admiration, Amusement, Approval, Caring, Desire, Excitement, Gratitude, Joy, Love, Optimism, Pride, Relief |
| **Negative** | Anger, Annoyance, Disappointment, Disapproval, Disgust, Embarrassment, Fear, Grief, Nervousness, Remorse, Sadness |
| **Neutral / Mixed** | Confusion, Curiosity, Realization, Surprise, Neutral |

---

## Methodology

### 1. Data Preprocessing
- Text cleaning and tokenization
- TF-IDF Vectorization (max 20,000 features, unigrams + bigrams)
- Class imbalance handled via `class_weight='balanced'`

### 2. Modeling
- **Logistic Regression** — primary model for sentiment and trust classification
- **One-vs-Rest Classifier** — used for multi-label emotion detection across all 27 classes

### 3. Explainability
- **SHAP** — global feature importance; identifies which words most influence predictions across the dataset
- **LIME** — local explanations; diagnoses individual misclassifications at the sample level

### 4. Evaluation
- Accuracy, Precision, Recall, F1-Score, ROC-AUC curves

---

## Project Structure
