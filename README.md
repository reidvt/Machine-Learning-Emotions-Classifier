Machine Learning Emotions Classifier
This repository contains a machine learning project focused on classifying emotions and sentiment from text data using the GoEmotions dataset. The project implements multi-label emotion classification, 3-class sentiment analysis, and a custom "Trust" metric.

Project Overview
The core objective is to analyze and categorize human emotions expressed in text. By leveraging a large-scale dataset of Reddit comments, the models are trained to recognize 27 distinct emotions plus a neutral category.

Key Features
3-Class Sentiment Classification: Maps emotions into Positive, Negative, and Neutral/Mixed buckets.

Trust Metric: A specialized classification grouping emotions into High, Medium, and Low Trust categories.

Interpretability: Integration with SHAP and LIME to explain model predictions and highlight influential words.

Performance Metrics: Detailed evaluation using Accuracy, Precision, Recall, F1-Score, and ROC-AUC curves.

Dataset
The project utilizes the GoEmotions dataset (211,225 Reddit comments).

Total Labels: 28 (27 emotions + 1 neutral).

Sentiment Mapping: * Positive: Admiration, Amusement, Approval, Caring, Desire, Excitement, Gratitude, Joy, Love, Optimism, Pride, Relief.

Negative: Anger, Annoyance, Disappointment, Disapproval, Disgust, Embarrassment, Fear, Grief, Nervousness, Remorse, Sadness.

Neutral/Mixed: Confusion, Curiosity, Realization, Surprise, Neutral.

Methodology
1. Data Preprocessing
Text cleaning and tokenization.

Feature extraction using TF-IDF Vectorization (max 20,000 features, n-grams 1-2).

Handling class imbalance using class_weight='balanced'.

2. Modeling
Logistic Regression: Primary model for 3-class sentiment and trust classification.

One-Vs-Rest Classifier: Implemented for the complex multi-label emotion task.

3. Evaluation
Models were evaluated on a 20% test split:

Sentiment Accuracy: ~61.23%

Trust Accuracy: ~62.07%

Installation & Usage
Prerequisites
Python 3.x

pandas, numpy, scikit-learn, matplotlib

shap, lime

Running the Project
Ensure combined.csv is in your project directory.

Open ML_Emotion_Project_Notebook.ipynb in Google Colab or a local Jupyter environment.

Execute the cells to train the models and view the SHAP/LIME visualizations.

Author
Reid VanTrieste
