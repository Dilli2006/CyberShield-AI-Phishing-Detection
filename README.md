# CyberShield – Phishing and Spam Detection System

## Overview

CyberShield is a multi-modal machine learning system designed to detect:

- SMS / Email spam messages
- Phishing and malicious URLs

The system applies Natural Language Processing (NLP) and hybrid feature engineering techniques to classify potential cyber threats in real time.

---

## Problem Statement

Phishing attacks and scam messages are increasingly common and difficult to detect using rule-based systems.  
This project explores machine learning approaches to detect suspicious textual and structural patterns in URLs and messages.

---

## System Architecture

### 1. SMS Spam Detection
- TF-IDF (word-level vectorization)
- Multinomial Naive Bayes classifier
- Accuracy: ~97%

### 2. URL Phishing Detection
- Character-level TF-IDF (2–6 n-grams)
- Structural URL features:
  - URL length
  - Number of dots
  - Digit count
  - Hyphen count
  - HTTPS presence
- Logistic Regression (class-balanced)
- Accuracy: ~99.6%

---

## Model Evaluation

Both models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

The URL detection model demonstrates strong generalization on structured phishing datasets, while the SMS classifier performs reliably on spam detection tasks.

---

## Features

- Real-time classification using Streamlit
- Confidence score output
- Multi-page interface
- Performance dashboard
- Confusion matrix visualization
- Modular training and inference pipelines

---

## Tech Stack

- Python
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Matplotlib
- Seaborn
- NLTK

---

## How to Run

```bash
pip install -r requirements.txt
python -m streamlit run app.py
