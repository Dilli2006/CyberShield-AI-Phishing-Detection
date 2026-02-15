# 🛡 CyberShield – AI-Based Phishing & Scam Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 🚀 Project Overview

CyberShield is a multi-modal AI system designed to detect:

- 📩 SMS / Email Spam Messages  
- 🌐 Phishing & Malicious URLs  

The system uses Natural Language Processing (NLP) and Hybrid Feature Engineering to identify cyber threats in real time.

---

## 🎯 Problem Statement

Phishing attacks and scam messages are increasing rapidly.  
Traditional rule-based systems fail to generalize to unseen attacks.

CyberShield leverages machine learning models trained on real datasets to classify suspicious patterns using:

- Character-level pattern analysis
- Structural URL features
- Text-based NLP modeling

---

## 🧠 Architecture

### 📩 SMS Detection Pipeline
- TF-IDF (word-level)
- Naive Bayes Classifier
- Accuracy: ~97%

### 🌐 URL Phishing Detection Pipeline
- Character-level TF-IDF (2–6 grams)
- Structural URL Features:
  - URL length
  - Number of dots
  - Digit count
  - HTTPS presence
- Logistic Regression (class-balanced)
- Accuracy: ~99.6%

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|--------|----------|-----------|--------|----------|
| SMS Detection | 97% | 96% | 95% | 96% |
| URL Detection | 99.6% | 99% | 100% | 99% |

✔ Confusion Matrix Visualization  
✔ Performance Dashboard  
✔ Confidence Scoring  

---

## 🖥 Features

- Real-time detection interface (Streamlit)
- Hybrid ML architecture
- Confidence score output
- Performance dashboard
- Clean modular code structure

---

## ⚙️ Tech Stack

- Python
- Scikit-learn
- Pandas & NumPy
- TF-IDF Vectorization
- Logistic Regression
- Naive Bayes
- Streamlit
- Matplotlib & Seaborn

---

## 🚀 How To Run

```bash
pip install -r requirements.txt
python -m streamlit run app.py
