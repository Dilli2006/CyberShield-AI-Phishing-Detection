# Phishing Scam Detection

Simple phishing/scam text classifier with a training script and Flask API.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Place a CSV at `data/dataset.csv` with columns:
- `text`: message content
- `label`: 0 for safe, 1 for phishing/scam

## Train

```bash
python train.py
```

## Run API

```bash
python app.py
```

Example request:

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"text\": \"Your account is locked, click here\"}"
```
