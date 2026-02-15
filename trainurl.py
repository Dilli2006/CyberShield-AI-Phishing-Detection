import pandas as pd
import pickle
import numpy as np
import re
from urllib.parse import urlparse

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack, csr_matrix

df = pd.read_csv("data/url_text.csv")
df["URL"] = df["URL"].str.lower()

X_text = df["URL"]
y = df["label"]


def extract_structural_features(url):
    parsed = urlparse(url)
    return [
        len(url),
        parsed.netloc.count("."),
        url.count("-"),
        sum(c.isdigit() for c in url),
        1 if parsed.scheme == "https" else 0,
    ]


structural_features = np.array(
    [extract_structural_features(url) for url in X_text]
)

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 6),
    max_features=10000
)

X_tfidf = vectorizer.fit_transform(X_text)
X_struct = csr_matrix(structural_features)
X_combined = hstack([X_tfidf, X_struct])

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42
)

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nHybrid URL Classifier Results")
print("--------------------------------")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

pickle.dump(model, open("models/url_hybrid_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/url_hybrid_vectorizer.pkl", "wb"))

print("\nHybrid URL model saved successfully.")
