import streamlit as st
import pickle
import re
import numpy as np
from urllib.parse import urlparse
from nltk.corpus import stopwords
from scipy.sparse import hstack, csr_matrix
import pandas as pd
from metrics import SMS_METRICS, URL_METRICS

sms_model = pickle.load(open("models/model.pkl", "rb"))
sms_vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

url_model = pickle.load(open("models/url_hybrid_model.pkl", "rb"))
url_vectorizer = pickle.load(open("models/url_hybrid_vectorizer.pkl", "rb"))

st.set_page_config(page_title="CyberShield", page_icon="shield", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    [data-testid="stSidebar"] .stRadio label {
        color: #e0e0e0 !important;
        font-weight: 500;
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d2ff, #7b2ff7, #ff6fd8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        text-align: center;
        color: #a0a0b8;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 2.5rem;
        letter-spacing: 0.3px;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 2rem;
        backdrop-filter: blur(16px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
        margin-bottom: 1.5rem;
    }

    .result-safe {
        background: linear-gradient(135deg, rgba(0, 200, 83, 0.12), rgba(0, 200, 83, 0.04));
        border: 1px solid rgba(0, 200, 83, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }

    .result-safe h3 {
        color: #00c853;
        font-size: 1.5rem;
        margin: 0 0 0.3rem 0;
    }

    .result-safe p {
        color: #a0a0b8;
        margin: 0;
    }

    .result-danger {
        background: linear-gradient(135deg, rgba(255, 23, 68, 0.12), rgba(255, 23, 68, 0.04));
        border: 1px solid rgba(255, 23, 68, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }

    .result-danger h3 {
        color: #ff1744;
        font-size: 1.5rem;
        margin: 0 0 0.3rem 0;
    }

    .result-danger p {
        color: #a0a0b8;
        margin: 0;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        backdrop-filter: blur(12px);
    }

    .metric-card h4 {
        color: #a0a0b8;
        font-weight: 400;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0 0 0.5rem 0;
    }

    .metric-card .value {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
    }

    .section-title {
        color: #ffffff;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(123, 47, 247, 0.4);
        display: inline-block;
    }

    .mode-label {
        color: #d0d0e0;
        font-size: 0.95rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, #7b2ff7, #00d2ff);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 2.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(123, 47, 247, 0.3);
    }

    .stButton > button:hover {
        box-shadow: 0 6px 25px rgba(123, 47, 247, 0.5);
        transform: translateY(-1px);
    }

    .stTextArea textarea, .stTextInput input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        font-size: 0.95rem !important;
    }

    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: rgba(123, 47, 247, 0.5) !important;
        box-shadow: 0 0 0 2px rgba(123, 47, 247, 0.15) !important;
    }

    .stRadio > div {
        gap: 0.5rem;
    }

    .footer-text {
        text-align: center;
        color: #606078;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        letter-spacing: 0.5px;
    }

    .dashboard-header {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.3rem;
    }

    .dashboard-sub {
        color: #a0a0b8;
        text-align: center;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }

    .info-block {
        background: rgba(123, 47, 247, 0.08);
        border-left: 3px solid #7b2ff7;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        color: #d0d0e0;
        font-size: 0.92rem;
        line-height: 1.7;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">CyberShield</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">AI-Powered Phishing & Scam Detection System</div>', unsafe_allow_html=True)

page = st.sidebar.radio("Navigation", ["Detection System", "Performance Dashboard"])


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)


def extract_structural_features(url):
    parsed = urlparse(url)
    return [
        len(url),
        parsed.netloc.count("."),
        url.count("-"),
        sum(c.isdigit() for c in url),
        1 if parsed.scheme == "https" else 0,
    ]


if page == "Detection System":

    mode = st.radio("Select Detection Mode", ["Message Detection", "URL Detection"])

    if mode == "Message Detection":

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Message Analysis</div>', unsafe_allow_html=True)
        message = st.text_area("Enter the SMS or email content below", height=150)

        if st.button("Analyze Message"):
            if message.strip():
                cleaned = clean_text(message)
                vector = sms_vectorizer.transform([cleaned])

                prediction = sms_model.predict(vector)[0]
                prob = sms_model.predict_proba(vector)[0]

                if prediction == 1:
                    st.markdown(f"""
                    <div class="result-danger">
                        <h3>SPAM DETECTED</h3>
                        <p>Confidence: {prob[1]*100:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-safe">
                        <h3>MESSAGE IS SAFE</h3>
                        <p>Confidence: {prob[0]*100:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Please enter a message to analyze.")
        st.markdown('</div>', unsafe_allow_html=True)

    elif mode == "URL Detection":

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">URL Analysis</div>', unsafe_allow_html=True)
        url = st.text_input("Enter the URL to verify")

        if st.button("Analyze URL"):
            if url.strip():
                url = url.lower()

                text_features = url_vectorizer.transform([url])
                struct_features = csr_matrix([extract_structural_features(url)])
                combined = hstack([text_features, struct_features])

                prediction = url_model.predict(combined)[0]
                prob = url_model.predict_proba(combined)[0]

                if prediction == 0:
                    st.markdown(f"""
                    <div class="result-danger">
                        <h3>PHISHING DETECTED</h3>
                        <p>Confidence: {prob[0]*100:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-safe">
                        <h3>URL IS LEGITIMATE</h3>
                        <p>Confidence: {prob[1]*100:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Please enter a URL to analyze.")
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "Performance Dashboard":

    st.markdown('<div class="dashboard-header">Performance Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-sub">Model accuracy and evaluation metrics</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">SMS Spam Detection</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (metric, value) in enumerate(SMS_METRICS.items()):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{metric}</h4>
                <div class="value">{value:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">URL Phishing Detection</div>', unsafe_allow_html=True)
    cols2 = st.columns(4)
    for i, (metric, value) in enumerate(URL_METRICS.items()):
        with cols2[i]:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{metric}</h4>
                <div class="value">{value:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-block">
        SMS model uses TF-IDF vectorization with Naive Bayes classification.<br>
        URL model uses hybrid character-level TF-IDF combined with structural feature engineering and Logistic Regression.<br>
        High recall is prioritized to minimize false negatives in phishing detection.<br>
        Hybrid feature engineering improves generalization across unseen data.
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="footer-text">CyberShield  |  NLP + Hybrid ML Architecture  |  AI & Data Science Project</div>', unsafe_allow_html=True)
