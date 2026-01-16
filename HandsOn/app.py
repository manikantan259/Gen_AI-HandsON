import streamlit as st
import requests
import json

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="LLM Hands-on UI", layout="centered")
st.title("🧠 LLM Hands-on Interface (Groq)")

# -----------------------------
# ZERO-SHOT SENTIMENT
# -----------------------------
st.header("Zero-shot Sentiment Classification")

zero_text = st.text_area(
    "Enter feedback (Zero-shot)",
    "The app works fine, but the UI feels outdated and slow."
)

if st.button("Run Zero-shot"):
    res = requests.post(
        f"{API_BASE}/sentiment/zero-shot",
        json={"feedback": zero_text}
    )
    st.success(f"Sentiment: {res.json()['sentiment']}")

# -----------------------------
# FEW-SHOT SENTIMENT
# -----------------------------
st.header("Few-shot Sentiment Classification")

few_text = st.text_area(
    "Enter feedback (Few-shot)",
    "The app crashes often and is very slow."
)

if st.button("Run Few-shot"):
    res = requests.post(
        f"{API_BASE}/sentiment/few-shot",
        json={"feedback": few_text}
    )
    st.success(f"Sentiment: {res.json()['sentiment']}")

# -----------------------------
# INFORMATION EXTRACTION
# -----------------------------
st.header("Information Extraction (JSON)")

info_text = st.text_area(
    "Enter text to extract information",
    "Jane Smith joined Acme Corp as a Senior Engineer in March 2022 and currently works remotely from Texas."
)

if st.button("Extract Information"):
    res = requests.post(
        f"{API_BASE}/extract-info",
        json={"text": info_text}
    )

    try:
        parsed = json.loads(res.text)
        st.json(parsed)
    except Exception:
        st.code(res.text)
