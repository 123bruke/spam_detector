"""Interactive Streamlit dashboard for the trained spam detectors."""
from __future__ import annotations
# importing the functions 
import json
import sys
from pathlib import Path

import joblib # saving the progress and reusing streming conditions 
import numpy as np
import pandas as pd
import streamlit as st 
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from spamdetector.models import SpamMLP

MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

st.set_page_config(page_title="Bayes + Neural Spam Detector", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_models():
    required = [
        MODEL_DIR / "naive_bayes.joblib",
        MODEL_DIR / "tfidf_vectorizer.joblib",
        MODEL_DIR / "neural_network.pt",
        MODEL_DIR / "neural_network_config.json",
    ]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing model files: " + ", ".join(missing) + ". Run `python train.py` first.")

    bayes = joblib.load(required[0])
    tfidf = joblib.load(required[1])
    config = json.loads(required[3].read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nn_model = SpamMLP(config["input_dim"]).to(device)
    try:
        state = torch.load(required[2], map_location=device, weights_only=True)
    except TypeError:  # compatibility with older PyTorch
        state = torch.load(required[2], map_location=device)
    nn_model.load_state_dict(state)
    nn_model.eval()
    return bayes, tfidf, nn_model, config, device


def predict_nn(text, tfidf, model, device):
    matrix = tfidf.transform([text])
    x = torch.tensor(matrix.toarray(), dtype=torch.float32, device=device)
    with torch.no_grad():
        return float(torch.sigmoid(model(x)).item())


st.title("🛡️ SMS Spam Detection Lab")
st.caption("Two approaches, one task: Bayes' theorem with Multinomial Naive Bayes and a trainable PyTorch neural network.")

try:
    bayes, tfidf, nn_model, nn_config, device = load_models()
except Exception as error:
    st.error(str(error))
    st.code("# From the project root\npip install -r requirements.txt\npython train.py\nstreamlit run app.py", language="bash")
    st.stop()

metrics_path = OUTPUT_DIR / "metrics.json"
summary = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
if "Synthetic demo" in summary.get("dataset_source", ""):
    st.warning("The current models were trained on the small synthetic demo dataset. Retrain with the full UCI SMS dataset for meaningful evaluation.")

left, right = st.columns([1.3, 1])
with left:
    st.subheader("Test a message")
    message = st.text_area(
        "Enter an SMS or short message",
        value="Congratulations! You have won a free prize. Call now to claim your reward.",
        height=145,
        placeholder="Type or paste a message here...",
    )
    model_choice = st.radio(
        "Choose detector",
        ["Multinomial Naive Bayes (Bayes' theorem)", "Neural Network (PyTorch MLP)"],
        horizontal=False,
    )
    threshold = st.slider("Spam decision threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05)
    run = st.button("Analyze message", type="primary", use_container_width=True)

with right:
    st.subheader("How the decision works")
    st.markdown("**Naive Bayes** estimates the posterior using the prior class frequency and token likelihoods:")
    st.latex(r"P(Spam\mid message)=\frac{P(message\mid Spam)P(Spam)}{P(message)}")
    st.markdown("**Neural network** learns nonlinear patterns from TF-IDF vectors and returns a sigmoid score. Its score is not automatically a calibrated probability.")

if run:
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        if model_choice.startswith("Multinomial"):
            detail = bayes.explain(message, top_n=10)
            spam_score = detail["spam_probability"]
            ham_score = detail["ham_probability"]
        else:
            spam_score = predict_nn(message, tfidf, nn_model, device)
            ham_score = 1.0 - spam_score
            detail = None

        is_spam = spam_score >= threshold
        st.divider()
        result_col, probability_col = st.columns([1, 1])
        with result_col:
            if is_spam:
                st.error("🚨 SPAM DETECTED")
            else:
                st.success("✅ LIKELY LEGITIMATE (HAM)")
            st.write(f"Decision threshold: **{threshold:.0%}**")
        with probability_col:
            st.metric("Spam score", f"{spam_score:.2%}")
            st.progress(min(max(spam_score, 0.0), 1.0))
            st.caption(f"Ham score: {ham_score:.2%}")

        if detail is not None:
            st.subheader("Bayes calculation for this message")
            st.write("The model uses a multinomial token likelihood. In log space, it calculates one score for each class:")
            st.latex(r"S_c=\log P(c)+\sum_w n_w\log P(w\mid c)")
            st.latex(r"P(Spam\mid message)=\frac{e^{S_{spam}}}{e^{S_{spam}}+e^{S_{ham}}}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Prior P(Spam)", f"{detail['prior_spam']:.2%}")
            c2.metric("Log score: spam", f"{detail['log_score_spam']:.3f}")
            c3.metric("Log score: ham", f"{detail['log_score_ham']:.3f}")
            st.markdown("**Tokens pushing the decision** (positive contribution supports spam; negative supports ham):")
            if detail["evidence_tokens"]:
                evidence = pd.DataFrame(detail["evidence_tokens"])
                evidence["direction"] = np.where(evidence["log_odds_contribution"] >= 0, "Spam", "Ham")
                evidence = evidence.rename(columns={"token": "Token", "count": "Count", "log_odds_contribution": "Log-odds contribution", "direction": "Evidence direction"})
                st.dataframe(evidence, use_container_width=True, hide_index=True)
            else:
                st.info("No tokens from the trained vocabulary were found in this message. The prediction is then driven mostly by the learned prior.")
        else:
            st.info("The neural network score is produced by a sigmoid over its output. Use the separate test metrics to assess generalization; the score is not guaranteed to be calibrated.")

st.divider()
st.header("Model evaluation and visualizations")
if summary:
    st.caption(f"Dataset: {summary.get('dataset_source', 'unknown')} · {summary.get('dataset_rows_after_deduplication', '?')} messages · stratified train/validation/test split")
    metrics = summary.get("test_metrics", {})
    if metrics:
        metric_df = pd.DataFrame(metrics).T
        wanted = [column for column in ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"] if column in metric_df.columns]
        st.dataframe((metric_df[wanted] * 100).round(2).rename_axis("Model").rename(columns=lambda c: c.replace("_", " ").title()), use_container_width=True)

plot_files = [
    ("01_class_distribution.png", "Dataset class distribution"),
    ("02_model_comparison.png", "Model metrics comparison"),
    ("03_confusion_matrices.png", "Confusion matrices"),
    ("04_roc_curves.png", "ROC curves"),
    ("05_neural_network_training.png", "Neural-network training history"),
    ("06_bayes_token_evidence.png", "Naive Bayes token evidence"),
]
for i in range(0, len(plot_files), 2):
    cols = st.columns(2)
    for col, (filename, caption) in zip(cols, plot_files[i:i+2]):
        path = OUTPUT_DIR / "plots" / filename
        with col:
            if path.exists():
                st.image(str(path), caption=caption, use_container_width=True)

st.caption("Educational project: predictions can be wrong. Do not use this demo as the sole basis for blocking important communications.")
