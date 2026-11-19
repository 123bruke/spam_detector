"""Local API for the React SMS spam detection dashboard."""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from spamdetector.models import SpamMLP

MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
FRONTEND_DIST = ROOT / "frontend" / "dist"
REQUIRED_MODELS = (
    MODEL_DIR / "naive_bayes.joblib",
    MODEL_DIR / "tfidf_vectorizer.joblib",
    MODEL_DIR / "neural_network.pt",
    MODEL_DIR / "neural_network_config.json",
)

app = FastAPI(title="SMS Spam Detection API", version="1.0.0")


class PredictionRequest(BaseModel):
    message: str = Field(min_length=1, max_length=5000)
    model: Literal["naive_bayes", "neural_network"] = "naive_bayes"
    threshold: float = Field(default=0.5, ge=0.1, le=0.9)


@lru_cache(maxsize=1)
def load_models():
    missing = [path.name for path in REQUIRED_MODELS if not path.exists()]
    if missing:
        raise HTTPException(
            status_code=503,
            detail="Missing model files: " + ", ".join(missing) + ". Run `python train.py` first.",
        )

    bayes = joblib.load(REQUIRED_MODELS[0])
    tfidf = joblib.load(REQUIRED_MODELS[1])
    config = json.loads(REQUIRED_MODELS[3].read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neural_network = SpamMLP(config["input_dim"]).to(device)
    try:
        state = torch.load(REQUIRED_MODELS[2], map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(REQUIRED_MODELS[2], map_location=device)
    neural_network.load_state_dict(state)
    neural_network.eval()
    return bayes, tfidf, neural_network, device


def predict_neural_network(message: str, tfidf, model, device) -> float:
    matrix = tfidf.transform([message])
    features = torch.tensor(matrix.toarray(), dtype=torch.float32, device=device)
    with torch.no_grad():
        return float(torch.sigmoid(model(features)).item())


@app.get("/api/health")
def health():
    missing = [path.name for path in REQUIRED_MODELS if not path.exists()]
    return {
        "status": "ready" if not missing else "training_required",
        "models_ready": not missing,
        "missing_models": missing,
    }


@app.get("/api/metrics")
def metrics():
    metrics_path = OUTPUT_DIR / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Evaluation metrics not found. Run `python train.py` first.")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


@app.post("/api/predict")
def predict(request: PredictionRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Enter a message to analyze.")

    bayes, tfidf, neural_network, device = load_models()
    evidence = None
    if request.model == "naive_bayes":
        detail = bayes.explain(message, top_n=8)
        spam_score = detail["spam_probability"]
        ham_score = detail["ham_probability"]
        evidence = detail["evidence_tokens"]
        model_name = "Multinomial Naive Bayes"
        score_note = "Bayes posterior based on the message tokens and learned class frequencies."
    else:
        spam_score = predict_neural_network(message, tfidf, neural_network, device)
        ham_score = 1.0 - spam_score
        detail = None
        model_name = "Neural Network (PyTorch MLP)"
        score_note = "Neural-network score; it is not guaranteed to be a calibrated probability."

    return {
        "model": request.model,
        "model_name": model_name,
        "spam_score": spam_score,
        "ham_score": ham_score,
        "threshold": request.threshold,
        "is_spam": spam_score >= request.threshold,
        "evidence_tokens": evidence,
        "bayes_detail": detail,
        "score_note": score_note,
    }


if FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


@app.get("/")
def frontend():
    index = FRONTEND_DIST / "index.html"
    if index.is_file():
        return FileResponse(index)
    return {
        "message": "React dashboard is not built yet.",
        "instructions": "Run `cd frontend; npm install; npm run build`, then restart the API.",
    }
