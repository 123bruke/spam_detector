"""Train a Bayes/Naive Bayes detector and a PyTorch neural network."""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from spamdetector.bayes import BayesSpamDetector
from spamdetector.data import load_dataset
from spamdetector.models import SpamMLP
from spamdetector.visualize import create_plots

MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)


def metric_pack(y_true, probabilities, threshold: float = 0.5):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "threshold": float(threshold),
        "probabilities": probabilities,
        "predictions": predictions,
    }


def train_neural_network(X_train, y_train, X_val, y_val, epochs=30, batch_size=64):
    """Fit a small MLP, keeping the weights with lowest validation loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpamMLP(X_train.shape[1]).to(device)

    # The dataset is somewhat imbalanced; positive weighting helps the network
    # pay attention to the less frequent spam class.
    positives = max(int(np.sum(y_train)), 1)
    negatives = max(len(y_train) - positives, 1)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    train_x = torch.tensor(X_train.toarray(), dtype=torch.float32)
    train_y = torch.tensor(np.asarray(y_train), dtype=torch.float32)
    val_x = torch.tensor(X_val.toarray(), dtype=torch.float32, device=device)
    val_y = torch.tensor(np.asarray(y_val), dtype=torch.float32, device=device)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience, stale_epochs = 8, 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = float(criterion(val_logits, val_y).item())
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
        val_predictions = (val_probs >= 0.5).astype(int)
        val_f1 = float(f1_score(y_val, val_predictions, zero_division=0))
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(batch_losses)),
            "val_loss": val_loss,
            "val_f1": val_f1,
        })

        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, history, device


def neural_probabilities(model, matrix, device):
    model.eval()
    values = torch.tensor(matrix.toarray(), dtype=torch.float32, device=device)
    with torch.no_grad():
        return torch.sigmoid(model(values)).cpu().numpy()


def json_safe_metrics(metrics):
    return {key: value for key, value in metrics.items() if key not in {"probabilities", "predictions"}}


def main():
    parser = argparse.ArgumentParser(description="Train the SMS spam detectors.")
    parser.add_argument("--demo", action="store_true", help="Force the small synthetic offline demo dataset.")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum neural-network epochs (early stopping is enabled).")
    args = parser.parse_args()

    set_seed(42)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data, source = load_dataset(force_demo=args.demo)
    texts = data["text"].astype(str).to_numpy()
    labels = data["label_id"].astype(int).to_numpy()

    # 70% train, 15% validation, 15% final test; preserve class proportions.
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    val_fraction_of_train_val = 0.15 / 0.85
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=val_fraction_of_train_val,
        random_state=42,
        stratify=y_train_val,
    )

    # Model 1: Multinomial Naive Bayes (Bayes' theorem + token independence assumption).
    bayes = BayesSpamDetector(alpha=0.5)
    bayes.fit(x_train, y_train)
    bayes_test_probs = bayes.predict_proba(x_test)[:, list(bayes.classifier.classes_).index(1)]
    bayes_metrics = metric_pack(y_test, bayes_test_probs)
    joblib.dump(bayes, MODEL_DIR / "naive_bayes.joblib")

    # Model 2: TF-IDF features feeding a real trainable neural network (PyTorch MLP).
    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_features=4000,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    X_train = tfidf.fit_transform(x_train)
    X_val = tfidf.transform(x_val)
    X_test = tfidf.transform(x_test)
    nn_model, history, device = train_neural_network(
        X_train, y_train, X_val, y_val, epochs=max(1, args.epochs)
    )
    nn_test_probs = neural_probabilities(nn_model, X_test, device)
    nn_metrics = metric_pack(y_test, nn_test_probs)

    joblib.dump(tfidf, MODEL_DIR / "tfidf_vectorizer.joblib")
    torch.save(nn_model.state_dict(), MODEL_DIR / "neural_network.pt")
    nn_metadata = {
        "input_dim": int(X_train.shape[1]),
        "threshold": 0.5,
        "architecture": "TF-IDF -> Dense(128, ReLU) -> Dropout(0.35) -> Dense(64, ReLU) -> Dropout(0.25) -> Dense(1)",
        "device_used_for_training": str(device),
    }
    (MODEL_DIR / "neural_network_config.json").write_text(json.dumps(nn_metadata, indent=2), encoding="utf-8")

    results = {"Multinomial Naive Bayes": bayes_metrics, "Neural Network (PyTorch MLP)": nn_metrics}
    predictions = {
        "Multinomial Naive Bayes": bayes_metrics["predictions"],
        "Neural Network (PyTorch MLP)": nn_metrics["predictions"],
    }
    create_plots(
        data=data,
        results=results,
        history=history,
        bayes_model=bayes,
        y_test=y_test,
        predictions=predictions,
        output_dir=PLOT_DIR,
    )

    summary = {
        "dataset_source": source,
        "dataset_rows_after_deduplication": int(len(data)),
        "class_counts": {"ham": int((labels == 0).sum()), "spam": int((labels == 1).sum())},
        "split_counts": {"train": int(len(y_train)), "validation": int(len(y_val)), "test": int(len(y_test))},
        "random_seed": 42,
        "test_metrics": {
            "Multinomial Naive Bayes": json_safe_metrics(bayes_metrics),
            "Neural Network (PyTorch MLP)": json_safe_metrics(nn_metrics),
        },
        "neural_network_epochs_completed": len(history),
        "note": "Synthetic-demo metrics are for pipeline smoke testing only, not a trustworthy benchmark.",
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame({
        "text": x_test,
        "actual_label": np.where(y_test == 1, "spam", "ham"),
        "bayes_spam_probability": bayes_test_probs,
        "bayes_prediction": np.where(bayes_metrics["predictions"] == 1, "spam", "ham"),
        "neural_spam_score": nn_test_probs,
        "neural_prediction": np.where(nn_metrics["predictions"] == 1, "spam", "ham"),
    }).to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    (OUTPUT_DIR / "training_history.csv").write_text(pd.DataFrame(history).to_csv(index=False), encoding="utf-8")

    print("\nDataset source:", source)
    print(f"Rows: {len(data)} | train: {len(y_train)} | validation: {len(y_val)} | test: {len(y_test)}")
    print("\nHeld-out test metrics (threshold = 0.50):")
    metric_table = pd.DataFrame({name: json_safe_metrics(m) for name, m in results.items()}).T
    print(metric_table[["accuracy", "precision", "recall", "f1", "roc_auc"]].round(4).to_string())
    print(f"\nSaved models to: {MODEL_DIR}")
    print(f"Saved metrics and Seaborn/Matplotlib charts to: {OUTPUT_DIR}")
    if "Synthetic demo" in source:
        print("\nIMPORTANT: Demo data was used. Run again with internet access to download the full UCI dataset.")


if __name__ == "__main__":
    main()
