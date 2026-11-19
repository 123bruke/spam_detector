"""Seaborn/Matplotlib plots for model analysis."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve

sns.set_theme(style="whitegrid", context="notebook")


def create_plots(data, results, history, bayes_model, y_test, predictions, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Class distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.countplot(data=data, x="label", order=["ham", "spam"], color="#4C78A8", ax=ax)
    ax.set(title="SMS dataset: class distribution", xlabel="Message class", ylabel="Number of messages")
    fig.tight_layout()
    fig.savefig(output_dir / "01_class_distribution.png", dpi=180)
    plt.close(fig)

    # 2. Model metric comparison
    rows = []
    for name, metrics in results.items():
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            rows.append({"Model": name, "Metric": metric.replace("_", " ").title(), "Score": metrics.get(metric, np.nan)})
    metric_df = pd.DataFrame(rows).dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=metric_df, x="Metric", y="Score", hue="Model", palette="deep", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set(title="Test-set performance comparison", ylabel="Score (0–1)")
    fig.tight_layout()
    fig.savefig(output_dir / "02_model_comparison.png", dpi=180)
    plt.close(fig)

    # 3. Side-by-side confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for ax, (name, y_pred) in zip(axes, predictions.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, labels=[0, 1], display_labels=["Ham", "Spam"],
            cmap="Blues", colorbar=False, ax=ax, values_format="d",
        )
        ax.set_title(f"{name}: confusion matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "03_confusion_matrices.png", dpi=180)
    plt.close(fig)

    # 4. ROC curves
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, metrics in results.items():
        fpr, tpr, _ = roc_curve(y_test, metrics["probabilities"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves on held-out test data")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "04_roc_curves.png", dpi=180)
    plt.close(fig)

    # 5. Neural network learning curve
    if history:
        history_df = pd.DataFrame(history)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        sns.lineplot(data=history_df, x="epoch", y="train_loss", label="Train", ax=axes[0])
        sns.lineplot(data=history_df, x="epoch", y="val_loss", label="Validation", ax=axes[0])
        axes[0].set(title="Neural-network loss", ylabel="Binary cross-entropy")
        sns.lineplot(data=history_df, x="epoch", y="val_f1", ax=axes[1], color="purple")
        axes[1].set(title="Validation F1 by epoch", ylabel="F1 score", ylim=(0, 1.05))
        fig.tight_layout()
        fig.savefig(output_dir / "05_neural_network_training.png", dpi=180)
        plt.close(fig)

    # 6. Most spam-indicative terms from the NB model
    vectorizer = bayes_model.vectorizer
    clf = bayes_model.classifier
    classes = list(clf.classes_)
    h_idx, s_idx = classes.index(0), classes.index(1)
    log_odds = clf.feature_log_prob_[s_idx] - clf.feature_log_prob_[h_idx]
    names = vectorizer.get_feature_names_out()
    top_spam = np.argsort(log_odds)[-12:]
    top_ham = np.argsort(log_odds)[:12]
    chosen = np.concatenate([top_ham, top_spam])
    term_df = pd.DataFrame({"Term": names[chosen], "Spam-vs-ham log likelihood": log_odds[chosen]})
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=term_df, x="Spam-vs-ham log likelihood", y="Term", color="#54A24B", ax=ax)
    ax.axvline(0, color="black", linewidth=1)
    ax.set(title="Naive Bayes: terms associated with each class")
    fig.tight_layout()
    fig.savefig(output_dir / "06_bayes_token_evidence.png", dpi=180)
    plt.close(fig)
