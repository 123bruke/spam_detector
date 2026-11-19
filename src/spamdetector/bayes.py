"""Multinomial Naive Bayes and interpretable Bayes-score calculations."""
from __future__ import annotations

import numpy as np
from scipy.special import softmax
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
# multi sections of the Native
class BayesSpamDetector:
    """A text classifier whose posterior follows Bayes' theorem."""

    def __init__(self, alpha: float = 0.5, max_features: int = 20000):
        self.vectorizer = CountVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_features=max_features,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.classifier = MultinomialNB(alpha=alpha)

    def fit(self, texts, labels):
        matrix = self.vectorizer.fit_transform(texts)
        self.classifier.fit(matrix, labels)
        return self

    def predict_proba(self, texts):
        matrix = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(matrix)

    def explain(self, text: str, top_n: int = 8) -> dict:
        """Return the posterior and token contributions to spam-vs-ham log odds.

        Multinomial NB calculates log P(class) + sum(count(token) *
        log P(token|class)). Subtracting the ham score from the spam score
        shows which observed tokens push the result toward either class.
        """
        row = self.vectorizer.transform([text])
        classes = list(self.classifier.classes_)
        ham_index = classes.index(0)
        spam_index = classes.index(1)

        log_scores = self.classifier.class_log_prior_ + row @ self.classifier.feature_log_prob_.T
        log_scores = np.asarray(log_scores).ravel()
        probabilities = softmax(log_scores)
        spam_probability = float(probabilities[spam_index])

        feature_names = self.vectorizer.get_feature_names_out()
        counts = row.toarray().ravel()
        spam_minus_ham = (
            self.classifier.feature_log_prob_[spam_index]
            - self.classifier.feature_log_prob_[ham_index]
        )
        active = np.flatnonzero(counts)
        evidence = [
            {
                "token": str(feature_names[index]),
                "count": int(counts[index]),
                "log_odds_contribution": float(counts[index] * spam_minus_ham[index]),
            }
            for index in active
        ]
        evidence.sort(key=lambda item: abs(item["log_odds_contribution"]), reverse=True)

        return {
            "spam_probability": spam_probability,
            "ham_probability": float(probabilities[ham_index]),
            "log_score_ham": float(log_scores[ham_index]),
            "log_score_spam": float(log_scores[spam_index]),
            "prior_ham": float(np.exp(self.classifier.class_log_prior_[ham_index])),
            "prior_spam": float(np.exp(self.classifier.class_log_prior_[spam_index])),
            "evidence_tokens": evidence[:top_n],
        }
