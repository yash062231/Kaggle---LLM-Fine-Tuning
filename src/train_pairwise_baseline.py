"""Train a text model with explicit pairwise response-comparison features."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import clean_conversations, make_labels, read_csv, validate_training_data  # noqa: E402


def clean_text(series: pd.Series) -> pd.Series:
    """Decode serialized conversation turns for the vectorizers."""
    return clean_conversations(series)


def make_features(frame: pd.DataFrame, response_vectorizer, prompt_vectorizer):
    """Build signed A-minus-B text features, absolute differences, and prompt features."""
    response_a = response_vectorizer.transform(clean_text(frame["response_a"]))
    response_b = response_vectorizer.transform(clean_text(frame["response_b"]))
    difference = response_a - response_b

    # Signed features tell the model which answer contains each term.
    # Absolute differences help it recognize when answers are lexically similar.
    absolute_difference = abs(difference)
    prompt_features = prompt_vectorizer.transform(clean_text(frame["prompt"]))
    return hstack([difference, absolute_difference, prompt_features], format="csr")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/raw/train.csv", help="Path to train.csv")
    parser.add_argument("--validation-size", type=float, default=0.2)
    args = parser.parse_args()

    train = read_csv(args.train)
    validate_training_data(train)
    labels = make_labels(train)
    fit_rows, valid_rows = train_test_split(
        np.arange(len(train)),
        test_size=args.validation_size,
        random_state=42,
        stratify=labels,
    )
    fit_data = train.iloc[fit_rows].copy()
    valid_data = train.iloc[valid_rows].copy()
    y_fit = labels.iloc[fit_rows].reset_index(drop=True)
    y_valid = labels.iloc[valid_rows].reset_index(drop=True)

    # Fit vocabularies on the fitting split only to avoid validation leakage.
    response_vectorizer = TfidfVectorizer(
        max_features=60_000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    response_text = pd.concat(
        [clean_text(fit_data["response_a"]), clean_text(fit_data["response_b"])],
        ignore_index=True,
    )
    response_vectorizer.fit(response_text)

    prompt_vectorizer = TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    prompt_vectorizer.fit(clean_text(fit_data["prompt"]))

    x_fit = make_features(fit_data, response_vectorizer, prompt_vectorizer)
    x_valid = make_features(valid_data, response_vectorizer, prompt_vectorizer)
    print(f"Fitting rows: {len(fit_data):,}; validation rows: {len(valid_data):,}")
    print(f"Feature columns: {x_fit.shape[1]:,}")

    # Reference predictions learned only from the fitting split.
    uniform = np.full((len(y_valid), 3), 1 / 3, dtype=float)
    prior = y_fit.value_counts(normalize=True).reindex([0, 1, 2]).to_numpy()
    prior_probs = np.tile(prior, (len(y_valid), 1))
    print(f"Uniform reference log loss: {log_loss(y_valid, uniform, labels=[0, 1, 2]):.5f}")
    print(f"Training-prior reference log loss: {log_loss(y_valid, prior_probs, labels=[0, 1, 2]):.5f}")

    model = LogisticRegression(C=2.0, max_iter=1_000, solver="lbfgs")
    model.fit(x_fit, y_fit)
    probabilities = model.predict_proba(x_valid)
    score = log_loss(y_valid, probabilities, labels=[0, 1, 2])
    print(f"Pairwise TF-IDF validation log loss: {score:.5f} (lower is better)")
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    actual = y_valid.to_numpy()
    print(f"Validation accuracy: {accuracy_score(actual, predictions):.4f}")
    print("Confusion matrix (rows=actual A/B/tie, columns=predicted A/B/tie):")
    print(confusion_matrix(actual, predictions, labels=[0, 1, 2]))
    print("Top-label confidence buckets (confidence range: rows, mean confidence, accuracy):")
    for lower, upper in [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.01)]:
        selected = (confidence >= lower) & (confidence < upper)
        if selected.any():
            print(f"  [{lower:.1f}, {upper:.2f}): n={selected.sum():,}, "
                  f"mean_confidence={confidence[selected].mean():.3f}, "
                  f"accuracy={(predictions[selected] == actual[selected]).mean():.3f}")


if __name__ == "__main__":
    main()


