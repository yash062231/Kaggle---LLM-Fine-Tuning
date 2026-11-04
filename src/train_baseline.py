"""Train a first TF-IDF + logistic regression benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import (  # noqa: E402
    TARGET_COLUMNS,
    make_labels,
    make_text,
    read_csv,
    validate_test_data,
    validate_training_data,
)


def build_model() -> Pipeline:
    """Make a simple, explainable baseline before trying neural networks."""
    return Pipeline(
        steps=[
            # TF-IDF gives more weight to terms that are useful in a document
            # but not common in every document.
            ("tfidf", TfidfVectorizer(
                max_features=100_000,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
                strip_accents="unicode",

            )),
            # Logistic regression learns class probabilities from those features.
            ("classifier", LogisticRegression(
                C=2.0,
                max_iter=1_000,
                solver="lbfgs",
            )),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/raw/train.csv", help="Path to train.csv")
    parser.add_argument("--test", default="data/raw/test.csv", help="Optional path to test.csv")
    parser.add_argument("--output", default="outputs/baseline_submission.csv", help="Submission output path")
    parser.add_argument("--validation-size", type=float, default=0.2)
    args = parser.parse_args()

    train = read_csv(args.train)
    validate_training_data(train)
    labels = make_labels(train)
    texts = make_text(train)
    print(f"Loaded {len(train):,} training rows.")
    print("Label counts (A win, B win, tie):", labels.value_counts().sort_index().tolist())
    print("Text length percentiles (characters):", texts.str.len().quantile([0.5, 0.9, 0.99]).round().astype(int).to_dict())

    # Keep a holdout set that the model does not see during fitting.
    # Stratification keeps each class represented in both parts.
    x_fit, x_valid, y_fit, y_valid = train_test_split(
        texts,
        labels,
        test_size=args.validation_size,
        random_state=42,
        stratify=labels,
    )
    # Compare the model against two simple references on the same holdout.
    # Uniform ignores the data; the prior uses only the fitting split's class rates.
    uniform_probs = np.full((len(y_valid), 3), 1 / 3, dtype=float)
    prior = y_fit.value_counts(normalize=True).reindex([0, 1, 2]).to_numpy()
    prior_probs = np.tile(prior, (len(y_valid), 1))
    print(f"Uniform reference log loss: {log_loss(y_valid, uniform_probs, labels=[0, 1, 2]):.5f}")
    print(f"Training-prior reference log loss: {log_loss(y_valid, prior_probs, labels=[0, 1, 2]):.5f}")
    model = build_model()
    model.fit(x_fit, y_fit)
    valid_probs = model.predict_proba(x_valid)
    score = log_loss(y_valid, valid_probs, labels=[0, 1, 2])
    print(f"Validation multiclass log loss: {score:.5f} (lower is better)")

    test_path = Path(args.test)
    if not test_path.exists():
        print(f"No test CSV at {test_path}; stopping after validation. Add it to create a submission.")
        return

    test = read_csv(str(test_path))
    validate_test_data(test)

    # Refit on all labeled data for the final predictions.
    model.fit(texts, labels)
    test_probs = model.predict_proba(make_text(test))
    # Ensure columns always follow Kaggle's requested target-column order.
    class_to_column = {0: 0, 1: 1, 2: 2}
    ordered_probs = test_probs[:, [list(model.classes_).index(class_id) for class_id in class_to_column]]
    submission = pd.DataFrame(ordered_probs, columns=TARGET_COLUMNS)
    submission.insert(0, "id", test["id"])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Wrote {len(submission):,} predictions to {output_path}.")


if __name__ == "__main__":
    main()


