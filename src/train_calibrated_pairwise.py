"""Test probability calibration for the pairwise TF-IDF model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import make_labels, read_csv, validate_training_data  # noqa: E402
from train_pairwise_baseline import clean_text, make_features  # noqa: E402


def probabilities_with_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Adjust confidence while preserving each row's most likely class."""
    logits = np.log(np.clip(probabilities, 1e-12, 1.0))
    return softmax(logits / temperature, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/raw/train.csv", help="Path to train.csv")
    parser.add_argument("--split-seed", type=int, default=42, help="Seed for repeatable data splits")
    args = parser.parse_args()

    data = read_csv(args.train)
    validate_training_data(data)
    labels = make_labels(data)

    # Keep the final validation rows untouched. Split the remaining data into
    # model-fitting and probability-calibration portions.
    fitting_rows, valid_rows = train_test_split(
        np.arange(len(data)), test_size=0.2, random_state=args.split_seed, stratify=labels
    )
    train_rows, calibration_rows = train_test_split(
        fitting_rows,
        test_size=0.125,
        random_state=args.split_seed + 1,
        stratify=labels.iloc[fitting_rows],
    )
    train_data = data.iloc[train_rows].copy()
    calibration_data = data.iloc[calibration_rows].copy()
    valid_data = data.iloc[valid_rows].copy()
    y_train = labels.iloc[train_rows].reset_index(drop=True)
    y_calibration = labels.iloc[calibration_rows].reset_index(drop=True)
    y_valid = labels.iloc[valid_rows].reset_index(drop=True)

    print(f"Model-fitting rows: {len(train_data):,}")
    print(f"Calibration rows: {len(calibration_data):,}")
    print(f"Untouched validation rows: {len(valid_data):,}")
    uniform_probs = np.full((len(y_valid), 3), 1 / 3, dtype=float)
    prior = y_train.value_counts(normalize=True).reindex([0, 1, 2]).to_numpy()
    prior_probs = np.tile(prior, (len(y_valid), 1))
    print(f"Uniform reference log loss: {log_loss(y_valid, uniform_probs, labels=[0, 1, 2]):.5f}")
    print(f"Training-prior reference log loss: {log_loss(y_valid, prior_probs, labels=[0, 1, 2]):.5f}")

    response_vectorizer = TfidfVectorizer(
        max_features=60_000, ngram_range=(1, 2), min_df=2,
        sublinear_tf=True, strip_accents="unicode",
    )
    response_text = pd.concat(
        [clean_text(train_data["response_a"]), clean_text(train_data["response_b"])],
        ignore_index=True,
    )
    response_vectorizer.fit(response_text)

    prompt_vectorizer = TfidfVectorizer(
        max_features=20_000, ngram_range=(1, 2), min_df=2,
        sublinear_tf=True, strip_accents="unicode",
    )
    prompt_vectorizer.fit(clean_text(train_data["prompt"]))

    x_train = make_features(train_data, response_vectorizer, prompt_vectorizer)
    x_calibration = make_features(calibration_data, response_vectorizer, prompt_vectorizer)
    x_valid = make_features(valid_data, response_vectorizer, prompt_vectorizer)

    model = LogisticRegression(C=2.0, max_iter=1_000, solver="lbfgs")
    model.fit(x_train, y_train)
    calibration_probs = model.predict_proba(x_calibration)
    valid_probs = model.predict_proba(x_valid)

    # Choose temperature only from the calibration split, never from validation.
    def calibration_loss(log_temperature: float) -> float:
        adjusted = probabilities_with_temperature(calibration_probs, np.exp(log_temperature))
        return log_loss(y_calibration, adjusted, labels=[0, 1, 2])

    result = minimize_scalar(
        calibration_loss,
        bounds=(np.log(0.25), np.log(8.0)),
        method="bounded",
    )
    temperature = float(np.exp(result.x))
    calibrated_valid = probabilities_with_temperature(valid_probs, temperature)

    raw_loss = log_loss(y_valid, valid_probs, labels=[0, 1, 2])
    calibrated_loss = log_loss(y_valid, calibrated_valid, labels=[0, 1, 2])
    raw_accuracy = accuracy_score(y_valid, valid_probs.argmax(axis=1))
    calibrated_accuracy = accuracy_score(y_valid, calibrated_valid.argmax(axis=1))
    print(f"Chosen temperature from calibration split: {temperature:.3f}")
    print(f"Raw validation log loss: {raw_loss:.5f}")
    print(f"Temperature-scaled validation log loss: {calibrated_loss:.5f}")
    print(f"Raw / scaled validation accuracy: {raw_accuracy:.4f} / {calibrated_accuracy:.4f}")
    print("Accuracy stays the same because temperature changes confidence, not class ranking.")


if __name__ == "__main__":
    main()


