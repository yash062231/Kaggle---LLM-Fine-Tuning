"""Compare a TF-IDF model trained with A/B-swapped examples."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import make_labels, make_text, read_csv, validate_training_data  # noqa: E402
from train_baseline import build_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/raw/train.csv", help="Path to train.csv")
    parser.add_argument("--validation-size", type=float, default=0.2)
    args = parser.parse_args()

    train = read_csv(args.train)
    validate_training_data(train)
    labels = make_labels(train)

    # Split row IDs so we can use raw rows to make correctly swapped examples.
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

    # Add a second version of every fitting row with the two answers exchanged.
    # If A originally won, B wins in the swapped version; ties stay ties.
    swapped_data = fit_data.copy()
    swapped_data["response_a"] = fit_data["response_b"].to_numpy()
    swapped_data["response_b"] = fit_data["response_a"].to_numpy()
    swapped_labels = y_fit.map({0: 1, 1: 0, 2: 2})

    x_fit = pd.concat(
        [make_text(fit_data).reset_index(drop=True), make_text(swapped_data).reset_index(drop=True)],
        ignore_index=True,
    )
    y_fit_augmented = pd.concat([y_fit, swapped_labels], ignore_index=True)
    x_valid = make_text(valid_data)

    print(f"Original fitting rows: {len(fit_data):,}; after swapping: {len(x_fit):,}")
    print("Validation rows:", f"{len(valid_data):,}")

    # Compare against constant predictions on the exact same holdout.
    uniform = np.full((len(y_valid), 3), 1 / 3, dtype=float)
    prior = y_fit.value_counts(normalize=True).reindex([0, 1, 2]).to_numpy()
    prior_probs = np.tile(prior, (len(y_valid), 1))
    print(f"Uniform reference log loss: {log_loss(y_valid, uniform, labels=[0, 1, 2]):.5f}")
    print(f"Training-prior reference log loss: {log_loss(y_valid, prior_probs, labels=[0, 1, 2]):.5f}")

    model = build_model()
    model.fit(x_fit, y_fit_augmented)
    valid_probs = model.predict_proba(x_valid)
    score = log_loss(y_valid, valid_probs, labels=[0, 1, 2])
    print(f"Swap-augmented validation log loss: {score:.5f} (lower is better)")


if __name__ == "__main__":
    main()
