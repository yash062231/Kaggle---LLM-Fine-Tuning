"""Utilities for the Kaggle LLM preference classification data."""

from __future__ import annotations

import json

import pandas as pd

TARGET_COLUMNS = ["winner_model_a", "winner_model_b", "winner_tie"]


def read_csv(path: str) -> pd.DataFrame:
    """Read a competition CSV and report helpful errors for common mistakes."""
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"The CSV is empty: {path}")
    return frame


def validate_training_data(frame: pd.DataFrame) -> None:
    """Check that the training file has the fields this baseline needs."""
    required = {"id", "prompt", "response_a", "response_b", *TARGET_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Training CSV is missing required columns: {', '.join(missing)}")


def validate_test_data(frame: pd.DataFrame) -> None:
    """Check that the test file has the fields this baseline needs."""
    required = {"id", "prompt", "response_a", "response_b"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Test CSV is missing required columns: {', '.join(missing)}")


def normalize_conversation(value: object) -> str:
    """Turn a CSV-encoded JSON list of conversation turns into readable text."""
    if pd.isna(value):
        return ""

    raw = str(value)
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        # Keep the function usable if a future row is already plain text.
        return raw

    if parsed is None:
        return ""
    if isinstance(parsed, list):
        # Null turns occur in this dataset; retain the turn boundary but no text.
        return " [TURN] ".join("" if turn is None else str(turn).strip() for turn in parsed)
    return str(parsed)


def clean_conversations(values: pd.Series) -> pd.Series:
    """Normalize a pandas text column containing serialized conversation turns."""
    return values.map(normalize_conversation)


def make_text(frame: pd.DataFrame) -> pd.Series:
    """Join prompt and both normalized answers into one document for TF-IDF."""
    prompt = clean_conversations(frame["prompt"])
    answer_a = clean_conversations(frame["response_a"])
    answer_b = clean_conversations(frame["response_b"])
    return (
        "Prompt: " + prompt + " Response A: " + answer_a + " Response B: " + answer_b
    )


def make_labels(frame: pd.DataFrame) -> pd.Series:
    """Convert the three one-hot target columns to class IDs 0, 1, and 2."""
    values = frame[TARGET_COLUMNS].fillna(0).to_numpy()
    if not ((values == 0) | (values == 1)).all():
        raise ValueError("Target columns must contain only 0/1 values.")
    row_sums = values.sum(axis=1)
    if not (row_sums == 1).all():
        raise ValueError(
            "Expected exactly one winning label per row; found rows with zero or multiple labels."
        )
    return pd.Series(values.argmax(axis=1), index=frame.index, name="label")
