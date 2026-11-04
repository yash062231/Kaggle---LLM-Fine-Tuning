"""Inspect DeBERTa token lengths to choose a fair sequence budget."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import clean_conversations  # noqa: E402


def token_lengths(tokenizer, texts: list[str]) -> np.ndarray:
    encoded = tokenizer(texts, add_special_tokens=False, truncation=False, padding=False)
    return np.asarray([len(ids) for ids in encoded["input_ids"]], dtype=np.int32)


def describe(name: str, lengths: np.ndarray) -> None:
    qs = np.quantile(lengths, [0.5, 0.9, 0.95, 0.99]).astype(int)
    print(f"{name}: median={qs[0]}, p90={qs[1]}, p95={qs[2]}, p99={qs[3]}, max={lengths.max()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/raw/train.csv", help="Path to train.csv")
    parser.add_argument("--model", default="microsoft/deberta-v3-small", help="Hugging Face model ID or local tokenizer path")
    parser.add_argument("--sample-size", type=int, default=2000, help="Rows to inspect (sampled reproducibly)")
    args = parser.parse_args()

    columns = ["prompt", "response_a", "response_b"]
    frame = pd.read_csv(args.train, usecols=columns)
    if len(frame) > args.sample_size:
        frame = frame.sample(n=args.sample_size, random_state=42)
    frame = frame.reset_index(drop=True)
    prompt = clean_conversations(frame["prompt"]).tolist()
    answer_a = clean_conversations(frame["response_a"]).tolist()
    answer_b = clean_conversations(frame["response_b"]).tolist()

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    prompt_lengths = token_lengths(tokenizer, prompt)
    a_lengths = token_lengths(tokenizer, answer_a)
    b_lengths = token_lengths(tokenizer, answer_b)

    # Treat prompt + A as the first segment and B as the second segment.
    # Pair encoding inserts the model's separator tokens between segments.
    first_segment = [f"Prompt: {p} [ANSWER A] {a}" for p, a in zip(prompt, answer_a)]
    second_segment = [f"Answer B: {b}" for b in answer_b]
    paired = tokenizer(
        first_segment,
        second_segment,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )["input_ids"]
    paired_lengths = np.asarray([len(ids) for ids in paired], dtype=np.int32)

    print(f"Rows inspected: {len(frame):,}")
    describe("Prompt", prompt_lengths)
    describe("Response A", a_lengths)
    describe("Response B", b_lengths)
    describe("Paired sequence including special tokens", paired_lengths)
    print("Fraction fitting without truncation:")
    for budget in (256, 384, 512, 768, 1024):
        print(f"  max_length={budget}: {(paired_lengths <= budget).mean():.1%}")
    print("Next step: use these measurements to choose a length budget and truncation policy.")


if __name__ == "__main__":
    main()
