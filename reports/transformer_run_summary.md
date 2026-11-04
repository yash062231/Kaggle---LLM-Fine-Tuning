# Transformer run summary

## Validation run

- Model: `microsoft/deberta-v3-small` (Kaggle model input)
- Maximum sequence length: 512 tokens
- Split: 70% fitting, 10% calibration, 20% untouched validation
- Epochs: 1
- Learning rate: `2e-5`
- Raw validation log loss: **1.07926**
- Calibrated validation log loss: **1.07933**
- Validation accuracy: **0.3887**

The raw probabilities were slightly better on the held-out validation split, so the final submission used raw softmax probabilities.

## Submission run

- Fit rows: 51,729
- Calibration rows: 5,748
- Validation rows: 0
- Output: `submission.csv` with 3 rows and columns `id`, `winner_model_a`, `winner_model_b`, `winner_tie`

The generated CSV remains a Kaggle output artifact and is not committed to this repository. The notebook can recreate it in Kaggle with `MODE = "submission"`.
