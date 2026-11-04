# Kaggle LLM Preference Classification

A learning-first project for the Kaggle **LLM Classification Finetuning** competition. We will first build a small, understandable text-classification baseline, then improve it and learn transformer fine-tuning step by step.

## The task

For each row, Kaggle gives a prompt and two chatbot answers. Predict three probabilities: the judge prefers answer A, prefers answer B, or considers them tied. The score is multiclass log loss, so better-calibrated probabilities matter, not just the most likely class.

## First learning milestone: a TF-IDF baseline

This starter uses TF-IDF features and logistic regression. It is intentionally small enough to understand before we move to GPU training.

### 1. Set up Python

From this folder in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Add the competition files

Download `train.csv` and `test.csv` from the competition and put them here:

```text
data/raw/train.csv
data/raw/test.csv
```

These files are ignored by Git because competition data should not be committed.

### 3. Inspect and train

```powershell
python src/train_baseline.py
```

The script prints row counts, label balance, text-length percentiles, and validation log loss. If `test.csv` is present, it also writes `outputs/baseline_submission.csv`.

## What the first lesson teaches

- **TF-IDF:** represent words and short phrases by how informative they are in each example.
- **Logistic regression:** a simple classifier that can produce probabilities.
- **Holdout validation:** reserve data the model does not train on to estimate how well it generalizes.
- **Log loss:** penalize predictions that assign low probability to the correct answer.

## Planned stages

1. Inspect the data and verify the competition's expected format.
2. Establish and understand the TF-IDF baseline.
3. Improve validation and check for duplicates and A/B position effects.
4. Fine-tune a small transformer encoder and compare it fairly to the baseline.
5. Package a reproducible Kaggle notebook for offline scoring.

## Current limitations

This is a starter benchmark, not a competition-ready model. The validation split is random and may need improvement if there are related or duplicate examples. It treats the prompt and both answers as one text document, so it does not yet model the comparison structure explicitly. The transformer stage comes after we understand these baseline results.

## First baseline result

The first 80/20 stratified validation run produced multiclass log loss **1.14068**. Uniform probabilities score about **1.09861**, so this simple TF-IDF model is currently worse than that reference. This is useful information: we have a working training and scoring pipeline, but the model needs better features or a different approach before it is useful. The generated submission contains only the 3 rows in the downloaded sample `test.csv`.

## Reference model comparison

On the same 20% validation split, uniform probabilities scored **1.09861** and probabilities based on the training split's class proportions scored **1.09722**. The TF-IDF model scored **1.14068**, so it is currently worse than both references. The training script now prints these reference scores beside the model score during future runs.

## A/B-swap experiment

We trained on both the original fitting examples and versions with response A/B exchanged and the A/B winner labels swapped. On the unchanged validation split, log loss improved from **1.14068** to **1.13142**, but still did not beat the training-prior reference (**1.09722**). This suggests the symmetry augmentation helped a little, while the text representation remains weak. The experiment is in `src/train_swap_augmented.py`.

## Pairwise TF-IDF experiment

The pairwise model used signed A-minus-B TF-IDF features, absolute feature differences, and prompt TF-IDF features. On the same 20% holdout it scored **1.19379**, worse than the original TF-IDF baseline (**1.14068**) and the training-prior reference (**1.09722**). Explicit pairwise lexical features alone did not help; the next step is to inspect prediction quality and probability calibration before choosing another model change.

## Probability calibration experiment

Using 40,233 fitting rows, 5,748 calibration rows, and the unchanged 11,496-row validation split, temperature scaling selected **3.482**. It reduced validation log loss from **1.19375** to **1.06222**. Validation accuracy remained **43.17%**, as temperature scaling changes confidence, not the most likely class. On this same split, uniform and training-prior references scored **1.09861** and **1.09722**, respectively, so calibrated pairwise predictions beat both references on this holdout. The script prints these references on future runs. This is one validation split, so the result needs a later robustness check.

### Second split check (seed 123)

On a second split, the raw pairwise model scored **1.18054** and temperature-scaled predictions scored **1.05948**, with accuracy **44.03%** unchanged. The chosen temperature was **3.355**. The direction and size of the log-loss improvement were similar to the first split, which is encouraging; both results are still local validation estimates, not Kaggle leaderboard scores.

## Data format correction before transformer work

A full scan showed that `prompt`, `response_a`, and `response_b` are JSON arrays of conversation turns, not plain text. Every row parsed as an array; some answer arrays contain null turns. The shared loader now joins turns with `[TURN]` and converts null turns to empty text. The earlier TF-IDF and pairwise scores were produced from serialized JSON text, so they are exploratory only and should be rerun with this corrected preprocessing before drawing final conclusions.

## Corrected baseline and transformer preparation

The decoded-conversation TF-IDF baseline scored **1.13803** on the seed-42 80/20 holdout (uniform **1.09861**, training-prior **1.09722**). This is the first baseline after JSON turn normalization; earlier scores used serialized JSON strings and remain exploratory only.

A reproducible 2,000-row tokenizer probe with `microsoft/deberta-v3-small` found paired prompt/A/B lengths of median **526**, p90 **1,281**, p95 **1,751**, p99 **3,085**, and max **8,353** tokens. The model config supports **512** positions, so long examples must be truncated. The probe script is `src/inspect_token_lengths.py`.

The local venv has the Transformers tokenizer and SentencePiece dependencies, but PyTorch and a local GPU are unavailable. We will do fine-tuning in a Kaggle GPU notebook. The competition requires Notebook submissions, disables internet during scoring, permits freely available pretrained models, and expects `submission.csv`; the model weights must therefore be available as a notebook input for the offline run.

The full transformer validation run scored raw log loss **1.07926** and calibrated log loss **1.07933** on the untouched 20% holdout, compared with **1.09722** for the training-prior reference. The final Kaggle run trained on 90% of the rows and generated a valid three-row `submission.csv` using raw softmax probabilities. Details are in `reports/transformer_run_summary.md`.


## First transformer run (Kaggle GPU)

The starter notebook is `notebooks/transformer_baseline.ipynb`. Begin with `MODE = "smoke"`, which fits 2,048 rows and uses 512 rows each for calibration and validation. This checks that the Kaggle GPU, data input, model files, and training loop work before spending time on the full run. If successful, change to `MODE = "validation"` for the full 70/10/20 split. Only after reviewing its held-out log loss should you switch to `MODE = "submission"`.

In Kaggle, create a notebook for **LLM Classification Finetuning**, add a public `microsoft/deberta-v3-small` model input, and select a GPU accelerator. The notebook searches `/kaggle/input` for both the competition folder and model folder automatically. It tokenizes one batch at a time so full validation does not keep every tokenized split in RAM. The notebook is prepared locally; it has not yet been submitted.
