# Triplet SAE Project

This project aims to discover and use sparse autoencoder (SAE) features that control triplet extraction behavior in a language model.

Goal: produce structured `(subject, relation, object)` outputs from text, similar to how JSON enforces structure, but focused on knowledge triplets.

## Problem Statement

We want to:

1. Collect text data from Wikipedia.
2. Use a stronger teacher model to extract triplets from that text.
3. Record activations from a target model while processing the same text.
4. Train/analyze an SAE over those activations.
5. Identify SAE features associated with better triplet generation.
6. Run ablation/intervention experiments to test causality.
7. Boost model behavior using those features.

## High-Level Pipeline

1. **Ingest data** from Wikipedia (articles -> cleaned text chunks).
2. **Generate labels** with a stronger model (triplets per chunk).
3. **Run target model** and cache intermediate activations.
4. **Train SAE** on cached activations.
5. **Map features -> behavior** via correlation and probing.
6. **Test causality** via ablation, activation patching, and steering.
7. **Evaluate** with triplet precision/recall/F1 and robustness checks.

## Proposed Repository Layout

```text
triplet_sae_project/
  README.md
  PLAN.md
  configs/
    data.yaml
    teacher.yaml
    model.yaml
    sae.yaml
    experiments.yaml
  data/
    raw/
    processed/
    labels/
    activations/
  scripts/
    01_fetch_wikipedia.py
    02_chunk_and_clean.py
    03_extract_triplets_teacher.py
    04_collect_activations.py
    05_train_sae.py
    06_feature_analysis.py
    07_ablation_and_steering.py
    08_evaluate.py
  src/
    data/
    teacher/
    models/
    sae/
    experiments/
    eval/
    utils/
  notebooks/
  outputs/
  tests/
```

## Data and Labeling

- **Source:** Wikipedia text.
- **Unit:** paragraph/chunk with metadata (`article_id`, `title`, `url`, `chunk_id`).
- **Teacher output format:** list of normalized triplets:
  - `subject` (string)
  - `relation` (string)
  - `object` (string)
  - optional: confidence score + evidence span

Recommended filtering:

- Deduplicate triplets per chunk.
- Drop malformed or empty fields.
- Normalize casing and punctuation.
- Keep relation vocabulary consistent when possible.

## SAE + Mechanistic Analysis

At minimum, track:

- model/layer/hook point for activations,
- SAE dictionary size and sparsity target,
- feature activation statistics by triplet quality bucket,
- feature relevance scores (correlation/probe weights),
- intervention effect size (before vs after).

Causal checks:

- Zero-ablation of selected features,
- Activation patching from high-quality contexts,
- Positive steering (scale selected features up).

## Evaluation

Primary metrics:

- Triplet precision/recall/F1 against teacher or curated validation set.
- Structural validity rate (`subject`, `relation`, `object` all present).

Secondary metrics:

- Calibration/confidence quality (if available),
- Robustness across domains and article lengths,
- Intervention stability (no catastrophic degradation on non-triplet text).

## Quick Start (Implementation Order)

1. Build dataset pipeline (`01` + `02` scripts).
2. Generate teacher triplets (`03`).
3. Capture target-model activations (`04`).
4. Train SAE (`05`).
5. Analyze candidate features (`06`).
6. Run ablations/interventions (`07`).
7. Evaluate and report (`08`).

Detailed roadmap: see `PLAN.md`.

