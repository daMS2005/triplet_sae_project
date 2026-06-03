# Triplet SAE Project

This repository studies triplet extraction as both:

- a `representation` problem, where we ask which sparse internal features are associated with structured extraction behavior, and
- an `adaptation` problem, where we train a compact instruction-tuned model to emit factual triples directly from text.

The project centers on factual `(subject, predicate, object)` extraction with optional qualifiers, for example:

```text
Marie Curie discovered polonium and won the Nobel Prize in Physics in 1903.
```

becomes:

```json
{
  "triples": [
    {
      "subject": "Marie Curie",
      "predicate": "discovered",
      "object": "polonium",
      "qualifiers": {}
    },
    {
      "subject": "Marie Curie",
      "predicate": "won",
      "object": "Nobel Prize in Physics",
      "qualifiers": {
        "year": "1903"
      }
    }
  ]
}
```

## Research Questions

The repository is built around three methodological questions:

1. `Can triplet extraction behavior be localized?`
   The interpretability path tests whether sparse autoencoder features become reliably more active when the model is operating in a triplet-extraction regime.

2. `Are those features merely correlated, or are they causally involved?`
   The intervention path tests this through ablation and steering rather than stopping at feature ranking.

3. `How far can a small open model be pushed with task-specific supervision?`
   The fine-tuning path trains and exports a Gemma 3 4B triplet extractor, then compares it against baseline behavior.

## Method Overview

The overall method has four stages.

### 1. Build labeled extraction data

Raw text is collected, cleaned, chunked, and paired with structured triplet labels.

Relevant scripts:

- [scripts/01_fetch_wikipedia.py](/Users/danielmora/triplet_sae_project/scripts/01_fetch_wikipedia.py)
- [scripts/01_fetch_dolma.py](/Users/danielmora/triplet_sae_project/scripts/01_fetch_dolma.py)
- [scripts/02_chunk_and_clean.py](/Users/danielmora/triplet_sae_project/scripts/02_chunk_and_clean.py)
- [scripts/03_extract_triplets_teacher.py](/Users/danielmora/triplet_sae_project/scripts/03_extract_triplets_teacher.py)
- [scripts/00_build_io_match.py](/Users/danielmora/triplet_sae_project/scripts/00_build_io_match.py)

The output of this stage is a matched JSONL format where each record contains:

- source text
- metadata such as `doc_id`, `title`, and `chunk_id`
- extracted triples

### 2. Collect activations and identify candidate SAE features

The interpretability pipeline works on model activations rather than just outputs.

Relevant scripts:

- [scripts/04_collect_activations.py](/Users/danielmora/triplet_sae_project/scripts/04_collect_activations.py)
- [scripts/06_feature_analysis.py](/Users/danielmora/triplet_sae_project/scripts/06_feature_analysis.py)

The activation collector records residual-stream activations and token-level SAE activations for selected Gemma layers. The feature analysis stage then compares matched triplet vs control conditions and ranks features using pooled statistics such as:

- mean activation
- max activation
- top-k token mean
- active fraction
- active-only mean

This is important because triplet-relevant signals are often sparse and position-specific, so simple sequence means can wash out useful structure.

### 3. Test causality through ablation and steering

This is a central part of the repository.

Relevant scripts:

- [scripts/07_ablation_and_steering.py](/Users/danielmora/triplet_sae_project/scripts/07_ablation_and_steering.py)
- [scripts/13_generate_with_sae_steering.py](/Users/danielmora/triplet_sae_project/scripts/13_generate_with_sae_steering.py)
- [configs/steering_feature_presets.json](/Users/danielmora/triplet_sae_project/configs/steering_feature_presets.json)

The core logic is:

- `ablation`: suppress selected SAE features and measure whether triplet behavior degrades
- `steering`: amplify selected SAE features during generation and inspect whether the output shifts toward more structured extraction behavior

This part of the project is meant to answer a methodological question, not just produce a demo:

- if a feature is ranked highly in triplet-vs-control analysis, does editing it actually change the model’s behavior in the expected direction?

### 4. Fine-tune and export a task-adapted model

Relevant scripts:

- [scripts/11_build_finetune_dataset.py](/Users/danielmora/triplet_sae_project/scripts/11_build_finetune_dataset.py)
- [scripts/12_finetune_gemma_lora.py](/Users/danielmora/triplet_sae_project/scripts/12_finetune_gemma_lora.py)
- [scripts/15_merge_lora_adapter.py](/Users/danielmora/triplet_sae_project/scripts/15_merge_lora_adapter.py)
- [scripts/16_run_quantized_triplet_inference.py](/Users/danielmora/triplet_sae_project/scripts/16_run_quantized_triplet_inference.py)
- [scripts/18_export_gptqmodel_native.py](/Users/danielmora/triplet_sae_project/scripts/18_export_gptqmodel_native.py)

This path treats triplet extraction as a standard supervised adaptation problem:

- build a chat-style dataset
- fine-tune Gemma 3 4B IT with LoRA
- merge the adapter if needed
- export quantized variants for deployment or comparison

## Repository Structure

```text
triplet_sae_project/
├── configs/                  # project configuration and steering presets
├── data/
│   ├── activations/          # activation caches
│   ├── labels/               # label JSONL outputs
│   ├── predictions/          # prediction files
│   ├── processed/            # processed chunked inputs
│   └── raw/                  # fetched raw corpora
├── scripts/                  # end-to-end pipeline and experiment runners
├── src/                      # reusable library code
├── tests/                    # lightweight regression tests
└── pyproject.toml
```

## Evaluation

The main evaluation entrypoints are:

- [scripts/08_evaluate.py](/Users/danielmora/triplet_sae_project/scripts/08_evaluate.py)
- [scripts/10_compare_triplet_extractors.py](/Users/danielmora/triplet_sae_project/scripts/10_compare_triplet_extractors.py)

The comparison workflow reports:

- exact SPO precision / recall / F1
- entity overlap
- predicate overlap
- qualifier usage statistics
- per-example comparison rows for manual inspection

This means the repository supports both:

- `behavioral evaluation` of generated triples, and
- `mechanistic evaluation` of whether internal sparse feature edits matter

## Public Model Release

The final public model release from this project is:

- [dams2005/gemma-3-4b-it-triplet-extractor](https://huggingface.co/dams2005/gemma-3-4b-it-triplet-extractor)

That release includes:

- a LoRA adapter
- a GPTQ 4-bit variant
- a GPTQ 8-bit variant

## Setup

```bash
python -m venv env
source env/bin/activate
pip install -e .
```

Install the optional fine-tuning stack when you want local model work:

```bash
pip install -e .[finetune]
```

If you use gated model checkpoints or external APIs, configure credentials through environment variables or a local `.env` file.

Notes:

- [requirements.txt](/Users/danielmora/triplet_sae_project/requirements.txt) mirrors the lightweight data and labeling stack.
- [requirements-finetune.txt](/Users/danielmora/triplet_sae_project/requirements-finetune.txt) mirrors the heavier Gemma / SAE / LoRA stack.
- GPTQ export depends on `gptqmodel`, which is intentionally kept outside the default install path because it is more environment-sensitive.

## Typical Workflows

### Build matched extraction data

```bash
python scripts/01_fetch_wikipedia.py
python scripts/02_chunk_and_clean.py
python scripts/03_extract_triplets_teacher.py
python scripts/00_build_io_match.py
```

### Run the interpretability path

```bash
python scripts/04_collect_activations.py
python scripts/06_feature_analysis.py
python scripts/07_ablation_and_steering.py
```

### Run preset-based steering generation

```bash
python scripts/13_generate_with_sae_steering.py
```

### Fine-tune Gemma

```bash
python scripts/11_build_finetune_dataset.py
python scripts/12_finetune_gemma_lora.py
```

### Compare extractor outputs

```bash
python scripts/10_compare_triplet_extractors.py
```

## Status

This repository includes:

- an end-to-end data and labeling pipeline
- a full SAE activation and feature-analysis path
- ablation and steering experiments as first-class methodology
- a fine-tuned Gemma 3 4B triplet extractor
- quantized export paths for downstream use
