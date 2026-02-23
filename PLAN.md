# Implementation Plan

This plan translates the project scope into concrete milestones and deliverables.

## Milestone 0 - Project Setup (Day 0-1)

Deliverables:

- Python environment and dependency lockfile.
- Base folder structure (`configs`, `scripts`, `src`, `data`, `outputs`, `tests`).
- Config templates for data, teacher, model, SAE, and experiments.

Acceptance criteria:

- `python -m scripts.01_fetch_wikipedia --help` style entry points exist.
- All config files load without errors.

## Milestone 1 - Wikipedia Data Pipeline (Day 1-3)

Tasks:

- Implement Wikipedia article fetcher.
- Clean and normalize text.
- Chunk text into model-friendly windows (token or sentence-based).
- Store chunk-level metadata.

Data schema (minimum):

- `article_id`
- `title`
- `source_url`
- `chunk_id`
- `chunk_text`

Acceptance criteria:

- Reproducible dataset build from seed list/category/query.
- Processed chunks saved as JSONL/Parquet.
- Basic data quality report (chunk count, avg length, empty ratio).

## Milestone 2 - Teacher Triplet Extraction (Day 3-5)

Tasks:

- Implement prompt template for triplet extraction.
- Add retries, validation, and rate-limit handling.
- Parse and normalize teacher output into canonical triplet schema.
- Add confidence/evidence if available.

Triplet schema:

```json
{
  "article_id": "str",
  "chunk_id": "str",
  "triplets": [
    {
      "subject": "str",
      "relation": "str",
      "object": "str",
      "confidence": 0.0,
      "evidence_span": "optional str"
    }
  ]
}
```

Acceptance criteria:

- >95% parse success rate on teacher responses.
- Malformed outputs filtered and logged.
- Labeled dataset persisted with version tag.

## Milestone 3 - Activation Collection (Day 5-7)

Tasks:

- Choose target model and hook points (layer/module).
- Run inference over labeled chunks.
- Save activations aligned to `chunk_id` and token positions.
- Add batching/checkpointing to survive interruptions.

Acceptance criteria:

- Activation files can be reloaded and matched to labels.
- Runtime/memory profile documented.

## Milestone 4 - SAE Training (Day 7-10)

Tasks:

- Train SAE on selected activation tensors.
- Sweep key hyperparameters:
  - dictionary size,
  - sparsity coefficient,
  - learning rate,
  - batch size.
- Save checkpoints and reconstruction/sparsity metrics.

Acceptance criteria:

- Stable training curve.
- Feature activations are sparse and non-degenerate.
- Best checkpoint selected with clear selection rule.

## Milestone 5 - Feature Attribution to Triplet Quality (Day 10-12)

Tasks:

- Define quality labels/score per sample (triplet precision proxy).
- Rank SAE features by association with quality.
- Train simple probe/logistic head on SAE codes for interpretability.
- Produce top feature report.

Acceptance criteria:

- Reproducible ranked feature list.
- At least one candidate feature set for intervention.

## Milestone 6 - Causal Experiments (Day 12-14)

Tasks:

- Zero-ablation on candidate features.
- Positive steering (scale-up feature activations).
- Optional patching from high-quality contexts.
- Measure downstream triplet extraction changes.

Acceptance criteria:

- Causal effect measured with confidence intervals.
- Negative controls included (random feature sets).

## Milestone 7 - Evaluation and Reporting (Day 14-16)

Tasks:

- Compute final metrics (precision/recall/F1, structural validity).
- Slice results by domain, article length, and relation type.
- Summarize gains vs baseline.
- Document limitations and failure cases.

Acceptance criteria:

- One reproducible evaluation script.
- Final report in `outputs/reports/`.

## Engineering Plan

Core modules:

- `src/data/`: fetching, cleaning, chunking, dataset I/O
- `src/teacher/`: prompting, parsing, normalization
- `src/models/`: target model loading/inference/hooks
- `src/sae/`: SAE model/training/inference
- `src/experiments/`: ablation, steering, patching
- `src/eval/`: metrics, aggregation, reporting

Quality controls:

- Unit tests for parsers and schema validators.
- Integration test for mini end-to-end run (small sample).
- Deterministic seeds and run metadata logging.

## Risks and Mitigations

- Teacher noise: use output validation and confidence filters.
- Relation inconsistency: normalize relation strings or map to ontology.
- Activation storage costs: use compressed formats and sampling strategy.
- Spurious SAE features: require causal confirmation, not correlation only.
- Overfitting to teacher style: include human-checked subset when possible.

## Immediate Next Tasks (This Week)

1. Scaffold folders and script entry points.
2. Implement Wikipedia fetch + chunk pipeline.
3. Define teacher extraction prompt and parser with strict schema checks.
4. Run a small pilot (100-500 chunks) and validate label quality.
5. Select target model and activation hook points.

