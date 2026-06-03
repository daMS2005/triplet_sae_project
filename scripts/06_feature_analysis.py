#!/usr/bin/env python3
"""Rank SAE features by paired triplet-vs-nontriplet activation differences.

This script answers the question:

    "Which SAE features are more active when the model is doing triplet extraction
    than when it is reading the same text under a non-triplet/control prompt?"

It expects activation `.pt` files produced by `04_collect_activations.py`. Each file
contains one record per text chunk, and each record includes `sae_acts`, a tensor of
shape:

    (number_of_tokens_in_chunk, number_of_sae_features)

The original version compressed each chunk down to one mean activation per SAE feature.
That is useful for broad "mode" features, but it is lossy for sparse syntax features.
This version can rank features with several token-pooling summaries:

    mean            average activation over all tokens
    max             largest token activation
    topk_mean       average of the top-k token activations
    active_fraction fraction of tokens above a threshold
    active_mean     mean activation only over active tokens
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


POOLING_CHOICES = ("mean", "max", "topk_mean", "active_fraction", "active_mean")
DEFAULT_TOP_K_TOKENS = 5
DEFAULT_ACTIVE_THRESHOLD = 1e-6


@dataclass(slots=True)
class PairSpec:
    # One paired comparison, for example:
    #   label        = "wiki"
    #   triplet_path = wiki activations from a triplet-extraction prompt
    #   control_path = wiki activations from a summary/non-triplet prompt
    label: str
    triplet_path: Path
    control_path: Path


@dataclass(slots=True)
class AnalysisConfig:
    # `pooling` controls the ranking objective. The script still writes the other
    # pooling deltas as auxiliary diagnostics, but this one decides row order.
    pooling: str
    top_k_tokens: int
    active_threshold: float


def _load_records(path: Path) -> dict[str, dict]:
    # Activation files are saved as a list of dictionaries. Reindexing them by
    # `chunk_id` lets us line up "same source text, different prompt/task" pairs.
    #
    # This pairing is important: if we compared random triplet chunks to random
    # summary chunks, a feature might look triplet-specific just because one corpus
    # happened to mention different entities, dates, or writing styles.
    records = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(records, list):
        raise TypeError(f"Expected {path} to contain a list of activation records.")

    by_chunk: dict[str, dict] = {}
    duplicate_count = 0
    for record in records:
        chunk_id = record.get("chunk_id")
        if isinstance(chunk_id, str) and chunk_id:
            if chunk_id in by_chunk:
                duplicate_count += 1
            by_chunk[chunk_id] = record
    if duplicate_count:
        print(
            f"warning: {path} contained {duplicate_count} duplicate chunk_id records; "
            "kept the last record for each duplicate.",
            file=sys.stderr,
        )
    return by_chunk


def _get_sae_acts(record: dict) -> torch.Tensor:
    # `sae_acts` is token-level SAE activation data:
    #   rows    = tokens in this prompt/chunk
    #   columns = SAE features
    sae_acts = record["sae_acts"]
    if not isinstance(sae_acts, torch.Tensor):
        raise TypeError("Expected 'sae_acts' to be a torch.Tensor.")
    acts = sae_acts.to(torch.float32)
    if acts.ndim != 2:
        raise ValueError(f"Expected sae_acts to have shape (seq_len, dict_size), got {tuple(acts.shape)}")
    if not torch.isfinite(acts).all():
        chunk_id = record.get("chunk_id", "<unknown>")
        raise ValueError(f"Non-finite sae_acts values found for chunk_id={chunk_id}")
    return acts


def _feature_summaries(record: dict, config: AnalysisConfig) -> dict[str, torch.Tensor]:
    acts = _get_sae_acts(record)
    seq_len = acts.shape[0]
    active_mask = acts > config.active_threshold
    active_counts = active_mask.sum(dim=0).clamp_min(1)
    active_sums = torch.where(active_mask, acts, torch.zeros_like(acts)).sum(dim=0)
    k = max(1, min(config.top_k_tokens, seq_len))

    # These summaries expose different kinds of SAE behavior:
    #   mean: broad mode features
    #   max/topk_mean: sparse spikes on a few important tokens
    #   active_fraction: how often a feature is present at all
    #   active_mean: how strong the feature is when it does fire
    return {
        "mean": acts.mean(dim=0),
        "max": acts.max(dim=0).values,
        "topk_mean": torch.topk(acts, k=k, dim=0).values.mean(dim=0),
        "active_fraction": active_mask.to(torch.float32).mean(dim=0),
        "active_mean": active_sums / active_counts,
    }


def _stack_feature_summaries(
    records: dict[str, dict],
    chunk_ids: list[str],
    config: AnalysisConfig,
) -> tuple[dict[str, torch.Tensor], int]:
    rows_by_pooling: dict[str, list[torch.Tensor]] = {name: [] for name in POOLING_CHOICES}
    expected_width: int | None = None

    for chunk_id in chunk_ids:
        summaries = _feature_summaries(records[chunk_id], config)
        width = int(summaries["mean"].shape[-1])
        if expected_width is None:
            expected_width = width
        elif width != expected_width:
            raise ValueError(
                f"Mixed SAE dictionary widths detected: expected {expected_width}, "
                f"got {width} for chunk_id={chunk_id}."
            )
        for pooling_name, tensor in summaries.items():
            rows_by_pooling[pooling_name].append(tensor)

    if expected_width is None:
        raise ValueError("No activation records were available to stack.")

    return {
        pooling_name: torch.stack(rows, dim=0)
        for pooling_name, rows in rows_by_pooling.items()
    }, expected_width


def _safe_float(value: float) -> float:
    # Do not silently turn NaN/inf into zero. If a statistic becomes non-finite,
    # that usually means the analysis assumptions broke and we should notice.
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"Non-finite statistic encountered: {value}")
    return float(value)


def _paired_t_stat(mean_delta: float, std_delta: float, n: int) -> float:
    # Paired t-statistic:
    #   mean_delta / standard_error_of_delta
    #
    # In plain English: "Is this feature consistently higher in triplet mode across
    # matched chunks, or did one weird chunk dominate the average?"
    #
    # We use it as a ranking signal, not as a formal publishable hypothesis test.
    if n <= 1 or std_delta <= 0.0:
        return 0.0
    return mean_delta / (std_delta / math.sqrt(n))


def _cohens_dz(mean_delta: float, std_delta: float) -> float:
    # Cohen's dz is a paired effect size. Larger positive values mean the triplet-vs-control
    # gap is large relative to how much that gap varies across chunks.
    if std_delta <= 0.0:
        return 0.0
    return mean_delta / std_delta


def _top_examples(
    chunk_ids: list[str],
    deltas: torch.Tensor,
    top_k: int,
    descending: bool,
) -> list[dict[str, float | str]]:
    # For each feature, keep a few concrete chunk IDs where the triplet-control gap
    # was largest. These are breadcrumbs for manual inspection:
    #   positive examples = "this feature really preferred triplet mode here"
    #   negative examples = "this feature was stronger in control mode here"
    ordered = sorted(
        zip(chunk_ids, deltas.tolist()),
        key=lambda item: float(item[1]),
        reverse=descending,
    )
    return [
        {
            "chunk_id": chunk_id,
            "delta_activation": _safe_float(delta),
        }
        for chunk_id, delta in ordered[:top_k]
    ]


def _dataset_summary_rows(chunk_ids: Iterable[str], pair_label: str, triplet_records: dict[str, dict]) -> list[dict]:
    # Lightweight metadata about which chunks participated in each paired comparison.
    # This goes into `analysis_summary.json` so we can audit the analysis later.
    rows = []
    for chunk_id in chunk_ids:
        record = triplet_records[chunk_id]
        rows.append(
            {
                "pair_label": pair_label,
                "chunk_id": chunk_id,
                "doc_id": record.get("doc_id", ""),
                "task_name": record.get("task_name", ""),
                "token_count": int(record.get("token_count", 0)),
            }
        )
    return rows


def analyze_pair(pair: PairSpec, top_k_examples: int, config: AnalysisConfig) -> tuple[list[dict], dict]:
    # Load the two activation files we want to compare:
    #   triplet_records = model was prompted to extract triples
    #   control_records = model saw the same text under a non-triplet task, usually summary
    triplet_records = _load_records(pair.triplet_path)
    control_records = _load_records(pair.control_path)

    # Compare only exact chunk matches so task effects are not confounded with document content.
    shared_chunk_ids = sorted(set(triplet_records) & set(control_records))
    if not shared_chunk_ids:
        raise ValueError(
            f"No shared chunk_ids between {pair.triplet_path} and {pair.control_path}."
        )

    # Build one matrix per condition and per pooling method:
    #   rows    = matched chunks
    #   columns = SAE features
    #
    # Example: triplet_matrices["max"][10, 210] is the maximum token activation of
    # feature 210 on chunk 10 when the model was in triplet-extraction mode.
    triplet_matrices, triplet_width = _stack_feature_summaries(triplet_records, shared_chunk_ids, config)
    control_matrices, control_width = _stack_feature_summaries(control_records, shared_chunk_ids, config)
    if triplet_width != control_width:
        raise ValueError(
            f"Triplet/control SAE widths differ for {pair.label}: {triplet_width} vs {control_width}."
        )

    # Positive deltas mean "more active in triplet mode than in control mode" for that feature.
    delta_matrices = {
        pooling_name: triplet_matrices[pooling_name] - control_matrices[pooling_name]
        for pooling_name in POOLING_CHOICES
    }
    rank_triplet_matrix = triplet_matrices[config.pooling]
    rank_control_matrix = control_matrices[config.pooling]
    rank_delta_matrix = delta_matrices[config.pooling]

    # Shapes:
    #   rank_triplet_matrix: (num_chunks, num_features)
    #   rank_control_matrix: (num_chunks, num_features)
    #   rank_delta_matrix:   (num_chunks, num_features)
    # Each column is one SAE feature tracked across all matched chunks.
    n_chunks, n_features = rank_delta_matrix.shape
    rank_triplet = rank_triplet_matrix.mean(dim=0)
    rank_control = rank_control_matrix.mean(dim=0)
    rank_delta = rank_delta_matrix.mean(dim=0)
    # Standard deviation across chunks tells us whether a feature's triplet preference
    # is consistent, or whether the mean was driven by a small number of extreme chunks.
    rank_std_delta = rank_delta_matrix.std(dim=0, unbiased=True) if n_chunks > 1 else torch.zeros_like(rank_delta)
    # Fraction of chunks where triplet activation > control activation. A feature with
    # high mean_delta but low positive_fraction is probably spiky rather than general.
    rank_positive_fraction = (rank_delta_matrix > 0).to(torch.float32).mean(dim=0)
    mean_triplet = triplet_matrices["mean"].mean(dim=0)
    mean_control = control_matrices["mean"].mean(dim=0)
    mean_delta = delta_matrices["mean"].mean(dim=0)
    auxiliary_delta_means = {
        pooling_name: delta_matrices[pooling_name].mean(dim=0)
        for pooling_name in POOLING_CHOICES
    }

    feature_rows: list[dict] = []
    for feature_id in range(n_features):
        # We summarize each feature with a few easy-to-read statistics:
        # - ranking-pooling activation in triplet/control mode
        # - ranking-pooling difference
        # - how often the ranking-pooling difference is positive
        # - sparse auxiliary deltas from other token pooling methods
        deltas = rank_delta_matrix[:, feature_id]
        rank_delta_value = _safe_float(rank_delta[feature_id].item())
        rank_std_delta_value = _safe_float(rank_std_delta[feature_id].item())
        rank_positive_fraction_value = _safe_float(rank_positive_fraction[feature_id].item())
        row = {
            "pair_label": pair.label,
            "feature_id": feature_id,
            "n_chunks": n_chunks,
            "ranking_pooling": config.pooling,
            "ranking_triplet_activation": _safe_float(rank_triplet[feature_id].item()),
            "ranking_control_activation": _safe_float(rank_control[feature_id].item()),
            "ranking_delta_activation": rank_delta_value,
            "ranking_std_delta_activation": rank_std_delta_value,
            "ranking_positive_delta_fraction": rank_positive_fraction_value,
            "ranking_paired_t_stat": _safe_float(_paired_t_stat(rank_delta_value, rank_std_delta_value, n_chunks)),
            "ranking_cohens_dz": _safe_float(_cohens_dz(rank_delta_value, rank_std_delta_value)),
            "mean_triplet_activation": _safe_float(mean_triplet[feature_id].item()),
            "mean_control_activation": _safe_float(mean_control[feature_id].item()),
            "mean_delta_activation": _safe_float(mean_delta[feature_id].item()),
            "max_delta_activation": _safe_float(auxiliary_delta_means["max"][feature_id].item()),
            "topk_mean_delta_activation": _safe_float(auxiliary_delta_means["topk_mean"][feature_id].item()),
            "active_fraction_delta": _safe_float(auxiliary_delta_means["active_fraction"][feature_id].item()),
            "active_mean_delta_activation": _safe_float(auxiliary_delta_means["active_mean"][feature_id].item()),
            "top_positive_examples": _top_examples(shared_chunk_ids, deltas, top_k_examples, descending=True),
            "top_negative_examples": _top_examples(shared_chunk_ids, deltas, top_k_examples, descending=False),
        }
        feature_rows.append(row)

    # Rank features so the most "triplet-enriched" ones come first under the selected
    # pooling method. Primary key: bigger average triplet-control gap.
    # Tie-breakers: more consistently positive across chunks, then larger t-stat.
    feature_rows.sort(
        key=lambda row: (
            float(row["ranking_delta_activation"]),
            float(row["ranking_positive_delta_fraction"]),
            float(row["ranking_paired_t_stat"]),
        ),
        reverse=True,
    )
    # The summary keeps the top slice lightweight for quick inspection and downstream
    # steering/ablation choices. The full JSONL/CSV still contains every feature.

    summary = {
        "pair_label": pair.label,
        "triplet_path": str(pair.triplet_path),
        "control_path": str(pair.control_path),
        "ranking_pooling": config.pooling,
        "top_k_tokens": config.top_k_tokens,
        "active_threshold": config.active_threshold,
        "n_chunks": n_chunks,
        "n_features": n_features,
        "shared_chunk_ids": shared_chunk_ids,
        "chunk_rows": _dataset_summary_rows(shared_chunk_ids, pair.label, triplet_records),
        "top_features": [
            {
                "feature_id": row["feature_id"],
                "ranking_delta_activation": row["ranking_delta_activation"],
                "ranking_positive_delta_fraction": row["ranking_positive_delta_fraction"],
                "ranking_paired_t_stat": row["ranking_paired_t_stat"],
                "ranking_cohens_dz": row["ranking_cohens_dz"],
                "mean_delta_activation": row["mean_delta_activation"],
                "max_delta_activation": row["max_delta_activation"],
                "topk_mean_delta_activation": row["topk_mean_delta_activation"],
                "active_fraction_delta": row["active_fraction_delta"],
            }
            for row in feature_rows[:20]
        ],
    }
    return feature_rows, summary


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    # JSONL stores one full feature row per line. This preserves nested fields like
    # top_positive_examples, which are awkward in CSV.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    # CSV is intentionally flatter and spreadsheet-friendly. The example lists are
    # JSON-encoded strings inside CSV cells.
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair_label",
        "feature_id",
        "n_chunks",
        "ranking_pooling",
        "ranking_triplet_activation",
        "ranking_control_activation",
        "ranking_delta_activation",
        "ranking_std_delta_activation",
        "ranking_positive_delta_fraction",
        "ranking_paired_t_stat",
        "ranking_cohens_dz",
        "mean_triplet_activation",
        "mean_control_activation",
        "mean_delta_activation",
        "max_delta_activation",
        "topk_mean_delta_activation",
        "active_fraction_delta",
        "active_mean_delta_activation",
        "top_positive_examples",
        "top_negative_examples",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = row.copy()
            csv_row["top_positive_examples"] = json.dumps(row["top_positive_examples"], ensure_ascii=False)
            csv_row["top_negative_examples"] = json.dumps(row["top_negative_examples"], ensure_ascii=False)
            writer.writerow(csv_row)


def _combine_global_ranking(rows: list[dict]) -> list[dict]:
    # Each pair/corpus produces its own ranking. This function groups rows by feature_id
    # so we can ask, "Which features are enriched across multiple datasets?"
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["feature_id"]), []).append(row)

    combined: list[dict] = []
    for feature_id, group in grouped.items():
        # Aggregate across corpora to surface features that generalize beyond one dataset.
        # A feature that ranks well in both wiki and dolma is more interesting than a
        # feature that only fires for one corpus's quirks.
        rank_delta_values = [float(row["ranking_delta_activation"]) for row in group]
        positive_fractions = [float(row["ranking_positive_delta_fraction"]) for row in group]
        t_values = [float(row["ranking_paired_t_stat"]) for row in group]
        mean_delta_values = [float(row["mean_delta_activation"]) for row in group]
        max_delta_values = [float(row["max_delta_activation"]) for row in group]
        topk_delta_values = [float(row["topk_mean_delta_activation"]) for row in group]
        active_fraction_values = [float(row["active_fraction_delta"]) for row in group]
        combined.append(
            {
                "feature_id": feature_id,
                "pair_count": len(group),
                "ranking_pooling": group[0]["ranking_pooling"],
                "mean_ranking_delta_activation": sum(rank_delta_values) / len(rank_delta_values),
                "mean_ranking_positive_delta_fraction": sum(positive_fractions) / len(positive_fractions),
                "mean_ranking_paired_t_stat": sum(t_values) / len(t_values),
                "mean_of_mean_delta_activation": sum(mean_delta_values) / len(mean_delta_values),
                "mean_max_delta_activation": sum(max_delta_values) / len(max_delta_values),
                "mean_topk_mean_delta_activation": sum(topk_delta_values) / len(topk_delta_values),
                "mean_active_fraction_delta": sum(active_fraction_values) / len(active_fraction_values),
                "pair_labels": [row["pair_label"] for row in group],
            }
        )

    combined.sort(
        key=lambda row: (
            float(row["mean_ranking_delta_activation"]),
            float(row["mean_ranking_positive_delta_fraction"]),
            float(row["mean_ranking_paired_t_stat"]),
        ),
        reverse=True,
    )
    return combined


def parse_args() -> argparse.Namespace:
    # The script can compare multiple pairs in one run. For example:
    #
    #   --pair wiki  wiki_triplet.pt  wiki_summary.pt
    #   --pair dolma dolma_triplet.pt dolma_summary.pt
    #
    # Then it writes both pair-specific rows and a combined global ranking.
    parser = argparse.ArgumentParser(
        description="Rank SAE features by paired triplet-vs-control activation differences."
    )
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        metavar=("LABEL", "TRIPLET_PT", "CONTROL_PT"),
        required=True,
        help="A paired comparison: label triplet_activations.pt control_activations.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directory to write ranking outputs.",
    )
    parser.add_argument(
        "--top-k-examples",
        type=int,
        default=5,
        metavar="N",
        help="How many top positive/negative chunk examples to store per feature (default: 5).",
    )
    parser.add_argument(
        "--pooling",
        choices=POOLING_CHOICES,
        default="mean",
        help=(
            "Token pooling method used for ranking. Use max/topk_mean/active_fraction/"
            "active_mean to find sparse syntax-like features (default: mean)."
        ),
    )
    parser.add_argument(
        "--top-k-tokens",
        type=int,
        default=DEFAULT_TOP_K_TOKENS,
        metavar="N",
        help=f"Number of token activations to average for --pooling topk_mean (default: {DEFAULT_TOP_K_TOKENS}).",
    )
    parser.add_argument(
        "--active-threshold",
        type=float,
        default=DEFAULT_ACTIVE_THRESHOLD,
        metavar="X",
        help=(
            "Activation threshold for active_fraction and active_mean pooling "
            f"(default: {DEFAULT_ACTIVE_THRESHOLD})."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AnalysisConfig(
        pooling=args.pooling,
        top_k_tokens=max(1, args.top_k_tokens),
        active_threshold=float(args.active_threshold),
    )
    # Convert raw CLI strings into typed PairSpec objects so the rest of the code can
    # talk in terms of pair.label / pair.triplet_path / pair.control_path.
    pairs = [PairSpec(label=label, triplet_path=Path(triplet), control_path=Path(control)) for label, triplet, control in args.pair]

    # Fail early if a path is wrong. This avoids running half an analysis and then
    # discovering later that one activation file was missing.
    for pair in pairs:
        if not pair.triplet_path.exists():
            raise FileNotFoundError(f"Triplet activation file not found: {pair.triplet_path}")
        if not pair.control_path.exists():
            raise FileNotFoundError(f"Control activation file not found: {pair.control_path}")

    all_rows: list[dict] = []
    summaries: list[dict] = []

    for pair in pairs:
        # Run the same analysis separately for each corpus pair, then combine them below.
        pair_rows, summary = analyze_pair(pair, top_k_examples=args.top_k_examples, config=config)
        all_rows.extend(pair_rows)
        summaries.append(summary)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ranking_jsonl = output_dir / "feature_ranking.jsonl"
    ranking_csv = output_dir / "feature_ranking.csv"
    global_json = output_dir / "global_feature_ranking.json"
    summary_json = output_dir / "analysis_summary.json"

    # Main outputs:
    #   feature_ranking.jsonl       = every feature for every pair, richest format
    #   feature_ranking.csv         = same ranking in spreadsheet-friendly form
    #   global_feature_ranking.json = one combined score per feature across pairs
    #   analysis_summary.json       = metadata/top-20 features per pair
    _write_jsonl(ranking_jsonl, all_rows)
    _write_csv(ranking_csv, all_rows)

    with global_json.open("w", encoding="utf-8") as f:
        json.dump(_combine_global_ranking(all_rows), f, ensure_ascii=False, indent=2)

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print(f"Wrote pairwise ranking -> {ranking_jsonl}")
    print(f"Wrote CSV summary    -> {ranking_csv}")
    print(f"Wrote global summary -> {global_json}")
    print(f"Wrote analysis meta  -> {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
