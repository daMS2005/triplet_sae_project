#!/usr/bin/env python3
"""Compare GPT-teacher triples against Gemma-generated triples."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).lower().strip())


def _load_jsonl(path: Path) -> dict[str, dict]:
    by_chunk: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            chunk_id = rec.get("chunk_id")
            if isinstance(chunk_id, str) and chunk_id:
                by_chunk[chunk_id] = rec
    return by_chunk


def _triples(record: dict) -> list[dict]:
    triples = record.get("triples") or []
    return [triple for triple in triples if isinstance(triple, dict)]


def _spo_set(triples: Iterable[dict]) -> set[tuple[str, str, str]]:
    out = set()
    for triple in triples:
        subject = _norm(triple.get("subject", ""))
        predicate = _norm(triple.get("predicate", ""))
        obj = _norm(triple.get("object", ""))
        if subject and predicate and obj:
            out.add((subject, predicate, obj))
    return out


def _entities(triples: Iterable[dict]) -> set[str]:
    out = set()
    for triple in triples:
        subject = _norm(triple.get("subject", ""))
        obj = _norm(triple.get("object", ""))
        if subject:
            out.add(subject)
        if obj:
            out.add(obj)
    return out


def _predicates(triples: Iterable[dict]) -> set[str]:
    return {_norm(triple.get("predicate", "")) for triple in triples if _norm(triple.get("predicate", ""))}


def _qualifier_key_counts(triples: Iterable[dict]) -> Counter:
    keys: Counter = Counter()
    for triple in triples:
        qualifiers = triple.get("qualifiers") or {}
        if isinstance(qualifiers, dict):
            keys.update(str(k) for k, v in qualifiers.items() if str(k).strip() and v not in ("", None))
    return keys


def _qualifier_stats(triples: list[dict]) -> dict[str, float | int]:
    triples_with_qualifiers = 0
    qualifier_count = 0
    for triple in triples:
        qualifiers = triple.get("qualifiers") or {}
        if not isinstance(qualifiers, dict):
            qualifiers = {}
        nonempty = {k: v for k, v in qualifiers.items() if str(k).strip() and v not in ("", None)}
        if nonempty:
            triples_with_qualifiers += 1
            qualifier_count += len(nonempty)
    return {
        "triples_with_qualifiers": triples_with_qualifiers,
        "qualifier_count": qualifier_count,
        "qualifier_rate": triples_with_qualifiers / len(triples) if triples else 0.0,
        "qualifiers_per_triple": qualifier_count / len(triples) if triples else 0.0,
    }


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def _overlap_metrics(gpt_triples: list[dict], gemma_triples: list[dict]) -> dict[str, float | int]:
    gpt_spo = _spo_set(gpt_triples)
    gemma_spo = _spo_set(gemma_triples)
    tp = len(gpt_spo & gemma_spo)
    fp = len(gemma_spo - gpt_spo)
    fn = len(gpt_spo - gemma_spo)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "exact_spo_tp": tp,
        "exact_spo_fp": fp,
        "exact_spo_fn": fn,
        "exact_spo_precision": precision,
        "exact_spo_recall": recall,
        "exact_spo_f1": f1,
        "entity_jaccard": _jaccard(_entities(gpt_triples), _entities(gemma_triples)),
        "predicate_jaccard": _jaccard(_predicates(gpt_triples), _predicates(gemma_triples)),
    }


def _compare_pair(label: str, gpt_path: Path, gemma_path: Path) -> tuple[list[dict], dict]:
    gpt = _load_jsonl(gpt_path)
    gemma = _load_jsonl(gemma_path)
    shared = sorted(set(gpt) & set(gemma))
    if not shared:
        raise ValueError(f"No shared chunk_ids for {label}: {gpt_path} vs {gemma_path}")

    rows: list[dict] = []
    gpt_qualifier_keys: Counter = Counter()
    gemma_qualifier_keys: Counter = Counter()

    for chunk_id in shared:
        gpt_record = gpt[chunk_id]
        gemma_record = gemma[chunk_id]
        gpt_triples = _triples(gpt_record)
        gemma_triples = _triples(gemma_record)
        gpt_entities = _entities(gpt_triples)
        gemma_entities = _entities(gemma_triples)
        gpt_preds = _predicates(gpt_triples)
        gemma_preds = _predicates(gemma_triples)
        gpt_q = _qualifier_stats(gpt_triples)
        gemma_q = _qualifier_stats(gemma_triples)
        gpt_qualifier_keys.update(_qualifier_key_counts(gpt_triples))
        gemma_qualifier_keys.update(_qualifier_key_counts(gemma_triples))
        overlap = _overlap_metrics(gpt_triples, gemma_triples)
        rows.append(
            {
                "dataset": label,
                "chunk_id": chunk_id,
                "title": gpt_record.get("title") or gemma_record.get("title") or "",
                "gpt_triples": len(gpt_triples),
                "gemma_triples": len(gemma_triples),
                "gpt_entities": len(gpt_entities),
                "gemma_entities": len(gemma_entities),
                "gpt_predicates": len(gpt_preds),
                "gemma_predicates": len(gemma_preds),
                "gpt_triples_with_qualifiers": gpt_q["triples_with_qualifiers"],
                "gemma_triples_with_qualifiers": gemma_q["triples_with_qualifiers"],
                "gpt_qualifier_count": gpt_q["qualifier_count"],
                "gemma_qualifier_count": gemma_q["qualifier_count"],
                "gpt_qualifier_rate": gpt_q["qualifier_rate"],
                "gemma_qualifier_rate": gemma_q["qualifier_rate"],
                "gpt_qualifiers_per_triple": gpt_q["qualifiers_per_triple"],
                "gemma_qualifiers_per_triple": gemma_q["qualifiers_per_triple"],
                **overlap,
            }
        )

    numeric_fields = [
        "gpt_triples",
        "gemma_triples",
        "gpt_entities",
        "gemma_entities",
        "gpt_predicates",
        "gemma_predicates",
        "gpt_triples_with_qualifiers",
        "gemma_triples_with_qualifiers",
        "gpt_qualifier_count",
        "gemma_qualifier_count",
        "gpt_qualifier_rate",
        "gemma_qualifier_rate",
        "gpt_qualifiers_per_triple",
        "gemma_qualifiers_per_triple",
        "exact_spo_precision",
        "exact_spo_recall",
        "exact_spo_f1",
        "entity_jaccard",
        "predicate_jaccard",
    ]
    summary = {
        "dataset": label,
        "gpt_path": str(gpt_path),
        "gemma_path": str(gemma_path),
        "chunk_count": len(rows),
        "means": {field: mean(float(row[field]) for row in rows) for field in numeric_fields},
        "totals": {
            "exact_spo_tp": sum(int(row["exact_spo_tp"]) for row in rows),
            "exact_spo_fp": sum(int(row["exact_spo_fp"]) for row in rows),
            "exact_spo_fn": sum(int(row["exact_spo_fn"]) for row in rows),
        },
        "gpt_top_qualifier_keys": gpt_qualifier_keys.most_common(20),
        "gemma_top_qualifier_keys": gemma_qualifier_keys.most_common(20),
    }
    totals = summary["totals"]
    tp = totals["exact_spo_tp"]
    fp = totals["exact_spo_fp"]
    fn = totals["exact_spo_fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    summary["micro_exact_spo"] = {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0,
    }
    return rows, summary


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Gemma triples to GPT teacher triples.")
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        metavar=("LABEL", "GPT_JSONL", "GEMMA_JSONL"),
        required=True,
        help="Dataset label plus GPT teacher JSONL plus Gemma prediction JSONL.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    args = parser.parse_args()

    all_rows: list[dict] = []
    summaries: list[dict] = []
    for label, gpt_path_raw, gemma_path_raw in args.pair:
        gpt_path = Path(gpt_path_raw)
        gemma_path = Path(gemma_path_raw)
        if not gpt_path.exists():
            raise FileNotFoundError(f"GPT file not found: {gpt_path}")
        if not gemma_path.exists():
            raise FileNotFoundError(f"Gemma file not found: {gemma_path}")
        rows, summary = _compare_pair(label, gpt_path, gemma_path)
        all_rows.extend(rows)
        summaries.append(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "per_chunk_comparison.jsonl", all_rows)
    _write_csv(args.output_dir / "per_chunk_comparison.csv", all_rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"datasets": summaries}, f, ensure_ascii=False, indent=2)

    print(f"Wrote comparison -> {args.output_dir}")
    for summary in summaries:
        means = summary["means"]
        micro = summary["micro_exact_spo"]
        print(
            f"{summary['dataset']}: chunks={summary['chunk_count']} "
            f"GPT triples={means['gpt_triples']:.2f} Gemma triples={means['gemma_triples']:.2f} "
            f"GPT ents={means['gpt_entities']:.2f} Gemma ents={means['gemma_entities']:.2f} "
            f"GPT qual_rate={means['gpt_qualifier_rate']:.2f} Gemma qual_rate={means['gemma_qualifier_rate']:.2f} "
            f"exact_f1={micro['f1']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
