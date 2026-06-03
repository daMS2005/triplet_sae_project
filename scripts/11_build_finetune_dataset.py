#!/usr/bin/env python3
"""Build supervised fine-tuning JSONL files from matched triplet data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


SYSTEM_PROMPT = (
    "You extract factual knowledge triples from source text.\n"
    "Return exactly one JSON object with key 'triples'.\n"
    "Each triple must have string fields 'subject', 'predicate', and 'object'.\n"
    "You may include a 'qualifiers' object for useful context like dates, locations, "
    "scores, roles, conditions, or event names.\n"
    "Only include facts explicitly grounded in the text.\n"
    "Do not include commentary before or after the JSON."
)

USER_TEMPLATES = [
    (
        "Generate factual knowledge triples for this paragraph.\n"
        "Use strict JSON in this shape:\n"
        "{{\"triples\":[{{\"subject\":\"...\",\"predicate\":\"...\",\"object\":\"...\",\"qualifiers\":{{}}}}]}}\n\n"
        "Paragraph:\n{text}"
    ),
    (
        "Extract subject-predicate-object triples from the paragraph below.\n"
        "Return strict JSON only in this shape:\n"
        "{{\"triples\":[{{\"subject\":\"...\",\"predicate\":\"...\",\"object\":\"...\",\"qualifiers\":{{}}}}]}}\n\n"
        "Paragraph:\n{text}"
    ),
    (
        "Read this paragraph and generate factual triples from it.\n"
        "Return a JSON object with key 'triples' and items shaped like "
        "{{\"subject\":\"...\",\"predicate\":\"...\",\"object\":\"...\",\"qualifiers\":{{}}}}.\n\n"
        "Paragraph:\n{text}"
    ),
    (
        "Produce knowledge triples for this paragraph.\n"
        "Return strict JSON only with a top-level 'triples' list.\n\n"
        "Paragraph:\n{text}"
    ),
]


def _pick_user_prompt(*, dataset: str, chunk_id: str, text: str, seed: int) -> str:
    # Pick a stable prompt variant per example so reruns are reproducible while the
    # training set still contains several prompt phrasings.
    key = f"{seed}:{dataset}:{chunk_id}".encode("utf-8")
    idx = int(hashlib.sha256(key).hexdigest(), 16) % len(USER_TEMPLATES)
    return USER_TEMPLATES[idx].format(text=text)


def _clean_triple(triple: dict) -> dict | None:
    subject = str(triple.get("subject") or "").strip()
    predicate = str(triple.get("predicate") or "").strip()
    obj = str(triple.get("object") or "").strip()
    if not subject or not predicate or not obj:
        return None

    qualifiers = triple.get("qualifiers") or {}
    if not isinstance(qualifiers, dict):
        qualifiers = {}
    qualifiers = {
        str(key): value
        for key, value in qualifiers.items()
        if str(key).strip() and value not in ("", None, [], {})
    }
    return {
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "qualifiers": qualifiers,
    }


def _load_examples(paths: list[str], max_triples: int | None, seed: int) -> list[dict]:
    examples: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for entry in paths:
        if "=" not in entry:
            raise ValueError(f"Expected DATASET=PATH, got: {entry}")
        dataset, path_text = entry.split("=", 1)
        path = Path(path_text)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                chunk_id = str(record.get("chunk_id") or "")
                text = str(record.get("text") or "").strip()
                raw_triples = record.get("triples") or []
                if not chunk_id or not text or not isinstance(raw_triples, list):
                    continue

                key = (dataset, chunk_id)
                if key in seen:
                    continue
                seen.add(key)

                triples = []
                for raw_triple in raw_triples:
                    if isinstance(raw_triple, dict):
                        cleaned = _clean_triple(raw_triple)
                        if cleaned is not None:
                            triples.append(cleaned)
                    if max_triples is not None and len(triples) >= max_triples:
                        break
                if not triples:
                    continue

                title = str(record.get("title") or "")
                assistant_json = json.dumps({"triples": triples}, ensure_ascii=False, separators=(",", ":"))
                user_prompt = _pick_user_prompt(
                    dataset=dataset,
                    chunk_id=chunk_id,
                    text=text,
                    seed=seed,
                )
                examples.append(
                    {
                        "dataset": dataset,
                        "doc_id": str(record.get("doc_id") or record.get("article_id") or ""),
                        "chunk_id": chunk_id,
                        "chunk_index": record.get("chunk_index"),
                        "title": title,
                        "text": text,
                        "triples": triples,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": assistant_json},
                        ],
                    }
                )
    return examples


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Gemma chat SFT data for triplet extraction.")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="DATASET=FILE",
        help="Matched input/output JSONL files.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, metavar="DIR")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.0)
    parser.add_argument(
        "--max-triples",
        type=int,
        default=32,
        help="Maximum teacher triples kept per example. Use 0 for no cap.",
    )
    args = parser.parse_args()

    if args.val_fraction < 0 or args.test_fraction < 0 or args.val_fraction + args.test_fraction >= 1:
        raise ValueError("Split fractions must be non-negative and sum to less than 1.")

    max_triples = None if args.max_triples <= 0 else args.max_triples
    examples = _load_examples(args.input, max_triples=max_triples, seed=args.seed)
    if not examples:
        raise ValueError("No SFT examples were built.")

    rng = random.Random(args.seed)
    rng.shuffle(examples)

    n_total = len(examples)
    n_test = int(round(n_total * args.test_fraction))
    n_val = int(round(n_total * args.val_fraction))
    test_rows = examples[:n_test]
    val_rows = examples[n_test : n_test + n_val]
    train_rows = examples[n_test + n_val :]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "train.jsonl", train_rows)
    _write_jsonl(args.output_dir / "validation.jsonl", val_rows)
    if test_rows:
        _write_jsonl(args.output_dir / "test.jsonl", test_rows)

    manifest = {
        "seed": args.seed,
        "inputs": args.input,
        "max_triples": max_triples,
        "total_examples": n_total,
        "train_examples": len(train_rows),
        "validation_examples": len(val_rows),
        "test_examples": len(test_rows),
        "system_prompt": SYSTEM_PROMPT,
        "user_templates": USER_TEMPLATES,
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
