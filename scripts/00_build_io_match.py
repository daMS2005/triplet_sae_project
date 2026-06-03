#!/usr/bin/env python3
"""Build an input/output matched JSONL by joining chunks with triplets on chunk_id."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_triples(path: Path) -> dict[str, dict]:
    triples_by_chunk: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            chunk_id = rec.get("chunk_id")
            if isinstance(chunk_id, str) and chunk_id:
                triples_by_chunk[chunk_id] = rec
    return triples_by_chunk


def build_match(chunks_path: Path, triples_path: Path, output_path: Path) -> None:
    triples_by_chunk = _load_triples(triples_path)
    matched = 0
    missing = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with chunks_path.open("r", encoding="utf-8") as chunks_f, output_path.open(
        "w", encoding="utf-8"
    ) as out_f:
        for line in chunks_f:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            chunk_id = chunk.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                continue

            triplet_rec = triples_by_chunk.get(chunk_id)
            if triplet_rec is None:
                missing += 1
                continue

            out_record = {
                "doc_id": chunk.get("doc_id") or chunk.get("article_id") or "",
                "title": chunk.get("title") or triplet_rec.get("title") or "",
                "chunk_id": chunk_id,
                "chunk_index": chunk.get("chunk_index"),
                "text": chunk.get("text") or "",
                "triples": triplet_rec.get("triples", []),
                "teacher_model": triplet_rec.get("model"),
                "teacher_error": triplet_rec.get("error"),
            }
            out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            matched += 1

    print(f"Matched {matched} chunks -> {output_path}")
    print(f"Skipped {missing} chunks with no triplet record")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Join chunk inputs with triplet outputs on chunk_id."
    )
    parser.add_argument(
        "--chunks", type=Path, required=True, metavar="FILE", help="Chunk JSONL input."
    )
    parser.add_argument(
        "--triples", type=Path, required=True, metavar="FILE", help="Triples JSONL input."
    )
    parser.add_argument(
        "--output", type=Path, required=True, metavar="FILE", help="Matched JSONL output."
    )
    args = parser.parse_args()

    if not args.chunks.exists():
        raise FileNotFoundError(f"Chunks file not found: {args.chunks}")
    if not args.triples.exists():
        raise FileNotFoundError(f"Triples file not found: {args.triples}")

    build_match(args.chunks, args.triples, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
