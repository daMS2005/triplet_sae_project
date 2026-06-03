#!/usr/bin/env python3
"""Clean and token-chunk a raw JSONL file (Wikipedia or Dolma) into fixed-size windows."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import tiktoken


CHUNK_SIZE = 4000   # tokens
OVERLAP = 200       # tokens
ENCODING = "o200k_base"  # tiktoken encoding; gpt-5-mini uses o200k_base


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_by_tokens(text: str, enc: tiktoken.Encoding) -> list[str]:
    tokens = enc.encode(text)
    step = CHUNK_SIZE - OVERLAP
    chunks = []
    for i in range(0, len(tokens), step):
        piece = tokens[i : i + CHUNK_SIZE]
        if not piece:
            break
        chunks.append(enc.decode(piece))
        if i + CHUNK_SIZE >= len(tokens):
            break
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chunk and clean a raw JSONL file into fixed-size token windows."
    )
    parser.add_argument(
        "--input", type=Path, required=True, metavar="FILE",
        help="Input JSONL file (Wikipedia or Dolma raw).",
    )
    parser.add_argument(
        "--output", type=Path, required=True, metavar="FILE",
        help="Output JSONL file for chunks.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    enc = tiktoken.get_encoding(ENCODING)
    total_chunks = 0

    with args.input.open("r", encoding="utf-8") as in_f, \
         args.output.open("w", encoding="utf-8") as out_f:

        for line in in_f:
            if not line.strip():
                continue
            doc = json.loads(line)

            # Normalize id: Wikipedia uses article_id, Dolma uses doc_id
            doc_id = doc.get("article_id") or doc.get("doc_id") or ""
            title = doc.get("title") or ""
            text = clean_text(doc.get("text") or "")
            if not text:
                continue

            for idx, chunk in enumerate(chunk_by_tokens(text, enc)):
                record = {
                    "doc_id": doc_id,
                    "title": title,
                    "chunk_id": f"{doc_id}_{idx}",
                    "chunk_index": idx,
                    "text": chunk,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"Wrote {total_chunks} chunks -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
