#!/usr/bin/env python3
"""Clean and token-chunk Wikipedia JSONL using GPT-5 mini tokenizer."""

from __future__ import annotations

import json
import re
from pathlib import Path
import tiktoken


# ---- Hardcoded config ----
MODEL_NAME = "gpt-5-mini"
CHUNK_SIZE = 4000        # tokens
OVERLAP = 200            # tokens
INPUT_PATH = Path("data/raw/wiki_random.jsonl")
OUTPUT_PATH = Path("data/processed/wiki_chunks.jsonl")
# --------------------------


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_by_tokens(text: str, enc) -> list[str]:
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
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.encoding_for_model(MODEL_NAME)

    total_chunks = 0

    with INPUT_PATH.open("r", encoding="utf-8") as in_f, \
         OUTPUT_PATH.open("w", encoding="utf-8") as out_f:

        for line in in_f:
            if not line.strip():
                continue

            article = json.loads(line)
            article_id = article.get("article_id")
            title = article.get("title")
            text = clean_text(article.get("text", ""))

            chunks = chunk_by_tokens(text, enc)

            for idx, chunk in enumerate(chunks):
                record = {
                    "article_id": article_id,
                    "title": title,
                    "chunk_id": f"{article_id}_{idx}",
                    "chunk_index": idx,
                    "text": chunk,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"Wrote {total_chunks} chunks -> {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())