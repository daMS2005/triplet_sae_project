#!/usr/bin/env python3
"""
Teacher-model triplet extraction step (GPT-5-mini).

- Input:  data/processed/wiki_chunks.jsonl  (one record per chunk)
- Output: data/processed/wiki_triples.jsonl (one record per chunk, append-only)

Resume behavior:
- Loads chunk_id from OUTPUT_PATH and skips already-seen chunk_ids.
Durability:
- Flush + fsync every 5 output records.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI


# ---- Hardcoded config ----
MODEL_NAME = "gpt-5-mini"
INPUT_PATH = Path("data/processed/wiki_chunks.jsonl")
OUTPUT_PATH = Path("data/processed/wiki_triples.jsonl")



SAVE_EVERY_N = 5  # flush + fsync every N written records

MAX_RETRIES = 1
INITIAL_BACKOFF_SECONDS = 1.0
# --------------------------


TRIPLES_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "name": "triples_response",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "triples": {
                "type": "array",
                "maxItems": 40,
                "items": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string"},
                        "predicate": {"type": "string"},
                        "object": {"type": "string"},
                        "qualifiers": {
                            "type": "object",
                            "additionalProperties": {
                                "type": ["string", "number", "boolean"]
                            },
                        },
                    },
                    "required": ["subject", "predicate", "object"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["triples"],
        "additionalProperties": False,
    },
}

SYSTEM_INSTRUCTIONS = (
    "You extract factual (subject, predicate, object) triples from text.\n"
    "Rules:\n"
    "- Only output facts explicitly supported by the input.\n"
    "- Prefer canonical entity names as they appear in the text.\n"
    "- Predicates should be short relation phrases (e.g., 'is a', 'was born in', 'located in').\n"
    "- If the text is ambiguous or not factual, omit the triple.\n"
    "- You may add qualifiers to the triple to provide more context.\n"
    "- Follow the JSON schema strictly."
)


def _load_done_chunk_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = rec.get("chunk_id")
            if isinstance(cid, str) and cid:
                done.add(cid)
    return done


def _load_pending_chunks(input_path: Path, done_chunk_ids: set[str]) -> list[dict]:
    pending: list[dict] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec.get("chunk_id")
            if not isinstance(cid, str) or not cid:
                continue
            if cid in done_chunk_ids:
                continue
            text = rec.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            pending.append(rec)
    return pending


def _call_with_retries(client: OpenAI, *, title: str, chunk_text: str) -> dict[str, Any]:
    backoff = INITIAL_BACKOFF_SECONDS
    last_err: Exception | None = None

    user_content = (
        f"Title: {title}\n\n"
        "Extract factual triples from the following text:\n"
        "-----\n"
        f"{chunk_text}\n"
        "-----"
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.responses.create(
                model=MODEL_NAME,
                input=[
                    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": user_content},
                ],
                text={"format": TRIPLES_SCHEMA},
    
            )

            data = resp.output_text
            print(resp)
            print(resp.output)
            print(resp.output_text)
            print(resp.status)
            print(resp.incomplete_details)
            return json.loads(data)

        except Exception as e:
            last_err = e
            print(f"[Attempt {attempt}] Error: {repr(e)}")
            if attempt < MAX_RETRIES:
                time.sleep(backoff)
                backoff *= 2

    # Preserve original error message
    raise RuntimeError(f"OpenAI call failed: {repr(last_err)}")


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    done_chunk_ids = _load_done_chunk_ids(OUTPUT_PATH)
    pending = _load_pending_chunks(INPUT_PATH, done_chunk_ids)

    print(f"Already done chunks: {len(done_chunk_ids)}")
    print(f"Pending chunks this run: {len(pending)}")

    client = OpenAI()

    processed = 0
    written = 0
    failed = 0

    # Append-only output, resume-safe via done_chunk_ids
    with OUTPUT_PATH.open("a", encoding="utf-8") as out_f:
        for rec in pending:
            chunk_id = rec["chunk_id"]
            title = str(rec.get("title") or "")
            text = str(rec.get("text") or "")

            processed += 1
            try:
                extracted = _call_with_retries(client, title=title, chunk_text=text)
                print(extracted)
                out_record = {
                    "article_id": rec.get("article_id"),
                    "title": rec.get("title"),
                    "chunk_id": chunk_id,
                    "chunk_index": rec.get("chunk_index"),
                    "triples": extracted["triples"],
                    "model": MODEL_NAME,
                }
            except Exception as e:
                failed += 1
                out_record = {
                    "article_id": rec.get("article_id"),
                    "title": rec.get("title"),
                    "chunk_id": chunk_id,
                    "chunk_index": rec.get("chunk_index"),
                    "error": str(e),
                    "model": MODEL_NAME,
                }

            out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            written += 1
            done_chunk_ids.add(chunk_id)

            # Durability: every N records, flush + fsync
            if written % SAVE_EVERY_N == 0:
                out_f.flush()
                os.fsync(out_f.fileno())

            if processed % 10 == 0:
                print(f"processed={processed} written={written} failed={failed}")

        # final flush
        out_f.flush()
        os.fsync(out_f.fileno())

    print(f"Done. processed={processed} written={written} failed={failed} -> {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())