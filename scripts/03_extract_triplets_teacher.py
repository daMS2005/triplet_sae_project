#!/usr/bin/env python3
"""Extract (subject, predicate, object) triples from text chunks using a teacher LLM.

Uses async OpenAI calls with a concurrency semaphore for throughput.

Resume behavior:
- Reads already-processed chunk_ids from the output file and skips them.

Durability:
- Writes are serialized through a lock; fsync every SAVE_EVERY_N records.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv()
from openai import AsyncOpenAI


MODEL_NAME = "gpt-5-mini"
CONCURRENCY = 20    # simultaneous in-flight API calls
SAVE_EVERY_N = 5
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1.0


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


async def _call_with_retries(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    *,
    title: str,
    chunk_text: str,
) -> dict[str, Any]:
    user_content = (
        f"Title: {title}\n\n"
        "Extract factual triples from the following text:\n"
        "-----\n"
        f"{chunk_text}\n"
        "-----"
    )
    backoff = INITIAL_BACKOFF_SECONDS
    last_err: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with sem:
                resp = await client.responses.create(
                    model=MODEL_NAME,
                    input=[
                        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                        {"role": "user", "content": user_content},
                    ],
                    text={"format": TRIPLES_SCHEMA},
                )
            return json.loads(resp.output_text)
        except Exception as e:
            last_err = e
            print(f"  [attempt {attempt}/{MAX_RETRIES}] error: {repr(e)}")
            if attempt < MAX_RETRIES:
                await asyncio.sleep(backoff)
                backoff *= 2

    raise RuntimeError(f"OpenAI call failed after {MAX_RETRIES} attempts: {repr(last_err)}")


async def _run(input_path: Path, output_path: Path, concurrency: int = CONCURRENCY) -> None:
    done_chunk_ids = _load_done_chunk_ids(output_path)
    pending = _load_pending_chunks(input_path, done_chunk_ids)
    print(f"Already done: {len(done_chunk_ids)}  |  Pending: {len(pending)}")

    if not pending:
        return

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(concurrency)

    counters = {"written": 0, "failed": 0}
    write_lock = asyncio.Lock()
    out_f = output_path.open("a", encoding="utf-8")

    async def process(rec: dict) -> None:
        chunk_id = rec["chunk_id"]
        title = str(rec.get("title") or "")
        text = str(rec.get("text") or "")

        try:
            extracted = await _call_with_retries(client, sem, title=title, chunk_text=text)
            out_record = {
                "doc_id": rec.get("doc_id") or rec.get("article_id"),
                "title": title,
                "chunk_id": chunk_id,
                "chunk_index": rec.get("chunk_index"),
                "triples": extracted["triples"],
                "model": MODEL_NAME,
            }
        except Exception as e:
            out_record = {
                "doc_id": rec.get("doc_id") or rec.get("article_id"),
                "title": title,
                "chunk_id": chunk_id,
                "chunk_index": rec.get("chunk_index"),
                "triples": [],
                "error": str(e),
                "model": MODEL_NAME,
            }
            async with write_lock:
                counters["failed"] += 1

        async with write_lock:
            out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            counters["written"] += 1
            if counters["written"] % SAVE_EVERY_N == 0:
                out_f.flush()
                os.fsync(out_f.fileno())
                print(f"  written={counters['written']} failed={counters['failed']}")

    try:
        await asyncio.gather(*[process(rec) for rec in pending])
    finally:
        out_f.flush()
        os.fsync(out_f.fileno())
        out_f.close()

    print(
        f"Done. written={counters['written']} failed={counters['failed']} -> {output_path}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract triplets from text chunks using an async teacher LLM."
    )
    parser.add_argument(
        "--input", type=Path, required=True, metavar="FILE",
        help="Input JSONL file of text chunks.",
    )
    parser.add_argument(
        "--output", type=Path, required=True, metavar="FILE",
        help="Output JSONL file for extracted triples (append-safe, resumable).",
    )
    parser.add_argument(
        "--concurrency", type=int, default=CONCURRENCY, metavar="N",
        help=f"Max simultaneous API calls (default: {CONCURRENCY}).",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    asyncio.run(_run(args.input, args.output, args.concurrency))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
