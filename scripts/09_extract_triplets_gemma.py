#!/usr/bin/env python3
"""Extract triples from matched chunks with Gemma instruction-tuned model.

This is the local-model counterpart to `03_extract_triplets_teacher.py`.
The GPT teacher triples already live in the matched input files, so this script only
adds Gemma predictions for the same chunks. A second script can then compare the two.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "google/gemma-3-4b-it"
DEFAULT_SEED = 7

SYSTEM_PROMPT = (
    "You extract factual knowledge triples from source text.\n"
    "Return exactly one JSON object with key 'triples'.\n"
    "Each triple must have string fields 'subject', 'predicate', and 'object'.\n"
    "You may include a 'qualifiers' object for useful context like dates, locations, "
    "scores, roles, conditions, or event names.\n"
    "Only include facts explicitly grounded in the text.\n"
    "Do not include commentary before or after the JSON."
)

USER_TEMPLATE = (
    "Extract up to {max_triples} factual (subject, predicate, object) triples from the text below.\n"
    "Use strict JSON in this shape:\n"
    "{{\"triples\":[{{\"subject\":\"...\",\"predicate\":\"...\",\"object\":\"...\",\"qualifiers\":{{}}}}]}}\n\n"
    "Title: {title}\n\n"
    "Text:\n{text}"
)


def _load_done_chunk_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk_id = rec.get("chunk_id")
            if isinstance(chunk_id, str) and chunk_id:
                done.add(chunk_id)
    return done


def _load_records(path: Path, *, sample: int | None, seed: int, done_chunk_ids: set[str]) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            chunk_id = rec.get("chunk_id")
            text = rec.get("text")
            if not isinstance(chunk_id, str) or not chunk_id:
                continue
            if chunk_id in done_chunk_ids:
                continue
            if not isinstance(text, str) or not text.strip():
                continue
            rows.append(rec)

    if sample is not None and sample > 0 and len(rows) > sample:
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = rows[:sample]
    return rows


def _extract_json_payload(text: str) -> dict | list | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()

    candidates: list[str] = []
    obj_start = stripped.find("{")
    obj_end = stripped.rfind("}")
    if obj_start != -1 and obj_end != -1 and obj_end >= obj_start:
        candidates.append(stripped[obj_start : obj_end + 1])

    list_start = stripped.find("[")
    list_end = stripped.rfind("]")
    if list_start != -1 and list_end != -1 and list_end >= list_start:
        candidates.append(stripped[list_start : list_end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _parse_truncated_triples(text: str) -> list[dict]:
    pattern = re.compile(
        r'"subject"\s*:\s*"(?P<subject>(?:\\.|[^"\\])*)"\s*,\s*'
        r'"predicate"\s*:\s*"(?P<predicate>(?:\\.|[^"\\])*)"\s*,\s*'
        r'"object"\s*:\s*"(?P<object>(?:\\.|[^"\\])*)"',
        re.DOTALL,
    )
    triples: list[dict] = []
    for match in pattern.finditer(text):
        subject = bytes(match.group("subject"), "utf-8").decode("unicode_escape").strip()
        predicate = bytes(match.group("predicate"), "utf-8").decode("unicode_escape").strip()
        obj = bytes(match.group("object"), "utf-8").decode("unicode_escape").strip()
        if subject and predicate and obj:
            triples.append({"subject": subject, "predicate": predicate, "object": obj, "qualifiers": {}})
    return triples


def _clean_triples(payload: dict | list | None, raw_text: str) -> tuple[list[dict], bool]:
    if payload is None:
        fallback = _parse_truncated_triples(raw_text)
        return fallback, bool(fallback)

    triples = payload.get("triples") if isinstance(payload, dict) else payload
    if not isinstance(triples, list):
        fallback = _parse_truncated_triples(raw_text)
        return fallback, bool(fallback)

    cleaned: list[dict] = []
    for item in triples:
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject") or "").strip()
        predicate = str(item.get("predicate") or "").strip()
        obj = str(item.get("object") or "").strip()
        qualifiers = item.get("qualifiers") or {}
        if not isinstance(qualifiers, dict):
            qualifiers = {}
        qualifiers = {str(k): v for k, v in qualifiers.items() if str(k).strip() and v not in ("", None)}
        if subject and predicate and obj:
            cleaned.append(
                {
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "qualifiers": qualifiers,
                }
            )
    return cleaned, bool(cleaned)


def _render_prompt(tokenizer, record: dict, max_triples: int) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                max_triples=max_triples,
                title=str(record.get("title") or ""),
                text=str(record.get("text") or ""),
            ),
        },
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _generate_one(
    *,
    model,
    tokenizer,
    record: dict,
    max_input_tokens: int,
    max_new_tokens: int,
    max_triples: int,
) -> dict:
    prompt = _render_prompt(tokenizer, record, max_triples)
    tokenized = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    input_ids = tokenized["input_ids"].to(model.device)
    attention_mask = tokenized["attention_mask"].to(model.device)

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated[0, input_ids.shape[1] :]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    triples, parse_success = _clean_triples(_extract_json_payload(generated_text), generated_text)
    return {
        "doc_id": record.get("doc_id") or record.get("article_id") or "",
        "title": record.get("title") or "",
        "chunk_id": record.get("chunk_id") or "",
        "chunk_index": record.get("chunk_index"),
        "triples": triples,
        "model": MODEL_NAME,
        "parse_success": parse_success,
        "generated_text": generated_text,
    }


def _write_records(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract triples from chunks with Gemma 12B IT.")
    parser.add_argument("--input", type=Path, required=True, help="Matched JSONL input.")
    parser.add_argument("--output", type=Path, required=True, help="Gemma prediction JSONL output.")
    parser.add_argument("--model", default=MODEL_NAME, help=f"Model name (default: {MODEL_NAME}).")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Optional PEFT/LoRA adapter directory to load on top of --model.",
    )
    parser.add_argument("--sample", type=int, default=None, help="Optional number of chunks to sample.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Sampling seed (default: {DEFAULT_SEED}).")
    parser.add_argument("--max-input-tokens", type=int, default=2048, help="Maximum prompt length.")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Maximum generation length.")
    parser.add_argument("--max-triples", type=int, default=40, help="Maximum requested triples per chunk.")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    done = _load_done_chunk_ids(args.output)
    rows = _load_records(args.input, sample=args.sample, seed=args.seed, done_chunk_ids=done)
    print(f"Already done: {len(done)}  |  Pending: {len(rows)}")
    if not rows:
        return 0

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if args.adapter is not None:
        from peft import PeftModel

        print(f"Loading adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    for i, record in enumerate(rows, start=1):
        out_record = _generate_one(
            model=model,
            tokenizer=tokenizer,
            record=record,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            max_triples=args.max_triples,
        )
        _write_records(args.output, [out_record])
        if i % 5 == 0 or i == len(rows):
            print(f"  [{i}/{len(rows)}] saved -> {args.output}")

    print(f"Done -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
