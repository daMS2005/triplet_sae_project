#!/usr/bin/env python3
"""Build task-conditioned prompt datasets from chunk or matched JSONL records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TASK_TEMPLATES: dict[str, dict[str, str]] = {
    "triplet_extract": {
        "system": (
            "You extract factual knowledge triples from source text. "
            "Be concise and literal."
        ),
        "user": (
            "Extract factual (subject, predicate, object) triples from the text below.\n\n"
            "Text:\n{text}\n\n"
            "Return only grounded triples."
        ),
        "fallback": (
            "Instruction: Extract factual (subject, predicate, object) triples from the "
            "text below.\n\nText:\n{text}\n\nResponse:"
        ),
    },
    "summary": {
        "system": (
            "You summarize source text faithfully and briefly."
        ),
        "user": (
            "Summarize the text below in 2 to 3 sentences.\n\n"
            "Text:\n{text}"
        ),
        "fallback": (
            "Instruction: Summarize the text below in 2 to 3 sentences.\n\n"
            "Text:\n{text}\n\nResponse:"
        ),
    },
    "qa": {
        "system": (
            "You answer questions about source text faithfully."
        ),
        "user": (
            "Read the text below and answer: What are the most important factual points?\n\n"
            "Text:\n{text}"
        ),
        "fallback": (
            "Instruction: Read the text below and answer: What are the most important "
            "factual points?\n\nText:\n{text}\n\nResponse:"
        ),
    },
}


def build_task_prompts(input_path: Path, output_path: Path, task: str) -> None:
    template = TASK_TEMPLATES[task]
    written = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as in_f, output_path.open(
        "w", encoding="utf-8"
    ) as out_f:
        for line in in_f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            source_text = str(record.get("text") or "").strip()
            if not source_text:
                continue

            prompted_text = template["fallback"].format(text=source_text)
            output_record = {
                **record,
                "source_text": source_text,
                "task_name": task,
                "messages": [
                    {"role": "system", "content": template["system"]},
                    {"role": "user", "content": template["user"].format(text=source_text)},
                ],
                "prompted_text": prompted_text,
            }
            out_f.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Built {written} task-conditioned records -> {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wrap JSONL text records in a task prompt for activation collection."
    )
    parser.add_argument(
        "--input", type=Path, required=True, metavar="FILE",
        help="Input JSONL file with a text field.",
    )
    parser.add_argument(
        "--output", type=Path, required=True, metavar="FILE",
        help="Output JSONL file with task prompt metadata.",
    )
    parser.add_argument(
        "--task", choices=sorted(TASK_TEMPLATES), required=True,
        help="Task prompt template to apply.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    build_task_prompts(args.input, args.output, args.task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
