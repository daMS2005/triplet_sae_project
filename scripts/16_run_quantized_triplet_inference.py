#!/usr/bin/env python3
"""Run triplet extraction with optional 8-bit or 4-bit bitsandbytes quantization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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


def _extract_json_payload(text: str):
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end >= start:
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _build_quant_config(mode: str | None):
    if mode is None or mode == "none":
        return None
    if mode == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if mode == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    raise ValueError(f"Unsupported quantization mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quantized Gemma triplet inference.")
    parser.add_argument("--model", required=True, help="Merged model path or HF model id.")
    parser.add_argument("--title", default="")
    parser.add_argument("--text", required=True)
    parser.add_argument("--quantization", choices=["none", "8bit", "4bit"], default="none")
    parser.add_argument("--max-triples", type=int, default=20)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--min-new-tokens", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    quant_config = _build_quant_config(args.quantization)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quant_config,
    )
    model.eval()

    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    max_triples=args.max_triples,
                    title=args.title,
                    text=args.text,
                ),
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    tokenized = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_input_tokens,
    )
    input_ids = tokenized["input_ids"].to(model.device)
    attention_mask = tokenized["attention_mask"].to(model.device)

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(
        generated[0, input_ids.shape[1] :],
        skip_special_tokens=True,
    ).strip()
    payload = _extract_json_payload(generated_text)
    print(json.dumps(
        {
            "model": args.model,
            "quantization": args.quantization,
            "title": args.title,
            "generated_text": generated_text,
            "parsed_payload": payload,
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
