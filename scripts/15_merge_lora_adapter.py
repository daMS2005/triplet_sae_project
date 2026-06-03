#!/usr/bin/env python3
"""Merge a PEFT/LoRA adapter into its base model and save a full checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_dtype(name: str):
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a base model.")
    parser.add_argument("--base-model", default="google/gemma-3-4b-it")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Dtype to load/save the merged model with.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map for loading the base model (default: auto).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.adapter.exists():
        raise FileNotFoundError(f"Adapter not found: {args.adapter}")

    dtype = _resolve_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=args.device_map,
    )
    model = PeftModel.from_pretrained(model, str(args.adapter))
    merged_model = model.merge_and_unload()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Merged model saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
