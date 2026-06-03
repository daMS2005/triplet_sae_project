#!/usr/bin/env python3
"""Quantize a standalone checkpoint with GPTQModel's native API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gptqmodel import GPTQModel, QuantizeConfig


def _load_jsonl_texts(path: Path, limit: int) -> list[str]:
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(texts) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = None
            for key in ("text", "input_text", "prompt", "content"):
                value = record.get(key)
                if isinstance(value, str) and value.strip():
                    text = value.strip()
                    break
            if text:
                texts.append(text)
    return texts


def _build_calibration_dataset(paths: list[Path], max_samples: int) -> list[str]:
    if max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if not paths:
        raise ValueError("At least one --calibration-file is required")

    per_file = max(1, max_samples // len(paths))
    dataset: list[str] = []
    for path in paths:
        dataset.extend(_load_jsonl_texts(path, per_file))

    if len(dataset) < max_samples:
        for path in paths:
            remaining = max_samples - len(dataset)
            if remaining <= 0:
                break
            extra = _load_jsonl_texts(path, per_file + remaining)
            dataset.extend(extra[per_file:])

    dataset = dataset[:max_samples]
    if not dataset:
        raise RuntimeError("Calibration dataset came back empty")
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a standalone GPTQModel checkpoint.")
    parser.add_argument("--model", required=True, help="Merged full-precision checkpoint path.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bits", type=int, choices=[4, 8], default=4)
    parser.add_argument(
        "--calibration-file",
        action="append",
        default=[],
        type=Path,
        help="JSONL file with text-bearing records. Repeat for multiple files.",
    )
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--desc-act", action="store_true")
    parser.add_argument("--sym", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    calibration_dataset = _build_calibration_dataset(args.calibration_file, args.max_samples)

    quant_config = QuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        desc_act=args.desc_act,
        sym=args.sym,
    )
    model = GPTQModel.load(args.model, quantize_config=quant_config)
    model.quantize(calibration_dataset, batch_size=args.batch_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.output_dir)

    metadata = {
        "source_model": args.model,
        "bits": args.bits,
        "calibration_files": [str(path) for path in args.calibration_file],
        "max_samples": args.max_samples,
        "group_size": args.group_size,
        "batch_size": args.batch_size,
        "desc_act": args.desc_act,
        "sym": args.sym,
    }
    (args.output_dir / "quantization_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), **metadata}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
