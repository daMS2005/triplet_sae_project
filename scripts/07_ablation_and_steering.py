#!/usr/bin/env python3
"""Run SAE-guided ablation and steering experiments on generation."""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "google/gemma-3-12b-it"
SAE_RELEASE = "gemma-scope-2-12b-it-res"
SAE_ID = "layer_24_width_16k_l0_medium"
TARGET_LAYER = 24
DEFAULT_MAX_INPUT_TOKENS = 512
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_SEED = 7
DEFAULT_ACTIVE_THRESHOLD = 1e-6


TRIPLET_SYSTEM_PROMPT = (
    "You extract factual knowledge triples from source text.\n"
    "Return exactly one JSON object with key 'triples'.\n"
    "Each triple must have string fields 'subject', 'predicate', and 'object'.\n"
    "Only include facts explicitly grounded in the text.\n"
    "Keep the output short and do not include any commentary before or after the JSON."
)

TRIPLET_USER_TEMPLATE = (
    "Extract up to 6 of the most important factual (subject, predicate, object) triples from the text below.\n"
    "Return strict JSON only, in this exact shape:\n"
    "{{\"triples\":[{{\"subject\":\"...\",\"predicate\":\"...\",\"object\":\"...\"}}]}}\n\n"
    "Text:\n{text}"
)

SUMMARY_SYSTEM_PROMPT = (
    "You summarize source text faithfully and briefly.\n"
    "Write 2 to 3 sentences.\n"
    "Do not use JSON or bullet points."
)

SUMMARY_USER_TEMPLATE = (
    "Summarize the text below in 2 to 3 sentences.\n\n"
    "Text:\n{text}"
)

WEAK_TRIPLET_SYSTEM_PROMPT = (
    "You help identify a few important factual relations in a passage.\n"
    "If there are clear facts, return a short JSON object with key 'triples'.\n"
    "Each triple should have 'subject', 'predicate', and 'object'.\n"
    "Keep it brief and only include the clearest relations."
)

WEAK_TRIPLET_USER_TEMPLATE = (
    "From the text below, extract up to 3 especially clear factual triples if they are obvious.\n"
    "Use this JSON shape:\n"
    "{{\"triples\":[{{\"subject\":\"...\",\"predicate\":\"...\",\"object\":\"...\"}}]}}\n"
    "If only a small number are obvious, that is fine.\n\n"
    "Text:\n{text}"
)


@dataclass(slots=True)
class Example:
    dataset: str
    chunk_id: str
    doc_id: str
    title: str
    text: str
    teacher_triples: list[dict]


@dataclass(slots=True)
class Condition:
    name: str
    edit_type: str
    layer_feature_ids: dict[int, list[int]]
    steering_strength: float
    steering_multiplier: float


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _normalize_sae_release(sae_release: str) -> str:
    aliases = {
        "-resid_post": "-res",
        "-attn_out": "-att",
        "-mlp_out": "-mlp",
    }
    for wrong_suffix, right_suffix in aliases.items():
        if sae_release.endswith(wrong_suffix):
            return sae_release[: -len(wrong_suffix)] + right_suffix
    return sae_release


def _parse_layer_from_sae_id(sae_id: str) -> int | None:
    match = re.search(r"(?:^|_)layer_(\d+)(?:_|$)", sae_id)
    return int(match.group(1)) if match else None


def _validate_layer_alignment(sae_id: str, target_layer: int, allow_mismatch: bool) -> None:
    sae_layer = _parse_layer_from_sae_id(sae_id)
    if sae_layer is None:
        raise ValueError(f"Could not parse layer number from SAE id: {sae_id}")
    if sae_layer != target_layer and not allow_mismatch:
        raise ValueError(
            f"SAE id {sae_id!r} is for layer {sae_layer}, but --target-layer is {target_layer}. "
            "Pass --allow-layer-mismatch only if you intentionally want this."
        )


def _sae_id_for_layer(sae_id_template: str, layer: int) -> str:
    if "{layer}" in sae_id_template:
        return sae_id_template.format(layer=layer)
    return sae_id_template


def _get_decoder_layers(model: torch.nn.Module):
    candidate_paths = (
        ("model", "layers"),
        ("model", "language_model", "layers"),
        ("language_model", "layers"),
    )
    for path in candidate_paths:
        current = model
        ok = True
        for attr in path:
            if not hasattr(current, attr):
                ok = False
                break
            current = getattr(current, attr)
        if ok:
            return current
    raise AttributeError("Could not find decoder layers on the loaded model.")


def _input_device(model: torch.nn.Module) -> torch.device:
    # With device_map="auto", `model.device` can be a leaky abstraction. The input
    # ids should start on the same device as the embedding table.
    return model.get_input_embeddings().weight.device


def _load_examples(paths: list[str], sample_per_file: int, seed: int) -> list[Example]:
    rng = random.Random(seed)
    examples: list[Example] = []
    for entry in paths:
        if "=" not in entry:
            raise ValueError(f"Expected DATASET=PATH format, got: {entry}")
        dataset, path_str = entry.split("=", 1)
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                text = str(record.get("text") or "").strip()
                triples = record.get("triples") or []
                # We only keep examples that already have teacher triples, because later we score
                # generated outputs against these labels.
                if not text or not triples:
                    continue
                rows.append(
                    Example(
                        dataset=dataset,
                        chunk_id=str(record.get("chunk_id") or ""),
                        doc_id=str(record.get("doc_id") or ""),
                        title=str(record.get("title") or ""),
                        text=text,
                        teacher_triples=list(triples),
                    )
                )
        if not rows:
            continue
        rng.shuffle(rows)
        # Keep the sampled set fixed so baseline vs edit conditions are directly comparable.
        examples.extend(rows[:sample_per_file])
    return examples


def _normalize_text(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"\s+", " ", value)
    return value


def _triples_to_set(triples: Iterable[dict]) -> set[tuple[str, str, str]]:
    normalized = set()
    for triple in triples:
        subject = _normalize_text(str(triple.get("subject") or ""))
        predicate = _normalize_text(str(triple.get("predicate") or ""))
        obj = _normalize_text(str(triple.get("object") or ""))
        if subject and predicate and obj:
            normalized.add((subject, predicate, obj))
    return normalized


def _compute_metrics(predicted: list[dict], gold: list[dict]) -> dict[str, float | int]:
    # Exact normalized triple matching is intentionally simple here. It is enough to tell whether
    # an edit makes the model recover more or fewer teacher-style triples.
    pred_set = _triples_to_set(predicted)
    gold_set = _triples_to_set(gold)
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_count": len(pred_set),
        "gold_count": len(gold_set),
    }


def _extract_json_payload(text: str) -> tuple[dict | list | None, bool]:
    stripped = text.strip()
    if stripped.startswith("```"):
        # Gemma often wraps JSON in markdown fences; strip them before parsing.
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()

    candidates: list[str] = []
    start_obj = stripped.find("{")
    end_obj = stripped.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj >= start_obj:
        candidates.append(stripped[start_obj : end_obj + 1])

    start_list = stripped.find("[")
    end_list = stripped.rfind("]")
    if start_list != -1 and end_list != -1 and end_list >= start_list:
        candidates.append(stripped[start_list : end_list + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate), True
        except json.JSONDecodeError:
            continue
    return None, False


def _parse_triples_from_truncated_json(text: str) -> list[dict]:
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
            triples.append(
                {
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                }
            )
    return triples


def _parse_generated_triples(text: str) -> tuple[list[dict], bool, bool]:
    payload, strict_json_success = _extract_json_payload(text)
    if payload is None:
        # Recover usable triples even when Gemma truncates or mangles the enclosing JSON.
        fallback = _parse_triples_from_truncated_json(text)
        return fallback, strict_json_success, bool(fallback)

    if isinstance(payload, dict):
        triples = payload.get("triples")
    else:
        triples = payload

    if not isinstance(triples, list):
        fallback = _parse_triples_from_truncated_json(text)
        return fallback, strict_json_success, bool(fallback)

    parsed = []
    for item in triples:
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject") or "").strip()
        predicate = str(item.get("predicate") or "").strip()
        obj = str(item.get("object") or "").strip()
        if subject and predicate and obj:
            parsed.append(
                {
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                }
            )
    if parsed:
        return parsed, strict_json_success, True
    fallback = _parse_triples_from_truncated_json(text)
    return fallback, strict_json_success, bool(fallback)


def _build_messages(text: str, prompt_mode: str) -> list[dict[str, str]]:
    # `triplet` is the strong extraction prompt.
    # `weak_triplet` is a softer "extract if obvious" prompt for near-boundary steering tests.
    # `summary` is the non-triplet control prompt.
    if prompt_mode == "summary":
        return [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": SUMMARY_USER_TEMPLATE.format(text=text)},
        ]
    if prompt_mode == "weak_triplet":
        return [
            {"role": "system", "content": WEAK_TRIPLET_SYSTEM_PROMPT},
            {"role": "user", "content": WEAK_TRIPLET_USER_TEMPLATE.format(text=text)},
        ]
    return [
        {"role": "system", "content": TRIPLET_SYSTEM_PROMPT},
        {"role": "user", "content": TRIPLET_USER_TEMPLATE.format(text=text)},
    ]


def _prepare_inputs(tokenizer, text: str, max_input_tokens: int, prompt_mode: str):
    messages = _build_messages(text, prompt_mode)
    # We always tokenize the full rendered chat prompt because the model sees roles/instructions too,
    # not just the raw passage text.
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    tokenized = tokenizer(
        rendered,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    return rendered, tokenized


class SAEEditHook:
    def __init__(
        self,
        model,
        sae,
        feature_ids: list[int],
        target_layer: int,
        edit_type: str,
        steering_strength: float,
        steering_multiplier: float,
        edit_positions: str,
        edit_last_k: int,
        steer_mode: str,
        active_threshold: float,
    ) -> None:
        self.model = model
        self.sae = sae
        self.feature_ids = sorted(set(int(fid) for fid in feature_ids))
        self.target_layer = int(target_layer)
        self.edit_type = str(edit_type)
        self.steering_strength = float(steering_strength)
        self.steering_multiplier = float(steering_multiplier)
        self.edit_positions = str(edit_positions)
        self.edit_last_k = int(edit_last_k)
        self.steer_mode = str(steer_mode)
        self.active_threshold = float(active_threshold)
        self._handle = None
        self.calls = 0
        self.positions_seen = 0
        self.positions_edited = 0
        self.latent_before_sum = 0.0
        self.latent_after_sum = 0.0
        self.latent_abs_delta_sum = 0.0
        self.active_before_sum = 0.0
        self.active_after_sum = 0.0

    def __enter__(self):
        decoder_layers = _get_decoder_layers(self.model)
        self._handle = decoder_layers[self.target_layer].register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False

    def diagnostics(self) -> dict[str, float | int | str]:
        denom = max(1, self.positions_edited * max(1, len(self.feature_ids)))
        return {
            "edit_positions": self.edit_positions,
            "edit_last_k": self.edit_last_k,
            "steer_mode": self.steer_mode,
            "active_threshold": self.active_threshold,
            "hook_calls": self.calls,
            "positions_seen": self.positions_seen,
            "positions_edited": self.positions_edited,
            "mean_target_latent_before": self.latent_before_sum / denom,
            "mean_target_latent_after": self.latent_after_sum / denom,
            "mean_abs_target_latent_delta": self.latent_abs_delta_sum / denom,
            "active_fraction_before": self.active_before_sum / denom,
            "active_fraction_after": self.active_after_sum / denom,
        }

    def _position_mask(self, batch: int, seq_len: int, device: torch.device) -> torch.Tensor:
        if self.edit_positions == "all":
            return torch.ones((batch, seq_len), dtype=torch.bool, device=device)
        if self.edit_positions == "final":
            mask = torch.zeros((batch, seq_len), dtype=torch.bool, device=device)
            mask[:, -1] = True
            return mask
        if self.edit_positions == "last_k":
            mask = torch.zeros((batch, seq_len), dtype=torch.bool, device=device)
            k = max(1, min(self.edit_last_k, seq_len))
            mask[:, -k:] = True
            return mask
        raise ValueError(f"Unknown edit position mode: {self.edit_positions}")

    def _hook(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        original_dtype = hidden.dtype
        original_device = hidden.device
        batch, seq_len, hidden_dim = hidden.shape
        position_mask = self._position_mask(batch, seq_len, hidden.device).reshape(batch * seq_len)
        selected_positions = int(position_mask.sum().item())
        self.calls += 1
        self.positions_seen += batch * seq_len
        self.positions_edited += selected_positions
        if selected_positions == 0:
            return output

        # Flatten batch and sequence so the SAE sees a simple matrix of token residuals.
        flat_hidden = hidden.reshape(batch * seq_len, hidden_dim).to(self.sae.device, dtype=torch.float32)
        with torch.no_grad():
            latents = self.sae.encode(flat_hidden)
            edited_latents = latents.clone()
            edit_mask = position_mask.to(self.sae.device)
            feature_tensor = torch.tensor(self.feature_ids, device=latents.device, dtype=torch.long)
            before = latents[edit_mask][:, feature_tensor]
            if self.edit_type == "ablate":
                # Ablation removes the chosen latent(s) only at the selected token positions.
                for feature_id in self.feature_ids:
                    edited_latents[edit_mask, feature_id] = 0
            elif self.edit_type == "steer":
                # Steering can either add a constant, multiply existing activation, or do both.
                # Multiplicative steering preserves the feature's sign/scale better than forcing
                # a fixed +constant onto every feature.
                for feature_id in self.feature_ids:
                    if self.steer_mode == "active_only":
                        active_mask = edit_mask & (latents[:, feature_id] > self.active_threshold)
                        target_mask = active_mask
                    else:
                        target_mask = edit_mask
                    edited_latents[target_mask, feature_id] = (
                        edited_latents[target_mask, feature_id] * self.steering_multiplier
                        + self.steering_strength
                    )
            after = edited_latents[edit_mask][:, feature_tensor]
            self.latent_before_sum += float(before.sum().item())
            self.latent_after_sum += float(after.sum().item())
            self.latent_abs_delta_sum += float((after - before).abs().sum().item())
            self.active_before_sum += float((before > self.active_threshold).to(torch.float32).sum().item())
            self.active_after_sum += float((after > self.active_threshold).to(torch.float32).sum().item())
            recon = self.sae.decode(latents)
            edited_recon = self.sae.decode(edited_latents)
            # Apply only the SAE reconstruction delta so we edit the target feature while preserving the original state.
            edited_hidden = flat_hidden + (edited_recon - recon)

        # Convert back to the model's original dtype/device so the rest of generation continues normally.
        edited_hidden = edited_hidden.reshape(batch, seq_len, hidden_dim).to(device=original_device, dtype=original_dtype)
        if isinstance(output, tuple):
            return (edited_hidden,) + tuple(output[1:])
        return edited_hidden


class MultiSAEEditHooks:
    def __init__(
        self,
        *,
        model,
        saes_by_layer: dict[int, SAE],
        layer_feature_ids: dict[int, list[int]],
        edit_type: str,
        steering_strength: float,
        steering_multiplier: float,
        edit_positions: str,
        edit_last_k: int,
        steer_mode: str,
        active_threshold: float,
    ) -> None:
        self.hooks = [
            SAEEditHook(
                model=model,
                sae=saes_by_layer[layer],
                feature_ids=feature_ids,
                target_layer=layer,
                edit_type=edit_type,
                steering_strength=steering_strength,
                steering_multiplier=steering_multiplier,
                edit_positions=edit_positions,
                edit_last_k=edit_last_k,
                steer_mode=steer_mode,
                active_threshold=active_threshold,
            )
            for layer, feature_ids in sorted(layer_feature_ids.items())
            if feature_ids
        ]

    def __enter__(self):
        for hook in self.hooks:
            hook.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        for hook in reversed(self.hooks):
            hook.__exit__(exc_type, exc, tb)
        return False

    def diagnostics(self) -> dict[str, dict[str, float | int | str]]:
        return {str(hook.target_layer): hook.diagnostics() for hook in self.hooks}


def _generate_for_condition(
    *,
    model,
    tokenizer,
    saes_by_layer: dict[int, SAE],
    condition: Condition,
    examples: list[Example],
    edit_positions: str,
    steer_mode: str,
    active_threshold: float,
    edit_last_k: int,
    max_input_tokens: int,
    max_new_tokens: int,
    prompt_mode: str,
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    total_tp = total_fp = total_fn = 0
    strict_json_successes = 0
    loose_parse_successes = 0
    total_predicted = 0

    # Baseline runs with no hook. Intervention runs can install hooks on multiple
    # layers at once, so we can test a cross-layer feature set as one causal edit.
    context = (
        MultiSAEEditHooks(
            model=model,
            saes_by_layer=saes_by_layer,
            layer_feature_ids=condition.layer_feature_ids,
            edit_type=condition.edit_type,
            steering_strength=condition.steering_strength,
            steering_multiplier=condition.steering_multiplier,
            edit_positions=edit_positions,
            edit_last_k=edit_last_k,
            steer_mode=steer_mode,
            active_threshold=active_threshold,
        )
        if condition.layer_feature_ids
        else None
    )

    manager = context if context is not None else _nullcontext()
    with manager:
        for example in examples:
            prompt_text, tokenized = _prepare_inputs(tokenizer, example.text, max_input_tokens, prompt_mode)
            input_device = _input_device(model)
            input_ids = tokenized["input_ids"].to(input_device)
            attention_mask = tokenized["attention_mask"].to(input_device)

            with torch.no_grad():
                # Generation happens on the normal Gemma model; the SAE only edits the chosen
                # layer's residuals in-flight.
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
            # Even for non-triplet prompts we still run the triple parser, so we can measure whether
            # steering made the output drift toward explicit triplet structure.
            parsed_triples, strict_json_success, loose_parse_success = _parse_generated_triples(generated_text)
            metrics = _compute_metrics(parsed_triples, example.teacher_triples)

            strict_json_successes += int(strict_json_success)
            loose_parse_successes += int(loose_parse_success)
            total_tp += int(metrics["tp"])
            total_fp += int(metrics["fp"])
            total_fn += int(metrics["fn"])
            total_predicted += int(metrics["predicted_count"])

            rows.append(
                {
                    "condition": condition.name,
                    "edit_type": condition.edit_type,
                    "dataset": example.dataset,
                    "chunk_id": example.chunk_id,
                    "doc_id": example.doc_id,
                    "title": example.title,
                    "feature_ids": [
                        feature_id
                        for _, feature_ids in sorted(condition.layer_feature_ids.items())
                        for feature_id in feature_ids
                    ],
                    "layer_feature_ids": {
                        str(layer): feature_ids
                        for layer, feature_ids in sorted(condition.layer_feature_ids.items())
                    },
                    "steering_strength": condition.steering_strength,
                    "steering_multiplier": condition.steering_multiplier,
                    "prompt_mode": prompt_mode,
                    "prompt_text": prompt_text,
                    "generated_text": generated_text,
                    "strict_json_success": strict_json_success,
                    "loose_parse_success": loose_parse_success,
                    "parse_success": loose_parse_success,
                    "predicted_triples": parsed_triples,
                    "teacher_triples": example.teacher_triples,
                    "metrics": metrics,
                }
            )

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    summary = {
        "condition": condition.name,
        "edit_type": condition.edit_type,
        "feature_ids": [
            feature_id
            for _, feature_ids in sorted(condition.layer_feature_ids.items())
            for feature_id in feature_ids
        ],
        "layer_feature_ids": {
            str(layer): feature_ids for layer, feature_ids in sorted(condition.layer_feature_ids.items())
        },
        "target_layers": sorted(condition.layer_feature_ids),
        "steering_strength": condition.steering_strength,
        "steering_multiplier": condition.steering_multiplier,
        "edit_positions": edit_positions,
        "edit_last_k": edit_last_k,
        "steer_mode": steer_mode,
        "active_threshold": active_threshold,
        "prompt_mode": prompt_mode,
        "example_count": len(examples),
        # Keep strict JSON success separate from loose recovery. Loose parsing is useful
        # diagnostically, but it should not be mistaken for valid JSON adherence.
        "strict_json_success_rate": strict_json_successes / len(examples) if examples else 0.0,
        "loose_parse_success_rate": loose_parse_successes / len(examples) if examples else 0.0,
        "parse_success_rate": loose_parse_successes / len(examples) if examples else 0.0,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "avg_predicted_triples": total_predicted / len(examples) if examples else 0.0,
        "intervention_diagnostics": context.diagnostics() if context is not None else {},
    }
    return rows, summary


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def _condition_checkpoint_paths(output_dir: Path, condition: Condition) -> tuple[Path, Path]:
    # Checkpoints are per condition, so a preemption only loses the condition that
    # was actively generating. Already-finished baselines/edits are reused on rerun.
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", condition.name).strip("_")
    checkpoint_dir = output_dir / "checkpoints"
    return checkpoint_dir / f"{safe_name}.rows.jsonl", checkpoint_dir / f"{safe_name}.summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline and SAE ablation experiments for triplet extraction."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="DATASET=FILE",
        help="Matched JSONL input file(s) to sample from.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--sample-per-file",
        type=int,
        default=4,
        metavar="N",
        help="How many examples to sample from each input file (default: 4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        metavar="N",
        help=f"Random seed for deterministic sampling (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--ablate",
        type=int,
        action="append",
        default=[],
        metavar="FEATURE_ID",
        help="Single feature to ablate. May be passed multiple times.",
    )
    parser.add_argument(
        "--ablate-set",
        action="append",
        default=[],
        metavar="NAME=ID1,ID2,...",
        help="Named feature set to ablate together.",
    )
    parser.add_argument(
        "--steer",
        type=int,
        action="append",
        default=[],
        metavar="FEATURE_ID",
        help="Single feature to positively steer. May be passed multiple times.",
    )
    parser.add_argument(
        "--steer-set",
        action="append",
        default=[],
        metavar="NAME=ID1,ID2,...",
        help="Named feature set to positively steer together.",
    )
    parser.add_argument(
        "--ablate-layer-set",
        action="append",
        default=[],
        metavar="NAME=LAYER:ID1,ID2|LAYER:ID3,ID4",
        help="Named multi-layer feature set to ablate together.",
    )
    parser.add_argument(
        "--steer-layer-set",
        action="append",
        default=[],
        metavar="NAME=LAYER:ID1,ID2|LAYER:ID3,ID4",
        help="Named multi-layer feature set to positively steer together.",
    )
    parser.add_argument(
        "--steer-strength",
        type=float,
        default=20.0,
        metavar="X",
        help="Additive latent steering strength for --steer / --steer-set (default: 20.0).",
    )
    parser.add_argument(
        "--steer-multiplier",
        type=float,
        default=1.0,
        metavar="X",
        help="Multiplicative latent steering factor for --steer / --steer-set (default: 1.0).",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("triplet", "summary", "weak_triplet"),
        default="triplet",
        help="Generation prompt style to use (default: triplet).",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"Model name (default: {MODEL_NAME}).",
    )
    parser.add_argument(
        "--sae-release",
        default=SAE_RELEASE,
        help=f"SAE release (default: {SAE_RELEASE}).",
    )
    parser.add_argument(
        "--sae-id",
        default=SAE_ID,
        help=f"SAE id (default: {SAE_ID}).",
    )
    parser.add_argument(
        "--sae-id-template",
        default="layer_{layer}_width_16k_l0_medium",
        help=(
            "SAE id template used for --ablate-layer-set / --steer-layer-set "
            "(default: layer_{layer}_width_16k_l0_medium)."
        ),
    )
    parser.add_argument(
        "--target-layer",
        type=int,
        default=TARGET_LAYER,
        metavar="N",
        help=f"Decoder layer where the SAE edit hook is installed (default: {TARGET_LAYER}).",
    )
    parser.add_argument(
        "--allow-layer-mismatch",
        action="store_true",
        help="Allow --sae-id layer and --target-layer to differ. Off by default to prevent invalid edits.",
    )
    parser.add_argument(
        "--edit-positions",
        choices=("final", "last_k", "all"),
        default="final",
        help=(
            "Token positions to edit inside each forward pass. 'final' edits only hidden[:, -1, :] "
            "and is safest for generation; 'last_k' edits a short suffix; 'all' preserves the old blunt "
            "behavior (default: final)."
        ),
    )
    parser.add_argument(
        "--edit-last-k",
        type=int,
        default=8,
        metavar="N",
        help="How many suffix token positions to edit when --edit-positions last_k is used (default: 8).",
    )
    parser.add_argument(
        "--steer-mode",
        choices=("always", "active_only"),
        default="always",
        help=(
            "For steering, either add to selected features at every edited position or only when "
            "the feature is already active above --active-threshold (default: always)."
        ),
    )
    parser.add_argument(
        "--active-threshold",
        type=float,
        default=DEFAULT_ACTIVE_THRESHOLD,
        metavar="X",
        help=f"Feature-active threshold for diagnostics and active_only steering (default: {DEFAULT_ACTIVE_THRESHOLD}).",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=DEFAULT_MAX_INPUT_TOKENS,
        metavar="N",
        help=f"Maximum prompt input length in tokens (default: {DEFAULT_MAX_INPUT_TOKENS}).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        metavar="N",
        help=f"Maximum generated tokens (default: {DEFAULT_MAX_NEW_TOKENS}).",
    )
    parser.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help="Compute device (default: auto).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable per-condition checkpoint reuse and recompute every condition.",
    )
    return parser.parse_args()


def _parse_layer_feature_spec(raw: str) -> tuple[str, dict[int, list[int]]]:
    if "=" not in raw:
        raise ValueError(f"Expected NAME=LAYER:ID1,ID2|LAYER:ID3,ID4 format, got {raw}")
    name, spec = raw.split("=", 1)
    layer_feature_ids: dict[int, list[int]] = {}
    for group in re.split(r"[|;]", spec):
        group = group.strip()
        if not group:
            continue
        if ":" not in group:
            raise ValueError(f"Expected LAYER:ID1,ID2 in multi-layer set {raw!r}, got {group!r}")
        layer_text, ids_text = group.split(":", 1)
        layer = int(layer_text)
        feature_ids = [int(part) for part in ids_text.split(",") if part.strip()]
        if feature_ids:
            layer_feature_ids.setdefault(layer, [])
            layer_feature_ids[layer].extend(feature_ids)
    if not layer_feature_ids:
        raise ValueError(f"No feature ids found in multi-layer set: {raw}")
    deduped = {
        layer: sorted(set(feature_ids))
        for layer, feature_ids in sorted(layer_feature_ids.items())
    }
    return name, deduped


def _parse_conditions(args: argparse.Namespace) -> list[Condition]:
    conditions = [
        Condition(
            name="baseline",
            edit_type="none",
            layer_feature_ids={},
            steering_strength=0.0,
            steering_multiplier=1.0,
        )
    ]
    # Every extra condition is the same examples plus one different latent edit, which keeps the
    # comparison as controlled as possible.
    for feature_id in args.ablate:
        conditions.append(
            Condition(
                name=f"ablate_{feature_id}",
                edit_type="ablate",
                layer_feature_ids={int(args.target_layer): [int(feature_id)]},
                steering_strength=0.0,
                steering_multiplier=1.0,
            )
        )
    for raw in args.ablate_set:
        if "=" not in raw:
            raise ValueError(f"Expected NAME=ID1,ID2,... for --ablate-set, got {raw}")
        name, ids = raw.split("=", 1)
        feature_ids = [int(part) for part in ids.split(",") if part.strip()]
        conditions.append(
            Condition(
                name=name,
                edit_type="ablate",
                layer_feature_ids={int(args.target_layer): feature_ids},
                steering_strength=0.0,
                steering_multiplier=1.0,
            )
        )
    for raw in args.ablate_layer_set:
        name, layer_feature_ids = _parse_layer_feature_spec(raw)
        conditions.append(
            Condition(
                name=name,
                edit_type="ablate",
                layer_feature_ids=layer_feature_ids,
                steering_strength=0.0,
                steering_multiplier=1.0,
            )
        )
    for feature_id in args.steer:
        conditions.append(
            Condition(
                name=f"steer_{feature_id}",
                edit_type="steer",
                layer_feature_ids={int(args.target_layer): [int(feature_id)]},
                steering_strength=float(args.steer_strength),
                steering_multiplier=float(args.steer_multiplier),
            )
        )
    for raw in args.steer_set:
        if "=" not in raw:
            raise ValueError(f"Expected NAME=ID1,ID2,... for --steer-set, got {raw}")
        name, ids = raw.split("=", 1)
        feature_ids = [int(part) for part in ids.split(",") if part.strip()]
        # Named groups let us test whether several candidate features act together more strongly than singles.
        conditions.append(
            Condition(
                name=name,
                edit_type="steer",
                layer_feature_ids={int(args.target_layer): feature_ids},
                steering_strength=float(args.steer_strength),
                steering_multiplier=float(args.steer_multiplier),
            )
        )
    for raw in args.steer_layer_set:
        name, layer_feature_ids = _parse_layer_feature_spec(raw)
        conditions.append(
            Condition(
                name=name,
                edit_type="steer",
                layer_feature_ids=layer_feature_ids,
                steering_strength=float(args.steer_strength),
                steering_multiplier=float(args.steer_multiplier),
            )
        )
    return conditions


def _sae_feature_count(sae) -> int:
    cfg = getattr(sae, "cfg", None)
    d_sae = getattr(cfg, "d_sae", None)
    if d_sae is not None:
        return int(d_sae)
    w_dec = getattr(sae, "W_dec", None)
    if isinstance(w_dec, torch.Tensor) and w_dec.ndim >= 1:
        return int(w_dec.shape[0])
    raise AttributeError("Could not determine SAE feature count for feature-id validation.")


def _validate_feature_ids(conditions: list[Condition], feature_counts_by_layer: dict[int, int]) -> None:
    invalid: list[tuple[str, int]] = []
    for condition in conditions:
        for layer, feature_ids in condition.layer_feature_ids.items():
            feature_count = feature_counts_by_layer[layer]
            for feature_id in feature_ids:
                if feature_id < 0 or feature_id >= feature_count:
                    invalid.append((f"{condition.name}:layer{layer}", feature_id))
    if invalid:
        details = ", ".join(f"{name}:{feature_id}" for name, feature_id in invalid[:20])
        raise ValueError(f"Invalid feature ids: {details}")


def main() -> int:
    args = parse_args()
    device = args.device or _best_device()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    examples = _load_examples(args.input, sample_per_file=args.sample_per_file, seed=args.seed)
    if not examples:
        raise ValueError("No examples loaded from the input files.")
    print(f"Loaded {len(examples)} examples")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    sae_release = _normalize_sae_release(args.sae_release)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = _parse_conditions(args)
    needed_layers = sorted(
        {
            layer
            for condition in conditions
            for layer in condition.layer_feature_ids
        }
    )
    uses_layer_sets = bool(args.ablate_layer_set or args.steer_layer_set)

    saes_by_layer: dict[int, SAE] = {}
    sae_ids_by_layer: dict[int, str] = {}
    for layer in needed_layers:
        sae_id = _sae_id_for_layer(args.sae_id_template, layer) if uses_layer_sets else args.sae_id
        _validate_layer_alignment(sae_id, layer, args.allow_layer_mismatch)
        print(f"Loading SAE for layer {layer}: {sae_release} / {sae_id}")
        sae, _, _ = SAE.from_pretrained(
            release=sae_release,
            sae_id=sae_id,
            device=device,
        )
        sae.eval()
        saes_by_layer[layer] = sae
        sae_ids_by_layer[layer] = sae_id

    _validate_feature_ids(
        conditions,
        {layer: _sae_feature_count(sae) for layer, sae in saes_by_layer.items()},
    )
    resume = not args.no_resume
    all_rows: list[dict] = []
    summaries: list[dict] = []
    for condition in conditions:
        rows_checkpoint, summary_checkpoint = _condition_checkpoint_paths(output_dir, condition)
        if resume and rows_checkpoint.exists() and summary_checkpoint.exists():
            print(f"Skipping completed condition from checkpoint: {condition.name}")
            all_rows.extend(_read_jsonl(rows_checkpoint))
            with summary_checkpoint.open("r", encoding="utf-8") as f:
                summaries.append(json.load(f))
            continue

        print(f"Running condition: {condition.name}")
        rows, summary = _generate_for_condition(
            model=model,
            tokenizer=tokenizer,
            saes_by_layer=saes_by_layer,
            condition=condition,
            examples=examples,
            edit_positions=args.edit_positions,
            steer_mode=args.steer_mode,
            active_threshold=args.active_threshold,
            edit_last_k=args.edit_last_k,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            prompt_mode=args.prompt_mode,
        )
        all_rows.extend(rows)
        summaries.append(summary)
        _write_jsonl(rows_checkpoint, rows)
        _write_json_atomic(summary_checkpoint, summary)
        print(f"Checkpointed condition: {condition.name} -> {summary_checkpoint}")
        print(json.dumps(summary, ensure_ascii=False))

    _write_jsonl(output_dir / "per_example_outputs.jsonl", all_rows)
    _write_json_atomic(
        output_dir / "summary.json",
        {
            "seed": args.seed,
            "sample_per_file": args.sample_per_file,
            "model": args.model,
            "sae_release": sae_release,
            "sae_id": args.sae_id,
            "sae_id_template": args.sae_id_template,
            "sae_ids_by_layer": {str(layer): sae_id for layer, sae_id in sae_ids_by_layer.items()},
            "target_layer": args.target_layer,
            "target_layers": needed_layers,
            "edit_positions": args.edit_positions,
            "edit_last_k": args.edit_last_k,
            "steer_mode": args.steer_mode,
            "active_threshold": args.active_threshold,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "prompt_mode": args.prompt_mode,
            "steer_strength": args.steer_strength,
            "steer_multiplier": args.steer_multiplier,
            "resume_enabled": resume,
            "conditions": summaries,
            "examples": [
                {
                    "dataset": example.dataset,
                    "chunk_id": example.chunk_id,
                    "doc_id": example.doc_id,
                    "title": example.title,
                }
                for example in examples
            ],
        },
    )

    print(f"Wrote outputs -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
