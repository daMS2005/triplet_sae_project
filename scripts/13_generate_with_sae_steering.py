#!/usr/bin/env python3
"""Generate text with an SAE steering preset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "google/gemma-3-12b-it"

TRIPLET_SYSTEM_PROMPT = (
    "You extract factual knowledge triples from source text.\n"
    "Return exactly one JSON object with key 'triples'.\n"
    "Each triple must have string fields 'subject', 'predicate', and 'object'.\n"
    "Only include facts explicitly grounded in the text.\n"
    "Keep the output short and do not include commentary before or after the JSON."
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

SUMMARY_USER_TEMPLATE = "Summarize the text below in 2 to 3 sentences.\n\nText:\n{text}"

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
    return model.get_input_embeddings().weight.device


def _build_messages(text: str, prompt_mode: str) -> list[dict[str, str]]:
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


class SAEEditHook:
    def __init__(
        self,
        *,
        model,
        sae,
        layer: int,
        feature_ids: list[int],
        mode: str,
        multiplier: float,
        strength: float,
        edit_positions: str,
        edit_last_k: int,
    ) -> None:
        self.model = model
        self.sae = sae
        self.layer = int(layer)
        self.feature_ids = sorted(set(int(fid) for fid in feature_ids))
        self.mode = mode
        self.multiplier = float(multiplier)
        self.strength = float(strength)
        self.edit_positions = edit_positions
        self.edit_last_k = int(edit_last_k)
        self._handle = None

    def __enter__(self):
        self._handle = _get_decoder_layers(self.model)[self.layer].register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False

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
        batch, seq_len, hidden_dim = hidden.shape
        original_dtype = hidden.dtype
        original_device = hidden.device
        flat_hidden = hidden.reshape(batch * seq_len, hidden_dim).to(self.sae.device, dtype=torch.float32)
        mask = self._position_mask(batch, seq_len, hidden.device).reshape(batch * seq_len).to(self.sae.device)
        if int(mask.sum().item()) == 0:
            return output

        with torch.no_grad():
            latents = self.sae.encode(flat_hidden)
            edited = latents.clone()
            for feature_id in self.feature_ids:
                if self.mode == "ablate":
                    edited[mask, feature_id] = 0
                else:
                    edited[mask, feature_id] = edited[mask, feature_id] * self.multiplier + self.strength
            recon = self.sae.decode(latents)
            edited_recon = self.sae.decode(edited)
            edited_hidden = flat_hidden + (edited_recon - recon)

        edited_hidden = edited_hidden.reshape(batch, seq_len, hidden_dim).to(
            device=original_device,
            dtype=original_dtype,
        )
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
        mode: str,
        multiplier: float,
        strength: float,
        edit_positions: str,
        edit_last_k: int,
    ) -> None:
        self.hooks = [
            SAEEditHook(
                model=model,
                sae=saes_by_layer[layer],
                layer=layer,
                feature_ids=feature_ids,
                mode=mode,
                multiplier=multiplier,
                strength=strength,
                edit_positions=edit_positions,
                edit_last_k=edit_last_k,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate with a stored SAE feature preset.")
    parser.add_argument("--preset", required=True, help="Preset name from the preset file.")
    parser.add_argument(
        "--preset-file",
        type=Path,
        default=Path("configs/steering_feature_presets.json"),
        help="JSON file with staged steering presets.",
    )
    parser.add_argument("--text", default=None, help="Inline source text to prompt on.")
    parser.add_argument("--text-file", type=Path, default=None, help="File containing source text.")
    parser.add_argument("--prompt-mode", choices=("triplet", "weak_triplet", "summary"), default=None)
    parser.add_argument("--mode", choices=("steer", "ablate"), default="steer")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--steer-multiplier", type=float, default=10.0)
    parser.add_argument("--steer-strength", type=float, default=0.0)
    parser.add_argument("--edit-positions", choices=("final", "last_k", "all"), default="final")
    parser.add_argument("--edit-last-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    if not args.preset_file.exists():
        raise FileNotFoundError(f"Preset file not found: {args.preset_file}")
    preset_data = json.loads(args.preset_file.read_text(encoding="utf-8"))
    presets = preset_data.get("presets") or {}
    if args.preset not in presets:
        raise KeyError(f"Preset {args.preset!r} not found in {args.preset_file}")
    preset = presets[args.preset]

    if args.text is not None:
        text = args.text.strip()
    elif args.text_file is not None:
        text = args.text_file.read_text(encoding="utf-8").strip()
    else:
        raise ValueError("Pass either --text or --text-file.")
    if not text:
        raise ValueError("Input text is empty.")

    prompt_mode = args.prompt_mode or preset.get("recommended_prompt_mode") or "summary"
    layer_feature_ids = {int(layer): [int(fid) for fid in feature_ids] for layer, feature_ids in (preset.get("layer_feature_ids") or {}).items()}
    sae_release = _normalize_sae_release(str(preset_data.get("sae_release") or "gemma-scope-2-12b-it-res"))
    sae_id_template = str(preset_data.get("sae_id_template") or "layer_{layer}_width_16k_l0_medium")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    saes_by_layer: dict[int, SAE] = {}
    for layer in sorted(layer_feature_ids):
        sae, _, _ = SAE.from_pretrained(
            release=sae_release,
            sae_id=sae_id_template.format(layer=layer),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        sae.eval()
        saes_by_layer[layer] = sae

    messages = _build_messages(text, prompt_mode)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokenized = tokenizer(
        rendered,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_input_tokens,
    )
    input_device = _input_device(model)
    input_ids = tokenized["input_ids"].to(input_device)
    attention_mask = tokenized["attention_mask"].to(input_device)

    with MultiSAEEditHooks(
        model=model,
        saes_by_layer=saes_by_layer,
        layer_feature_ids=layer_feature_ids,
        mode=args.mode,
        multiplier=args.steer_multiplier,
        strength=args.steer_strength,
        edit_positions=args.edit_positions,
        edit_last_k=args.edit_last_k,
    ):
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

    new_tokens = generated[0, input_ids.shape[1] :]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    print(generated_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
