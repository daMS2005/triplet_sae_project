#!/usr/bin/env python3
"""Collect residual-stream activations and SAE feature activations from Gemma 3 12B IT.

For each text chunk in the input JSONL:
  1. Tokenize (truncated to MAX_SEQ_LEN tokens).
  2. Run a forward pass through google/gemma-3-12b-it with hooks on one or more target layers.
  3. Pass each captured residual stream through the matching GemmaScope SAE to get sparse feature acts.
  4. Save per-chunk results to a .pt file.

Output record per chunk:
  {
    "chunk_id":       str,
    "doc_id":         str,
    "task_name":      str,
    "model_input_text": str,
    "input_ids":      Tensor (seq_len,)      -- token ids fed to the model,
    "tokens":         list[str]              -- decoded tokens,
    "primary_layer":  int,
    "target_layers":  list[int],
    "mean_resid":     Tensor (hidden_dim,)   -- mean-pooled residual stream for the primary layer,
    "sae_acts":       Tensor (seq_len, dict_size) -- SAE acts for the primary layer,
    "mean_resid_by_layer": dict[str, Tensor] -- mean residual per collected layer,
    "sae_acts_by_layer": dict[str, Tensor]   -- token-level SAE acts per collected layer,
    "token_count":    int,
  }

Resume-safe: already-processed chunk_ids in the output file are skipped.

GemmaScope SAE release: gemma-scope-2-12b-it-res
Available layers: 12, 24, 31, 41
Available widths: 16k, 65k, 262k, 1m
Available l0 sparsities: small, medium, big
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE


# ---------------------------------------------------------------------------
# Defaults (override via CLI)
# ---------------------------------------------------------------------------

MODEL_NAME   = "google/gemma-3-12b-it"
SAE_RELEASE  = "gemma-scope-2-12b-it-res"
SAE_ID       = "layer_24_width_16k_l0_medium"   # layer 24, 16k dict, l0≈60
TARGET_LAYER = 24
MAX_SEQ_LEN  = 512    # truncate chunks to this many tokens
BATCH_SIZE   = 1
DEFAULT_LAYERS = [12, 24, 31, 41]


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _load_done_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    data = torch.load(output_path, map_location="cpu", weights_only=False)
    return {rec["chunk_id"] for rec in data}


def _load_existing(output_path: Path) -> list[dict]:
    if not output_path.exists():
        return []
    return torch.load(output_path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def _parse_sae_layer(sae_id: str) -> int:
    """Extract layer index from SAE id string like 'layer_24_width_16k_l0_medium'."""
    # The SAE name already encodes which model layer it was trained on, so we recover it here
    # instead of keeping a second manual config that could drift out of sync.
    parts = sae_id.split("_")
    return int(parts[1])


def _get_decoder_layers(model: torch.nn.Module):
    """Return the decoder layer list for different HF model wrappers."""
    # Different Hugging Face model classes nest the transformer blocks differently.
    # We probe the common paths so the same script works across Gemma wrapper variants.
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
    raise AttributeError(
        "Could not find decoder layers on the loaded model. "
        "Tried: model.layers, model.language_model.layers, language_model.layers."
    )


def _normalize_sae_release(sae_release: str) -> str:
    """Handle common GemmaScope alias mistakes."""
    aliases = {
        "-resid_post": "-res",
        "-attn_out": "-att",
        "-mlp_out": "-mlp",
    }
    for wrong_suffix, right_suffix in aliases.items():
        if sae_release.endswith(wrong_suffix):
            normalized = sae_release[: -len(wrong_suffix)] + right_suffix
            print(f"Normalizing SAE release '{sae_release}' -> '{normalized}'")
            return normalized
    return sae_release


def _parse_layers(raw_layers: str | None, fallback_layer: int) -> list[int]:
    if raw_layers is None:
        return [int(fallback_layer)]
    parsed = []
    for part in raw_layers.split(","):
        part = part.strip()
        if not part:
            continue
        parsed.append(int(part))
    if not parsed:
        return [int(fallback_layer)]
    # Preserve user order while dropping duplicates.
    return list(dict.fromkeys(parsed))


def _replace_sae_layer(sae_id: str, layer: int) -> str:
    current_layer = _parse_sae_layer(sae_id)
    return sae_id.replace(f"layer_{current_layer}_", f"layer_{int(layer)}_", 1)


def _render_model_input(tokenizer, record: dict, max_seq_len: int):
    messages = record.get("messages")
    if isinstance(messages, list) and hasattr(tokenizer, "apply_chat_template"):
        # Recreate the exact chat-formatted prompt so activations reflect the real task setup.
        rendered_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        tokens = tokenizer(
            rendered_text,
            return_tensors="pt",
            max_length=max_seq_len,
            truncation=True,
        )
        return rendered_text, tokens

    model_input_text = str(
        record.get("prompted_text")
        or record.get("model_input_text")
        or record.get("text")
        or ""
    )
    # Fallback path for plain-text records that were not built from chat messages.
    tokens = tokenizer(
        model_input_text,
        return_tensors="pt",
        max_length=max_seq_len,
        truncation=True,
    )
    return model_input_text, tokens


def collect(
    input_path: Path,
    output_path: Path,
    model_name: str,
    sae_release: str,
    sae_id: str,
    target_layers: list[int],
    max_seq_len: int,
    device: str,
) -> None:
    sae_release = _normalize_sae_release(sae_release)
    primary_layer = int(target_layers[0])

    # ---- Load already-done chunk ids ----
    # The output file is a growing list of per-chunk activation records.
    # On reruns we skip chunk_ids that are already present so large jobs can resume safely.
    done_ids = _load_done_ids(output_path)
    results  = _load_existing(output_path)

    # ---- Load chunks ----
    chunks = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("chunk_id") and rec["chunk_id"] not in done_ids:
                chunks.append(rec)

    print(f"Already done: {len(done_ids)}  |  Pending: {len(chunks)}")
    if not chunks:
        print("Nothing to do.")
        return

    # ---- Load tokenizer + model ----
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    # Gemma 3 is wrapped differently across HF releases, so we resolve the real decoder block list once.
    decoder_layers = _get_decoder_layers(model)

    # ---- Load SAEs for each requested layer ----
    saes_by_layer: dict[int, SAE] = {}
    for layer in target_layers:
        layer_sae_id = _replace_sae_layer(sae_id, layer)
        print(f"Loading SAE for layer {layer}: {sae_release} / {layer_sae_id}")
        try:
            sae, _, _ = SAE.from_pretrained(
                release=sae_release,
                sae_id=layer_sae_id,
                device=device,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load one of the requested SAEs. GemmaScope releases usually use '-res', "
                "'-att', or '-mlp' rather than '-resid_post', '-attn_out', or '-mlp_out'. "
                f"Received release='{sae_release}', requested layer={layer}, base_sae_id='{sae_id}', "
                f"model='{model_name}'."
            ) from exc
        sae.eval()
        saes_by_layer[int(layer)] = sae

    # ---- Hooks: capture residual streams after each requested layer ----
    _hook_store: dict[int, torch.Tensor] = {}

    def _make_resid_hook(layer: int):
        def _resid_hook(module, input, output):
            # Gemma decoder layer output is (hidden_state, ...) tuple.
            hidden = output[0] if isinstance(output, tuple) else output
            _hook_store[int(layer)] = hidden.detach().float()  # (batch, seq, hidden)
        return _resid_hook

    hook_handles = [
        decoder_layers[int(layer)].register_forward_hook(_make_resid_hook(int(layer)))
        for layer in target_layers
    ]

    try:
        for i, chunk in enumerate(chunks):
            chunk_id = chunk["chunk_id"]
            # Convert this dataset row into the exact string Gemma will read.
            # For prompted/chat datasets this includes the system/user wrapper, not just the raw text.
            model_input_text, tokenized = _render_model_input(tokenizer, chunk, max_seq_len)
            input_ids      = tokenized["input_ids"].to(model.device)
            attention_mask = tokenized["attention_mask"].to(model.device)
            seq_len        = input_ids.shape[1]
            # Save the exact tokenization used so later analyses can map feature spikes back to tokens.
            token_ids_cpu  = input_ids[0].detach().cpu()
            decoded_tokens = tokenizer.convert_ids_to_tokens(token_ids_cpu.tolist())

            with torch.no_grad():
                # The forward pass triggers all requested hooks, which capture residual streams by layer.
                model(input_ids=input_ids, attention_mask=attention_mask)

            mask = attention_mask[0].float().unsqueeze(-1).cpu()
            mean_resid_by_layer: dict[str, torch.Tensor] = {}
            sae_acts_by_layer: dict[str, torch.Tensor] = {}
            for layer in target_layers:
                resid = _hook_store[int(layer)][0]  # (seq_len, hidden_dim)

                # Mean-pool over non-padding tokens.
                mean_resid = (resid.cpu() * mask).sum(0) / mask.sum()

                with torch.no_grad():
                    # Encode token-wise residuals into the GemmaScope latent basis for this layer.
                    sae_acts = saes_by_layer[int(layer)].encode(resid.to(saes_by_layer[int(layer)].device))

                mean_resid_by_layer[str(int(layer))] = mean_resid.to(torch.bfloat16)
                sae_acts_by_layer[str(int(layer))] = sae_acts.cpu().to(torch.float32)

            results.append({
                "chunk_id":         chunk_id,
                "doc_id":           chunk.get("doc_id") or chunk.get("article_id") or "",
                "task_name":        chunk.get("task_name") or "plain_text",
                "model_input_text": model_input_text,
                "input_ids":        token_ids_cpu.to(torch.int32),
                "tokens":           decoded_tokens,
                "primary_layer":    primary_layer,
                "target_layers":    [int(layer) for layer in target_layers],
                # Keep the old single-layer keys pointing at the primary layer for backward compatibility.
                "mean_resid":       mean_resid_by_layer[str(primary_layer)],
                "sae_acts":         sae_acts_by_layer[str(primary_layer)],
                "mean_resid_by_layer": mean_resid_by_layer,
                "sae_acts_by_layer":   sae_acts_by_layer,
                "token_count":      seq_len,
            })
            done_ids.add(chunk_id)

            if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
                # Save every few chunks so a long VM/Slurm run does not lose all progress on interruption.
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(results, output_path)
                print(f"  [{i+1}/{len(chunks)}] saved -> {output_path}")

    finally:
        for hook_handle in hook_handles:
            hook_handle.remove()

    print(f"Done. {len(results)} total records -> {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect Gemma 3 12B IT residual-stream + SAE activations for text chunks."
    )
    parser.add_argument(
        "--input", type=Path, required=True, metavar="FILE",
        help="Input JSONL file of text chunks.",
    )
    parser.add_argument(
        "--output", type=Path, required=True, metavar="FILE",
        help="Output .pt file (list of dicts, resume-safe).",
    )
    parser.add_argument(
        "--model", default=MODEL_NAME, metavar="HF_REPO",
        help=f"HuggingFace model id (default: {MODEL_NAME}).",
    )
    parser.add_argument(
        "--sae-release", default=SAE_RELEASE, metavar="RELEASE",
        help=f"SAELens release name (default: {SAE_RELEASE}).",
    )
    parser.add_argument(
        "--sae-id", default=SAE_ID, metavar="SAE_ID",
        help=(
            f"SAE identifier within the release (default: {SAE_ID}). "
            "Available layers: 12, 24, 31, 41. "
            "Widths: 16k, 65k, 262k, 1m. "
            "Sparsities: small (l0≈20), medium (l0≈60), big (l0≈150)."
        ),
    )
    parser.add_argument(
        "--layers", default=None, metavar="L1,L2,...",
        help=(
            "Comma-separated model layers to collect in one pass. "
            "If omitted, the script uses the layer encoded in --sae-id. "
            f"Useful GemmaScope layers for this release include: {', '.join(str(x) for x in DEFAULT_LAYERS)}."
        ),
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=MAX_SEQ_LEN, metavar="N",
        help=f"Truncate chunks to this many tokens (default: {MAX_SEQ_LEN}).",
    )
    parser.add_argument(
        "--device", default=None, metavar="DEVICE",
        help="Compute device: cuda, mps, or cpu (default: auto-detect).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    device = args.device or _best_device()
    print(f"Device: {device}")
    target_layers = _parse_layers(args.layers, _parse_sae_layer(args.sae_id))
    print(f"Target layers: {target_layers}")

    collect(
        input_path  = args.input,
        output_path = args.output,
        model_name  = args.model,
        sae_release = args.sae_release,
        sae_id      = args.sae_id,
        target_layers = target_layers,
        max_seq_len = args.max_seq_len,
        device      = device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
