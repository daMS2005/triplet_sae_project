#!/usr/bin/env python3
"""Fine-tune Gemma for triplet extraction with LoRA or QLoRA."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = "google/gemma-3-4b-it"
IGNORE_INDEX = -100


class ChatSFTDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int) -> None:
        self.rows = []
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        if not self.rows:
            raise ValueError(f"No rows found in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        row = self.rows[idx]
        messages = row["messages"]
        prompt_messages = messages[:-1]
        assistant_text = str(messages[-1]["content"]).strip()

        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = prompt_text + assistant_text + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        full = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = full["input_ids"]
        attention_mask = full["attention_mask"]

        labels = input_ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [IGNORE_INDEX] * prompt_len
        if all(label == IGNORE_INDEX for label in labels):
            # If a very long prompt consumed the whole window, keep the final token
            # trainable instead of producing an all-ignored example.
            labels[-1] = input_ids[-1]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def _load_model(args: argparse.Namespace):
    quantization_config = None
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def _maybe_apply_lora(model, args: argparse.Namespace):
    if not args.use_lora:
        return model
    # Some environments used for inference/export also have GPTQ/AWQ packages
    # installed. PEFT will try to route LoRA injection through those quantized
    # adapters if it detects them, which breaks normal dense-model continuation
    # training. We explicitly disable that detection here so continuation runs
    # use the standard dense LoRA path.
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    import peft.import_utils as peft_import_utils
    import peft.tuners.lora.awq as peft_lora_awq
    import peft.tuners.lora.model as peft_lora_model

    peft_import_utils.is_gptqmodel_available = lambda: False
    peft_lora_awq.dispatch_awq = lambda *args, **kwargs: None
    peft_lora_model.dispatch_awq = peft_lora_awq.dispatch_awq

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    if args.init_adapter is not None:
        model = PeftModel.from_pretrained(model, str(args.init_adapter), is_trainable=True)
        model.print_trainable_parameters()
        return model

    target_modules = [part.strip() for part in args.lora_target_modules.split(",") if part.strip()]
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA/QLoRA fine-tuning for triplet extraction.")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument(
        "--init-adapter",
        type=Path,
        default=None,
        help="Optional existing LoRA adapter directory to continue fine-tuning from.",
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    # Default to standard LoRA unless 4-bit is explicitly requested. This keeps
    # the training path usable on environments where bitsandbytes is unavailable.
    parser.add_argument("--use-4bit", action="store_true", default=False)
    parser.add_argument("--no-4bit", action="store_false", dest="use_4bit")
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--no-lora", action="store_false", dest="use_lora")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    return parser.parse_args()


def _find_last_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.split("-")[-1])
        except ValueError:
            continue
        checkpoints.append((step, path))
    if not checkpoints:
        return None
    checkpoints.sort()
    return checkpoints[-1][1]


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    train_dataset = ChatSFTDataset(args.train_file, tokenizer, args.max_length)
    eval_dataset = ChatSFTDataset(args.validation_file, tokenizer, args.max_length)

    model = _load_model(args)
    model = _maybe_apply_lora(model, args)

    training_kwargs = dict(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=False,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    signature = inspect.signature(TrainingArguments)
    if "eval_strategy" in signature.parameters:
        training_kwargs["eval_strategy"] = "steps"
    else:
        training_kwargs["evaluation_strategy"] = "steps"
    training_kwargs["save_strategy"] = "steps"
    training_args = TrainingArguments(**training_kwargs)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=IGNORE_INDEX,
        pad_to_multiple_of=8,
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer_signature = inspect.signature(Trainer)
    if "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint = _find_last_checkpoint(args.output_dir)
    if last_checkpoint is not None:
        print(f"Resuming from checkpoint: {last_checkpoint}")
    else:
        print("Starting training from scratch.")

    trainer.train(resume_from_checkpoint=str(last_checkpoint) if last_checkpoint is not None else None)
    trainer.save_model(args.output_dir / "final_adapter")
    tokenizer.save_pretrained(args.output_dir / "final_adapter")

    metrics = trainer.evaluate()
    with (args.output_dir / "final_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
