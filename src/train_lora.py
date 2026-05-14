"""Fine-tune a base model with LoRA for a specific emotional style.

Usage:
    python src/train_lora.py --style empathetic --config configs/empathetic.yaml
    python src/train_lora.py --style rational   --config configs/rational.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    set_seed,
)

from utils import (
    STYLE_SYSTEM_PROMPTS,
    save_jsonl,
    load_jsonl,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StyleTrainingArgs:
    """Style-specific training arguments."""

    style: str = field(
        default="empathetic",
        metadata={"help": "Style name: empathetic, rational, encouraging, calm_safe"},
    )
    model_name: str = field(
        default="./models/Qwen/Qwen3-1.7B",
        metadata={"help": "Base model name or path"},
    )
    data_path: str = field(
        default="data/train/empathetic.jsonl",
        metadata={"help": "Path to training data (JSONL format)"},
    )
    output_dir: str = field(
        default="outputs/lora/empathetic",
        metadata={"help": "Output directory for LoRA adapter"},
    )

    # LoRA hyperparams
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "Target modules for LoRA"},
    )


def load_and_format_data(
    data_path: str,
    tokenizer,
    style: str,
    max_length: int = 1024,
) -> Dataset:
    """Load training data and format with style-specific system prompt via Qwen3 chat template."""
    raw = load_jsonl(data_path)
    system_prompt = STYLE_SYSTEM_PROMPTS.get(style, STYLE_SYSTEM_PROMPTS["calm_safe"])

    formatted = []
    for item in raw:
        user_msg = item.get("user", item.get("input", item.get("instruction", "")))
        assistant_msg = item.get("assistant", item.get("output", item.get("response", ""))

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        formatted.append({"text": text})

    dataset = Dataset.from_list(formatted)

    def tokenize_fn(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        result["labels"] = [ids.copy() for ids in result["input_ids"]]
        return result

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    return dataset


def train(args: StyleTrainingArgs, training_args: TrainingArguments):
    """Run LoRA fine-tuning for a single style."""

    set_seed(training_args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading base model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if training_args.fp16 else torch.float32,
        trust_remote_code=True,
        device_map="auto",
    )

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    # Load data
    logger.info(f"Loading data from: {args.data_path}")
    dataset = load_and_format_data(args.data_path, tokenizer, args.style, training_args.max_seq_length or 1024)

    # Only use training data (no eval split for simplicity)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    logger.info(f"Training LoRA for style: {args.style}")
    trainer.train()

    # Save adapter
    logger.info(f"Saving LoRA adapter to: {output_dir}")
    peft_model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save training config
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(
            {
                "style": args.style,
                "model_name": args.model_name,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("Training complete.")


def main():
    parser = HfArgumentParser((StyleTrainingArgs, TrainingArguments))
    style_args, training_args = parser.parse_args_into_dataclasses()
    train(style_args, training_args)


if __name__ == "__main__":
    main()
