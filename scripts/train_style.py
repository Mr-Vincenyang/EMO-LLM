#!/usr/bin/env python3
"""Train a single style LoRA on a specific GPU.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_style.py --style empathetic
    CUDA_VISIBLE_DEVICES=1 python scripts/train_style.py --style rational
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import STYLE_SYSTEM_PROMPTS, load_jsonl

LOG_DIR = Path("log")
MODEL_PATH = "./models/Qwen/Qwen3-1.7B"
OUTPUT_DIR = Path("outputs/lora_v2")


def train_style(
    style: str = "empathetic",
    data_dir: str = "data/train",
    num_epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 4,
):
    gpu_id = torch.cuda.current_device() if torch.cuda.is_available() else -1
    run_id = f"v2_{style}_{datetime.now():%H%M%S}"
    log_file = LOG_DIR / f"train_{run_id}.log"
    data_path = Path(data_dir) / f"{style}.jsonl"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(style)
    logger.info(f"Style: {style} | GPU: {gpu_id} | Log: {log_file}")

    output = OUTPUT_DIR / style
    output.mkdir(parents=True, exist_ok=True)

    raw = load_jsonl(str(data_path))
    logger.info(f"Samples: {len(raw)}")

    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16, trust_remote_code=True, device_map="auto"
    )
    gpu_mem_model = torch.cuda.max_memory_allocated(gpu_id) / 1e9
    logger.info(f"Model loaded. GPU mem: {gpu_mem_model:.1f}GB")

    system_prompt = STYLE_SYSTEM_PROMPTS[style]
    formatted = []
    for item in raw:
        user_msg = item.get("user", "")
        assistant_msg = item.get("assistant", "")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        formatted.append({"text": text})

    dataset = Dataset.from_list(formatted)

    def tokenize_fn(examples):
        r = tokenizer(examples["text"], truncation=True, max_length=1024, padding=False)
        r["labels"] = [ids.copy() for ids in r["input_ids"]]
        return r

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {trainable:,} ({100*trainable/sum(p.numel() for p in peft_model.parameters()):.2f}%)")

    training_args = TrainingArguments(
        output_dir=f"outputs/tmp_{style}",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_epochs,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        fp16=True,
        report_to="none",
        seed=42,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=peft_model, padding=True, label_pad_token_id=-100
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    gpu_mem_before = torch.cuda.memory_allocated(gpu_id) / 1e9
    t0 = time.time()

    logger.info("Starting training...")
    result = trainer.train()

    train_time = time.time() - t0
    gpu_mem_peak = torch.cuda.max_memory_allocated(gpu_id) / 1e9

    peft_model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))

    metrics = {
        "style": style,
        "gpu": gpu_id,
        "samples": len(raw),
        "trainable_params": trainable,
        "train_time_s": round(train_time, 1),
        "gpu_mem_model_gb": round(gpu_mem_model, 2),
        "gpu_mem_before_train_gb": round(gpu_mem_before, 2),
        "gpu_mem_peak_gb": round(gpu_mem_peak, 2),
        "train_loss": round(result.training_loss, 4),
        "epochs": num_epochs,
        "batch_size": batch_size * grad_accum,
        "timestamp": datetime.now().isoformat(),
    }

    # Write metrics
    metrics_path = LOG_DIR / f"metrics_{run_id}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    logger.info(f"Training complete. Time: {train_time:.0f}s, Loss: {result.training_loss:.4f}")
    logger.info(f"Peak GPU: {gpu_mem_peak:.1f}GB | Adapter saved: {output}")
    logger.info(f"Metrics: {metrics_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", required=True, choices=["empathetic", "rational", "encouraging", "calm_safe"])
    parser.add_argument("--data_dir", default="data/train", help="Directory with style JSONL files")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    args = parser.parse_args()
    train_style(args.style, args.data_dir, args.epochs, args.batch_size, args.grad_accum)


if __name__ == "__main__":
    main()
