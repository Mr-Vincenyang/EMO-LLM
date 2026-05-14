#!/usr/bin/env python3
"""Run the full experiment pipeline with logging.

1. Train 4 style LoRAs
2. Log metrics (loss, time, GPU memory, params)
3. Generate comparison outputs with different weight combinations
"""

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
DATA_DIR = Path("data/train")
OUTPUT_DIR = Path("outputs/lora")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"experiment_{datetime.now():%Y%m%d_%H%M%S}.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def log_metrics(metrics: dict, step: str):
    path = LOG_DIR / "metrics.jsonl"
    record = {"timestamp": datetime.now().isoformat(), "step": step, **metrics}
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"Metrics [{step}]: {json.dumps(metrics, ensure_ascii=False)}")


def train_one_style(style: str, model, tokenizer, training_args: TrainingArguments) -> dict:
    """Train a single style LoRA. Returns training metrics."""
    data_path = DATA_DIR / f"{style}.jsonl"
    output = OUTPUT_DIR / style
    output.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training style: {style}")
    logger.info(f"  Data: {data_path} ({data_path.stat().st_size / 1024:.0f}KB)")
    logger.info(f"  Output: {output}")

    system_prompt = STYLE_SYSTEM_PROMPTS[style]
    raw = load_jsonl(str(data_path))
    logger.info(f"  Samples: {len(raw)}")

    if len(raw) == 0:
        logger.warning(f"  No data for {style}, skipping")
        return {"style": style, "samples": 0, "trainable_params": 0, "train_time_s": 0}

    # Format data with Qwen3 chat template
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
    total = sum(p.numel() for p in peft_model.parameters())
    logger.info(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    gpu_mem_before = torch.cuda.memory_allocated(0) / 1e9

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=peft_model, padding=True, label_pad_token_id=-100)

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    t0 = time.time()
    result = trainer.train()
    train_time = time.time() - t0

    gpu_mem_after = torch.cuda.max_memory_allocated(0) / 1e9

    peft_model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))

    # Remove LoRA layers without merging, returning clean base model
    base_model = peft_model.unload()

    # Collect metrics
    metrics = {
        "style": style,
        "samples": len(raw),
        "trainable_params": trainable,
        "train_time_s": round(train_time, 1),
        "gpu_mem_before_gb": round(gpu_mem_before, 2),
        "gpu_mem_peak_gb": round(gpu_mem_after, 2),
        "train_loss": round(result.training_loss, 4),
        "epochs": training_args.num_train_epochs,
    }
    log_metrics(metrics, f"train_{style}")
    return metrics, base_model


def main():
    logger.info("=" * 60)
    logger.info("EMO-LLM Experiment Pipeline")
    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"Start: {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info("=" * 60)

    # GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU: {gpu_name} ({gpu_mem:.0f}GB)")

    set_seed(42)

    # Load model once
    logger.info("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16, trust_remote_code=True, device_map="auto"
    )
    logger.info("Model loaded.")

    training_args = TrainingArguments(
        output_dir="outputs/tmp",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        fp16=True,
        report_to="none",
        seed=42,
    )

    styles = ["empathetic", "rational", "encouraging", "calm_safe"]
    results = {}

    for style in styles:
        m, model = train_one_style(style, model, tokenizer, training_args)
        results[style] = m

    # Final summary
    logger.info("=" * 60)
    logger.info("Training Complete - Summary")
    logger.info("=" * 60)
    for style, m in results.items():
        logger.info(
            f"  {style}: {m.get('samples', 0)} samples, "
            f"loss={m.get('train_loss', 'N/A'):.4f}, "
            f"time={m.get('train_time_s', 'N/A')}s"
        )

    log_metrics({"all_styles_complete": True, "results": results}, "complete")
    logger.info(f"Logs saved to: {LOG_DIR.absolute()}")


if __name__ == "__main__":
    main()
