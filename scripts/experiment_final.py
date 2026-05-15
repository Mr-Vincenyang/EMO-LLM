#!/usr/bin/env python3
"""Final experiment: LoRA for quality, Prompt for style.

Design: train ONE LoRA on ALL ESConv data with neutral prompt.
Then 2×2 ablation: ±LoRA × ±Style Prompt.

Metrics:
  - Style separation: can the system produce distinct styles?
  - Quality: response length, lexical diversity, PPL
  - Combined score: separation × quality

The claim: LoRA + Style Prompt achieves both high quality AND style control,
outperforming either component alone.
"""

import json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
    Trainer, DataCollatorForSeq2Seq, set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import STYLE_SYSTEM_PROMPTS, load_jsonl, QWEN3_NON_THINKING_GENERATION

MODEL_PATH = "./models/Qwen/Qwen3-1.7B"
LOG_DIR = Path("log")
LOG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("outputs/lora_final")
OUTPUT_DIR.mkdir(exist_ok=True)

TEST_INPUTS = [
    "I'm feeling really stressed at work and don't know how to cope.",
    "My friend and I had a big argument, feeling terrible.",
    "I feel completely lost about my future career.",
    "I've been feeling anxious and can't sleep well at night.",
    "My parents keep pressuring me about my life choices.",
]

NEUTRAL_PROMPT = "You are a supportive emotional support assistant. Respond with empathy, warmth, and practical help."


def train_quality_lora():
    """Train ONE LoRA on ALL ESConv data for emotional support quality."""
    print("[1/3] Training quality LoRA on all ESConv data...")

    all_data = []
    for sf in Path("data/train").glob("*.jsonl"):
        if sf.stem == "all_styles":
            continue
        all_data.extend(load_jsonl(str(sf)))
    print(f"  Samples: {len(all_data)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    formatted = []
    for item in all_data:
        messages = [
            {"role": "system", "content": NEUTRAL_PROMPT},
            {"role": "user", "content": item["user"]},
            {"role": "assistant", "content": item["assistant"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        formatted.append({"text": text})

    dataset = Dataset.from_list(formatted)

    def tok_fn(ex):
        r = tokenizer(ex["text"], truncation=True, max_length=1024, padding=False)
        r["labels"] = [ids.copy() for ids in r["input_ids"]]
        return r

    dataset = dataset.map(tok_fn, batched=True, remove_columns=dataset.column_names)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16, trust_remote_code=True, device_map="auto"
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable:,}")

    training_args = TrainingArguments(
        output_dir="outputs/tmp_final",
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        num_train_epochs=2, learning_rate=2e-4,
        warmup_ratio=0.1, lr_scheduler_type="cosine",
        logging_steps=20, save_strategy="no",
        fp16=True, report_to="none", seed=42,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=peft_model, padding=True, label_pad_token_id=-100)
    trainer = Trainer(model=peft_model, args=training_args, train_dataset=dataset, data_collator=data_collator)

    t0 = time.time()
    result = trainer.train()
    train_time = time.time() - t0

    peft_model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    metrics = {"loss": round(result.training_loss, 4), "time_s": round(train_time, 1), "trainable": trainable, "samples": len(all_data)}
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Done: loss={metrics['loss']}, time={train_time:.0f}s")
    return metrics


def generate(model, tokenizer, system_prompt, user_input):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    defaults = QWEN3_NON_THINKING_GENERATION
    output = model.generate(
        **inputs, max_new_tokens=128,
        temperature=defaults["temperature"], top_p=defaults["top_p"],
        top_k=defaults["top_k"], do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def compute_quality_metrics(texts):
    """Response quality: avg length, lexical diversity (TTR), distinct-N."""
    lengths = []
    ttrs = []
    for text in texts:
        words = text.split()
        lengths.append(len(words))
        ttrs.append(len(set(words)) / len(words) if words else 0)

    from collections import Counter
    uniq_2, total_2 = Counter(), 0
    for text in texts:
        words = text.lower().split()
        uniq_2.update(tuple(words[i:i+2]) for i in range(len(words)-1))
        total_2 += max(len(words)-1, 0)
    d2 = len(uniq_2) / total_2 if total_2 > 0 else 0

    return {
        "avg_length": round(float(np.mean(lengths)), 1),
        "avg_ttr": round(float(np.mean(ttrs)), 4),
        "distinct_2": round(d2, 4),
    }


def compute_style_separation(per_style):
    """N-gram overlap: intra vs cross."""
    styles = ["empathetic", "rational", "encouraging", "calm_safe"]
    intra_vals, cross_vals = [], []
    for s in styles:
        samples = per_style[s]
        for i in range(len(samples)):
            for j in range(i+1, len(samples)):
                wa, wb = set(samples[i].lower().split()), set(samples[j].lower().split())
                intra_vals.append(len(wa & wb) / len(wa | wb) if wa | wb else 0)
        other = []
        for os_ in styles:
            if os_ != s:
                other.extend(per_style[os_])
        for hyp in samples:
            for ref in other[:len(samples)]:
                wa, wb = set(hyp.lower().split()), set(ref.lower().split())
                cross_vals.append(len(wa & wb) / len(wa | wb) if wa | wb else 0)
    return {
        "intra_overlap": round(float(np.mean(intra_vals)), 4),
        "cross_overlap": round(float(np.mean(cross_vals)), 4),
        "separation": round(float(np.mean(intra_vals) - np.mean(cross_vals)), 4),
    }


def run_ablation():
    print("\n[2/3] Loading models...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16, trust_remote_code=True, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    lora_model = PeftModel.from_pretrained(base_model, str(OUTPUT_DIR))

    styles = ["empathetic", "rational", "encouraging", "calm_safe"]

    print("\n[3/3] Running 2×2 ablation: ±LoRA × ±Style Prompt\n")

    configs = [
        ("Base (no LoRA, no style)", base_model, lambda s: NEUTRAL_PROMPT),
        ("Base + Style Prompt", base_model, lambda s: STYLE_SYSTEM_PROMPTS[s]),
        ("LoRA (no style prompt)", lora_model, lambda s: NEUTRAL_PROMPT),
        ("LoRA + Style Prompt", lora_model, lambda s: STYLE_SYSTEM_PROMPTS[s]),
    ]

    all_results = {}
    for config_name, model, prompt_fn in configs:
        per_style = {s: [] for s in styles}
        all_texts = []
        for user_input in TEST_INPUTS:
            for style in styles:
                resp = generate(model, tokenizer, prompt_fn(style), user_input)
                per_style[style].append(resp)
                all_texts.append(resp)

        quality = compute_quality_metrics(all_texts)
        style = compute_style_separation(per_style)
        all_results[config_name] = {**quality, **style}

    # ── Print Results Table ──
    print("=" * 90)
    print(f"  {'Config':<30s} {'Separation':>7s} {'Length':>7s} {'TTR':>7s} {'Dist-2':>7s}")
    print("  " + "-" * 60)
    for name, m in all_results.items():
        print(f"  {name:<30s} {m['separation']:>+7.4f} {m['avg_length']:>7.1f} {m['avg_ttr']:>7.4f} {m['distinct_2']:>7.4f}")

    # ── Qualitative sample ──
    print("\n" + "=" * 90)
    print("  QUALITATIVE SAMPLE (input: 'feeling stressed at work')")
    print("=" * 90)
    for config_name, model, prompt_fn in configs:
        resp = generate(model, tokenizer, prompt_fn("rational"), TEST_INPUTS[0])
        print(f"  [{config_name[:20]:20s}] {resp[:100]}...")

    # ── Analysis ──
    base = all_results["Base (no LoRA, no style)"]
    prompt = all_results["Base + Style Prompt"]
    lora = all_results["LoRA (no style prompt)"]
    combined = all_results["LoRA + Style Prompt"]

    print("\n" + "=" * 90)
    print("  ANALYSIS")
    print("=" * 90)
    print(f"  Prompt effect on style:    {prompt['separation'] - base['separation']:+.4f} separation")
    print(f"  LoRA effect on quality:    {lora['avg_length'] - base['avg_length']:+.1f} words longer")
    print(f"  Combined (LoRA+Style):     sep={combined['separation']:+.4f}, len={combined['avg_length']:.1f}, TTR={combined['avg_ttr']:.4f}")
    print(f"  Combined vs Prompt-only:   {'✅ better' if combined['separation'] >= prompt['separation'] and combined['avg_length'] > prompt['avg_length'] else 'Trade-off: check details'}")

    # Save
    out = {"config": {"data": "ALL ESConv (10K), neutral prompt", "lora": "single LoRA for quality, 17M params", "no_self_distillation": True}, "results": all_results}
    with open(LOG_DIR / "experiment_final_results.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Results: log/experiment_final_results.json")

    return all_results


def main():
    print("=" * 60)
    print("  FINAL EXPERIMENT")
    print("  LoRA → Quality Enhancement | Prompt → Style Control")
    print("  No self-distillation. Authentic ESConv data.")
    print("=" * 60)
    set_seed(42)
    train_quality_lora()
    run_ablation()
    print("\nDone.")


if __name__ == "__main__":
    main()
