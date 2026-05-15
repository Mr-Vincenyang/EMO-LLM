#!/usr/bin/env python3
"""v3: 4 LoRAs on SAME ESConv data, each with DIFFERENT contrastive prompt.

Question: can LoRA learn style-specific behavior even with identical training targets,
if the system prompts are contrastive enough?

This avoids v1's problem (data split by strategy → homogeneous subsets)
and v2's problem (self-distillation from own generation).

If LoRA+Prompt > Prompt-only in style separation → LoRA adds value beyond prompt.
"""

import json, subprocess, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import STYLE_SYSTEM_PROMPTS, QWEN3_NON_THINKING_GENERATION

MODEL_PATH = "./models/Qwen/Qwen3-1.7B"
LOG_DIR = Path("log")
LOG_DIR.mkdir(exist_ok=True)

TEST_INPUTS = [
    "I'm feeling really stressed at work and don't know how to cope.",
    "My friend and I had a big argument, feeling terrible.",
    "I feel completely lost about my future career.",
    "I've been feeling anxious and can't sleep well at night.",
    "My parents keep pressuring me about my life choices.",
]


def train_all():
    """Launch 4 parallel LoRA trainings."""
    print("[1/3] Training 4 LoRAs in parallel...")
    styles = ["empathetic", "rational", "encouraging", "calm_safe"]
    gpus = [0, 0, 1, 1]

    procs = []
    for style, gpu in zip(styles, gpus):
        env = {**__import__("os").environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        p = subprocess.Popen(
            ["python", "scripts/train_v3_worker.py", style, "0"],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        procs.append((style, p))

    for style, p in procs:
        out, _ = p.communicate(timeout=600)
        lines = out.decode().split("\n")
        done = [l for l in lines if "DONE:" in l]
        print(f"  {style}: {done[0] if done else 'FAILED'}")
        if not done:
            print(out.decode()[-300:])

    # Verify
    for style in styles:
        if not (Path(f"outputs/lora_v3/{style}") / "adapter_model.safetensors").exists():
            raise RuntimeError(f"Training failed for {style}")
    print("  All 4 trained successfully.")


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


def compute_metrics(all_gen):
    """Compute style separation + distinct-N for a set of generations."""
    styles = ["empathetic", "rational", "encouraging", "calm_safe"]
    per_style = {s: all_gen[s] for s in styles}
    all_texts = []
    for s in styles:
        all_texts.extend(per_style[s])

    # Style separation
    intra_vals, cross_vals = [], []
    for s in styles:
        samples = per_style[s]
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
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

    intra_mean = float(np.mean(intra_vals))
    cross_mean = float(np.mean(cross_vals))

    # Distinct-N
    from collections import Counter
    uniq_2, total_2 = Counter(), 0
    uniq_3, total_3 = Counter(), 0
    for text in all_texts:
        words = text.lower().split()
        uniq_2.update(tuple(words[i:i+2]) for i in range(len(words)-1))
        total_2 += max(len(words)-1, 0)
        uniq_3.update(tuple(words[i:i+3]) for i in range(len(words)-2))
        total_3 += max(len(words)-2, 0)
    d2 = len(uniq_2) / total_2 if total_2 > 0 else 0
    d3 = len(uniq_3) / total_3 if total_3 > 0 else 0

    return {
        "style_separation": round(intra_mean - cross_mean, 4),
        "intra_overlap": round(intra_mean, 4),
        "cross_overlap": round(cross_mean, 4),
        "distinct_2": round(d2, 4),
        "distinct_3": round(d3, 4),
    }


def evaluate():
    print("[2/3] Loading models...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16, trust_remote_code=True, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    styles = ["empathetic", "rational", "encouraging", "calm_safe"]
    lora_models = {}
    for style in styles:
        lora_models[style] = PeftModel.from_pretrained(
            base_model, f"outputs/lora_v3/{style}"
        )

    # ── Generate ──
    print("[3/3] Generating and evaluating...")
    configs = {
        "Prompt-only": (base_model, lambda style: STYLE_SYSTEM_PROMPTS[style]),
        "LoRA+Prompt": (lora_models, lambda style: STYLE_SYSTEM_PROMPTS[style]),
    }

    results = {}
    for config_name, (model_src, prompt_fn) in configs.items():
        print(f"\n  [{config_name}]")
        per_style = defaultdict(list)
        for user_input in TEST_INPUTS:
            for style in styles:
                if isinstance(model_src, dict):
                    model = model_src[style]
                else:
                    model = model_src
                resp = generate(model, tokenizer, prompt_fn(style), user_input)
                per_style[style].append(resp)
                if style == styles[0]:
                    print(f"    {resp[:80]}...")
        results[config_name] = compute_metrics(per_style)
        per_style_dict = dict(per_style)
        results[config_name]["_raw"] = {s: per_style_dict[s] for s in styles}

    # ── Print comparison ──
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    for name, m in results.items():
        print(f"  {name}:")
        print(f"    separation={m['style_separation']}, D2={m['distinct_2']}, D3={m['distinct_3']}")

    sep_prompt = results["Prompt-only"]["style_separation"]
    sep_lora = results["LoRA+Prompt"]["style_separation"]
    delta = round(sep_lora - sep_prompt, 4)
    print(f"\n  LoRA contribution to style separation: {delta:+.4f}")
    print(f"  {'✅ LoRA adds value beyond prompt' if delta > 0.005 else '≈ No significant improvement' if abs(delta) < 0.005 else '❌ LoRA degrades style control'}")

    # ── Save ──
    out = {"config": {"data": "ESConv ALL samples, 1000 per style", "prompts": "contrastive DO/DON'T", "lora_rank": 16, "no_self_distillation": True, "no_generated_data": True}, "results": results}
    with open(LOG_DIR / "experiment_v3_results.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Results: log/experiment_v3_results.json")

    return results


def main():
    print("=" * 60)
    print("  v3: 4 LoRAs, SAME authentic ESConv data")
    print("  DIFFERENT contrastive prompts per style")
    print("  No self-distillation. No generated data.")
    print("  Key question: can LoRA add value beyond prompt?")
    print("=" * 60)
    train_all()
    evaluate()
    print("\nDone.")


if __name__ == "__main__":
    main()
