#!/usr/bin/env python3
"""Validate the core idea: LoRA interpolation for emotion-style controllable generation.

Tests:
  1. Pure styles (weight=1.0) produce distinct, style-specific responses
  2. Interpolated weights produce blended outputs
  3. Continuous weight sweep shows smooth style transition
"""

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from interpolate import InterpolatableLoRA
from generate import generate_with_interpolation

MODEL_PATH = "./models/Qwen/Qwen3-1.7B"
LORA_DIR = Path("outputs/lora")
LOG_DIR = Path("log")
LOG_DIR.mkdir(exist_ok=True)

TEST_INPUTS = [
    "最近工作压力很大，感觉快撑不住了",
    "和朋友闹矛盾了，心里好难受",
    "对未来感到很迷茫，不知道该选什么",
]


def load_model_and_loras():
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16, trust_remote_code=True, device_map="auto"
    )

    lora_paths = {}
    for style in ["empathetic", "rational", "encouraging", "calm_safe"]:
        p = LORA_DIR / style
        if (p / "adapter_model.safetensors").exists():
            lora_paths[style] = str(p)
        else:
            print(f"  WARNING: {style} adapter not found at {p}")

    if not lora_paths:
        raise RuntimeError("No LoRA adapters found!")
    print(f"Loaded {len(lora_paths)} adapters: {list(lora_paths.keys())}")
    return model, tokenizer, lora_paths


def run_validation():
    model, tokenizer, lora_paths = load_model_and_loras()

    wrapper = InterpolatableLoRA(
        base_model=model,
        lora_paths=lora_paths,
        interpolation_mode="weight",
    )

    results = []
    all_log = []

    # ── Test 1: Pure styles ──
    print("\n" + "=" * 60)
    print("TEST 1: Pure style responses (single weight = 1.0)")
    print("=" * 60)

    for user_input in TEST_INPUTS:
        print(f"\n>>> 用户: {user_input}")
        for style in ["empathetic", "rational", "encouraging", "calm_safe"]:
            weights = {s: 0.0 for s in lora_paths}
            weights[style] = 1.0

            t0 = time.time()
            response, info = generate_with_interpolation(
                wrapper, tokenizer, user_input, weights,
                max_new_tokens=128, temperature=0.7, top_p=0.8, top_k=20
            )
            elapsed = time.time() - t0

            zh_name = {"empathetic": "共情", "rational": "理性",
                       "encouraging": "鼓励", "calm_safe": "安全"}.get(style, style)
            print(f"  [{zh_name}] ({elapsed:.1f}s) {response[:120]}...")

            results.append({
                "test": "pure_style",
                "input": user_input,
                "style": style,
                "response": response,
                "time_s": round(elapsed, 2),
            })
            all_log.append(f"[pure] {style:15s} → {response[:100]}")

    # ── Test 2: Interpolation ──
    print("\n" + "=" * 60)
    print("TEST 2: Weight interpolation (blended styles)")
    print("=" * 60)

    interp_configs = [
        {"empathetic": 0.7, "rational": 0.0, "encouraging": 0.3, "calm_safe": 0.0},
        {"empathetic": 0.3, "rational": 0.7, "encouraging": 0.0, "calm_safe": 0.0},
        {"empathetic": 0.0, "rational": 0.5, "encouraging": 0.5, "calm_safe": 0.0},
        {"empathetic": 0.3, "rational": 0.2, "encouraging": 0.3, "calm_safe": 0.2},  # balanced
        {"empathetic": 0.0, "rational": 0.0, "encouraging": 0.0, "calm_safe": 1.0},  # safety
    ]

    for user_input in TEST_INPUTS:
        print(f"\n>>> 用户: {user_input}")
        for w in interp_configs:
            label = "+".join(f"{k[:3]}:{v}" for k, v in w.items() if v > 0)
            t0 = time.time()
            response, info = generate_with_interpolation(
                wrapper, tokenizer, user_input, w,
                max_new_tokens=128, temperature=0.7, top_p=0.8, top_k=20
            )
            elapsed = time.time() - t0

            print(f"  [{label}] ({elapsed:.1f}s) {response[:120]}...")
            results.append({
                "test": "interpolation",
                "input": user_input,
                "weights": w,
                "response": response,
                "time_s": round(elapsed, 2),
            })
            all_log.append(f"[interp] {label} → {response[:100]}")

    # ── Test 3: Continuous sweep ──
    print("\n" + "=" * 60)
    print("TEST 3: Continuous empathy → rationality sweep")
    print("=" * 60)

    user_input = TEST_INPUTS[0]
    print(f"\n>>> 用户: {user_input}")
    print(f"{'Empathy%':>8s} {'Rational%':>9s} | Response preview")

    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        w = {
            "empathetic": 1.0 - alpha,
            "rational": alpha,
            "encouraging": 0.0,
            "calm_safe": 0.0,
        }
        response, _ = generate_with_interpolation(
            wrapper, tokenizer, user_input, w,
            max_new_tokens=128, temperature=0.7, top_p=0.8, top_k=20
        )
        e_pct = (1 - alpha) * 100
        r_pct = alpha * 100
        print(f"  {e_pct:6.0f}% {r_pct:7.0f}%  | {response[:100]}...")
        all_log.append(f"[sweep] E{e_pct:.0f}_R{r_pct:.0f} → {response[:100]}")

    # Save results
    output_path = LOG_DIR / "validation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nFull results saved to {output_path}")

    # Write readable log
    log_path = LOG_DIR / "validation_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_log))
    print(f"Readable log saved to {log_path}")

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_validation()
