#!/usr/bin/env python3
"""Validate v2 LoRA interpolation and compare with v1."""

import json, sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from interpolate import InterpolatableLoRA
from generate import generate_with_interpolation

MODEL_PATH = "./models/Qwen/Qwen3-1.7B"

TEST_INPUTS = [
    "I'm feeling really stressed at work and don't know how to cope.",
    "My friend and I had a big argument, feeling terrible.",
    "I feel completely lost about my future career.",
]


def run_validation(lora_dir: str, label: str):
    print(f"\n{'='*70}")
    print(f"  {label}: {lora_dir}")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16, trust_remote_code=True, device_map="auto"
    )

    lora_paths = {s: str(Path(lora_dir) / s) for s in ["empathetic", "rational", "encouraging", "calm_safe"]}
    wrapper = InterpolatableLoRA(model, lora_paths, interpolation_mode="weight")

    results = []
    for user_input in TEST_INPUTS:
        print(f"\n  User: {user_input}")
        for style in ["empathetic", "rational", "encouraging", "calm_safe"]:
            w = {s: 0.0 for s in lora_paths}
            w[style] = 1.0
            resp, _ = generate_with_interpolation(
                wrapper, tokenizer, user_input, w,
                max_new_tokens=128, temperature=0.7, top_p=0.8, top_k=20
            )
            zh = {"empathetic": "共情", "rational": "理性", "encouraging": "鼓励", "calm_safe": "安全"}[style]
            print(f"    [{zh}] {resp[:150]}")
            results.append({"label": label, "input": user_input, "style": style, "response": resp})
            print()

    return results


def main():
    print("Loading model...")
    v1_results = run_validation("outputs/lora", "v1 (ESConv strategy mapping)")
    v2_results = run_validation("outputs/lora_v2", "v2 (contrastive prompts)")

    all_results = {"v1": v1_results, "v2": v2_results}
    with open("log/validate_v2_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to log/validate_v2_comparison.json")

    # Quick comparison
    print("\n" + "=" * 70)
    print("  SIDE-BY-SIDE: Same input, same style — v1 vs v2")
    print("=" * 70)
    for i, inp in enumerate(TEST_INPUTS):
        print(f"\n  User: {inp}")
        for style in ["empathetic", "rational", "encouraging", "calm_safe"]:
            v1_resp = next(r["response"] for r in v1_results if r["input"] == inp and r["style"] == style)
            v2_resp = next(r["response"] for r in v2_results if r["input"] == inp and r["style"] == style)
            zh = {"empathetic": "共情", "rational": "理性", "encouraging": "鼓励", "calm_safe": "安全"}[style]
            print(f"    [{zh}]")
            print(f"      v1: {v1_resp[:120]}...")
            print(f"      v2: {v2_resp[:120]}...")


if __name__ == "__main__":
    main()
