#!/usr/bin/env python3
"""Head-to-head comparison: Base Model (prompt-only) vs LoRA-interpolated Model."""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from interpolate import InterpolatableLoRA
from generate import generate_with_interpolation
from utils import STYLE_SYSTEM_PROMPTS

MODEL_PATH = "./models/Qwen/Qwen3-1.7B"
LORA_DIR = Path("outputs/lora")

INPUTS = [
    "I'm feeling really stressed at work and don't know how to cope.",
    "My friend and I had a big argument, feeling terrible.",
    "I feel completely lost about my future career.",
]

SYSTEM_PROMPTS = {
    "共情": "You are a warm, empathetic listener. Respond with gentleness and compassion. Prioritize understanding and validating the user's emotions, providing emotional companionship.",
    "理性": "You are a rational, analytical advisor. Objectively analyze the user's situation. Break down causes clearly and provide structured advice and solutions.",
    "鼓励": "You are a positive, energetic motivator. Respond with enthusiasm and affirmation. Highlight the user's strengths and potential, encourage action and build confidence.",
    "安全": "You are a calm, professional psychological safety officer. Respond with steadiness and security. Avoid excessive emotional expression. Focus on safety and potential risks.",
}


def baseline_generate(tokenizer, model, sys_prompt, user_input):
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_input},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs, max_new_tokens=128, temperature=0.7, top_p=0.8, top_k=20, do_sample=True
    )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading models...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16, trust_remote_code=True, device_map="auto"
    )

    lora_paths = {s: str(LORA_DIR / s) for s in ["empathetic", "rational", "encouraging", "calm_safe"]}
    wrapper = InterpolatableLoRA(base_model, lora_paths, interpolation_mode="weight")

    style_map = {"共情": "empathetic", "理性": "rational", "鼓励": "encouraging", "安全": "calm_safe"}

    results = {}
    for user_input in INPUTS:
        print(f"\n{'='*70}")
        print(f"User: {user_input}")
        print(f"{'='*70}")
        print(f"{'':>6s} {'BASELINE (prompt only)':<50s} | {'LORA (trained adapter)':<50s}")
        print(f"{'':>6s} {'─'*50} | {'─'*50}")

        for zh_style, en_style in style_map.items():
            # Baseline
            b_resp = baseline_generate(tokenizer, base_model, SYSTEM_PROMPTS[zh_style], user_input)

            # LoRA
            w = {s: 0.0 for s in ["empathetic", "rational", "encouraging", "calm_safe"]}
            w[en_style] = 1.0
            l_resp, _ = generate_with_interpolation(
                wrapper, tokenizer, user_input, w,
                max_new_tokens=128, temperature=0.7, top_p=0.8, top_k=20
            )

            print(f"  {zh_style:4s} {b_resp[:80]:80s} | {l_resp[:80]:80s}")

            results[f"{user_input[:30]}/{zh_style}"] = {
                "baseline": b_resp,
                "lora": l_resp,
            }

    with open("log/compare_baseline_vs_lora.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to log/compare_baseline_vs_lora.json")


if __name__ == "__main__":
    main()
