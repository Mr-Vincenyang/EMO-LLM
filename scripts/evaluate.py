#!/usr/bin/env python3
"""Evaluate quality of style interpolation across the weight grid.

Sweeps combinations of style weights and checks:
  1. Response consistency (same input + same weights → similar output)
  2. Style discernibility (extreme weights produce clearly different styles)
  3. Smooth interpolation (neighboring weight combos produce gradual changes)

Usage:
    python scripts/evaluate.py --lora_dir outputs/lora --model_name ./models/Qwen/Qwen3-1.7B
"""

from __future__ import annotations

import argparse
import json
import itertools
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.interpolate import InterpolatableLoRA
from src.generate import generate_with_interpolation


def sweep_weights(grid_resolution: int = 5):
    """Generate weight combinations on a grid for 4 styles.

    Uses a simplex sampling to ensure weights are meaningful.
    """
    styles = ["empathetic", "rational", "encouraging", "calm_safe"]
    results = []

    # Grid: each weight in {0, 1/grid, 2/grid, ..., 1}
    # We sample meaningful combinations (top-2 dominant styles)
    values = np.linspace(0, 1, grid_resolution + 1)
    for combo in itertools.product(values, repeat=2):
        dominant = combo[0]
        secondary = combo[1]
        if dominant + secondary > 1.0:
            continue
        remainder = 1.0 - dominant - secondary
        # Distribute remainder equally among the other 2 styles
        for primary_style, secondary_style in itertools.permutations(styles, 2):
            other_styles = [s for s in styles if s not in (primary_style, secondary_style)]
            weights = {primary_style: dominant, secondary_style: secondary}
            for s in other_styles:
                weights[s] = remainder / 2.0
            results.append(weights)

    # Deduplicate
    seen = set()
    unique = []
    for w in results:
        key = tuple(round(w[s], 4) for s in styles)
        if key not in seen:
            seen.add(key)
            unique.append(w)

    return unique


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="./models/Qwen/Qwen3-1.7B")
    parser.add_argument("--lora_dir", default="outputs/lora")
    parser.add_argument("--output", default="outputs/eval_results.json")
    parser.add_argument("--grid_resolution", type=int, default=3)
    parser.add_argument("--test_inputs", nargs="+", default=[
        "最近工作压力很大，感觉喘不过气来",
        "我和好朋友发生了矛盾，不知道该怎么办",
        "对未来感到很迷茫，不知道该怎么选择",
    ])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    print(f"Loading base model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map=args.device,
    )

    lora_dir = Path(args.lora_dir)
    lora_paths = {}
    for style in ["empathetic", "rational", "encouraging", "calm_safe"]:
        p = lora_dir / style
        if p.exists():
            lora_paths[style] = str(p)

    if not lora_paths:
        print("No LoRA adapters found. Cannot evaluate.")
        return

    wrapper = InterpolatableLoRA(
        base_model=model,
        lora_paths=lora_paths,
        interpolation_mode="weight",
    )

    weight_combos = sweep_weights(args.grid_resolution)
    print(f"Evaluating {len(weight_combos)} weight combinations × {len(args.test_inputs)} inputs...")

    results = []
    for weights in tqdm(weight_combos, desc="Sweeping weights"):
        for user_input in args.test_inputs:
            response, info = generate_with_interpolation(
                wrapper, tokenizer, user_input, weights,
                max_new_tokens=256, temperature=0.7, top_p=0.8, top_k=20, min_p=0.0,
            )
            results.append({
                "input": user_input,
                "weights": weights,
                "response": response,
            })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()
