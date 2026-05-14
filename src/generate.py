"""Controllable generation via LoRA interpolation.

Given trained style LoRAs, generate responses with continuous style blending.

Two interpolation modes are supported:
  - "weight": Blend LoRA parameters in weight space (preferred, more efficient).
  - "logit":  Blend logits from separate forward passes (more flexible).

Usage:
    python src/generate.py \
        --lora_paths empathetic=outputs/lora/empathetic,encouraging=outputs/lora/encouraging \
        --weights empathetic=0.7,encouraging=0.3 \
        --input "最近工作压力很大，感觉喘不过气来"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

from interpolate import InterpolatableLoRA
from utils import (
    STYLE_NAMES,
    format_messages,
    apply_chat_template,
    normalize_weights,
    extract_response,
    QWEN3_NON_THINKING_GENERATION,
)


def parse_key_value_pairs(arg: str) -> dict[str, float]:
    """Parse 'key1=0.7,key2=0.3' into dict."""
    result = {}
    for pair in arg.split(","):
        key, _, value = pair.partition("=")
        result[key.strip()] = float(value)
    return result


def load_base_model(model_name: str, device: str = "auto"):
    """Load base model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map=device,
    )
    return model, tokenizer


@torch.no_grad()
def generate_with_interpolation(
    model: InterpolatableLoRA,
    tokenizer,
    user_input: str,
    weights: dict[str, float],
    max_new_tokens: int = 512,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
) -> tuple[str, dict]:
    """Generate a response using interpolated LoRA weights.

    Defaults to Qwen3 non-thinking mode recommended params if not specified.
    """
    weights = normalize_weights(weights.copy())

    # Update model weights for weight-space interpolation
    if model.mode == "weight":
        model.set_weights(weights)

    # Build prompt via Qwen3 chat template
    messages = format_messages(user_input, weights)
    prompt = apply_chat_template(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Default to Qwen3 non-thinking mode params
    defaults = QWEN3_NON_THINKING_GENERATION
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature if temperature is not None else defaults["temperature"],
        top_p=top_p if top_p is not None else defaults["top_p"],
        top_k=top_k if top_k is not None else defaults["top_k"],
        min_p=min_p if min_p is not None else defaults["min_p"],
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    output_ids = model.generate(**inputs, generation_config=gen_config)
    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=False)

    response = extract_response(full_output)

    info = {
        "weights": weights,
        "style_labels": {k: STYLE_NAMES.get(k, k) for k in weights},
    }
    return response, info


def main():
    parser = argparse.ArgumentParser(description="Controllable generation via LoRA interpolation")
    parser.add_argument("--model_name", default="./models/Qwen/Qwen3-1.7B", help="Base model name or path")
    parser.add_argument(
        "--lora_paths", required=True,
        help="Comma-separated style=path pairs, e.g. empathetic=outputs/lora/empathetic,rational=outputs/lora/rational",
    )
    parser.add_argument(
        "--weights", default=None,
        help="Comma-separated style=weight pairs, e.g. empathetic=0.7,rational=0.3. "
             "If not provided, defaults to equal blending.",
    )
    parser.add_argument("--input", default=None, help="User input text")
    parser.add_argument("--mode", default="weight", choices=["weight", "logit"],
                        help="Interpolation mode")
    parser.add_argument("--device", default="auto", help="Device for inference")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    args = parser.parse_args()

    lora_paths = {}
    for pair in args.lora_paths.split(","):
        key, _, path = pair.partition("=")
        lora_paths[key.strip()] = path.strip()

    print(f"Loading base model: {args.model_name}")
    model, tokenizer = load_base_model(args.model_name, args.device)

    print(f"Loading {len(lora_paths)} LoRA adapters...")
    wrapper = InterpolatableLoRA(
        base_model=model,
        lora_paths=lora_paths,
        interpolation_mode=args.mode,
    )

    if args.weights:
        weights = parse_key_value_pairs(args.weights)
    else:
        n = len(lora_paths)
        weights = {k: 1.0 / n for k in lora_paths}

    user_input = args.input
    if not user_input:
        user_input = input("Enter your message: ")

    print(f"\nStyle weights: {weights}")
    print(f"User input: {user_input}\n")

    response, info = generate_with_interpolation(
        wrapper, tokenizer, user_input, weights,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
    )

    print(f"Response: {response}")
    print(f"\nStyle breakdown: {json.dumps(info, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
