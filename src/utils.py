"""Utility functions for style-controllable LoRA interpolation."""

import json
import re
from typing import Any

import torch


STYLE_NAMES = {
    "empathetic": "温柔共情型",
    "rational": "理性分析型",
    "encouraging": "鼓励激励型",
    "calm_safe": "冷静安全型",
}

STYLE_SYSTEM_PROMPTS = {
    "empathetic": (
        "You are a warm, empathetic listener. "
        "Respond with gentleness and compassion. "
        "Prioritize understanding and validating the user's emotions, "
        "providing emotional companionship and comfort."
    ),
    "rational": (
        "You are a rational, analytical advisor. "
        "Objectively analyze the user's situation. "
        "Break down causes clearly and provide structured advice and solutions."
    ),
    "encouraging": (
        "You are a positive, energetic motivator. "
        "Respond with enthusiasm and affirmation. "
        "Highlight the user's strengths and potential, "
        "encourage action and build confidence."
    ),
    "calm_safe": (
        "You are a calm, professional psychological safety officer. "
        "Respond with steadiness and security. "
        "Avoid excessive emotional expression. Ensure the user feels safe and respected. "
        "Pay special attention to negative emotions and potential risks."
    ),
}

# Qwen3 non-thinking mode recommended params
QWEN3_NON_THINKING_GENERATION = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
}


def build_system_prompt(style_weights: dict[str, float]) -> str:
    """Build a single blended system prompt from style weights."""
    parts = []
    for style, weight in style_weights.items():
        if weight > 0.01 and style in STYLE_SYSTEM_PROMPTS:
            parts.append(f"[{STYLE_NAMES.get(style, style)} × {weight:.0%}] {STYLE_SYSTEM_PROMPTS[style]}")
    return "\n".join(parts) if parts else STYLE_SYSTEM_PROMPTS["calm_safe"]


def format_messages(
    user_input: str,
    style_weights: dict[str, float],
    history: list[dict] | None = None,
) -> list[dict]:
    """Build chat messages with system prompt + optional history + user input.

    Uses the messages format expected by Qwen3's apply_chat_template.
    """
    system_prompt = build_system_prompt(style_weights)
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_input})
    return messages


def apply_chat_template(tokenizer, messages: list[dict], **kwargs) -> str:
    """Apply Qwen3 chat template with thinking disabled.

    enable_thinking=False: no <think>...</think> blocks, behaving like Qwen2.5-Instruct.
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
        **kwargs,
    )


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Normalize style weights to sum to 1."""
    total = sum(weights.values())
    if total == 0:
        n = len(weights)
        return {k: 1.0 / n for k in weights}
    return {k: v / total for k, v in weights.items()}


def extract_response(output: str) -> str:
    """Extract the assistant response, stripping think blocks."""
    # Try ChatML format
    match = re.search(r"<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)", output, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        text = output

    # Strip <think>...</think> blocks (may appear even with enable_thinking=False)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def load_jsonl(filepath: str) -> list[dict[str, Any]]:
    """Load a JSONL file."""
    data = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: list[dict[str, Any]], filepath: str) -> None:
    """Save data as JSONL."""
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


@torch.no_grad()
def average_lora_weights(lora_weights_list: list[dict], weights: list[float]) -> dict:
    """Weighted average of LoRA parameter dictionaries."""
    if len(lora_weights_list) != len(weights):
        raise ValueError(f"Mismatched lengths: {len(lora_weights_list)} LoRAs vs {len(weights)} weights")

    weight_sum = sum(weights)
    normalized = [w / weight_sum for w in weights]

    averaged = {}
    for key in lora_weights_list[0]:
        stacked = torch.stack([lw[key].float() for lw in lora_weights_list])
        avg = sum(normalized[i] * stacked[i] for i in range(len(normalized)))
        averaged[key] = avg.to(lora_weights_list[0][key].dtype)

    return averaged
