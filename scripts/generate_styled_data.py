#!/usr/bin/env python3
"""Generate contrastive multi-style training data.

Uses Qwen3 base model + strong contrastive prompts to generate
4 distinct style responses for each input, creating differentiated
training targets that will produce stylistically separate LoRA adapters.

Output: data/train_v2/{empathetic,rational,encouraging,calm_safe}.jsonl
"""

import json
import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_jsonl, save_jsonl

MODEL_PATH = "./models/Qwen/Qwen3-1.7B"
OUTPUT_DIR = Path("data/train_v2")
OUTPUT_DIR.mkdir(exist_ok=True)

CONTRASTIVE_SYSTEM_PROMPTS = {
    "empathetic": (
        "You ONLY validate feelings and offer emotional comfort. "
        "NEVER give advice, suggestions, or analyze. "
        "NEVER ask probing questions. "
        "DO: 'That sounds really hard. I hear you. You're not alone in this.' "
        "DON'T: 'Let's break this down into steps.' 'Have you tried...' 'What specifically...' "
        "Show warmth, understanding, and unconditional acceptance."
    ),
    "rational": (
        "You ONLY give logical, structured analysis. "
        "NEVER express emotion, sympathy, or self-disclose. "
        "NEVER offer reassurance without analysis. "
        "DO: 'Let's analyze this systematically. 1. Identify the core issue. 2. Evaluate options. 3. Take action.' "
        "DON'T: 'I'm sorry to hear that.' 'I understand how you feel.' 'That must be so hard.' "
        "Be professional, objective, and solution-oriented."
    ),
    "encouraging": (
        "You ONLY affirm strengths and motivate action. "
        "NEVER analyze problems or ask probing questions. "
        "NEVER be neutral or cautious. "
        "DO: 'You are stronger than you think! You've got this. I believe in you.' "
        "DON'T: 'Let's look at this objectively.' 'What specifically is the problem?' 'Be careful.' "
        "Be energetic, optimistic, and empowering."
    ),
    "calm_safe": (
        "You ONLY assess safety and provide professional, measured support. "
        "NEVER express strong emotion, give casual advice, or self-disclose. "
        "NEVER be overly optimistic or dismissive. "
        "DO: 'Are you safe right now? Your well-being is the priority. Let's take this one step at a time.' "
        "DON'T: 'Oh no, that's terrible!' 'Just cheer up!' 'I've been there too.' "
        "Be steady, professional, and safety-focused."
    ),
}


def generate_styled_responses(tokenizer, model, user_input: str) -> dict[str, str]:
    """Generate one response per style using contrastive prompts."""
    responses = {}
    for style, sys_prompt in CONTRASTIVE_SYSTEM_PROMPTS.items():
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_input},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        resp = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        # Strip any residual think tags
        import re
        resp = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL).strip()
        responses[style] = resp
    return responses


def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16, trust_remote_code=True, device_map="auto"
    )
    print("Model loaded.")

    # Load ESConv user utterances
    print("Loading ESConv data...")
    ds = load_from_disk("data/esconv")
    user_inputs = []
    seen = set()
    for row in ds["train"]:
        item = json.loads(row["text"])
        for turn in item["dialog"]:
            if turn["speaker"] == "usr":
                text = turn["text"].strip()
                if text and len(text) > 10 and text not in seen:
                    seen.add(text)
                    user_inputs.append(text)

    print(f"Unique user utterances: {len(user_inputs)}")

    # Sample: take up to 500 for each style (2000 total generations)
    # Actually use all unique inputs but cap at 500 for time
    import random
    random.seed(42)
    if len(user_inputs) > 200:
        user_inputs = random.sample(user_inputs, 200)

    print(f"Generating {len(user_inputs)} inputs × 4 styles = {len(user_inputs) * 4} responses...")

    style_data = {s: [] for s in CONTRASTIVE_SYSTEM_PROMPTS}

    for user_input in tqdm(user_inputs, desc="Generating"):
        try:
            responses = generate_styled_responses(tokenizer, model, user_input)
            for style, resp in responses.items():
                if resp:  # skip empty
                    style_data[style].append({
                        "user": user_input,
                        "assistant": resp,
                    })
        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Save
    for style, records in style_data.items():
        fpath = OUTPUT_DIR / f"{style}.jsonl"
        save_jsonl(records, str(fpath))
        print(f"  {style}: {len(records)} samples → {fpath}")

    # Also save merged
    all_records = []
    for records in style_data.values():
        all_records.extend(records)
    save_jsonl(all_records, str(OUTPUT_DIR / "all_styles.jsonl"))
    print(f"  all_styles: {len(all_records)} samples")

    print("\nDone. Data saved to data/train_v2/")


if __name__ == "__main__":
    main()
