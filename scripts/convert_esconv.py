#!/usr/bin/env python3
"""Convert ESConv dataset to style-specific JSONL training data.

Maps supporter strategies to emotion styles:
  empathetic:    Reflection of feelings, Restatement or Paraphrasing, Self-disclosure
  rational:      Providing Suggestions, Information, Question
  encouraging:   Affirmation and Reassurance, Self-disclosure
  calm_safe:     Others, Question (cautious), Information (professional)
"""

import json
from pathlib import Path

from datasets import load_from_disk

STRATEGY_TO_STYLE = {
    "Reflection of feelings": "empathetic",
    "Restatement or Paraphrasing": "empathetic",
    "Self-disclosure": ["empathetic", "encouraging"],  # split between two
    "Providing Suggestions": "rational",
    "Information": "rational",
    "Question": "rational",
    "Affirmation and Reassurance": "encouraging",
    "Others": "calm_safe",
}


def convert_esconv_to_style_data(dataset_path: str, output_dir: str):
    ds = load_from_disk(dataset_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    style_data = {"empathetic": [], "rational": [], "encouraging": [], "calm_safe": []}
    self_disclosure_count = 0
    skipped = 0

    for row in ds["train"]:
        item = json.loads(row["text"])
        dialog = item["dialog"]
        situation = item.get("situation", "")
        emotion = item.get("emotion_type", "")

        for i in range(len(dialog) - 1):
            turn = dialog[i]
            next_turn = dialog[i + 1]

            if turn["speaker"] != "usr" or next_turn["speaker"] != "sys":
                continue

            user_msg = turn["text"]
            assistant_msg = next_turn["text"]
            strategy = next_turn.get("strategy", "Others")

            style = STRATEGY_TO_STYLE.get(strategy)
            if style is None:
                skipped += 1
                continue

            # Self-disclosure goes equally to empathetic and encouraging
            if isinstance(style, list):
                style = style[self_disclosure_count % 2]
                self_disclosure_count += 1

            # Add context: prepend situation if available
            user_with_context = f"[背景: {situation}] [情绪: {emotion}]\n{user_msg}" if situation else user_msg

            style_data[style].append({
                "user": user_with_context,
                "assistant": assistant_msg,
            })

    # Write JSONL per style
    for style, records in style_data.items():
        fpath = out / f"{style}.jsonl"
        with open(fpath, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{style}: {len(records)} samples → {fpath}")

    # Also write a merged file
    all_records = []
    for records in style_data.values():
        all_records.extend(records)
    fpath = out / "all_styles.jsonl"
    with open(fpath, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"all_styles: {len(all_records)} samples → {fpath}")
    print(f"Skipped (unknown strategy): {skipped}")


if __name__ == "__main__":
    convert_esconv_to_style_data("data/esconv", "data/train")
