#!/usr/bin/env python3
"""SPR-VR: Sentence-level Prompt Routing with Verification Repair.

Controls emotional support reply structure by assigning each sentence
a function (EMPATHY or ADVICE), routing to the appropriate Prompt Expert,
and verifying+repairing the result.

4 methods compared:
  1. Prompt-only — ask model directly for a given E/A ratio
  2. Structured Prompt — give explicit E/E/A/A sentence plan, one-shot generation
  3. SPR — sentence-level prompt routing, no repair
  4. SPR-VR — SPR + Verifier Repair (our main method)
"""

import json, re, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MODEL_PATH = "./models/Qwen/Qwen3-1.7B"
LOG_DIR = Path("log")
LOG_DIR.mkdir(exist_ok=True)


# ─── Model Loading ───────────────────────────────────────────
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16, trust_remote_code=True, device_map="auto"
    )
    return model, tokenizer


def generate(model, tokenizer, system_prompt, user_content, max_tokens=200):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs, max_new_tokens=max_tokens, temperature=0.7, top_p=0.8, top_k=20,
        do_sample=True, pad_token_id=tokenizer.pad_token_id,
    )
    resp = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    # Strip think tags
    resp = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL).strip()
    return resp


# ─── Verifier ────────────────────────────────────────────────
VERIFIER_PROMPT = """Classify the primary function of this sentence in an emotional support reply.

User input: {user_input}

Sentence: {sentence}

Choose exactly one label:
EMPATHY — primarily expresses understanding, validation, emotional acceptance.
ADVICE — primarily provides suggestions, action steps, or solutions.
OTHER — neither category is clearly dominant.

Output only the label."""


def verify_sentence(model, tokenizer, user_input, sentence):
    """Classify a sentence as EMPATHY / ADVICE / OTHER."""
    prompt = VERIFIER_PROMPT.format(user_input=user_input, sentence=sentence)
    resp = generate(model, tokenizer, "You are a precise sentence classifier.", prompt, max_tokens=20)
    resp = resp.strip().upper()
    for label in ["EMPATHY", "ADVICE", "OTHER"]:
        if label in resp:
            return label
    return "OTHER"


# ─── Semantic Planner ────────────────────────────────────────
PLANNER_PROMPT = """Analyze this user's emotional expression and create a brief support plan.

User input: {user_input}

Output a JSON object with these fields:
- emotion: the primary emotion expressed (1-3 words)
- situation: brief summary of the user's situation (1 sentence)
- support_focus: what emotional support angle to take (1 sentence)
- advice_focus: what practical advice to offer (1 sentence)

Output only the JSON object."""


def create_semantic_plan(model, tokenizer, user_input):
    """Create a shared semantic plan for coherent multi-sentence generation."""
    prompt = PLANNER_PROMPT.format(user_input=user_input)
    resp = generate(model, tokenizer, "You are a psychological support planner.", prompt, max_tokens=250)
    # Try to parse JSON
    try:
        # Find JSON block
        match = re.search(r'\{.*\}', resp, re.DOTALL)
        if match:
            plan = json.loads(match.group())
            return plan
    except json.JSONDecodeError:
        pass
    # Fallback: use raw response as plan
    return {"emotion": "distress", "situation": user_input,
            "support_focus": "Validate feelings and offer support",
            "advice_focus": "Suggest practical steps"}


# ─── Prompt Experts ──────────────────────────────────────────
EMPATHY_SYSTEM = "You are an emotional support reply generator. You are writing ONE sentence of a multi-sentence reply."

EMPATHY_USER = """You are writing ONE sentence of an emotional support reply.

User's message: {user_input}

Semantic plan: {semantic_plan}

Previous sentences: {previous}

Your sentence function: EMPATHY — express understanding, validation, and emotional acceptance.
Requirements:
1. Write exactly ONE natural sentence. No bullet points, no labels.
2. Focus on validating emotions and showing understanding.
3. Do NOT give advice or suggestions in this sentence.
4. Flow naturally from previous sentences. Do not repeat.
5. Do not output the word "EMPATHY" or any label.

Generate only the sentence:"""

ADVICE_SYSTEM = "You are an emotional support reply generator. You are writing ONE sentence of a multi-sentence reply."

ADVICE_USER = """You are writing ONE sentence of an emotional support reply.

User's message: {user_input}

Semantic plan: {semantic_plan}

Previous sentences: {previous}

Your sentence function: ADVICE — provide a practical, gentle, actionable suggestion.
Requirements:
1. Write exactly ONE natural sentence. No bullet points, no labels.
2. Provide a specific, gentle, and actionable suggestion.
3. You may briefly acknowledge emotions, but the FOCUS must be advice.
4. Flow naturally from previous sentences. Do not repeat.
5. Do not output the word "ADVICE" or any label.

Generate only the sentence:"""


# ─── Style Budget Planner ────────────────────────────────────
def get_sentence_plan(alpha):
    """Map target empathy ratio to sentence plan. E=EMPATHY, A=ADVICE. 4 sentences."""
    n_empathy = int(round(alpha * 4))
    n_advice = 4 - n_empathy
    # Empathy sentences first, then advice (natural emotional support flow)
    return ["EMPATHY"] * n_empathy + ["ADVICE"] * n_advice


# ─── Method 1: Prompt-only ───────────────────────────────────
def method_prompt_only(model, tokenizer, user_input, alpha):
    prompt = f"""Generate a 4-sentence emotional support reply.
Target composition: approximately {alpha*100:.0f}% empathy/validation, {(1-alpha)*100:.0f}% advice/suggestions.
Make the reply natural, coherent, and helpful.

User input: {user_input}

Your 4-sentence reply (no labels, just the sentences):"""
    resp = generate(model, tokenizer, "You are a helpful emotional support assistant.", prompt, max_tokens=250)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', resp) if s.strip()]
    return sentences[:4] if len(sentences) >= 4 else sentences


# ─── Method 2: Structured Prompt ─────────────────────────────
def method_structured_prompt(model, tokenizer, user_input, alpha):
    plan = get_sentence_plan(alpha)
    plan_str = "\n".join(f"Sentence {i+1}: {p}" for i, p in enumerate(plan))

    prompt = f"""Generate a 4-sentence emotional support reply following this EXACT sentence plan:

{plan_str}

Requirements:
- EMPATHY sentences: express understanding, validation, emotional acceptance.
- ADVICE sentences: provide specific, gentle, actionable suggestions.
- The reply must be natural and coherent as a whole.
- Do NOT output labels like "EMPATHY:" or "ADVICE:" in the response.
- Output exactly 4 sentences.

User input: {user_input}

Your reply:"""
    resp = generate(model, tokenizer, "You are a helpful emotional support assistant.", prompt, max_tokens=250)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', resp) if s.strip()]
    return sentences[:4] if len(sentences) >= 4 else sentences


# ─── Method 3: SPR (no repair) ───────────────────────────────
def method_spr(model, tokenizer, user_input, alpha):
    plan = get_sentence_plan(alpha)
    sem_plan = create_semantic_plan(model, tokenizer, user_input)
    sem_plan_str = json.dumps(sem_plan, ensure_ascii=False)

    sentences = []
    for i, sentence_type in enumerate(plan):
        prev = "\n".join(f"[{i+1}] {s}" for i, s in enumerate(sentences)) if sentences else "(none)"

        if sentence_type == "EMPATHY":
            user_content = EMPATHY_USER.format(
                user_input=user_input, semantic_plan=sem_plan_str, previous=prev
            )
            sent = generate(model, tokenizer, EMPATHY_SYSTEM, user_content, max_tokens=80)
        else:
            user_content = ADVICE_USER.format(
                user_input=user_input, semantic_plan=sem_plan_str, previous=prev
            )
            sent = generate(model, tokenizer, ADVICE_SYSTEM, user_content, max_tokens=80)

        sent = sent.strip().strip('"').strip("'")
        sentences.append(sent)

    return sentences


# ─── Method 4: SPR-VR (with Verifier Repair) ──────────────────
def method_spr_vr(model, tokenizer, user_input, alpha):
    plan = get_sentence_plan(alpha)
    sem_plan = create_semantic_plan(model, tokenizer, user_input)
    sem_plan_str = json.dumps(sem_plan, ensure_ascii=False)

    sentences = []
    repair_count = 0
    max_repairs_per_sentence = 1

    for i, target_type in enumerate(plan):
        prev = "\n".join(f"[{i+1}] {s}" for i, s in enumerate(sentences)) if sentences else "(none)"

        for attempt in range(max_repairs_per_sentence + 1):
            if target_type == "EMPATHY":
                user_content = EMPATHY_USER.format(
                    user_input=user_input, semantic_plan=sem_plan_str, previous=prev
                )
                sent = generate(model, tokenizer, EMPATHY_SYSTEM, user_content, max_tokens=80)
            else:
                user_content = ADVICE_USER.format(
                    user_input=user_input, semantic_plan=sem_plan_str, previous=prev
                )
                sent = generate(model, tokenizer, ADVICE_SYSTEM, user_content, max_tokens=80)

            sent = sent.strip().strip('"').strip("'")

            # Verify
            predicted = verify_sentence(model, tokenizer, user_input, sent)

            if predicted == target_type:
                sentences.append(sent)
                break
            elif attempt < max_repairs_per_sentence:
                repair_count += 1
                # Continue loop to retry
            else:
                # Max retries reached, keep the sentence
                sentences.append(sent)

    return sentences


# ─── Evaluation ──────────────────────────────────────────────
def evaluate_sentences(model, tokenizer, user_input, sentences):
    """Classify each sentence and compute actual empathy ratio."""
    labels = []
    for sent in sentences:
        if not sent.strip():
            labels.append("OTHER")
        else:
            labels.append(verify_sentence(model, tokenizer, user_input, sent))
    n_empathy = labels.count("EMPATHY")
    n_total = max(len(labels), 1)
    return labels, n_empathy / n_total


def compute_metrics(results):
    """Compute MAE, Pearson R, R², Exact Match from results dict."""
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    methods = ["Prompt-only", "Structured Prompt", "SPR", "SPR-VR"]

    metrics = {}
    for method in methods:
        actuals = []
        targets = []
        exact_matches = 0
        total = 0

        for alpha in alphas:
            for item in results:
                entry = item.get(f"{method}_{alpha}")
                if not entry:
                    continue
                actual_alpha = entry.get("actual_alpha", 0)
                sentence_labels = entry.get("labels", [])
                plan = get_sentence_plan(alpha)

                actuals.append(actual_alpha)
                targets.append(alpha)
                total += 1

                if sentence_labels[:4] == plan[:4]:
                    exact_matches += 1

        if actuals:
            mae = float(np.mean(np.abs(np.array(actuals) - np.array(targets))))
            pearson = float(np.corrcoef(actuals, targets)[0, 1]) if len(actuals) > 2 else 0.0
            ss_res = np.sum((np.array(actuals) - np.array(targets)) ** 2)
            ss_tot = np.sum((np.array(targets) - np.mean(targets)) ** 2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
            exact_rate = exact_matches / total if total > 0 else 0.0

            metrics[method] = {
                "MAE": round(mae, 4),
                "Pearson": round(pearson, 4),
                "R2": round(r2, 4),
                "ExactMatch": round(exact_rate, 4),
            }

    return metrics


def main():
    print("=" * 60)
    print("  SPR-VR: Sentence-level Prompt Routing with Verification")
    print("=" * 60)

    # Load test inputs
    with open("data/eval/test_inputs_30.json") as f:
        test_inputs = json.load(f)
    print(f"\nTest inputs: {len(test_inputs)}")

    # Load model
    print("Loading model...")
    model, tokenizer = load_model()

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    methods = {
        "Prompt-only": method_prompt_only,
        "Structured Prompt": method_structured_prompt,
        "SPR": method_spr,
        "SPR-VR": method_spr_vr,
    }

    all_results = []
    total = len(test_inputs) * len(alphas) * len(methods)

    print(f"\nRunning experiment: {len(test_inputs)} inputs × {len(alphas)} α × {len(methods)} methods")
    print(f"Total generations: ~{total} (SPR/SPR-VR will generate 4 sentences each ≈ {total * 2})")

    idx = 0
    for user_input in tqdm(test_inputs, desc="Inputs"):
        entry = {"input": user_input}

        for alpha in alphas:
            for method_name, method_fn in methods.items():
                t0 = time.time()
                try:
                    sentences = method_fn(model, tokenizer, user_input, alpha)
                    # Ensure 4 sentences
                    if len(sentences) < 4:
                        sentences = sentences + [""] * (4 - len(sentences))
                    sentences = sentences[:4]

                    labels, actual_alpha = evaluate_sentences(model, tokenizer, user_input, sentences)
                except Exception as e:
                    print(f"\n  Error [{method_name} α={alpha}]: {e}")
                    sentences = [""] * 4
                    labels = ["OTHER"] * 4
                    actual_alpha = 0.0

                elapsed = time.time() - t0
                entry[f"{method_name}_{alpha}"] = {
                    "sentences": sentences,
                    "labels": labels,
                    "actual_alpha": actual_alpha,
                    "target_alpha": alpha,
                    "time_s": round(elapsed, 1),
                }
                idx += 1

        all_results.append(entry)
        # Save incrementally
        if len(all_results) % 5 == 0:
            with open(LOG_DIR / "spr_vr_results.json", "w") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Final save
    with open(LOG_DIR / "spr_vr_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Compute metrics
    metrics = compute_metrics(all_results)

    # ── Print results ──
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    # Table 1: Main metrics
    print(f"\n{'Method':<20s} {'MAE↓':>8s} {'Pearson↑':>10s} {'R²↑':>8s} {'ExactMatch↑':>12s}")
    print("-" * 60)
    for method in ["Prompt-only", "Structured Prompt", "SPR", "SPR-VR"]:
        m = metrics.get(method, {})
        print(f"{method:<20s} {m.get('MAE', 0):>8.4f} {m.get('Pearson', 0):>10.4f} {m.get('R2', 0):>8.4f} {m.get('ExactMatch', 0):>12.4f}")

    # Table 2: Per-alpha breakdown
    print(f"\n{'Target α':<10s}", end="")
    for method in ["Prompt-only", "Structured Prompt", "SPR", "SPR-VR"]:
        print(f" {method:>18s}", end="")
    print()
    print("-" * 85)

    for alpha in alphas:
        print(f"{alpha:<10.2f}", end="")
        for method_name in ["Prompt-only", "Structured Prompt", "SPR", "SPR-VR"]:
            vals = []
            for entry in all_results:
                key = f"{method_name}_{alpha}"
                if key in entry:
                    vals.append(entry[key]["actual_alpha"])
            avg = np.mean(vals) if vals else 0.0
            print(f" {avg:>18.4f}", end="")
        print()

    # Save metrics
    with open(LOG_DIR / "spr_vr_metrics.json", "w") as f:
        json.dump({
            "metrics": metrics,
            "per_alpha": {},
            "num_inputs": len(test_inputs),
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to log/spr_vr_results.json")
    print(f"Metrics saved to log/spr_vr_metrics.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
