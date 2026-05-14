#!/usr/bin/env python3
"""Quantitative evaluation of LoRA interpolation for emotion-style control.

Metrics:
  1. Style Discriminability  — Can a classifier tell which style was used?
  2. Interpolation Smoothness — Do neighboring weights produce gradual changes?
  3. Perplexity — Are generated responses fluent?
  4. Distinct-N — Response diversity
  5. Self-BLEU (inter-style) — Are different styles truly different?
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cosine, euclidean
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from interpolate import InterpolatableLoRA
from generate import generate_with_interpolation

LOG_DIR = Path("log")
MODEL_PATH = "./models/Qwen/Qwen3-1.7B"
LORA_DIR = Path("outputs/lora_v2")

TEST_INPUTS = [
    "I'm feeling really stressed at work lately and don't know how to cope.",
    "My friend and I had a big argument and now I feel terrible.",
    "I feel lost about my future and don't know what career to choose.",
    "I've been feeling anxious and can't sleep well at night.",
    "My parents keep pressuring me about my life choices.",
]


def load_model_and_loras():
    print("Loading model and adapters...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16, trust_remote_code=True, device_map="auto"
    )
    lora_paths = {s: str(LORA_DIR / s) for s in ["empathetic", "rational", "encouraging", "calm_safe"]}
    wrapper = InterpolatableLoRA(model, lora_paths, interpolation_mode="weight")
    return wrapper, tokenizer


def compute_ppl(model, tokenizer, texts: list[str]) -> float:
    """Compute perplexity using the base model."""
    losses = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            losses.append(outputs.loss.item())
    return float(np.exp(np.mean(losses)))


def compute_distinct_n(texts: list[str], n: int = 2) -> float:
    """Distinct-N: ratio of unique n-grams to total n-grams."""
    from collections import Counter

    total_ngrams = 0
    unique_ngrams = Counter()
    for text in texts:
        words = text.lower().split()
        ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
        total_ngrams += len(ngrams)
        unique_ngrams.update(ngrams)
    return len(unique_ngrams) / total_ngrams if total_ngrams > 0 else 0.0


def ngram_overlap(text_a: str, text_b: str, n: int = 2) -> float:
    """Simple n-gram overlap between two texts (0–1)."""
    words_a = text_a.lower().split()
    words_b = text_b.lower().split()
    ngrams_a = {tuple(words_a[i:i+n]) for i in range(len(words_a)-n+1)}
    ngrams_b = {tuple(words_b[i:i+n]) for i in range(len(words_b)-n+1)}
    union = ngrams_a | ngrams_b
    inter = ngrams_a & ngrams_b
    return len(inter) / len(union) if union else 0.0


def compute_self_bleu(samples_per_style: dict[str, list[str]]) -> dict:
    """Style separation via n-gram overlap (replaces BLEU without nltk).

    Higher intra-style overlap + lower cross-style overlap = better style separation.
    """
    styles = sorted(samples_per_style.keys())

    intra_overlap = {}
    cross_overlap = {}

    for s in styles:
        samples = samples_per_style[s]
        if len(samples) < 2:
            continue

        # Within-style overlap
        intra_scores = []
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                intra_scores.append(ngram_overlap(samples[i], samples[j], n=2))
        intra_overlap[s] = float(np.mean(intra_scores)) if intra_scores else 0.0

        # Cross-style overlap
        other_samples = []
        for os_ in styles:
            if os_ != s:
                other_samples.extend(samples_per_style[os_])
        cross_scores = []
        for hyp in samples:
            for ref in other_samples[:len(samples)]:
                cross_scores.append(ngram_overlap(hyp, ref, n=2))
        cross_overlap[s] = float(np.mean(cross_scores)) if cross_scores else 0.0

    return {"intra_style": intra_overlap, "cross_style": cross_overlap}


def compute_embedding_distance(samples_per_style: dict[str, list[str]], tokenizer, model) -> dict:
    """Cosine distance between style centroids in embedding space.
    Larger distance = better style separation.
    """
    style_embeddings = {}
    for style, texts in samples_per_style.items():
        embs = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
            with torch.no_grad():
                hidden = model(**inputs, output_hidden_states=True).hidden_states[-1]
                emb = hidden[:, -1, :].cpu().numpy()
                embs.append(emb[0])
        style_embeddings[style] = np.mean(embs, axis=0)

    distances = {}
    styles = sorted(style_embeddings.keys())
    for i, s1 in enumerate(styles):
        for s2 in styles[i + 1:]:
            distances[f"{s1}_vs_{s2}"] = float(cosine(style_embeddings[s1], style_embeddings[s2]))
    return distances


def compute_interpolation_smoothness(
    wrapper, tokenizer, inputs: list[str], steps: int = 9
) -> dict:
    """Measure how smoothly responses change along the empathy→rationality axis.
    Smoothness = low variance in adjacent embedding distances.
    """
    all_smoothness = []
    for user_input in inputs[:3]:
        responses = []
        for alpha in np.linspace(0, 1, steps):
            w = {"empathetic": 1 - alpha, "rational": alpha, "encouraging": 0.0, "calm_safe": 0.0}
            resp, _ = generate_with_interpolation(
                wrapper, tokenizer, user_input, w,
                max_new_tokens=128, temperature=0.7, top_p=0.8, top_k=20
            )
            responses.append(resp)

        # Measure adjacent differences via n-gram overlap
        adj_diffs = []
        for i in range(len(responses) - 1):
            d = 1.0 - ngram_overlap(responses[i], responses[i + 1], n=2)
            adj_diffs.append(d)

        # Smoothness: coefficient of variation of adjacent differences
        diffs = np.array(adj_diffs)
        cv = float(np.std(diffs) / np.mean(diffs)) if np.mean(diffs) > 0 else 0.0
        all_smoothness.append({
            "adjacent_distances": adj_diffs,
            "mean_distance": float(np.mean(diffs)),
            "cv": cv,
            "smoothness_score": 1.0 / (1.0 + cv),  # higher = smoother
        })

    return {
        "avg_smoothness": float(np.mean([s["smoothness_score"] for s in all_smoothness])),
        "avg_distance": float(np.mean([s["mean_distance"] for s in all_smoothness])),
        "per_input": all_smoothness,
    }


def main():
    print("=" * 60)
    print("QUANTITATIVE EVALUATION")
    print("=" * 60)

    wrapper, tokenizer = load_model_and_loras()

    # Generate samples: 4 pure styles × 5 inputs = 20 responses
    print("\nGenerating evaluation samples...")
    samples_per_style = defaultdict(list)
    all_generations = []

    for user_input in tqdm(TEST_INPUTS, desc="Generating"):
        for style in ["empathetic", "rational", "encouraging", "calm_safe"]:
            w = {s: 0.0 for s in ["empathetic", "rational", "encouraging", "calm_safe"]}
            w[style] = 1.0
            resp, _ = generate_with_interpolation(
                wrapper, tokenizer, user_input, w,
                max_new_tokens=128, temperature=0.7, top_p=0.8, top_k=20
            )
            samples_per_style[style].append(resp)
            all_generations.append(resp)

    # ── Metric 1: Perplexity ──
    print("\n[1/5] Computing Perplexity...")
    ppl = compute_ppl(wrapper.peft_model, tokenizer, all_generations)
    print(f"  Overall PPL: {ppl:.2f}")

    # ── Metric 2: Distinct-N ──
    print("\n[2/5] Computing Distinct-N...")
    distinct2 = compute_distinct_n(all_generations, n=2)
    distinct3 = compute_distinct_n(all_generations, n=3)
    print(f"  Distinct-2: {distinct2:.4f}")
    print(f"  Distinct-3: {distinct3:.4f}")

    # ── Metric 3: Self-BLEU (style separation) ──
    print("\n[3/5] Computing Self-BLEU...")
    bleu_results = compute_self_bleu(samples_per_style)
    for style in sorted(bleu_results["intra_style"]):
        intra = bleu_results["intra_style"][style]
        cross = bleu_results["cross_style"][style]
        separation = intra - cross
        print(f"  {style:15s} | intra={intra:.4f} | cross={cross:.4f} | separation={separation:.4f}")

    # ── Metric 4: Embedding distance (style separation) ──
    print("\n[4/5] Computing Style Embedding Distances...")
    emb_dists = compute_embedding_distance(samples_per_style, tokenizer, wrapper.peft_model)
    for pair, dist in sorted(emb_dists.items(), key=lambda x: -x[1]):
        print(f"  {pair:30s} | cos_dist={dist:.4f}")

    # ── Metric 5: Interpolation Smoothness ──
    print("\n[5/5] Computing Interpolation Smoothness...")
    smoothness = compute_interpolation_smoothness(wrapper, tokenizer, TEST_INPUTS)
    print(f"  Smoothness score: {smoothness['avg_smoothness']:.4f}  (1.0 = perfectly smooth)")
    print(f"  Avg step distance: {smoothness['avg_distance']:.4f}")

    # ── Summary ──
    results = {
        "perplexity": ppl,
        "distinct_2": distinct2,
        "distinct_3": distinct3,
        "self_bleu": bleu_results,
        "style_embedding_distances": emb_dists,
        "interpolation_smoothness": smoothness,
        "num_samples": len(all_generations),
    }

    output_path = LOG_DIR / "eval_metrics.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    # Print interpretation
    print("\n--- Interpretation ---")
    print(f"PPL {ppl:.0f}: {'Good' if ppl < 50 else 'High — may indicate overfitting to ESConv style'}")
    intra_mean = np.mean(list(bleu_results["intra_style"].values()))
    cross_mean = np.mean(list(bleu_results["cross_style"].values()))
    print(f"Self-BLEU separation {intra_mean - cross_mean:.4f}: {'Clear style separation' if intra_mean - cross_mean > 0.05 else 'Subtle style differences'}")
    avg_emb_dist = np.mean(list(emb_dists.values()))
    print(f"Avg embedding cosine distance {avg_emb_dist:.4f}: {'Strong separation' if avg_emb_dist > 0.3 else 'Moderate' if avg_emb_dist > 0.1 else 'Weak separation'}")

    # Save human-readable report
    report_path = LOG_DIR / "eval_report.txt"
    with open(report_path, "w") as f:
        f.write("EMO-LLM Quantitative Evaluation Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Perplexity: {ppl:.2f}\n")
        f.write(f"Distinct-2: {distinct2:.4f}\n")
        f.write(f"Distinct-3: {distinct3:.4f}\n\n")
        f.write("Style Separation (Self-BLEU):\n")
        for s in sorted(bleu_results["intra_style"]):
            f.write(f"  {s}: intra={bleu_results['intra_style'][s]:.4f} cross={bleu_results['cross_style'][s]:.4f}\n")
        f.write(f"\nStyle Embedding Cosine Distances:\n")
        for pair, dist in sorted(emb_dists.items(), key=lambda x: -x[1]):
            f.write(f"  {pair}: {dist:.4f}\n")
        f.write(f"\nInterpolation Smoothness: {smoothness['avg_smoothness']:.4f}\n")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
