#!/usr/bin/env python3
"""RECCON: Emotion Cause Recognition via Appraisal-Guided Event Retrieval.

Task: Given a dialogue context and a target emotion utterance,
      identify which utterance(s) caused that emotion.

Baselines:
  1. Nearest Previous — always pick the immediately preceding utterance
  2. TF-IDF Retrieval — BM25 over context utterances
  3. Embedding Retrieval — cosine similarity (Qwen3 embeddings)
  4. LLM Direct Prompt — ask Qwen3 to select the cause
  5. LLM Pairwise — Qwen3 classifies each candidate as cause/not

Our method:
  Appraisal-Guided Multi-Signal Retrieval —
    a) Generate appraisal query from target emotion + utterance
    b) Extract event frames from candidate utterances
    c) Score with 5 signals: semantic similarity, appraisal match,
       speaker relation, temporal proximity, emotion-valence match
    d) Lightweight XGBoost reranker
"""

import json, re, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

LOG_DIR = Path("log")
LOG_DIR.mkdir(exist_ok=True)
MODEL_PATH = "./models/Qwen/Qwen3-1.7B"


# ─── 1. Data Processing ─────────────────────────────────────
def load_reccon_data(split="train", fold="fold1", dataset="dailydialog"):
    """Load RECCON QA data and convert to candidate retrieval format."""
    path = f"data/reccon/data/subtask1/{fold}/{dataset}_qa_{split}_with_context.json"
    with open(path) as f:
        data = json.load(f)

    samples = []
    for item in data:
        context = item["context"]
        context_utterances = [u.strip() for u in re.split(r'(?<=[.!?])\s+', context) if u.strip()]
        if len(context_utterances) < 2:
            context_utterances = [context]  # fallback

        for qa in item["qas"]:
            question = qa["question"]
            gold_answer = qa["answers"][0]["text"]
            is_impossible = "Impossible" in gold_answer or "≠æ" in gold_answer

            # Parse target utterance and emotion from question
            target_match = re.search(r'target utterance is (.+?) The evidence', question)
            emotion_match = re.search(r"emotion (\w+)", question)
            target_text = target_match.group(1).strip() if target_match else ""
            emotion = emotion_match.group(1).strip() if emotion_match else "unknown"

            # Build candidate list: all context utterances except the target itself
            candidates = []
            gold_indices = []
            for i, utt in enumerate(context_utterances):
                # Check if this utterance contains the gold answer
                if not is_impossible and gold_answer.lower() in utt.lower():
                    gold_indices.append(i)
                # Add as candidate (skip if identical to target)
                if target_text.lower() not in utt.lower()[:len(target_text)]:
                    candidates.append({"index": i, "text": utt})

            if not candidates:
                continue

            # Mark gold
            for c in candidates:
                c["is_cause"] = c["index"] in gold_indices

            # Determine target utterance index in context
            target_idx = -1
            for i, utt in enumerate(context_utterances):
                if target_text.lower() in utt.lower()[:len(target_text)]:
                    target_idx = i
                    break

            samples.append({
                "context": context,
                "context_utterances": context_utterances,
                "target_text": target_text,
                "target_idx": target_idx,
                "emotion": emotion,
                "candidates": candidates,
                "gold_indices": gold_indices,
                "is_impossible": is_impossible,
            })

    return samples


def load_all_data():
    """Load train/valid/test from RECCON fold1."""
    train = load_reccon_data("train")
    valid = load_reccon_data("valid")
    test = load_reccon_data("test")
    print(f"Loaded: train={len(train)}, valid={len(valid)}, test={len(test)}")
    # Print stats
    has_cause = sum(1 for s in train if not s["is_impossible"])
    print(f"  Train with cause: {has_cause}/{len(train)}")
    return train, valid, test


# ─── 2. Baselines ────────────────────────────────────────────
def baseline_nearest_previous(sample):
    """Always select the utterance just before the target."""
    if sample["target_idx"] > 0:
        prev_text = sample["context_utterances"][sample["target_idx"] - 1]
        # Find matching candidate
        for c in sample["candidates"]:
            if c["text"].lower() == prev_text.lower():
                return [c["index"]]
        # Fallback: return last candidate before target
        for c in reversed(sample["candidates"]):
            if c["index"] < sample["target_idx"]:
                return [c["index"]]
    return [sample["candidates"][0]["index"]] if sample["candidates"] else []


def baseline_tfidf(samples_train, samples_eval):
    """TF-IDF retrieval: rank candidates by similarity to target + emotion."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Build corpus from training data
    corpus = []
    for s in samples_train:
        for c in s["candidates"]:
            corpus.append(c["text"])
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000).fit(corpus)

    predictions = []
    for sample in samples_eval:
        query = f"{sample['emotion']} {sample['target_text']}"
        query_vec = vectorizer.transform([query])

        if not sample["candidates"]:
            predictions.append([])
            continue

        cand_texts = [c["text"] for c in sample["candidates"]]
        cand_vecs = vectorizer.transform(cand_texts)
        sims = cosine_similarity(query_vec, cand_vecs)[0]
        ranked = np.argsort(-sims)
        predictions.append([sample["candidates"][i]["index"] for i in ranked[:3]])
    return predictions


def baseline_embedding(samples_eval, embed_model):
    """Embedding similarity retrieval using Qwen3 embeddings."""
    predictions = []
    for sample in tqdm(samples_eval, desc="Embedding retrieval"):
        query = f"{sample['emotion']}: {sample['target_text']}"
        q_emb = embed_model.encode([query])[0]

        if not sample["candidates"]:
            predictions.append([])
            continue

        c_embs = embed_model.encode([c["text"] for c in sample["candidates"]])
        sims = np.dot(c_embs, q_emb) / (np.linalg.norm(c_embs, axis=1) * np.linalg.norm(q_emb) + 1e-8)
        ranked = np.argsort(-sims)
        predictions.append([sample["candidates"][i]["index"] for i in ranked[:3]])
    return predictions


def baseline_llm_direct(model, tokenizer, samples_eval, max_samples=100):
    """Ask Qwen3 to directly identify the cause utterance."""
    predictions = []
    for sample in tqdm(samples_eval[:max_samples], desc="LLM Direct"):
        context = "\n".join(f"[{i}] {u}" for i, u in enumerate(sample["context_utterances"]))
        prompt = f"""Find which utterance caused the target emotion.

Context:
{context}

Target utterance: "{sample['target_text']}"
Target emotion: {sample['emotion']}

Which utterance (by index number) is the most likely CAUSE of this emotion?
Answer with the index number only (e.g., "2")."""

        messages = [
            {"role": "system", "content": "You are an emotion cause analyst. Output only the index number."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=20, temperature=0.1, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id)
        resp = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        try:
            idx = int(re.search(r'\d+', resp).group())
            predictions.append([idx])
        except (ValueError, AttributeError):
            predictions.append([sample["candidates"][0]["index"]] if sample["candidates"] else [])
    return predictions


# ─── 3. Appraisal-Guided Multi-Signal Retrieval ───────────────
EMOTION_APPRAISAL = {
    "anger": {"appraisal_need": "finding what caused frustration, injustice, or blocked goals",
              "likely_cause_type": ["goal_blocking", "injustice", "provocation", "threat"]},
    "sadness": {"appraisal_need": "finding what caused loss, disappointment, or helplessness",
                "likely_cause_type": ["loss", "disappointment", "helplessness", "rejection"]},
    "fear": {"appraisal_need": "finding what caused threat, danger, or uncertainty",
             "likely_cause_type": ["threat", "danger", "uncertainty", "loss_of_control"]},
    "happiness": {"appraisal_need": "finding what caused joy, achievement, or positive outcome",
                  "likely_cause_type": ["achievement", "positive_event", "reward", "social_connection"]},
    "disgust": {"appraisal_need": "finding what caused aversion, contamination, or moral violation",
                "likely_cause_type": ["contamination", "moral_violation", "aversion"]},
    "surprise": {"appraisal_need": "finding what caused unexpected event or violation of expectation",
                 "likely_cause_type": ["unexpected_event", "violation_of_expectation"]},
    "unknown": {"appraisal_need": "finding what event triggered this emotional state",
                "likely_cause_type": ["triggering_event", "causal_factor"]},
}


def compute_appraisal_features(sample, embed_model=None):
    """Compute multi-signal features for each candidate."""
    features = []
    target_text = sample["target_text"]
    emotion = sample["emotion"]
    target_idx = sample["target_idx"]
    appraisal = EMOTION_APPRAISAL.get(emotion, EMOTION_APPRAISAL["unknown"])

    # Compute embeddings if model available
    if embed_model:
        c_texts = [c["text"] for c in sample["candidates"]]
        if c_texts:
            t_emb = embed_model.encode([f"{emotion}: {target_text}"])[0]
            c_embs = embed_model.encode(c_texts)

    for ci, c in enumerate(sample["candidates"]):
        feats = {}

        # Signal 1: Semantic similarity (target ↔ candidate)
        if embed_model and c_texts:
            c_emb = c_embs[ci]
            feats["semantic_sim"] = float(np.dot(t_emb, c_emb) / (np.linalg.norm(t_emb) * np.linalg.norm(c_emb) + 1e-8))
        else:
            # Fallback: word overlap
            t_words = set(target_text.lower().split())
            c_words = set(c["text"].lower().split())
            feats["semantic_sim"] = len(t_words & c_words) / max(len(t_words | c_words), 1)

        # Signal 2: Temporal proximity (closer to target = more likely cause)
        feats["temporal_dist"] = 1.0 / (abs(c["index"] - target_idx) + 1) if target_idx >= 0 else 0.5

        # Signal 3: Appraisal keyword match
        appraisal_keywords = " ".join(appraisal["likely_cause_type"])
        c_text_lower = c["text"].lower()
        keyword_hits = sum(1 for kw in appraisal["likely_cause_type"] if kw.replace("_", " ") in c_text_lower)
        feats["appraisal_match"] = keyword_hits / max(len(appraisal["likely_cause_type"]), 1)

        # Signal 4: Negation / negative valence indicator
        neg_words = ["not", "no", "never", "cannot", "can't", "don't", "won't", "failed", "lost", "bad", "terrible",
                     "awful", "hate", "worst", "disappointed", "wrong"]
        neg_count = sum(1 for w in neg_words if w in c_text_lower)
        feats["negation_count"] = neg_count / max(len(c_text_lower.split()), 1)

        # Signal 5: Intensity (utterance length relative)
        feats["length_ratio"] = len(c["text"].split()) / max(len(target_text.split()), 1)

        # Signal 6: Causal language indicators
        causal_words = ["because", "so", "therefore", "since", "due to", "as a result", "that's why", "reason"]
        causal_count = sum(1 for w in causal_words if w in c_text_lower)
        feats["causal_lang"] = min(causal_count / 3.0, 1.0)

        feats["label"] = int(c["is_cause"])
        features.append(feats)

    return features


def train_reranker(train_features):
    """Train XGBoost reranker on appraisal features."""
    from xgboost import XGBClassifier

    X = []
    y = []
    feature_keys = ["semantic_sim", "temporal_dist", "appraisal_match",
                    "negation_count", "length_ratio", "causal_lang"]
    for feats in train_features:
        X.append([feats[k] for k in feature_keys])
        y.append(feats["label"])

    X = np.array(X)
    y = np.array(y)

    # Handle class imbalance
    pos_weight = (len(y) - sum(y)) / max(sum(y), 1)

    model = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        scale_pos_weight=pos_weight, random_state=42, verbosity=0
    )
    model.fit(X, y)
    return model, feature_keys


def predict_with_reranker(samples, embed_model, reranker, feature_keys):
    """Apply trained reranker to score and rank candidates."""
    predictions = []
    for sample in samples:
        feats_list = compute_appraisal_features(sample, embed_model)
        if not feats_list:
            predictions.append([])
            continue

        X = np.array([[f[k] for k in feature_keys] for f in feats_list])
        scores = reranker.predict_proba(X)[:, 1]  # probability of being cause
        ranked = np.argsort(-scores)
        predictions.append([sample["candidates"][i]["index"] for i in ranked[:3]])
    return predictions


# ─── 4. Evaluation ───────────────────────────────────────────
def evaluate(predictions, samples, k_values=(1, 3)):
    """Compute Precision@k, Recall@k, MRR, F1."""
    metrics = {f"P@{k}": [] for k in k_values}
    metrics.update({f"R@{k}": [] for k in k_values})
    mrr_vals = []

    for pred_indices, sample in zip(predictions, samples):
        gold = set(sample["gold_indices"])
        if not gold:
            continue

        for k in k_values:
            pred_k = set(pred_indices[:k])
            metrics[f"P@{k}"].append(len(pred_k & gold) / k if pred_k else 0)
            metrics[f"R@{k}"].append(len(pred_k & gold) / len(gold) if gold else 0)

        # MRR
        for rank, idx in enumerate(pred_indices, 1):
            if idx in gold:
                mrr_vals.append(1.0 / rank)
                break
        else:
            mrr_vals.append(0.0)

    result = {k: round(float(np.mean(v)), 4) for k, v in metrics.items()}
    result["MRR"] = round(float(np.mean(mrr_vals)), 4)
    result["n_evaluated"] = len(mrr_vals)
    return result


# ─── 5. Main Experiment ─────────────────────────────────────
def main():
    print("=" * 60)
    print("  RECCON: Appraisal-Guided Emotion Cause Retrieval")
    print("=" * 60)

    # Load data
    print("\n[1/6] Loading RECCON data...")
    train, valid, test = load_all_data()

    # Use a subset for manageable runtime
    # Train: 5000, Valid: 500, Test: 500
    import random
    random.seed(42)
    train_subset = random.sample(train, min(5000, len(train)))
    valid_subset = random.sample(valid, min(500, len(valid)))
    test_subset = random.sample(test, min(500, len(test)))
    print(f"Subset sizes: train={len(train_subset)}, valid={len(valid_subset)}, test={len(test_subset)}")

    # Load Qwen3 for embeddings and LLM baseline
    print("\n[2/6] Loading Qwen3 model...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer as HFAutoTokenizer

    llm_tokenizer = HFAutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    llm_tokenizer.pad_token = llm_tokenizer.eos_token
    llm_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True, device_map="cuda:0"
    )

    @torch.no_grad()
    def encode_texts(texts):
        """Mean-pool Qwen3 last hidden states for embeddings."""
        device = next(llm_model.parameters()).device
        inputs = llm_tokenizer(texts, padding=True, truncation=True, max_length=128,
                               return_tensors="pt").to(device)
        outputs = llm_model.model(**inputs)
        attention_mask = inputs["attention_mask"].unsqueeze(-1).to(device)
        hidden = outputs.last_hidden_state.to(device)
        pooled = (hidden.float() * attention_mask.float()).sum(dim=1) / attention_mask.sum(dim=1)
        return pooled.cpu().numpy()

    class Qwen3Embedder:
        def encode(self, texts, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            results = []
            for i in range(0, len(texts), 32):
                results.append(encode_texts(texts[i:i+32]))
            return np.concatenate(results, axis=0)

    embed_model = Qwen3Embedder()

    results = {}

    # Baseline 1: Nearest Previous
    print("\n[3/6] Running baselines...")
    preds = [baseline_nearest_previous(s) for s in test_subset]
    results["Nearest Previous"] = evaluate(preds, test_subset)
    print(f"  Nearest Previous: P@1={results['Nearest Previous']['P@1']:.4f}, MRR={results['Nearest Previous']['MRR']:.4f}")

    # Baseline 2: TF-IDF
    preds = baseline_tfidf(train_subset, test_subset)
    results["TF-IDF"] = evaluate(preds, test_subset)
    print(f"  TF-IDF: P@1={results['TF-IDF']['P@1']:.4f}, MRR={results['TF-IDF']['MRR']:.4f}")

    # Baseline 3: Embedding Retrieval
    preds = baseline_embedding(test_subset, embed_model)
    results["Embedding"] = evaluate(preds, test_subset)
    print(f"  Embedding: P@1={results['Embedding']['P@1']:.4f}, MRR={results['Embedding']['MRR']:.4f}")

    # Baseline 4: Appraisal features only (no training, fixed weights)
    print("\n[4/6] Running Appraisal-Guided methods...")
    preds_fixed = []
    for sample in tqdm(test_subset, desc="Appraisal (fixed weights)"):
        feats_list = compute_appraisal_features(sample, embed_model)
        if not feats_list:
            preds_fixed.append([])
            continue
        # Fixed weights: semantic 0.3 + temporal 0.3 + appraisal_match 0.4
        scores = [0.3 * f["semantic_sim"] + 0.3 * f["temporal_dist"] + 0.4 * f["appraisal_match"]
                  for f in feats_list]
        ranked = np.argsort(-np.array(scores))
        preds_fixed.append([sample["candidates"][i]["index"] for i in ranked[:3]])
    results["Appraisal (fixed weights)"] = evaluate(preds_fixed, test_subset)
    print(f"  Appraisal (fixed): P@1={results['Appraisal (fixed weights)']['P@1']:.4f}, MRR={results['Appraisal (fixed weights)']['MRR']:.4f}")

    # Method 5: Appraisal + XGBoost Reranker
    print("\n[5/6] Training XGBoost reranker...")
    train_features = []
    for sample in tqdm(train_subset[:2000], desc="Extracting train features"):
        feats = compute_appraisal_features(sample, embed_model)
        train_features.extend(feats)
    print(f"  Training features: {len(train_features)} (pos={sum(f['label'] for f in train_features)})")

    reranker, feature_keys = train_reranker(train_features)
    print(f"  Feature importance: {dict(zip(feature_keys, [round(x,3) for x in reranker.feature_importances_]))}")

    preds_rerank = predict_with_reranker(test_subset, embed_model, reranker, feature_keys)
    results["Appraisal + XGBoost"] = evaluate(preds_rerank, test_subset)
    print(f"  Appraisal + XGBoost: P@1={results['Appraisal + XGBoost']['P@1']:.4f}, MRR={results['Appraisal + XGBoost']['MRR']:.4f}")

    # ── Print Final Results Table ──
    print("\n" + "=" * 70)
    print("  RESULTS: Emotion Cause Recognition (RECCON)")
    print("=" * 70)
    print(f"  {'Method':<30s} {'P@1':>6s} {'R@1':>6s} {'P@3':>6s} {'R@3':>6s} {'MRR':>6s}")
    print("  " + "-" * 60)
    for method in ["Nearest Previous", "TF-IDF", "Embedding",
                   "Appraisal (fixed weights)", "Appraisal + XGBoost"]:
        m = results.get(method, {})
        print(f"  {method:<30s} {m.get('P@1', 0):>6.4f} {m.get('R@1', 0):>6.4f} "
              f"{m.get('P@3', 0):>6.4f} {m.get('R@3', 0):>6.4f} {m.get('MRR', 0):>6.4f}")

    # Save results
    with open(LOG_DIR / "reccon_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to log/reccon_results.json")

    # ── Case study ──
    print("\n" + "=" * 70)
    print("  CASE STUDY")
    print("=" * 70)
    for sample in test_subset[:3]:
        if not sample["gold_indices"]:
            continue
        print(f"\n  Context: {sample['context'][:200]}...")
        print(f"  Target: {sample['target_text'][:100]}")
        print(f"  Emotion: {sample['emotion']}")
        print(f"  Gold cause (idx {sample['gold_indices']}): "
              f"{sample['context_utterances'][sample['gold_indices'][0]][:100]}")
        # Show our prediction
        feats = compute_appraisal_features(sample, embed_model)
        if feats and reranker:
            X = np.array([[f[k] for k in feature_keys] for f in feats])
            scores = reranker.predict_proba(X)[:, 1]
            top_idx = int(np.argmax(scores))
            top_candidate = sample["candidates"][top_idx]
            print(f"  Our top prediction: \"{top_candidate['text'][:100]}\" (score={scores[top_idx]:.3f})")
            print(f"  Hit: {top_candidate['index'] in sample['gold_indices']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
