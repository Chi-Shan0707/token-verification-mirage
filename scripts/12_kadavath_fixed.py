#!/usr/bin/env python3
"""Kadavath et al. (2022) — P(True) Self-Evaluation, FIXED VERSION.

Key fix: Do NOT show the ground truth to the model. The model must
evaluate its own answer without knowing the correct answer.

Two variants:
A) Zero-shot: model evaluates its own proposed answer directly
B) Few-shot with brainstorming: show other samples from same problem first
"""

import json
import os
import re
import time
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

QWEN_JSONL = "outputs/bigmath_merged64/bigmath_qwen_64run.jsonl"
MODEL_PATH = "data/models/qwen2.5-coder-7b"
SAVE_DIR = "outputs/baseline_repro"


def extract_boxed_answer(text):
    for pat in [r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
                r'boxed\{([^}]+)\}']:
        matches = re.findall(pat, text)
        if matches:
            return re.sub(r'[,\s]+', '', matches[-1].strip())
    return None


def extract_final_numeric(text):
    nums = re.findall(r'-?\d+\.?\d*', text)
    return nums[-1] if nums else None


def load_hard_problems(jsonl_path):
    problems = defaultdict(list)
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            problems[rec["problem_id"]].append(rec)
    for pid in problems:
        problems[pid].sort(key=lambda r: r["run_id"])
    result = {}
    for pid, recs in problems.items():
        if recs[0]["difficulty_tier"] != "hard":
            continue
        labels = np.array([r["is_correct"] for r in recs])
        if labels.sum() < 2 or (len(labels) - labels.sum()) < 2:
            continue
        result[pid] = {"labels": labels, "recs": recs}
    return result


def loo_auroc_scalar(values, labels):
    n = len(values)
    c_idx = np.where(labels)[0]
    w_idx = np.where(~labels)[0]
    if len(c_idx) < 2 or len(w_idx) < 2:
        return np.nan, np.nan
    scores = np.empty(n)
    for hold in range(n):
        mask = np.ones(n, dtype=bool)
        mask[hold] = False
        cm = values[mask & labels].mean()
        wm = values[mask & (~labels)].mean()
        scores[hold] = abs(values[hold] - wm) - abs(values[hold] - cm)
    raw = roc_auc_score(labels, scores)
    return raw, max(raw, 1.0 - raw)


def bootstrap_median_ci(data, n_boot=10000, ci=0.95):
    if len(data) < 2:
        return np.nan, (np.nan, np.nan)
    rng = np.random.default_rng(42)
    meds = np.empty(n_boot)
    for b in range(n_boot):
        meds[b] = np.median(rng.choice(data, size=len(data), replace=True))
    lo = np.percentile(meds, (1 - ci) / 2 * 100)
    hi = np.percentile(meds, (1 + ci) / 2 * 100)
    return np.median(data), (lo, hi)


def main():
    from vllm import LLM, SamplingParams

    t0 = time.time()
    
    print("Loading Qwen2.5-Coder-7B (TP=2)...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=2,
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=0.9,
        disable_log_stats=True,
    )
    print(f"Model loaded in {time.time()-t0:.1f}s")

    problems = load_hard_problems(QWEN_JSONL)
    print(f"  {len(problems)} hard WP-eligible problems")

    sp = SamplingParams(temperature=0.0, max_tokens=10, n=1)

    # ============================================================
    # Variant A: Zero-shot, no ground truth
    # ============================================================
    print("\n=== Variant A: Zero-shot P(True), NO ground truth ===")
    
    prompts_a = []
    meta_a = []
    for pid, pdata in problems.items():
        for run_idx, rec in enumerate(pdata["recs"]):
            text = rec["generated_text"]
            proposed = extract_boxed_answer(text)
            if proposed is None:
                proposed = extract_final_numeric(text) or "unknown"
            
            last_part = text[-600:] if len(text) > 600 else text
            
            prompt = f"""Below is a mathematical reasoning trace followed by a proposed answer.

Reasoning (last part):
...{last_part}

Proposed answer: {proposed}

Is the proposed answer correct? Answer only "True" or "False".

Answer:"""
            prompts_a.append(prompt)
            meta_a.append((pid, run_idx))

    print(f"  Generating {len(prompts_a)} prompts...")
    outputs_a = llm.generate(prompts_a, sp)
    
    scores_a = defaultdict(lambda: {"scores": {}, "labels": {}})
    for i, out in enumerate(outputs_a):
        pid, run_idx = meta_a[i]
        resp = out.outputs[0].text.strip().lower()
        if "true" in resp and "false" not in resp:
            score = 1.0
        elif "false" in resp:
            score = 0.0
        else:
            score = 0.5
        scores_a[pid]["scores"][run_idx] = score
        scores_a[pid]["labels"][run_idx] = problems[pid]["labels"][run_idx]

    pairs_a = []
    for pid in scores_a:
        s = np.array([v for _, v in sorted(scores_a[pid]["scores"].items())])
        l = np.array([v for _, v in sorted(scores_a[pid]["labels"].items())])
        if len(l) >= 4 and len(set(l)) >= 2 and np.std(s) > 0:
            raw, da = loo_auroc_scalar(s, l)
            if not np.isnan(da):
                pairs_a.append((raw, da))

    if pairs_a:
        raws = np.array([p[0] for p in pairs_a])
        das = np.array([p[1] for p in pairs_a])
        med, (lo, hi) = bootstrap_median_ci(das)
        flip = np.mean(raws < 0.5)
        print(f"  Kadavath-P(True)-zero-shot: DA={med:.3f} [{lo:.3f},{hi:.3f}] flip={flip*100:.1f}% n={len(das)}")

    # ============================================================
    # Variant B: Few-shot with brainstorming (other runs as context)
    # ============================================================
    print("\n=== Variant B: Few-shot P(True) with brainstorming ===")
    
    prompts_b = []
    meta_b = []
    for pid, pdata in problems.items():
        recs = pdata["recs"]
        for run_idx, rec in enumerate(recs):
            text = rec["generated_text"]
            proposed = extract_boxed_answer(text)
            if proposed is None:
                proposed = extract_final_numeric(text) or "unknown"
            
            # Pick 3 other runs as brainstorming context (avoid the target itself)
            other_indices = [j for j in range(len(recs)) if j != run_idx]
            np.random.seed(42 + hash(pid) % 10000)
            brainstorm_idx = np.random.choice(other_indices, min(3, len(other_indices)), replace=False)
            
            brainstorm_text = ""
            for bi, bidx in enumerate(brainstorm_idx):
                bans = extract_boxed_answer(recs[bidx]["generated_text"])
                if bans is None:
                    bans = extract_final_numeric(recs[bidx]["generated_text"]) or "?"
                brainstorm_text += f"{bi+1}. {bans}\n"
            
            prompt = f"""Here are some proposed answers to a math problem:
{brainstorm_text}
Now evaluate this proposed answer: {proposed}

Is this proposed answer correct? Answer only "True" or "False".

Answer:"""
            prompts_b.append(prompt)
            meta_b.append((pid, run_idx))

    print(f"  Generating {len(prompts_b)} prompts...")
    outputs_b = llm.generate(prompts_b, sp)

    scores_b = defaultdict(lambda: {"scores": {}, "labels": {}})
    for i, out in enumerate(outputs_b):
        pid, run_idx = meta_b[i]
        resp = out.outputs[0].text.strip().lower()
        if "true" in resp and "false" not in resp:
            score = 1.0
        elif "false" in resp:
            score = 0.0
        else:
            score = 0.5
        scores_b[pid]["scores"][run_idx] = score
        scores_b[pid]["labels"][run_idx] = problems[pid]["labels"][run_idx]

    pairs_b = []
    for pid in scores_b:
        s = np.array([v for _, v in sorted(scores_b[pid]["scores"].items())])
        l = np.array([v for _, v in sorted(scores_b[pid]["labels"].items())])
        if len(l) >= 4 and len(set(l)) >= 2 and np.std(s) > 0:
            raw, da = loo_auroc_scalar(s, l)
            if not np.isnan(da):
                pairs_b.append((raw, da))

    if pairs_b:
        raws = np.array([p[0] for p in pairs_b])
        das = np.array([p[1] for p in pairs_b])
        med, (lo, hi) = bootstrap_median_ci(das)
        flip = np.mean(raws < 0.5)
        print(f"  Kadavath-P(True)-few-shot:  DA={med:.3f} [{lo:.3f},{hi:.3f}] flip={flip*100:.1f}% n={len(das)}")

    # Save
    save_path = os.path.join(SAVE_DIR, "kadavath_fixed_results.json")
    with open(save_path, "w") as f:
        json.dump({
            "zero_shot": pairs_a,
            "few_shot": pairs_b,
            "total_time_s": time.time() - t0,
        }, f, indent=2)
    print(f"\nSaved to {save_path}")
    print(f"Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
