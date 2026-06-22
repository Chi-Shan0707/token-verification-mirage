#!/usr/bin/env python3
"""Per-method direction-flip rates for BigMath Hard problems."""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
import iisignature

QWEN_JSONL = "outputs/bigmath_traces/bigmath_qwencoder_traces.jsonl"
LLAMA_JSONL = "data/bigmath_llama_traces/bigmath_llama3_8b_traces.jsonl"
QWEN_NPZ_DIR = Path("outputs/bigmath_traces/bigmath_qwencoder_traces_npz")
LLAMA_NPZ_DIR = Path("data/bigmath_llama_traces/bigmath_llama3_8b_traces_npz")


def load_traces(jsonl_path, npz_dir):
    """Load traces, group by hard problem_id, return eligible problems."""
    problems = defaultdict(list)
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            if d["difficulty_tier"] != "hard":
                continue
            problems[d["problem_id"]].append(d)

    eligible = {}
    for pid, traces in problems.items():
        n_correct = sum(t["is_correct"] for t in traces)
        n_wrong = len(traces) - n_correct
        if n_correct >= 2 and n_wrong >= 2:
            eligible[pid] = traces
    return eligible


def compute_entropy_slope(entropy):
    """Linear regression slope of entropy over time."""
    x = np.arange(len(entropy), dtype=np.float64).reshape(-1, 1)
    y = entropy.astype(np.float64).reshape(-1, 1)
    reg = LinearRegression().fit(x, y)
    return float(reg.coef_[0, 0])


def compute_signature_2d(entropy):
    """Depth-3 signature of [t, entropy] path."""
    t = np.arange(len(entropy), dtype=np.float64)
    path = np.stack([t, entropy.astype(np.float64)], axis=1)
    sig = iisignature.sig(path, 3)
    return sig


def compute_loo_centroid_distance(sigs, labels):
    """LOO centroid distance for signature method."""
    n = len(sigs)
    sigs = np.array(sigs)
    correct_idx = [i for i in range(n) if labels[i]]
    wrong_idx = [i for i in range(n) if not labels[i]]

    if len(correct_idx) < 2 or len(wrong_idx) < 2:
        return None

    centroid_correct = np.mean(sigs[correct_idx], axis=0)
    centroid_wrong = np.mean(sigs[wrong_idx], axis=0)

    scores = np.empty(n)
    for i in range(n):
        if labels[i]:
            loo_centroid = (centroid_correct * len(correct_idx) - sigs[i]) / (len(correct_idx) - 1)
        else:
            loo_centroid = (centroid_wrong * len(wrong_idx) - sigs[i]) / (len(wrong_idx) - 1)
        d_correct = np.linalg.norm(sigs[i] - loo_centroid)
        d_wrong = np.linalg.norm(sigs[i] - (centroid_wrong if labels[i] else centroid_correct))
        scores[i] = d_wrong - d_correct

    return scores


def compute_per_problem_auroc(pid, traces, npz_dir):
    """Compute AUROC for all 6 methods on a single problem."""
    labels = []
    entropy_arrays = []
    npz_loaded = {}

    for t in traces:
        labels.append(t["is_correct"])
        npz_path = npz_dir / Path(t["npz_path"]).name
        data = np.load(npz_path)
        entropy = data["tok_entropy"].astype(np.float64)
        entropy_arrays.append(entropy)
        npz_loaded[t["run_id"]] = data

    labels = np.array(labels, dtype=bool)
    n = len(labels)
    n_correct = labels.sum()
    n_wrong = n - n_correct

    if n_correct < 2 or n_wrong < 2:
        return None

    results = {}

    # 1. Final-token entropy
    scores = np.array([e[-1] for e in entropy_arrays])
    results["Final-token entropy"] = roc_auc_score(labels, scores)

    # 2. Entropy mean
    scores = np.array([np.mean(e) for e in entropy_arrays])
    results["Entropy mean"] = roc_auc_score(labels, scores)

    # 3. Entropy std
    scores = np.array([np.std(e) for e in entropy_arrays])
    results["Entropy std"] = roc_auc_score(labels, scores)

    # 4. Length
    scores = np.array([len(e) for e in entropy_arrays], dtype=np.float64)
    results["Length"] = roc_auc_score(labels, scores)

    # 5. Entropy slope
    slopes = [compute_entropy_slope(e) for e in entropy_arrays]
    scores = np.array(slopes)
    results["Entropy slope"] = roc_auc_score(labels, scores)

    # 6. Signature 2D (depth-3, centroid-distance LOO)
    sigs = [compute_signature_2d(e) for e in entropy_arrays]
    loo_scores = compute_loo_centroid_distance(sigs, labels)
    if loo_scores is not None:
        results["Signature 2D"] = roc_auc_score(labels, loo_scores)
    else:
        results["Signature 2D"] = None

    return results


def main():
    print("Loading traces...")
    qwen_problems = load_traces(QWEN_JSONL, QWEN_NPZ_DIR)
    llama_problems = load_traces(LLAMA_JSONL, LLAMA_NPZ_DIR)

    print(f"Qwen eligible: {len(qwen_problems)} problems")
    print(f"Llama eligible: {len(llama_problems)} problems")

    methods = [
        "Final-token entropy",
        "Entropy mean",
        "Entropy std",
        "Length",
        "Entropy slope",
        "Signature 2D",
    ]

    # Collect per-problem AUROCs
    qwen_aurocs = {m: [] for m in methods}
    llama_aurocs = {m: [] for m in methods}

    print("\nComputing Qwen per-problem AUROCs...")
    for pid, traces in qwen_problems.items():
        result = compute_per_problem_auroc(pid, traces, QWEN_NPZ_DIR)
        if result:
            for m in methods:
                if result[m] is not None:
                    qwen_aurocs[m].append(result[m])

    print("Computing Llama per-problem AUROCs...")
    for pid, traces in llama_problems.items():
        result = compute_per_problem_auroc(pid, traces, LLAMA_NPZ_DIR)
        if result:
            for m in methods:
                if result[m] is not None:
                    llama_aurocs[m].append(result[m])

    # Compute flip rates and medians
    print("\n" + "=" * 90)
    print(f"{'Method':<22} | {'Flip Rate (Q/L)':<18} | {'Raw Median (Q/L)':<20} | {'DA Median (Q/L)':<20}")
    print("-" * 90)

    for m in methods:
        q_vals = np.array(qwen_aurocs[m])
        l_vals = np.array(llama_aurocs[m])

        if len(q_vals) == 0 or len(l_vals) == 0:
            print(f"{m:<22} | INSUFFICIENT DATA")
            continue

        # Flip rate = fraction with AUROC < 0.5
        q_flip = np.mean(q_vals < 0.5)
        l_flip = np.mean(l_vals < 0.5)

        # Raw median
        q_raw_med = np.median(q_vals)
        l_raw_med = np.median(l_vals)

        # Direction-agnostic: map AUROC < 0.5 to 1 - AUROC
        q_da = np.where(q_vals < 0.5, 1 - q_vals, q_vals)
        l_da = np.where(l_vals < 0.5, 1 - l_vals, l_vals)
        q_da_med = np.median(q_da)
        l_da_med = np.median(l_da)

        n_q = len(q_vals)
        n_l = len(l_vals)

        print(
            f"{m:<22} | {q_flip:.2f}/{l_flip:.2f}  ({n_q}/{n_l})  "
            f"| {q_raw_med:.4f}/{l_raw_med:.4f}        "
            f"| {q_da_med:.4f}/{l_da_med:.4f}"
        )

    print("=" * 90)
    print("\nFlip Rate = fraction of problems where raw AUROC < 0.5")
    print("DA = Direction-Agnostic (AUROC < 0.5 mapped to 1 - AUROC)")
    print("Numbers in parentheses = problem count (Qwen/Llama)")


if __name__ == "__main__":
    main()
