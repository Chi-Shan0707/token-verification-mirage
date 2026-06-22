#!/usr/bin/env python3
"""Baseline reproduction: Malinin & Gales (2021) — Corrected per-run LOO scoring.

Key insight: the ensemble-level scores (Total/Data/MI/RMI) are identical for all
runs within a problem, so they cannot discriminate. The discriminative signal
comes from per-run deviation from the ensemble consensus.

We compute per-run "surprisal" scores: how much does this run's token-level
uncertainty deviate from what the ensemble predicts?

Three per-run scoring strategies:
1. Per-run mean entropy (simple scalar baseline from their framework)
2. Per-run entropy deviation from ensemble mean entropy trajectory  
3. LOO ensemble divergence: hold out this run, compute ensemble posterior,
   then score = KL(run_distribution || LOO_posterior) aggregated over tokens
"""

import json
import os
import time
import numpy as np
from collections import defaultdict
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

QWEN_JSONL = "outputs/bigmath_merged64/bigmath_qwen_64run.jsonl"
QWEN_NPZ_DIR = "outputs/bigmath_merged64/bigmath_qwen_64run_npz/"
LLAMA_JSONL = "outputs/bigmath_merged64/bigmath_llama_64run.jsonl"
LLAMA_NPZ_DIR = "outputs/bigmath_merged64/bigmath_llama_64run_npz/"


def load_traces(jsonl_path, npz_dir, tier_filter="hard"):
    problems = defaultdict(list)
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            pid = rec["problem_id"]
            npz_path = os.path.join(npz_dir, f"{pid}_run{rec['run_id']:03d}.npz")
            rec["npz_path"] = npz_path
            problems[pid].append(rec)
    for pid in problems:
        problems[pid].sort(key=lambda r: r["run_id"])
    result = {}
    for pid, recs in problems.items():
        tier = recs[0]["difficulty_tier"]
        if tier != tier_filter:
            continue
        labels = np.array([r["is_correct"] for r in recs])
        n_correct = labels.sum()
        n_wrong = len(labels) - n_correct
        if n_correct < 2 or n_wrong < 2:
            continue
        result[pid] = {"tier": tier, "labels": labels, "recs": recs}
    return result


def compute_mg_loo_scores(args):
    pid, pdata = args
    labels = pdata["labels"]
    recs = pdata["recs"]
    n_runs = len(recs)
    eps = 1e-12

    all_topk_logprobs = []
    all_tok_entropy = []
    seq_lengths = []
    for rec in recs:
        npz = np.load(rec["npz_path"], allow_pickle=True)
        all_topk_logprobs.append(npz["topk_logprobs"].astype(np.float64))
        all_tok_entropy.append(npz["tok_entropy"].astype(np.float64))
        seq_lengths.append(len(all_tok_entropy[-1]))

    min_len = min(seq_lengths)
    if min_len < 5:
        return pid, {}

    # Align to last min_len tokens
    aligned_topk = np.zeros((n_runs, min_len, 20))
    aligned_entropy = np.zeros((n_runs, min_len))
    for i in range(n_runs):
        start = seq_lengths[i] - min_len
        aligned_topk[i] = all_topk_logprobs[i][start:start+min_len]
        aligned_entropy[i] = all_tok_entropy[i][start:start+min_len]

    # Convert logprobs to probabilities per run
    run_probs = np.zeros((n_runs, min_len, 20))
    for i in range(n_runs):
        for t in range(min_len):
            lp = aligned_topk[i, t]
            lp_shifted = lp - lp.max()
            run_probs[i, t] = np.exp(lp_shifted)
            run_probs[i, t] /= run_probs[i, t].sum()

    # --- Score 1: Per-run mean entropy (MG-style scalar) ---
    score_mean_entropy = np.array([float(aligned_entropy[i].mean()) for i in range(n_runs)])

    # --- Score 2: Per-run final entropy (MG-style positional scalar) ---
    score_final_entropy = np.array([float(aligned_entropy[i, -1]) for i in range(n_runs)])

    # --- Score 3: LOO posterior KL divergence ---
    # For each held-out run, compute KL(run_distribution || LOO_posterior) at each token
    score_kl_loo = np.zeros(n_runs)
    score_kl_correct = np.zeros(n_runs)

    correct_idx = np.where(labels)[0]
    wrong_idx = np.where(~labels)[0]

    for hold in range(n_runs):
        mask = np.ones(n_runs, dtype=bool)
        mask[hold] = False

        # LOO ensemble posterior
        post_loo = run_probs[mask].mean(axis=0)  # (min_len, 20)

        # KL(posterior || run_distribution) — measures how "surprising" this run is
        kl = np.sum(run_probs[hold] * (np.log(run_probs[hold] + eps) - np.log(post_loo + eps)), axis=1)
        score_kl_loo[hold] = float(np.mean(kl))

        # Also: distance to correct vs wrong centroid (LOO)
        rem_probs = run_probs[mask]
        rem_labels = labels[mask]

        c_probs = rem_probs[rem_labels].mean(axis=0)
        w_probs = rem_probs[~rem_labels].mean(axis=0)

        dist_w = np.mean(np.abs(run_probs[hold] - w_probs))
        dist_c = np.mean(np.abs(run_probs[hold] - c_probs))
        score_kl_correct[hold] = dist_w - dist_c

    # --- Score 4: Token-level MI deviation (per-run knowledge uncertainty) ---
    # For each run, compute how much its token entropy deviates from
    # the "expected" entropy under the ensemble posterior
    ensemble_posterior = run_probs.mean(axis=0)
    expected_entropy = -np.sum(ensemble_posterior * np.log(ensemble_posterior + eps), axis=1)

    score_mi_deviation = np.zeros(n_runs)
    for i in range(n_runs):
        actual_entropy = aligned_entropy[i]
        deviation = actual_entropy - expected_entropy
        score_mi_deviation[i] = float(np.mean(deviation))

    # --- Score 5: Sequence-level MI with LOO ---
    # Hold out one run, compute ensemble posterior, get total/data/knowledge unc
    # Then score held-out run by its "knowledge uncertainty surprise"
    score_knowledge_surprise = np.zeros(n_runs)
    for hold in range(n_runs):
        mask = np.ones(n_runs, dtype=bool)
        mask[hold] = False
        post_loo = run_probs[mask].mean(axis=0)

        # Total uncertainty from LOO posterior
        total_unc = -np.sum(post_loo * np.log(post_loo + eps), axis=1)

        # Data uncertainty from remaining runs
        data_unc = np.zeros(min_len)
        for j in np.where(mask)[0]:
            data_unc += -np.sum(run_probs[j] * np.log(run_probs[j] + eps), axis=1)
        data_unc /= mask.sum()

        # Knowledge unc = total - data
        knowledge_unc = total_unc - data_unc

        # Surprise: how much does the held-out run's entropy exceed the knowledge unc?
        score_knowledge_surprise[hold] = float(np.mean(aligned_entropy[hold] - knowledge_unc))

    # --- Compute LOO AUROC for each scoring method ---
    method_scores = {
        "MG-MeanEntropy": score_mean_entropy,
        "MG-FinalEntropy": score_final_entropy,
        "MG-LOO-KL": score_kl_loo,
        "MG-LOO-CentroidDist": score_kl_correct,
        "MG-MIDeviation": score_mi_deviation,
        "MG-KnowledgeSurprise": score_knowledge_surprise,
    }

    results = {}
    for name, scores in method_scores.items():
        if np.any(np.isnan(scores)) or np.std(scores) < 1e-15:
            results[name] = (np.nan, np.nan)
            continue
        try:
            raw = roc_auc_score(labels, scores)
            da = max(raw, 1.0 - raw)
            results[name] = (raw, da)
        except Exception:
            results[name] = (np.nan, np.nan)

    return pid, results


def bootstrap_median_ci(data, n_boot=10000, ci=0.95):
    if len(data) < 2:
        return np.nan, (np.nan, np.nan)
    rng = np.random.default_rng(42)
    medians = np.empty(n_boot)
    for b in range(n_boot):
        medians[b] = np.median(rng.choice(data, size=len(data), replace=True))
    lo = np.percentile(medians, (1 - ci) / 2 * 100)
    hi = np.percentile(medians, (1 + ci) / 2 * 100)
    return np.median(data), (lo, hi)


def main():
    t0 = time.time()

    configs = [
        ("Qwen", QWEN_JSONL, QWEN_NPZ_DIR),
        ("Llama", LLAMA_JSONL, LLAMA_NPZ_DIR),
    ]

    for model_name, jsonl_path, npz_dir in configs:
        print(f"\n{'='*70}")
        print(f"Malinin & Gales (2021) LOO Baseline — {model_name} Hard")
        print(f"{'='*70}")

        problems = load_traces(jsonl_path, npz_dir, tier_filter="hard")
        print(f"  {len(problems)} hard WP-eligible problems")

        work = [(pid, pdata) for pid, pdata in problems.items()]

        with Pool() as pool:
            all_results = pool.map(compute_mg_loo_scores, work, chunksize=4)

        method_names = [
            "MG-MeanEntropy", "MG-FinalEntropy", "MG-LOO-KL",
            "MG-LOO-CentroidDist", "MG-MIDeviation", "MG-KnowledgeSurprise"
        ]

        tier_results = defaultdict(list)
        for pid, res in all_results:
            for mn in method_names:
                if mn in res and not np.isnan(res[mn][1]):
                    tier_results[mn].append(res[mn])

        print(f"\n{'Method':<28} | {'DA Median':>10} | {'95% CI':>24} | {'Flip Rate':>9} | {'Raw Median':>10}")
        print("-" * 95)

        for mn in method_names:
            pairs = tier_results.get(mn, [])
            if not pairs:
                print(f"{mn:<28} | {'N/A':>10}")
                continue
            raws = np.array([p[0] for p in pairs])
            das = np.array([p[1] for p in pairs])
            flip_rate = np.mean(raws < 0.5)
            med_da, (ci_lo, ci_hi) = bootstrap_median_ci(das)
            med_raw = np.median(raws)
            print(f"{mn:<28} | {med_da:>10.3f} | [{ci_lo:.3f}, {ci_hi:.3f}]     | {flip_rate*100:>6.1f}%    | {med_raw:>10.3f}")

        print(f"  Time: {time.time() - t0:.1f}s")

    print(f"\nTotal: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
