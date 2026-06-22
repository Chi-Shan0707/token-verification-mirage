#!/usr/bin/env python3
"""Baseline reproduction: Malinin & Gales (2021) token-level uncertainty.

Computes Total Uncertainty (entropy), Data Uncertainty (avg conditional entropy),
Knowledge Uncertainty (MI = Total - Data), and Reverse Mutual Information (RMI)
from top-k logprobs already stored in NPZ files.

Uses 64 runs per problem (within-problem) as the "ensemble".
Applies strict LOO protocol from our paper.
"""

import json
import os
import sys
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

VOCAB_LOG_BASE = np.e


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


def compute_malinin_gales_scores_for_problem(args):
    pid, pdata = args

    labels = pdata["labels"]
    recs = pdata["recs"]
    n_runs = len(recs)

    all_topk_logprobs = []
    all_tok_entropy = []
    seq_lengths = []

    for rec in recs:
        npz = np.load(rec["npz_path"], allow_pickle=True)
        topk_lp = npz["topk_logprobs"].astype(np.float64)
        tok_ent = npz["tok_entropy"].astype(np.float64)
        all_topk_logprobs.append(topk_lp)
        all_tok_entropy.append(tok_ent)
        seq_lengths.append(len(tok_ent))

    min_len = min(seq_lengths)
    if min_len < 5:
        return pid, {k: (np.nan, np.nan) for k in [
            "MG-TotalUnc", "MG-DataUnc", "MG-KnowledgeUnc(MI)", "MG-RMI",
            "MG-TotalUnc-mc", "MG-KnowledgeUnc-mc"
        ]}

    # Align all sequences to min_len (truncate from start to keep answer region)
    aligned_topk = np.zeros((n_runs, min_len, 20))
    aligned_entropy = np.zeros((n_runs, min_len))
    for i in range(n_runs):
        start = seq_lengths[i] - min_len
        aligned_topk[i] = all_topk_logprobs[i][start:start+min_len]
        aligned_entropy[i] = all_tok_entropy[i][start:start+min_len]

    # --- Method A: Posterior entropy from top-k approximation ---
    # Total Uncertainty = H[P(y_l | prefix, D)] where D = all 64 runs
    # Approximate posterior as mean of per-run softmax over top-k

    # Convert logprobs to probabilities (per run)
    run_probs = np.zeros((n_runs, min_len, 20))
    for i in range(n_runs):
        for t in range(min_len):
            lp = aligned_topk[i, t]
            lp_shifted = lp - lp.max()
            run_probs[i, t] = np.exp(lp_shifted)
            run_probs[i, t] /= run_probs[i, t].sum()

    # Ensemble posterior = average of per-run distributions
    posterior = run_probs.mean(axis=0)  # (min_len, 20)

    # Total Uncertainty: H[predictive posterior] per token
    eps = 1e-12
    total_unc_per_tok = -np.sum(posterior * np.log(posterior + eps), axis=1)  # (min_len,)

    # Data Uncertainty: E[H[p(y|prefix, theta_m)]]
    per_run_entropy = -np.sum(run_probs * np.log(run_probs + eps), axis=2)  # (n_runs, min_len)
    data_unc_per_tok = per_run_entropy.mean(axis=0)  # (min_len,)

    # Knowledge Uncertainty (MI) = Total - Data
    knowledge_unc_per_tok = total_unc_per_tok - data_unc_per_tok  # (min_len,)

    # RMI: (1/M) * sum_m KL(posterior || p(y|prefix, theta_m))
    rmi_per_tok = np.zeros(min_len)
    for i in range(n_runs):
        kl = np.sum(posterior * (np.log(posterior + eps) - np.log(run_probs[i] + eps)), axis=1)
        rmi_per_tok += kl
    rmi_per_tok /= n_runs

    # Aggregate to sequence-level scores (length-normalized mean)
    def seq_score(arr):
        return float(np.mean(arr))

    scores_all_runs = {
        "MG-TotalUnc": np.array([seq_score(total_unc_per_tok)] * n_runs),
        "MG-DataUnc": np.array([seq_score(data_unc_per_tok)] * n_runs),
        "MG-KnowledgeUnc(MI)": np.array([seq_score(knowledge_unc_per_tok)] * n_runs),
        "MG-RMI": np.array([seq_score(rmi_per_tok)] * n_runs),
    }

    # --- Method B: Per-run score using MC agreement ---
    # For each run i, use its own entropy trajectory as the "observed" signal
    # Score = total uncertainty from the ensemble minus the run's own conditional entropy
    # This gives a per-run discriminative score

    # Per-run total uncertainty score (higher = more uncertain = likely wrong)
    per_run_total = np.array([seq_score(aligned_entropy[i]) for i in range(n_runs)])

    # Per-run knowledge uncertainty: use leave-one-out posterior
    per_run_knowledge_loo = np.zeros(n_runs)
    per_run_rmi_loo = np.zeros(n_runs)
    for hold in range(n_runs):
        mask = np.ones(n_runs, dtype=bool)
        mask[hold] = False
        post_loo = run_probs[mask].mean(axis=0)  # (min_len, 20)

        # Total unc from LOO posterior
        total_loo = -np.sum(post_loo * np.log(post_loo + eps), axis=1)

        # Data unc from remaining runs
        data_loo = per_run_entropy[mask].mean(axis=0)

        # Knowledge unc (MI)
        knowledge_loo = total_loo - data_loo
        per_run_knowledge_loo[hold] = seq_score(knowledge_loo)

        # RMI from LOO posterior
        rmi_loo = np.zeros(min_len)
        for j in np.where(mask)[0]:
            kl = np.sum(post_loo * (np.log(post_loo + eps) - np.log(run_probs[j] + eps)), axis=1)
            rmi_loo += kl
        rmi_loo /= mask.sum()
        per_run_rmi_loo[hold] = seq_score(rmi_loo)

    scores_all_runs["MG-TotalUnc-mc"] = per_run_total
    scores_all_runs["MG-KnowledgeUnc-mc"] = per_run_knowledge_loo

    # --- LOO AUROC for each method ---
    results = {}
    correct_idx = np.where(labels)[0]
    wrong_idx = np.where(~labels)[0]

    for method_name, scores in scores_all_runs.items():
        if np.all(np.isnan(scores)):
            results[method_name] = (np.nan, np.nan)
            continue

        # For uncertainty methods, higher = more uncertain = likely wrong
        # So we need direction-agnostic scoring
        raw = roc_auc_score(labels, scores)
        da = max(raw, 1.0 - raw)
        results[method_name] = (raw, da)

    return pid, results


def bootstrap_median_ci(data, n_boot=10000, ci=0.95):
    if len(data) < 2:
        return np.nan, (np.nan, np.nan)
    rng = np.random.default_rng(42)
    medians = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(data, size=len(data), replace=True)
        medians[b] = np.median(sample)
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
        print(f"Malinin & Gales (2021) Baseline — {model_name}")
        print(f"{'='*70}")

        problems = load_traces(jsonl_path, npz_dir, tier_filter="hard")
        print(f"  {len(problems)} hard WP-eligible problems")

        work = [(pid, pdata) for pid, pdata in problems.items()]

        with Pool() as pool:
            all_results = pool.map(compute_malinin_gales_scores_for_problem, work, chunksize=4)

        method_names = [
            "MG-TotalUnc", "MG-DataUnc", "MG-KnowledgeUnc(MI)", "MG-RMI",
            "MG-TotalUnc-mc", "MG-KnowledgeUnc-mc"
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

        elapsed = time.time() - t0
        print(f"\n  Time: {elapsed:.1f}s")

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
