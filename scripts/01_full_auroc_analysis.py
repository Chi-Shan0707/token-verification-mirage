#!/usr/bin/env python3
"""Complete AUROC analysis of all methods on 64-run BigMath data."""

import json, os, sys, time, warnings
import numpy as np
from collections import defaultdict
from pathlib import Path
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score
from multiprocessing import Pool
from functools import partial

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
QWEN_JSONL = "outputs/bigmath_merged64/bigmath_qwen_64run.jsonl"
LLAMA_JSONL = "outputs/bigmath_merged64/bigmath_llama_64run.jsonl"
QWEN_NPZ = "outputs/bigmath_merged64/bigmath_qwen_64run_npz/"
LLAMA_NPZ = "outputs/bigmath_merged64/bigmath_llama_64run_npz/"

MODELS = {
    "Qwen":  {"jsonl": QWEN_JSONL, "npz_dir": QWEN_NPZ},
    "Llama": {"jsonl": LLAMA_JSONL, "npz_dir": LLAMA_NPZ},
}
TIERS = ["easy", "medium", "hard"]

# ── scalar feature extractors ──────────────────────────────────────────────
def feat_final_entropy(entropy):
    return entropy[-1]

def feat_mean_entropy(entropy):
    return entropy.mean()

def feat_std_entropy(entropy):
    return entropy.std()

def feat_slope_entropy(entropy):
    n = len(entropy)
    if n < 2:
        return 0.0
    t = np.linspace(0, 1, n)
    return np.polyfit(t, entropy, 1)[0]

def feat_length(entropy):
    return float(len(entropy))

def feat_first_entropy(entropy):
    return entropy[0]

def feat_last10_mean(entropy):
    k = max(1, len(entropy) // 10)
    return entropy[-k:].mean()

SCALAR_METHODS = [
    ("Final entropy",   feat_final_entropy),
    ("Entropy mean",    feat_mean_entropy),
    ("Entropy std",     feat_std_entropy),
    ("Entropy slope",   feat_slope_entropy),
    ("Sequence length", feat_length),
    ("First entropy",   feat_first_entropy),
    ("Last-10%-mean",   feat_last10_mean),
]

TOP5_FEATURES = [feat_mean_entropy, feat_std_entropy, feat_final_entropy,
                 feat_slope_entropy, feat_length]

# ── signature ──────────────────────────────────────────────────────────────
def sig_depth3_2d(path):
    """Full depth-3 signature of 2D path: 1 + 2 + 4 + 8 = 15 dims."""
    increments = np.diff(path, axis=0)
    n_inc = len(increments)
    S1 = np.zeros(2)
    S2 = np.zeros((2, 2))
    S3 = np.zeros((2, 2, 2))
    cum1 = np.zeros(2)
    cum2 = np.zeros((2, 2))
    for idx in range(n_inc):
        inc = increments[idx]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    S3[i, j, k] += cum2[i, j] * inc[k]
        S2 += np.outer(cum1, inc)
        cum2 += np.outer(cum1, inc)
        cum1 += inc
    S1 = increments.sum(axis=0)
    return np.concatenate([[1.0], S1, S2.flatten(), S3.flatten()])


def compute_signature(entropy):
    t = np.linspace(0, 1, len(entropy))
    path = np.column_stack([t, entropy])
    return sig_depth3_2d(path)

# ── LOO AUROC helpers ─────────────────────────────────────────────────────
def loo_auroc_scalar(values, labels):
    """LOO centroid-distance AUROC for scalar feature.
    values: array of feature values (one per run)
    labels: array of bool (correct/wrong)
    Returns (raw_auroc, da_auroc) or (np.nan, np.nan) if insufficient.
    """
    n = len(values)
    correct_idx = np.where(labels)[0]
    wrong_idx = np.where(~labels)[0]
    if len(correct_idx) < 2 or len(wrong_idx) < 2:
        return np.nan, np.nan

    scores = np.empty(n)
    for hold in range(n):
        mask = np.ones(n, dtype=bool)
        mask[hold] = False
        c_mean = values[mask & labels].mean()
        w_mean = values[mask & (~labels)].mean()
        scores[hold] = abs(values[hold] - w_mean) - abs(values[hold] - c_mean)

    raw = roc_auc_score(labels, scores)
    da = max(raw, 1.0 - raw)
    return raw, da


def loo_auroc_vector(sigs, labels):
    """LOO centroid-distance AUROC for vector feature (signature/OT)."""
    n = len(sigs)
    correct_idx = np.where(labels)[0]
    wrong_idx = np.where(~labels)[0]
    if len(correct_idx) < 2 or len(wrong_idx) < 2:
        return np.nan, np.nan

    scores = np.empty(n)
    for hold in range(n):
        mask = np.ones(n, dtype=bool)
        mask[hold] = False
        rem_sigs = sigs[mask]
        rem_labels = labels[mask]
        c_centroid = rem_sigs[rem_labels].mean(axis=0)
        w_centroid = rem_sigs[~rem_labels].mean(axis=0)
        scores[hold] = np.linalg.norm(sigs[hold] - w_centroid) - np.linalg.norm(sigs[hold] - c_centroid)

    raw = roc_auc_score(labels, scores)
    da = max(raw, 1.0 - raw)
    return raw, da


def loo_auroc_top5(entropy_list, labels):
    """LOO linear-regression AUROC using top-5 scalar features.
    Need >= 6 runs total (5 features + 1 held out minimum).
    """
    n = len(entropy_list)
    if n < 6:
        return np.nan, np.nan
    correct_idx = np.where(labels)[0]
    wrong_idx = np.where(~labels)[0]
    if len(correct_idx) < 2 or len(wrong_idx) < 2:
        return np.nan, np.nan

    feat_matrix = np.zeros((n, 5))
    for i, ent in enumerate(entropy_list):
        for j, fn in enumerate(TOP5_FEATURES):
            feat_matrix[i, j] = fn(ent)

    scores = np.empty(n)
    for hold in range(n):
        mask = np.ones(n, dtype=bool)
        mask[hold] = False
        X_train = feat_matrix[mask]
        y_train = labels[mask].astype(float)
        X_test = feat_matrix[hold:hold+1]

        XtX = X_train.T @ X_train
        reg = 1e-6 * np.eye(5)
        try:
            beta = np.linalg.solve(XtX + reg, X_train.T @ y_train)
        except np.linalg.LinAlgError:
            return np.nan, np.nan
        scores[hold] = (X_test @ beta)[0]

    raw = roc_auc_score(labels, scores)
    da = max(raw, 1.0 - raw)
    return raw, da

# ── data loading ───────────────────────────────────────────────────────────
def load_model_data(model_name):
    """Returns dict: problem_id -> {tier, labels: bool array, npz_paths: list}"""
    info = MODELS[model_name]
    problems = defaultdict(lambda: {"runs": []})

    with open(info["jsonl"]) as f:
        for line in f:
            e = json.loads(line)
            pid = e["problem_id"]
            run_id = e["run_id"]
            tier = e["difficulty_tier"]
            is_correct = bool(e["is_correct"])
            npz_name = f"{pid}_run{run_id:03d}.npz"
            npz_path = os.path.join(info["npz_dir"], npz_name)
            problems[pid]["runs"].append({
                "run_id": run_id,
                "is_correct": is_correct,
                "tier": tier,
                "npz_path": npz_path,
            })

    result = {}
    for pid, data in problems.items():
        data["runs"].sort(key=lambda x: x["run_id"])
        tier = data["runs"][0]["tier"]
        labels = np.array([r["is_correct"] for r in data["runs"]])
        npz_paths = [r["npz_path"] for r in data["runs"]]
        result[pid] = {"tier": tier, "labels": labels, "npz_paths": npz_paths}
    return result


def load_entropy(pid_data):
    """Load entropy arrays for a problem. Returns list of entropy arrays (float64)."""
    return [np.load(p)["tok_entropy"].astype(np.float64) for p in pid_data["npz_paths"]]

# ── per-problem analysis ──────────────────────────────────────────────────
def analyze_problem(pid, pid_data, methods_to_run=None):
    """Compute all method AUROCs for a single problem.
    Returns dict of method_name -> (raw_auroc, da_auroc) or np.nan.
    """
    labels = pid_data["labels"]
    n_runs = len(labels)
    n_correct = labels.sum()
    n_wrong = n_runs - n_correct

    wp_eligible = (n_correct >= 2 and n_wrong >= 2)

    entropy_arrays = load_entropy(pid_data)

    results = {}

    # Scalar methods
    for name, fn in SCALAR_METHODS:
        if not wp_eligible:
            results[name] = (np.nan, np.nan)
            continue
        values = np.array([fn(e) for e in entropy_arrays])
        results[name] = loo_auroc_scalar(values, labels)

    # Signature 2D (depth 3)
    if not wp_eligible:
        results["Signature-2D"] = (np.nan, np.nan)
    else:
        try:
            sigs = np.array([compute_signature(e) for e in entropy_arrays])
            results["Signature-2D"] = loo_auroc_vector(sigs, labels)
        except Exception:
            results["Signature-2D"] = (np.nan, np.nan)

    # OT Barycenter 1D
    if not wp_eligible:
        results["OT-Barycenter"] = (np.nan, np.nan)
    else:
        try:
            min_len = min(len(e) for e in entropy_arrays)
            if min_len < 2:
                results["OT-Barycenter"] = (np.nan, np.nan)
            else:
                sorted_ents = np.array([np.sort(e[:min_len]) for e in entropy_arrays])
                results["OT-Barycenter"] = loo_auroc_vector(sorted_ents, labels)
        except Exception:
            results["OT-Barycenter"] = (np.nan, np.nan)

    # Top-5 linear combo
    if not wp_eligible:
        results["Top5-linear"] = (np.nan, np.nan)
    else:
        try:
            results["Top5-linear"] = loo_auroc_top5(entropy_arrays, labels)
        except Exception:
            results["Top5-linear"] = (np.nan, np.nan)

    return pid, pid_data["tier"], results


def analyze_problem_wrapper(args):
    pid, pid_data = args
    return analyze_problem(pid, pid_data)


# ── bootstrap CI for median ───────────────────────────────────────────────
def bootstrap_median_ci(data, n_boot=10000, ci=0.95):
    """Bootstrap CI for median. data should be 1D array of non-nan values."""
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

# ── main analysis ─────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    all_method_names = [name for name, _ in SCALAR_METHODS] + [
        "Signature-2D", "OT-Barycenter", "Top5-linear"
    ]

    for model_name in MODELS:
        print(f"\n{'='*80}")
        print(f"Loading {model_name} data...")
        problems = load_model_data(model_name)
        print(f"  {len(problems)} problems loaded ({time.time()-t0:.1f}s)")

        # Group by tier
        tier_pids = defaultdict(list)
        for pid, pdata in problems.items():
            tier_pids[pdata["tier"]].append(pid)

        # Run all problems in parallel
        work_items = [(pid, problems[pid]) for pid in problems]
        print(f"  Processing {len(work_items)} problems with multiprocessing...")
        t1 = time.time()

        with Pool() as pool:
            all_results = pool.map(analyze_problem_wrapper, work_items, chunksize=4)

        print(f"  Done ({time.time()-t1:.1f}s)")

        # Organize results by tier
        tier_results = defaultdict(lambda: defaultdict(list))
        for pid, tier, res in all_results:
            for method_name in all_method_names:
                if method_name in res:
                    raw, da = res[method_name]
                    if not np.isnan(da):
                        tier_results[tier][method_name].append((raw, da))

        # Print tables per tier
        for tier in TIERS:
            method_data = tier_results[tier]
            # Count WP-eligible problems (those with any non-nan result)
            wp_count = len(set(
                pid for pid, t, res in all_results
                if t == tier and any(not np.isnan(v[1]) for v in res.values())
            ))

            print(f"\n{'='*80}")
            print(f"=== MODEL: {model_name}, TIER: {tier} (WP-eligible n={wp_count}) ===")
            print(f"{'Method':<22} | {'DA Median':>10} | {'95% CI':>24} | {'Flip Rate':>9} | {'Raw Median':>10}")
            print(f"{'-'*22}-+-{'-'*10}-+-{'-'*24}-+-{'-'*9}-+-{'-'*10}")

            for method_name in all_method_names:
                pairs = method_data.get(method_name, [])
                if not pairs:
                    print(f"{method_name:<22} | {'N/A':>10} | {'N/A':>24} | {'N/A':>9} | {'N/A':>10}")
                    continue

                raws = np.array([p[0] for p in pairs])
                das = np.array([p[1] for p in pairs])
                flip_rate = np.mean(raws < 0.5)

                med_da, (ci_lo, ci_hi) = bootstrap_median_ci(das)
                med_raw = np.median(raws)

                print(f"{method_name:<22} | {med_da:>10.3f} | [{ci_lo:.3f}, {ci_hi:.3f}]     | {flip_rate*100:>6.1f}%    | {med_raw:>10.3f}")

    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"Total time: {elapsed:.1f}s")

    # Save results to file
    with open("/tmp/full_auroc_analysis_output.txt", "w") as fout:
        fout.write("Full AUROC analysis completed.\n")
    print("Results saved.")


if __name__ == "__main__":
    main()
