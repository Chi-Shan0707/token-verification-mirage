#!/usr/bin/env python3
"""Permutation null baseline and protocol sensitivity analysis on 64-run merged data."""

import json, os, sys, time
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from multiprocessing import Pool
import iisignature

N_PERM = 1000
N_BOOT = 2000
RS_BOOT = 42

JSONL = {
    "Qwen":  "outputs/bigmath_merged64/bigmath_qwen_64run.jsonl",
    "Llama": "outputs/bigmath_merged64/bigmath_llama_64run.jsonl",
}
NPZ_DIR = {
    "Qwen":  "outputs/bigmath_merged64/bigmath_qwen_64run_npz",
    "Llama": "outputs/bigmath_merged64/bigmath_llama_64run_npz",
}


def da_auroc(scores, labels):
    u = np.unique(labels)
    if len(u) < 2:
        return np.nan
    a = roc_auc_score(labels, scores)
    return max(a, 1.0 - a)


def load_features_for_model(model_name):
    jsonl_path = JSONL[model_name]
    npz_dir = NPZ_DIR[model_name]

    problems = defaultdict(list)
    with open(jsonl_path) as f:
        for line in f:
            problems[json.loads(line)["problem_id"]].append(json.loads(line))

    wp = {}
    for pid, runs in problems.items():
        nc = sum(r["is_correct"] for r in runs)
        nw = len(runs) - nc
        if nc >= 2 and nw >= 2 and runs[0].get("difficulty_tier") == "hard":
            wp[pid] = runs

    print(f"  {model_name}: {len(problems)} problems, {len(wp)} WP-eligible", flush=True)

    features = {}
    for pid, runs in wp.items():
        ent_list, sig_list, lab_list = [], [], []
        for r in runs:
            npz_path = os.path.join(npz_dir, f"{pid}_run{r['run_id']:03d}.npz")
            if not os.path.exists(npz_path):
                continue
            d = np.load(npz_path)
            ent = float(d["tok_entropy"][-1])
            t = np.linspace(0, 1, len(d["tok_entropy"]))
            path2d = np.column_stack([
                t.astype(np.float64),
                d["tok_entropy"].astype(np.float64),
            ])
            sig = iisignature.sig(path2d, 3)
            ent_list.append(ent)
            sig_list.append(sig)
            lab_list.append(1 if r["is_correct"] else 0)
        if len(set(lab_list)) >= 2:
            features[pid] = {
                "entropy": np.array(ent_list),
                "signature": np.array(sig_list),
                "labels": np.array(lab_list),
            }
    return features


# ---------- Part 1 ----------

def _perm_task(args):
    entropies, labels, n_perm, seed = args
    rng = np.random.default_rng(seed)
    n = len(entropies)
    out = np.empty(n_perm)
    for i in range(n_perm):
        pl = rng.permutation(labels)
        a = roc_auc_score(pl, entropies)
        out[i] = max(a, 1.0 - a)
    return out


def part1_permutation(features, model_name):
    print(f"\n{'='*60}")
    print(f"Part 1: Permutation Null — {model_name}")
    print(f"{'='*60}")

    tasks = []
    idx = 0
    for pid, feat in features.items():
        tasks.append((feat["entropy"], feat["labels"], N_PERM, idx))
        idx += 1

    with Pool() as pool:
        results = pool.map(_perm_task, tasks, chunksize=10)

    all_aurocs = np.concatenate(results)
    n_total = len(all_aurocs)

    pct = np.percentile(all_aurocs, [5, 25, 50, 75, 95])
    print(f"  Problems:        {len(features)}")
    print(f"  Total AUROCs:    {n_total}")
    print(f"  Mean:            {all_aurocs.mean():.4f}")
    print(f"  Median:          {np.median(all_aurocs):.4f}")
    print(f"  P5:              {pct[0]:.4f}")
    print(f"  P25:             {pct[1]:.4f}")
    print(f"  P50:             {pct[2]:.4f}")
    print(f"  P75:             {pct[3]:.4f}")
    print(f"  P95:             {pct[4]:.4f}")
    print(f"  Frac > 0.60:     {(all_aurocs > 0.60).mean():.4f}")
    print(f"  Frac > 0.65:     {(all_aurocs > 0.65).mean():.4f}")
    print(f"  Frac > 0.70:     {(all_aurocs > 0.70).mean():.4f}")
    return all_aurocs


# ---------- Part 2 ----------

def _centroid_score_vec(feat_matrix, labels, mean_c, mean_w):
    d_w = np.linalg.norm(feat_matrix - mean_w, axis=1)
    d_c = np.linalg.norm(feat_matrix - mean_c, axis=1)
    return d_w - d_c


def _per_problem_protocol(feat_arr, labels):
    n = len(labels)
    c_mask = labels == 1
    w_mask = labels == 0

    # In-sample
    mc = feat_arr[c_mask].mean(axis=0)
    mw = feat_arr[w_mask].mean(axis=0)
    is_scores = _centroid_score_vec(feat_arr, labels, mc, mw)
    is_auroc = da_auroc(is_scores, labels)

    # LOO
    loo_scores = np.empty(n)
    for i in range(n):
        rem = np.delete(feat_arr, i, axis=0)
        rl = np.delete(labels, i)
        mc_loo = rem[rl == 1].mean(axis=0)
        mw_loo = rem[rl == 0].mean(axis=0)
        loo_scores[i] = np.linalg.norm(feat_arr[i] - mw_loo) - np.linalg.norm(feat_arr[i] - mc_loo)
    loo_auroc = da_auroc(loo_scores, labels)

    return loo_auroc, is_auroc


def part2_protocol(features, model_name):
    print(f"\n{'='*60}")
    print(f"Part 2: Protocol Sensitivity — {model_name}")
    print(f"{'='*60}")

    pids = sorted(features.keys())
    n_prob = len(pids)

    rng_boot = np.random.default_rng(RS_BOOT)

    for method, key in [("Final-token entropy", "entropy"), ("Signature 2D", "signature")]:
        loo_pp = np.empty(n_prob)
        is_pp = np.empty(n_prob)
        for i, pid in enumerate(pids):
            f = features[pid]
            arr = f[key]
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            loo_pp[i], is_pp[i] = _per_problem_protocol(arr, f["labels"])

        # Global pooled
        _feats = []
        for pid in pids:
            a = features[pid][key]
            _feats.append(a.reshape(-1, 1) if a.ndim == 1 else a)
        all_feat = np.concatenate(_feats, axis=0)
        all_lab = np.concatenate([features[pid]["labels"] for pid in pids])
        g_mc = all_feat[all_lab == 1].mean(axis=0)
        g_mw = all_feat[all_lab == 0].mean(axis=0)
        g_scores = _centroid_score_vec(all_feat, all_lab, g_mc, g_mw)
        global_auroc = da_auroc(g_scores, all_lab)

        # Bootstrap CI for LOO (over problems)
        boot_loo = np.empty(N_BOOT)
        boot_is = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = rng_boot.choice(n_prob, n_prob, replace=True)
            boot_loo[b] = np.nanmedian(loo_pp[idx])
            boot_is[b] = np.nanmedian(is_pp[idx])
        loo_mean = np.nanmedian(loo_pp)
        loo_ci = (np.percentile(boot_loo, 2.5), np.percentile(boot_loo, 97.5))
        is_mean = np.nanmedian(is_pp)
        delta_is = is_mean - loo_mean

        # Bootstrap for global (resample problems, pool)
        boot_gl = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = rng_boot.choice(n_prob, n_prob, replace=True)
            bf = np.concatenate([
                (features[pids[j]][key].reshape(-1, 1) if features[pids[j]][key].ndim == 1 else features[pids[j]][key])
                for j in idx
            ], axis=0)
            bl = np.concatenate([features[pids[j]]["labels"] for j in idx])
            mc_b = bf[bl == 1].mean(axis=0)
            mw_b = bf[bl == 0].mean(axis=0)
            sc_b = _centroid_score_vec(bf, bl, mc_b, mw_b)
            boot_gl[b] = da_auroc(sc_b, bl)
        gl_ci = (np.percentile(boot_gl, 2.5), np.percentile(boot_gl, 97.5))
        delta_gl = global_auroc - loo_mean

        print(f"\nMethod: {method}, Model: {model_name} Hard")
        print(f"  LOO:       {loo_mean:.3f} [{loo_ci[0]:.3f}, {loo_ci[1]:.3f}]")
        print(f"  In-sample: {is_mean:.3f} (Δ={delta_is:+.3f})")
        print(f"  Global:    {global_auroc:.3f} (Δ={delta_gl:+.3f})")


def main():
    t0 = time.time()
    for model in ["Qwen", "Llama"]:
        print(f"\n>>> Loading {model} ...", flush=True)
        features = load_features_for_model(model)

        part1_permutation(features, model)
        part2_protocol(features, model)

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
