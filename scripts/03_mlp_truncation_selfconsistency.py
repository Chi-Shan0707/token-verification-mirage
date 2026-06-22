#!/usr/bin/env python3
"""
Three analyses on 64-run merged BigMath data:
1. Cross-Problem MLP Verifier (GroupKFold 5-fold)
2. Prefix Truncation Ablation (Hard WP-eligible)
3. Self-Consistency Baseline (Majority Voting + pass@N)
"""

import json
import os
import time
import numpy as np
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.special import comb
import warnings
warnings.filterwarnings('ignore')

NCPUS = min(cpu_count(), 16)
print(f"Using {NCPUS} CPUs")

BASE = 'outputs/bigmath_merged64'
MODELS = {
    'Qwen': {
        'jsonl': os.path.join(BASE, 'bigmath_qwen_64run.jsonl'),
        'npz_dir': os.path.join(BASE, 'bigmath_qwen_64run_npz/'),
    },
    'Llama': {
        'jsonl': os.path.join(BASE, 'bigmath_llama_64run.jsonl'),
        'npz_dir': os.path.join(BASE, 'bigmath_llama_64run_npz/'),
    },
}


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def npz_path(npz_dir, problem_id, run_id):
    return os.path.join(npz_dir, f"{problem_id}_run{run_id:03d}.npz")


# ============================================================
# Analysis 1: Cross-Problem MLP Verifier
# ============================================================

def extract_features(npz_p):
    d = np.load(npz_p)
    ent = d['tok_entropy'].astype(np.float64)
    n = len(ent)
    s = max(0, int(n * 0.9))
    return np.array([
        np.mean(ent),
        np.std(ent) if n > 1 else 0.0,
        float(ent[0]),
        float(ent[-1]),
        float(n),
        np.mean(ent[s:]),
    ])


def analysis1():
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Cross-Problem MLP Verifier (GroupKFold 5-fold)")
    print("=" * 70)

    for model_name, paths in MODELS.items():
        print(f"\n=== Cross-Problem MLP: {model_name} ===")
        t0 = time.time()

        records = load_jsonl(paths['jsonl'])
        ndir = paths['npz_dir']

        npz_paths = [npz_path(ndir, r['problem_id'], r['run_id']) for r in records]

        with Pool(NCPUS) as pool:
            features = np.array(pool.map(extract_features, npz_paths))

        labels = np.array([r['is_correct'] for r in records])
        pids = np.array([r['problem_id'] for r in records])
        tier_map = {r['problem_id']: r['difficulty_tier'] for r in records}

        unique_pids, groups = np.unique(pids, return_inverse=True)
        print(f"  {len(records)} traces, {len(unique_pids)} problems, loaded in {time.time()-t0:.1f}s")

        gkf = GroupKFold(n_splits=5)
        problem_auroc = {}

        for fold_i, (tr, te) in enumerate(gkf.split(features, labels, groups)):
            t1 = time.time()
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(features[tr])
            X_te = scaler.transform(features[te])
            y_tr, y_te = labels[tr], labels[te]

            mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=30,
                                solver='adam', random_state=42)
            mlp.fit(X_tr, y_tr)

            y_prob = mlp.predict_proba(X_te)
            y_score = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob[:, 0]

            te_pids = pids[te]
            for pid in np.unique(te_pids):
                mask = te_pids == pid
                yt, ys = y_te[mask], y_score[mask]
                if len(set(yt)) < 2:
                    continue
                auc = roc_auc_score(yt, ys)
                problem_auroc[pid] = max(auc, 1 - auc)

            print(f"  Fold {fold_i+1}/5 done in {time.time()-t1:.1f}s")

        rng = np.random.RandomState(42)
        print(f"\n{'Tier':<8}| {'n_problems':>10} | {'Median AUROC':>12} | {'95% CI':>20}")
        print("-" * 58)

        for tier_label in ['easy', 'medium', 'hard', 'All']:
            if tier_label == 'All':
                pset = list(problem_auroc.keys())
            else:
                pset = [p for p in problem_auroc if tier_map[p] == tier_label]
            aucs = np.array([problem_auroc[p] for p in pset])
            n = len(aucs)
            if n == 0:
                continue
            med = np.median(aucs)
            boot_idx = rng.randint(0, n, size=(10000, n))
            boot_meds = np.median(aucs[boot_idx], axis=1)
            ci_lo, ci_hi = np.percentile(boot_meds, [2.5, 97.5])
            print(f"{tier_label:<8}| {n:>10} | {med:>12.3f} | [{ci_lo:.3f}, {ci_hi:.3f}]")


# ============================================================
# Analysis 2: Prefix Truncation Ablation
# ============================================================

def truncation_worker(args):
    npz_paths, is_corrects = args
    ent_arrays = []
    for p in npz_paths:
        d = np.load(p)
        ent_arrays.append(d['tok_entropy'].astype(np.float64))

    y_true = np.array(is_corrects)
    if len(set(y_true)) < 2:
        return None

    res_final = []
    res_mean = []
    for k in range(1, 11):
        finals, means = [], []
        ok = True
        for ent in ent_arrays:
            n = len(ent)
            if n == 0:
                ok = False
                break
            cut = max(1, int(n * k / 10))
            t = ent[:cut]
            finals.append(float(t[-1]))
            means.append(float(np.mean(t)))
        if not ok:
            res_final.append(np.nan)
            res_mean.append(np.nan)
            continue

        auc = roc_auc_score(y_true, np.array(finals))
        res_final.append(max(auc, 1 - auc))
        auc = roc_auc_score(y_true, np.array(means))
        res_mean.append(max(auc, 1 - auc))

    return (res_final, res_mean)


def analysis2():
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Prefix Truncation Ablation (Hard WP-eligible)")
    print("=" * 70)

    for model_name, paths in MODELS.items():
        print(f"\n=== Prefix Truncation: {model_name} Hard ===")
        t0 = time.time()

        records = load_jsonl(paths['jsonl'])
        ndir = paths['npz_dir']

        hard_probs = defaultdict(list)
        for r in records:
            if r['difficulty_tier'] == 'hard':
                hard_probs[r['problem_id']].append(r)

        problem_args = []
        for pid, recs in hard_probs.items():
            corrects = [r['is_correct'] for r in recs]
            if len(set(corrects)) < 2:
                continue
            recs_s = sorted(recs, key=lambda x: x['run_id'])
            npaths = [npz_path(ndir, pid, r['run_id']) for r in recs_s]
            problem_args.append((npaths, corrects))

        print(f"  {len(hard_probs)} hard problems, {len(problem_args)} WP-eligible")

        with Pool(NCPUS) as pool:
            results = pool.map(truncation_worker, problem_args)

        print(f"  Computed in {time.time()-t0:.1f}s")

        print(f"\n{'% used':>7}| {'Final-entropy DA Median':>24} | {'Entropy-mean DA Median':>23}")
        print("-" * 60)
        for k in range(1, 11):
            finals = [r[0][k-1] for r in results if r is not None and not np.isnan(r[0][k-1])]
            means = [r[1][k-1] for r in results if r is not None and not np.isnan(r[1][k-1])]
            mf = np.median(finals) if finals else float('nan')
            mm = np.median(means) if means else float('nan')
            print(f"{k*10:>6}% | {mf:>24.3f} | {mm:>23.3f}")


# ============================================================
# Analysis 3: Self-Consistency Baseline
# ============================================================

def pass_at_n(n_total, n_correct, n):
    if n_correct == 0:
        return 0.0
    if n_total - n_correct < n:
        return 1.0
    return 1.0 - float(comb(n_total - n_correct, n, exact=True)) / float(comb(n_total, n, exact=True))


def analysis3():
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Self-Consistency Baseline (Majority Voting)")
    print("=" * 70)

    Ns = [4, 8, 16, 32, 64]
    hdr = f"{'Model':<7}| {'Tier':<7}| {'N_problems':>10} | {'Maj Acc':>8} | " + " | ".join([f"pass@{n}" for n in Ns])
    print(f"\n{hdr}")
    print("-" * len(hdr))

    for model_name, paths in MODELS.items():
        records = load_jsonl(paths['jsonl'])
        tier_probs = defaultdict(lambda: defaultdict(list))
        for r in records:
            tier_probs[r['difficulty_tier']][r['problem_id']].append(r['is_correct'])

        for tier in ['easy', 'medium', 'hard']:
            probs = tier_probs[tier]
            n_probs = len(probs)
            maj_correct = 0
            pr = {n: [] for n in Ns}

            for pid, corrects in probs.items():
                ca = np.array(corrects)
                nt = len(ca)
                nc = int(ca.sum())
                if nc > nt / 2:
                    maj_correct += 1
                for n in Ns:
                    pr[n].append(pass_at_n(nt, nc, min(n, nt)))

            ma = maj_correct / n_probs * 100
            ps = " | ".join([f"{np.mean(pr[n])*100:>7.1f}%" for n in Ns])
            print(f"{model_name:<7}| {tier:<7}| {n_probs:>10} | {ma:>7.1f}% | {ps}")


# ============================================================
if __name__ == '__main__':
    t = time.time()
    analysis1()
    analysis2()
    analysis3()
    print(f"\nTotal time: {time.time() - t:.1f}s")
