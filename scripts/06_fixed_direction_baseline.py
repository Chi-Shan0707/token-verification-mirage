#!/usr/bin/env python3
"""Fixed-direction baseline experiment.

Compares direction-agnostic (DA) scoring (max(AUROC, 1-AUROC)) against
a fixed global direction learned from training data using GroupKFold.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from collections import defaultdict
import warnings
import iisignature
import ot

warnings.filterwarnings('ignore')

NPZ_CACHE = {}
NPZ_REMAP = {}


def set_npz_dir(npz_dir):
    """Set the NPZ directory for remapping paths."""
    global NPZ_REMAP
    NPZ_REMAP.clear()
    import os
    for fname in os.listdir(npz_dir):
        NPZ_REMAP[fname] = os.path.join(npz_dir, fname)


def load_data(jsonl_path):
    """Load JSONL and group traces by problem_id."""
    traces = defaultdict(list)
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            traces[rec['problem_id']].append(rec)
    return traces


def _resolve_npz_path(npz_path):
    """Resolve NPZ path: use remap if available, else try original."""
    fname = Path(npz_path).name
    if fname in NPZ_REMAP:
        return NPZ_REMAP[fname]
    return npz_path


def get_npz(npz_path):
    """Load and cache NPZ data."""
    resolved = _resolve_npz_path(npz_path)
    if resolved not in NPZ_CACHE:
        NPZ_CACHE[resolved] = dict(np.load(resolved, allow_pickle=True))
    return NPZ_CACHE[resolved]


def get_hard_wp_eligible(traces):
    """Get problem IDs that are hard and have >= 1 correct and >= 1 wrong."""
    eligible = []
    for pid, recs in traces.items():
        if recs[0]['difficulty_tier'] != 'hard':
            continue
        n_correct = sum(1 for r in recs if r['is_correct'])
        n_wrong = len(recs) - n_correct
        if n_correct >= 1 and n_wrong >= 1:
            eligible.append(pid)
    return eligible


# --- Scalar score methods ---

def score_final_entropy(rec):
    return float(get_npz(rec['npz_path'])['tok_entropy'][-1])


def score_first_entropy(rec):
    return float(get_npz(rec['npz_path'])['tok_entropy'][0])


def score_entropy_mean(rec):
    return float(np.mean(get_npz(rec['npz_path'])['tok_entropy']))


# --- LOO centroid methods ---

def compute_signature_2d(tok_entropy):
    """2D path: (normalized_position, entropy), level-2 signature."""
    ent = tok_entropy.astype(np.float64)
    T = len(ent)
    path = np.column_stack([np.linspace(0, 1, T), ent])
    return iisignature.sig(path, 2)


def compute_loo_signature_scores(recs):
    """LOO centroid distance using signature 2D."""
    sigs = []
    labels = []
    for rec in recs:
        npz = get_npz(rec['npz_path'])
        sig = compute_signature_2d(npz['tok_entropy'])
        sigs.append(sig)
        labels.append(rec['is_correct'])

    sigs = np.array(sigs)
    labels = np.array(labels)
    correct_idx = np.where(labels)[0]
    wrong_idx = np.where(~labels)[0]

    scores = np.full(len(recs), np.nan)
    for i in range(len(recs)):
        ci = correct_idx[correct_idx != i] if i in correct_idx else correct_idx
        wi = wrong_idx[wrong_idx != i] if i in wrong_idx else wrong_idx

        if len(ci) == 0 or len(wi) == 0:
            continue

        c_cent = np.mean(sigs[ci], axis=0)
        w_cent = np.mean(sigs[wi], axis=0)
        scores[i] = np.linalg.norm(sigs[i] - w_cent) - np.linalg.norm(sigs[i] - c_cent)

    return scores


def wasserstein_1d(a, b):
    """1D Wasserstein distance using scipy (handles unequal lengths)."""
    from scipy.stats import wasserstein_distance as wd
    return float(wd(a, b))


def _quantile_function(values, n_grid=128):
    """Evaluate quantile function on a uniform grid."""
    sorted_v = np.sort(values)
    probs = np.linspace(0, 1, n_grid)
    return np.interp(probs, np.linspace(0, 1, len(sorted_v)), sorted_v)


def compute_loo_ot_scores(recs):
    """LOO barycenter distance using 1D optimal transport."""
    entropies = []
    labels = []
    for rec in recs:
        npz = get_npz(rec['npz_path'])
        entropies.append(npz['tok_entropy'].astype(np.float64))
        labels.append(rec['is_correct'])

    labels = np.array(labels)
    correct_idx = np.where(labels)[0]
    wrong_idx = np.where(~labels)[0]

    scores = np.full(len(recs), np.nan)
    for i in range(len(recs)):
        ci = correct_idx[correct_idx != i] if i in correct_idx else correct_idx
        wi = wrong_idx[wrong_idx != i] if i in wrong_idx else wrong_idx

        if len(ci) == 0 or len(wi) == 0:
            continue

        # Barycenter = mean quantile function (exact for 1D Wasserstein)
        N_GRID = 128
        c_qf = np.mean([_quantile_function(entropies[j], N_GRID) for j in ci], axis=0)
        w_qf = np.mean([_quantile_function(entropies[j], N_GRID) for j in wi], axis=0)
        qi = _quantile_function(entropies[i], N_GRID)

        dist_c = float(np.mean(np.abs(qi - c_qf)))
        dist_w = float(np.mean(np.abs(qi - w_qf)))

        scores[i] = dist_w - dist_c

    return scores


# --- Per-problem AUROC helpers ---

def per_problem_auroc(scores, labels, direction='da'):
    """Compute per-problem AUROC.

    direction='da': max(AUROC, 1-AUROC)
    direction='high': use raw scores (high -> correct)
    direction='low': negate scores (low -> correct)
    """
    valid = ~np.isnan(scores)
    s, y = scores[valid], labels[valid]
    if len(set(y)) < 2 or len(s) < 2:
        return np.nan
    if direction == 'da':
        a = roc_auc_score(y, s)
        return max(a, 1 - a)
    elif direction == 'high':
        return roc_auc_score(y, s)
    elif direction == 'low':
        return roc_auc_score(y, -s)


def compute_all_problem_data(traces, eligible_pids):
    """Pre-compute scores for all eligible problems."""
    problem_data = {}
    method_names = ['Final entropy', 'First entropy', 'Entropy mean',
                    'Signature 2D', 'OT Barycenter']

    for pid in eligible_pids:
        recs = traces[pid]
        labels = np.array([r['is_correct'] for r in recs])

        s_final = np.array([score_final_entropy(r) for r in recs])
        s_first = np.array([score_first_entropy(r) for r in recs])
        s_mean = np.array([score_entropy_mean(r) for r in recs])
        s_sig = compute_loo_signature_scores(recs)
        s_ot = compute_loo_ot_scores(recs)

        problem_data[pid] = {
            'labels': labels,
            'scores': {
                'Final entropy': s_final,
                'First entropy': s_first,
                'Entropy mean': s_mean,
                'Signature 2D': s_sig,
                'OT Barycenter': s_ot,
            }
        }

    return problem_data, method_names


# --- Main experiment ---

def run_experiment(model_name, jsonl_path):
    print(f"\nLoading {model_name} data...")
    traces = load_data(jsonl_path)
    eligible_pids = sorted(get_hard_wp_eligible(traces))
    n_problems = len(eligible_pids)
    print(f"  Hard WP-eligible: {n_problems}")

    print("  Computing per-problem scores...")
    problem_data, method_names = compute_all_problem_data(traces, eligible_pids)
    print("  Done computing scores.")

    # GroupKFold
    problem_ids = np.array(eligible_pids)
    X = np.arange(len(problem_ids))
    y_dummy = np.ones(len(problem_ids))
    gkf = GroupKFold(n_splits=5)

    results = {m: {'da_aurocs': [], 'fixed_aurocs': [],
                   'dir_correct': [], 'train_high_fracs': []}
               for m in method_names}

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y_dummy, groups=problem_ids)):
        train_pids = problem_ids[train_idx]
        test_pids = problem_ids[test_idx]

        for method in method_names:
            # Learn global direction from TRAIN
            train_high_count = 0
            train_total = 0
            for pid in train_pids:
                d = problem_data[pid]
                sc = d['scores'][method]
                lb = d['labels']
                valid = ~np.isnan(sc)
                if valid.sum() < 2 or len(set(lb[valid])) < 2:
                    continue
                raw = roc_auc_score(lb[valid], sc[valid])
                if raw >= 0.5:
                    train_high_count += 1
                train_total += 1

            if train_total == 0:
                continue

            high_frac = train_high_count / train_total
            global_dir = 'high' if high_frac >= 0.5 else 'low'
            results[method]['train_high_fracs'].append(high_frac)

            # Evaluate on TEST
            for pid in test_pids:
                d = problem_data[pid]
                sc = d['scores'][method]
                lb = d['labels']

                # DA AUROC
                da = per_problem_auroc(sc, lb, 'da')
                if np.isnan(da):
                    continue
                results[method]['da_aurocs'].append(da)

                # Fixed-direction AUROC
                fixed = per_problem_auroc(sc, lb, global_dir)
                if np.isnan(fixed):
                    continue
                results[method]['fixed_aurocs'].append(fixed)

                # Direction correctness: compare learned direction to true direction
                raw = per_problem_auroc(sc, lb, 'high')
                true_dir = 'high' if raw >= 0.5 else 'low'
                results[method]['dir_correct'].append(global_dir == true_dir)

    return results, n_problems, method_names


def print_results(model_name, results, n_problems, method_names):
    print(f"\n=== {model_name} Hard (n={n_problems}) ===")
    print(f"{'Method':<16} | {'DA Med AUROC':>12} | {'Fixed-Dir Med AUROC':>19} | {'Gap':>7} | {'% Train high→corr':>19}")
    print("-" * 90)

    for method in method_names:
        r = results[method]
        da_med = np.median(r['da_aurocs']) if r['da_aurocs'] else np.nan
        fx_med = np.median(r['fixed_aurocs']) if r['fixed_aurocs'] else np.nan
        gap = da_med - fx_med
        avg_hf = np.mean(r['train_high_fracs']) * 100 if r['train_high_fracs'] else np.nan
        print(f"{method:<16} | {da_med:>12.4f} | {fx_med:>19.4f} | {gap:>7.4f} | {avg_hf:>18.1f}%")

    for method in method_names:
        r = results[method]
        nc = sum(r['dir_correct'])
        nt = len(r['dir_correct'])
        if nt > 0:
            print(f"Per-problem direction accuracy on test ({method}): {nc}/{nt} = {nc/nt*100:.1f}% problems have correct direction")


if __name__ == '__main__':
    configs = [
        ('Qwen', 'outputs/bigmath_merged64/bigmath_qwen_64run.jsonl',
         'outputs/bigmath_merged64/bigmath_qwen_64run_npz'),
        ('Llama', 'outputs/bigmath_merged64/bigmath_llama_64run.jsonl',
         'outputs/bigmath_merged64/bigmath_llama_64run_npz'),
    ]

    for model_name, jsonl_path, npz_dir in configs:
        NPZ_CACHE.clear()
        set_npz_dir(npz_dir)
        results, n_problems, method_names = run_experiment(model_name, jsonl_path)
        print_results(model_name, results, n_problems, method_names)

    print("\nDone.")
