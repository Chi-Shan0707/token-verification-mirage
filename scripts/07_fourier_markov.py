import json
import numpy as np
import os
from collections import defaultdict
from sklearn.metrics import roc_auc_score

def load_problems(jsonl_path, npz_dir):
    problems = defaultdict(list)
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            problems[rec['problem_id']].append(rec)

    result = {}
    for pid, runs in problems.items():
        tier = runs[0]['difficulty_tier']
        entries = []
        for r in sorted(runs, key=lambda x: x['run_id']):
            rid = r['run_id']
            npz_name = f"{pid}_run{rid:03d}.npz"
            npz_path = os.path.join(npz_dir, npz_name)
            if not os.path.exists(npz_path):
                continue
            data = np.load(npz_path)
            entropy = data['tok_entropy'].astype(np.float64)
            entries.append({
                'run_id': rid,
                'is_correct': r['is_correct'],
                'entropy': entropy,
            })
        if entries:
            result[pid] = {'tier': tier, 'runs': entries}
    return result


def fourier_spectral_entropy_4d(entropy_seq):
    fft_vals = np.fft.rfft(entropy_seq)
    power = np.abs(fft_vals) ** 2
    power = power[1:]  # drop DC
    if len(power) < 4:
        power = np.pad(power, (0, 4 - len(power)))
    n = len(power)
    band_size = n // 4
    features = np.zeros(4)
    eps = 1e-12
    for i in range(4):
        start = i * band_size
        end = start + band_size if i < 3 else n
        band = power[start:end]
        total = band.sum()
        if total < eps:
            features[i] = 0.0
        else:
            p = band / total
            p = p[p > eps]
            features[i] = -np.sum(p * np.log(p))
    return features


def markov_diag_4d(entropy_seq):
    vals = entropy_seq
    if len(vals) < 2:
        return np.zeros(4)
    q25, q50, q75 = np.percentile(vals, [25, 50, 75])
    states = np.searchsorted([q25, q50, q75], vals).astype(int)
    counts = np.zeros((4, 4))
    for s, ns in zip(states[:-1], states[1:]):
        counts[s, ns] += 1
    diag = np.zeros(4)
    for i in range(4):
        row_sum = counts[i].sum()
        if row_sum > 0:
            diag[i] = counts[i, i] / row_sum
    return diag


def markov_full_16d(entropy_seq):
    vals = entropy_seq
    if len(vals) < 2:
        return np.zeros(16)
    q25, q50, q75 = np.percentile(vals, [25, 50, 75])
    states = np.searchsorted([q25, q50, q75], vals).astype(int)
    counts = np.zeros((4, 4))
    for s, ns in zip(states[:-1], states[1:]):
        counts[s, ns] += 1
    mat = np.zeros((4, 4))
    for i in range(4):
        row_sum = counts[i].sum()
        if row_sum > 0:
            mat[i] = counts[i] / row_sum
    return mat.flatten()


def compute_features(runs, feat_fn):
    features = []
    for r in runs:
        feat = feat_fn(r['entropy'])
        features.append(feat)
    return np.array(features)


def loo_auroc(features, labels):
    n = len(labels)
    if n < 3:
        return None
    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos < 2 or n_neg < 2:
        return None

    features = np.array(features)
    labels = np.array(labels)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    scores = np.zeros(n)
    for i in range(n):
        others = np.arange(n) != i
        pos_mask = others & (labels == 1)
        neg_mask = others & (labels == 0)
        pos_feats = features[pos_mask]
        neg_feats = features[neg_mask]
        centroid_pos = pos_feats.mean(axis=0)
        centroid_neg = neg_feats.mean(axis=0)
        d_pos = np.sum((features[i] - centroid_pos) ** 2)
        d_neg = np.sum((features[i] - centroid_neg) ** 2)
        scores[i] = d_neg - d_pos

    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        return None
    return max(auroc, 1 - auroc)


def bootstrap_ci(aurocs, n_boot=10000):
    aurocs = np.array(aurocs)
    n = len(aurocs)
    rng = np.random.RandomState(42)
    boot_medians = np.zeros(n_boot)
    for b in range(n_boot):
        sample = rng.choice(aurocs, size=n, replace=True)
        boot_medians[b] = np.median(sample)
    lo = np.percentile(boot_medians, 2.5)
    hi = np.percentile(boot_medians, 97.5)
    med = np.median(aurocs)
    flip = np.mean(aurocs < 0.5) * 100
    return med, lo, hi, flip


def evaluate_method(problems, feat_fn, feat_name):
    print(f"\n=== {feat_name} ===")
    print(f"{'Model':<7} | {'Tier':<5} | {'n_wp':>4} | {'DA Median AUROC':>15} | {'95% CI':>25} | {'Flip Rate':>9}")
    print("-" * 90)

    for model_name, model_problems in problems:
        for tier in ['hard', 'medium', 'easy']:
            aurocs = []
            for pid, pdata in model_problems.items():
                if pdata['tier'] != tier:
                    continue
                runs = pdata['runs']
                labels = np.array([r['is_correct'] for r in runs])
                n_pos = labels.sum()
                n_neg = len(labels) - n_pos
                if n_pos < 2 or n_neg < 2:
                    continue
                features = compute_features(runs, feat_fn)
                auroc = loo_auroc(features, labels)
                if auroc is not None:
                    aurocs.append(auroc)

            if len(aurocs) == 0:
                print(f"{model_name:<7} | {tier:<5} | {0:>4} | {'N/A':>15} | {'N/A':>25} | {'N/A':>9}")
                continue
            med, lo, hi, flip = bootstrap_ci(aurocs)
            print(f"{model_name:<7} | {tier:<5} | {len(aurocs):>4} | {med:>15.3f} | [{lo:.3f}, {hi:.3f}]       | {flip:>6.1f}%")


def main():
    configs = [
        ('Qwen', 'outputs/bigmath_merged64/bigmath_qwen_64run.jsonl',
         'outputs/bigmath_merged64/bigmath_qwen_64run_npz/'),
        ('Llama', 'outputs/bigmath_merged64/bigmath_llama_64run.jsonl',
         'outputs/bigmath_merged64/bigmath_llama_64run_npz/'),
    ]

    all_problems = []
    for name, jsonl, npz_dir in configs:
        print(f"Loading {name}...")
        probs = load_problems(jsonl, npz_dir)
        all_problems.append((name, probs))
        print(f"  {len(probs)} problems loaded")

    evaluate_method(all_problems, fourier_spectral_entropy_4d, "Fourier Spectral Entropy")
    evaluate_method(all_problems, markov_diag_4d, "Markov Transition (4-state diagonal)")
    evaluate_method(all_problems, markov_full_16d, "Markov Transition (4x4 full matrix)")


if __name__ == '__main__':
    main()
