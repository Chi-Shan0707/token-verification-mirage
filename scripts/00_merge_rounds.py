#!/usr/bin/env python3
"""
Merge round1 + round2 BigMath traces into 64-run-per-problem JSONL files,
then run the complete paper analysis pipeline.
"""
import json, os, sys, numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

VENV = 'data/venv/bin/python3'

# ── Paths ──
R1_QWEN = 'outputs/bigmath_traces/bigmath_qwencoder_traces.jsonl'
R1_LLAMA = 'data/bigmath_llama_traces/bigmath_llama3_8b_traces.jsonl'
R2_QWEN = 'data/bigmath_round2/bigmath_qwen_round2_traces.jsonl'
R2_LLAMA = 'data/bigmath_round2/bigmath_llama_round2_traces.jsonl'
NPZ_R1_QWEN = 'outputs/bigmath_traces/bigmath_qwencoder_traces_npz'
NPZ_R1_LLAMA = 'data/bigmath_llama_traces/bigmath_llama3_8b_traces_npz'
NPZ_R2_QWEN = 'data/bigmath_round2/bigmath_qwen_round2_traces_npz'
NPZ_R2_LLAMA = 'data/bigmath_round2/bigmath_llama_round2_traces_npz'

MERGED_DIR = 'outputs/bigmath_merged64'
os.makedirs(MERGED_DIR, exist_ok=True)

def merge_traces(r1_path, r2_path, out_path, model_name):
    """Merge round1 and round2, renumbering run_ids in round2 to 32-63."""
    r1 = [json.loads(l) for l in open(r1_path)]
    r2 = [json.loads(l) for l in open(r2_path)]
    
    merged = []
    for d in r1:
        d['run_id'] = int(d['run_id'])
        d['round'] = 1
        merged.append(d)
    for d in r2:
        d['run_id'] = int(d['run_id']) + 32
        d['round'] = 2
        merged.append(d)
    
    with open(out_path, 'w') as f:
        for d in merged:
            f.write(json.dumps(d) + '\n')
    
    problems = set(d['problem_id'] for d in merged)
    rpp = Counter(d['problem_id'] for d in merged)
    correct = sum(d['is_correct'] for d in merged)
    print(f"Merged {model_name}: {len(merged)} traces, {len(problems)} problems, "
          f"{dict(Counter(rpp.values()))}, accuracy={correct/len(merged)*100:.1f}%")
    return merged

def merge_npz(r1_dir, r2_dir, out_dir, model_name):
    """Create symlinks for NPZ files, renaming round2 to run032-run063."""
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for f in sorted(os.listdir(r1_dir)):
        src = os.path.join(r1_dir, f)
        dst = os.path.join(out_dir, f)
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)
        count += 1
    
    for f in sorted(os.listdir(r2_dir)):
        # Rename: run000 -> run032, run001 -> run033, etc.
        parts = f.replace('.npz', '').split('_run')
        base = parts[0]
        run_num = int(parts[1])
        new_name = f"{base}_run{run_num + 32:03d}.npz"
        src = os.path.join(r2_dir, f)
        dst = os.path.join(out_dir, new_name)
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)
        count += 1
    
    print(f"Merged NPZ {model_name}: {count} files -> {out_dir}")

# ── Step 1: Merge ──
print("=" * 60)
print("STEP 1: Merging round1 + round2")
print("=" * 60)

MERGED_QWEN = os.path.join(MERGED_DIR, 'bigmath_qwen_64run.jsonl')
MERGED_LLAMA = os.path.join(MERGED_DIR, 'bigmath_llama_64run.jsonl')
NPZ_QWEN = os.path.join(MERGED_DIR, 'bigmath_qwen_64run_npz')
NPZ_LLAMA = os.path.join(MERGED_DIR, 'bigmath_llama_64run_npz')

merge_traces(R1_QWEN, R2_QWEN, MERGED_QWEN, 'Qwen')
merge_traces(R1_LLAMA, R2_LLAMA, MERGED_LLAMA, 'Llama')
merge_npz(NPZ_R1_QWEN, NPZ_R2_QWEN, NPZ_QWEN, 'Qwen')
merge_npz(NPZ_R1_LLAMA, NPZ_R2_LLAMA, NPZ_LLAMA, 'Llama')

# ── Step 2: Load & organize ──
print("\n" + "=" * 60)
print("STEP 2: Computing WP-eligible problems per difficulty")
print("=" * 60)

def load_data(jsonl_path, npz_dir):
    data = [json.loads(l) for l in open(jsonl_path)]
    for d in data:
        d['run_id'] = int(d.get('run_id', 0))
    return data

def get_wp_eligible(data, difficulty):
    """Return problems with ≥2 correct and ≥2 wrong in given difficulty."""
    problems = defaultdict(list)
    for d in data:
        if d['difficulty_tier'] == difficulty:
            problems[d['problem_id']].append(d)
    
    eligible = {}
    for pid, runs in problems.items():
        correct = sum(r['is_correct'] for r in runs)
        wrong = len(runs) - correct
        if correct >= 2 and wrong >= 2:
            eligible[pid] = runs
    return eligible

for model_name, data_path in [('Qwen', MERGED_QWEN), ('Llama', MERGED_LLAMA)]:
    data = load_data(data_path, None)
    print(f"\n{model_name}: {len(data)} total traces")
    for diff in ['easy', 'medium', 'hard']:
        elig = get_wp_eligible(data, diff)
        total_correct = []
        for pid, runs in elig.items():
            correct = sum(r['is_correct'] for r in runs)
            wrong = len(runs) - correct
            total_correct.append((correct, wrong, len(runs)))
        avg_correct = np.mean([c for c,w,n in total_correct])
        avg_wrong = np.mean([w for c,w,n in total_correct])
        avg_n = np.mean([n for c,w,n in total_correct])
        print(f"  {diff}: {len(elig)} WP-eligible problems, "
              f"avg {avg_correct:.1f} correct / {avg_wrong:.1f} wrong / {avg_n:.1f} total runs")

print("\nDone. Merged data ready at:", MERGED_DIR)
