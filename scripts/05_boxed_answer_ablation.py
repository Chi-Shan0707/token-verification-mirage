#!/usr/bin/env python3
"""
Boxed-answer ablation experiment (clean version).
Tests whether final-token entropy AUROC is driven by post-\\boxed{} formatting artifacts.
"""

import json
import os
import numpy as np
from collections import defaultdict
from tokenizers import Tokenizer

def load_tokenizer(model_key):
    if model_key == "qwen":
        return Tokenizer.from_file("data/models/qwen2.5-coder-7b/tokenizer.json")
    else:
        return Tokenizer.from_file("data/models/LLM-Research/Meta-Llama-3___1-8B-Instruct/tokenizer.json")

def find_boxed_token_pos(tokenizer, text):
    """Find token index for the LAST '\\boxed{' in text using offset mapping.
    Returns token_pos (the token containing the '\\' of '\\boxed{') or None.
    """
    boxed_str = "\\boxed{"
    char_idx = text.rfind(boxed_str)
    if char_idx < 0:
        return None, None
    enc = tokenizer.encode(text)
    for i, (s, e) in enumerate(enc.offsets):
        if s <= char_idx < e:
            return i, char_idx
        if s > char_idx:
            return i, char_idx
    return None, char_idx

def da_auroc(labels, scores):
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    a = (gt + 0.5 * eq) / (len(pos) * len(neg))
    return max(a, 1.0 - a)

def flip_rate(labels, scores):
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    flips = (pos[:, None] < neg[None, :]).sum()
    return flips / (len(pos) * len(neg))

def run_model(model_key, jsonl_path, npz_dir):
    tokenizer = load_tokenizer(model_key)
    
    traces = []
    with open(jsonl_path) as f:
        for line in f:
            traces.append(json.loads(line))
    
    hard = [t for t in traces if t['difficulty_tier'] == 'hard']
    by_problem = defaultdict(list)
    for t in hard:
        by_problem[t['problem_id']].append(t)
    
    wp = {k: v for k, v in by_problem.items()
          if any(t['is_correct'] for t in v) and any(not t['is_correct'] for t in v)}
    
    n_wp = len(wp)
    total = sum(len(v) for v in wp.values())
    
    cond_data = defaultdict(lambda: defaultdict(list))
    
    n_boxed = 0
    n_no_boxed = 0
    no_box_correct = 0
    no_box_incorrect = 0
    cnt = defaultdict(int)
    
    for pid, ptraces in sorted(wp.items()):
        for t in ptraces:
            text = t['generated_text']
            correct = t['is_correct']
            tlen = t['total_length']
            
            bname = os.path.basename(t['npz_path'])
            npath = os.path.join(npz_dir, bname)
            if not os.path.exists(npath):
                npath = os.path.join(npz_dir, f"{pid}_run{t['run_id']:03d}.npz")
            if not os.path.exists(npath):
                continue
            
            ent = np.load(npath)['tok_entropy']
            use_len = min(len(ent), tlen)
            if use_len == 0:
                continue
            
            # A: full sequence - last token entropy
            cond_data['full'][pid].append((correct, float(ent[-1])))
            cnt['full'] += 1
            
            # E: mask last 5
            if use_len > 5:
                cond_data['mask5'][pid].append((correct, float(ent[-6])))
                cnt['mask5'] += 1
            
            tok_pos, _ = find_boxed_token_pos(tokenizer, text)
            
            if tok_pos is None:
                n_no_boxed += 1
                no_box_correct += int(correct)
                no_box_incorrect += int(not correct)
                continue
            
            n_boxed += 1
            tok_pos = min(tok_pos, use_len - 1)
            
            # B: pre-answer last entropy
            if tok_pos > 0:
                cond_data['pre_last'][pid].append((correct, float(ent[tok_pos - 1])))
                cnt['pre_last'] += 1
            
            # D: pre-answer mean entropy
            if tok_pos > 0:
                cond_data['pre_mean'][pid].append((correct, float(np.mean(ent[:tok_pos]))))
                cnt['pre_mean'] += 1
            
            # C: post-answer mean entropy
            if tok_pos < use_len:
                post = ent[tok_pos:]
                if len(post) > 0:
                    cond_data['post_mean'][pid].append((correct, float(np.mean(post))))
                    cnt['post_mean'] += 1
    
    # Compute per-problem metrics then aggregate
    results = {}
    for cname in ['full', 'pre_last', 'pre_mean', 'post_mean', 'mask5']:
        aurocs, flips = [], []
        for pid, pairs in cond_data[cname].items():
            labels = [p[0] for p in pairs]
            scores = [p[1] for p in pairs]
            if len(set(labels)) < 2:
                continue
            a = da_auroc(labels, scores)
            f = flip_rate(labels, scores)
            if not np.isnan(a):
                aurocs.append(a)
                flips.append(f)
        results[cname] = {
            'med_auroc': np.median(aurocs) if aurocs else np.nan,
            'med_flip': np.median(flips) if flips else np.nan,
            'mean_auroc': np.mean(aurocs) if aurocs else np.nan,
            'std_auroc': np.std(aurocs) if aurocs else np.nan,
            'n_prob': len(aurocs),
            'n_traces': cnt[cname],
            'aurocs': aurocs,
        }
    
    return results, n_wp, total, n_boxed, n_no_boxed, no_box_correct, no_box_incorrect

def print_results(label, res, n_wp, total, n_boxed, n_no_box, nb_c, nb_i):
    print(f"\n{'='*70}")
    print(f"=== {label} Hard (n={n_wp} WP-eligible, {total} traces) ===")
    print(f"{'='*70}")
    print(f"{'Condition':<24} | {'DA Med AUROC':>12} | {'Mean±Std':>12} | {'Flip Rate':>9} | {'n_prob':>6} | {'n_traces':>9}")
    print(f"{'-'*24}-+-{'-'*12}-+-{'-'*12}-+-{'-'*9}-+-{'-'*6}-+-{'-'*9}")
    
    for key, lbl in [('full','Full sequence'), ('pre_last','Pre-answer (last ent)'),
                      ('pre_mean','Pre-answer (mean ent)'), ('post_mean','Post-answer (mean ent)'),
                      ('mask5','Mask last 5 tokens')]:
        r = res[key]
        med = f"{r['med_auroc']:.3f}"
        meanstd = f"{r['mean_auroc']:.3f}±{r['std_auroc']:.3f}"
        flip = f"{r['med_flip']*100:.1f}%"
        print(f"{lbl:<24} | {med:>12} | {meanstd:>12} | {flip:>9} | {r['n_prob']:>6} | {r['n_traces']:>9}")
    
    frac = n_boxed / (n_boxed + n_no_box) * 100 if (n_boxed + n_no_box) > 0 else 0
    print(f"\nFraction with \\boxed{{: {frac:.1f}%  ({n_boxed}/{n_boxed + n_no_box})")
    if n_no_box > 0:
        print(f"Without \\boxed{{: {n_no_box} traces, {nb_c} correct, {nb_i} incorrect "
              f"({nb_c/n_no_box*100:.1f}% correct)")
    
    full_a = res['full']['med_auroc']
    pre_a = res['pre_last']['med_auroc']
    pre_m = res['pre_mean']['med_auroc']
    post_a = res['post_mean']['med_auroc']
    mask_a = res['mask5']['med_auroc']
    
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC SUMMARY")
    print(f"{'='*70}")
    print(f"  Full sequence AUROC:          {full_a:.3f}")
    print(f"  Pre-answer (last ent) AUROC:  {pre_a:.3f}  (delta from full: {pre_a - full_a:+.3f})")
    print(f"  Pre-answer (mean ent) AUROC:  {pre_m:.3f}  (delta from full: {pre_m - full_a:+.3f})")
    print(f"  Post-answer (mean ent) AUROC: {post_a:.3f}  (delta from full: {post_a - full_a:+.3f})")
    print(f"  Mask last 5 AUROC:            {mask_a:.3f}  (delta from full: {mask_a - full_a:+.3f})")
    
    if abs(pre_a - full_a) < 0.02:
        print(f"\n  >> Pre-answer AUROC ≈ Full AUROC (|Δ| < 0.02)")
        print(f"     The formatting tokens after \\boxed{{}} do NOT drive the signal.")
        print(f"     Entropy discriminates from REASONING, not answer formatting.")
    elif pre_a > full_a:
        print(f"\n  >> Pre-answer AUROC > Full AUROC!")
        print(f"     Post-\\boxed{{}} tokens actually ADD NOISE to the signal.")
    else:
        print(f"\n  >> Pre-answer AUROC < Full AUROC (Δ = {pre_a - full_a:.3f})")
        print(f"     Some discriminative signal may come from answer-formatting region.")

if __name__ == "__main__":
    models = [
        ("Qwen", "qwen",
         "outputs/bigmath_merged64/bigmath_qwen_64run.jsonl",
         "outputs/bigmath_merged64/bigmath_qwen_64run_npz/"),
        ("Llama", "llama",
         "outputs/bigmath_merged64/bigmath_llama_64run.jsonl",
         "outputs/bigmath_merged64/bigmath_llama_64run_npz/"),
    ]
    
    for label, key, jsonl, npz_dir in models:
        res, n_wp, tot, n_b, n_nb, nbc, nbi = run_model(key, jsonl, npz_dir)
        print_results(label, res, n_wp, tot, n_b, n_nb, nbc, nbi)
    
    print(f"\n{'='*70}")
    print("DONE")
