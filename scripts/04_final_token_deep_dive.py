#!/usr/bin/env python3
"""
Investigate WHY final-token entropy is such a strong predictor.
Four analyses: last-N pattern, EOS confound, answer-entropy, trajectory shape.
"""

import json
import os
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
MODELS = {
    "Qwen": {
        "jsonl": "outputs/bigmath_merged64/bigmath_qwen_64run.jsonl",
        "npz_dir": "outputs/bigmath_merged64/bigmath_qwen_64run_npz/",
        "eos_ids": {151645, 2},
        "max_tokens": 8192,
    },
    "Llama": {
        "jsonl": "outputs/bigmath_merged64/bigmath_llama_64run.jsonl",
        "npz_dir": "outputs/bigmath_merged64/bigmath_llama_64run_npz/",
        "eos_ids": {128001, 2},
        "max_tokens": 16384,
    },
}


def load_hard_traces(jsonl_path):
    traces = []
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("difficulty_tier") == "hard":
                traces.append(d)
    return traces


def get_npz_path(npz_dir, problem_id, run_id):
    return os.path.join(npz_dir, f"{problem_id}_run{run_id:03d}.npz")


def load_entropy(npz_path):
    arr = np.load(npz_path)
    return arr["tok_entropy"].astype(np.float64)


def safe_auroc(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return roc_auc_score(y_true, y_pred)
    except Exception:
        return np.nan


def loo_auroc_by_problem(features_dict, correctness_dict):
    """
    Leave-one-problem-out AUROC.
    For each problem, assign its traces to test; train on all other problems.
    Returns mean AUROC across folds.
    """
    problems = sorted(set(features_dict.keys()))
    all_y_true = []
    all_y_score = []

    for holdout_pid in problems:
        train_feat = []
        train_corr = []
        test_feat = []
        test_corr = []

        for pid in problems:
            fvals = features_dict[pid]
            cvals = correctness_dict[pid]
            if pid == holdout_pid:
                test_feat.extend(fvals)
                test_corr.extend(cvals)
            else:
                train_feat.extend(fvals)
                train_corr.extend(cvals)

        train_corr = np.array(train_corr)
        train_feat = np.array(train_feat)
        test_corr = np.array(test_corr)
        test_feat = np.array(test_feat)

        if len(np.unique(train_corr)) < 2 or len(np.unique(test_corr)) < 2:
            continue

        # Simple threshold: use mean difference direction from train
        mean_correct = train_feat[train_corr == 1].mean()
        mean_incorrect = train_feat[train_corr == 0].mean()
        if mean_correct == mean_incorrect:
            continue

        # AUROC on test (score = feature itself, direction handled by safe_auroc)
        auc = safe_auroc(test_corr, test_feat)
        if not np.isnan(auc):
            all_y_true.extend(test_corr.tolist())
            all_y_score.extend(test_feat.tolist())

    if len(all_y_true) > 0 and len(np.unique(all_y_true)) >= 2:
        return safe_auroc(all_y_true, all_y_score)
    return np.nan


def loo_auroc_binary(feature_dict, correctness_dict):
    """LOO AUROC for binary features (higher = more likely correct)."""
    return loo_auroc_by_problem(feature_dict, correctness_dict)


# ── Main ────────────────────────────────────────────────────────────────────

for model_name, cfg in MODELS.items():
    print("=" * 80)
    print(f"  MODEL: {model_name}")
    print(f"  max_tokens={cfg['max_tokens']}, EOS IDs={cfg['eos_ids']}")
    print("=" * 80)

    traces = load_hard_traces(cfg["jsonl"])
    print(f"  Loaded {len(traces)} hard traces")

    # ── Preload all data ────────────────────────────────────────────────
    data = []
    skip = 0
    for t in traces:
        npz_path = get_npz_path(cfg["npz_dir"], t["problem_id"], t["run_id"])
        if not os.path.exists(npz_path):
            skip += 1
            continue
        entropy = load_entropy(npz_path)
        token_ids = t.get("token_ids", [])
        if len(token_ids) == 0 or len(entropy) == 0:
            skip += 1
            continue
        data.append({
            "problem_id": t["problem_id"],
            "run_id": t["run_id"],
            "is_correct": t["is_correct"],
            "total_length": t["total_length"],
            "generated_text": t.get("generated_text", ""),
            "token_ids": token_ids,
            "entropy": entropy,
        })
    print(f"  Usable traces: {len(data)} (skipped {skip})")

    # Organize by problem
    by_problem = defaultdict(list)
    for d in data:
        by_problem[d["problem_id"]].append(d)

    # ═══════════════════════════════════════════════════════════════════════
    # ANALYSIS A: Last-N tokens entropy pattern
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("  ANALYSIS A: Last-N Tokens Entropy Pattern")
    print("─" * 60)

    last_n_values = [1, 3, 5, 10, 20]

    # Per-trace: compute mean entropy of last-N tokens
    feat_lastN = {n: defaultdict(list) for n in last_n_values}
    corr_lastN = defaultdict(list)

    for d in data:
        pid = d["problem_id"]
        ent = d["entropy"]
        length = len(ent)
        for n in last_n_values:
            if length >= n:
                val = float(ent[-n:].mean())
            else:
                val = float(ent.mean())
            feat_lastN[n][pid].append(val)
        corr_lastN[pid].append(int(d["is_correct"]))

    print(f"\n  {'Last-N':<10} {'LOO AUROC':<12} {'Global AUROC':<14}")
    print(f"  {'─'*10} {'─'*12} {'─'*14}")

    # Also compute global AUROC
    for n in last_n_values:
        all_feat = []
        all_corr = []
        for pid in feat_lastN[n]:
            all_feat.extend(feat_lastN[n][pid])
            all_corr.extend(corr_lastN[pid])

        loo = loo_auroc_by_problem(feat_lastN[n], corr_lastN)
        glob = safe_auroc(all_corr, all_feat)
        print(f"  last-{n:<5} {loo:<12.4f} {glob:<14.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # ANALYSIS B: EOS / Formatting confound check
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("  ANALYSIS B: EOS / Max-Token Confound Check")
    print("─" * 60)

    max_tok_limit = cfg["max_tokens"]

    ends_eos_correct = 0
    ends_eos_incorrect = 0
    no_eos_correct = 0
    no_eos_incorrect = 0
    hits_max_correct = 0
    hits_max_incorrect = 0
    not_max_correct = 0
    not_max_incorrect = 0

    feat_eos = defaultdict(list)
    feat_maxtok = defaultdict(list)
    corr_B = defaultdict(list)

    for d in data:
        pid = d["problem_id"]
        last_tok = d["token_ids"][-1]
        is_eos = last_tok in cfg["eos_ids"]
        hits_max = d["total_length"] >= (max_tok_limit - 1)

        corr_B[pid].append(int(d["is_correct"]))
        feat_eos[pid].append(int(is_eos))
        feat_maxtok[pid].append(int(hits_max))

        if is_eos:
            if d["is_correct"]:
                ends_eos_correct += 1
            else:
                ends_eos_incorrect += 1
        else:
            if d["is_correct"]:
                no_eos_correct += 1
            else:
                no_eos_incorrect += 1

        if hits_max:
            if d["is_correct"]:
                hits_max_correct += 1
            else:
                hits_max_incorrect += 1
        else:
            if d["is_correct"]:
                not_max_correct += 1
            else:
                not_max_incorrect += 1

    total = len(data)
    total_correct = sum(1 for d in data if d["is_correct"])
    total_incorrect = total - total_correct

    print(f"\n  Overall: {total_correct} correct / {total_incorrect} incorrect "
          f"({total_correct/total*100:.1f}% correct)")

    eos_total = ends_eos_correct + ends_eos_incorrect
    no_eos_total = no_eos_correct + no_eos_incorrect
    print(f"\n  Ends with EOS ({eos_total} traces, {eos_total/total*100:.1f}%):")
    if eos_total > 0:
        print(f"    Correct:   {ends_eos_correct}/{eos_total} = {ends_eos_correct/eos_total*100:.1f}%")
        print(f"    Incorrect: {ends_eos_incorrect}/{eos_total} = {ends_eos_incorrect/eos_total*100:.1f}%")
    else:
        print(f"    (no traces end with EOS)")

    print(f"\n  Does NOT end with EOS ({no_eos_total} traces, {no_eos_total/total*100:.1f}%):")
    if no_eos_total > 0:
        print(f"    Correct:   {no_eos_correct}/{no_eos_total} = {no_eos_correct/no_eos_total*100:.1f}%")
        print(f"    Incorrect: {no_eos_incorrect}/{no_eos_total} = {no_eos_incorrect/no_eos_total*100:.1f}%")
    else:
        print(f"    (all traces end with EOS)")

    # Risk ratio
    if ends_eos_incorrect > 0 and no_eos_incorrect > 0:
        p_correct_given_eos = ends_eos_correct / eos_total
        p_correct_given_no_eos = no_eos_correct / no_eos_total
        print(f"\n  P(correct|EOS)    = {p_correct_given_eos:.4f}")
        print(f"  P(correct|no_EOS) = {p_correct_given_no_eos:.4f}")
        if p_correct_given_no_eos > 0:
            print(f"  Risk ratio = {p_correct_given_eos / p_correct_given_no_eos:.3f}")

    hits_total = hits_max_correct + hits_max_incorrect
    not_max_total = not_max_correct + not_max_incorrect
    print(f"\n  Hits max_tokens ({hits_total} traces, {hits_total/total*100:.1f}%):")
    print(f"    Correct:   {hits_max_correct}/{hits_total}" +
          (f" = {hits_max_correct/hits_total*100:.1f}%" if hits_total > 0 else ""))
    print(f"    Incorrect: {hits_max_incorrect}/{hits_total}" +
          (f" = {hits_max_incorrect/hits_total*100:.1f}%" if hits_total > 0 else ""))

    print(f"\n  Below max_tokens ({not_max_total} traces, {not_max_total/total*100:.1f}%):")
    print(f"    Correct:   {not_max_correct}/{not_max_total} = {not_max_correct/not_max_total*100:.1f}%")
    print(f"    Incorrect: {not_max_incorrect}/{not_max_total} = {not_max_incorrect/not_max_total*100:.1f}%")

    if hits_total > 0:
        p_correct_given_hits = hits_max_correct / hits_total
        p_correct_given_not = not_max_correct / not_max_total
        print(f"\n  P(correct|hits_max)    = {p_correct_given_hits:.4f}")
        print(f"  P(correct|below_max)   = {p_correct_given_not:.4f}")

    print(f"\n  LOO AUROC Results:")
    loo_eos = loo_auroc_binary(feat_eos, corr_B)
    loo_max = loo_auroc_binary(feat_maxtok, corr_B)

    # For EOS: lower = more likely correct (correct traces end with EOS more)
    # So we need to flip: score = 1 - eos
    feat_eos_flip = {pid: [1 - v for v in vals] for pid, vals in feat_eos.items()}
    loo_eos_flip = loo_auroc_by_problem(feat_eos_flip, corr_B)

    # For max_tok: hitting max = more likely incorrect
    feat_max_flip = {pid: [1 - v for v in vals] for pid, vals in feat_maxtok.items()}
    loo_max_flip = loo_auroc_by_problem(feat_max_flip, corr_B)

    print(f"    ends_with_EOS as feature (higher EOS → ?): AUROC = {loo_eos:.4f}")
    print(f"    ends_with_EOS (flipped: lower EOS → correct): AUROC = {loo_eos_flip:.4f}")
    print(f"    hits_max_tokens (higher → ?): AUROC = {loo_max:.4f}")
    print(f"    hits_max_tokens (flipped: lower → correct): AUROC = {loo_max_flip:.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # ANALYSIS C: Answer-entropy separation
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("  ANALYSIS C: Answer-Entropy Separation")
    print("─" * 60)

    pre_answer_ent_correct = []
    pre_answer_ent_incorrect = []
    post_answer_ent_correct = []
    post_answer_ent_incorrect = []

    no_boxed = 0
    for d in data:
        text = d["generated_text"]
        ent = d["entropy"]
        is_corr = d["is_correct"]

        # Find last occurrence of \boxed{
        boxed_idx = text.rfind("\\boxed{")
        if boxed_idx == -1:
            boxed_idx = text.rfind("\\boxed ")
            if boxed_idx == -1:
                no_boxed += 1
                continue

        # Approximate token position by character fraction
        char_frac = boxed_idx / max(len(text), 1)
        token_pos = int(char_frac * len(ent))
        token_pos = max(1, min(token_pos, len(ent) - 1))

        pre_ent = float(ent[:token_pos].mean()) if token_pos > 0 else np.nan
        post_ent = float(ent[token_pos:].mean()) if token_pos < len(ent) else np.nan

        if is_corr:
            pre_answer_ent_correct.append(pre_ent)
            post_answer_ent_correct.append(post_ent)
        else:
            pre_answer_ent_incorrect.append(pre_ent)
            post_answer_ent_incorrect.append(post_ent)

    print(f"\n  Traces with \\boxed found: {len(data) - no_boxed}/{len(data)} "
          f"({(len(data)-no_boxed)/len(data)*100:.1f}%)")

    def fmt_stats(vals):
        vals = [v for v in vals if not np.isnan(v)]
        if not vals:
            return "N/A"
        return f"mean={np.mean(vals):.4f}, median={np.median(vals):.4f}, std={np.std(vals):.4f}"

    print(f"\n  PRE-answer entropy:")
    print(f"    Correct:   {fmt_stats(pre_answer_ent_correct)}")
    print(f"    Incorrect: {fmt_stats(pre_answer_ent_incorrect)}")

    print(f"\n  POST-answer entropy:")
    print(f"    Correct:   {fmt_stats(post_answer_ent_correct)}")
    print(f"    Incorrect: {fmt_stats(post_answer_ent_incorrect)}")

    # Effect size: Cohen's d for post-answer entropy
    if post_answer_ent_correct and post_answer_ent_incorrect:
        c_arr = np.array([x for x in post_answer_ent_correct if not np.isnan(x)])
        i_arr = np.array([x for x in post_answer_ent_incorrect if not np.isnan(x)])
        if len(c_arr) > 1 and len(i_arr) > 1:
            pooled_std = np.sqrt(
                (c_arr.std()**2 * (len(c_arr)-1) + i_arr.std()**2 * (len(i_arr)-1))
                / (len(c_arr) + len(i_arr) - 2)
            )
            cohens_d = (c_arr.mean() - i_arr.mean()) / pooled_std if pooled_std > 0 else 0
            print(f"\n  POST-answer entropy Cohen's d: {cohens_d:.4f}")

    # LOO AUROC for post-answer mean entropy
    feat_post_ent = defaultdict(list)
    corr_C = defaultdict(list)
    for d in data:
        pid = d["problem_id"]
        text = d["generated_text"]
        ent = d["entropy"]
        boxed_idx = text.rfind("\\boxed{")
        if boxed_idx == -1:
            boxed_idx = text.rfind("\\boxed ")
            if boxed_idx == -1:
                continue
        char_frac = boxed_idx / max(len(text), 1)
        token_pos = max(1, min(int(char_frac * len(ent)), len(ent) - 1))
        post_ent = float(ent[token_pos:].mean())
        feat_post_ent[pid].append(post_ent)
        corr_C[pid].append(int(d["is_correct"]))

    if feat_post_ent:
        loo_post = loo_auroc_by_problem(feat_post_ent, corr_C)
        # Also pre-answer
        feat_pre_ent = defaultdict(list)
        corr_C2 = defaultdict(list)
        for d in data:
            pid = d["problem_id"]
            text = d["generated_text"]
            ent = d["entropy"]
            boxed_idx = text.rfind("\\boxed{")
            if boxed_idx == -1:
                boxed_idx = text.rfind("\\boxed ")
                if boxed_idx == -1:
                    continue
            char_frac = boxed_idx / max(len(text), 1)
            token_pos = max(1, min(int(char_frac * len(ent)), len(ent) - 1))
            pre_ent = float(ent[:token_pos].mean())
            feat_pre_ent[pid].append(pre_ent)
            corr_C2[pid].append(int(d["is_correct"]))

        loo_pre = loo_auroc_by_problem(feat_pre_ent, corr_C2)
        print(f"\n  LOO AUROC (post-answer mean entropy): {loo_post:.4f}")
        print(f"  LOO AUROC (pre-answer mean entropy):  {loo_pre:.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # ANALYSIS D: Entropy trajectory shape at sequence end
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("  ANALYSIS D: Entropy Trajectory Shape at Sequence End")
    print("─" * 60)

    slope_correct = []
    slope_incorrect = []

    feat_slope = defaultdict(list)
    corr_D = defaultdict(list)

    for d in data:
        pid = d["problem_id"]
        ent = d["entropy"]
        length = len(ent)

        # Last 10% of tokens
        n_tail = max(2, int(length * 0.1))
        tail_ent = ent[-n_tail:]

        # Linear regression slope
        x = np.arange(len(tail_ent), dtype=np.float64)
        y = tail_ent.astype(np.float64)
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
        else:
            slope = 0.0

        feat_slope[pid].append(float(slope))
        corr_D[pid].append(int(d["is_correct"]))

        if d["is_correct"]:
            slope_correct.append(float(slope))
        else:
            slope_incorrect.append(float(slope))

    print(f"\n  Entropy slope (last 10% of tokens):")
    print(f"    Correct:   mean={np.mean(slope_correct):.6f}, "
          f"median={np.median(slope_correct):.6f}, "
          f"frac_positive={np.mean(np.array(slope_correct)>0):.3f}")
    print(f"    Incorrect: mean={np.mean(slope_incorrect):.6f}, "
          f"median={np.median(slope_incorrect):.6f}, "
          f"frac_positive={np.mean(np.array(slope_incorrect)>0):.3f}")

    loo_slope = loo_auroc_by_problem(feat_slope, corr_D)
    print(f"\n  LOO AUROC (ending entropy slope): {loo_slope:.4f}")

    # Also: last-10% mean entropy for comparison
    feat_tail_mean = defaultdict(list)
    corr_D2 = defaultdict(list)
    for d in data:
        pid = d["problem_id"]
        ent = d["entropy"]
        n_tail = max(2, int(len(ent) * 0.1))
        tail_mean = float(ent[-n_tail:].mean())
        feat_tail_mean[pid].append(tail_mean)
        corr_D2[pid].append(int(d["is_correct"]))

    loo_tail = loo_auroc_by_problem(feat_tail_mean, corr_D2)
    print(f"  LOO AUROC (last-10% mean entropy): {loo_tail:.4f}")

    # ── Summary comparison ───────────────────────────────────────────────
    print("\n" + "═" * 60)
    print(f"  SUMMARY for {model_name}")
    print("═" * 60)
    print(f"  Feature                                    LOO AUROC")
    print(f"  {'─'*50}")
    print(f"  Last-1 token entropy:                      {looo_1:.4f}" if 'looo_1' in dir() else "", end="")
    # Recompute the key AUCs for summary
    loo_l1 = loo_auroc_by_problem(feat_lastN[1], corr_lastN)
    loo_l3 = loo_auroc_by_problem(feat_lastN[3], corr_lastN)
    loo_l5 = loo_auroc_by_problem(feat_lastN[5], corr_lastN)
    loo_l10 = loo_auroc_by_problem(feat_lastN[10], corr_lastN)
    loo_l20 = loo_auroc_by_problem(feat_lastN[20], corr_lastN)

    print(f"\n  Feature                                    LOO AUROC")
    print(f"  {'─'*50}")
    print(f"  Last-1 token mean entropy:                 {loo_l1:.4f}")
    print(f"  Last-3 tokens mean entropy:                {loo_l3:.4f}")
    print(f"  Last-5 tokens mean entropy:                {loo_l5:.4f}")
    print(f"  Last-10 tokens mean entropy:               {loo_l10:.4f}")
    print(f"  Last-20 tokens mean entropy:               {loo_l20:.4f}")
    print(f"  Ends with EOS (flipped):                   {loo_eos_flip:.4f}")
    print(f"  Hits max_tokens (flipped):                 {loo_max_flip:.4f}")
    print(f"  Post-answer mean entropy:                  {loo_post:.4f}")
    print(f"  Pre-answer mean entropy:                   {loo_pre:.4f}")
    print(f"  Ending entropy slope:                      {loo_slope:.4f}")
    print(f"  Last-10% mean entropy:                     {loo_tail:.4f}")

    # Confound analysis
    print(f"\n  Confound Check:")
    # Is last-1 entropy still predictive after controlling for EOS?
    # Stratified: among EOS-ending traces, compute AUROC
    eos_feat = defaultdict(list)
    eos_corr = defaultdict(list)
    noeos_feat = defaultdict(list)
    noeos_corr = defaultdict(list)
    for d in data:
        pid = d["problem_id"]
        ent = d["entropy"]
        last_tok = d["token_ids"][-1]
        is_eos = last_tok in cfg["eos_ids"]
        last_ent = float(ent[-1])
        if is_eos:
            eos_feat[pid].append(last_ent)
            eos_corr[pid].append(int(d["is_correct"]))
        else:
            noeos_feat[pid].append(last_ent)
            noeos_corr[pid].append(int(d["is_correct"]))

    loo_eos_strat = loo_auroc_by_problem(eos_feat, eos_corr)
    loo_noeos_strat = loo_auroc_by_problem(noeos_feat, noeos_corr)
    print(f"    Last-1 entropy AUROC (only EOS-ending traces):     {loo_eos_strat:.4f}")
    print(f"    Last-1 entropy AUROC (only non-EOS traces):        {loo_noeos_strat:.4f}")

    # Among non-max-token traces
    below_feat = defaultdict(list)
    below_corr = defaultdict(list)
    for d in data:
        pid = d["problem_id"]
        if d["total_length"] < (max_tok_limit - 1):
            below_feat[pid].append(float(d["entropy"][-1]))
            below_corr[pid].append(int(d["is_correct"]))
    loo_below = loo_auroc_by_problem(below_feat, below_corr)
    print(f"    Last-1 entropy AUROC (only below max_tok traces):  {loo_below:.4f}")

    print()

print("\n" + "=" * 80)
print("  ANALYSIS COMPLETE")
print("=" * 80)
