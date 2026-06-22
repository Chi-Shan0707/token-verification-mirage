#!/usr/bin/env python3
"""Baseline reproductions for the entropy-trajectory baseline (2026) and Kadavath et al. (2022).

Uses Qwen2.5-Coder-7B loaded with vLLM on dual GPUs.
Single-model setting: only Qwen-Coder for both generation and evaluation.

METHOD 1: Entropy-trajectory baseline
- Split CoT into steps (paragraph boundaries as fallback)
- At each step prefix, sample m=5 short continuations
- Extract answers from continuations, compute answer distribution entropy
- Check if entropy trajectory is monotonically decreasing
- Score: -violation_count or monotonicity label

METHOD 2: Kadavath et al. — P(True) self-evaluation
- For each trace, construct evaluation prompt
- Ask model to assess P(True) of the proposed answer
- Extract probability as verification score

Both methods evaluated under our strict LOO protocol on hard WP-eligible problems.
"""

import json
import os
import re
import time
import numpy as np
from collections import Counter, defaultdict
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

QWEN_JSONL = "outputs/bigmath_merged64/bigmath_qwen_64run.jsonl"
QWEN_NPZ_DIR = "outputs/bigmath_merged64/bigmath_qwen_64run_npz/"
MODEL_PATH = "data/models/qwen2.5-coder-7b"

# ============================================================
# Data loading
# ============================================================

def load_hard_problems(jsonl_path, npz_dir):
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
        if tier != "hard":
            continue
        labels = np.array([r["is_correct"] for r in recs])
        n_correct = labels.sum()
        n_wrong = len(labels) - n_correct
        if n_correct < 2 or n_wrong < 2:
            continue
        result[pid] = {
            "tier": tier, "labels": labels, "recs": recs,
            "problem_text": recs[0].get("generated_text", "")[:200],
            "ground_truth": recs[0].get("ground_truth", ""),
        }
    return result


# ============================================================
# Utility: answer extraction
# ============================================================

def extract_boxed_answer(text):
    """Extract answer from \\boxed{...} pattern."""
    patterns = [
        r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
        r'boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
        r'\\boxed\s*\{([^}]+)\}',
    ]
    for pat in patterns:
        matches = re.findall(pat, text)
        if matches:
            ans = matches[-1].strip()
            ans = re.sub(r'[,\s]+', '', ans)
            return ans
    return None


def extract_final_numeric(text):
    """Extract last numeric value as fallback answer."""
    nums = re.findall(r'-?\d+\.?\d*', text)
    if nums:
        return nums[-1]
    return None


# ============================================================
# Entropy-trajectory baseline
# ============================================================

def entropy_split_steps(cot_text):
    """Split CoT into step prefixes. Use paragraph boundaries."""
    paragraphs = [p for p in cot_text.split('\n\n') if p.strip()]
    if len(paragraphs) < 2:
        sentences = re.split(r'(?<=[.!?。！？])\s+', cot_text)
        paragraphs = [s for s in sentences if s.strip()]
    if len(paragraphs) < 2:
        return [cot_text]
    
    prefixes = []
    accumulated = ""
    for i, para in enumerate(paragraphs):
        if i > 0:
            accumulated += "\n\n"
        accumulated += para
        prefixes.append(accumulated)
    return prefixes


def entropy_compute_trajectory(step_prefixes, llm, problem_text, m=5, tau=0.7, max_new_tokens=150):
    """Compute entropy trajectory using Entropy-trajectory baseline's method."""
    trajectory = []
    
    for k, prefix in enumerate(step_prefixes):
        prompt = prefix
        completions_answers = []
        
        try:
            outputs = llm.generate(
                [prompt] * m,
                sampling_params=None,  # will set below
            )
        except Exception:
            trajectory.append(None)
            continue
        
        for output in outputs:
            completion = output.outputs[0].text
            ans = extract_boxed_answer(completion)
            if ans is None:
                ans = extract_final_numeric(completion)
            if ans is not None:
                completions_answers.append(ans)
        
        if len(completions_answers) < 2:
            trajectory.append(None)
            continue
        
        freq = Counter(completions_answers)
        total = len(completions_answers)
        entropy = -sum((c / total) * np.log(c / total) for c in freq.values())
        trajectory.append(entropy)
    
    return trajectory


def entropy_check_monotonicity(trajectory, epsilon=0.01):
    """Check monotonicity and count violations."""
    valid_traj = [(i, h) for i, h in enumerate(trajectory) if h is not None]
    if len(valid_traj) < 2:
        return False, len(valid_traj), 0
    
    violation_count = 0
    for j in range(1, len(valid_traj)):
        _, h_prev = valid_traj[j - 1]
        _, h_curr = valid_traj[j]
        if h_curr > h_prev + epsilon:
            violation_count += 1
    
    is_monotone = (violation_count == 0)
    return is_monotone, len(valid_traj), violation_count


# ============================================================
# Kadavath et al. — P(True) self-evaluation
# ============================================================

def kadavath_p_true_prompt(question, proposed_answer, baseline_answers=None):
    """Construct P(True) evaluation prompt."""
    prompt = ""
    if baseline_answers:
        prompt += "Here are some proposed answers to the question:\n"
        for i, ans in enumerate(baseline_answers[:5]):
            prompt += f"{i+1}. {ans}\n"
        prompt += "\n"
    
    prompt += f"""Question: {question}

Proposed answer: {proposed_answer}

Is the proposed answer correct? Answer only "True" or "False".

The probability the proposed answer is True is:"""
    return prompt


# ============================================================
# LOO AUROC evaluation
# ============================================================

def loo_auroc_scalar(values, labels):
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


# ============================================================
# Main
# ============================================================

def main():
    from vllm import LLM, SamplingParams
    
    t0 = time.time()
    
    print("Loading Qwen2.5-Coder-7B with vLLM (tensor_parallel=2)...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=2,
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=0.9,
    )
    print(f"Model loaded in {time.time()-t0:.1f}s")
    
    # Load data
    print("Loading data...")
    problems = load_hard_problems(QWEN_JSONL, QWEN_NPZ_DIR)
    n_problems = len(problems)
    print(f"  {n_problems} hard WP-eligible problems")
    
    # ============================================================
    # Phase 1: Entropy-trajectory baseline — Entropy Trajectory
    # ============================================================
    print(f"\n{'='*70}")
    print("Entropy-trajectory baseline — Entropy Trajectory Monotonicity")
    print(f"{'='*70}")
    
    entropy_results = {}  # pid -> {scores, labels}
    entropy_sampling = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=150,
        n=1,
    )
    
    processed = 0
    for pid, pdata in problems.items():
        recs = pdata["recs"]
        labels = pdata["labels"]
        n_runs = len(recs)
        
        violation_scores = np.full(n_runs, np.nan)
        monotone_labels = np.full(n_runs, np.nan)
        
        for run_idx, rec in enumerate(recs):
            text = rec["generated_text"]
            
            # Split into steps
            prefixes = entropy_split_steps(text)
            
            # Compute entropy trajectory
            trajectory = []
            for k, prefix in enumerate(prefixes):
                try:
                    prompts = [prefix] * 5
                    outputs = llm.generate(prompts, entropy_sampling)
                    
                    answers = []
                    for output in outputs:
                        completion = output.outputs[0].text
                        ans = extract_boxed_answer(completion)
                        if ans is None:
                            ans = extract_final_numeric(completion)
                        if ans is not None:
                            answers.append(ans)
                    
                    if len(answers) >= 2:
                        freq = Counter(answers)
                        total = len(answers)
                        h = -sum((c / total) * np.log(c / total) for c in freq.values())
                        trajectory.append(h)
                    else:
                        trajectory.append(None)
                except Exception as e:
                    trajectory.append(None)
            
            is_mono, n_valid, violations = entropy_check_monotonicity(trajectory)
            if n_valid >= 2:
                violation_scores[run_idx] = -float(violations)
                monotone_labels[run_idx] = float(is_mono)
        
        entropy_results[pid] = {
            "violation_scores": violation_scores,
            "monotone_labels": monotone_labels,
            "labels": labels,
        }
        
        processed += 1
        if processed % 5 == 0:
            elapsed = time.time() - t0
            rate = processed / elapsed
            eta = (n_problems - processed) / rate if rate > 0 else 0
            print(f"  [{processed}/{n_problems}] {pid} done ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")
    
    # Evaluate Entropy-trajectory baseline results
    print(f"\n--- Entropy-trajectory baseline Results (Qwen Hard, n={n_problems}) ---")
    
    # Method 1: -violation_count as score
    entropy_viol_pairs = []
    for pid, data in entropy_results.items():
        scores = data["violation_scores"]
        labels = data["labels"]
        valid = ~np.isnan(scores)
        if valid.sum() >= 4:
            raw, da = loo_auroc_scalar(scores[valid], labels[valid])
            if not np.isnan(da):
                entropy_viol_pairs.append((raw, da))
    
    if entropy_viol_pairs:
        raws = np.array([p[0] for p in entropy_viol_pairs])
        das = np.array([p[1] for p in entropy_viol_pairs])
        med_da, (ci_lo, ci_hi) = bootstrap_median_ci(das)
        flip = np.mean(raws < 0.5)
        print(f"  Entropy-ViolationCount: DA={med_da:.3f} [{ci_lo:.3f},{ci_hi:.3f}] flip={flip*100:.1f}%")
    
    # Method 2: monotonicity as binary classifier
    entropy_mono_correct = 0
    entropy_mono_total = 0
    mono_auroc_pairs = []
    for pid, data in entropy_results.items():
        scores = data["monotone_labels"]
        labels = data["labels"]
        valid = ~np.isnan(scores)
        if valid.sum() >= 4:
            raw, da = loo_auroc_scalar(scores[valid], labels[valid])
            if not np.isnan(da):
                mono_auroc_pairs.append((raw, da))
                mono_rate = scores[labels[valid] == 1].mean()
                non_mono_rate = scores[labels[valid] == 0].mean()
    
    if mono_auroc_pairs:
        raws = np.array([p[0] for p in mono_auroc_pairs])
        das = np.array([p[1] for p in mono_auroc_pairs])
        med_da, (ci_lo, ci_hi) = bootstrap_median_ci(das)
        flip = np.mean(raws < 0.5)
        print(f"  Entropy-Monotonicity:   DA={med_da:.3f} [{ci_lo:.3f},{ci_hi:.3f}] flip={flip*100:.1f}%")
    
    # ============================================================
    # Phase 2: Kadavath et al. — P(True)
    # ============================================================
    print(f"\n{'='*70}")
    print("Kadavath et al. (2022) — P(True) Self-Evaluation")
    print(f"{'='*70}")
    
    kad_sampling = SamplingParams(
        temperature=0.0,
        max_tokens=10,
        n=1,
    )
    
    kad_results = {}
    processed = 0
    
    for pid, pdata in problems.items():
        recs = pdata["recs"]
        labels = pdata["labels"]
        n_runs = len(recs)
        ground_truth = pdata["ground_truth"]
        
        # Collect proposed answers for brainstorming context
        proposed_answers = []
        for rec in recs:
            ans = extract_boxed_answer(rec["generated_text"])
            if ans is None:
                ans = extract_final_numeric(rec["generated_text"])
            proposed_answers.append(ans if ans else "unknown")
        
        p_true_scores = np.full(n_runs, np.nan)
        
        # Build prompts for all runs
        prompts = []
        valid_indices = []
        for run_idx, rec in enumerate(recs):
            text = rec["generated_text"]
            proposed = proposed_answers[run_idx]
            
            # Zero-shot version (no brainstorming to avoid leakage)
            prompt = f"""Given the following mathematical problem and a proposed solution:

Problem reference answer: {ground_truth}

Proposed solution (last part):
...{text[-500:]}

The proposed answer is: {proposed}

Is the proposed answer True or False? The probability the proposed answer is True is:"""
            prompts.append(prompt)
            valid_indices.append(run_idx)
        
        # Batch generate
        try:
            outputs = llm.generate(prompts, kad_sampling)
            for i, output in enumerate(outputs):
                response = output.outputs[0].text.strip()
                # Extract probability
                if "true" in response.lower():
                    p_true_scores[valid_indices[i]] = 1.0
                elif "false" in response.lower():
                    p_true_scores[valid_indices[i]] = 0.0
                else:
                    nums = re.findall(r'(\d+)', response)
                    if nums:
                        p_true_scores[valid_indices[i]] = min(float(nums[0]) / 100.0, 1.0)
        except Exception as e:
            print(f"  Error on {pid}: {e}")
        
        kad_results[pid] = {"scores": p_true_scores, "labels": labels}
        processed += 1
        if processed % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{processed}/{n_problems}] ({elapsed:.0f}s)")
    
    # Evaluate Kadavath results
    print(f"\n--- Kadavath Results (Qwen Hard, n={n_problems}) ---")
    
    kad_pairs = []
    for pid, data in kad_results.items():
        scores = data["scores"]
        labels = data["labels"]
        valid = ~np.isnan(scores)
        if valid.sum() >= 4 and len(set(labels[valid])) >= 2:
            raw, da = loo_auroc_scalar(scores[valid], labels[valid])
            if not np.isnan(da):
                kad_pairs.append((raw, da))
    
    if kad_pairs:
        raws = np.array([p[0] for p in kad_pairs])
        das = np.array([p[1] for p in kad_pairs])
        med_da, (ci_lo, ci_hi) = bootstrap_median_ci(das)
        flip = np.mean(raws < 0.5)
        print(f"  Kadavath-P(True):  DA={med_da:.3f} [{ci_lo:.3f},{ci_hi:.3f}] flip={flip*100:.1f}%")
    
    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f}min)")
    
    # Save results
    save_path = "outputs/baseline_reproduction_results.json"
    save_data = {
        "entropy_violation_pairs": entropy_viol_pairs if entropy_viol_pairs else [],
        "entropy_mono_pairs": mono_auroc_pairs if mono_auroc_pairs else [],
        "kad_ptrue_pairs": kad_pairs if kad_pairs else [],
        "total_time_s": total_time,
    }
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    main()
