#!/usr/bin/env python3
"""Baseline reproductions: Entropy-trajectory baseline and Kadavath et al. (2022)
on Qwen2.5-Coder-7B BigMath hard WP-eligible data.

Single-model setting: Qwen-Coder only.
Uses vLLM tensor_parallel=2 for fast batched inference.

OPTIMIZED: Batches all generation requests per phase for maximum throughput.
"""

import json
import os
import re
import sys
import time
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

QWEN_JSONL = "outputs/bigmath_merged64/bigmath_qwen_64run.jsonl"
QWEN_NPZ_DIR = "outputs/bigmath_merged64/bigmath_qwen_64run_npz/"
MODEL_PATH = "data/models/qwen2.5-coder-7b"

SAVE_DIR = "outputs/baseline_repro"
os.makedirs(SAVE_DIR, exist_ok=True)


def load_hard_problems(jsonl_path):
    problems = defaultdict(list)
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            problems[rec["problem_id"]].append(rec)
    for pid in problems:
        problems[pid].sort(key=lambda r: r["run_id"])
    result = {}
    for pid, recs in problems.items():
        if recs[0]["difficulty_tier"] != "hard":
            continue
        labels = np.array([r["is_correct"] for r in recs])
        if labels.sum() < 2 or (len(labels) - labels.sum()) < 2:
            continue
        result[pid] = {
            "labels": labels,
            "recs": recs,
            "ground_truth": recs[0].get("ground_truth", ""),
        }
    return result


def extract_boxed_answer(text):
    for pat in [r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
                r'boxed\{([^}]+)\}']:
        matches = re.findall(pat, text)
        if matches:
            return re.sub(r'[,\s]+', '', matches[-1].strip())
    return None


def extract_final_numeric(text):
    nums = re.findall(r'-?\d+\.?\d*', text)
    return nums[-1] if nums else None


def split_into_checkpoints(text, n_checkpoints=4):
    """Split trace into n_checkpoints prefix positions for the entropy-trajectory baseline"""
    paras = [p for p in text.split('\n\n') if p.strip()]
    if len(paras) < 2:
        paras = [p for p in re.split(r'(?<=[.!?])\s+', text) if p.strip()]
    if len(paras) < 2:
        return [text]
    
    accumulated = ""
    prefixes = []
    for para in paras:
        if prefixes:
            accumulated += "\n\n"
        accumulated += para
        prefixes.append(accumulated)
    
    if len(prefixes) <= n_checkpoints:
        return prefixes
    
    indices = set()
    for frac in [0.25, 0.5, 0.75, 1.0]:
        idx = min(int(len(prefixes) * frac), len(prefixes) - 1)
        indices.add(idx)
    return [prefixes[i] for i in sorted(indices)]


def loo_auroc_scalar(values, labels):
    n = len(values)
    c_idx = np.where(labels)[0]
    w_idx = np.where(~labels)[0]
    if len(c_idx) < 2 or len(w_idx) < 2:
        return np.nan, np.nan
    scores = np.empty(n)
    for hold in range(n):
        mask = np.ones(n, dtype=bool)
        mask[hold] = False
        cm = values[mask & labels].mean()
        wm = values[mask & (~labels)].mean()
        scores[hold] = abs(values[hold] - wm) - abs(values[hold] - cm)
    raw = roc_auc_score(labels, scores)
    return raw, max(raw, 1.0 - raw)


def bootstrap_median_ci(data, n_boot=10000, ci=0.95):
    if len(data) < 2:
        return np.nan, (np.nan, np.nan)
    rng = np.random.default_rng(42)
    meds = np.empty(n_boot)
    for b in range(n_boot):
        meds[b] = np.median(rng.choice(data, size=len(data), replace=True))
    lo = np.percentile(meds, (1 - ci) / 2 * 100)
    hi = np.percentile(meds, (1 + ci) / 2 * 100)
    return np.median(data), (lo, hi)


def print_method_result(name, pairs):
    if not pairs:
        print(f"  {name:<30} | N/A")
        return
    raws = np.array([p[0] for p in pairs])
    das = np.array([p[1] for p in pairs])
    flip = np.mean(raws < 0.5)
    med, (lo, hi) = bootstrap_median_ci(das)
    print(f"  {name:<30} | DA={med:.3f} [{lo:.3f},{hi:.3f}] flip={flip*100:.1f}% n={len(das)}")


# ============================================================
# Phase 1: Entropy-trajectory baseline
# ============================================================

def run_entropy_trajectory(llm, problems):
    print(f"\n{'='*70}")
    print("Phase 1: Entropy-trajectory baseline — Entropy Trajectory Monotonicity")
    print(f"{'='*70}")
    
    from vllm import SamplingParams
    entropy_sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=150, n=1)
    
    M_SAMPLES = 5
    EPSILON = 0.01
    
    all_prompts = []
    prompt_meta = []
    
    print("  Building prompt list...")
    for pid, pdata in problems.items():
        for run_idx, rec in enumerate(pdata["recs"]):
            text = rec["generated_text"]
            prefixes = split_into_checkpoints(text, n_checkpoints=4)
            for cp_idx, prefix in enumerate(prefixes):
                for sample_i in range(M_SAMPLES):
                    all_prompts.append(prefix)
                    prompt_meta.append({
                        "pid": pid, "run_idx": run_idx,
                        "cp_idx": cp_idx, "sample_i": sample_i,
                    })
    
    n_total = len(all_prompts)
    print(f"  Total prompts to generate: {n_total}")
    print(f"  (≈{n_total/62:.0f} per problem, {n_total/62/64:.1f} per run)")
    
    BATCH_SIZE = 256
    all_completions = []
    
    t0 = time.time()
    for batch_start in range(0, n_total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_total)
        batch = all_prompts[batch_start:batch_end]
        
        try:
            outputs = llm.generate(batch, entropy_sp)
            for out in outputs:
                all_completions.append(out.outputs[0].text)
        except Exception as e:
            print(f"  Batch error at {batch_start}: {e}")
            all_completions.extend([""] * (batch_end - batch_start))
        
        if batch_start % (BATCH_SIZE * 20) == 0:
            elapsed = time.time() - t0
            rate = (batch_start + len(batch)) / elapsed
            eta = (n_total - batch_start - len(batch)) / rate if rate > 0 else 0
            print(f"  [{batch_start}/{n_total}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s")
    
    gen_time = time.time() - t0
    print(f"  Generation done in {gen_time:.1f}s ({n_total/gen_time:.0f} prompts/s)")
    
    print("  Extracting answers and computing trajectories...")
    
    problem_trajectories = defaultdict(lambda: defaultdict(list))
    
    for i, completion in enumerate(all_completions):
        meta = prompt_meta[i]
        pid = meta["pid"]
        run_idx = meta["run_idx"]
        cp_idx = meta["cp_idx"]
        
        ans = extract_boxed_answer(completion)
        if ans is None:
            ans = extract_final_numeric(completion)
        
        problem_trajectories[pid][run_idx].append({
            "cp_idx": cp_idx,
            "answer": ans,
        })
    
    print("  Computing monotonicity scores...")
    
    entropy_viol_pairs = []
    entropy_range_pairs = []
    
    for pid, pdata in problems.items():
        labels = pdata["labels"]
        n_runs = len(pdata["recs"])
        
        violation_scores = np.full(n_runs, np.nan)
        entropy_range_scores = np.full(n_runs, np.nan)
        
        for run_idx in range(n_runs):
            if run_idx not in problem_trajectories[pid]:
                continue
            
            traj_data = problem_trajectories[pid][run_idx]
            grouped = defaultdict(list)
            for item in traj_data:
                grouped[item["cp_idx"]].append(item["answer"])
            
            trajectory = []
            for cp_idx in sorted(grouped.keys()):
                answers = [a for a in grouped[cp_idx] if a is not None]
                if len(answers) < 2:
                    trajectory.append(None)
                    continue
                freq = Counter(answers)
                total = len(answers)
                h = -sum((c / total) * np.log(c / total) for c in freq.values())
                trajectory.append(h)
            
            valid_h = [(i, h) for i, h in enumerate(trajectory) if h is not None]
            if len(valid_h) < 2:
                continue
            
            violations = 0
            for j in range(1, len(valid_h)):
                _, h_prev = valid_h[j - 1]
                _, h_curr = valid_h[j]
                if h_curr > h_prev + EPSILON:
                    violations += 1
            
            violation_scores[run_idx] = -float(violations)
            
            if len(valid_h) >= 2:
                h_vals = [h for _, h in valid_h]
                entropy_range_scores[run_idx] = h_vals[0] - h_vals[-1]
        
        valid = ~np.isnan(violation_scores)
        if valid.sum() >= 4 and len(set(labels[valid])) >= 2:
            raw, da = loo_auroc_scalar(violation_scores[valid], labels[valid])
            if not np.isnan(da):
                entropy_viol_pairs.append((raw, da))
        
        valid2 = ~np.isnan(entropy_range_scores)
        if valid2.sum() >= 4 and len(set(labels[valid2])) >= 2:
            raw, da = loo_auroc_scalar(entropy_range_scores[valid2], labels[valid2])
            if not np.isnan(da):
                entropy_range_pairs.append((raw, da))
    
    print(f"\n  --- Entropy-trajectory baseline Results (Qwen Hard, n={len(problems)}) ---")
    print_method_result("Entropy-ViolationCount", entropy_viol_pairs)
    print_method_result("Entropy-Range (H0-HN)", entropy_range_pairs)
    
    return {"entropy_viol": entropy_viol_pairs, "entropy_range": entropy_range_pairs}


# ============================================================
# Phase 2: Kadavath et al. — P(True) Self-Evaluation
# ============================================================

def run_kadavath(llm, problems):
    print(f"\n{'='*70}")
    print("Phase 2: Kadavath et al. (2022) — P(True) Self-Evaluation")
    print(f"{'='*70}")
    
    from vllm import SamplingParams
    kad_sp = SamplingParams(temperature=0.0, max_tokens=5, n=1)
    
    print("  Building prompts...")
    
    all_prompts = []
    prompt_pids = []
    prompt_run_indices = []
    
    for pid, pdata in problems.items():
        gt = pdata["ground_truth"]
        for run_idx, rec in enumerate(pdata["recs"]):
            text = rec["generated_text"]
            proposed = extract_boxed_answer(text)
            if proposed is None:
                proposed = extract_final_numeric(text)
            if proposed is None:
                proposed = "unknown"
            
            prompt = f"""Below is a mathematical problem and a proposed solution.

Proposed final answer: {proposed}
Correct answer: {gt}

Is the proposed answer correct? Respond with only "True" or "False".

Answer:"""
            all_prompts.append(prompt)
            prompt_pids.append(pid)
            prompt_run_indices.append(run_idx)
    
    n_total = len(all_prompts)
    print(f"  Total prompts: {n_total}")
    
    BATCH_SIZE = 512
    all_responses = []
    
    t0 = time.time()
    for batch_start in range(0, n_total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_total)
        batch = all_prompts[batch_start:batch_end]
        
        try:
            outputs = llm.generate(batch, kad_sp)
            for out in outputs:
                all_responses.append(out.outputs[0].text.strip())
        except Exception as e:
            print(f"  Batch error at {batch_start}: {e}")
            all_responses.extend([""] * (batch_end - batch_start))
        
        if batch_start % (BATCH_SIZE * 10) == 0:
            elapsed = time.time() - t0
            rate = (batch_start + len(batch)) / elapsed
            eta = (n_total - batch_start - len(batch)) / rate if rate > 0 else 0
            print(f"  [{batch_start}/{n_total}] {elapsed:.0f}s, ETA {eta:.0f}s")
    
    gen_time = time.time() - t0
    print(f"  Generation done in {gen_time:.1f}s")
    
    print("  Computing P(True) scores...")
    
    problem_scores = defaultdict(lambda: {"scores": [], "labels": []})
    
    for i, response in enumerate(all_responses):
        pid = prompt_pids[i]
        run_idx = prompt_run_indices[i]
        
        if "true" in response.lower() and "false" not in response.lower():
            score = 1.0
        elif "false" in response.lower():
            score = 0.0
        else:
            nums = re.findall(r'(\d+)', response)
            score = min(float(nums[0]) / 100.0, 1.0) if nums else 0.5
        
        problem_scores[pid]["scores"].append((run_idx, score))
        problem_scores[pid]["labels"].append(problems[pid]["labels"][run_idx])
    
    kad_ptrue_pairs = []
    kad_binary_pairs = []
    
    for pid, data in problem_scores.items():
        scores_raw = np.array([s for _, s in sorted(data["scores"])])
        labels = np.array(data["labels"])
        
        if len(labels) < 4 or len(set(labels)) < 2:
            continue
        
        raw, da = loo_auroc_scalar(scores_raw, labels)
        if not np.isnan(da):
            kad_ptrue_pairs.append((raw, da))
        
        binary = (scores_raw > 0.5).astype(float)
        if np.std(binary) > 0:
            raw2, da2 = loo_auroc_scalar(binary, labels)
            if not np.isnan(da2):
                kad_binary_pairs.append((raw2, da2))
    
    print(f"\n  --- Kadavath Results (Qwen Hard, n={len(problems)}) ---")
    print_method_result("Kadavath-P(True)-continuous", kad_ptrue_pairs)
    print_method_result("Kadavath-P(True)-binary", kad_binary_pairs)
    
    return {"kad_ptrue": kad_ptrue_pairs, "kad_binary": kad_binary_pairs}


# ============================================================
# Main
# ============================================================

def main():
    from vllm import LLM
    
    t0 = time.time()
    
    print("Loading Qwen2.5-Coder-7B (tensor_parallel=2)...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=2,
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=0.9,
        disable_log_stats=True,
    )
    print(f"Model loaded in {time.time()-t0:.1f}s")
    
    print("Loading data...")
    problems = load_hard_problems(QWEN_JSONL)
    print(f"  {len(problems)} hard WP-eligible problems")
    
    entropy_results = run_entropy_trajectory(llm, problems)
    kad_results = run_kadavath(llm, problems)
    
    total_time = time.time() - t0
    
    print(f"\n{'='*70}")
    print("SUMMARY — All Baseline Results (Qwen Hard)")
    print(f"{'='*70}")
    print_method_result("Entropy-ViolationCount", entropy_results.get("entropy_viol", []))
    print_method_result("Entropy-Range", entropy_results.get("entropy_range", []))
    print_method_result("Kadavath-P(True)", kad_results.get("kad_ptrue", []))
    print_method_result("Kadavath-Binary", kad_results.get("kad_binary", []))
    
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f}min)")
    
    save_path = os.path.join(SAVE_DIR, "baseline_repro_results.json")
    with open(save_path, "w") as f:
        json.dump({
            "entropy_viol": entropy_results.get("entropy_viol", []),
            "entropy_range": entropy_results.get("entropy_range", []),
            "kad_ptrue": kad_results.get("kad_ptrue", []),
            "kad_binary": kad_results.get("kad_binary", []),
            "total_time_s": total_time,
            "n_problems": len(problems),
            "model": "Qwen2.5-Coder-7B",
            "tier": "hard",
        }, f, indent=2)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
