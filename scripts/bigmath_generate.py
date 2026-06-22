#!/usr/bin/env python3
"""
bigmath_generate.py
Generate traces on BigMath curated problems using vLLM.
Saves BOTH trace_jsonl (human-readable) AND npz (top-20 logprobs arrays).

Usage:
    conda activate vllm_5090
    CUDA_VISIBLE_DEVICES=0 python bigmath_generate.py --model_path data/models/qwen2.5-coder-7b --out_file bigmath_qwencoder_traces
    CUDA_VISIBLE_DEVICES=1 python bigmath_generate.py --model_path data/models/DeepSeek-R1-Distill-Qwen-7B --out_file bigmath_deepseek_traces
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
from tqdm import tqdm

from vllm import LLM, SamplingParams

# ── Config ──────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = "data/models/DeepSeek-R1-Distill-Qwen-7B"
DEFAULT_DATASET_PATH = "outputs/bigmath_curated_400.jsonl"
DEFAULT_OUT_DIR = "outputs/bigmath_traces"
DEFAULT_N_SAMPLES = 32
DEFAULT_TEMPERATURE = 0.6
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TOP_LOGPROBS = 20


# ── System Prompt for Math Answer Verification ──────────────────────
MATH_SYSTEM_PROMPT = (
    "Solve the problem step by step. "
    "Think through the reasoning carefully, then output your final answer within \\boxed{...}."
)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_problems(path: str) -> list[dict]:
    problems = []
    with open(path) as f:
        for line in f:
            problems.append(json.loads(line))
    return problems


def build_messages(problem: dict) -> list[dict]:
    """Build chat messages for the problem."""
    return [
        {"role": "system", "content": MATH_SYSTEM_PROMPT},
        {"role": "user", "content": problem["prompt"]},
    ]


def extract_topk_arrays(token_ids: list[int], logprobs_list: list[dict], top_k: int = 20):
    """
    Extract per-position top-k logprob arrays.

    Returns:
        topk_token_ids:  np.float16  [seq_len, top_k]  - token ids (cast to float)
        topk_logprobs:   np.float16  [seq_len, top_k]  - log probabilities
        topk_ranks:      np.int8     [seq_len, top_k]  - rank (0-based inside top-k)
        tok_logprob:     np.float16  [seq_len]          - chosen token's logprob
        tok_entropy:     np.float16  [seq_len]          - entropy over top-k
        tok_top1_prob:   np.float16  [seq_len]          - max prob
        tok_margin:      np.float16  [seq_len]          - top1 - top2 prob gap
    """
    seq_len = len(logprobs_list)
    topk_ids = np.zeros((seq_len, top_k), dtype=np.int32)
    topk_logprobs = np.zeros((seq_len, top_k), dtype=np.float16)
    topk_ranks = np.zeros((seq_len, top_k), dtype=np.int8)
    arr_logprob = np.zeros(seq_len, dtype=np.float16)
    arr_entropy = np.zeros(seq_len, dtype=np.float16)
    arr_top1_prob = np.zeros(seq_len, dtype=np.float16)
    arr_margin = np.zeros(seq_len, dtype=np.float16)

    for pos, lp_dict in enumerate(logprobs_list):
        chosen_id = token_ids[pos]

        # Build sorted top-k in this position
        ids = list(lp_dict.keys())
        lps = np.array([lp_dict[tid].logprob for tid in ids], dtype=np.float64)
        probs = np.exp(lps)

        # Sort by prob descending
        order = np.argsort(-probs)
        ids_sorted = [ids[i] for i in order]
        lps_sorted = lps[order]
        probs_sorted = probs[order]

        n_save = min(len(ids_sorted), top_k)
        for k in range(n_save):
            topk_ids[pos, k] = ids_sorted[k]  # int32, no overflow
            topk_logprobs[pos, k] = lps_sorted[k]
            topk_ranks[pos, k] = k

        # Chosen token logprob
        if chosen_id in lp_dict:
            arr_logprob[pos] = lp_dict[chosen_id].logprob
        else:
            arr_logprob[pos] = lps_sorted[0]  # fallback to top1

        # Top1 prob
        top1_prob = float(probs_sorted[0]) if n_save > 0 else 0.0
        arr_top1_prob[pos] = top1_prob

        # Margin
        top2_prob = float(probs_sorted[1]) if n_save > 1 else 0.0
        arr_margin[pos] = top1_prob - top2_prob

        # Entropy over normalized top-k
        if n_save > 0:
            probs_norm = probs_sorted / (probs_sorted.sum() + 1e-10)
            entropy = -np.sum(probs_norm * np.log(probs_norm + 1e-10))
            arr_entropy[pos] = entropy

    return {
        "topk_token_ids": topk_ids,
        "topk_logprobs": topk_logprobs,
        "topk_ranks": topk_ranks,
        "tok_logprob": arr_logprob,
        "tok_entropy": arr_entropy,
        "tok_top1_prob": arr_top1_prob,
        "tok_margin": arr_margin,
    }


def verify_correctness(generated_text: str, ground_truth: str) -> bool:
    """Extract final answer from generated text and compare to ground truth.
    
    Handles multiple output formats:
    - \\boxed{...} (standard LaTeX)
    - \\boxed (...)
    - The answer is ... (plain text)
    - Final Answer: ... 
    """
    import re
    pred = None
    
    # Try \boxed{...} first
    boxed_matches = re.findall(r'\\boxed\s*\{([^}]*)\}', generated_text)
    if boxed_matches:
        pred = boxed_matches[-1].strip()
    else:
        # Try \boxed (...)
        paren_matches = re.findall(r'\\boxed\s*\(([^)]*)\)', generated_text)
        if paren_matches:
            pred = paren_matches[-1].strip()
        else:
            # Fallback: last non-empty line that looks like an answer
            lines = generated_text.strip().split('\n')
            for line in reversed(lines):
                line = line.strip()
                if line and not line.startswith(('think', 'Wait', 'Hmm', 'Let', 'I ', 'We ', 'So ', 'But', 'First', 'Now', 'Then', 'Thus', 'Therefore')):
                    # Try to extract from patterns like "answer is X" or "= X"
                    ans_match = re.search(r'(?:answer|result|value)\s*(?:is|=)\s*(.+)', line, re.IGNORECASE)
                    if ans_match:
                        pred = ans_match.group(1).strip()
                        break
                    # Try simple "= X"
                    eq_match = re.search(r'=\s*(.+)$', line)
                    if eq_match:
                        pred = eq_match.group(1).strip()
                        break
    
    if pred is None:
        return False
    
    # Normalize both
    gt = ground_truth.strip()
    gt = re.sub(r'\\boxed\s*\{|\}', '', gt).strip()
    gt = re.sub(r'\\boxed\s*\(|\)', '', gt).strip()
    
    # Remove spaces, standardize
    pred_norm = pred.replace(" ", "").replace(",", "").replace("$", "")
    gt_norm = gt.replace(" ", "").replace(",", "").replace("$", "")
    
    # Try exact match
    if pred_norm == gt_norm:
        return True
    
    # Try numeric comparison if both are numbers
    try:
        pred_num = float(pred_norm)
        gt_num = float(gt_norm)
        return abs(pred_num - gt_num) < 1e-6
    except (ValueError, TypeError):
        pass
    
    return False


def main():
    parser = argparse.ArgumentParser(description="BigMath Trace Generation (vLLM, top-20 logprobs)")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset_path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--out_file", default="bigmath_traces")
    parser.add_argument("--n_samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--append", action="store_true", help="Append to existing JSONL instead of overwriting")
    parser.add_argument("--skip_existing_npz", type=int, default=0, help="Skip problems that already have this many npz files")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model_name = os.path.basename(args.model_path.rstrip("/"))
    trace_path = os.path.join(args.out_dir, f"{args.out_file}.jsonl")
    npz_dir = os.path.join(args.out_dir, f"{args.out_file}_npz")
    os.makedirs(npz_dir, exist_ok=True)

    log(f"Model: {args.model_path} ({model_name})")
    log(f"Dataset: {args.dataset_path}")
    log(f"Output trace: {trace_path}")
    log(f"Output npz dir: {npz_dir}")
    log(f"N samples per problem: {args.n_samples}")
    log(f"Temperature: {args.temperature}")
    log(f"Top logprobs: 20")

    # Load problems
    problems = load_problems(args.dataset_path)
    log(f"Loaded {len(problems)} problems")

    # Load model
    log("Loading model...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=args.max_new_tokens + 512,
    )
    log("Model loaded.")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_new_tokens,
        logprobs=20,
    )

    # Generate
    stats = {"total_runs": 0, "correct": 0, "wrong": 0}

    existing_npz_counts = {}
    if args.skip_existing_npz > 0:
        import glob as _glob
        for f in os.listdir(npz_dir):
            if f.endswith(".npz"):
                pid = f.rsplit("_run", 1)[0]
                existing_npz_counts[pid] = existing_npz_counts.get(pid, 0) + 1

    jsonl_mode = "a" if args.append else "w"
    with open(trace_path, jsonl_mode) as f_trace:
        for prob_idx, problem in enumerate(tqdm(problems, desc="Problems")):
            problem_id = problem["problem_id"]

            if args.skip_existing_npz > 0 and existing_npz_counts.get(problem_id, 0) >= args.skip_existing_npz:
                log(f"  Skip {problem_id} (already has {existing_npz_counts[problem_id]} npz)")
                continue

            messages = build_messages(problem)

            # Prepare batch of identical prompts
            all_messages = [messages] * args.n_samples

            outputs = llm.chat(all_messages, sampling_params=sampling_params)

            for run_idx, output in enumerate(outputs):
                generated_text = output.outputs[0].text
                token_ids = list(output.outputs[0].token_ids)
                logprobs_list = output.outputs[0].logprobs  # list[dict[int, Logprob]]

                # Check correctness
                is_correct = verify_correctness(generated_text, problem["solution"])

                stats["total_runs"] += 1
                if is_correct:
                    stats["correct"] += 1
                else:
                    stats["wrong"] += 1

                # Extract top-k arrays
                if logprobs_list and len(logprobs_list) > 0:
                    topk_arrays = extract_topk_arrays(token_ids, logprobs_list, top_k=20)
                else:
                    topk_arrays = None

                # Save npz
                npz_filename = f"{problem_id}_run{run_idx:03d}.npz"
                npz_path = os.path.join(npz_dir, npz_filename)
                if topk_arrays is not None:
                    np.savez_compressed(npz_path, **topk_arrays)

                # Save trace JSONL record
                trace_record = {
                    "problem_id": problem_id,
                    "run_id": run_idx,
                    "model": model_name,
                    "is_correct": bool(is_correct),
                    "generated_text": generated_text,
                    "token_ids": token_ids,
                    "total_length": len(token_ids),
                    "ground_truth": problem["solution"],
                    "llama8b_solve_rate": problem["llama8b_solve_rate"],
                    "difficulty_tier": problem["difficulty_tier"],
                    "generation_params": {
                        "temperature": args.temperature,
                        "top_p": 0.95,
                        "max_tokens": args.max_new_tokens,
                    },
                    "npz_path": npz_path,
                }
                f_trace.write(json.dumps(trace_record, ensure_ascii=False) + "\n")

            if (prob_idx + 1) % 10 == 0:
                acc = stats["correct"] / max(stats["total_runs"], 1)
                log(f"  Problem {prob_idx+1}/{len(problems)} | "
                    f"Running acc: {acc:.3f} ({stats['correct']}/{stats['total_runs']})")

    # Final stats
    total = stats["total_runs"]
    acc = stats["correct"] / max(total, 1)
    log(f"\nDone! {total} runs generated.")
    log(f"Overall accuracy: {acc:.4f} ({stats['correct']}/{total})")

    # Save metadata
    meta = {
        "model_path": args.model_path,
        "model_name": model_name,
        "dataset_path": args.dataset_path,
        "n_problems": len(problems),
        "n_samples_per_problem": args.n_samples,
        "total_runs": total,
        "correct": stats["correct"],
        "wrong": stats["wrong"],
        "accuracy": float(acc),
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "top_logprobs": 20,
        "trace_path": trace_path,
        "npz_dir": npz_dir,
        "timestamp": datetime.now().isoformat(),
    }
    meta_path = os.path.join(args.out_dir, f"{args.out_file}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    log(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
