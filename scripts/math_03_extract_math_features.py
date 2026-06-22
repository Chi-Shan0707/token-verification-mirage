#!/usr/bin/env python3
"""
math_03_extract_math_features.py

Extract token-level statistics from Math CoT parquet files.
Reads parquet (action_log_probs, action_entropy lists), outputs feature JSONL.

Usage:
    python math_03_extract_math_features.py \
        --in_dir math/ \
        --out_path outputs/math/math_features.jsonl \
        --max_records 1000
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np

from shared_utils_features import enrich_record, build_empty_arrays, SIGNAL_NAMES

DEFAULT_IN_DIR = "math/"
DEFAULT_OUT = "outputs/math/math_features.jsonl"
DEFAULT_MAX_RECORDS = 1000


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def find_parquet_files(in_dir: str) -> list[str]:
    """Find all .parquet files under in_dir."""
    files = []
    for root, _, names in os.walk(in_dir):
        for n in names:
            if n.endswith(".parquet"):
                files.append(os.path.join(root, n))
    return sorted(files)


def process_math_parquet(pq_path: str, max_records: int, f_out) -> tuple[int, int, int]:
    """
    Process one parquet file. Returns (written, correct_count, wrong_count).
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        log("ERROR: pyarrow not installed. Run: pip install pyarrow")
        return 0, 0, 0

    log(f"Reading {pq_path} ...")
    table = pq.read_table(pq_path)
    df = table.to_pandas()
    del table

    written = 0
    correct_count = 0
    wrong_count = 0

    for _, row in df.iterrows():
        if written >= max_records:
            break

        question = row.get("question", "")
        responses = row.get("responses", [])

        for resp in responses:
            if written >= max_records:
                break

            acc_reward = float(resp.get("acc_reward", 0.0))
            log_probs = resp.get("action_log_probs", [])
            entropies = resp.get("action_entropy", [])
            # Ensure they are lists, not numpy arrays
            if hasattr(log_probs, 'tolist'):
                log_probs = log_probs.tolist()
            if hasattr(entropies, 'tolist'):
                entropies = entropies.tolist()

            # Build record
            record = {
                "domain": "math",
                "question_preview": str(question)[:200],
                "correct": acc_reward >= 0.5,
                "acc_reward": acc_reward,
                "seq_length": len(log_probs),
            }

            # Map math signals to unified naming
            # Math provides: action_log_probs -> tok_logprob
            #                action_entropy   -> tok_neg_entropy (negated)
            arrays = build_empty_arrays()
            if log_probs:
                arrays["tok_logprob"] = np.array(log_probs, dtype=np.float16)
            if entropies:
                # Store as negative entropy for consistency with code domain
                arrays["tok_neg_entropy"] = np.array([-e for e in entropies], dtype=np.float16)

            record = enrich_record(record, arrays)

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()
            written += 1

            if record["correct"]:
                correct_count += 1
            else:
                wrong_count += 1

            if written % 500 == 0:
                log(f"Written {written} math traces...")

    del df
    return written, correct_count, wrong_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract features from Math CoT parquet")
    parser.add_argument("--in_dir", default=DEFAULT_IN_DIR)
    parser.add_argument("--out_path", default=DEFAULT_OUT)
    parser.add_argument("--max_records", type=int, default=DEFAULT_MAX_RECORDS)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)

    pq_files = find_parquet_files(args.in_dir)
    if not pq_files:
        log(f"No parquet files found in {args.in_dir}")
        return

    log(f"Found {len(pq_files)} parquet files")

    total_written = 0
    total_correct = 0
    total_wrong = 0
    remaining = args.max_records

    with open(args.out_path, "w", encoding="utf-8") as f_out:
        for pq_path in pq_files:
            if remaining <= 0:
                break

            written, correct, wrong = process_math_parquet(pq_path, remaining, f_out)
            total_written += written
            total_correct += correct
            total_wrong += wrong
            remaining -= written

    log(f"Done. Written={total_written}, Correct={total_correct}, Wrong={total_wrong}")
    log(f"Correctness ratio: {100*total_correct/total_written:.1f}%" if total_written > 0 else "N/A")
    log(f"Output: {args.out_path}")


if __name__ == "__main__":
    main()
