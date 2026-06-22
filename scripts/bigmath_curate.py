#!/usr/bin/env python3
"""Curate BigMath: select 400 problems across 3 difficulty tiers."""
import json
import os
import pyarrow.parquet as pq
import numpy as np
from datetime import datetime

PARQUET_PATH = "data/datasets/bigmath/downloads/0cf600ebeff531417bc7a5ac6a4dc5eae2faa9473bc184fa4b16ede47dbf941e"
OUT_PATH = "outputs/bigmath_curated_400.jsonl"
SEED = 42

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    dt = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = pq.read_table(PARQUET_PATH).to_pandas()
    sr = df["llama8b_solve_rate"]

    # Define tiers
    hard = df[sr < 0.3]
    medium = df[(sr >= 0.3) & (sr < 0.6)]
    easy = df[sr >= 0.6]

    N_HARD, N_MEDIUM, N_EASY = 140, 130, 130
    rng = np.random.default_rng(SEED)

    selected = []
    for label, subset, n in [("hard", hard, N_HARD), ("medium", medium, N_MEDIUM), ("easy", easy, N_EASY)]:
        indices = rng.choice(len(subset), size=min(n, len(subset)), replace=False)
        for idx in indices:
            row = subset.iloc[idx]
            selected.append({
                "problem_id": f"bigmath_{label}_{len(selected):04d}",
                "prompt": str(row["prompt"]),
                "solution": str(row["solution"]),
                "source": str(row["source"]),
                "domain": list(row["domain"]),
                "llama8b_solve_rate": float(row["llama8b_solve_rate"]),
                "difficulty_tier": label,
            })

    with open(OUT_PATH, "w") as f:
        for rec in selected:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    hard_count = sum(1 for r in selected if r["difficulty_tier"] == "hard")
    medium_count = sum(1 for r in selected if r["difficulty_tier"] == "medium")
    easy_count = sum(1 for r in selected if r["difficulty_tier"] == "easy")

    print(f"Curated {len(selected)} problems -> {OUT_PATH}")
    print(f"  Hard (sr<0.3): {hard_count}")
    print(f"  Medium (0.3<=sr<0.6): {medium_count}")
    print(f"  Easy (sr>=0.6): {easy_count}")

    rates = [r["llama8b_solve_rate"] for r in selected]
    print(f"  Solve rate range: [{min(rates):.3f}, {max(rates):.3f}]")
    print(f"  Mean solve rate: {np.mean(rates):.3f}")

if __name__ == "__main__":
    main()
