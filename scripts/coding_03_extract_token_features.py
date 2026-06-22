#!/usr/bin/env python3
"""coding_03_extract_token_features.py

Compute rich token-level signals for each trace using the Qwen-Coder model.

Five signal families (per completion token):
  1. tok_conf  – top-1 probability (model confidence)
  2. tok_gini  – Gini coefficient of full distribution (sharpness)
  3. tok_neg_entropy – negative entropy (uncertainty strength)
  4. tok_logprob – log-prob of the actual chosen token (evidence)
  5. tok_selfcert – self-certification: p(chosen) / p(top-1)

For each family we store:
  - Full arrays (for downstream analysis)
  - Summary stats: mean, std, min, max, p25, p50, p75
  - Trend: linear slope over token positions
  - Segment-separated stats (CoT tokens vs code tokens)

Usage:
    conda activate qwenenv
    python coding_verifier/scripts/coding_03_extract_token_features.py
"""

import argparse
import json
import os
import warnings

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_PATH = (
    "models/"
    "Qwen2.5-Coder-3B-Instruct"
)
DEFAULT_IN_PATH = (
    "coding_verifier/outputs/humaneval_traces_with_correctness.jsonl"
)
DEFAULT_OUT_PATH = (
    "coding_verifier/outputs/humaneval_traces_with_token_features.jsonl"
)
DEFAULT_MAX_TOKENS = 4096


# ──────────────────────────────────────────────────────────────────────
# Core per-position computations
# ──────────────────────────────────────────────────────────────────────

def gini_coefficient(probs: np.ndarray) -> float:
    """Gini coefficient of a probability distribution (0 = uniform, 1 = peak)."""
    sorted_p = np.sort(probs)
    n = len(sorted_p)
    if n == 0:
        return 0.0
    cumsum = np.cumsum(sorted_p)
    return (n + 1 - 2.0 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0.0


def linear_trend(arr: np.ndarray) -> float:
    """Slope of a simple linear fit over positions. Negative = decreasing."""
    n = len(arr)
    if n < 3:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    slope = (n * np.sum(x * arr) - np.sum(x) * np.sum(arr)) / \
            (n * np.sum(x * x) - np.sum(x) ** 2)
    return float(slope)


def summarize(arr: np.ndarray, prefix: str) -> dict:
    """Compute summary statistics for a 1-D array."""
    if len(arr) == 0:
        return {f"{prefix}_mean": np.nan, f"{prefix}_std": np.nan,
                f"{prefix}_min": np.nan, f"{prefix}_max": np.nan,
                f"{prefix}_p25": np.nan, f"{prefix}_p50": np.nan,
                f"{prefix}_p75": np.nan, f"{prefix}_trend": np.nan,
                f"{prefix}_array": []}
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_p25": float(np.percentile(arr, 25)),
        f"{prefix}_p50": float(np.percentile(arr, 50)),
        f"{prefix}_p75": float(np.percentile(arr, 75)),
        f"{prefix}_trend": linear_trend(arr),
        f"{prefix}_array": arr.tolist(),
    }


# ──────────────────────────────────────────────────────────────────────
# Time-axis (temporal) features
# ──────────────────────────────────────────────────────────────────────

def autocorrelation(arr: np.ndarray, lag: int = 1) -> float:
    """Lag-k autocorrelation of a 1-D array (Pearson)."""
    n = len(arr)
    if n < lag + 2:
        return 0.0
    a = arr[:n - lag]
    b = arr[lag:]
    a_centered = a - np.mean(a)
    b_centered = b - np.mean(b)
    denom = np.sqrt(np.sum(a_centered ** 2) * np.sum(b_centered ** 2))
    return float(np.sum(a_centered * b_centered) / denom) if denom > 0 else 0.0


def rolling_std(arr: np.ndarray, window: int = 16) -> float:
    """Mean of rolling window standard deviations (local volatility)."""
    n = len(arr)
    if n < window:
        return float(np.std(arr)) if n > 1 else 0.0
    counts = min(n - window + 1, 50)  # sample up to 50 windows
    step = max(1, (n - window) // counts) if counts > 0 else 1
    stds = []
    for i in range(0, n - window + 1, step):
        stds.append(np.std(arr[i:i + window]))
    return float(np.mean(stds))


def cumulative_drift(arr: np.ndarray) -> float:
    """Cumulative drift: difference between last-window mean and first-window mean."""
    n = len(arr)
    if n < 10:
        return 0.0
    w = max(5, n // 10)
    return float(np.mean(arr[-w:]) - np.mean(arr[:w]))


def low_confidence_burst_ratio(arr: np.ndarray, threshold_pct: float = 25.0) -> float:
    """Fraction of tokens that fall below the given percentile in a run of 3+."""
    if len(arr) < 5:
        return 0.0
    thresh = np.percentile(arr, threshold_pct)
    below = (arr < thresh).astype(int)
    # Count positions that are part of a burst (3+ consecutive below threshold)
    burst_count = 0
    run_len = 0
    for v in below:
        if v:
            run_len += 1
        else:
            if run_len >= 3:
                burst_count += run_len
            run_len = 0
    if run_len >= 3:
        burst_count += run_len
    return float(burst_count / len(arr))


def compute_time_axis_features(arr: np.ndarray, prefix: str) -> dict:
    """Compute time-axis features for a single signal array."""
    features = {}
    if len(arr) < 3:
        for suffix in ["autocorr_lag1", "autocorr_lag5", "rolling_std",
                        "cumulative_drift", "burst_ratio"]:
            features[f"{prefix}_{suffix}"] = np.nan
        return features

    features[f"{prefix}_autocorr_lag1"] = autocorrelation(arr, lag=1)
    features[f"{prefix}_autocorr_lag5"] = autocorrelation(arr, lag=5)
    features[f"{prefix}_rolling_std"] = rolling_std(arr, window=16)
    features[f"{prefix}_cumulative_drift"] = cumulative_drift(arr)
    features[f"{prefix}_burst_ratio"] = low_confidence_burst_ratio(arr)
    return features


def compute_regime_change(cot_arr: np.ndarray, code_arr: np.ndarray, signal_name: str) -> dict:
    """Difference between CoT-segment and code-segment for a signal."""
    features = {}
    if len(cot_arr) > 0 and len(code_arr) > 0:
        features[f"{signal_name}_regime_cot_mean"] = float(np.mean(cot_arr))
        features[f"{signal_name}_regime_code_mean"] = float(np.mean(code_arr))
        features[f"{signal_name}_regime_delta"] = float(np.mean(code_arr) - np.mean(cot_arr))
        features[f"{signal_name}_regime_ratio"] = (
            float(np.mean(code_arr) / np.mean(cot_arr))
            if abs(np.mean(cot_arr)) > 1e-10 else np.nan
        )
    else:
        features[f"{signal_name}_regime_cot_mean"] = np.nan
        features[f"{signal_name}_regime_code_mean"] = np.nan
        features[f"{signal_name}_regime_delta"] = np.nan
        features[f"{signal_name}_regime_ratio"] = np.nan
    return features


# ──────────────────────────────────────────────────────────────────────
# Main computation
# ──────────────────────────────────────────────────────────────────────

def compute_token_signals(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    completion: str,
    cot_text: str,
    code_text: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict | None:
    """Compute five token-level signal families for a single trace.

    Returns a flat dict with:
      - {signal}_{stat} for full completion
      - cot_{signal}_{stat} for CoT portion only
      - code_{signal}_{stat} for code portion only
      - completion_token_count, cot_token_count, code_token_count
    """
    device = model.device

    # --- Tokenize prompt alone for boundary ---
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    prompt_len = len(prompt_ids)

    # --- Tokenize full sequence ---
    full_text = prompt + completion
    inputs = tokenizer(
        full_text, return_tensors="pt", truncation=True, max_length=max_tokens,
    ).to(device)
    input_ids = inputs["input_ids"][0]
    seq_len = len(input_ids)

    if seq_len <= prompt_len + 1:
        return None

    # --- Find CoT/code boundary in token space ---
    # Tokenize prompt+cot to find where code tokens start
    if cot_text and len(cot_text) > 0:
        cot_full_ids = tokenizer(prompt + cot_text, return_tensors="pt",
                                 truncation=True, max_length=max_tokens)["input_ids"][0]
        # Account for possible truncation differences
        # Align: find prompt+cot length in the full tokenized sequence
        cot_boundary = min(len(cot_full_ids), seq_len)
        # Clamp: must be >= prompt_len and < seq_len
        cot_boundary = max(prompt_len + 1, min(cot_boundary, seq_len - 1))
    else:
        cot_boundary = prompt_len  # no CoT, everything is code

    # --- Forward pass ---
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]  # (seq_len, vocab_size)

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # At position i, logits[i] predicts token input_ids[i+1].
    # Completion tokens: positions prompt_len .. seq_len-2
    #   predicting tokens:   prompt_len+1 .. seq_len-1

    comp_start = prompt_len
    comp_end = seq_len - 1  # exclusive for the "predicting from" positions

    # Per-position arrays (over completion only)
    arr_logprob = []      # log p(chosen_token)
    arr_conf = []         # max p (top-1 probability)
    arr_gini = []         # Gini of full distribution
    arr_neg_entropy = []  # -H(p)
    arr_selfcert = []     # p(chosen) / p(top-1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for i in range(comp_start, comp_end):
            next_id = input_ids[i + 1].item()
            lp = log_probs[i]           # (vocab,)
            p = probs[i]                # (vocab,)

            # tok_logprob: log-prob of actual token
            arr_logprob.append(lp[next_id].item())

            # tok_conf: top-1 probability
            top1_val = torch.max(p).item()
            arr_conf.append(top1_val)

            # tok_neg_entropy: negative of entropy
            entropy = -(p * lp).sum().item()
            arr_neg_entropy.append(-entropy)

            # tok_gini: Gini coefficient (computed on CPU numpy for speed)
            p_np = p.cpu().numpy()
            arr_gini.append(gini_coefficient(p_np))

            # tok_selfcert: p(chosen) / p(top-1), clipped to [0, 1]
            arr_selfcert.append(min(p[next_id].item() / top1_val, 1.0) if top1_val > 1e-10 else 0.0)

    arr_logprob = np.array(arr_logprob, dtype=np.float64)
    arr_conf = np.array(arr_conf, dtype=np.float64)
    arr_gini = np.array(arr_gini, dtype=np.float64)
    arr_neg_entropy = np.array(arr_neg_entropy, dtype=np.float64)
    arr_selfcert = np.array(arr_selfcert, dtype=np.float64)

    n_comp = len(arr_logprob)

    # --- Segment boundaries (relative to completion arrays) ---
    cot_end_rel = max(0, min(cot_boundary - prompt_len, n_comp))
    code_start_rel = cot_end_rel

    arrs = {
        "tok_logprob": arr_logprob,
        "tok_conf": arr_conf,
        "tok_gini": arr_gini,
        "tok_neg_entropy": arr_neg_entropy,
        "tok_selfcert": arr_selfcert,
    }

    # --- Build output dict ---
    result: dict = {}
    result["completion_token_count"] = n_comp
    result["cot_token_count"] = cot_end_rel
    result["code_token_count"] = n_comp - code_start_rel

    # Full completion summaries
    for name, arr in arrs.items():
        result.update(summarize(arr, name))

    # CoT-only summaries
    for name, arr in arrs.items():
        if cot_end_rel > 0:
            result.update(summarize(arr[:cot_end_rel], f"cot_{name}"))
        else:
            for suffix in ["mean", "std", "min", "max", "p25", "p50", "p75", "trend"]:
                result[f"cot_{name}_{suffix}"] = np.nan
            result[f"cot_{name}_array"] = []

    # Code-only summaries
    for name, arr in arrs.items():
        if code_start_rel < n_comp:
            result.update(summarize(arr[code_start_rel:], f"code_{name}"))
        else:
            for suffix in ["mean", "std", "min", "max", "p25", "p50", "p75", "trend"]:
                result[f"code_{name}_{suffix}"] = np.nan
            result[f"code_{name}_array"] = []

    # --- Time-axis features (full completion) ---
    for name, arr in arrs.items():
        result.update(compute_time_axis_features(arr, name))

    # --- Regime change features (CoT vs code delta) ---
    for name, arr in arrs.items():
        cot_arr = arr[:cot_end_rel] if cot_end_rel > 0 else np.array([])
        code_arr = arr[code_start_rel:] if code_start_rel < n_comp else np.array([])
        result.update(compute_regime_change(cot_arr, code_arr, name))

    return result


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute token-level signals for each trace."
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--in_path", default=DEFAULT_IN_PATH)
    parser.add_argument("--out_path", default=DEFAULT_OUT_PATH)
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    print("Loading model ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    total = 0
    success = 0

    with open(args.in_path, "r") as f_in, open(args.out_path, "w") as f_out:
        for line in tqdm(f_in, desc="Token features"):
            total += 1
            record = json.loads(line)

            stats = compute_token_signals(
                model,
                tokenizer,
                record.get("prompt", ""),
                record.get("generated_text", ""),
                record.get("cot", ""),
                record.get("code", ""),
                max_tokens=args.max_tokens,
            )

            if stats is not None:
                record.update(stats)
                success += 1

            f_out.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(f"Done. Token features computed for {success}/{total} traces.")


if __name__ == "__main__":
    main()
