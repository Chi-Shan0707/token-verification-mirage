#!/usr/bin/env python3
"""
shared_utils_features.py

Shared feature extraction utilities for both Code and Math domains.
All domain-agnostic logic lives here.

Feature naming convention (per signal family):
  {signal}_{stat}           – full sequence
  {signal}_prefix_{stat}    – first 20% of sequence
  {signal}_recency_{stat}   – last 20% of sequence
  {signal}_tail_{stat}      – alias for recency (last 20%)
  {signal}_slope            – linear trend over full sequence

Stats computed:
  mean, var, tail_mean (last 20%), max
  + prefix_mean, recency_mean, slope
"""

import numpy as np

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

TAIL_FRAC = 0.20   # for tail_mean / recency
PREFIX_FRAC = 0.20  # for prefix

# Five signal families (unified naming across domains)
SIGNAL_NAMES = ["tok_logprob", "tok_conf", "tok_gini", "tok_neg_entropy", "tok_selfcert"]

# Base stats
BASE_STAT_NAMES = ["mean", "var", "tail_mean", "max"]

# Extended stats (prefix, recency, slope)
EXTENDED_STAT_NAMES = ["mean", "var", "tail_mean", "max", "prefix_mean", "recency_mean", "slope"]


# ──────────────────────────────────────────────────────────────────────
# Core: compute stats from a 1-D float array
# ──────────────────────────────────────────────────────────────────────

def compute_base_stats(arr: np.ndarray) -> dict[str, float | None]:
    """
    Compute 4 base statistics on a 1-D array:
      - mean, var, tail_mean (last 20%), max
    """
    if len(arr) == 0:
        return {k: None for k in BASE_STAT_NAMES}

    arr_f = arr.astype(np.float64)
    tail_n = max(1, int(len(arr_f) * TAIL_FRAC))

    return {
        "mean": float(arr_f.mean()),
        "var": float(arr_f.var()),
        "tail_mean": float(arr_f[-tail_n:].mean()),
        "max": float(arr_f.max()),
    }


def compute_extended_stats(arr: np.ndarray) -> dict[str, float | None]:
    """
    Compute extended statistics on a 1-D array:
      - mean, var, tail_mean, max           (base)
      - prefix_mean: mean of first 20%
      - recency_mean: mean of last 20% (same as tail_mean)
      - slope: linear trend coefficient over positions
    """
    if len(arr) == 0:
        return {k: None for k in EXTENDED_STAT_NAMES}

    arr_f = arr.astype(np.float64)
    n = len(arr_f)
    prefix_n = max(1, int(n * PREFIX_FRAC))
    tail_n = max(1, int(n * TAIL_FRAC))

    # Linear trend (slope)
    if n >= 3:
        x = np.arange(n, dtype=np.float64)
        denom = n * np.sum(x * x) - np.sum(x) ** 2
        if abs(denom) > 1e-12:
            slope = float((n * np.sum(x * arr_f) - np.sum(x) * np.sum(arr_f)) / denom)
        else:
            slope = 0.0
    else:
        slope = 0.0

    return {
        "mean": float(arr_f.mean()),
        "var": float(arr_f.var()),
        "tail_mean": float(arr_f[-tail_n:].mean()),
        "max": float(arr_f.max()),
        "prefix_mean": float(arr_f[:prefix_n].mean()),
        "recency_mean": float(arr_f[-tail_n:].mean()),
        "slope": slope,
    }


def enrich_record(record: dict, arrays: dict[str, np.ndarray]) -> dict:
    """
    Enrich a record with extended stats for each signal family.

    Fields added per signal:
      {signal}_mean, {signal}_var, {signal}_tail_mean, {signal}_max,
      {signal}_prefix_mean, {signal}_recency_mean, {signal}_slope
    """
    for sig_name in SIGNAL_NAMES:
        if sig_name in arrays and len(arrays[sig_name]) > 0:
            stats = compute_extended_stats(arrays[sig_name])
            for stat_name, val in stats.items():
                record[f"{sig_name}_{stat_name}"] = val
        else:
            for stat_name in EXTENDED_STAT_NAMES:
                record[f"{sig_name}_{stat_name}"] = None
    return record


def build_empty_arrays() -> dict[str, np.ndarray]:
    """Return a dict of empty arrays for all signal families."""
    return {name: np.array([], dtype=np.float16) for name in SIGNAL_NAMES}
