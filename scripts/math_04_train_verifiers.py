#!/usr/bin/env python3
"""
math_04_train_verifiers.py

Train simple verifiers on Math token-level features.
Balanced sampling: 1000 correct + 1000 wrong (or max available).

Models:
  1. Logistic Regression (linear)
  2. Random Forest (nonlinear)
  3. MLP (nonlinear)

Evaluation: GroupKFold by question (or simple stratified KFold if no group).
Metrics: AUROC, Accuracy, F1.

Usage:
    python math_04_train_verifiers.py \
        --in_path outputs/math/math_features.jsonl \
        --out_path results/math_verifier_results.json
"""

import argparse
import json
import os
import random
import warnings
from datetime import datetime

import numpy as np

warnings.filterwarnings("ignore")

DEFAULT_IN = "outputs/math/math_features.jsonl"
DEFAULT_OUT = "results/math_verifier_results.json"
RANDOM_SEED = 42


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_balanced_data(path: str, n_per_class: int = 1000) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load data and balance to n_per_class per class."""
    correct, wrong = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("correct"):
                correct.append(d)
            else:
                wrong.append(d)

    random.seed(RANDOM_SEED)
    n = min(n_per_class, len(correct), len(wrong))
    
    # If not enough wrong samples, cap to available
    if len(wrong) < n_per_class:
        log(f"WARNING: Only {len(wrong)} wrong samples available (requested {n_per_class})")
        n = min(len(correct), len(wrong))
    
    correct_sample = random.sample(correct, n)
    wrong_sample = random.sample(wrong, n)

    all_samples = correct_sample + wrong_sample
    random.shuffle(all_samples)

    # Extract features
    feature_names = []
    for key in all_samples[0].keys():
        if key.startswith(("tok_logprob", "tok_neg_entropy", "tok_conf", "tok_gini", "tok_selfcert")):
            if key.split("_")[-1] in ["mean", "var", "tail_mean", "max", "prefix_mean", "recency_mean", "slope"]:
                feature_names.append(key)

    feature_names = sorted(feature_names)
    log(f"Using {len(feature_names)} features: {feature_names}")

    X = []
    y = []
    for s in all_samples:
        row = [s.get(k, 0.0) if s.get(k) is not None else 0.0 for k in feature_names]
        X.append(row)
        y.append(1.0 if s["correct"] else 0.0)

    return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64), feature_names


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name: str) -> dict:
    """Train and evaluate a model."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model.fit(X_train_s, y_train)
    y_proba = model.predict_proba(X_test_s)[:, 1]
    y_pred = model.predict(X_test_s)

    return {
        "model": model_name,
        "auroc": float(roc_auc_score(y_test, y_proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train verifiers on Math features")
    parser.add_argument("--in_path", default=DEFAULT_IN)
    parser.add_argument("--out_path", default=DEFAULT_OUT)
    parser.add_argument("--n_per_class", type=int, default=1000)
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)

    try:
        from sklearn.model_selection import StratifiedKFold
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
    except ImportError:
        log("ERROR: scikit-learn not installed. Run: pip install scikit-learn")
        return

    X, y, feature_names = load_balanced_data(args.in_path, args.n_per_class)
    log(f"Data shape: X={X.shape}, y={y.shape}")
    log(f"Class balance: correct={int(y.sum())}, wrong={int(len(y) - y.sum())}")

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=RANDOM_SEED)

    all_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        log(f"\n--- Fold {fold_idx + 1}/{args.n_folds} ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        models = [
            ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced")),
            ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, class_weight="balanced")),
            ("MLP", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=RANDOM_SEED, early_stopping=True)),
        ]

        for name, model in models:
            result = evaluate_model(model, X_train, X_test, y_train, y_test, name)
            result["fold"] = fold_idx + 1
            all_results.append(result)
            log(f"  {name:20s}: AUROC={result['auroc']:.4f}, Acc={result['accuracy']:.4f}, F1={result['f1']:.4f}")

    # Aggregate
    log("\n" + "=" * 60)
    log("AGGREGATE RESULTS (mean ± std across folds)")
    log("=" * 60)

    summary = {}
    for name in ["LogisticRegression", "RandomForest", "MLP"]:
        rows = [r for r in all_results if r["model"] == name]
        for metric in ["auroc", "accuracy", "f1"]:
            vals = [r[metric] for r in rows]
            summary[f"{name}_{metric}"] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }
            log(f"  {name:20s} {metric:10s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Save
    output = {
        "feature_names": feature_names,
        "n_samples": len(y),
        "n_folds": args.n_folds,
        "per_fold_results": all_results,
        "summary": summary,
    }

    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    log(f"\nResults saved to {args.out_path}")


if __name__ == "__main__":
    main()
