# Analysis Scripts

This directory contains the scripts used for the ICML 2026 AI4Math Workshop paper.

## Execution Order

| Step | Script | Purpose |
|---:|---|---|
| 00 | `00_merge_rounds.py` | Merge BigMath generation rounds into 64-run files. |
| 01 | `01_full_auroc_analysis.py` | Main within-problem LOO AUROC analysis and bootstrap CIs. |
| 02 | `02_permutation_null_and_protocol.py` | Direction-agnostic permutation null and protocol sensitivity. |
| 03 | `03_mlp_truncation_selfconsistency.py` | Cross-problem MLP, prefix truncation, self-consistency, and pass@k. |
| 04 | `04_final_token_deep_dive.py` | Final-token entropy diagnostics. |
| 05 | `05_boxed_answer_ablation.py` | Pre-answer, post-answer, and boxed-answer ablations. |
| 06 | `06_fixed_direction_baseline.py` | Fixed-direction AUROC baselines. |
| 07 | `07_fourier_markov.py` | Fourier spectral and Markov transition features. |
| 08 | `08_flip_rates.py` | Direction-flip-rate analysis. |
| 09 | `09_baseline_malinin_gales.py` | Malinin & Gales-style uncertainty baseline. |
| 09b | `09b_baseline_malinin_gales_loo.py` | LOO version of the uncertainty baseline. |
| 10 | `10_baseline_entropy_kadavath.py` | Entropy-trajectory and Kadavath-style P(True) baselines. |
| 11 | `11_baseline_entropy_kadavath_batched.py` | Batched runner for the prior-inspired baselines. |
| 12 | `12_kadavath_fixed.py` | Fixed-direction analysis for P(True). |

## Utilities

- `bigmath_curate.py`: curate the BigMath subset.
- `bigmath_generate.py`: generate BigMath traces.
- `math_03_extract_math_features.py`: extract token features for the MATH experiments.
- `math_04_train_verifiers.py`: train MATH verifiers.
- `shared_utils_features.py`: shared feature extraction utilities.
- `generate_fig_overview_icml.py`: regenerate the overview figure.
- `generate_fig_performance_band.py`: auxiliary figure-generation script.

## Expected Inputs

Most scripts assume generated traces and token arrays under:

```text
outputs/bigmath_merged64/
  bigmath_qwen_64run.jsonl
  bigmath_qwen_64run_npz/
  bigmath_llama_64run.jsonl
  bigmath_llama_64run_npz/
```

Model-inference baselines additionally expect local model checkpoints under `models/` or an equivalent path edited in the script.
