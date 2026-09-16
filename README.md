# token-verification-mirage

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](requirements.txt)

**Contributor:** [Yuhan Chi](https://chi-shan0707.github.io/) ([@Chi-Shan0707](https://github.com/Chi-Shan0707))

**A controlled audit of token-level verification signals for LLM math reasoning.**

This repository contains the paper artifacts and analysis code for:

> **Token-Level Verification under Controlled Evaluation: Protocol Sensitivity Shapes Apparent Performance**  
> Yuhan Chi, Fudan University  
> Accepted as a poster at the **ICML 2026 Workshop on AI for Math (AI4Math)**  
> [OpenReview](https://openreview.net/forum?id=wRImV3kfR1) · [Paper PDF](https://openreview.net/pdf?id=wRImV3kfR1) · [Workshop](https://ai4math2026.github.io/)

## Overview

Token entropy, log-probability, and confidence scores are attractive verification signals because they are available during generation. This project does **not** propose a new verification method. Instead, it audits what these shallow token-level signals can and cannot distinguish—correct from incorrect math reasoning traces—after common evaluation artifacts are controlled for.

The paper is a diagnostic study. Before token-level signals are used as low-cost filters, auxiliary rerankers, or baselines for stronger verifiers, their evaluation protocol should not overstate their standalone discriminative power. The main deliverable is a reproducible evaluation framework—the Standard Evaluation Protocol (SEP) below—plus the analysis code in `scripts/`.

The four core controls:

- **Within-problem evaluation:** compare runs from the same problem rather than pooling all problems.
- **Leave-one-run-out (LOO) scoring:** avoid in-sample scoring for within-problem methods.
- **Fixed-direction reporting:** separate deployable fixed-direction AUROC from oracle-style direction-agnostic (DA) AUROC.
- **Permutation-null calibration:** calibrate DA AUROC against a null baseline.

## Observed Results

The numbers below are the paper's observations within its studied scope (see *Scope and Caveats*). They should be read as findings about a specific setting, not as general claims.

| Setting | Observation |
|---|---|
| Protocol sensitivity | On WP-eligible BigMath hard problems, protocol choices shift AUROC by up to about **0.18**. |
| Shallow token statistics | Under within-problem LOO and DA scoring, the analyzed methods cluster around **0.60--0.75 AUROC**, with substantially overlapping bootstrap CIs. |
| Final-token entropy | Final-token entropy drops from **0.72--0.75** DA AUROC to **0.47--0.48** fixed-direction AUROC (below chance). |
| Direction-agnostic null | The permutation null is about **0.58--0.60**, not 0.50. |
| Semantic self-evaluation | A Kadavath-style P(True) baseline reaches higher DA AUROC than the shallow token statistics in the same evaluation setting. |

A note on interpretation: the overlapping CIs do **not** establish that the methods are equivalent. With roughly 62--64 hard problems per model, the minimum detectable effect is about 0.12 AUROC. DA AUROC is a per-problem oracle (it selects the better scoring direction post-hoc) and should be treated as an upper bound, not a deployment estimate.

## Scope and Caveats

These results are specific to, and should be understood within, the following limits:

- Three model instances at 7B--32B scale (Qwen, Llama) on text-only mathematical word problems (MATH, BigMath). Reasoning-specialized architectures, multimodal math, and structured generations are untested.
- The WP-eligibility filter (≥2 correct and ≥2 wrong runs per problem) excludes problems a model always solves or never solves—about 56% of hard problems for Llama. Results apply to problems where within-problem evaluation is feasible.
- Sample size limits the minimum detectable effect to ≈0.12 AUROC at 80% power and α=0.05; smaller method differences cannot be resolved.
- The observed AUROC band is not an information-theoretic ceiling. It is specific to shallow statistical features on WP-eligible hard problems, modulated by model capability, and partially metric-induced (the DA null is ≈0.58--0.60).

## Repository Layout

```text
paper/
  paper.tex                  ICML camera-ready LaTeX source
  paper.pdf                  latest ICML camera-ready PDF
  icml2026.sty/.bst          ICML style files used by paper.tex
  algorithm*.sty             algorithm environment dependencies
  fig_overview_icml.pdf      overview figure used by paper.tex

scripts/
  bigmath_curate.py                          curate the BigMath subset (400 problems)
  bigmath_generate.py                        generate BigMath traces with vLLM
  00_merge_rounds.py                         merge two generation rounds into 64 runs/problem
  01_full_auroc_analysis.py                  main within-problem LOO AUROC analysis
  02_permutation_null_and_protocol.py        permutation null and protocol sensitivity
  03_mlp_truncation_selfconsistency.py       MLP, truncation, and pass@k analyses
  04_final_token_deep_dive.py                final-token entropy diagnostics
  05_boxed_answer_ablation.py                pre/post-answer ablations
  06_fixed_direction_baseline.py             fixed-direction AUROC
  07_fourier_markov.py                       Fourier and Markov features
  08_flip_rates.py                           direction-flip analysis
  09_baseline_malinin_gales.py               Malinin & Gales uncertainty baseline
  09b_baseline_malinin_gales_loo.py          LOO uncertainty baseline
  10_baseline_entropy_kadavath.py            entropy-trajectory and P(True) baselines
  11_baseline_entropy_kadavath_batched.py    batched baseline runner
  12_kadavath_fixed.py                       fixed-direction P(True) analysis
  shared_utils_features.py                   shared feature extraction utilities
  README.md                                  script index

requirements.txt
LICENSE
README.md
```

## Installation

```bash
git clone https://github.com/Chi-Shan0707/token-verification-mirage.git
cd token-verification-mirage
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Generation scripts additionally require model access and a working vLLM environment.

## Data Layout

Large generated traces and token arrays are not included in the repository. The analysis scripts expect local files using this layout:

```text
outputs/
  bigmath_curated_400.jsonl
  bigmath_merged64/
    bigmath_qwen_64run.jsonl
    bigmath_qwen_64run_npz/
    bigmath_llama_64run.jsonl
    bigmath_llama_64run_npz/

models/
  qwen2.5-coder-7b/
  LLM-Research/Meta-Llama-3___1-8B-Instruct/
```

Each run is stored as `{problem_id}_run{run_id:03d}.npz` with per-position top-20 logprob arrays (`topk_token_ids`, `topk_logprobs`, `topk_ranks`, `tok_logprob`, `tok_entropy`, `tok_top1_prob`, `tok_margin`); `tok_entropy` is the main input to most analyses.

Use `scripts/bigmath_curate.py`, `scripts/bigmath_generate.py`, and `scripts/00_merge_rounds.py` to recreate the expected BigMath inputs if traces are available locally. Note that a few scripts still reference pre-merge paths (for example `scripts/08_flip_rates.py`); check the path constants near the top of each script before running.

## Re-Running the Analyses

After preparing the data layout above, run the analysis-only scripts in order:

```bash
python scripts/01_full_auroc_analysis.py
python scripts/02_permutation_null_and_protocol.py
python scripts/03_mlp_truncation_selfconsistency.py
python scripts/04_final_token_deep_dive.py
python scripts/05_boxed_answer_ablation.py
python scripts/06_fixed_direction_baseline.py
python scripts/07_fourier_markov.py
python scripts/08_flip_rates.py
```

The prior-inspired baselines are in `scripts/09*` through `scripts/12*`. These require additional model inference and are substantially more expensive than the analysis-only scripts.

## Paper Build

```bash
cd paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
```

The minimal source needed for the paper is:

```text
paper.tex
fig_overview_icml.pdf
```

## Citation

```bibtex
@misc{chi2026tokenverificationmirage,
  title={Token-Level Verification under Controlled Evaluation: Protocol Sensitivity Shapes Apparent Performance},
  author={Chi, Yuhan},
  year={2026},
  note={Accepted as a poster at the ICML 2026 Workshop on AI for Math (AI4Math)},
  url={https://openreview.net/forum?id=wRImV3kfR1}
}
```

## License

MIT
