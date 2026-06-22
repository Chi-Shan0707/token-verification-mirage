# token-verification-mirage

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](requirements.txt)

**Contributor:** [Yuhan Chi](https://chi-shan0707.github.io/) ([@Chi-Shan0707](https://github.com/Chi-Shan0707))

**Controlled evaluation for token-level verification signals in LLM math reasoning.**

This repository contains the paper artifacts and analysis code for:

> **Token-Level Verification under Controlled Evaluation: Protocol Sensitivity Shapes Apparent Performance**  
> Yuhan Chi, Fudan University  
> Accepted as a poster at the **ICML 2026 Workshop on AI for Math (AI4Math)**  
> [OpenReview](https://openreview.net/forum?id=wRImV3kfR1) · [Paper PDF](https://openreview.net/pdf?id=wRImV3kfR1) · [Workshop](https://ai4math2026.github.io/)

## Overview

Token entropy, log-probability, and confidence scores are attractive verification signals because they are available during generation. This project evaluates whether these shallow token-level signals can distinguish correct from incorrect math reasoning traces after controlling common evaluation artifacts.

The analysis focuses on four controls:

- **Within-problem evaluation:** compare runs from the same problem rather than pooling all problems.
- **Leave-one-run-out scoring:** avoid in-sample scoring for within-problem methods.
- **Fixed-direction reporting:** separate deployable fixed-direction AUROC from direction-agnostic AUROC.
- **Permutation-null calibration:** calibrate direction-agnostic AUROC against a null baseline.

## Main Results

| Setting | Result |
|---|---|
| Protocol sensitivity | On WP-eligible BigMath hard problems, protocol choices shift AUROC by up to about **0.18**. |
| Shallow token statistics | Under within-problem LOO and direction-agnostic scoring, analyzed methods cluster around **0.60--0.75 AUROC**. |
| Final-token entropy | Final-token entropy drops from **0.72--0.75** direction-agnostic AUROC to **0.47--0.48** fixed-direction AUROC. |
| Direction-agnostic null | The permutation null is about **0.58--0.60**, not 0.50. |
| Semantic self-evaluation | Kadavath-style P(True) achieves higher direction-agnostic AUROC than shallow token statistics in the same evaluation setting. |

## Repository Layout

```text
paper/
  paper.tex                  ICML camera-ready LaTeX source
  paper.pdf                  latest ICML camera-ready PDF
  icml2026.sty/.bst          ICML style files used by paper.tex
  algorithm*.sty             algorithm environment dependencies
  fig_overview_icml.pdf      overview figure used by paper.tex

scripts/
  00_merge_rounds.py                         merge generation rounds
  01_full_auroc_analysis.py                  main LOO AUROC analysis
  02_permutation_null_and_protocol.py        permutation null and protocol sensitivity
  03_mlp_truncation_selfconsistency.py       MLP, truncation, and pass@k analyses
  04_final_token_deep_dive.py                final-token entropy diagnostics
  05_boxed_answer_ablation.py                pre/post-answer ablations
  06_fixed_direction_baseline.py             fixed-direction AUROC
  07_fourier_markov.py                       Fourier and Markov features
  08_flip_rates.py                           direction-flip analysis
  09_baseline_malinin_gales.py               uncertainty baseline
  09b_baseline_malinin_gales_loo.py          LOO uncertainty baseline
  10_baseline_entropy_kadavath.py            entropy-trajectory and P(True) baselines
  11_baseline_entropy_kadavath_batched.py    batched baseline runner
  12_kadavath_fixed.py                       fixed-direction P(True) analysis
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

Generation scripts additionally require model access and a working vLLM/Transformers environment.

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

Use `scripts/bigmath_curate.py`, `scripts/bigmath_generate.py`, and `scripts/00_merge_rounds.py` to recreate the expected BigMath inputs if traces are available locally.

## Re-running the Analyses

After preparing the data layout above, run the main scripts in order:

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

Prior-inspired baselines are in `scripts/09*` through `scripts/12*`. Some of these require model inference and are more expensive than the analysis-only scripts.

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
