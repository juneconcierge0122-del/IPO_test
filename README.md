# IPO_test — FinText TSFM checkpoints, evaluated

Work on the time-series foundation models published at
[huggingface.co/FinText](https://huggingface.co/FinText) (360 checkpoints: TimesFM 8M/20M and
Chronos Tiny/Mini/Small, one per training-cutoff year 2000–2023, in Global / US / Augmented
variants), from the paper

> Rahimikia, Ni & Wang, *Re(Visiting) Time Series Foundation Models in Finance*,
> [arXiv:2511.18578](https://arxiv.org/abs/2511.18578) (Nov 2025)

Two questions: **do these checkpoints reproduce the paper's results**, and **are they useful
on IPOs** (Taiwan and U.S.), where a stock has almost no price history.

## Read this first

**[`HANDOFF.md`](HANDOFF.md)** — the single source of truth (in Chinese): metric definitions,
checkpoint defects, the replication, and every experiment with its numbers and its caveats.
Superseded result files live in `out/archive/` with an explanation of why.

## Headline findings

- **The paper's core pattern reproduces.** TimesFM_20M_US long-short Sharpe runs
  −1.62 → 0.68 → 2.31 → **3.06** across context windows 5 / 21 / 252 / 512 (paper: −18.22%
  annualised at window 5, +30.36% / Sharpe 3.66 at window 512), and the long leg dominates,
  as in the paper. Caveat: the U.S. panel is a rate-limited yfinance download — **973
  currently-listed tickers, alphabetically truncated (no N–Z)** — so levels are not
  representative of the U.S. market and carry survivorship bias.
- **The economics sit in the less liquid names.** On the 500 most liquid of those tickers,
  Chronos_Small_US matches the paper's R²_OOS almost exactly (−0.54…−0.76% vs −0.59%) while
  its long-short return turns *negative*.
- **The published TimesFM weights are not clean.** Every checkpoint ships uninitialised
  memory in `self_attn.scaling`; the quantile heads are unusable (0% monotone);
  `TimesFM_20M_2023_Augmented` diverges outright; every `Augmented` checkpoint is
  miscalibrated. The loader works around what it can.
- **TimesFM cannot be evaluated on short contexts.** `patch_len=32`, so under 32 days of
  context the 9-layer transformer degenerates to an MLP — exactly the regime an IPO study
  lives in. Chronos (one token per day) is unaffected.
- **Pooling test years overstates skill.** On the Taiwan index, pooled correlation made
  TimesFM look positive (+0.053) where the per-year value is significantly negative
  (−0.074, 0 of 14 checkpoint years positive). Aggregate per year.
- **No confirmed IPO signal** on Taiwan (881 IPOs, 17 point-in-time cohorts) once the
  paper's own return definition is used; the sole surviving variant does not clear a
  multiple-testing correction.

## Layout

```
scripts/   loaders, panel builders, experiments (see HANDOFF.md §6 for the map)
out/       valid result tables; out/archive/ holds superseded ones with a README
```

`data/` is not tracked — rebuild with the `build_*` scripts (note the yfinance
rate-limiting caveat in HANDOFF.md §1.5).

## Environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate fintext
```

Python 3.11 / torch 2.7.1+cu126 / transformers 4.51.3 / chronos-forecasting 1.5.2 /
timesfm 1.2.7 (`--no-deps`, only for its torch decoder). Two machine-specific traps
(pip user-site leakage, CUDA compat-driver mismatch) are documented in HANDOFF.md §5.

## Caveats that apply to every number here

Backtests are gross of transaction costs and assume unlimited daily short selling, as the
paper's are. The U.S. universe is a truncated, survivorship-biased yfinance sample. The
long-leg returns have not yet been tested against microstructure artefacts (bid-ask bounce
under daily rebalancing); see HANDOFF.md §1.7.
