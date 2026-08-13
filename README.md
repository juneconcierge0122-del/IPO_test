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

**[`HANDOFF.md`](HANDOFF.md)** — the full write-up (in Chinese): environment traps, checkpoint
defects, the paper's protocol, every experiment with its numbers, and open next steps.

## Headline findings

- **The paper's U.S. result reproduces qualitatively.** TimesFM_20M_US long-short Sharpe runs
  −1.62 → 0.69 → 2.31 → **3.06** across context windows 5 / 21 / 252 / 512, against the paper's
  −18.22% (window 5) and +30.36% / Sharpe 3.66 (window 512).
- **But the economics live in the illiquid tail.** Restricted to the 500 most liquid U.S. names,
  Chronos_Small_US matches the paper's R²_OOS almost exactly (−0.54…−0.76% vs −0.59%) while its
  long-short return turns *negative*. The statistical claim replicates; the economic one does not
  survive outside small caps.
- **The published TimesFM weights are not clean.** Every checkpoint ships uninitialised memory in
  `self_attn.scaling`; the quantile heads are unusable (0% monotone); `TimesFM_20M_2023_Augmented`
  diverges outright. The loader here works around what it can.
- **TimesFM cannot be evaluated on short contexts.** `patch_len=32`, so a context under 32 days
  leaves a single patch and the 9-layer transformer degenerates to an MLP — which is exactly the
  regime an IPO study lives in.
- **No confirmed IPO signal** for either family once the paper's own return definition
  (total return minus the risk-free rate, not market-adjusted) is used.

## Layout

```
scripts/
  fintext_tsfm.py         loader for both families (handles the TimesFMForHF weights)
  tw_riskfree.py          CBC overnight interbank rate -> daily risk-free
  build_tw_ipo_panel.py   Taiwan IPO panel, four return definitions
  build_us_panel.py       U.S. panel from yfinance + Ken French daily RF
  run_experiment.py       Taiwan IPO setup A, year by year
  replicate_paper.py      U.S. replication of the paper's protocol
  market_backtest.py      Taiwan index walk-forward (checkpoint year x test year)
  inspect_raw_output.py   dissects the raw output of both families
  check_signal_source.py  is the IC just momentum in disguise?
  smoke_test.py           checkpoint load + inference health check
out/                      result tables (CSV)
```

`data/` is not tracked — rebuild it with the two `build_*` scripts.

## Environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate fintext
```

Python 3.11 / torch 2.7.1+cu126 / transformers 4.51.3 / chronos-forecasting 1.5.2 /
timesfm 1.2.7 (installed `--no-deps`, only for its torch decoder).
Two machine-specific traps — a `pip install.user` default that leaks into conda envs, and a
CUDA driver/compat mismatch that breaks `cuInit` — are documented in `HANDOFF.md` §0.

## Caveats that apply to every number here

Backtests are gross of transaction costs and assume unlimited daily short selling, as the paper's
are. The U.S. universe comes from yfinance and therefore excludes delisted names, which flatters
the short leg. Several experiments deviate from the paper's protocol; each says so where it does.
