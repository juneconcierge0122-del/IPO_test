"""Walk-forward test of the FinText checkpoints on the Taiwan market itself (0050).

Unlike the IPO setup, the index has a full 512-day history at every point, so TimesFM
gets all 16 patches and its attention is not degenerate -- this is the models' home turf
and the right place to ask "do these checkpoints work at all?".

Every checkpoint is tested on each calendar year strictly after its training cut-off:
the 2021 checkpoint is evaluated on 2022, 2023, 2024, 2025 and 2026 separately, so
performance decay as the checkpoint goes stale is visible.

Protocol
  context   : trailing 512 daily returns of 0050
  forecast  : next H trading days, re-forecast every STEP trading days
  strategy  : hold 0050 for the next STEP days if the predicted cumulative H-day return
              is > 0, else hold cash; compared against buy-and-hold over the same days
"""

import argparse
import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, "/home/ebenezer0616/IPO_test/scripts")
from fintext_tsfm import load_chronos, load_timesfm

PANEL = "/home/ebenezer0616/finlab_db/etl#adj_close.feather"
CTX = 512
ANN = 245


def market_returns(ticker="0050"):
    px = pd.read_feather(PANEL).set_index("date")
    px.index = pd.to_datetime(px.index)
    s = px[ticker].sort_index().dropna()
    return s.pct_change().dropna()


def make_windows(r, step, hor):
    """Anchor points t: context r[t-CTX:t], target r[t:t+hor]."""
    idx, ctx, fut = [], [], []
    for t in range(CTX, len(r) - hor, step):
        idx.append(r.index[t])
        ctx.append(r.values[t - CTX:t])
        fut.append(r.values[t:t + hor])
    return pd.DatetimeIndex(idx), np.asarray(ctx, np.float32), np.asarray(fut, np.float64)


def predict(repo, ctx, device, hor, batch=128, chronos_batch=8, num_samples=100):
    if "TimesFM" in repo:
        m = load_timesfm(repo, device=device)
        out = [m.forecast(list(ctx[i:i + batch]), horizon=hor)[0].numpy()
               for i in range(0, len(ctx), batch)]
    else:
        # a 512-step context x num_samples blows up the T5 encoder: keep the batch small
        m = load_chronos(repo, device=device)
        out = []
        for i in range(0, len(ctx), chronos_batch):
            s = m.predict(torch.tensor(ctx[i:i + chronos_batch]), prediction_length=hor,
                          num_samples=num_samples)
            out.append(np.median(s.numpy(), axis=1))
    del m
    torch.cuda.empty_cache()
    return np.concatenate(out).astype(np.float64)


def stats_for(pred, fut, r, dates, step):
    """pred/fut: [N, H] daily returns. Strategy holds the next `step` days when bullish."""
    p = np.clip(np.nan_to_num(pred), -1, 1)
    mse, mse0 = ((p - fut) ** 2).mean(), (fut ** 2).mean()
    cum_p = (1 + p).prod(axis=1) - 1
    cum_a = (1 + fut).prod(axis=1) - 1
    # realised return of the next `step` days after each anchor (non-overlapping)
    pos = r.index.get_indexer(dates)
    nxt = np.array([(1 + r.values[i:i + step]).prod() - 1 for i in pos])
    long_only = nxt
    timed = np.where(cum_p > 0, nxt, 0.0)
    # These checkpoints are structurally bullish (they predict a positive drift almost
    # always), so a raw sign rule degenerates into buy-and-hold. The de-meaned rule asks
    # the weaker question "is today's forecast above this model's own past forecasts?",
    # using an expanding (causal) median.
    med = pd.Series(cum_p).expanding(min_periods=20).median().shift(1).values
    timed_dm = np.where(np.nan_to_num(cum_p - med, nan=0.0) > 0, nxt, 0.0)
    n_per_year = ANN / step
    def ann(x):
        return (1 + x.mean()) ** n_per_year - 1
    def sharpe(x):
        return x.mean() / x.std() * np.sqrt(n_per_year) if x.std() > 0 else np.nan
    return {
        "n_fc": len(p),
        "R2_oos_%": 100 * (1 - mse / mse0),
        "dir_hit_%": 100 * (np.sign(cum_p) == np.sign(cum_a)).mean(),
        "corr": stats.spearmanr(cum_p, cum_a).statistic if np.std(cum_p) > 1e-12 else np.nan,
        "bull_%": 100 * (cum_p > 0).mean(),
        "strat_ann_%": 100 * ann(timed),
        "strat_dm_ann_%": 100 * ann(timed_dm),
        "bh_ann_%": 100 * ann(long_only),
        "strat_sharpe": sharpe(timed),
        "strat_dm_sharpe": sharpe(timed_dm),
        "bh_sharpe": sharpe(long_only),
        "pred_cum_mean_%": 100 * cum_p.mean(),
        "pred_cum_std_%": 100 * cum_p.std(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="0050")
    ap.add_argument("--hor", type=int, default=20)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--ckpt-years", type=int, nargs=2, default=[2010, 2023])
    ap.add_argument("--models", nargs="+",
                    default=["TimesFM_20M_Global", "TimesFM_20M_US",
                             "Chronos_Small_Global", "Chronos_Small_US"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="/home/ebenezer0616/IPO_test/out/market_backtest.csv")
    args = ap.parse_args()

    r = market_returns(args.ticker)
    dates, ctx, fut = make_windows(r, args.step, args.hor)
    yrs = dates.year.values
    print(f"{args.ticker}: {len(r)} daily returns {r.index[0].date()} -> {r.index[-1].date()}")
    print(f"anchors: {len(dates)} (every {args.step} days), horizon {args.hor} days, "
          f"context {CTX}\n")

    rows = []
    for y in range(args.ckpt_years[0], args.ckpt_years[1] + 1):
        test = yrs > y                                     # strictly after the training cut-off
        if test.sum() == 0:
            continue
        sub_yrs = yrs[test]
        for tag in args.models:
            family, variant = tag.rsplit("_", 1)
            repo = f"FinText/{family}_{y}_{variant}"
            try:
                pred = predict(repo, ctx[test], args.device, args.hor)
            except Exception as e:  # noqa: BLE001
                print(f"  SKIP {repo}: {type(e).__name__}: {str(e)[:70]}")
                continue
            sub_dates, sub_fut = dates[test], fut[test]
            for ty in sorted(set(sub_yrs.tolist())):
                k = sub_yrs == ty
                if k.sum() < 10:
                    continue
                rows.append({"ckpt_year": y, "test_year": ty, "model": tag,
                             **stats_for(pred[k], sub_fut[k], r, sub_dates[k], args.step)})
            rows.append({"ckpt_year": y, "test_year": "ALL", "model": tag,
                         **stats_for(pred, sub_fut, r, sub_dates, args.step)})
        print(f"  ckpt {y}: tested on {sorted(set(sub_yrs.tolist()))}")

    res = pd.DataFrame(rows)
    res.to_csv(args.out, index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 400)

    allrows = res[res.test_year == "ALL"]
    print("\n=== pooled over each checkpoint's whole out-of-sample period ===")
    print(allrows[["ckpt_year", "model", "n_fc", "R2_oos_%", "dir_hit_%", "corr", "bull_%",
                   "strat_ann_%", "bh_ann_%", "strat_sharpe", "bh_sharpe"]]
          .to_string(index=False, float_format=lambda x: f"{x:8.2f}"))

    per = res[res.test_year != "ALL"].copy()
    per["test_year"] = per.test_year.astype(int)
    for col in ("R2_oos_%", "dir_hit_%"):
        print(f"\n=== {col}: checkpoint year (rows) x test year (cols) ===")
        for tag in args.models:
            sub = per[per.model == tag]
            if sub.empty:
                continue
            print(f"\n-- {tag} --")
            print(sub.pivot(index="ckpt_year", columns="test_year", values=col)
                  .to_string(float_format=lambda x: f"{x:7.2f}"))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
