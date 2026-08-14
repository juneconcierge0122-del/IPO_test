"""Replicate the evaluation protocol of arXiv:2511.18578 on U.S. daily excess returns.

Protocol, verbatim from the paper:
  * input    : the trailing C daily excess returns of a single stock (C in 5/21/252/512)
  * output   : the ONE-day-ahead excess return, taken as the conditional *mean*
               (Chronos: mean over sampled paths; TimesFM: the mean head)
  * models   : point-in-time, the checkpoint for test year Y is the min(2023, Y-1) one
  * R2_OOS   : 1 - sum_it (r_{t+1} - rhat_{t+1})^2 / sum_it (r_{t+1})^2   (Gu et al. 2020,
               benchmarked against a naive forecast of zero, NOT the historical mean)
  * direction: overall / up / down accuracy and macro-F1 on sign(r_{t+1})
  * portfolio: each day rank the cross-section by rhat_{t+1}, sort into ten deciles, go long
               the top decile and short the bottom decile equal-weighted, zero cost,
               rebalanced daily, no transaction costs

Deviations from the paper, all forced by data/compute and reported with the results:
  * universe is currently-traded tickers from yfinance, not CRSP -> survivorship bias
  * Chronos uses fewer sampled paths than the paper's default (--num-samples)
  * test years are limited by the price history we could download
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/home/ebenezer0616/IPO_test/scripts")
from fintext_tsfm import load_chronos, load_timesfm

DATA = "/home/ebenezer0616/IPO_test/data/us_panel.npz"
ANN = 252


def metrics(pred, actual, dates, stock_idx, quantile=10):
    """pred/actual: 1-D aligned arrays; dates: day index per observation.

    quantile: cross-section split for the long-short (10 = deciles, the paper's choice;
    small universes like TW50 pass 5 for quintiles so each leg still holds ~10 names).
    """
    err = pred - actual
    r2 = 100 * (1 - (err ** 2).sum() / (actual ** 2).sum())
    if pred.std() == 0:
        # a constant forecast (e.g. the zero benchmark) has no sign and no cross-sectional
        # ranking; reporting an accuracy or a decile spread for it would be meaningless
        return {"n_obs": len(pred), "n_days": len(np.unique(dates)), "R2_OOS_%": r2,
                "acc_%": np.nan, "acc_up_%": np.nan, "acc_dn_%": np.nan, "macroF1_%": np.nan,
                "LS_ann_%": np.nan, "LS_sharpe": np.nan, "pred_std": 0.0}

    up, dn = actual > 0, actual < 0
    ph = np.sign(pred) == np.sign(actual)
    acc = 100 * ph.mean()
    acc_up = 100 * ph[up].mean() if up.any() else np.nan
    acc_dn = 100 * ph[dn].mean() if dn.any() else np.nan
    # macro-F1 over {up, down} using sign(pred) as the predicted class
    f1s = []
    for cls, mask in (("up", up), ("dn", dn)):
        pos = (pred > 0) if cls == "up" else (pred < 0)
        tp = (pos & mask).sum()
        prec = tp / pos.sum() if pos.sum() else 0.0
        rec = tp / mask.sum() if mask.sum() else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    macro_f1 = 100 * np.mean(f1s)

    # decile long-short, equal weighted, rebalanced daily.
    # rf cancels in the zero-cost spread, so the LS series is the same whether it is built
    # from excess or raw returns; the individual legs are excess returns (rf does not cancel).
    from scipy import stats as _st
    ls, lng, sht, ics = [], [], [], []
    for d in np.unique(dates):
        m = dates == d
        if m.sum() < 2 * quantile:             # need enough names for both extreme baskets
            continue
        p, a = pred[m], actual[m]
        if p.std() > 1e-12:
            ics.append(_st.spearmanr(p, a).statistic)   # daily cross-sectional rank IC
        k = max(1, len(p) // quantile)
        o = np.argsort(p)
        top, bot = a[o[-k:]].mean(), a[o[:k]].mean()
        ls.append(top - bot); lng.append(top); sht.append(-bot)
    ls, lng, sht = map(np.asarray, (ls, lng, sht))

    def perf(x):
        sd = x.std(ddof=1)
        curve = np.cumprod(1 + x)
        return (100 * x.mean() * ANN,
                (x.mean() / sd) * np.sqrt(ANN) if sd > 0 else np.nan,
                100 * sd * np.sqrt(ANN),
                1e4 * x.mean(),
                100 * (1 - curve / np.maximum.accumulate(curve)).max(),
                100 * x.min())
    ann, shp, sd, bps, mdd, dd1 = perf(ls)
    return {"n_obs": len(pred), "n_days": len(ls), "R2_OOS_%": r2,
            "acc_%": acc, "acc_up_%": acc_up, "acc_dn_%": acc_dn, "macroF1_%": macro_f1,
            "LS_ann_%": ann, "LS_sharpe": shp, "LS_std_%": sd, "LS_bps_day": bps,
            "LS_maxdd_%": mdd, "LS_maxdd1d_%": dd1,
            "LS_skew": float(pd.Series(ls).skew()), "LS_kurt": float(pd.Series(ls).kurt()),
            "IC_mean": float(np.mean(ics)) if ics else np.nan,
            "IC_t": float(np.mean(ics) / (np.std(ics, ddof=1) / np.sqrt(len(ics))))
                    if len(ics) > 1 and np.std(ics, ddof=1) > 0 else np.nan,
            "IC_pos_%": 100 * float(np.mean(np.array(ics) > 0)) if ics else np.nan,
            "long_ann_%": perf(lng)[0], "long_sharpe": perf(lng)[1],
            "short_ann_%": perf(sht)[0], "short_sharpe": perf(sht)[1],
            "pred_std": pred.std()}


def build_windows(ex, valid, day_ids, ctx_len, lag=1):
    """Return (contexts [N, ctx_len], targets [N], day index [N], column index [N]).

    lag=1 targets day t+1 (the paper's protocol). lag=2 skips a day and targets t+2:
    the model's forecast is unchanged, so any drop in portfolio return between lag=1
    and lag=2 measures the one-day microstructure component (bid-ask bounce and
    short-term reversal harvested by daily rebalancing), not model skill.
    """
    ctxs, tgts, dids, cols = [], [], [], []
    for t in day_ids:
        win = ex[t - ctx_len + 1: t + 1]                # context ends on day t
        nxt = ex[t + lag]                               # target day
        ok = np.isfinite(win).all(axis=0) & np.isfinite(nxt) & valid[t]
        if not ok.any():
            continue
        idx = np.flatnonzero(ok)
        ctxs.append(win[:, idx].T.astype(np.float32))
        tgts.append(nxt[idx].astype(np.float64))
        dids.append(np.full(len(idx), t))
        cols.append(idx)
    if not ctxs:
        return None
    return (np.concatenate(ctxs), np.concatenate(tgts),
            np.concatenate(dids), np.concatenate(cols))


def predict(repo, ctx, device, num_samples, tf_batch=1024, ch_batch=64):
    if "TimesFM" in repo:
        m = load_timesfm(repo, device=device)
        out = [m.forecast(list(ctx[i:i + tf_batch]), horizon=1)[0].numpy()[:, 0]
               for i in range(0, len(ctx), tf_batch)]
    else:
        m = load_chronos(repo, device=device)
        out = []
        for i in range(0, len(ctx), ch_batch):
            s = m.predict(torch.tensor(ctx[i:i + ch_batch]), prediction_length=1,
                          num_samples=num_samples)          # [B, S, 1]
            out.append(s.numpy()[:, :, 0].mean(axis=1))     # conditional MEAN, per the paper
    del m
    torch.cuda.empty_cache()
    return np.concatenate(out).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2017, 2023],
                    help="two ints = inclusive range of TEST years")
    ap.add_argument("--windows", type=int, nargs="+", default=[5, 21, 252, 512])
    ap.add_argument("--families", nargs="+", default=["TimesFM_20M"])
    ap.add_argument("--variants", nargs="+", default=["US", "Global", "Augmented"])
    ap.add_argument("--universe-top", type=int, default=0,
                    help="keep only the N most liquid tickers (0 = all)")
    ap.add_argument("--drop-microcap", type=float, default=0.05,
                    help="exclude the bottom q of each day's size distribution, as the paper "
                         "does with market cap; dollar volume is the proxy used here")
    ap.add_argument("--num-samples", type=int, default=20, help="Chronos sampled paths")
    ap.add_argument("--lag", type=int, default=1,
                    help="target day t+lag; 2 = skip a day, isolating microstructure effects")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="/home/ebenezer0616/IPO_test/out/us_replication.csv")
    ap.add_argument("--data", default=DATA,
                    help="panel npz; pass data/us_panel_973broken.npz to reproduce the "
                         "973-ticker mixed-cap runs on the identical universe")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    dates = pd.to_datetime(d["dates"])
    ex = d["exret"].astype(np.float64)
    dv = d["dollar_vol"]
    tickers = d["tickers"]

    if args.universe_top:
        liq = np.nan_to_num(np.nanmedian(dv, axis=0))
        keep = np.argsort(liq)[::-1][:args.universe_top]
        ex, dv, tickers = ex[:, keep], dv[:, keep], tickers[keep]
    # a stock is investable on day t if it traded that day
    valid = np.isfinite(ex)
    if args.drop_microcap > 0:
        # paper: drop the bottom 5% of the country-day market-cap distribution. We have no
        # historical share counts, so dollar volume stands in as the size proxy.
        dvm = np.where(np.isfinite(dv) & (dv > 0), dv, np.nan)
        cut = np.nanquantile(np.where(valid, dvm, np.nan), args.drop_microcap, axis=1)
        before = valid.sum()
        valid &= dvm >= cut[:, None]
        print(f"microcap filter (bottom {args.drop_microcap:.0%} by dollar volume): "
              f"{before:,} -> {valid.sum():,} stock-days")
    print(f"panel: {ex.shape[0]} days x {ex.shape[1]} tickers, "
          f"{dates[0].date()} -> {dates[-1].date()}  device={args.device}")

    y0, y1 = min(args.years), max(args.years)
    rows = []
    for ctx_len in args.windows:
        for year in range(y0, y1 + 1):
            day_ids = np.flatnonzero((dates.year == year))
            day_ids = day_ids[(day_ids >= ctx_len - 1) & (day_ids < len(dates) - args.lag)]
            if len(day_ids) == 0:
                continue
            built = build_windows(ex, valid, day_ids, ctx_len, lag=args.lag)
            if built is None:
                continue
            C, T, D, _ = built
            ck = min(2023, year - 1)
            print(f"\nwindow={ctx_len} year={year} ckpt={ck}: {len(C):,} stock-days "
                  f"over {len(np.unique(D))} days", flush=True)

            rows.append({"window": ctx_len, "year": year, "ckpt": "-", "model": "zero",
                         **metrics(np.zeros_like(T), T, D, None)})
            for fam in args.families:
                for var in args.variants:
                    repo = f"FinText/{fam}_{ck}_{var}"
                    try:
                        P = predict(repo, C, args.device, args.num_samples)
                    except Exception as e:  # noqa: BLE001
                        print(f"  SKIP {repo}: {type(e).__name__}: {str(e)[:70]}")
                        continue
                    rows.append({"window": ctx_len, "year": year, "ckpt": ck,
                                 "model": f"{fam}_{var}", **metrics(P, T, D, None)})
                    r = rows[-1]
                    print(f"  {fam}_{var:10s} R2={r['R2_OOS_%']:8.3f}%  acc={r['acc_%']:5.2f}%  "
                          f"LS={r['LS_ann_%']:7.2f}%/yr  Sharpe={r['LS_sharpe']:6.2f}", flush=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)

    res = pd.DataFrame(rows)
    res.to_csv(args.out, index=False)
    pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 500)
    print("\n=== pooled over test years, by window and model ===")
    g = res.groupby(["window", "model"]).apply(lambda x: pd.Series({
        "years": x.year.nunique(),
        "R2_OOS_%": np.average(x["R2_OOS_%"], weights=x.n_obs),
        "acc_%": np.average(x["acc_%"], weights=x.n_obs),
        "macroF1_%": np.average(x["macroF1_%"], weights=x.n_obs),
        "LS_ann_%": np.average(x["LS_ann_%"], weights=x.n_days),
        "LS_sharpe": np.average(x["LS_sharpe"], weights=x.n_days),
    }), include_groups=False)
    print(g.to_string(float_format=lambda x: f"{x:9.3f}"))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
