"""Paper-protocol replication on point-in-time Taiwan 50 constituents.

Same protocol as run_sp500.py: for test year Y the cross-section is the TW50 membership
as of Dec 31, Y-1 (reconstructed from the zh.wikipedia constituents + change-history
tables, snapshot 2026-08-13; exactly 50 members per year-end, all present in the finlab
panel). Excess return = finlab adjusted-close total return minus the CBC overnight
interbank rate. One-day-ahead forecasts, context swept over 5/21/252/512.

With only 50 names a decile basket holds 5 stocks, so the long-short here uses
QUINTILES (10 stocks per leg) - noted in the results.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/home/ebenezer0616/IPO_test/scripts")
from replicate_paper import build_windows, metrics, predict
from tw_riskfree import daily_rf

PANEL = "/home/ebenezer0616/finlab_db/etl#adj_close.feather"
MEMBERS = "/home/ebenezer0616/IPO_test/data/tw50_membership.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2017, 2023])
    ap.add_argument("--windows", type=int, nargs="+", default=[512, 252, 21, 5])
    ap.add_argument("--families", nargs="+", default=["TimesFM_20M", "Chronos_Small"])
    ap.add_argument("--variants", nargs="+", default=["US"])
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--ckpt-dir", default=None,
                    help="local checkpoint root {dir}/{year}/ instead of the HF repos "
                         "(for the fintext.ai Synthetic checkpoints); the path must contain "
                         "'TimesFM' or 'Chronos' so predict() picks the right loader")
    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--quantile", type=int, default=5,
                    help="cross-section split; 5 = quintiles (10 stocks per leg)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="/home/ebenezer0616/IPO_test/out/tw50_results.csv")
    args = ap.parse_args()

    membership = {k: set(v) for k, v in json.load(open(MEMBERS)).items()}
    all_ids = sorted(set().union(*membership.values()))

    px = pd.read_feather(PANEL).set_index("date")
    px.index = pd.to_datetime(px.index)
    px = px[all_ids].sort_index()
    ret = px.pct_change(fill_method=None)
    ret = ret.where(ret.abs() < 0.5)
    rf = daily_rf(px.index)
    ex = ret.sub(rf, axis=0).values.astype(np.float64)
    dates = px.index
    col = {t: j for j, t in enumerate(px.columns)}
    finite = np.isfinite(ex)
    print(f"panel: {ex.shape[0]} days x {ex.shape[1]} TW50-history tickers  "
          f"device={args.device}  quantile={args.quantile}")

    y0, y1 = min(args.years), max(args.years)
    rows = []
    for ctx_len in args.windows:
        for year in range(y0, y1 + 1):
            snap = membership.get(f"{year-1}-12-31")
            if snap is None:
                continue
            cols = np.array(sorted(col[t] for t in snap if t in col))
            valid = np.zeros_like(finite)
            valid[:, cols] = finite[:, cols]
            day_ids = np.flatnonzero(dates.year == year)
            day_ids = day_ids[(day_ids >= ctx_len - 1) & (day_ids < len(dates) - args.lag)]
            built = build_windows(ex, valid, day_ids, ctx_len, lag=args.lag)
            if built is None:
                continue
            C, T, D, _ = built
            ck = min(2023, year - 1)
            print(f"\nwindow={ctx_len} year={year} ckpt={ck}: members={len(cols)} "
                  f"-> {len(C):,} stock-days", flush=True)
            rows.append({"window": ctx_len, "year": year, "ckpt": "-", "model": "zero",
                         **metrics(np.zeros_like(T), T, D, None, quantile=args.quantile)})
            for fam in args.families:
                for var in args.variants:
                    repo = (f"{args.ckpt_dir}/{ck}" if args.ckpt_dir
                            else f"FinText/{fam}_{ck}_{var}")
                    try:
                        P = predict(repo, C.astype(np.float32), args.device, args.num_samples)
                    except Exception as e:  # noqa: BLE001
                        print(f"  SKIP {repo}: {type(e).__name__}: {str(e)[:70]}")
                        continue
                    rows.append({"window": ctx_len, "year": year, "ckpt": ck,
                                 "model": f"{fam}_{var}", **metrics(P, T, D, None, quantile=args.quantile)})
                    r = rows[-1]
                    print(f"  {fam}_{var:8s} R2={r['R2_OOS_%']:8.3f}%  acc={r['acc_%']:5.2f}%  "
                          f"LS={r['LS_ann_%']:8.2f}%/yr  Sharpe={r['LS_sharpe']:6.2f}", flush=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)

    res = pd.DataFrame(rows)
    res.to_csv(args.out, index=False)
    pd.set_option("display.width", 250, "display.max_columns", 60)
    print("\n=== pooled over test years ===")
    g = res[res.model != "zero"].groupby(["window", "model"]).apply(
        lambda x: pd.Series({
            "years": x.year.nunique(),
            "R2_OOS_%": np.average(x["R2_OOS_%"], weights=x.n_obs),
            "acc_%": np.average(x["acc_%"], weights=x.n_obs),
            "LS_ann_%": np.average(x["LS_ann_%"], weights=x.n_days),
            "LS_sharpe": np.average(x["LS_sharpe"], weights=x.n_days),
            "long_ann_%": np.average(x["long_ann_%"], weights=x.n_days),
            "short_ann_%": np.average(x["short_ann_%"], weights=x.n_days),
        }), include_groups=False)
    print(g.to_string(float_format=lambda x: f"{x:9.3f}"))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
