"""Paper-protocol replication restricted to point-in-time S&P 500 members.

Same protocol as replicate_paper.py (1-step horizon, decile long-short, daily rebalance),
but for test year Y the cross-section is the S&P 500 membership as of Dec 31, Y-1
(reconstructed from Wikipedia's constituents + historical-changes tables), so the universe
never contains a stock the index had not yet admitted.

This is THE clean large-cap test: if the checkpoints' long-short returns on the 973-ticker
panel came from illiquid names and microstructure, they should vanish here.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/home/ebenezer0616/IPO_test/scripts")
from replicate_paper import build_windows, metrics, predict

DATA = "/home/ebenezer0616/IPO_test/data/sp500_panel.npz"
MEMBERS = "/home/ebenezer0616/IPO_test/data/sp500_membership.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2017, 2023])
    ap.add_argument("--windows", type=int, nargs="+", default=[512, 21, 5])
    ap.add_argument("--families", nargs="+", default=["TimesFM_20M", "Chronos_Small"])
    ap.add_argument("--variants", nargs="+", default=["US"])
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--ckpt-dir", default=None,
                    help="local checkpoint root {dir}/{year}/ (fintext.ai Synthetic); the "
                         "path must contain 'TimesFM' or 'Chronos' for loader dispatch")
    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="/home/ebenezer0616/IPO_test/out/sp500_results.csv")
    args = ap.parse_args()

    d = np.load(DATA, allow_pickle=True)
    dates = pd.to_datetime(d["dates"])
    ex = d["exret"].astype(np.float64)
    tickers = np.array([str(t) for t in d["tickers"]])
    col = {t: j for j, t in enumerate(tickers)}
    membership = {k: set(v) for k, v in json.load(open(MEMBERS)).items()}
    finite = np.isfinite(ex)
    print(f"panel: {ex.shape[0]} days x {ex.shape[1]} tickers  device={args.device}")

    y0, y1 = min(args.years), max(args.years)
    rows = []
    for ctx_len in args.windows:
        for year in range(y0, y1 + 1):
            snap = membership.get(f"{year-1}-12-31")
            if snap is None:
                print(f"  no membership snapshot for {year-1}-12-31, skip {year}")
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
            print(f"\nwindow={ctx_len} year={year} ckpt={ck}: members={len(snap)} "
                  f"in-panel={len(cols)} -> {len(C):,} stock-days", flush=True)
            rows.append({"window": ctx_len, "year": year, "ckpt": "-", "model": "zero",
                         "n_members": len(cols), **metrics(np.zeros_like(T), T, D, None)})
            for fam in args.families:
                for var in args.variants:
                    repo = (f"{args.ckpt_dir}/{ck}" if args.ckpt_dir
                            else f"FinText/{fam}_{ck}_{var}")
                    try:
                        P = predict(repo, C, args.device, args.num_samples)
                    except Exception as e:  # noqa: BLE001
                        print(f"  SKIP {repo}: {type(e).__name__}: {str(e)[:70]}")
                        continue
                    rows.append({"window": ctx_len, "year": year, "ckpt": ck,
                                 "model": f"{fam}_{var}", "n_members": len(cols),
                                 **metrics(P, T, D, None)})
                    r = rows[-1]
                    print(f"  {fam}_{var:8s} R2={r['R2_OOS_%']:8.3f}%  acc={r['acc_%']:5.2f}%  "
                          f"LS={r['LS_ann_%']:8.2f}%/yr  Sharpe={r['LS_sharpe']:6.2f}  "
                          f"long={r.get('long_ann_%', float('nan')):7.2f}%", flush=True)
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
