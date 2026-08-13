"""IPO experiment, one setting only: context = first 30 post-listing daily excess returns,
predict days 31-60. Strict same-year cohorts, point-in-time checkpoints (cohort Y uses
min(2023, Y-1)). One script for both markets so every number is computed by the same code.

  python run_ipo.py --market tw            # data/tw_ipo_panel.npz (exrf arrays)
  python run_ipo.py --market us            # built on the fly from data/us_panel.npz

Cross-sections are 20-300 names, so the long-short uses TERTILES (top/bottom third by the
predicted 30-day cumulative excess return). Each cohort produces one 30-trading-day
long-short return; "annualised" scales it by 252/30 and the Sharpe uses the same period
count - i.e. the annualisation assumes the trade could be redeployed back-to-back, which
is generous and stated here rather than hidden.

TimesFM caveat, always attached to its rows: a 30-day context is a single 32-day patch,
so the transformer degenerates to an MLP; its numbers measure that fallback, not the model.
"""

import argparse
import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, "/home/ebenezer0616/IPO_test/scripts")
from fintext_tsfm import load_chronos, load_timesfm

CTX, HOR = 30, 30
PERIODS_PER_YEAR = 252 / HOR


def load_tw():
    d = np.load("/home/ebenezer0616/IPO_test/data/tw_ipo_panel.npz", allow_pickle=True)
    return (d["ctx_exrf"].astype(np.float64), d["hor_exrf"].astype(np.float64),
            d["ipo_year"].astype(int))


def load_us():
    d = np.load("/home/ebenezer0616/IPO_test/data/us_panel.npz", allow_pickle=True)
    dates = pd.to_datetime(d["dates"])
    ex = d["exret"].astype(np.float64)
    ctxs, hors, years = [], [], []
    buffer_end = dates[0] + pd.Timedelta(days=180)     # left-censored: already listed at start
    for j in range(ex.shape[1]):
        fin = np.flatnonzero(np.isfinite(ex[:, j]))
        if len(fin) < CTX + HOR:
            continue
        f = fin[0]
        if dates[f] <= buffer_end:
            continue
        w = ex[f:f + CTX + HOR, j]
        if not np.isfinite(w).all():
            continue
        if (dates[f + CTX + HOR - 1] - dates[f]).days > 100:   # long gaps -> not a clean IPO
            continue
        ctxs.append(w[:CTX]); hors.append(w[CTX:]); years.append(dates[f].year)
    return np.asarray(ctxs), np.asarray(hors), np.asarray(years)


def forecast(repo, ctx, device, num_samples):
    if "TimesFM" in repo:
        m = load_timesfm(repo, device=device)
        out = [m.forecast(list(ctx[i:i + 256]), horizon=HOR)[0].numpy()
               for i in range(0, len(ctx), 256)]
        del m
    else:
        p = load_chronos(repo, device=device)
        out = []
        for i in range(0, len(ctx), 64):
            s = p.predict(torch.tensor(ctx[i:i + 64], dtype=torch.float32),
                          prediction_length=HOR, num_samples=num_samples)
            out.append(s.numpy().mean(axis=1))          # conditional mean, per the paper
        del p
    torch.cuda.empty_cache()
    return np.concatenate(out).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["tw", "us"], required=True)
    ap.add_argument("--years", type=int, nargs="+", default=[2017, 2023])
    ap.add_argument("--families", nargs="+", default=["TimesFM_20M", "Chronos_Small"])
    ap.add_argument("--variants", nargs="+", default=["US"])
    ap.add_argument("--ckpt-dir", default=None,
                    help="local checkpoint root {dir}/{year}/ instead of the HF FinText repos "
                         "(used for the fintext.ai Synthetic checkpoints)")
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--min-n", type=int, default=15)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_path = args.out or f"/home/ebenezer0616/IPO_test/out/ipo_{args.market}_results.csv"

    ctx, hor, yr = load_tw() if args.market == "tw" else load_us()
    y0, y1 = min(args.years), max(args.years)
    sel = (yr >= y0) & (yr <= y1)
    ctx, hor, yr = ctx[sel], hor[sel], yr[sel]
    cohorts = [y for y in range(y0, y1 + 1) if (yr == y).sum() >= args.min_n]
    print(f"market={args.market}  IPOs={len(ctx)}  cohorts={ {y: int((yr==y).sum()) for y in cohorts} }")

    rows = []
    for fam in args.families:
        for var in args.variants:
            per = []
            for y in cohorts:
                m = yr == y
                ck = min(2023, y - 1)
                repo = (f"{args.ckpt_dir}/{ck}" if args.ckpt_dir
                        else f"FinText/{fam}_{ck}_{var}")
                try:
                    P = forecast(repo, ctx[m].astype(np.float32), args.device, args.num_samples)
                except Exception as e:  # noqa: BLE001
                    print(f"  SKIP {repo}: {type(e).__name__}: {str(e)[:70]}")
                    continue
                A = hor[m]
                P = np.clip(np.nan_to_num(P), -1, 1)
                r2 = 100 * (1 - ((P - A) ** 2).sum() / (A ** 2).sum())
                acc = 100 * (np.sign(P) == np.sign(A)).mean()
                cum_p = (1 + P).prod(axis=1) - 1
                cum_a = (1 + A).prod(axis=1) - 1
                ic = stats.spearmanr(cum_p, cum_a).statistic if cum_p.std() > 1e-12 else np.nan
                k = max(1, m.sum() // 3)
                o = np.argsort(cum_p)
                ls = cum_a[o[-k:]].mean() - cum_a[o[:k]].mean()
                lng, sht = cum_a[o[-k:]].mean(), -cum_a[o[:k]].mean()
                per.append({"cohort": y, "ckpt": ck, "n": int(m.sum()), "R2_OOS_%": r2,
                            "acc_%": acc, "IC": ic, "LS_30d_%": 100 * ls,
                            "long_30d_%": 100 * lng, "short_30d_%": 100 * sht})
                print(f"  {fam}_{var} {y}: n={m.sum():3d} ckpt={ck} R2={r2:8.2f}% "
                      f"acc={acc:5.2f}% IC={ic:+.3f} LS30d={100*ls:+6.2f}%", flush=True)
            if not per:
                continue
            p = pd.DataFrame(per)
            ls = p["LS_30d_%"] / 100
            ann = 100 * ls.mean() * PERIODS_PER_YEAR
            shp = (ls.mean() / ls.std(ddof=1)) * np.sqrt(PERIODS_PER_YEAR) \
                if ls.std(ddof=1) > 0 else np.nan
            for r in per:
                rows.append({"model": f"{fam}_{var}", **r})
            rows.append({"model": f"{fam}_{var}", "cohort": "ALL",
                         "ckpt": "-", "n": int(p.n.sum()),
                         "R2_OOS_%": np.average(p["R2_OOS_%"], weights=p.n),
                         "acc_%": np.average(p["acc_%"], weights=p.n),
                         "IC": p.IC.mean(), "LS_30d_%": p["LS_30d_%"].mean(),
                         "long_30d_%": p["long_30d_%"].mean(),
                         "short_30d_%": p["short_30d_%"].mean(),
                         "LS_ann_%": ann, "LS_sharpe": shp})

    res = pd.DataFrame(rows)
    res.to_csv(out_path, index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print("\n=== summary (cohort = ALL rows) ===")
    print(res[res.cohort == "ALL"].to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
