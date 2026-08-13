"""Setup A, year by year: feed the first 30 post-IPO daily excess returns, forecast the next 30.

Every IPO cohort is evaluated with its own point-in-time checkpoint: IPOs listed in year Y
use the min(2023, Y-1) checkpoint, so the model's pre-training data ends before the IPO.

Reported per (IPO year, model):
  R2_oos_%   out-of-sample R^2 on daily excess returns vs a zero forecast (>0 = beats "no signal")
  dir_hit_%  daily sign accuracy
  cum_sign_% sign accuracy of the 30-day cumulative excess return
  IC         Spearman corr. of predicted vs realised 30-day cumulative excess return
  T3-T1_%    realised cum. excess return of the top predicted tertile minus the bottom
  blowup_%   share of predicted daily returns with |r| > 50% (numerical divergence)

  python run_experiment.py                       # all years, all variants
  python run_experiment.py --years 2020 2026     # year range
"""

import argparse
import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, "/home/ebenezer0616/IPO_test/scripts")
from fintext_tsfm import load_chronos, load_timesfm

DATA = "/home/ebenezer0616/IPO_test/data/tw_ipo_panel.npz"
HOR = 30
CLIP = 1.0          # daily excess return clip used only so cumulative products stay finite


def forecast_timesfm(repo, ctx, device, batch=256):
    m = load_timesfm(repo, device=device)
    out = []
    for i in range(0, len(ctx), batch):
        mean, _ = m.forecast(list(ctx[i:i + batch]), horizon=HOR)
        out.append(mean.numpy())          # mean head only: the quantile heads are unusable
    del m
    torch.cuda.empty_cache()
    return np.concatenate(out).astype(np.float64)


def forecast_chronos(repo, ctx, device, batch=64, num_samples=100):
    p = load_chronos(repo, device=device)
    out = []
    for i in range(0, len(ctx), batch):
        s = p.predict(torch.tensor(ctx[i:i + batch], dtype=torch.float32),
                      prediction_length=HOR, num_samples=num_samples)  # [B, S, H]
        out.append(np.median(s.numpy(), axis=1))
    del p
    torch.cuda.empty_cache()
    return np.concatenate(out).astype(np.float64)


def evaluate(pred, actual):
    """pred/actual: [N, HOR] daily excess returns."""
    blow = float((np.abs(pred) > 0.5).mean())
    p = np.clip(np.nan_to_num(pred, nan=0.0, posinf=CLIP, neginf=-CLIP), -CLIP, CLIP)
    mse = ((p - actual) ** 2).mean()
    mse0 = (actual ** 2).mean()
    cum_p = (1 + p).prod(axis=1) - 1
    cum_a = (1 + actual).prod(axis=1) - 1
    nz = p != 0
    ic = np.nan
    if np.std(cum_p) > 1e-12:
        ic = stats.spearmanr(cum_p, cum_a).statistic
    order = np.argsort(cum_p)
    k = max(1, len(order) // 3)
    spread = cum_a[order[-k:]].mean() - cum_a[order[:k]].mean() if np.std(cum_p) > 1e-12 else np.nan
    return {
        "n": len(pred),
        "R2_oos_%": 100 * (1 - mse / mse0),
        "dir_hit_%": 100 * (np.sign(p[nz]) == np.sign(actual[nz])).mean() if nz.any() else np.nan,
        "cum_sign_%": 100 * (np.sign(cum_p) == np.sign(cum_a)).mean() if np.std(cum_p) > 1e-12 else np.nan,
        "IC": ic,
        "T3-T1_%": 100 * spread if spread == spread else np.nan,
        "blowup_%": 100 * blow,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=None,
                    help="two ints = inclusive range, or an explicit list")
    ap.add_argument("--variants", nargs="+", default=["Global", "US", "Augmented"])
    ap.add_argument("--families", nargs="+", default=["TimesFM_20M", "Chronos_Small"])
    ap.add_argument("--min-n", type=int, default=10, help="skip cohorts smaller than this")
    ap.add_argument("--returns", default="exrf", choices=["exrf", "exmkt", "ret", "retpt"],
                    help="exrf = total return minus daily risk-free (FinText's definition); "
                         "exmkt = minus cross-sectional median; ret = raw total return")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="/home/ebenezer0616/IPO_test/out/results_by_year.csv")
    args = ap.parse_args()

    d = np.load(DATA, allow_pickle=True)
    yr, ck = d["ipo_year"], d["ckpt_year"]
    if args.years:
        sel = ((yr >= min(args.years)) & (yr <= max(args.years))) if len(args.years) == 2 \
            else np.isin(yr, args.years)
    else:
        sel = np.ones(len(yr), bool)
    ctx = d[f"ctx_{args.returns}"][sel]
    act = d[f"hor_{args.returns}"][sel]
    yr, ck = yr[sel], ck[sel]
    cohorts = [y for y in sorted(set(yr.tolist())) if (yr == y).sum() >= args.min_n]
    print(f"device={args.device}  IPOs={len(ctx)}  returns={args.returns}  cohorts={cohorts}\n")

    specs = [(f, v) for f in args.families for v in args.variants]
    rows = []

    for y in cohorts:
        m = yr == y
        c, a = ctx[m].astype(np.float64), act[m].astype(np.float64)
        cy = int(ck[m][0])                                   # min(2023, y-1)
        rows.append({"ipo_year": y, "ckpt_year": "-", "model": "baseline:zero",
                     **evaluate(np.zeros_like(a), a)})
        mom = np.repeat(c.mean(axis=1, keepdims=True), HOR, axis=1)
        rows.append({"ipo_year": y, "ckpt_year": "-", "model": "baseline:momentum",
                     **evaluate(mom, a)})
        rows.append({"ipo_year": y, "ckpt_year": "-", "model": "baseline:reversal",
                     **evaluate(-mom, a)})
        for family, variant in specs:
            repo = f"FinText/{family}_{cy}_{variant}"
            try:
                pred = (forecast_timesfm(repo, c, args.device) if family.startswith("TimesFM")
                        else forecast_chronos(repo, c, args.device))
            except Exception as e:  # noqa: BLE001 - a missing checkpoint must not kill the sweep
                print(f"  SKIP {repo}: {type(e).__name__}: {str(e)[:80]}")
                continue
            rows.append({"ipo_year": y, "ckpt_year": cy, "model": f"{family}_{variant}",
                         **evaluate(pred, a)})
        print(f"  {y}: n={m.sum():3d} ckpt={cy} done")

    res = pd.DataFrame(rows)
    res.to_csv(args.out, index=False)
    pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 500)

    print("\n=== IC by IPO year (Spearman, predicted vs realised 30d cum excess return) ===")
    print(res.pivot(index="ipo_year", columns="model", values="IC")
             .to_string(float_format=lambda x: f"{x:6.3f}"))
    print("\n=== R2_oos_% by IPO year (daily excess returns; >0 beats a zero forecast) ===")
    print(res.pivot(index="ipo_year", columns="model", values="R2_oos_%")
             .to_string(float_format=lambda x: f"{x:8.2f}"))

    print("\n=== pooled across years (mean of yearly stats, n-weighted where sensible) ===")
    g = res.groupby("model").apply(
        lambda x: pd.Series({
            "years": x.ipo_year.nunique(),
            "n_total": x.n.sum(),
            "IC_mean": x.IC.mean(),
            "IC_t": x.IC.mean() / (x.IC.std(ddof=1) / np.sqrt(x.IC.notna().sum()))
                    if x.IC.notna().sum() > 1 else np.nan,
            "IC_pos_yrs_%": 100 * (x.IC > 0).mean(),
            "R2_oos_mean_%": np.average(x["R2_oos_%"], weights=x.n),
            "dir_hit_mean_%": np.average(x["dir_hit_%"].fillna(0), weights=x.n),
            "T3-T1_mean_%": x["T3-T1_%"].mean(),
            "blowup_%": np.average(x["blowup_%"], weights=x.n),
        }), include_groups=False)
    print(g.to_string(float_format=lambda x: f"{x:8.3f}"))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
