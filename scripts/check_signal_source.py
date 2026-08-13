"""Is the Chronos IC just momentum in disguise?

For each IPO cohort, rank-regress the realised 30-day cumulative excess return on the
context's own cumulative return (momentum), and measure the model's IC against the
residual. If the IC survives, the model carries information beyond "the first 30 days
kept going".  Also dumps the per-year long-short tertile return.
"""

import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, "/home/ebenezer0616/IPO_test/scripts")
from run_experiment import forecast_chronos

MODELS = ["Chronos_Small_Augmented", "Chronos_Small_Global", "Chronos_Small_US"]
HOR = 30
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def rank(x):
    return stats.rankdata(x) / len(x)


d = np.load("/home/ebenezer0616/IPO_test/data/tw_ipo_panel.npz", allow_pickle=True)
yr, ck = d["ipo_year"], d["ckpt_year"]
rows, preds = [], {}

for tag in MODELS:
    family, variant = tag.rsplit("_", 1)
    for y in sorted(set(yr.tolist())):
        m = yr == y
        if m.sum() < 20:
            continue
        c, a = d["ctx_excess"][m].astype(np.float64), d["hor_excess"][m].astype(np.float64)
        cy = int(ck[m][0])
        p = forecast_chronos(f"FinText/{family}_{cy}_{variant}", c, DEV)
        preds[(tag, y)] = p

        cum_p = (1 + p).prod(axis=1) - 1
        cum_a = (1 + a).prod(axis=1) - 1
        mom = (1 + c).prod(axis=1) - 1

        ic = stats.spearmanr(cum_p, cum_a).statistic
        ic_mom = stats.spearmanr(mom, cum_a).statistic
        # residualise the realised return on momentum, in rank space
        rm, ra = rank(mom), rank(cum_a)
        beta = np.polyfit(rm, ra, 1)[0]
        resid = ra - beta * rm
        ic_resid = stats.spearmanr(cum_p, resid).statistic
        # long-short tertile, equal weighted
        o = np.argsort(cum_p)
        k = max(1, len(o) // 3)
        ls = cum_a[o[-k:]].mean() - cum_a[o[:k]].mean()

        rows.append({"model": tag, "ipo_year": y, "n": int(m.sum()), "IC": ic,
                     "IC_momentum": ic_mom, "IC_resid_of_momentum": ic_resid,
                     "LS_tertile_%": 100 * ls,
                     "corr_pred_momentum": stats.spearmanr(cum_p, mom).statistic})
    print(f"  {tag} done")

res = pd.DataFrame(rows)
res.to_csv("/home/ebenezer0616/IPO_test/out/signal_source.csv", index=False)
pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 100)

print("\n=== per-year, Chronos_Small_Augmented ===")
print(res[res.model == "Chronos_Small_Augmented"]
      .to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

print("\n=== summary across the 17 cohorts ===")
g = res.groupby("model").apply(lambda x: pd.Series({
    "IC_mean": x.IC.mean(),
    "IC_t": x.IC.mean() / (x.IC.std(ddof=1) / np.sqrt(len(x))),
    "IC_resid_mean": x.IC_resid_of_momentum.mean(),
    "IC_resid_t": x.IC_resid_of_momentum.mean() / (x.IC_resid_of_momentum.std(ddof=1) / np.sqrt(len(x))),
    "pos_yrs_%": 100 * (x.IC > 0).mean(),
    "LS_mean_%": x["LS_tertile_%"].mean(),
    "LS_t": x["LS_tertile_%"].mean() / (x["LS_tertile_%"].std(ddof=1) / np.sqrt(len(x))),
    "corr_w_momentum": x.corr_pred_momentum.mean(),
}), include_groups=False)
print(g.to_string(float_format=lambda x: f"{x:8.3f}"))
print("\nsaved -> /home/ebenezer0616/IPO_test/out/signal_source.csv")
