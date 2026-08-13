"""Taiwan daily risk-free rate, from the CBC's own open data.

Primary series: 金融業隔夜拆款利率 (overnight interbank call loan rate) -- the actual
traded short rate, daily since 2002. This is the right risk-free proxy for daily excess
returns. Note it sits far below the 重貼現率 policy rate throughout (0.08-0.83% vs
1.125-2.0% since 2009) because Taiwan's interbank market runs in structural excess
liquidity, so the two are not interchangeable.

  data/overnight.csv  CBC WebF2.csv, Big5 -> UTF-8, covers 2002-05-02 .. 2026-03-18
  data/gap_*.html     CBC's paginated HTML, used to extend past the CSV's last refresh

Refresh with:
  curl -sL -o data/overnight_big5.csv https://www.cbc.gov.tw/public/data/OpenData/WebF2.csv
  iconv -f big5 -t utf-8 data/overnight_big5.csv > data/overnight.csv
  for p in 1 2 3 4 5; do curl -sL "https://www.cbc.gov.tw/tw/lp-641-1-$p-20.html" -o data/gap_$p.html; done
"""

import glob
import re

import numpy as np
import pandas as pd

DATA = "/home/ebenezer0616/IPO_test/data"
TRADING_DAYS = 252
ROW_RE = re.compile(r"<span>(\d{4}/\d{2}/\d{2})</span>.*?<span>([\d.]+)</span>", re.DOTALL)


def _overnight_annual():
    """Daily overnight interbank rate as a decimal annual rate, indexed by date."""
    df = pd.read_csv(f"{DATA}/overnight.csv")
    s = pd.Series(df.iloc[:, 1].astype(float).values,
                  index=pd.to_datetime(df.iloc[:, 0], format="%Y/%m/%d"))
    rows = {}
    for f in sorted(glob.glob(f"{DATA}/gap_*.html")):
        for d, r in ROW_RE.findall(open(f, encoding="utf-8").read()):
            rows[pd.Timestamp(d.replace("/", "-"))] = float(r)
    if rows:
        s = pd.concat([s, pd.Series(rows)])
    return (s[~s.index.duplicated(keep="last")].sort_index() / 100).rename("rf_annual")


def daily_rf(index, source="overnight"):
    """Daily risk-free rate aligned to `index` (a DatetimeIndex)."""
    if source != "overnight":
        raise ValueError(source)
    ann = _overnight_annual()
    ann = ann.reindex(ann.index.union(index)).ffill().bfill().reindex(index)
    return (1 + ann) ** (1 / TRADING_DAYS) - 1


if __name__ == "__main__":
    ann = _overnight_annual()
    print(f"overnight interbank rate: {len(ann)} obs, "
          f"{ann.index[0].date()} -> {ann.index[-1].date()}")
    idx = pd.DatetimeIndex(sorted(ann.index[ann.index >= "2007-01-01"]))
    rf = daily_rf(idx)
    print(f"daily rf: mean {rf.mean():.8f} ({rf.mean()*TRADING_DAYS:.4%}/yr), "
          f"min {rf.min():.8f}, max {rf.max():.8f}")
    print("\nannualised by year:")
    print((ann[ann.index >= "2007-01-01"].groupby(lambda t: t.year).mean())
          .to_string(float_format=lambda x: f"{x:.3%}"))
