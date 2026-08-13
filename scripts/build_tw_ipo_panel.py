"""Build the Taiwan IPO evaluation panel (setup A: first 30 days -> next 30 days).

Source: ~/finlab_db/etl#adj_close.feather  (wide: date x stock_id, adjusted close)
The series is an *adjusted* close, so cash dividends are already reinvested in it and
    one_day_return_t = (P_t - P_{t-1} + dividends paid in (t-1, t]) / P_{t-1}
is simply its percentage change. Three return definitions are stored:

  ret        total return as above
  ex_rf      ret - daily risk-free rate            <- FinText's "excess return"
  ex_mkt     ret - cross-sectional median return   <- market-adjusted
  ret_pt     (P_t - P_{t-1} + div) / P_t           <- same but divided by the *current*
                                                      price; kept to show it barely differs

Output: data/tw_ipo_panel.npz  +  data/tw_ipo_meta.csv
"""

import json
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ebenezer0616/IPO_test/scripts")
from tw_riskfree import daily_rf

PANEL = "/home/ebenezer0616/finlab_db/etl#adj_close.feather"
OUT = "/home/ebenezer0616/IPO_test/data"

CTX = 30       # context: 30 daily returns after listing
HOR = 30       # horizon: next 30 daily returns
PANEL_START_BUFFER = "2008-07-01"   # panel starts 2007-04-23; ignore left-censored names
MIN_PRICE = 1.0

TWSE_API = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
TPEX_API = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O"


def _fetch_listing_dates():
    """{stock_id: listing_date} from TWSE + TPEx open data. Best effort."""
    out = {}
    for url in (TWSE_API, TPEX_API):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                rows = json.load(r)
            for row in rows:
                sid = (row.get("公司代號") or "").strip()
                d = (row.get("上市日期") or row.get("上櫃日期") or "").strip()
                if sid and len(d) == 8 and d.isdigit():
                    out[sid] = pd.Timestamp(d)
            print(f"  listing dates from {url.split('/')[2]}: {len(rows)} rows")
        except Exception as e:  # noqa: BLE001 - metadata is optional
            print(f"  WARN: {url.split('/')[2]} failed ({type(e).__name__}: {e})")
    return out


def _is_common_stock(sid):
    """TW common stock: 4 numeric digits, not an ETF (00xx) / TDR (91xx) / warrant (letters)."""
    return sid.isdigit() and len(sid) == 4 and not sid.startswith("00") and not sid.startswith("91")


def main():
    os.makedirs(OUT, exist_ok=True)
    px = pd.read_feather(PANEL).set_index("date")
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()
    print(f"panel: {px.shape[0]} days x {px.shape[1]} ids, {px.index[0].date()} -> {px.index[-1].date()}")

    common = [c for c in px.columns if _is_common_stock(c)]
    px_c = px[common]
    print(f"common stocks: {len(common)}")

    ret_all = px_c.pct_change(fill_method=None)   # never pad: suspended days must stay NaN
    ret_all = ret_all.where(ret_all.abs() < 0.5)  # drop obvious data errors
    mkt = ret_all.median(axis=1)                  # median is robust to IPO-day spikes
    rf = daily_rf(px_c.index)
    print(f"risk-free: {rf.mean()*252:.3%}/yr average, "
          f"{rf.min()*252:.3%} - {rf.max()*252:.3%} over the sample")

    listing = _fetch_listing_dates()

    rows = []
    acc = {k: [] for k in ("ctx_ret", "hor_ret", "ctx_exrf", "hor_exrf",
                           "ctx_exmkt", "hor_exmkt", "ctx_retpt", "hor_retpt")}
    buffer_ts = pd.Timestamp(PANEL_START_BUFFER)

    for sid in common:
        s = px_c[sid].dropna()
        if len(s) < CTX + HOR + 1:
            continue
        first = s.index[0]
        if first < buffer_ts or s.iloc[0] < MIN_PRICE:
            continue
        w = s.iloc[: CTX + HOR + 1]
        if (w.index[-1] - w.index[0]).days > 150:          # illiquid / long trading gaps
            continue

        ret = w.pct_change().dropna()                       # (P_t - P_{t-1}) / P_{t-1}
        retpt = (w.diff() / w).dropna()                     # ... / P_t  (as literally written)
        if len(ret) != CTX + HOR or not np.isfinite(ret.values).all():
            continue
        exrf = ret - rf.reindex(ret.index)
        exmkt = ret - mkt.reindex(ret.index)
        if exrf.isna().any() or exmkt.isna().any():
            continue

        rows.append({
            "stock_id": sid,
            "first_trade": first.date(),
            "listing_date": listing.get(sid, pd.NaT),
            "ipo_year": first.year,
            "ckpt_year": min(2023, first.year - 1),        # point-in-time: train cut-off < IPO
            "first_close": float(w.iloc[0]),
            "ctx_cum_exrf": float((1 + exrf.values[:CTX]).prod() - 1),
            "hor_cum_exrf": float((1 + exrf.values[CTX:]).prod() - 1),
        })
        for key, v in (("ret", ret), ("exrf", exrf), ("exmkt", exmkt), ("retpt", retpt)):
            acc[f"ctx_{key}"].append(v.values[:CTX])
            acc[f"hor_{key}"].append(v.values[CTX:])

    meta = pd.DataFrame(rows)
    meta = meta[meta.ckpt_year >= 2000].reset_index(drop=True)
    arrs = {k: np.asarray(v, dtype=np.float32) for k, v in acc.items()}

    np.savez(os.path.join(OUT, "tw_ipo_panel.npz"),
             stock_id=meta.stock_id.values, ipo_year=meta.ipo_year.values,
             ckpt_year=meta.ckpt_year.values, **arrs)
    meta.to_csv(os.path.join(OUT, "tw_ipo_meta.csv"), index=False)

    print(f"\nIPO sample: {len(meta)} names, {meta.ipo_year.min()}-{meta.ipo_year.max()}")
    matched = meta.listing_date.notna().sum()
    print(f"listing date matched in TWSE/TPEx open data: {matched}/{len(meta)}")
    if matched:
        dd = (pd.to_datetime(meta.listing_date) - pd.to_datetime(meta.first_trade)).dt.days.dropna()
        print(f"  |listing_date - first_trade|: median {dd.abs().median():.0f}d, "
              f"<=7d for {(dd.abs() <= 7).mean():.0%}")

    print("\nhow much the return definition actually matters (daily, pooled):")
    for k in ("ret", "exrf", "exmkt", "retpt"):
        v = np.concatenate([arrs[f"ctx_{k}"].ravel(), arrs[f"hor_{k}"].ravel()])
        print(f"  {k:6s} mean {v.mean():+.6f}  std {v.std():.6f}")
    base = np.concatenate([arrs["ctx_ret"].ravel(), arrs["hor_ret"].ravel()])
    for k in ("exrf", "exmkt", "retpt"):
        v = np.concatenate([arrs[f"ctx_{k}"].ravel(), arrs[f"hor_{k}"].ravel()])
        print(f"  ret vs {k:6s}: mean abs diff {np.abs(base - v).mean():.6f}, "
              f"corr {np.corrcoef(base, v)[0,1]:.5f}")


if __name__ == "__main__":
    main()
