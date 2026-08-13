"""Build the U.S. daily excess-return panel for replicating arXiv:2511.18578.

Paper protocol:
  r_t     = (S_t + D_t - S_{t-1}) / S_{t-1}      total return, dividends included
  r_ex_t  = r_t - rf_t                            rf = daily risk-free rate

Sources here (the paper uses CRSP, which we do not have):
  prices  yfinance auto-adjusted close -> pct_change is the total return above
  rf      Kenneth French's daily research factors (the same 1-month T-bill series
          CRSP-based papers use), so the risk-free leg matches the paper exactly

KNOWN DEVIATION -- survivorship bias: the ticker list is *currently traded* names, so
firms that delisted during the sample are missing. CRSP includes them with delisting
returns. This biases the long-short backtest upward and cannot be fixed with free data;
it is reported alongside the results rather than silently ignored.

Output: data/us_panel.npz (dates, tickers, ret, exret) + data/us_universe.csv
"""

import argparse
import io
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import yfinance as yf

OUT = "/home/ebenezer0616/IPO_test/data"
NASDAQ_SYMS = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
FF_DAILY = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
            "F-F_Research_Data_Factors_daily_CSV.zip")


def fetch_risk_free():
    """Daily risk-free rate (decimal) from Ken French's research factors."""
    with urllib.request.urlopen(FF_DAILY, timeout=120) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    raw = z.read(z.namelist()[0]).decode("latin-1")
    rows = [l for l in raw.splitlines() if l[:8].strip().isdigit() and len(l[:8].strip()) == 8]
    df = pd.DataFrame([l.split(",") for l in rows]).iloc[:, [0, 4]]
    df.columns = ["date", "RF"]
    rf = pd.Series(df.RF.astype(float).values / 100.0,
                   index=pd.to_datetime(df.date.str.strip(), format="%Y%m%d"))
    return rf.sort_index()


def fetch_universe():
    """Currently-traded U.S. common stocks (no ETFs, no test issues, no warrants/units)."""
    req = urllib.request.Request(NASDAQ_SYMS, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        txt = r.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(txt), sep="|")
    df = df[df["Nasdaq Traded"] == "Y"]
    df = df[(df["ETF"] == "N") & (df["Test Issue"] == "N")]
    df = df[df["Security Name"].str.contains("Common Stock", na=False)]
    # drop warrants / units / preferred / rights, and anything with a share-class suffix char
    bad = df["Security Name"].str.contains("Warrant|Unit|Preferred|Right|Depositary", case=False,
                                           na=False)
    df = df[~bad]
    sym = df["NASDAQ Symbol"].astype(str)
    sym = sym[~sym.str.contains(r"[\$\.\^]", regex=True)]
    return sorted(set(sym))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01", help="price history start (needs 512 "
                    "trading days of burn-in before the first test year)")
    ap.add_argument("--end", default="2024-01-01")
    ap.add_argument("--max-tickers", type=int, default=2000,
                    help="cap the download; the panel is later filtered by liquidity anyway")
    ap.add_argument("--chunk", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    rf = fetch_risk_free()
    print(f"risk-free: {len(rf)} days, {rf.index[0].date()} -> {rf.index[-1].date()}, "
          f"mean {rf.mean()*252:.2%}/yr")

    syms = fetch_universe()
    print(f"universe: {len(syms)} currently-traded U.S. common stocks (capped at {args.max_tickers})")
    syms = syms[:args.max_tickers]

    closes, volumes = [], []
    for i in range(0, len(syms), args.chunk):
        part = syms[i:i + args.chunk]
        d = yf.download(part, start=args.start, end=args.end, auto_adjust=True,
                        progress=False, threads=True, group_by="column")
        if d is None or len(d) == 0:
            print(f"  chunk {i//args.chunk}: empty"); continue
        closes.append(d["Close"])
        volumes.append(d["Volume"] * d["Close"])
        print(f"  chunk {i//args.chunk+1}/{-(-len(syms)//args.chunk)}: "
              f"{d['Close'].shape[1]} tickers, {d['Close'].notna().any().sum()} with data", flush=True)

    px = pd.concat(closes, axis=1).sort_index()
    dv = pd.concat(volumes, axis=1).sort_index()
    px = px.loc[:, ~px.columns.duplicated()]
    dv = dv.loc[:, ~dv.columns.duplicated()]
    px.index = pd.to_datetime(px.index).tz_localize(None)
    dv.index = px.index

    ret = px.pct_change(fill_method=None)
    ret = ret.where(ret.abs() < 10.0)                      # paper: returns beyond +-1000% are errors
    exret = ret.sub(rf.reindex(ret.index), axis=0)

    keep = exret.notna().sum() > 252                       # need at least a year of data
    px, dv, ret, exret = px.loc[:, keep], dv.loc[:, keep], ret.loc[:, keep], exret.loc[:, keep]
    print(f"\npanel: {exret.shape[0]} days x {exret.shape[1]} tickers, "
          f"{exret.index[0].date()} -> {exret.index[-1].date()}")

    np.savez_compressed(
        os.path.join(OUT, "us_panel.npz"),
        dates=exret.index.values.astype("datetime64[D]"),
        tickers=np.array(exret.columns, dtype=object),
        ret=ret.values.astype(np.float32),
        exret=exret.values.astype(np.float32),
        dollar_vol=dv.values.astype(np.float32),
    )
    pd.DataFrame({"ticker": exret.columns,
                  "n_obs": exret.notna().sum().values,
                  "median_dollar_vol": dv.median().values}).to_csv(
        os.path.join(OUT, "us_universe.csv"), index=False)
    print(f"saved -> {OUT}/us_panel.npz")


if __name__ == "__main__":
    main()
