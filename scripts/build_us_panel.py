"""Build the full U.S. daily excess-return panel from yfinance.

Universe: currently-traded U.S. common stocks from nasdaqtraded.txt (ETFs, test issues,
warrants/units/preferred excluded). Excess return per the paper: total return including
dividends (yfinance auto-adjusted close pct-change) minus Ken French's daily risk-free rate.

Hardened after the first attempt was silently rate-limited into a 973-ticker,
alphabetically-truncated panel (chunks 12-19 came back empty and were concatenated anyway):
  - small chunks with a pause between them, exponential backoff and retries on empty chunks
  - sanity gates (ticker count, alphabet coverage) before anything is written
  - temp file ending in .npz (np.savez appends the suffix otherwise) + atomic rename

Known residual limitation: yfinance has no delisted names -> survivorship bias, reported
not fixed. Output: data/us_panel.npz + data/us_universe.csv, and a per-year count of
first-trading-day cohorts (the IPO sample for run_ipo_us.py).
"""

import io
import os
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import yfinance as yf

OUT = "/home/ebenezer0616/IPO_test/data"
START, END = "2014-06-01", "2024-01-01"
CHUNK, PAUSE, RETRIES = 50, 20, 3
NASDAQ_SYMS = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
FF_DAILY = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
            "F-F_Research_Data_Factors_daily_CSV.zip")


def fetch_risk_free():
    with urllib.request.urlopen(FF_DAILY, timeout=120) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    raw = z.read(z.namelist()[0]).decode("latin-1")
    rows = [l for l in raw.splitlines() if l[:8].strip().isdigit() and len(l[:8].strip()) == 8]
    df = pd.DataFrame([l.split(",") for l in rows]).iloc[:, [0, 4]]
    return pd.Series(df[4].astype(float).values / 100.0,
                     index=pd.to_datetime(df[0].str.strip(), format="%Y%m%d")).sort_index()


def fetch_universe():
    req = urllib.request.Request(NASDAQ_SYMS, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        txt = r.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(txt), sep="|")
    df = df[(df["Nasdaq Traded"] == "Y") & (df["ETF"] == "N") & (df["Test Issue"] == "N")]
    df = df[df["Security Name"].str.contains("Common Stock", na=False)]
    bad = df["Security Name"].str.contains("Warrant|Unit|Preferred|Right|Depositary",
                                           case=False, na=False)
    sym = df.loc[~bad, "NASDAQ Symbol"].astype(str)
    sym = sym[~sym.str.contains(r"[\$\.\^]", regex=True)]
    return sorted(set(sym))


def download(tickers):
    closes, volumes = [], []
    for i in range(0, len(tickers), CHUNK):
        part = tickers[i:i + CHUNK]
        got = None
        for attempt in range(RETRIES):
            d = yf.download(part, start=START, end=END, auto_adjust=True,
                            progress=False, threads=True, group_by="column")
            if d is not None and len(d) and d["Close"].notna().any().sum() > 0:
                got = d
                break
            wait = PAUSE * (2 ** attempt)
            print(f"  chunk {i//CHUNK+1}: empty, backing off {wait}s", flush=True)
            time.sleep(wait)
        if got is None:
            print(f"  chunk {i//CHUNK+1}: FAILED after {RETRIES} retries", flush=True)
            continue
        closes.append(got["Close"])
        volumes.append(got["Volume"] * got["Close"])
        print(f"  chunk {i//CHUNK+1}/{-(-len(tickers)//CHUNK)}: "
              f"{got['Close'].notna().any().sum()}/{len(part)} with data", flush=True)
        time.sleep(PAUSE)
    return closes, volumes


def main():
    rf = fetch_risk_free()
    tickers = fetch_universe()
    print(f"universe: {len(tickers)} U.S. common stocks; rf mean {rf.mean()*252:.2%}/yr",
          flush=True)

    closes, volumes = download(tickers)
    px = pd.concat(closes, axis=1).sort_index()
    dv = pd.concat(volumes, axis=1).sort_index()
    px = px.loc[:, ~px.columns.duplicated()]
    dv = dv.loc[:, ~dv.columns.duplicated()]
    px.index = pd.to_datetime(px.index).tz_localize(None)

    have = px.notna().sum() > 60          # keep young IPOs; 60 obs = enough for ctx30+hor30
    px, dv = px.loc[:, have], dv.loc[:, have]

    # sanity gates before writing anything
    letters = {c[0] for c in px.columns}
    assert len(px.columns) >= 2500, f"only {len(px.columns)} tickers - refusing to write"
    assert len(letters) >= 20, f"alphabet truncated: {sorted(letters)}"
    print(f"\ncoverage: {len(px.columns)}/{len(tickers)} tickers, "
          f"letters {min(letters)}-{max(letters)}")

    ret = px.pct_change(fill_method=None)
    ret = ret.where(ret.abs() < 10.0)                     # paper: +-1000% bound
    exret = ret.sub(rf.reindex(ret.index), axis=0)

    tmp = os.path.join(OUT, "us_panel_tmp.npz")           # must end in .npz (savez appends it)
    np.savez_compressed(tmp,
        dates=exret.index.values.astype("datetime64[D]"),
        tickers=np.array(exret.columns, dtype=object),
        ret=ret.values.astype(np.float32),
        exret=exret.values.astype(np.float32),
        dollar_vol=dv.values.astype(np.float32))
    os.replace(tmp, os.path.join(OUT, "us_panel.npz"))
    pd.DataFrame({"ticker": exret.columns, "n_obs": exret.notna().sum().values,
                  "median_dollar_vol": dv.median().values}).to_csv(
        os.path.join(OUT, "us_universe.csv"), index=False)

    first = exret.notna().idxmax()[exret.notna().any()]
    cohort = pd.Series(pd.to_datetime(first.values)).dt.year.value_counts().sort_index()
    print(f"panel: {exret.shape[0]} days x {exret.shape[1]} tickers, "
          f"{exret.index[0].date()} -> {exret.index[-1].date()}")
    print("first-trading-day cohorts (IPO sample):")
    print(cohort.to_string())
    print("saved -> data/us_panel.npz")


if __name__ == "__main__":
    main()
