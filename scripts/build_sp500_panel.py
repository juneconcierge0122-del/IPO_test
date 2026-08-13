"""Download daily prices for every 2016-2026 S&P 500 member and build the excess-return panel.

Universe: data/sp500_all_tickers.json — reconstructed point-in-time membership from the
Wikipedia constituents page + the historical-changes table (snapshot 2026-08-13), so it
includes names later REMOVED from the index. Tickers that were acquired/delisted will have
no yfinance data; the miss count is reported (that residue is the remaining survivorship bias).

Hardened against the rate-limiting that corrupted the first US panel:
  - small chunks with a pause between them, exponential backoff and retries on empty chunks
  - the panel is only written if coverage is sane, via a temp file + atomic rename
"""

import io
import json
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


def download(tickers):
    closes, volumes, failed = [], [], []
    for i in range(0, len(tickers), CHUNK):
        part = [t.replace(".", "-") for t in tickers[i:i + CHUNK]]  # BRK.B -> BRK-B for yahoo
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
            failed.extend(part)
            continue
        n_ok = got["Close"].notna().any().sum()
        closes.append(got["Close"])
        volumes.append(got["Volume"] * got["Close"])
        print(f"  chunk {i//CHUNK+1}/{-(-len(tickers)//CHUNK)}: {n_ok}/{len(part)} with data",
              flush=True)
        time.sleep(PAUSE)
    return closes, volumes, failed


def main():
    tickers = json.load(open(os.path.join(OUT, "sp500_all_tickers.json")))
    rf = fetch_risk_free()
    print(f"universe: {len(tickers)} historical S&P 500 members; rf mean {rf.mean()*252:.2%}/yr")

    closes, volumes, failed = download(tickers)
    px = pd.concat(closes, axis=1).sort_index()
    dv = pd.concat(volumes, axis=1).sort_index()
    px = px.loc[:, ~px.columns.duplicated()]
    dv = dv.loc[:, ~dv.columns.duplicated()]
    px.index = pd.to_datetime(px.index).tz_localize(None)

    have = px.notna().sum() > 252
    px, dv = px.loc[:, have], dv.loc[:, have]
    got = set(c.replace("-", ".") for c in px.columns)
    missing = sorted(set(tickers) - got)
    print(f"\ncoverage: {len(got)}/{len(tickers)} tickers "
          f"({len(missing)} missing -> residual survivorship bias)")
    print("missing sample:", missing[:20])

    # sanity gates before writing anything
    first_letters = {c[0] for c in px.columns}
    assert len(px.columns) >= 500, f"only {len(px.columns)} tickers - refusing to write"
    assert len(first_letters) >= 20, f"alphabet truncated: {sorted(first_letters)}"

    ret = px.pct_change(fill_method=None)
    ret = ret.where(ret.abs() < 10.0)                     # paper: +-1000% bound
    exret = ret.sub(rf.reindex(ret.index), axis=0)

    tmp = os.path.join(OUT, "sp500_panel_tmp.npz")
    np.savez_compressed(tmp,
        dates=exret.index.values.astype("datetime64[D]"),
        tickers=np.array([c.replace("-", ".") for c in exret.columns], dtype=object),
        ret=ret.values.astype(np.float32),
        exret=exret.values.astype(np.float32),
        dollar_vol=dv.values.astype(np.float32))
    os.replace(tmp, os.path.join(OUT, "sp500_panel.npz"))  # tmp name must end in .npz: savez appends it otherwise
    print(f"panel: {exret.shape[0]} days x {exret.shape[1]} tickers, "
          f"{exret.index[0].date()} -> {exret.index[-1].date()}")
    print("saved -> data/sp500_panel.npz")


if __name__ == "__main__":
    main()
