"""
Refresh data/prices.csv from Stooq.

Stooq (https://stooq.com) provides free, daily-frequency US equity history
without an API key. The 'close' column is the regular close, not adjusted
for dividends; sector dividend yields are 1-2 % annually so absolute Sharpe
ratios are biased slightly downward, but the cross-sectional comparison
between optimisation methods is unaffected because every sector faces the
same omission.

Usage:
    python fetch_prices.py

The script writes to data/prices.csv in the repository root.
"""

import sys
import time
from pathlib import Path
import pandas as pd

TICKERS = ["XLK", "XLV", "XLF", "XLY", "XLP", "XLE", "XLI",
           "XLB", "XLU", "XLRE", "XLC", "SPY"]
START   = "20150101"
END     = "20250501"
OUT     = Path(__file__).resolve().parent / "data" / "prices.csv"


def fetch_one(t):
    url = (f"https://stooq.com/q/d/l/?s={t.lower()}.us&i=d"
           f"&d1={START}&d2={END}")
    df = pd.read_csv(url, parse_dates=["Date"])
    if df.empty:
        raise RuntimeError("empty response")
    return df.set_index("Date")["Close"].rename(t)


def main():
    OUT.parent.mkdir(exist_ok=True)
    series, missing = [], []
    for t in TICKERS:
        try:
            s = fetch_one(t)
            print(f"{t:5s}  {len(s):>5d} rows  "
                  f"{s.index.min().date()} -> {s.index.max().date()}")
            series.append(s)
        except Exception as e:
            print(f"{t:5s}  failed: {e}", file=sys.stderr)
            missing.append(t)
        time.sleep(0.4)        # polite

    if missing:
        print(f"\n{len(missing)} ticker(s) failed: {missing}", file=sys.stderr)
        print("not overwriting data/prices.csv", file=sys.stderr)
        sys.exit(1)

    px = pd.concat(series, axis=1).sort_index().dropna(how="all")
    px.to_csv(OUT)
    print(f"\nwrote {len(px)} rows x {px.shape[1]} tickers to {OUT}")


if __name__ == "__main__":
    main()
