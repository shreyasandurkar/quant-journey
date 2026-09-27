"""Data access layer: prices + a static fundamentals/sector snapshot from yfinance."""
import os
import time
import pandas as pd
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# A liquid, sector-diverse subset of large-cap US equities (proxy for an
# S&P 500 style universe). A handful of names that were delisted/acquired
# are intentionally kept out of the "current" list -- see README for the
# survivorship-bias discussion.
UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "ORCL", "CRM", "ADBE",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "SCHW", "BLK", "AXP", "SPGI",
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT", "DHR", "BMY",
    "XOM", "CVX", "COP", "SLB", "EOG",
    "PG", "KO", "PEP", "WMT", "COST", "MCD", "NKE", "HD", "LOW", "TGT",
    "CAT", "BA", "HON", "UNP", "GE", "RTX", "LMT", "DE",
    "LIN", "SHW", "NEM",
    "NEE", "DUK", "SO",
    "T", "VZ", "CMCSA", "DIS",
]

FIELDS = ["trailingPE", "returnOnEquity", "sector", "marketCap"]


def download_prices(start="2015-01-01", end=None, interval="1mo"):
    """Monthly adjusted close prices for the whole universe (one bulk call)."""
    cache_path = os.path.join(CACHE_DIR, f"prices_{interval}.csv")
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    raw = yf.download(UNIVERSE, start=start, end=end, interval=interval,
                       auto_adjust=True, progress=False, group_by="ticker",
                       threads=True)
    closes = {}
    for tkr in UNIVERSE:
        try:
            closes[tkr] = raw[tkr]["Close"]
        except (KeyError, TypeError):
            continue
    prices = pd.DataFrame(closes).sort_index()
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.8))
    prices.to_csv(cache_path)
    return prices


def download_fundamental_snapshot():
    """Static snapshot of value/quality/sector data (yfinance has no clean
    point-in-time fundamentals history for free), used as of "today" for
    every rebalance date. This is a documented simplification -- see README.
    """
    cache_path = os.path.join(CACHE_DIR, "fundamentals.csv")
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, index_col=0)

    rows = {}
    for tkr in UNIVERSE:
        try:
            info = yf.Ticker(tkr).get_info()
            rows[tkr] = {f: info.get(f) for f in FIELDS}
        except Exception:
            continue
        time.sleep(0.05)
    df = pd.DataFrame(rows).T
    df.to_csv(cache_path)
    return df


if __name__ == "__main__":
    prices = download_prices()
    fundamentals = download_fundamental_snapshot()
    print(prices.shape, fundamentals.shape)
