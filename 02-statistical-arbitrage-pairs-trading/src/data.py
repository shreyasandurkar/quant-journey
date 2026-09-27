"""Data access for the pairs trading engine: daily prices for a candidate
universe of sector-matched large caps, via yfinance, with local caching.
"""
import os
import pandas as pd
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Candidate pairs, grouped by sector/business model similarity -- a sensible
# prior before running statistical tests (never mine cointegration blindly
# across unrelated names; it's a recipe for spurious regressions).
CANDIDATE_PAIRS = [
    ("KO", "PEP"),      # Beverages
    ("XOM", "CVX"),     # Integrated oil majors
    ("JPM", "BAC"),     # Money-center banks
    ("HD", "LOW"),      # Home improvement retail
    ("MA", "V"),         # Payment networks
    ("GS", "MS"),        # Investment banks
    ("UNP", "CSX"),      # Rail (CSX added below as it's not in universe otherwise)
]

ALL_TICKERS = sorted({t for pair in CANDIDATE_PAIRS for t in pair})


def download_prices(start="2018-01-01", end=None):
    cache_path = os.path.join(CACHE_DIR, "daily_prices.csv")
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    raw = yf.download(ALL_TICKERS, start=start, end=end, interval="1d",
                       auto_adjust=True, progress=False, group_by="ticker",
                       threads=True)
    closes = {}
    for tkr in ALL_TICKERS:
        try:
            closes[tkr] = raw[tkr]["Close"]
        except (KeyError, TypeError):
            continue
    prices = pd.DataFrame(closes).sort_index().ffill().dropna()
    prices.to_csv(cache_path)
    return prices


if __name__ == "__main__":
    df = download_prices()
    print(df.shape)
    print(df.tail())
