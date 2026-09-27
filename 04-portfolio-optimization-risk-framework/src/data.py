"""Data access: multi-asset universe of daily prices for the portfolio
optimization framework (equities across sectors + a bond ETF + gold, to
give the optimizers genuine diversification benefit to work with).
"""
import os
import pandas as pd
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",   # tech
    "JPM", "BAC", "GS",                          # financials
    "JNJ", "UNH", "LLY",                         # healthcare
    "XOM", "CVX",                                # energy
    "PG", "KO", "WMT",                           # staples
    "CAT", "HON",                                # industrials
    "NEE", "DUK",                                # utilities
    "TLT",                                       # long-term treasuries
    "GLD",                                       # gold
]


def download_prices(start="2015-01-01", end=None):
    cache_path = os.path.join(CACHE_DIR, "daily_prices.csv")
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    raw = yf.download(UNIVERSE, start=start, end=end, interval="1d",
                       auto_adjust=True, progress=False, group_by="ticker",
                       threads=True)
    closes = {}
    for tkr in UNIVERSE:
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
