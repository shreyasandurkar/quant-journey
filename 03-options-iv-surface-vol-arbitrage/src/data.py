"""Data access: live option chain snapshot + underlying price history."""
import os
import datetime as dt
import pandas as pd
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

TICKER = "SPY"
RISK_FREE_RATE = 0.045  # approx short-term T-bill yield, held flat for simplicity


def download_option_chain(ticker: str = TICKER, max_expiries: int = 8) -> pd.DataFrame:
    """Snapshot of calls across all available expiries (yfinance only
    exposes the current chain -- there is no free historical options
    endpoint, see README limitations)."""
    cache_path = os.path.join(CACHE_DIR, f"{ticker}_chain.csv")
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["expiry"])
        return df

    tk = yf.Ticker(ticker)
    expiries = tk.options[:max_expiries]
    spot = tk.history(period="1d")["Close"].iloc[-1]
    today = dt.date.today()

    rows = []
    for exp in expiries:
        chain = tk.option_chain(exp)
        calls = chain.calls.copy()
        calls["kind"] = "call"
        puts = chain.puts.copy()
        puts["kind"] = "put"
        both = pd.concat([calls, puts])
        both["expiry"] = pd.Timestamp(exp)
        both["T"] = (pd.Timestamp(exp).date() - today).days / 365.0
        both["spot"] = spot
        rows.append(both[["strike", "lastPrice", "bid", "ask", "impliedVolatility",
                           "volume", "openInterest", "kind", "expiry", "T", "spot"]])
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(cache_path, index=False)
    return df


def download_underlying_history(ticker: str = TICKER, period: str = "5y") -> pd.Series:
    cache_path = os.path.join(CACHE_DIR, f"{ticker}_hist.csv")
    if os.path.exists(cache_path):
        s = pd.read_csv(cache_path, index_col=0, parse_dates=True)["Close"]
        return s
    hist = yf.download(ticker, period=period, interval="1d", auto_adjust=True,
                        progress=False)["Close"]
    hist = hist.squeeze()
    hist.to_csv(cache_path, header=["Close"])
    return hist


def download_vix_history(period: str = "5y") -> pd.Series:
    """Real historical VIX levels, used as the implied-vol input to the
    delta-hedged backtest (yfinance has no free historical option-price
    archive, so VIX is the best available proxy for market-implied vol)."""
    cache_path = os.path.join(CACHE_DIR, "vix_hist.csv")
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)["Close"]
    hist = yf.download("^VIX", period=period, interval="1d", auto_adjust=True,
                        progress=False)["Close"]
    hist = hist.squeeze()
    hist.to_csv(cache_path, header=["Close"])
    return hist


if __name__ == "__main__":
    chain = download_option_chain()
    hist = download_underlying_history()
    print(chain.shape, hist.shape)
    print(chain.head())
