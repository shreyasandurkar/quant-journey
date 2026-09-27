"""Delta-hedged short-volatility backtest using the classic gamma P&L
approximation, driven by *real* historical VIX (as the implied-vol proxy)
and realized SPY returns -- see README for why this approximation is used
in place of historical option prices (which are not freely available).
"""
import numpy as np
import pandas as pd

from src.black_scholes import bs_price, bs_delta
from scipy.stats import norm


def bs_gamma(S, K, T, r, sigma, q=0.0):
    if sigma <= 0 or T <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def realized_vol(prices: pd.Series, window: int = 21) -> pd.Series:
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)


def gamma_pnl_backtest(prices: pd.Series, vix: pd.Series, tenor_days: int = 21,
                        r: float = 0.045) -> pd.DataFrame:
    """Roll a fresh ATM straddle every `tenor_days`. Each day, mark the
    daily delta-hedged P&L of the SHORT straddle position using:

        dPnL_t = 0.5 * Gamma_t * S_t^2 * (sigma_impl^2 - r_t^2 * 252) * dt

    where sigma_impl is fixed at the VIX level observed at trade inception
    and r_t is that day's realized log return. This is the standard
    "vol trading" P&L attribution used to isolate the variance risk
    premium (VRP) captured by a short-vol, delta-hedged book.
    """
    df = pd.DataFrame({"S": prices}).join(vix.rename("vix")).dropna()
    df["log_ret"] = np.log(df["S"] / df["S"].shift(1))

    daily_pnl = pd.Series(0.0, index=df.index)
    entry_idx = 0
    n = len(df)
    dt = 1.0 / 252

    while entry_idx < n - 1:
        end_idx = min(entry_idx + tenor_days, n - 1)
        K = df["S"].iloc[entry_idx]
        sigma_impl = df["vix"].iloc[entry_idx] / 100.0

        for i in range(entry_idx + 1, end_idx + 1):
            T_remaining = max((end_idx - i + 1), 1) / 252
            S_prev = df["S"].iloc[i - 1]
            gamma = bs_gamma(S_prev, K, T_remaining, r, sigma_impl)
            r_t = df["log_ret"].iloc[i]
            pnl = 0.5 * gamma * S_prev ** 2 * (sigma_impl ** 2 - r_t ** 2 * 252) * dt
            daily_pnl.iloc[i] += pnl

        entry_idx = end_idx

    notional = df["S"].iloc[0] ** 2  # normalize P&L into a return-like series
    out = pd.DataFrame({
        "S": df["S"],
        "vix": df["vix"],
        "daily_pnl": daily_pnl,
        "daily_return": daily_pnl / notional,
    })
    return out
