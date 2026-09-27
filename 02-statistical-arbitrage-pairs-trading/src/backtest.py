"""Signal generation and backtest engine for the pairs trade."""
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore", category=FutureWarning)


def rolling_ols_beta(y: pd.Series, x: pd.Series, window: int = 60) -> pd.Series:
    """Rolling-window static hedge ratio (baseline to compare vs Kalman)."""
    betas = pd.Series(index=y.index, dtype=float)
    for i in range(window, len(y)):
        yw, xw = y.iloc[i - window:i], x.iloc[i - window:i]
        X = np.column_stack([np.ones(window), xw.values])
        b, *_ = np.linalg.lstsq(X, yw.values, rcond=None)
        betas.iloc[i] = b[1]
    return betas


def rolling_regime_filter(spread: pd.Series, window: int = 120, p_thresh: float = 0.10) -> pd.Series:
    """Rolling ADF p-value on the trailing window; True = still stationary
    enough to trust mean-reversion (a simple structural-break / regime
    filter -- if the trailing spread stops looking stationary, e.g. after
    an earnings surprise breaks the economic link, we stand aside).
    """
    ok = pd.Series(False, index=spread.index)
    vals = spread.values
    for i in range(window, len(spread)):
        try:
            p = adfuller(vals[i - window:i], autolag="AIC")[1]
            ok.iloc[i] = p < p_thresh
        except Exception:
            ok.iloc[i] = False
    return ok


def zscore_signal(spread: pd.Series, window: int = 20) -> pd.Series:
    mu = spread.rolling(window).mean()
    sd = spread.rolling(window).std()
    return (spread - mu) / sd


def generate_positions(z: pd.Series, regime_ok: pd.Series, entry: float = 2.0,
                        exit: float = 0.5, stop: float = 4.0) -> pd.Series:
    """State machine: +1 = long spread (long y, short beta*x), -1 = short
    spread, 0 = flat. Enter at |z|>entry, exit at |z|<exit or reversion,
    hard stop-loss if |z|>stop (spread diverging further = broken
    cointegration, cut losses rather than average down). Requires the
    regime filter to be OK to *enter* a new trade (but not to exit one).
    """
    pos = pd.Series(0, index=z.index, dtype=int)
    state = 0
    for i in range(len(z)):
        zi = z.iloc[i]
        if pd.isna(zi):
            pos.iloc[i] = state
            continue
        if state == 0:
            if regime_ok.iloc[i]:
                if zi > entry:
                    state = -1  # spread too high -> short spread
                elif zi < -entry:
                    state = 1   # spread too low -> long spread
        elif state == 1:
            if zi >= -exit or zi < -stop:
                state = 0
        elif state == -1:
            if zi <= exit or zi > stop:
                state = 0
        pos.iloc[i] = state
    return pos


def backtest_pair(y: pd.Series, x: pd.Series, beta: pd.Series, positions: pd.Series,
                   cost_bps: float = 5.0) -> pd.DataFrame:
    """Daily P&L of a beta-neutral spread position: long/short 1 unit of y
    vs beta units of x, position sized in the prior signal's direction.
    """
    y_ret = y.pct_change()
    x_ret = x.pct_change()
    b = beta.reindex(y.index).ffill()

    # Spread return approximated as y_ret - beta * x_ret (dollar-neutral proxy)
    spread_ret = y_ret - b * x_ret
    strat_ret = positions.shift(1).fillna(0) * spread_ret

    trade = positions.diff().abs().fillna(0)
    cost = trade * (cost_bps / 1e4) * (1 + b.abs())
    net_ret = strat_ret - cost

    return pd.DataFrame({"gross_return": strat_ret, "net_return": net_ret,
                          "position": positions, "beta": b, "trades": trade})
