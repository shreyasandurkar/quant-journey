"""Performance metrics for the daily-return P&L series."""
import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252


def sharpe_ratio(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.std() == 0 or len(r) == 0:
        return np.nan
    return (r.mean() / r.std()) * np.sqrt(PERIODS_PER_YEAR)


def annualized_return(returns: pd.Series) -> float:
    r = returns.dropna()
    return r.mean() * PERIODS_PER_YEAR


def max_drawdown(cum_pnl: pd.Series) -> float:
    peak = cum_pnl.cummax()
    return (cum_pnl - peak).min()
