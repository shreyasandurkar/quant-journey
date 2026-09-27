"""Performance metrics (daily-periodicity version)."""
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
    return (1 + r).prod() ** (PERIODS_PER_YEAR / len(r)) - 1 if len(r) else np.nan


def annualized_vol(returns: pd.Series) -> float:
    return returns.dropna().std() * np.sqrt(PERIODS_PER_YEAR)


def max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns.fillna(0)).cumprod()
    peak = cum.cummax()
    return (cum / peak - 1.0).min()


def hit_rate(returns: pd.Series) -> float:
    r = returns.dropna()
    r = r[r != 0]
    return (r > 0).mean() if len(r) else np.nan


def cumulative_curve(returns: pd.Series) -> pd.Series:
    return (1 + returns.fillna(0)).cumprod()


def summary_table(net: pd.Series, gross: pd.Series) -> pd.DataFrame:
    rows = {}
    for label, r in [("Gross", gross), ("Net of costs", net)]:
        rows[label] = {
            "Ann. Return": annualized_return(r),
            "Ann. Vol": annualized_vol(r),
            "Sharpe": sharpe_ratio(r),
            "Max Drawdown": max_drawdown(r),
            "Hit Rate": hit_rate(r),
        }
    return pd.DataFrame(rows).T
