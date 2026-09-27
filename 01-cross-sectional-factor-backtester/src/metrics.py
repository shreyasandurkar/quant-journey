"""Performance evaluation metrics for the long-short factor portfolio."""
import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 12


def annualized_return(returns: pd.Series) -> float:
    r = returns.dropna()
    return (1 + r).prod() ** (PERIODS_PER_YEAR / len(r)) - 1 if len(r) else np.nan


def annualized_vol(returns: pd.Series) -> float:
    return returns.dropna().std() * np.sqrt(PERIODS_PER_YEAR)


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    r = returns.dropna() - rf / PERIODS_PER_YEAR
    if r.std() == 0 or len(r) == 0:
        return np.nan
    return (r.mean() / r.std()) * np.sqrt(PERIODS_PER_YEAR)


def information_ratio(returns: pd.Series, benchmark: pd.Series) -> float:
    active = (returns - benchmark).dropna()
    if active.std() == 0 or len(active) == 0:
        return np.nan
    return (active.mean() / active.std()) * np.sqrt(PERIODS_PER_YEAR)


def max_drawdown(returns: pd.Series) -> float:
    r = returns.dropna()
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1.0
    return dd.min()


def cumulative_curve(returns: pd.Series) -> pd.Series:
    return (1 + returns.fillna(0)).cumprod()


def summary_table(net_returns: pd.Series, gross_returns: pd.Series,
                   turnover: pd.Series, benchmark: pd.Series = None) -> pd.DataFrame:
    rows = {}
    for label, r in [("Gross", gross_returns), ("Net of costs", net_returns)]:
        rows[label] = {
            "Ann. Return": annualized_return(r),
            "Ann. Vol": annualized_vol(r),
            "Sharpe": sharpe_ratio(r),
            "Max Drawdown": max_drawdown(r),
            "Avg Monthly Turnover": turnover.mean(),
        }
        if benchmark is not None:
            rows[label]["Information Ratio"] = information_ratio(r, benchmark)
    return pd.DataFrame(rows).T
