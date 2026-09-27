"""Performance & risk metrics: Sharpe, drawdown, historical VaR/CVaR."""
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


def historical_var(returns: pd.Series, alpha: float = 0.05) -> float:
    return -np.percentile(returns.dropna(), alpha * 100)


def historical_cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    r = returns.dropna()
    var = np.percentile(r, alpha * 100)
    tail = r[r <= var]
    return -tail.mean() if len(tail) else np.nan


def cumulative_curve(returns: pd.Series) -> pd.Series:
    return (1 + returns.fillna(0)).cumprod()


def summary_table(strategy_returns: dict) -> pd.DataFrame:
    rows = {}
    for name, r in strategy_returns.items():
        rows[name] = {
            "Ann. Return": annualized_return(r),
            "Ann. Vol": annualized_vol(r),
            "Sharpe": sharpe_ratio(r),
            "Max Drawdown": max_drawdown(r),
            "VaR (95%, daily)": historical_var(r),
            "CVaR (95%, daily)": historical_cvar(r),
        }
    return pd.DataFrame(rows).T
