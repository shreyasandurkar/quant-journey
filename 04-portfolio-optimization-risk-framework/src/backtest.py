"""Rolling monthly rebalance backtest comparing portfolio construction
methods, with position limits and a turnover cap applied uniformly.
"""
import numpy as np
import pandas as pd

from src.covariance import sample_cov, ledoit_wolf_shrinkage, rmt_filtered_cov
from src.optimizers import min_variance_weights, max_sharpe_weights, hrp_weights


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.to_series().groupby([index.year, index.month]).apply(lambda x: x.iloc[-1]).values


def apply_turnover_cap(target_w: pd.Series, prev_w: pd.Series, max_turnover: float) -> pd.Series:
    all_idx = target_w.index.union(prev_w.index)
    tw = target_w.reindex(all_idx).fillna(0.0)
    pw = prev_w.reindex(all_idx).fillna(0.0)
    delta = tw - pw
    turnover = delta.abs().sum() / 2.0
    if turnover > max_turnover and turnover > 0:
        scale = max_turnover / turnover
        tw = pw + delta * scale
    return tw


def run_strategy_backtest(prices: pd.DataFrame, lookback: int = 252,
                           max_weight: float = 0.25, max_turnover: float = 0.30,
                           cost_bps: float = 10.0) -> dict:
    """Rolling monthly rebalance for four allocation methods:
    Equal-Weight, Min-Variance (sample cov), Min-Variance (Ledoit-Wolf
    shrinkage cov), and Hierarchical Risk Parity.
    """
    daily_ret = prices.pct_change().dropna()
    rebal_dates = month_end_dates(daily_ret.index)

    methods = ["Equal-Weight", "MinVar-Sample", "MinVar-LedoitWolf", "HRP"]
    prev_w = {m: pd.Series(dtype=float) for m in methods}
    port_ret = {m: pd.Series(index=daily_ret.index, dtype=float) for m in methods}
    weights_hist = {m: {} for m in methods}
    turnovers = {m: [] for m in methods}

    for i, dt in enumerate(rebal_dates):
        loc = daily_ret.index.get_loc(dt)
        if loc < lookback:
            continue
        window = daily_ret.iloc[loc - lookback:loc]

        n = window.shape[1]
        ew = pd.Series(1.0 / n, index=window.columns)

        cov_sample = sample_cov(window)
        w_mv_sample = min_variance_weights(cov_sample, max_weight)

        cov_lw, _ = ledoit_wolf_shrinkage(window)
        w_mv_lw = min_variance_weights(cov_lw, max_weight)

        w_hrp = hrp_weights(window)

        targets = {"Equal-Weight": ew, "MinVar-Sample": w_mv_sample,
                   "MinVar-LedoitWolf": w_mv_lw, "HRP": w_hrp}

        next_dt = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else daily_ret.index[-1]
        period_mask = (daily_ret.index > dt) & (daily_ret.index <= next_dt)
        period_rets = daily_ret.loc[period_mask]

        for m in methods:
            tw = apply_turnover_cap(targets[m], prev_w[m], max_turnover)
            union_idx = tw.index.union(prev_w[m].index)
            delta = tw.reindex(union_idx).fillna(0.0) - prev_w[m].reindex(union_idx).fillna(0.0)
            turnover = delta.abs().sum() / 2.0
            cost = turnover * (cost_bps / 1e4)

            if len(period_rets) > 0:
                w_drift = tw.copy()
                for d in period_rets.index:
                    r = period_rets.loc[d].reindex(w_drift.index).fillna(0.0)
                    port_ret[m].loc[d] = (w_drift * r).sum() - (cost if d == period_rets.index[0] else 0.0)
                    w_drift = w_drift * (1 + r)
                    w_drift /= w_drift.sum()

            weights_hist[m][dt] = tw
            turnovers[m].append(turnover)
            prev_w[m] = tw

    return {
        "returns": {m: port_ret[m].dropna() for m in methods},
        "weights": weights_hist,
        "turnover": {m: np.mean(turnovers[m]) for m in methods},
    }
