"""Long-short decile portfolio construction with transaction-cost drag."""
import numpy as np
import pandas as pd


def decile_weights(score_row: pd.Series, n_deciles: int = 10) -> pd.Series:
    """Equal-weight long the top decile, equal-weight short the bottom
    decile, dollar-neutral (long weights sum to +1, short to -1)."""
    s = score_row.dropna()
    if len(s) < n_deciles * 2:
        return pd.Series(dtype=float)

    ranks = s.rank(pct=True)
    top = ranks[ranks >= 1 - 1.0 / n_deciles].index
    bottom = ranks[ranks <= 1.0 / n_deciles].index

    w = pd.Series(0.0, index=s.index)
    w[top] = 1.0 / len(top)
    w[bottom] = -1.0 / len(bottom)
    return w


def forward_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Return earned from t to t+1 (next-period realized return)."""
    return prices.shift(-1) / prices - 1.0


def run_backtest(score: pd.DataFrame, prices: pd.DataFrame,
                  n_deciles: int = 10, cost_bps: float = 10.0,
                  max_turnover: float = None) -> dict:
    """Monthly rebalanced long-short backtest with linear transaction costs.

    cost_bps: round-trip cost per unit of turnover, in basis points.
    max_turnover: optional cap; if the target turnover exceeds it the
    trade is scaled down (proxy for a rebalancing turnover limit).
    """
    dates = score.index
    rets = forward_returns(prices)

    weights_hist = {}
    prev_w = pd.Series(dtype=float)
    gross_rets, net_rets, turnovers = [], [], []

    for dt in dates:
        target_w = decile_weights(score.loc[dt], n_deciles)
        if target_w.empty:
            gross_rets.append(np.nan)
            net_rets.append(np.nan)
            turnovers.append(np.nan)
            continue

        all_names = target_w.index.union(prev_w.index)
        tw = target_w.reindex(all_names).fillna(0.0)
        pw = prev_w.reindex(all_names).fillna(0.0)

        delta = tw - pw
        turnover = delta.abs().sum() / 2.0  # one-way turnover
        if max_turnover is not None and turnover > max_turnover and turnover > 0:
            scale = max_turnover / turnover
            tw = pw + delta * scale
            turnover = max_turnover

        r = rets.loc[dt].reindex(tw.index).fillna(0.0)
        gross = float((tw * r).sum())
        cost = float(delta.abs().sum() * (cost_bps / 1e4))
        net = gross - cost

        gross_rets.append(gross)
        net_rets.append(net)
        turnovers.append(turnover)
        weights_hist[dt] = tw
        prev_w = tw

    out = pd.DataFrame({
        "gross_return": gross_rets,
        "net_return": net_rets,
        "turnover": turnovers,
    }, index=dates)
    return {"returns": out, "weights": weights_hist}
