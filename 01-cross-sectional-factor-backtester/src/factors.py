"""Cross-sectional factor construction: momentum (dynamic) + value/quality
(static snapshot), z-scored, winsorized, and industry/size-neutralized.
"""
import numpy as np
import pandas as pd


def momentum_12_1(prices: pd.DataFrame) -> pd.DataFrame:
    """12-1 month momentum: cumulative return over the last 12 months,
    skipping the most recent month to avoid short-term reversal."""
    ret_12 = prices.shift(1) / prices.shift(12) - 1.0
    return ret_12


def winsorize(df: pd.DataFrame, limits=(0.01, 0.99)) -> pd.DataFrame:
    lo = df.quantile(limits[0], axis=1)
    hi = df.quantile(limits[1], axis=1)
    return df.clip(lower=lo, upper=hi, axis=0)


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


def build_raw_factors(prices: pd.DataFrame, fundamentals: pd.DataFrame) -> dict:
    """Return {factor_name: DataFrame[date x ticker]} of raw factor values."""
    tickers = [t for t in prices.columns if t in fundamentals.index]
    prices = prices[tickers]

    mom = momentum_12_1(prices)

    earnings_yield = 1.0 / fundamentals.loc[tickers, "trailingPE"].astype(float)
    roe = fundamentals.loc[tickers, "returnOnEquity"].astype(float)

    # Broadcast the static snapshot across every rebalance date.
    value = pd.DataFrame(np.tile(earnings_yield.values, (len(prices.index), 1)),
                          index=prices.index, columns=tickers)
    quality = pd.DataFrame(np.tile(roe.values, (len(prices.index), 1)),
                            index=prices.index, columns=tickers)

    return {"Momentum": mom, "Value_EY": value, "Quality_ROE": quality}


def neutralize_factor(factor_row: pd.Series, sector: pd.Series,
                       log_mcap: pd.Series) -> pd.Series:
    """Cross-sectional OLS regression of the factor on sector dummies +
    log market cap; return the residual (idiosyncratic alpha signal)."""
    valid = factor_row.dropna().index.intersection(sector.dropna().index) \
        .intersection(log_mcap.dropna().index)
    if len(valid) < 10:
        return pd.Series(dtype=float)

    y = factor_row.loc[valid].values.astype(float)
    dummies = pd.get_dummies(sector.loc[valid], prefix="sec", drop_first=True).astype(float)
    X = np.column_stack([np.ones(len(valid)), log_mcap.loc[valid].values.astype(float),
                         dummies.values])

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return pd.Series(resid, index=valid)


def neutralize_panel(factor: pd.DataFrame, sector: pd.Series,
                      market_cap: pd.Series) -> pd.DataFrame:
    log_mcap = np.log(market_cap.astype(float))
    out = {}
    for dt, row in factor.iterrows():
        out[dt] = neutralize_factor(row, sector, log_mcap)
    return pd.DataFrame(out).T.reindex(columns=factor.columns)


def composite_score(factors: dict, weights: dict = None) -> pd.DataFrame:
    """Equal- (or user-) weighted average of winsorized, z-scored,
    neutralized factors into a single composite alpha score."""
    names = list(factors.keys())
    weights = weights or {n: 1.0 / len(names) for n in names}
    total = None
    for name in names:
        z = zscore(winsorize(factors[name]))
        total = z * weights[name] if total is None else total + z * weights[name]
    return total
