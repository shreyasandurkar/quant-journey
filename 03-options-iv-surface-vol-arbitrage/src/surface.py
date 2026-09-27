"""Implied volatility surface construction + static arbitrage checks."""
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from src.black_scholes import implied_vol, bs_price
from src.data import RISK_FREE_RATE


def compute_iv_surface(chain: pd.DataFrame, r: float = RISK_FREE_RATE,
                        min_volume: int = 1) -> pd.DataFrame:
    """Compute our own Newton-Raphson IV (not yfinance's) from mid prices,
    using out-of-the-money options only (more reliable quotes/liquidity)."""
    df = chain.copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df = df[(df["mid"] > 0.01) & (df["T"] > 1 / 365) & (df["volume"].fillna(0) >= min_volume)]

    # Keep OTM options: calls with strike >= spot, puts with strike <= spot
    otm_calls = df[(df.kind == "call") & (df.strike >= df.spot)]
    otm_puts = df[(df.kind == "put") & (df.strike <= df.spot)]
    df = pd.concat([otm_calls, otm_puts])

    ivs = []
    for row in df.itertuples():
        iv = implied_vol(row.mid, row.spot, row.strike, row.T, r, kind=row.kind)
        ivs.append(iv)
    df = df.assign(iv=ivs)
    df = df[(df.iv > 0.01) & (df.iv < 3.0)]
    df["moneyness"] = np.log(df.strike / df.spot)
    return df


def fit_smile_splines(surface: pd.DataFrame) -> dict:
    """One cubic spline of IV vs log-moneyness per expiry."""
    splines = {}
    for exp, grp in surface.groupby("expiry"):
        grp = grp.sort_values("moneyness").drop_duplicates("moneyness")
        if len(grp) < 5:
            continue
        splines[exp] = CubicSpline(grp["moneyness"].values, grp["iv"].values,
                                    extrapolate=False)
    return splines


def check_butterfly_arbitrage(surface: pd.DataFrame, r: float = RISK_FREE_RATE) -> pd.DataFrame:
    """Static butterfly-arbitrage check: for a fixed expiry, call prices
    must be a convex function of strike. Discretely, for consecutive
    strikes K1<K2<K3 (evenly enough spaced), the butterfly price
        C(K1) - 2*C(K2) + C(K3)
    must be >= 0 (a long butterfly has non-negative value / non-negative
    risk-neutral density). We reprice with the fitted smile so the check
    is smile-consistent rather than raw-quote-noise-sensitive.
    """
    violations = []
    for exp, grp in surface.groupby("expiry"):
        grp = grp[grp.kind == "call"].sort_values("strike").drop_duplicates("strike")
        if len(grp) < 3:
            continue
        strikes = grp["strike"].values
        T = grp["T"].iloc[0]
        spot = grp["spot"].iloc[0]
        ivs = grp["iv"].values
        prices = np.array([bs_price(spot, K, T, r, iv, kind="call")
                            for K, iv in zip(strikes, ivs)])
        for i in range(1, len(strikes) - 1):
            k1, k2, k3 = strikes[i - 1], strikes[i], strikes[i + 1]
            # only check roughly-evenly-spaced triples
            if abs((k3 - k2) - (k2 - k1)) > 0.25 * (k2 - k1):
                continue
            fly = prices[i - 1] - 2 * prices[i] + prices[i + 1]
            if fly < -1e-6:
                violations.append({"expiry": exp, "K1": k1, "K2": k2, "K3": k3,
                                    "butterfly_value": fly})
    return pd.DataFrame(violations)


def check_calendar_arbitrage(surface: pd.DataFrame) -> pd.DataFrame:
    """Static calendar-arbitrage check: total implied variance
    w(k,T) = iv(k,T)^2 * T at fixed log-moneyness k must be non-decreasing
    in T (a longer-dated option cannot be cheaper in variance terms than a
    shorter-dated one at the same moneyness, or you could sell the near
    option and buy the far one for a free lunch).
    """
    splines = fit_smile_splines(surface)
    expiries = sorted(splines.keys())
    grid = np.linspace(-0.15, 0.15, 31)
    violations = []
    for k in grid:
        w_prev, exp_prev, T_prev = None, None, None
        for exp in expiries:
            iv = splines[exp](k)
            if np.isnan(iv):
                continue
            T = surface[surface.expiry == exp]["T"].iloc[0]
            w = iv ** 2 * T
            if w_prev is not None and w < w_prev - 1e-6:
                violations.append({"moneyness": k, "expiry_short": exp_prev,
                                    "expiry_long": exp, "w_short": w_prev, "w_long": w})
            w_prev, exp_prev, T_prev = w, exp, T
    return pd.DataFrame(violations)
