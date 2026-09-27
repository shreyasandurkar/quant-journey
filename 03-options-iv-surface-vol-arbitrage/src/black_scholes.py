"""Black-Scholes pricing, Greeks, and Newton-Raphson implied volatility."""
import numpy as np
from scipy.stats import norm


def bs_price(S, K, T, r, sigma, q=0.0, kind="call"):
    if sigma <= 0 or T <= 0:
        intrinsic = max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
        return intrinsic
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_vega(S, K, T, r, sigma, q=0.0):
    if sigma <= 0 or T <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def bs_delta(S, K, T, r, sigma, q=0.0, kind="call"):
    if sigma <= 0 or T <= 0:
        if kind == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if kind == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    return np.exp(-q * T) * (norm.cdf(d1) - 1.0)


def implied_vol(price, S, K, T, r, q=0.0, kind="call", tol=1e-6, max_iter=100):
    """Newton-Raphson with a bisection fallback (robust to bad starting
    points / near-zero vega, which trips up pure Newton-Raphson)."""
    if price <= 0 or T <= 0:
        return np.nan
    intrinsic = max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
    if price < intrinsic - 1e-8:
        return np.nan  # violates no-arbitrage bound, cannot invert

    sigma = 0.3
    for _ in range(max_iter):
        px = bs_price(S, K, T, r, sigma, q, kind)
        vega = bs_vega(S, K, T, r, sigma, q)
        diff = px - price
        if abs(diff) < tol:
            return sigma
        if vega < 1e-8:
            break
        sigma -= diff / vega
        if sigma <= 0 or sigma > 5:
            break

    # Bisection fallback
    lo, hi = 1e-4, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        px = bs_price(S, K, T, r, mid, q, kind)
        if px > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            return mid
    return np.nan
