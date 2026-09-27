"""Portfolio construction: Mean-Variance (Markowitz), Hierarchical Risk
Parity (HRP), and a simple Black-Litterman posterior-return blend.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def min_variance_weights(cov: pd.DataFrame, max_weight: float = 0.25) -> pd.Series:
    n = len(cov)
    x0 = np.ones(n) / n
    bounds = [(0.0, max_weight)] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    def obj(w):
        return w @ cov.values @ w

    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                    options={"maxiter": 500, "ftol": 1e-10})
    w = res.x if res.success else x0
    return pd.Series(w, index=cov.columns)


def max_sharpe_weights(mu: pd.Series, cov: pd.DataFrame, rf: float = 0.0,
                        max_weight: float = 0.25) -> pd.Series:
    n = len(cov)
    x0 = np.ones(n) / n
    bounds = [(0.0, max_weight)] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    def neg_sharpe(w):
        ret = w @ mu.values - rf
        vol = np.sqrt(w @ cov.values @ w)
        return -ret / vol if vol > 0 else 1e6

    res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=cons,
                    options={"maxiter": 500, "ftol": 1e-10})
    w = res.x if res.success else x0
    return pd.Series(w, index=cov.columns)


def _quasi_diag(link) -> list:
    link = link.astype(int)
    root = link[-1, 0], link[-1, 1]
    n_leaves = link.shape[0] + 1

    def recurse(node):
        if node < n_leaves:
            return [node]
        i = node - n_leaves
        left, right = link[i, 0], link[i, 1]
        return recurse(left) + recurse(right)

    return recurse(2 * n_leaves - 2)


def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    """Hierarchical Risk Parity (Lopez de Prado, 2016):
    1. Hierarchically cluster assets by correlation distance.
    2. Quasi-diagonalize the covariance matrix by cluster order.
    3. Recursive bisection: split risk equally between sub-clusters,
       weighted by inverse-variance, no matrix inversion required.
    """
    corr = returns.corr()
    dist = np.sqrt(0.5 * (1 - corr))
    condensed = squareform(dist.values, checks=False)
    link = linkage(condensed, method="single")

    order = _quasi_diag(link)
    tickers = corr.columns[order].tolist()

    cov = returns.cov()
    weights = pd.Series(1.0, index=tickers)
    clusters = [tickers]

    while clusters:
        clusters = [c[start:end] for c in clusters
                    for start, end in [(0, len(c) // 2), (len(c) // 2, len(c))]
                    if len(c) > 1]
        for i in range(0, len(clusters), 2):
            if i + 1 >= len(clusters):
                continue
            c0, c1 = clusters[i], clusters[i + 1]

            def cluster_var(c):
                sub_cov = cov.loc[c, c]
                ivp = 1.0 / np.diag(sub_cov)
                ivp /= ivp.sum()
                return ivp @ sub_cov.values @ ivp

            v0, v1 = cluster_var(c0), cluster_var(c1)
            alpha = 1 - v0 / (v0 + v1)
            weights[c0] *= alpha
            weights[c1] *= (1 - alpha)

    weights /= weights.sum()
    return weights.reindex(returns.columns)


def black_litterman_posterior(mu_prior: pd.Series, cov: pd.DataFrame,
                               views: dict, tau: float = 0.05,
                               view_confidence: float = 0.5) -> pd.Series:
    """Simplified Black-Litterman: blend an equilibrium prior (e.g.
    reverse-optimized market-cap-weighted return) with a small number of
    absolute analyst 'views' (ticker -> expected annual return), weighted
    by `view_confidence` (0 = ignore views, 1 = fully trust views).
    """
    assets = mu_prior.index
    P = np.zeros((len(views), len(assets)))
    Q = np.zeros(len(views))
    for i, (tkr, view_ret) in enumerate(views.items()):
        P[i, assets.get_loc(tkr)] = 1.0
        Q[i] = view_ret

    omega = np.diag(np.diag(P @ (tau * cov.values) @ P.T)) / max(view_confidence, 1e-6)
    tau_sigma = tau * cov.values

    middle = np.linalg.inv(np.linalg.inv(tau_sigma) + P.T @ np.linalg.inv(omega) @ P)
    posterior = middle @ (np.linalg.inv(tau_sigma) @ mu_prior.values + P.T @ np.linalg.inv(omega) @ Q)
    return pd.Series(posterior, index=assets)
