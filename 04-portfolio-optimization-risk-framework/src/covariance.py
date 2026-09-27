"""Covariance matrix estimators: sample, Ledoit-Wolf shrinkage, and
Random Matrix Theory (Marchenko-Pastur) eigenvalue filtering.
"""
import numpy as np
import pandas as pd


def sample_cov(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.cov()


def ledoit_wolf_shrinkage(returns: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf (2004) shrinkage toward the constant-correlation
    target, with the analytic shrinkage intensity from their paper
    (simplified single-parameter version)."""
    X = returns.values
    T, N = X.shape
    X = X - X.mean(axis=0)
    sample = (X.T @ X) / T

    var = np.diag(sample)
    std = np.sqrt(var)
    corr = sample / np.outer(std, std)
    avg_corr = (corr.sum() - N) / (N * (N - 1))
    target = avg_corr * np.outer(std, std)
    np.fill_diagonal(target, var)

    # Estimate shrinkage intensity (Ledoit-Wolf formula, simplified)
    Xc = X
    pi_mat = np.zeros((N, N))
    for t in range(T):
        outer_t = np.outer(Xc[t], Xc[t])
        pi_mat += (outer_t - sample) ** 2
    pi_mat /= T
    pi_hat = pi_mat.sum()

    rho_hat = 0.0
    gamma_hat = ((target - sample) ** 2).sum()
    kappa_hat = (pi_hat - rho_hat) / gamma_hat if gamma_hat > 0 else 0.0
    shrinkage = max(0.0, min(1.0, kappa_hat / T))

    shrunk = shrinkage * target + (1 - shrinkage) * sample
    return pd.DataFrame(shrunk, index=returns.columns, columns=returns.columns), shrinkage


def rmt_filtered_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """Random Matrix Theory filtering: eigenvalues of the correlation
    matrix below the Marchenko-Pastur upper bound are treated as noise
    and replaced by their average (preserving the trace), keeping only
    eigenvalues/eigenvectors that carry genuine cross-sectional signal.
    """
    X = returns.values
    T, N = X.shape
    std = X.std(axis=0)
    corr = np.corrcoef(X, rowvar=False)

    q = N / T
    lambda_max = (1 + np.sqrt(q)) ** 2

    eigvals, eigvecs = np.linalg.eigh(corr)
    noise_mask = eigvals <= lambda_max
    signal_mask = ~noise_mask

    avg_noise_eigval = eigvals[noise_mask].mean() if noise_mask.any() else 0.0
    cleaned_eigvals = eigvals.copy()
    cleaned_eigvals[noise_mask] = avg_noise_eigval

    cleaned_corr = eigvecs @ np.diag(cleaned_eigvals) @ eigvecs.T
    # rescale to unit diagonal (numerical correlation matrix)
    d = np.sqrt(np.diag(cleaned_corr))
    cleaned_corr = cleaned_corr / np.outer(d, d)

    cleaned_cov = cleaned_corr * np.outer(std, std)
    return pd.DataFrame(cleaned_cov, index=returns.columns, columns=returns.columns), int(signal_mask.sum())
