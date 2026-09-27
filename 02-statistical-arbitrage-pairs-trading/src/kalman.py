"""Kalman filter for dynamic hedge-ratio estimation (time-varying beta),
as an alternative to a static/rolling-OLS regression.

State-space form:
    state:  x_t = [alpha_t, beta_t]'   (random walk)
    obs:    y_t = alpha_t + beta_t * x_t_price + eps_t
"""
import numpy as np
import pandas as pd


def kalman_dynamic_beta(y: pd.Series, x: pd.Series, delta: float = 1e-4,
                         obs_var: float = 1e-3) -> pd.DataFrame:
    """Simple 2-state (intercept, slope) Kalman filter.

    delta: controls state transition variance (how fast beta can drift).
    obs_var: observation noise variance.
    Returns DataFrame with columns alpha, beta, spread (= y - alpha - beta*x).
    """
    n = len(y)
    Vw = delta / (1 - delta) * np.eye(2)  # state transition covariance
    Ve = obs_var

    theta = np.zeros((n, 2))  # [alpha, beta] over time
    P = np.eye(2)
    theta_t = np.array([0.0, 1.0])

    alphas, betas = np.zeros(n), np.zeros(n)
    for t in range(n):
        F = np.array([1.0, x.iloc[t]])

        if t > 0:
            P = P + Vw  # predict step (random-walk state)

        y_hat = F @ theta_t
        e = y.iloc[t] - y_hat
        Q = F @ P @ F.T + Ve
        K = P @ F / Q  # Kalman gain

        theta_t = theta_t + K * e
        P = P - np.outer(K, F) @ P

        alphas[t], betas[t] = theta_t
        theta[t] = theta_t

    out = pd.DataFrame({"alpha": alphas, "beta": betas}, index=y.index)
    out["spread"] = y.values - out["alpha"].values - out["beta"].values * x.values
    return out
