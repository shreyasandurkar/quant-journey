"""Cointegration testing and Ornstein-Uhlenbeck spread modeling."""
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

warnings.filterwarnings("ignore", category=FutureWarning)


def engle_granger_beta(y: pd.Series, x: pd.Series) -> float:
    """Static OLS hedge ratio: y = beta * x + c."""
    X = np.column_stack([np.ones(len(x)), x.values])
    beta, *_ = np.linalg.lstsq(X, y.values, rcond=None)
    return beta[1]


def adf_test(series: pd.Series) -> dict:
    stat, pvalue, lags, nobs, crit, *_ = adfuller(series.dropna(), autolag="AIC")
    return {"stat": stat, "pvalue": pvalue, "lags": lags, "nobs": nobs, "crit_5%": crit["5%"]}


def johansen_test(y: pd.Series, x: pd.Series) -> dict:
    """Johansen trace test for cointegration rank on the bivariate system.
    Returns the trace statistic and 95% critical value for r<=0 (no
    cointegration) -- rejecting it implies at least one cointegrating vector.
    """
    data = pd.concat([y, x], axis=1).dropna()
    result = coint_johansen(data.values, det_order=0, k_ar_diff=1)
    trace_stat = result.lr1[0]
    crit_95 = result.cvt[0, 1]
    return {"trace_stat": trace_stat, "crit_95%": crit_95,
            "cointegrated": trace_stat > crit_95}


def screen_pair(y: pd.Series, x: pd.Series) -> dict:
    """Run EG-beta + spread ADF + Johansen; return a one-row diagnostic dict."""
    beta = engle_granger_beta(y, x)
    spread = y - beta * x
    adf = adf_test(spread)
    joh = johansen_test(y, x)
    return {
        "beta": beta,
        "adf_pvalue": adf["pvalue"],
        "adf_stat": adf["stat"],
        "adf_crit_5%": adf["crit_5%"],
        "johansen_trace": joh["trace_stat"],
        "johansen_crit_95%": joh["crit_95%"],
        "johansen_cointegrated": joh["cointegrated"],
        "stationary_5%": adf["pvalue"] < 0.05,
    }


def fit_ou_process(spread: pd.Series, periods_per_year: int = 252) -> dict:
    """Fit spread_t = spread_{t-1}*phi + c + eps via OLS, map to
    continuous-time OU: dS = kappa*(theta - S)dt + sigma dW.
    Returns half-life in trading days, theta (long-run mean), sigma.
    """
    s = spread.dropna()
    s_lag = s.shift(1).dropna()
    s_now = s.loc[s_lag.index]
    X = np.column_stack([np.ones(len(s_lag)), s_lag.values])
    coef, *_ = np.linalg.lstsq(X, s_now.values, rcond=None)
    c, phi = coef
    resid = s_now.values - X @ coef
    kappa = -np.log(phi) * periods_per_year if 0 < phi < 1 else np.nan
    theta = c / (1 - phi) if phi != 1 else s.mean()
    half_life = np.log(2) / kappa * periods_per_year if kappa and kappa > 0 else np.nan
    sigma = resid.std() * np.sqrt(periods_per_year)
    return {"phi": phi, "kappa": kappa, "theta": theta,
            "half_life_days": np.log(2) / (-np.log(phi)) if 0 < phi < 1 else np.nan,
            "sigma_annual": sigma}
