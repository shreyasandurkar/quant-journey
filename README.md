# Journey with Quant

A collection of quantitative research projects, each in its own
self-contained folder, written like short research notes rather than
plain scripts: hypothesis, methodology, honest results, and a critical
evaluation of *why* each strategy is limited or fails in places.

Each project pulls real market data (via `yfinance`), runs vectorized
NumPy/Pandas/SciPy analysis, and saves reproducible figures + CSV outputs.

## Projects

| # | Project | Core QR Skills |
|---|---|---|
| 1 | [Cross-Sectional Factor Backtester](01-cross-sectional-factor-backtester/) | Cross-sectional regression, sector/size neutralization, survivorship bias |
| 2 | [Statistical Arbitrage & Pairs Trading Engine](02-statistical-arbitrage-pairs-trading/) | Cointegration (ADF, Johansen), Kalman filtering, Ornstein-Uhlenbeck processes |
| 3 | [Options IV Surface & Volatility Arbitrage](03-options-iv-surface-vol-arbitrage/) | Black-Scholes inversion, no-arbitrage checks, variance risk premium |
| 4 | [Portfolio Optimization & Risk Modeling Framework](04-portfolio-optimization-risk-framework/) | Covariance shrinkage/RMT, convex optimization, HRP, VaR/CVaR |

## Philosophy

Each project's README documents:

1. **The hypothesis being tested**, stated up front.
2. **Methodology**, with the actual math being implemented.
3. **Results**, including the honest headline numbers (not cherry-picked).
4. **A "why this fails or is limited" section** — every project here has
   at least one genuine negative or marginal result, disclosed
   deliberately rather than hidden, along with the data limitations of
   using free sources (`yfinance`) instead of institutional data feeds.
5. **What a production version would need** to be closer to
   institutional-grade.

## Setup

Each project has its own `requirements.txt` and is runnable independently:

```bash
cd 01-cross-sectional-factor-backtester
pip install -r requirements.txt
python run_backtest.py
```

Downloaded data is cached locally under each project's `cache/` folder
(gitignored) so repeated runs are fast and don't hammer Yahoo Finance.
