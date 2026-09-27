"""Main entry point for the Portfolio Optimization & Risk Framework."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data import download_prices
from src.covariance import sample_cov, ledoit_wolf_shrinkage, rmt_filtered_cov
from src.backtest import run_strategy_backtest
from src.metrics import summary_table, cumulative_curve, max_drawdown

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams["figure.dpi"] = 110


def main():
    prices = download_prices()
    daily_ret = prices.pct_change().dropna()

    # ---- Part A: covariance estimator comparison on the full sample ----
    cov_sample = sample_cov(daily_ret)
    cov_lw, shrinkage = ledoit_wolf_shrinkage(daily_ret)
    cov_rmt, n_signal = rmt_filtered_cov(daily_ret)
    print(f"Ledoit-Wolf shrinkage intensity: {shrinkage:.3f}")
    print(f"RMT: {n_signal} / {len(cov_sample)} eigenvalues kept as signal "
          f"(rest treated as noise)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (name, cov) in zip(axes, [("Sample", cov_sample), ("Ledoit-Wolf", cov_lw),
                                        ("RMT-Filtered", cov_rmt)]):
        corr = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_title(f"{name} Correlation")
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.7, label="Correlation")
    fig.savefig(os.path.join(FIG_DIR, "01_covariance_comparison.png")); plt.close(fig)

    # ---- Part B: rolling backtest across allocation methods ----
    bt = run_strategy_backtest(prices, lookback=252, max_weight=0.25,
                                max_turnover=0.30, cost_bps=10.0)
    returns = bt["returns"]

    table = summary_table(returns)
    table.to_csv(os.path.join(OUT_DIR, "performance_summary.csv"))
    print("\nPerformance summary:\n", table.round(4))

    turnover_df = pd.Series(bt["turnover"])
    turnover_df.to_csv(os.path.join(OUT_DIR, "avg_turnover.csv"))
    print("\nAverage monthly turnover:\n", turnover_df.round(3))

    # ---- Figure 2: cumulative returns ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m, r in returns.items():
        cumulative_curve(r).plot(ax=ax, label=m)
    ax.set_title("Cumulative Growth of $1: Allocation Method Comparison")
    ax.set_ylabel("Growth of $1")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "02_cumulative_returns.png")); plt.close(fig)

    # ---- Figure 3: drawdowns ----
    fig, ax = plt.subplots(figsize=(9, 5))
    for m, r in returns.items():
        cum = cumulative_curve(r)
        dd = cum / cum.cummax() - 1.0
        dd.plot(ax=ax, label=f"{m} (Max {max_drawdown(r):.1%})")
    ax.set_title("Drawdown Comparison")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "03_drawdowns.png")); plt.close(fig)

    # ---- Figure 4: HRP weight allocation over time (stacked area) ----
    hrp_w = pd.DataFrame(bt["weights"]["HRP"]).T.fillna(0.0).sort_index()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.stackplot(hrp_w.index, hrp_w.T.values, labels=hrp_w.columns)
    ax.set_title("HRP Portfolio Weight Allocation Over Time")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7)
    ax.set_ylabel("Weight")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "04_hrp_weights_over_time.png")); plt.close(fig)

    # ---- Figure 5: rolling 6m volatility, Sample-MinVar vs HRP ----
    fig, ax = plt.subplots(figsize=(9, 4.5))
    returns["MinVar-Sample"].rolling(126).std().mul(np.sqrt(252)).plot(ax=ax, label="MinVar (Sample Cov)")
    returns["HRP"].rolling(126).std().mul(np.sqrt(252)).plot(ax=ax, label="HRP")
    returns["Equal-Weight"].rolling(126).std().mul(np.sqrt(252)).plot(ax=ax, label="Equal-Weight", linestyle="--")
    ax.set_title("Rolling 6-Month Annualized Volatility")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "05_rolling_volatility.png")); plt.close(fig)

    print("\nSaved figures to", FIG_DIR)


if __name__ == "__main__":
    main()
