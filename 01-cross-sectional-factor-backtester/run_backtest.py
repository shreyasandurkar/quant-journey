"""Main entry point: builds factors, neutralizes, backtests, and saves
figures + a results table for the Cross-Sectional Factor Backtester.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data import download_prices, download_fundamental_snapshot
from src.factors import build_raw_factors, neutralize_panel, composite_score, zscore, winsorize
from src.backtest import run_backtest, forward_returns
from src.metrics import summary_table, cumulative_curve, max_drawdown

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["figure.dpi"] = 110


def equal_weight_benchmark(prices: pd.DataFrame) -> pd.Series:
    rets = forward_returns(prices)
    return rets.mean(axis=1)


def information_coefficient(score: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    """Spearman rank IC between this month's score and next month's return."""
    rets = forward_returns(prices)
    ic = {}
    for dt in score.index:
        s = score.loc[dt].dropna()
        r = rets.loc[dt].reindex(s.index).dropna()
        common = s.index.intersection(r.index)
        if len(common) < 10:
            continue
        ic[dt] = s.loc[common].rank().corr(r.loc[common].rank())
    return pd.Series(ic)


def main():
    prices = download_prices()
    fundamentals = download_fundamental_snapshot()
    fundamentals = fundamentals.dropna(subset=["sector", "marketCap"])

    raw_factors = build_raw_factors(prices, fundamentals)

    sector = fundamentals["sector"]
    market_cap = fundamentals["marketCap"]

    neutral_factors = {}
    for name, factor in raw_factors.items():
        w = winsorize(factor)
        neutral_factors[name] = neutralize_panel(w, sector, market_cap)

    composite = composite_score(neutral_factors)
    composite = composite.dropna(how="all").iloc[12:]  # need 12m history for momentum

    bt = run_backtest(composite, prices, n_deciles=5, cost_bps=15.0, max_turnover=0.60)
    returns = bt["returns"].dropna()

    benchmark = equal_weight_benchmark(prices).reindex(returns.index)
    table = summary_table(returns["net_return"], returns["gross_return"],
                           returns["turnover"], benchmark)
    table.to_csv(os.path.join(OUT_DIR, "performance_summary.csv"))
    print(table.round(4))

    ic = information_coefficient(composite, prices)
    ic.to_csv(os.path.join(OUT_DIR, "monthly_ic.csv"))
    print(f"\nMean IC: {ic.mean():.4f}  |  IC IR: {ic.mean() / ic.std():.4f}")

    # ---- Figure 1: cumulative gross vs net vs benchmark ----
    fig, ax = plt.subplots(figsize=(9, 5))
    cumulative_curve(returns["gross_return"]).plot(ax=ax, label="Long-Short (Gross)")
    cumulative_curve(returns["net_return"]).plot(ax=ax, label="Long-Short (Net of Costs)")
    cumulative_curve(benchmark).plot(ax=ax, label="Equal-Weight Universe", linestyle="--")
    ax.set_title("Cross-Sectional Factor Portfolio: Cumulative Growth of $1")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "01_cumulative_returns.png"))
    plt.close(fig)

    # ---- Figure 2: drawdown ----
    fig, ax = plt.subplots(figsize=(9, 4))
    cum = cumulative_curve(returns["net_return"])
    dd = cum / cum.cummax() - 1.0
    ax.fill_between(dd.index, dd.values, 0, color="firebrick", alpha=0.5)
    ax.set_title(f"Drawdown, Net of Costs (Max = {max_drawdown(returns['net_return']):.1%})")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "02_drawdown.png"))
    plt.close(fig)

    # ---- Figure 3: rolling 12m IC ----
    fig, ax = plt.subplots(figsize=(9, 4))
    ic.rolling(12).mean().plot(ax=ax, color="darkgreen")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Rolling 12-Month Mean Information Coefficient (Composite Score)")
    ax.set_ylabel("Spearman IC")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "03_rolling_ic.png"))
    plt.close(fig)

    # ---- Figure 4: turnover ----
    fig, ax = plt.subplots(figsize=(9, 4))
    returns["turnover"].plot(ax=ax, color="slategray")
    ax.set_title("Monthly One-Way Portfolio Turnover")
    ax.set_ylabel("Turnover")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "04_turnover.png"))
    plt.close(fig)

    # ---- Figure 5: factor correlation heatmap (post-neutralization) ----
    z_neutral = pd.concat({k: zscore(v).mean(axis=1) for k, v in neutral_factors.items()}, axis=1)
    corr = pd.concat({k: v.stack() for k, v in neutral_factors.items()}, axis=1).corr()
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Neutralized Factor Correlation")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "05_factor_correlation.png"))
    plt.close(fig)

    print("\nSaved figures to", FIG_DIR)


if __name__ == "__main__":
    main()
