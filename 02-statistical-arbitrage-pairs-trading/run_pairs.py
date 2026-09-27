"""Main entry point: screens candidate pairs for cointegration, fits an OU
process + Kalman-filtered dynamic hedge ratio, runs the trading engine, and
saves figures + results for the Statistical Arbitrage project.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

from src.data import download_prices, CANDIDATE_PAIRS
from src.cointegration import screen_pair, fit_ou_process
from src.kalman import kalman_dynamic_beta
from src.backtest import rolling_ols_beta, rolling_regime_filter, zscore_signal, generate_positions, backtest_pair
from src.metrics import summary_table, cumulative_curve, max_drawdown

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams["figure.dpi"] = 110


def main():
    prices = download_prices()

    # ---- Step 1: screen all candidate pairs for cointegration ----
    screen_rows = {}
    for y_t, x_t in CANDIDATE_PAIRS:
        if y_t not in prices.columns or x_t not in prices.columns:
            continue
        screen_rows[f"{y_t}-{x_t}"] = screen_pair(prices[y_t], prices[x_t])
    screen_df = pd.DataFrame(screen_rows).T.sort_values("adf_pvalue")
    screen_df.to_csv(os.path.join(OUT_DIR, "cointegration_screen.csv"))
    print("Cointegration screen (sorted by ADF p-value):\n", screen_df.round(4))

    best_pair_name = screen_df.index[0]
    y_tkr, x_tkr = best_pair_name.split("-")
    print(f"\nSelected pair: {y_tkr} / {x_tkr}")

    y, x = prices[y_tkr], prices[x_tkr]

    # ---- Step 2: OU process fit on the static-beta spread ----
    static_beta = screen_df.loc[best_pair_name, "beta"]
    static_spread = y - static_beta * x
    ou = fit_ou_process(static_spread)
    print("OU fit:", ou)

    # ---- Step 3: dynamic beta via Kalman filter vs rolling OLS ----
    kf = kalman_dynamic_beta(y, x)
    roll_beta = rolling_ols_beta(y, x, window=60)

    z = zscore_signal(kf["spread"], window=20)
    regime_ok = rolling_regime_filter(kf["spread"], window=120, p_thresh=0.10)
    positions = generate_positions(z, regime_ok, entry=2.0, exit=0.5, stop=4.0)

    result_kf = backtest_pair(y, x, kf["beta"], positions, cost_bps=5.0)
    result_roll = backtest_pair(y, x, roll_beta, positions, cost_bps=5.0)

    table = summary_table(result_kf["net_return"], result_kf["gross_return"])
    table.to_csv(os.path.join(OUT_DIR, "performance_summary_kalman.csv"))
    print("\nKalman-beta strategy performance:\n", table.round(4))

    table_roll = summary_table(result_roll["net_return"], result_roll["gross_return"])
    table_roll.to_csv(os.path.join(OUT_DIR, "performance_summary_rolling_ols.csv"))
    print("\nRolling-OLS-beta strategy performance:\n", table_roll.round(4))

    # ---- Figure 1: normalized price series ----
    fig, ax = plt.subplots(figsize=(9, 5))
    (y / y.iloc[0]).plot(ax=ax, label=y_tkr)
    (x / x.iloc[0]).plot(ax=ax, label=x_tkr)
    ax.set_title(f"Normalized Prices: {y_tkr} vs {x_tkr}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "01_normalized_prices.png")); plt.close(fig)

    # ---- Figure 2: dynamic beta, Kalman vs rolling OLS ----
    fig, ax = plt.subplots(figsize=(9, 4.5))
    kf["beta"].plot(ax=ax, label="Kalman Filter Beta")
    roll_beta.plot(ax=ax, label="60-Day Rolling OLS Beta", linestyle="--")
    ax.axhline(static_beta, color="black", linewidth=0.8, label="Full-Sample Static Beta")
    ax.set_title(f"Dynamic Hedge Ratio: {y_tkr} ~ beta * {x_tkr}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "02_dynamic_beta.png")); plt.close(fig)

    # ---- Figure 3: spread + z-score with entry/exit bands and trades ----
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    kf["spread"].plot(ax=axes[0], color="steelblue")
    axes[0].set_title(f"Kalman-Filtered Spread: {y_tkr} - beta*{x_tkr}")
    axes[0].grid(alpha=0.3)

    z.plot(ax=axes[1], color="darkorange", label="Z-score")
    for level, style in [(2.0, "--"), (-2.0, "--"), (0.5, ":"), (-0.5, ":"), (4.0, "-"), (-4.0, "-")]:
        axes[1].axhline(level, color="gray", linestyle=style, linewidth=0.8)
    axes[1].fill_between(z.index, -100, 100, where=~regime_ok.values, color="red",
                          alpha=0.08, transform=axes[1].get_xaxis_transform())
    axes[1].set_ylim(-6, 6)
    axes[1].set_title("Z-score (entry=±2, exit=±0.5, stop=±4); red = regime filter off")
    axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "03_spread_zscore.png")); plt.close(fig)

    # ---- Figure 4: cumulative returns, Kalman vs rolling-OLS vs buy&hold spread ----
    fig, ax = plt.subplots(figsize=(9, 5))
    cumulative_curve(result_kf["net_return"]).plot(ax=ax, label="Kalman Beta (Net)")
    cumulative_curve(result_roll["net_return"]).plot(ax=ax, label="Rolling OLS Beta (Net)")
    ax.set_title(f"Pairs Trade Cumulative Growth of $1: {y_tkr}/{x_tkr}")
    ax.set_ylabel("Growth of $1")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "04_cumulative_returns.png")); plt.close(fig)

    # ---- Figure 5: drawdown ----
    fig, ax = plt.subplots(figsize=(9, 4))
    cum = cumulative_curve(result_kf["net_return"])
    dd = cum / cum.cummax() - 1.0
    ax.fill_between(dd.index, dd.values, 0, color="firebrick", alpha=0.5)
    ax.set_title(f"Drawdown, Kalman Strategy (Max = {max_drawdown(result_kf['net_return']):.1%})")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "05_drawdown.png")); plt.close(fig)

    print("\nSaved figures to", FIG_DIR)


if __name__ == "__main__":
    main()
