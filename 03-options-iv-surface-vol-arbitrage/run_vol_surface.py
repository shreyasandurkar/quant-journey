"""Main entry point for the IV Surface & Volatility Arbitrage project."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data import download_option_chain, download_underlying_history, download_vix_history, TICKER
from src.surface import compute_iv_surface, fit_smile_splines, check_butterfly_arbitrage, check_calendar_arbitrage
from src.vol_arb_backtest import gamma_pnl_backtest, realized_vol
from src.metrics import sharpe_ratio, annualized_return, max_drawdown

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams["figure.dpi"] = 110


def main():
    # ---- Part A: IV surface construction + arbitrage checks ----
    chain = download_option_chain(TICKER)
    surface = compute_iv_surface(chain)
    surface.to_csv(os.path.join(OUT_DIR, "iv_surface_points.csv"), index=False)

    butterfly_violations = check_butterfly_arbitrage(surface)
    calendar_violations = check_calendar_arbitrage(surface)
    butterfly_violations.to_csv(os.path.join(OUT_DIR, "butterfly_violations.csv"), index=False)
    calendar_violations.to_csv(os.path.join(OUT_DIR, "calendar_violations.csv"), index=False)
    print(f"Butterfly arbitrage violations found: {len(butterfly_violations)}")
    print(f"Calendar arbitrage violations found: {len(calendar_violations)}")

    # ---- Figure 1: smile per expiry ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    splines = fit_smile_splines(surface)
    for exp, grp in surface.groupby("expiry"):
        grp = grp.sort_values("moneyness")
        ax.scatter(grp["moneyness"], grp["iv"] * 100, s=10, alpha=0.5)
        if exp in splines:
            xs = np.linspace(grp["moneyness"].min(), grp["moneyness"].max(), 100)
            ax.plot(xs, splines[exp](xs) * 100, label=pd.Timestamp(exp).date().isoformat())
    ax.set_xlabel("Log-Moneyness  ln(K/S)")
    ax.set_ylabel("Implied Volatility (%)")
    ax.set_title(f"{TICKER} Volatility Smiles by Expiry (Newton-Raphson IV)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "01_vol_smiles.png")); plt.close(fig)

    # ---- Figure 2: 3D-style surface (scatter, moneyness x T x IV) ----
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(surface["moneyness"], surface["T"], surface["iv"] * 100,
                     c=surface["iv"] * 100, cmap="viridis", s=12)
    ax.set_xlabel("Log-Moneyness")
    ax.set_ylabel("Time to Maturity (yrs)")
    ax.set_zlabel("IV (%)")
    ax.set_title(f"{TICKER} Implied Volatility Surface")
    fig.colorbar(sc, ax=ax, shrink=0.6, label="IV (%)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "02_iv_surface_3d.png")); plt.close(fig)

    # ---- Part B: delta-hedged short-vol backtest (VRP capture) ----
    spy = download_underlying_history(TICKER)
    vix = download_vix_history()
    bt = gamma_pnl_backtest(spy, vix, tenor_days=21)
    bt["cum_pnl"] = bt["daily_pnl"].cumsum()
    bt.to_csv(os.path.join(OUT_DIR, "vrp_backtest.csv"))

    rv = realized_vol(spy, window=21).reindex(bt.index)
    sharpe = sharpe_ratio(bt["daily_return"])
    mdd = max_drawdown(bt["cum_pnl"])
    print(f"\nShort-vol delta-hedged strategy (idealized gamma-PnL model): "
          f"Sharpe={sharpe:.2f}, Cumulative P&L (per $1 gamma notional)="
          f"{bt['cum_pnl'].iloc[-1]:.2f}, Max DD={mdd:.2f}")
    print("NOTE: this Sharpe is a theoretical upper bound -- see README "
          "Section 5 for why real-world short-vol Sharpe is much lower.")

    vrp = (vix.reindex(bt.index) / 100.0) - rv
    print(f"Mean VIX - Realized Vol (VRP) = {vrp.mean():.4f} "
          f"({(vrp > 0).mean():.1%} of days VIX > realized)")

    # ---- Figure 3: VIX vs realized vol ----
    fig, ax = plt.subplots(figsize=(9, 5))
    (vix.reindex(bt.index) / 100.0 * 100).plot(ax=ax, label="VIX (Implied)")
    (rv * 100).plot(ax=ax, label="21-Day Realized Vol", alpha=0.8)
    ax.set_ylabel("Annualized Vol (%)")
    ax.set_title(f"{TICKER}/VIX: Implied vs Realized Volatility (Variance Risk Premium)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "03_vrp_implied_vs_realized.png")); plt.close(fig)

    # ---- Figure 4: cumulative gamma P&L of short-vol book ----
    fig, ax = plt.subplots(figsize=(9, 5))
    bt["cum_pnl"].plot(ax=ax, color="darkgreen")
    ax.set_title("Cumulative Gamma P&L: Delta-Hedged Short ATM Straddle, Rolled Monthly")
    ax.set_ylabel("Cumulative P&L (per $1 notional$^2$)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "04_short_vol_cumulative_pnl.png")); plt.close(fig)

    # ---- Figure 5: rolling 6m Sharpe of the short-vol book ----
    fig, ax = plt.subplots(figsize=(9, 4))
    roll_sharpe = bt["daily_return"].rolling(126).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan)
    roll_sharpe.plot(ax=ax, color="purple")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Rolling 6-Month Sharpe Ratio of the Short-Vol Book")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "05_rolling_sharpe.png")); plt.close(fig)

    print("\nSaved figures to", FIG_DIR)


if __name__ == "__main__":
    main()
