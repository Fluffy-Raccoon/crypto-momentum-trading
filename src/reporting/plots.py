"""Chart generation for backtest results."""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plot_equity_curve(
    equity_curve: pd.DataFrame,
    benchmark_curve: pd.DataFrame | None = None,
    output_path: str | Path = "equity_curve.png",
) -> Path:
    """Plot equity curve (log scale) with optional benchmark.

    Args:
        equity_curve: DataFrame with 'timestamp' and 'equity'.
        benchmark_curve: Optional benchmark equity curve.
        output_path: Where to save the PNG.

    Returns:
        Path to saved file.
    """
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(equity_curve["timestamp"], equity_curve["equity"], label="Strategy", linewidth=1.5)

    if benchmark_curve is not None and len(benchmark_curve) > 0:
        ax.plot(
            benchmark_curve["timestamp"],
            benchmark_curve["equity"],
            label="BTC Buy & Hold",
            linewidth=1.0,
            alpha=0.7,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity ($, log scale)")
    ax.set_title("Equity Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved equity curve to {output_path}")
    return output_path


def plot_drawdown(
    equity_curve: pd.DataFrame,
    output_path: str | Path = "drawdown.png",
) -> Path:
    """Plot underwater drawdown chart.

    Args:
        equity_curve: DataFrame with 'equity' column.
        output_path: Where to save the PNG.

    Returns:
        Path to saved file.
    """
    output_path = Path(output_path)
    equity = equity_curve["equity"].values
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(
        equity_curve["timestamp"], drawdown, 0,
        color="red", alpha=0.3, label="Drawdown",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Underwater Drawdown")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved drawdown chart to {output_path}")
    return output_path


def plot_monthly_returns_heatmap(
    equity_curve: pd.DataFrame,
    output_path: str | Path = "monthly_returns.png",
) -> Path:
    """Plot monthly returns heatmap (year x month grid).

    Args:
        equity_curve: DataFrame with 'timestamp' and 'equity'.
        output_path: Where to save the PNG.

    Returns:
        Path to saved file.
    """
    output_path = Path(output_path)

    df = equity_curve.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    monthly = df["equity"].resample("ME").last()
    monthly_returns = monthly.pct_change().dropna()

    if len(monthly_returns) == 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "Insufficient data for monthly heatmap", ha="center", va="center")
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    pivot = pd.DataFrame({
        "year": monthly_returns.index.year,
        "month": monthly_returns.index.month,
        "return": monthly_returns.values,
    })
    table = pivot.pivot_table(index="year", columns="month", values="return", aggfunc="first")

    fig, ax = plt.subplots(figsize=(12, max(4, len(table) * 0.5 + 1)))
    im = ax.imshow(table.values, cmap="RdYlGn", aspect="auto", vmin=-0.2, vmax=0.2)

    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels(table.index)
    ax.set_title("Monthly Returns Heatmap")

    # Add text annotations
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            val = table.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1%}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved monthly heatmap to {output_path}")
    return output_path


def plot_rolling_sharpe(
    equity_curve: pd.DataFrame,
    window: int = 90,
    output_path: str | Path = "rolling_sharpe.png",
) -> Path:
    """Plot rolling 90-day Sharpe ratio.

    Args:
        equity_curve: DataFrame with 'equity' column.
        window: Rolling window in days.
        output_path: Where to save the PNG.

    Returns:
        Path to saved file.
    """
    output_path = Path(output_path)

    daily_returns = equity_curve["equity"].pct_change().dropna()
    rolling_mean = daily_returns.rolling(window).mean()
    rolling_std = daily_returns.rolling(window).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(365)

    fig, ax = plt.subplots(figsize=(12, 4))
    timestamps = equity_curve["timestamp"].iloc[1:]  # skip first (NaN from pct_change)
    if len(timestamps) == len(rolling_sharpe):
        ax.plot(timestamps, rolling_sharpe, linewidth=1)
    else:
        ax.plot(rolling_sharpe.values, linewidth=1)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title(f"Rolling {window}-Day Sharpe Ratio")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved rolling Sharpe to {output_path}")
    return output_path


def plot_position_concentration(
    positions_over_time: pd.DataFrame,
    output_path: str | Path = "position_concentration.png",
) -> Path:
    """Plot number of positions over time.

    Args:
        positions_over_time: DataFrame with 'timestamp' and 'symbol'.
        output_path: Where to save the PNG.

    Returns:
        Path to saved file.
    """
    output_path = Path(output_path)

    if len(positions_over_time) == 0:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, "No position data", ha="center", va="center")
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    count_by_date = positions_over_time.groupby("timestamp")["symbol"].nunique()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(count_by_date.index, count_by_date.values, alpha=0.5)
    ax.plot(count_by_date.index, count_by_date.values, linewidth=1)
    ax.set_xlabel("Date")
    ax.set_ylabel("# Positions")
    ax.set_title("Position Concentration Over Time")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved position concentration to {output_path}")
    return output_path
