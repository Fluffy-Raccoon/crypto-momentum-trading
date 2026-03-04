"""Tests for chart generation."""

import numpy as np
import pandas as pd
import pytest

from src.reporting import plots


@pytest.fixture
def equity_curve():
    """Sample equity curve for plotting."""
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    equity = 1000.0 * np.cumprod(1 + np.random.default_rng(42).normal(0.001, 0.02, n))
    return pd.DataFrame({
        "timestamp": dates,
        "equity": equity,
        "num_positions": np.random.default_rng(42).integers(0, 4, n),
    })


@pytest.fixture
def positions_df():
    """Sample positions over time."""
    dates = pd.date_range("2023-01-01", periods=50, freq="D", tz="UTC")
    records = []
    for d in dates:
        for sym in ["BTC/USDT", "ETH/USDT"]:
            records.append({"timestamp": d, "symbol": sym, "weight": 0.2})
    return pd.DataFrame(records)


class TestPlots:
    """Tests for plot generation functions."""

    def test_equity_curve_png(self, equity_curve, tmp_path):
        """Should generate equity curve PNG."""
        path = plots.plot_equity_curve(equity_curve, output_path=tmp_path / "eq.png")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_equity_curve_with_benchmark(self, equity_curve, tmp_path):
        """Should generate equity curve with benchmark overlay."""
        benchmark = equity_curve.copy()
        benchmark["equity"] = benchmark["equity"] * 0.9
        path = plots.plot_equity_curve(
            equity_curve, benchmark, output_path=tmp_path / "eq_bench.png"
        )
        assert path.exists()

    def test_drawdown_png(self, equity_curve, tmp_path):
        """Should generate drawdown chart."""
        path = plots.plot_drawdown(equity_curve, output_path=tmp_path / "dd.png")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_monthly_returns_heatmap(self, equity_curve, tmp_path):
        """Should generate monthly returns heatmap."""
        path = plots.plot_monthly_returns_heatmap(
            equity_curve, output_path=tmp_path / "monthly.png"
        )
        assert path.exists()
        assert path.stat().st_size > 0

    def test_monthly_returns_heatmap_short_data(self, tmp_path):
        """Short data should produce a chart without error."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D", tz="UTC")
        df = pd.DataFrame({"timestamp": dates, "equity": np.full(10, 100.0)})
        path = plots.plot_monthly_returns_heatmap(df, output_path=tmp_path / "short.png")
        assert path.exists()

    def test_rolling_sharpe_png(self, equity_curve, tmp_path):
        """Should generate rolling Sharpe chart."""
        path = plots.plot_rolling_sharpe(
            equity_curve, window=30, output_path=tmp_path / "sharpe.png"
        )
        assert path.exists()
        assert path.stat().st_size > 0

    def test_position_concentration_png(self, positions_df, tmp_path):
        """Should generate position concentration chart."""
        path = plots.plot_position_concentration(
            positions_df, output_path=tmp_path / "pos.png"
        )
        assert path.exists()
        assert path.stat().st_size > 0

    def test_position_concentration_empty(self, tmp_path):
        """Empty positions should produce a chart without error."""
        empty = pd.DataFrame(columns=["timestamp", "symbol", "weight"])
        path = plots.plot_position_concentration(empty, output_path=tmp_path / "empty.png")
        assert path.exists()
