"""Tests for performance metrics computation."""

import numpy as np
import pandas as pd
import pytest

from src.reporting.metrics import (
    compute_avg_holding_period,
    compute_avg_win_loss_ratio,
    compute_cagr,
    compute_exposure,
    compute_max_drawdown,
    compute_metrics,
    compute_profit_factor,
    compute_sharpe,
    compute_sortino,
    compute_win_rate,
)


@pytest.fixture
def linear_equity():
    """Linear growth equity curve — predictable metrics."""
    n = 365
    dates = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    equity = np.linspace(1000, 2000, n)  # 100% return over 1 year
    return pd.DataFrame({
        "timestamp": dates,
        "equity": equity,
        "num_positions": np.ones(n),
    })


@pytest.fixture
def flat_equity():
    """Flat equity curve — zero returns."""
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({
        "timestamp": dates,
        "equity": np.full(n, 1000.0),
        "num_positions": np.zeros(n),
    })


@pytest.fixture
def drawdown_equity():
    """Equity curve with a single large drawdown."""
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    # Rise to 1200, then drop to 900, then recover to 1100
    equity = np.concatenate([
        np.linspace(1000, 1200, 30),
        np.linspace(1200, 900, 20),
        np.linspace(900, 1100, 50),
    ])
    return pd.DataFrame({
        "timestamp": dates,
        "equity": equity,
        "num_positions": np.ones(n),
    })


@pytest.fixture
def sample_trade_log():
    """Sample trade log with wins and losses."""
    return pd.DataFrame({
        "entry_date": pd.to_datetime(["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]),
        "exit_date": pd.to_datetime(["2023-01-10", "2023-01-25", "2023-02-10", "2023-02-20"]),
        "symbol": ["BTC", "ETH", "SOL", "ADA"],
        "pnl": [100.0, -50.0, 200.0, -30.0],
    })


class TestMetrics:
    """Tests for individual metric functions."""

    def test_cagr_linear(self, linear_equity):
        """Linear 100% growth over 1 year -> CAGR ~ 100%."""
        cagr = compute_cagr(linear_equity)
        assert 0.9 < cagr < 1.1  # ~100%

    def test_cagr_flat(self, flat_equity):
        """Flat equity -> CAGR = 0."""
        cagr = compute_cagr(flat_equity)
        assert cagr == pytest.approx(0.0, abs=0.001)

    def test_cagr_empty(self):
        """Empty equity curve -> CAGR = 0."""
        df = pd.DataFrame(columns=["timestamp", "equity"])
        assert compute_cagr(df) == 0.0

    def test_sharpe_flat(self, flat_equity):
        """Flat equity -> Sharpe = 0."""
        sharpe = compute_sharpe(flat_equity)
        assert sharpe == 0.0

    def test_sharpe_positive_for_growth(self, linear_equity):
        """Growing equity -> positive Sharpe."""
        sharpe = compute_sharpe(linear_equity)
        assert sharpe > 0

    def test_sortino_flat(self, flat_equity):
        """Flat equity -> Sortino = 0."""
        sortino = compute_sortino(flat_equity)
        assert sortino == 0.0

    def test_max_drawdown_flat(self, flat_equity):
        """Flat equity -> max DD = 0."""
        dd_pct, dd_days = compute_max_drawdown(flat_equity)
        assert dd_pct == 0.0
        assert dd_days == 0

    def test_max_drawdown_known(self, drawdown_equity):
        """Known drawdown: peak 1200, trough 900 -> DD = -25%."""
        dd_pct, dd_days = compute_max_drawdown(drawdown_equity)
        assert dd_pct == pytest.approx(-0.25, abs=0.02)
        assert dd_days > 0

    def test_profit_factor(self, sample_trade_log):
        """Profit factor = gross wins / gross losses."""
        pf = compute_profit_factor(sample_trade_log)
        # Wins: 100 + 200 = 300, Losses: 50 + 30 = 80
        assert pf == pytest.approx(300 / 80)

    def test_profit_factor_no_losses(self):
        """All wins -> profit factor = inf."""
        log = pd.DataFrame({"pnl": [100.0, 50.0]})
        assert compute_profit_factor(log) == float("inf")

    def test_profit_factor_empty(self):
        """Empty trade log -> profit factor = 0."""
        log = pd.DataFrame(columns=["pnl"])
        assert compute_profit_factor(log) == 0.0

    def test_win_rate(self, sample_trade_log):
        """Win rate = 2 wins / 4 trades = 50%."""
        wr = compute_win_rate(sample_trade_log)
        assert wr == pytest.approx(0.5)

    def test_win_rate_empty(self):
        """Empty trade log -> win rate = 0."""
        log = pd.DataFrame(columns=["pnl"])
        assert compute_win_rate(log) == 0.0

    def test_avg_win_loss_ratio(self, sample_trade_log):
        """Avg win/loss = (150) / (40) = 3.75."""
        awl = compute_avg_win_loss_ratio(sample_trade_log)
        # Avg win = (100+200)/2 = 150, avg loss = (50+30)/2 = 40
        assert awl == pytest.approx(150 / 40)

    def test_avg_holding_period(self, sample_trade_log):
        """Average holding period should be ~9 days."""
        ahp = compute_avg_holding_period(sample_trade_log)
        # (9, 10, 9, 5) -> avg ~8.25
        assert 5 < ahp < 12

    def test_exposure(self, linear_equity):
        """All positions open -> exposure = 100%."""
        exp = compute_exposure(linear_equity)
        assert exp == pytest.approx(1.0)

    def test_exposure_no_positions(self, flat_equity):
        """No positions -> exposure = 0%."""
        exp = compute_exposure(flat_equity)
        assert exp == pytest.approx(0.0)

    def test_compute_metrics_full(self, linear_equity, sample_trade_log):
        """compute_metrics should return a complete PerformanceMetrics."""
        metrics = compute_metrics(linear_equity, sample_trade_log)
        assert metrics.cagr > 0
        assert metrics.total_trades == 4
        assert metrics.win_rate == pytest.approx(0.5)
        d = metrics.to_dict()
        assert "CAGR" in d
        assert "Sharpe Ratio" in d
