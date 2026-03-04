"""Tests for HTML tearsheet generation."""

import numpy as np
import pandas as pd
import pytest

from src.contracts import BacktestResult
from src.reporting.tearsheet import generate_tearsheet


@pytest.fixture
def mock_result():
    """Minimal BacktestResult for tearsheet generation."""
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    equity = 1000.0 * np.cumprod(1 + np.random.default_rng(42).normal(0.001, 0.02, n))

    equity_curve = pd.DataFrame({
        "timestamp": dates,
        "equity": equity,
        "num_positions": np.random.default_rng(42).integers(0, 4, n),
    })

    trade_log = pd.DataFrame({
        "entry_date": pd.to_datetime(["2023-01-05", "2023-02-01"]),
        "exit_date": pd.to_datetime(["2023-01-15", "2023-02-10"]),
        "symbol": ["BTC/USDT", "ETH/USDT"],
        "side": ["long", "long"],
        "entry_price": [100.0, 50.0],
        "exit_price": [110.0, 48.0],
        "size": [1.0, 2.0],
        "pnl": [10.0, -4.0],
        "entry_cost": [0.1, 0.05],
        "exit_cost": [0.11, 0.048],
    })

    daily_returns = equity_curve["equity"].pct_change().fillna(0)
    daily_returns.index = equity_curve["timestamp"]

    positions = pd.DataFrame({
        "timestamp": dates[:10],
        "symbol": ["BTC/USDT"] * 10,
        "weight": [0.2] * 10,
    })

    return BacktestResult(
        equity_curve=equity_curve,
        trade_log=trade_log,
        daily_returns=daily_returns,
        positions_over_time=positions,
        signal_name="test_signal",
        config={},
    )


class TestTearsheet:
    """Tests for tearsheet generation."""

    def test_generates_html(self, mock_result, tmp_path):
        """Should generate an HTML tearsheet file."""
        path = generate_tearsheet(mock_result, tmp_path)
        assert path.exists()
        assert path.suffix == ".html"

    def test_html_contains_signal_name(self, mock_result, tmp_path):
        """HTML should contain the signal name."""
        path = generate_tearsheet(mock_result, tmp_path)
        html = path.read_text()
        assert "test_signal" in html

    def test_html_contains_metrics(self, mock_result, tmp_path):
        """HTML should contain performance metrics."""
        path = generate_tearsheet(mock_result, tmp_path)
        html = path.read_text()
        assert "CAGR" in html
        assert "Sharpe" in html
        assert "Max Drawdown" in html

    def test_html_contains_charts(self, mock_result, tmp_path):
        """HTML should embed chart images."""
        path = generate_tearsheet(mock_result, tmp_path)
        html = path.read_text()
        assert "data:image/png;base64" in html

    def test_generates_metrics_json(self, mock_result, tmp_path):
        """Should generate a metrics JSON file alongside."""
        generate_tearsheet(mock_result, tmp_path)
        json_path = tmp_path / "metrics_test_signal.json"
        assert json_path.exists()

    def test_generates_trade_csv(self, mock_result, tmp_path):
        """Should generate a trades CSV file."""
        generate_tearsheet(mock_result, tmp_path)
        csv_path = tmp_path / "trades_test_signal.csv"
        assert csv_path.exists()

    def test_chart_pngs_generated(self, mock_result, tmp_path):
        """Individual chart PNGs should be created."""
        generate_tearsheet(mock_result, tmp_path)
        assert (tmp_path / "equity_curve.png").exists()
        assert (tmp_path / "drawdown.png").exists()
        assert (tmp_path / "monthly_returns.png").exists()
        assert (tmp_path / "rolling_sharpe.png").exists()
        assert (tmp_path / "position_concentration.png").exists()
