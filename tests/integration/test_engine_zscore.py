"""Integration test: full backtest with Z-score signal."""

import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.signals.momentum_zscore import MomentumZScore


@pytest.fixture
def backtest_config():
    """Config with small windows for fast integration tests."""
    return {
        "data": {
            "top_n_coins": 5,
            "exclude": [],
        },
        "signals": {
            "momentum_zscore": {
                "lookback_days": 5,
                "entry_threshold": 1.0,
                "exit_threshold": 0.0,
                "zscore_window": 30,
            },
        },
        "portfolio": {
            "initial_capital": 10000.0,
            "max_positions": 3,
            "risk_per_position_pct": 1.5,
            "vol_lookback_days": 14,
        },
        "costs": {
            "commission_pct": 0.10,
            "slippage_pct": 0.02,
        },
        "backtest": {
            "walk_forward": {
                "formation_window_days": 60,
                "roll_step_days": 30,
            },
            "benchmark": "BTC",
        },
    }


class TestEngineZScore:
    """Integration tests for BacktestEngine with Z-score signal."""

    def test_full_backtest_runs(self, backtest_config, multi_coin_ohlcv):
        """Full backtest should complete without errors."""
        engine = BacktestEngine(backtest_config)
        signal = MomentumZScore(
            lookback_days=5, zscore_window=30,
            entry_threshold=1.0, exit_threshold=0.0,
        )
        result = engine.run(signal, multi_coin_ohlcv)

        assert result is not None
        assert result.signal_name == "momentum_zscore_5"

    def test_equity_curve_no_nans(self, backtest_config, multi_coin_ohlcv):
        """Equity curve should have no NaN values."""
        engine = BacktestEngine(backtest_config)
        signal = MomentumZScore(
            lookback_days=5, zscore_window=30,
            entry_threshold=1.0, exit_threshold=0.0,
        )
        result = engine.run(signal, multi_coin_ohlcv)

        assert result.equity_curve["equity"].notna().all()

    def test_starting_equity(self, backtest_config, multi_coin_ohlcv):
        """Starting equity should be close to initial capital."""
        engine = BacktestEngine(backtest_config)
        signal = MomentumZScore(
            lookback_days=5, zscore_window=30,
            entry_threshold=1.0, exit_threshold=0.0,
        )
        result = engine.run(signal, multi_coin_ohlcv)

        initial = backtest_config["portfolio"]["initial_capital"]
        first_equity = result.equity_curve.iloc[0]["equity"]
        assert abs(first_equity - initial) / initial < 0.1

    def test_window_no_overlap(self, backtest_config):
        """Walk-forward windows should have no lookahead bias."""
        engine = BacktestEngine(backtest_config)
        start = pd.Timestamp("2020-01-01", tz="UTC")
        end = pd.Timestamp("2020-12-31", tz="UTC")

        windows = engine.generate_windows(start, end)
        for w in windows:
            assert w.formation_end < w.trading_start

    def test_daily_returns_series(self, backtest_config, multi_coin_ohlcv):
        """Daily returns should be a valid Series."""
        engine = BacktestEngine(backtest_config)
        signal = MomentumZScore(
            lookback_days=5, zscore_window=30,
            entry_threshold=1.0, exit_threshold=0.0,
        )
        result = engine.run(signal, multi_coin_ohlcv)

        assert len(result.daily_returns) > 0
        assert result.daily_returns.notna().all()
