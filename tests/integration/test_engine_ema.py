"""Integration test: full backtest with EMA crossover signal."""

import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.signals.ema_crossover import EMACrossover


@pytest.fixture
def backtest_config():
    """Config with small windows for fast integration tests."""
    return {
        "data": {
            "top_n_coins": 5,
            "exclude": [],
        },
        "signals": {
            "ema_crossover": {"fast_period": 5, "slow_period": 10},
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


class TestEngineEMA:
    """Integration tests for BacktestEngine with EMA crossover."""

    def test_full_backtest_runs(self, backtest_config, multi_coin_ohlcv):
        """Full backtest should complete without errors."""
        engine = BacktestEngine(backtest_config)
        signal = EMACrossover(fast_period=5, slow_period=10)
        result = engine.run(signal, multi_coin_ohlcv)

        assert result is not None
        assert result.signal_name == "ema_crossover_5_10"

    def test_equity_curve_length(self, backtest_config, multi_coin_ohlcv):
        """Equity curve should have entries for trading days."""
        engine = BacktestEngine(backtest_config)
        signal = EMACrossover(fast_period=5, slow_period=10)
        result = engine.run(signal, multi_coin_ohlcv)

        assert len(result.equity_curve) > 0
        assert "equity" in result.equity_curve.columns
        assert "timestamp" in result.equity_curve.columns

    def test_no_nan_in_equity(self, backtest_config, multi_coin_ohlcv):
        """Equity curve should have no NaN values."""
        engine = BacktestEngine(backtest_config)
        signal = EMACrossover(fast_period=5, slow_period=10)
        result = engine.run(signal, multi_coin_ohlcv)

        assert result.equity_curve["equity"].notna().all()

    def test_starting_equity(self, backtest_config, multi_coin_ohlcv):
        """First equity entry should be close to initial capital."""
        engine = BacktestEngine(backtest_config)
        signal = EMACrossover(fast_period=5, slow_period=10)
        result = engine.run(signal, multi_coin_ohlcv)

        initial = backtest_config["portfolio"]["initial_capital"]
        first_equity = result.equity_curve.iloc[0]["equity"]
        # Allow some deviation from entry costs on day 0
        assert abs(first_equity - initial) / initial < 0.1

    def test_trade_log_populated(self, backtest_config, multi_coin_ohlcv):
        """Trade log should contain trades."""
        engine = BacktestEngine(backtest_config)
        signal = EMACrossover(fast_period=5, slow_period=10)
        result = engine.run(signal, multi_coin_ohlcv)

        assert len(result.trade_log) > 0
        required_cols = ["entry_date", "exit_date", "symbol", "pnl"]
        for col in required_cols:
            assert col in result.trade_log.columns

    def test_pnl_reconciliation(self, backtest_config, multi_coin_ohlcv):
        """Final equity should approximately equal initial + sum(pnl) - sum(costs)."""
        engine = BacktestEngine(backtest_config)
        signal = EMACrossover(fast_period=5, slow_period=10)
        result = engine.run(signal, multi_coin_ohlcv)

        initial = backtest_config["portfolio"]["initial_capital"]
        if len(result.trade_log) > 0:
            total_pnl = result.trade_log["pnl"].sum()
            final_equity = result.equity_curve.iloc[-1]["equity"]
            # Allow 5% tolerance for mark-to-market vs realized P&L differences
            assert abs(final_equity - (initial + total_pnl)) / initial < 0.05

    def test_window_generation(self, backtest_config):
        """Walk-forward windows should not overlap."""
        engine = BacktestEngine(backtest_config)
        start = pd.Timestamp("2020-01-01", tz="UTC")
        end = pd.Timestamp("2020-12-31", tz="UTC")

        windows = engine.generate_windows(start, end)
        assert len(windows) > 0

        for w in windows:
            assert w.formation_end < w.trading_start, "Formation must end before trading starts"

    def test_max_positions_respected(self, backtest_config, multi_coin_ohlcv):
        """Should never exceed max_positions."""
        engine = BacktestEngine(backtest_config)
        signal = EMACrossover(fast_period=5, slow_period=10)
        result = engine.run(signal, multi_coin_ohlcv)

        max_pos = backtest_config["portfolio"]["max_positions"]
        equity_curve = result.equity_curve
        assert (equity_curve["num_positions"] <= max_pos).all()
