"""Property-based tests using hypothesis for key invariants."""

from datetime import timedelta

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from src.backtest.costs import TransactionCostModel
from src.portfolio.position_sizer import VolatilityPositionSizer
from src.signals.ema_crossover import EMACrossover
from src.signals.momentum_zscore import MomentumZScore

# --- Strategies ---

@st.composite
def price_series(draw, min_length=50, max_length=200):
    """Generate a random price series as a DataFrame."""
    n = draw(st.integers(min_value=min_length, max_value=max_length))
    start_price = draw(st.floats(min_value=10.0, max_value=1000.0))
    returns = draw(
        st.lists(
            st.floats(min_value=-0.05, max_value=0.05),
            min_size=n,
            max_size=n,
        )
    )
    close = start_price * np.cumprod(1 + np.array(returns))
    close = np.maximum(close, 0.01)  # prevent negative prices

    dates = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({
        "timestamp": dates,
        "open": close * 0.999,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.full(n, 1e7),
        "symbol": "TEST/USDT",
    })


# --- Signal boundedness ---

class TestSignalBoundedness:
    """Signal outputs must always be in {0, 1}."""

    @given(prices=price_series(min_length=50, max_length=150))
    @settings(max_examples=30, deadline=timedelta(seconds=10))
    def test_ema_signal_bounded(self, prices):
        """EMA crossover signal values must be in {0, 1}."""
        signal = EMACrossover(fast_period=5, slow_period=10)
        result = signal.generate(prices)
        assert set(result.unique()).issubset({0, 1}), f"Got values: {result.unique()}"

    @given(prices=price_series(min_length=100, max_length=150))
    @settings(max_examples=30, deadline=timedelta(seconds=10))
    def test_zscore_signal_bounded(self, prices):
        """Momentum Z-score signal values must be in {0, 1}."""
        signal = MomentumZScore(
            lookback_days=5, zscore_window=20,
            entry_threshold=1.0, exit_threshold=0.0,
        )
        result = signal.generate(prices)
        assert set(result.unique()).issubset({0, 1}), f"Got values: {result.unique()}"

    @given(prices=price_series(min_length=50, max_length=150))
    @settings(max_examples=30, deadline=timedelta(seconds=10))
    def test_ema_output_length(self, prices):
        """EMA signal length must match input length."""
        signal = EMACrossover(fast_period=5, slow_period=10)
        result = signal.generate(prices)
        assert len(result) == len(prices)

    @given(prices=price_series(min_length=100, max_length=150))
    @settings(max_examples=30, deadline=timedelta(seconds=10))
    def test_zscore_output_length(self, prices):
        """Z-score signal length must match input length."""
        signal = MomentumZScore(lookback_days=5, zscore_window=20)
        result = signal.generate(prices)
        assert len(result) == len(prices)


# --- Position size positivity ---

class TestPositionSizePositivity:
    """Position sizes must be positive for valid inputs."""

    @given(
        equity=st.floats(min_value=100.0, max_value=1e6),
        prices=price_series(min_length=50, max_length=100),
    )
    @settings(max_examples=30, deadline=timedelta(seconds=10))
    def test_positive_size(self, equity, prices):
        """For positive equity and prices, position size > 0."""
        config = {
            "portfolio": {
                "risk_per_position_pct": 1.5,
                "max_positions": 5,
                "vol_lookback_days": 14,
            }
        }
        sizer = VolatilityPositionSizer(config)
        size = sizer.compute_position_size(equity, prices)
        assert size > 0, f"Size should be positive, got {size}"
        assert np.isfinite(size), f"Size should be finite, got {size}"


# --- Cost non-negativity ---

class TestCostNonNegativity:
    """Transaction costs must never be negative."""

    @given(trade_value=st.floats(min_value=-1e6, max_value=1e6))
    @settings(max_examples=50, deadline=timedelta(seconds=5))
    def test_cost_non_negative(self, trade_value):
        """Cost must be >= 0 for any trade value."""
        config = {"costs": {"commission_pct": 0.10, "slippage_pct": 0.02}}
        model = TransactionCostModel(config)
        cost = model.compute_cost(trade_value)
        assert cost >= 0, f"Cost should be non-negative, got {cost}"


# --- Max drawdown monotonicity ---

class TestDrawdownMonotonicity:
    """Max drawdown of a subseries <= max drawdown of the full series."""

    @given(prices=price_series(min_length=50, max_length=200))
    @settings(max_examples=20, deadline=timedelta(seconds=10))
    def test_drawdown_monotonic(self, prices):
        """Max drawdown of subseries <= max drawdown of full series."""
        from src.reporting.metrics import compute_max_drawdown

        equity_full = pd.DataFrame({
            "timestamp": prices["timestamp"],
            "equity": prices["close"],
        })
        dd_full, _ = compute_max_drawdown(equity_full)

        # Take the first half as subseries
        mid = len(prices) // 2
        equity_sub = equity_full.iloc[:mid].copy()
        dd_sub, _ = compute_max_drawdown(equity_sub)

        # Subseries drawdown should be >= full drawdown (both are negative or zero)
        # i.e., the full series can only have a worse (more negative) drawdown
        assert dd_sub >= dd_full or abs(dd_sub - dd_full) < 1e-10, (
            f"Subseries DD {dd_sub} > full DD {dd_full}"
        )
