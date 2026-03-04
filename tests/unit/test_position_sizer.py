"""Tests for volatility-scaled position sizing."""

import math

import numpy as np
import pandas as pd
import pytest

from src.portfolio.position_sizer import MIN_ANNUALIZED_VOL, VolatilityPositionSizer


@pytest.fixture
def sizer_config():
    """Standard config for position sizer tests."""
    return {
        "portfolio": {
            "risk_per_position_pct": 1.5,
            "max_positions": 5,
            "vol_lookback_days": 30,
        }
    }


@pytest.fixture
def known_vol_prices():
    """Prices with known daily volatility for hand calculation."""
    n = 50
    dates = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    # Constant daily returns of +/- 2% alternating -> known std
    returns = np.array([0.02, -0.02] * 25)
    close = 100.0 * np.cumprod(1 + returns)
    return pd.DataFrame({
        "timestamp": dates,
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.full(n, 1e7),
        "symbol": "TEST/USDT",
    })


class TestVolatilityPositionSizer:
    """Tests for VolatilityPositionSizer."""

    def test_known_volatility_calculation(self, sizer_config, known_vol_prices):
        """Verify position size matches hand calculation for known volatility."""
        sizer = VolatilityPositionSizer(sizer_config)
        equity = 5000.0

        ann_vol = sizer.compute_annualized_vol(known_vol_prices)
        expected_daily_std = known_vol_prices["close"].pct_change().dropna().tail(30).std()
        expected_ann_vol = expected_daily_std * math.sqrt(365)

        assert abs(ann_vol - expected_ann_vol) < 0.001
        size = sizer.compute_position_size(equity, known_vol_prices)
        expected_size = (equity * 0.015) / expected_ann_vol
        max_size = equity / 5
        expected_size = min(expected_size, max_size)
        assert abs(size - expected_size) < 0.01

    def test_clamp_max_position(self, sizer_config):
        """Position should not exceed equity / max_positions."""
        sizer = VolatilityPositionSizer(sizer_config)
        equity = 5000.0

        # Very low volatility -> large raw size, should be clamped
        n = 50
        close = np.linspace(100, 100.001, n)  # almost flat -> near-zero vol
        prices = pd.DataFrame({"close": close})

        size = sizer.compute_position_size(equity, prices)
        max_allowed = equity / 5  # max_positions = 5
        assert size <= max_allowed + 0.01

    def test_zero_volatility_graceful(self, sizer_config, flat_ohlcv):
        """Near-zero volatility should not cause division by zero."""
        sizer = VolatilityPositionSizer(sizer_config)
        equity = 5000.0

        # Should not raise
        size = sizer.compute_position_size(equity, flat_ohlcv)
        assert size > 0
        assert math.isfinite(size)

    def test_insufficient_data(self, sizer_config):
        """Single row of data should return size based on MIN_ANNUALIZED_VOL."""
        sizer = VolatilityPositionSizer(sizer_config)
        prices = pd.DataFrame({"close": [100.0]})

        vol = sizer.compute_annualized_vol(prices)
        assert vol == MIN_ANNUALIZED_VOL

    def test_compute_units(self, sizer_config, known_vol_prices):
        """compute_units should return correct fractional units."""
        sizer = VolatilityPositionSizer(sizer_config)
        equity = 5000.0
        current_price = 100.0

        units = sizer.compute_units(equity, known_vol_prices, current_price)
        size_usd = sizer.compute_position_size(equity, known_vol_prices)
        expected_units = size_usd / current_price

        assert abs(units - expected_units) < 0.001

    def test_compute_units_zero_price(self, sizer_config, known_vol_prices):
        """Zero price should return 0 units."""
        sizer = VolatilityPositionSizer(sizer_config)
        units = sizer.compute_units(5000.0, known_vol_prices, 0.0)
        assert units == 0.0

    def test_positive_size_for_positive_inputs(self, sizer_config, sample_ohlcv):
        """For any positive equity and prices, size should be positive."""
        sizer = VolatilityPositionSizer(sizer_config)
        size = sizer.compute_position_size(5000.0, sample_ohlcv)
        assert size > 0
