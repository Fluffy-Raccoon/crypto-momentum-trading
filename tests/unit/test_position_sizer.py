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

    # --- Timeframe-aware annualization tests ---

    def test_4h_annualization_factor(self, known_vol_prices):
        """4h timeframe should use 6 bars/day * 365 = 2190 bars/year."""
        config_4h = {
            "portfolio": {
                "risk_per_position_pct": 1.5,
                "max_positions": 5,
                "vol_lookback_days": 30,
            },
            "data": {"timeframe": "4h"},
        }
        sizer = VolatilityPositionSizer(config_4h)
        assert sizer._bars_per_year == 6 * 365

    def test_1d_annualization_factor(self, known_vol_prices):
        """1d timeframe should use 1 bar/day * 365 = 365 bars/year."""
        config_1d = {
            "portfolio": {
                "risk_per_position_pct": 1.5,
                "max_positions": 5,
                "vol_lookback_days": 30,
            },
            "data": {"timeframe": "1d"},
        }
        sizer = VolatilityPositionSizer(config_1d)
        assert sizer._bars_per_year == 365

    def test_1h_annualization_factor(self):
        """1h timeframe should use 24 bars/day * 365 = 8760 bars/year."""
        config_1h = {
            "portfolio": {
                "risk_per_position_pct": 1.5,
                "max_positions": 5,
                "vol_lookback_days": 30,
            },
            "data": {"timeframe": "1h"},
        }
        sizer = VolatilityPositionSizer(config_1h)
        assert sizer._bars_per_year == 24 * 365

    def test_default_timeframe_is_1d(self):
        """Missing timeframe config should default to 1d (365 bars/year)."""
        config_no_tf = {
            "portfolio": {
                "risk_per_position_pct": 1.5,
                "max_positions": 5,
                "vol_lookback_days": 30,
            },
        }
        sizer = VolatilityPositionSizer(config_no_tf)
        assert sizer._bars_per_year == 365

    def test_4h_vol_higher_than_1d_same_data(self, known_vol_prices):
        """4h annualized vol should be higher than 1d because sqrt(2190) > sqrt(365)."""
        config_4h = {
            "portfolio": {
                "risk_per_position_pct": 1.5,
                "max_positions": 5,
                "vol_lookback_days": 30,
            },
            "data": {"timeframe": "4h"},
        }
        config_1d = {
            "portfolio": {
                "risk_per_position_pct": 1.5,
                "max_positions": 5,
                "vol_lookback_days": 30,
            },
            "data": {"timeframe": "1d"},
        }
        sizer_4h = VolatilityPositionSizer(config_4h)
        sizer_1d = VolatilityPositionSizer(config_1d)

        vol_4h = sizer_4h.compute_annualized_vol(known_vol_prices)
        vol_1d = sizer_1d.compute_annualized_vol(known_vol_prices)

        # Same bar returns but 4h multiplies by sqrt(2190) vs sqrt(365)
        assert vol_4h > vol_1d
