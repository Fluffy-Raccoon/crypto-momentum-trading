"""Shared test fixtures for the crypto momentum backtester."""

import numpy as np
import pandas as pd
import pytest
import yaml


@pytest.fixture
def config():
    """Load config.yaml with test-friendly overrides (smaller windows for speed)."""
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    # Override for faster tests
    cfg["backtest"]["walk_forward"]["formation_window_days"] = 60
    cfg["backtest"]["walk_forward"]["roll_step_days"] = 15
    cfg["signals"]["momentum_zscore"]["zscore_window"] = 30
    cfg["portfolio"]["vol_lookback_days"] = 14
    return cfg


@pytest.fixture
def sample_ohlcv():
    """A deterministic 500-row OHLCV DataFrame using a seeded random walk."""
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")

    # Random walk for close prices starting at 100
    returns = rng.normal(0.0005, 0.03, size=n)
    close = 100.0 * np.cumprod(1 + returns)

    # Derive OHLV from close
    high = close * (1 + rng.uniform(0, 0.03, size=n))
    low = close * (1 - rng.uniform(0, 0.03, size=n))
    open_ = close * (1 + rng.normal(0, 0.01, size=n))
    volume = rng.uniform(1e6, 1e8, size=n)

    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "symbol": "BTC/USDT",
        }
    )


@pytest.fixture
def trending_ohlcv():
    """Synthetic uptrend OHLCV (for signal entry tests)."""
    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")

    # Steady uptrend: 0.5% daily growth
    close = 100.0 * np.cumprod(np.ones(n) * 1.005)
    high = close * 1.01
    low = close * 0.99
    open_ = close * 0.999
    volume = np.full(n, 5e7)

    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "symbol": "ETH/USDT",
        }
    )


@pytest.fixture
def mean_reverting_ohlcv():
    """Synthetic sine wave OHLCV (for signal exit tests)."""
    n = 365
    dates = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")

    t = np.arange(n)
    close = 100.0 + 20.0 * np.sin(2 * np.pi * t / 90)  # 90-day cycle
    high = close * 1.01
    low = close * 0.99
    open_ = close * 1.001
    volume = np.full(n, 3e7)

    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "symbol": "SOL/USDT",
        }
    )


@pytest.fixture
def flat_ohlcv():
    """Constant-price OHLCV (edge case testing)."""
    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")

    close = np.full(n, 100.0)
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": close.copy(),
            "high": close.copy(),
            "low": close.copy(),
            "close": close.copy(),
            "volume": np.full(n, 1e7),
            "symbol": "FLAT/USDT",
        }
    )


@pytest.fixture
def multi_coin_ohlcv():
    """Synthetic OHLCV data for 5 coins, 365 days (for integration tests)."""
    rng = np.random.default_rng(123)
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "DOT/USDT"]
    n = 365
    dates = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    result = {}

    for i, sym in enumerate(symbols):
        drift = 0.0003 * (i + 1)  # different drift per coin
        returns = rng.normal(drift, 0.04, size=n)
        close = (50.0 + i * 20) * np.cumprod(1 + returns)
        high = close * (1 + rng.uniform(0, 0.02, size=n))
        low = close * (1 - rng.uniform(0, 0.02, size=n))
        open_ = close * (1 + rng.normal(0, 0.005, size=n))
        volume = rng.uniform(1e6, 1e8, size=n) * (5 - i)  # BTC has highest volume

        result[sym] = pd.DataFrame(
            {
                "timestamp": dates,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "symbol": sym,
            }
        )

    return result
