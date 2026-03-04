"""Tests for the coin universe selection logic."""

import numpy as np
import pandas as pd
import pytest

from src.data.universe import CoinUniverse


@pytest.fixture
def universe_config():
    """Standard config for universe tests."""
    return {
        "data": {
            "top_n_coins": 3,
            "exclude": ["USDT", "USDC", "BUSD"],
        }
    }


@pytest.fixture
def volume_ohlcv_data():
    """Synthetic OHLCV data with known volume rankings.

    BTC: highest volume, ETH: second, SOL: third, ADA: fourth, USDC: stablecoin.
    """
    n = 60
    dates = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    base_close = np.linspace(100, 110, n)

    def make_df(symbol: str, volume_base: float) -> pd.DataFrame:
        return pd.DataFrame({
            "timestamp": dates,
            "open": base_close * 0.99,
            "high": base_close * 1.02,
            "low": base_close * 0.98,
            "close": base_close,
            "volume": np.full(n, volume_base),
            "symbol": symbol,
        })

    return {
        "BTC/USDT": make_df("BTC/USDT", 1e9),
        "ETH/USDT": make_df("ETH/USDT", 5e8),
        "SOL/USDT": make_df("SOL/USDT", 2e8),
        "ADA/USDT": make_df("ADA/USDT", 1e8),
        "USDC/USDT": make_df("USDC/USDT", 3e9),  # Should be excluded
    }


class TestCoinUniverse:
    """Tests for CoinUniverse."""

    def test_correct_top_n_selection(self, universe_config, volume_ohlcv_data):
        """Should return top-N symbols by volume."""
        universe = CoinUniverse(universe_config)
        result = universe.get_universe(
            volume_ohlcv_data,
            pd.Timestamp("2023-02-28", tz="UTC"),
        )
        assert result == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    def test_stablecoins_excluded(self, universe_config, volume_ohlcv_data):
        """Stablecoins from the exclude list should never appear."""
        universe = CoinUniverse(universe_config)
        result = universe.get_universe(
            volume_ohlcv_data,
            pd.Timestamp("2023-02-28", tz="UTC"),
        )
        for sym in result:
            base = sym.split("/")[0]
            assert base not in universe_config["data"]["exclude"]

    def test_no_lookahead(self, universe_config):
        """Universe determined at date T should not use data from after T."""
        n = 60
        dates = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
        close = np.full(n, 100.0)

        # ADA has low volume in first 30 days, very high after
        ada_vol = np.concatenate([np.full(30, 1e6), np.full(30, 1e12)])
        sol_vol = np.full(n, 5e8)

        data = {
            "ADA/USDT": pd.DataFrame({
                "timestamp": dates, "open": close, "high": close,
                "low": close, "close": close, "volume": ada_vol, "symbol": "ADA/USDT",
            }),
            "SOL/USDT": pd.DataFrame({
                "timestamp": dates, "open": close, "high": close,
                "low": close, "close": close, "volume": sol_vol, "symbol": "SOL/USDT",
            }),
        }

        config = {"data": {"top_n_coins": 1, "exclude": []}}
        universe = CoinUniverse(config)

        # At day 30, ADA still has low volume
        result_early = universe.get_universe(data, pd.Timestamp("2023-01-30", tz="UTC"))
        assert result_early == ["SOL/USDT"]

        # At day 60, ADA now dominates
        result_late = universe.get_universe(data, pd.Timestamp("2023-03-01", tz="UTC"))
        assert result_late == ["ADA/USDT"]

    def test_insufficient_data_excluded(self, universe_config):
        """Symbols with fewer than vol_window days of data should be excluded."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D", tz="UTC")
        close = np.full(10, 100.0)

        data = {
            "NEW/USDT": pd.DataFrame({
                "timestamp": dates, "open": close, "high": close,
                "low": close, "close": close, "volume": np.full(10, 1e12),
                "symbol": "NEW/USDT",
            }),
        }

        universe = CoinUniverse(universe_config)
        result = universe.get_universe(data, pd.Timestamp("2023-01-10", tz="UTC"))
        assert result == []

    def test_get_all_candidate_symbols(self, universe_config):
        """Should return candidate list with stablecoins filtered."""
        universe = CoinUniverse(universe_config)
        candidates = universe.get_all_candidate_symbols()
        assert len(candidates) > 0
        for sym in candidates:
            base = sym.split("/")[0]
            assert base not in universe_config["data"]["exclude"]

    def test_empty_data(self, universe_config):
        """Empty input should return empty universe."""
        universe = CoinUniverse(universe_config)
        result = universe.get_universe({}, pd.Timestamp("2023-01-01", tz="UTC"))
        assert result == []

    def test_fewer_coins_than_top_n(self, universe_config, volume_ohlcv_data):
        """If fewer valid coins than top_n, return all valid ones."""
        config = {**universe_config, "data": {**universe_config["data"], "top_n_coins": 100}}
        universe = CoinUniverse(config)
        result = universe.get_universe(
            volume_ohlcv_data,
            pd.Timestamp("2023-02-28", tz="UTC"),
        )
        # Should get all non-stablecoin coins (BTC, ETH, SOL, ADA = 4)
        assert len(result) == 4

    def test_volume_ranking_order(self, universe_config, volume_ohlcv_data):
        """Result should be ordered by volume descending."""
        config = {**universe_config, "data": {**universe_config["data"], "top_n_coins": 10}}
        universe = CoinUniverse(config)
        result = universe.get_universe(
            volume_ohlcv_data,
            pd.Timestamp("2023-02-28", tz="UTC"),
        )
        # BTC (1e9) > ETH (5e8) > SOL (2e8) > ADA (1e8)
        assert result == ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"]
