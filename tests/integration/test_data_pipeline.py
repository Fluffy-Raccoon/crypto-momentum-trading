"""Integration test: data fetch -> cache -> re-fetch pipeline."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.fetcher import BinanceFetcher


@pytest.fixture
def pipeline_config(tmp_path):
    """Config for data pipeline tests."""
    return {
        "data": {
            "exchange": "binance",
            "base_currency": "USDT",
            "timeframe": "1d",
            "cache_dir": str(tmp_path / "cache"),
        }
    }


@pytest.fixture
def mock_ohlcv_data():
    """Standard mock OHLCV data."""
    base_ts = int(pd.Timestamp("2023-01-01", tz="UTC").timestamp() * 1000)
    day_ms = 86400 * 1000
    return [
        [base_ts + i * day_ms, 100.0 + i, 105.0 + i, 95.0 + i, 102.0 + i, 1e6]
        for i in range(30)
    ]


class TestDataPipeline:
    """Integration tests for the data fetch -> cache pipeline."""

    def test_fetch_cache_refetch(self, pipeline_config, mock_ohlcv_data):
        """Fetch -> cache -> re-fetch should hit cache (no second API call)."""
        mock_exchange = MagicMock()
        fetcher = BinanceFetcher(pipeline_config, exchange=mock_exchange)

        with patch.object(fetcher, "_fetch_all_pages", return_value=mock_ohlcv_data) as mock_pages:
            # First fetch
            df1 = fetcher.fetch_ohlcv("BTC/USDT", "2023-01-01", "2023-01-30")
            first_calls = mock_pages.call_count

            # Second fetch — should hit cache
            df2 = fetcher.fetch_ohlcv("BTC/USDT", "2023-01-01", "2023-01-30")
            assert mock_pages.call_count == first_calls, "Second fetch should use cache"

        assert len(df1) == len(df2)
        assert df1["timestamp"].tolist() == df2["timestamp"].tolist()

    def test_parquet_schema(self, pipeline_config, mock_ohlcv_data):
        """Cached parquet should have correct columns and dtypes."""
        mock_exchange = MagicMock()
        fetcher = BinanceFetcher(pipeline_config, exchange=mock_exchange)

        with patch.object(fetcher, "_fetch_all_pages", return_value=mock_ohlcv_data):
            fetcher.fetch_ohlcv("BTC/USDT", "2023-01-01", "2023-01-30")

        # Read parquet directly
        cache_path = fetcher._cache_path("BTC/USDT")
        df = pd.read_parquet(cache_path)

        required = ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

        assert df["timestamp"].duplicated().sum() == 0, "No duplicate timestamps"

    def test_multiple_symbols(self, pipeline_config, mock_ohlcv_data):
        """fetch_multiple should cache each symbol independently."""
        mock_exchange = MagicMock()
        fetcher = BinanceFetcher(pipeline_config, exchange=mock_exchange)

        with patch.object(fetcher, "_fetch_all_pages", return_value=mock_ohlcv_data):
            result = fetcher.fetch_multiple(
                ["BTC/USDT", "ETH/USDT"], "2023-01-01", "2023-01-30"
            )

        assert "BTC/USDT" in result
        assert "ETH/USDT" in result

        # Both should have cache files
        assert fetcher._cache_path("BTC/USDT").exists()
        assert fetcher._cache_path("ETH/USDT").exists()
