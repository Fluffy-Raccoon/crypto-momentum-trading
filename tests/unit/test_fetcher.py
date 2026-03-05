"""Tests for the Binance OHLCV data fetcher."""

from unittest.mock import MagicMock, patch

import ccxt
import pandas as pd
import pytest

from src.contracts import REQUIRED_OHLCV_COLS
from src.data.fetcher import BinanceFetcher


@pytest.fixture
def fetcher_config(tmp_path):
    """Config pointing to a temporary cache directory."""
    return {
        "data": {
            "exchange": "binance",
            "base_currency": "USDT",
            "timeframe": "1d",
            "cache_dir": str(tmp_path / "cache"),
        }
    }


@pytest.fixture
def mock_exchange():
    """A mock ccxt exchange that never hits the network."""
    return MagicMock(spec=ccxt.binance)


@pytest.fixture
def mock_raw_ohlcv():
    """Raw OHLCV data as ccxt returns it (list of lists)."""
    base_ts = int(pd.Timestamp("2023-01-01", tz="UTC").timestamp() * 1000)
    day_ms = 86400 * 1000
    return [
        [base_ts + i * day_ms, 100.0 + i, 105.0 + i, 95.0 + i, 102.0 + i, 1e6 + i * 1000]
        for i in range(10)
    ]


def make_fetcher(config, exchange):
    """Create a BinanceFetcher with a mock exchange injected."""
    return BinanceFetcher(config, exchange=exchange)


class TestBinanceFetcher:
    """Tests for BinanceFetcher."""

    def test_fetch_creates_cache_file(self, fetcher_config, mock_exchange, mock_raw_ohlcv):
        """First fetch should write a Parquet cache file."""
        fetcher = make_fetcher(fetcher_config, mock_exchange)
        # Patch _fetch_all_pages to avoid the pagination loop
        with patch.object(fetcher, "_fetch_all_pages", return_value=mock_raw_ohlcv):
            df = fetcher.fetch_ohlcv("BTC/USDT", "2023-01-01", "2023-01-10")

        cache_path = fetcher._cache_path("BTC/USDT")
        assert cache_path.exists()
        assert len(df) > 0

    def test_cache_hit_no_api_call(self, fetcher_config, mock_exchange, mock_raw_ohlcv):
        """Second fetch should use cache without calling the API."""
        fetcher = make_fetcher(fetcher_config, mock_exchange)

        with patch.object(fetcher, "_fetch_all_pages", return_value=mock_raw_ohlcv) as mock_pages:
            # First fetch populates cache
            fetcher.fetch_ohlcv("BTC/USDT", "2023-01-01", "2023-01-10")
            call_count_after_first = mock_pages.call_count

            # Second fetch should hit cache (end_date is within cached range)
            df2 = fetcher.fetch_ohlcv("BTC/USDT", "2023-01-01", "2023-01-10")
            assert mock_pages.call_count == call_count_after_first
            assert len(df2) > 0

    def test_output_has_correct_columns(self, fetcher_config, mock_exchange, mock_raw_ohlcv):
        """Output DataFrame must have all required OHLCV columns."""
        fetcher = make_fetcher(fetcher_config, mock_exchange)
        with patch.object(fetcher, "_fetch_all_pages", return_value=mock_raw_ohlcv):
            df = fetcher.fetch_ohlcv("BTC/USDT", "2023-01-01", "2023-01-10")

        for col in REQUIRED_OHLCV_COLS:
            assert col in df.columns, f"Missing column: {col}"

    def test_output_dtypes(self, fetcher_config, mock_exchange, mock_raw_ohlcv):
        """Prices should be float64, timestamp should be datetime with UTC."""
        fetcher = make_fetcher(fetcher_config, mock_exchange)
        with patch.object(fetcher, "_fetch_all_pages", return_value=mock_raw_ohlcv):
            df = fetcher.fetch_ohlcv("BTC/USDT", "2023-01-01", "2023-01-10")

        for col in ["open", "high", "low", "close", "volume"]:
            assert df[col].dtype == "float64", f"{col} should be float64"
        assert df["timestamp"].dt.tz is not None, "timestamp should be tz-aware"

    def test_empty_api_response(self, fetcher_config, mock_exchange):
        """Empty API response should return empty DataFrame without error."""
        fetcher = make_fetcher(fetcher_config, mock_exchange)
        with patch.object(fetcher, "_fetch_all_pages", return_value=[]):
            df = fetcher.fetch_ohlcv("FAKE/USDT", "2023-01-01", "2023-01-10")

        assert len(df) == 0

    def test_retry_on_network_error(self, fetcher_config, mock_exchange):
        """Should retry on NetworkError with exponential backoff."""
        fetcher = make_fetcher(fetcher_config, mock_exchange)
        mock_exchange.fetch_ohlcv.side_effect = [
            ccxt.NetworkError("timeout"),
            ccxt.NetworkError("timeout"),
            [[int(pd.Timestamp("2023-01-01", tz="UTC").timestamp() * 1000),
              100.0, 105.0, 95.0, 102.0, 1e6]],
        ]

        with patch("src.data.fetcher.time.sleep"):
            result = fetcher._fetch_from_api("BTC/USDT", 0)

        assert len(result) == 1
        assert mock_exchange.fetch_ohlcv.call_count == 3

    def test_retry_exhausted_raises(self, fetcher_config, mock_exchange):
        """Should raise after exhausting all retries."""
        fetcher = make_fetcher(fetcher_config, mock_exchange)
        mock_exchange.fetch_ohlcv.side_effect = ccxt.NetworkError("timeout")

        with patch("src.data.fetcher.time.sleep"):
            with pytest.raises(ccxt.NetworkError):
                fetcher._fetch_from_api("BTC/USDT", 0)

    def test_fetch_multiple_skips_failures(self, fetcher_config, mock_exchange, mock_raw_ohlcv):
        """fetch_multiple should skip symbols that fail and continue."""
        fetcher = make_fetcher(fetcher_config, mock_exchange)

        def pages_side_effect(symbol, since):
            if "FAIL" in symbol:
                raise ccxt.ExchangeError("not found")
            return mock_raw_ohlcv

        with patch.object(fetcher, "_fetch_all_pages", side_effect=pages_side_effect):
            result = fetcher.fetch_multiple(
                ["BTC/USDT", "FAIL/USDT", "ETH/USDT"],
                "2023-01-01",
                "2023-01-10",
            )

        assert "BTC/USDT" in result
        assert "ETH/USDT" in result
        assert "FAIL/USDT" not in result

    def test_incremental_fetch(self, fetcher_config, mock_exchange):
        """After caching 5 days, fetching beyond cache triggers new fetch."""
        day_ms = 86400 * 1000
        base_ts = int(pd.Timestamp("2023-01-01", tz="UTC").timestamp() * 1000)

        first_batch = [
            [base_ts + i * day_ms, 100.0, 105.0, 95.0, 102.0, 1e6]
            for i in range(5)
        ]
        second_batch = [
            [base_ts + i * day_ms, 100.0, 105.0, 95.0, 102.0, 1e6]
            for i in range(5, 10)
        ]

        fetcher = make_fetcher(fetcher_config, mock_exchange)
        call_count = [0]

        def pages_side_effect(symbol, since):
            call_count[0] += 1
            if call_count[0] == 1:
                return first_batch
            return second_batch

        with patch.object(fetcher, "_fetch_all_pages", side_effect=pages_side_effect):
            # First fetch: gets days 0-4
            fetcher.fetch_ohlcv("BTC/USDT", "2023-01-01", "2023-01-05")
            assert call_count[0] == 1

            # Second fetch with later end_date: triggers incremental fetch
            fetcher.fetch_ohlcv("BTC/USDT", "2023-01-01", "2023-01-12")
            assert call_count[0] == 2

    def test_symbol_name_in_output(self, fetcher_config, mock_exchange, mock_raw_ohlcv):
        """Symbol column should match the requested symbol."""
        fetcher = make_fetcher(fetcher_config, mock_exchange)
        with patch.object(fetcher, "_fetch_all_pages", return_value=mock_raw_ohlcv):
            df = fetcher.fetch_ohlcv("ETH/USDT", "2023-01-01", "2023-01-10")

        assert (df["symbol"] == "ETH/USDT").all()

    def test_no_duplicate_timestamps(self, fetcher_config, mock_exchange, mock_raw_ohlcv):
        """Fetched data should have no duplicate timestamps."""
        fetcher = make_fetcher(fetcher_config, mock_exchange)
        with patch.object(fetcher, "_fetch_all_pages", return_value=mock_raw_ohlcv):
            df = fetcher.fetch_ohlcv("BTC/USDT", "2023-01-01", "2023-01-10")

        assert df["timestamp"].is_unique

    def test_fetch_all_pages_pagination(self, fetcher_config, mock_exchange):
        """_fetch_all_pages should paginate until empty response."""
        fetcher = make_fetcher(fetcher_config, mock_exchange)
        base_ts = int(pd.Timestamp("2023-01-01", tz="UTC").timestamp() * 1000)
        day_ms = 86400 * 1000

        page1 = [[base_ts + i * day_ms, 100.0, 105.0, 95.0, 102.0, 1e6] for i in range(3)]
        page2 = [[base_ts + (i + 3) * day_ms, 100.0, 105.0, 95.0, 102.0, 1e6] for i in range(3)]

        mock_exchange.fetch_ohlcv.side_effect = [page1, page2, []]

        with patch("src.data.fetcher.time.sleep"):
            result = fetcher._fetch_all_pages("BTC/USDT", base_ts)

        assert len(result) == 6

    def test_raw_to_dataframe(self, fetcher_config, mock_exchange, mock_raw_ohlcv):
        """_raw_to_dataframe should produce proper DataFrame from raw data."""
        fetcher = make_fetcher(fetcher_config, mock_exchange)
        df = fetcher._raw_to_dataframe(mock_raw_ohlcv, "BTC/USDT")

        assert len(df) == 10
        assert "symbol" in df.columns
        assert (df["symbol"] == "BTC/USDT").all()
        assert df["timestamp"].dt.tz is not None

    # --- Timeframe-aware cache path tests ---

    def test_cache_path_includes_timeframe(self, fetcher_config, mock_exchange):
        """Cache file path should include the timeframe to prevent collisions."""
        fetcher = make_fetcher(fetcher_config, mock_exchange)
        path = fetcher._cache_path("BTC/USDT")
        assert "_1d.parquet" in str(path)

    def test_cache_path_different_timeframes(self, tmp_path, mock_exchange):
        """Different timeframes should produce different cache paths."""
        config_1d = {
            "data": {"timeframe": "1d", "cache_dir": str(tmp_path / "cache")},
        }
        config_4h = {
            "data": {"timeframe": "4h", "cache_dir": str(tmp_path / "cache")},
        }
        fetcher_1d = make_fetcher(config_1d, mock_exchange)
        fetcher_4h = make_fetcher(config_4h, mock_exchange)

        path_1d = fetcher_1d._cache_path("BTC/USDT")
        path_4h = fetcher_4h._cache_path("BTC/USDT")

        assert path_1d != path_4h
        assert "BTC_USDT_1d.parquet" in str(path_1d)
        assert "BTC_USDT_4h.parquet" in str(path_4h)

    def test_cache_path_symbol_sanitized(self, fetcher_config, mock_exchange):
        """Slash in symbol name should be replaced with underscore."""
        fetcher = make_fetcher(fetcher_config, mock_exchange)
        path = fetcher._cache_path("ETH/USDT")
        assert "/" not in path.name
        assert "ETH_USDT" in path.name
