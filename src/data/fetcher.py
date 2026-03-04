"""Binance OHLCV data fetcher with local Parquet caching."""

import logging
import time
from pathlib import Path

import ccxt
import pandas as pd

from src.contracts import REQUIRED_OHLCV_COLS, validate_ohlcv

logger = logging.getLogger(__name__)


class BinanceFetcher:
    """Fetches daily OHLCV data from Binance with incremental Parquet caching."""

    def __init__(self, config: dict, exchange: ccxt.Exchange | None = None) -> None:
        """Initialize fetcher from config dict.

        Args:
            config: Must contain 'data' key with exchange, base_currency,
                    timeframe, cache_dir settings.
            exchange: Optional ccxt exchange instance (for testing/DI).
        """
        data_cfg = config["data"]
        self._timeframe = data_cfg.get("timeframe", "1d")
        self._cache_dir = Path(data_cfg.get("cache_dir", "src/data/cache"))
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._exchange = exchange or ccxt.binance({"enableRateLimit": True})
        self._max_retries = 3
        self._base_delay = 1.0

    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: str,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a single symbol, using cache when available.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT").
            start_date: Start date as "YYYY-MM-DD".
            end_date: End date as "YYYY-MM-DD", or None for today.

        Returns:
            DataFrame conforming to OHLCV contract.
        """
        cached = self._load_cache(symbol)
        start_ts = pd.Timestamp(start_date, tz="UTC")
        end_ts = pd.Timestamp(end_date, tz="UTC") if end_date else pd.Timestamp.now(tz="UTC")

        if cached is not None and len(cached) > 0:
            last_cached = cached["timestamp"].max()
            if last_cached >= end_ts:
                logger.info(f"Cache hit for {symbol}, no fetch needed")
                mask = (cached["timestamp"] >= start_ts) & (cached["timestamp"] <= end_ts)
                return cached.loc[mask].reset_index(drop=True)
            # Fetch only missing data
            fetch_since = int((last_cached + pd.Timedelta(days=1)).timestamp() * 1000)
        else:
            cached = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
            fetch_since = int(start_ts.timestamp() * 1000)

        new_data = self._fetch_all_pages(symbol, fetch_since)
        if new_data:
            new_df = self._raw_to_dataframe(new_data, symbol)
            combined = pd.concat([cached, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
            combined = combined.reset_index(drop=True)
            self._save_cache(symbol, combined)
        else:
            combined = cached

        if combined.empty:
            combined["symbol"] = pd.Series(dtype="str")
            for col in REQUIRED_OHLCV_COLS:
                if col not in combined.columns:
                    combined[col] = pd.Series(dtype="float64")
            return combined

        if "symbol" not in combined.columns:
            combined["symbol"] = symbol

        mask = (combined["timestamp"] >= start_ts) & (combined["timestamp"] <= end_ts)
        result = combined.loc[mask].reset_index(drop=True)
        validate_ohlcv(result)
        return result

    def fetch_multiple(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV for multiple symbols.

        Args:
            symbols: List of trading pairs.
            start_date: Start date as "YYYY-MM-DD".
            end_date: End date or None.

        Returns:
            Dict mapping symbol to OHLCV DataFrame.
        """
        result = {}
        for sym in symbols:
            try:
                result[sym] = self.fetch_ohlcv(sym, start_date, end_date)
            except Exception:
                logger.warning(f"Failed to fetch {sym}, skipping", exc_info=True)
        return result

    def _fetch_all_pages(self, symbol: str, since: int) -> list:
        """Fetch all OHLCV pages from since timestamp until now."""
        all_data: list = []
        current_since = since

        while True:
            batch = self._fetch_from_api(symbol, current_since)
            if not batch:
                break
            all_data.extend(batch)
            last_ts = batch[-1][0]
            if last_ts == current_since:
                break
            current_since = last_ts + 1
            time.sleep(0.1)  # Rate limit courtesy

        return all_data

    def _fetch_from_api(self, symbol: str, since: int, limit: int = 1000) -> list:
        """Fetch a single page of OHLCV data with retry logic.

        Args:
            symbol: Trading pair.
            since: Timestamp in milliseconds.
            limit: Max rows per request.

        Returns:
            List of [timestamp, open, high, low, close, volume] lists.
        """
        for attempt in range(self._max_retries):
            try:
                data = self._exchange.fetch_ohlcv(
                    symbol, self._timeframe, since=since, limit=limit
                )
                return data
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
                delay = self._base_delay * (2**attempt)
                logger.warning(
                    f"Attempt {attempt + 1}/{self._max_retries} failed for {symbol}: {e}. "
                    f"Retrying in {delay}s"
                )
                if attempt < self._max_retries - 1:
                    time.sleep(delay)
                else:
                    raise
        return []  # unreachable, but satisfies type checker

    def _raw_to_dataframe(self, raw: list, symbol: str) -> pd.DataFrame:
        """Convert raw ccxt OHLCV list to DataFrame."""
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["symbol"] = symbol
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype("float64")
        return df

    def _load_cache(self, symbol: str) -> pd.DataFrame | None:
        """Load cached Parquet file for a symbol."""
        path = self._cache_path(symbol)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df
        except Exception:
            logger.warning(f"Failed to read cache for {symbol}", exc_info=True)
            return None

    def _save_cache(self, symbol: str, df: pd.DataFrame) -> None:
        """Save DataFrame to Parquet cache."""
        path = self._cache_path(symbol)
        df.to_parquet(path, index=False)
        logger.info(f"Cached {len(df)} rows for {symbol} at {path}")

    def _cache_path(self, symbol: str) -> Path:
        """Get the cache file path for a symbol."""
        safe_name = symbol.replace("/", "_")
        return self._cache_dir / f"{safe_name}.parquet"
