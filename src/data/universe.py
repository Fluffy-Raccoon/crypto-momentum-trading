"""Coin universe selection logic — top-N by trailing volume."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Broad set of historically top-50 coins (excluding stablecoins).
# Used to determine which symbols to pre-fetch data for.
CANDIDATE_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT",
    "SOL/USDT", "DOT/USDT", "DOGE/USDT", "AVAX/USDT", "MATIC/USDT",
    "LINK/USDT", "LTC/USDT", "UNI/USDT", "ATOM/USDT", "ETC/USDT",
    "XLM/USDT", "ALGO/USDT", "FIL/USDT", "TRX/USDT", "NEAR/USDT",
    "ICP/USDT", "AAVE/USDT", "GRT/USDT", "FTM/USDT", "SAND/USDT",
    "MANA/USDT", "AXS/USDT", "THETA/USDT", "EOS/USDT", "XTZ/USDT",
    "RUNE/USDT", "CRV/USDT", "SUSHI/USDT", "1INCH/USDT", "ENJ/USDT",
    "COMP/USDT", "SNX/USDT", "MKR/USDT", "YFI/USDT", "ZEC/USDT",
    "DASH/USDT", "NEO/USDT", "WAVES/USDT", "BAT/USDT", "ZIL/USDT",
    "LUNA/USDT", "FTT/USDT", "VET/USDT", "EGLD/USDT", "HBAR/USDT",
]


class CoinUniverse:
    """Determines the top-N coin universe at each rebalance date.

    Uses trailing 30-day average daily volume as a proxy for market cap ranking.
    Only uses data available as of the given date (no lookahead).
    """

    def __init__(self, config: dict) -> None:
        """Initialize from config.

        Args:
            config: Must contain 'data' key with top_n_coins and exclude list.
        """
        data_cfg = config["data"]
        self._top_n = data_cfg.get("top_n_coins", 20)
        self._exclude = set(data_cfg.get("exclude", []))
        self._vol_window = 30  # trailing days for volume ranking

    def get_universe(
        self,
        ohlcv_data: dict[str, pd.DataFrame],
        as_of_date: pd.Timestamp,
    ) -> list[str]:
        """Return top-N symbols by trailing volume as of the given date.

        Args:
            ohlcv_data: Dict mapping symbol to OHLCV DataFrame.
            as_of_date: Universe is determined using only data up to this date.

        Returns:
            List of symbol strings, sorted by volume descending.
        """
        volume_scores: dict[str, float] = {}

        for symbol, df in ohlcv_data.items():
            # Check stablecoin exclusion (match against base currency name)
            base = symbol.split("/")[0] if "/" in symbol else symbol
            if base in self._exclude:
                continue

            # Only use data available at as_of_date (no lookahead)
            mask = df["timestamp"] <= as_of_date
            available = df.loc[mask]

            if len(available) < self._vol_window:
                continue

            # Trailing 30-day average volume
            trailing = available.tail(self._vol_window)
            avg_volume = trailing["volume"].mean()
            volume_scores[symbol] = avg_volume

        # Sort by volume descending, take top N
        ranked = sorted(volume_scores.items(), key=lambda x: x[1], reverse=True)
        universe = [sym for sym, _ in ranked[: self._top_n]]

        logger.info(
            f"Universe as of {as_of_date.date()}: {len(universe)} coins "
            f"(from {len(volume_scores)} candidates)"
        )
        return universe

    def get_all_candidate_symbols(self) -> list[str]:
        """Return the broad list of symbols to pre-fetch data for.

        Returns:
            List of historically top-50 coin trading pairs.
        """
        return [s for s in CANDIDATE_SYMBOLS if s.split("/")[0] not in self._exclude]
