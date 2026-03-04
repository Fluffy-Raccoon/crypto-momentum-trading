"""Volatility-scaled position sizing."""

import logging
import math

import pandas as pd

logger = logging.getLogger(__name__)

# Minimum annualized volatility to prevent division by near-zero
MIN_ANNUALIZED_VOL = 0.01


class VolatilityPositionSizer:
    """Sizes positions based on realized volatility.

    position_size_usd = (equity * risk_per_position_pct) / annualized_vol
    Clamped so no single position exceeds equity / max_positions.
    """

    def __init__(self, config: dict) -> None:
        """Initialize from config.

        Args:
            config: Must contain 'portfolio' key with risk_per_position_pct,
                    max_positions, vol_lookback_days.
        """
        port_cfg = config["portfolio"]
        self._risk_pct = port_cfg["risk_per_position_pct"] / 100.0
        self._max_positions = port_cfg["max_positions"]
        self._vol_lookback = port_cfg.get("vol_lookback_days", 30)

    def compute_position_size(
        self,
        equity: float,
        prices: pd.DataFrame,
    ) -> float:
        """Compute position size in USD for a single asset.

        Args:
            equity: Current account equity in USD.
            prices: OHLCV DataFrame with 'close' column (needs vol_lookback rows).

        Returns:
            Position size in USD.
        """
        ann_vol = self.compute_annualized_vol(prices)
        raw_size = (equity * self._risk_pct) / ann_vol
        max_size = equity / self._max_positions
        return min(raw_size, max_size)

    def compute_annualized_vol(self, prices: pd.DataFrame) -> float:
        """Compute annualized volatility from trailing daily returns.

        Args:
            prices: OHLCV DataFrame with 'close' column.

        Returns:
            Annualized volatility (daily_std * sqrt(365)).
        """
        close = prices["close"]
        if len(close) < 2:
            return MIN_ANNUALIZED_VOL

        daily_returns = close.pct_change().dropna().tail(self._vol_lookback)
        if len(daily_returns) == 0:
            return MIN_ANNUALIZED_VOL

        daily_std = daily_returns.std()
        if daily_std == 0 or math.isnan(daily_std):
            return MIN_ANNUALIZED_VOL

        ann_vol = daily_std * math.sqrt(365)
        return max(ann_vol, MIN_ANNUALIZED_VOL)

    def compute_units(
        self,
        equity: float,
        prices: pd.DataFrame,
        current_price: float,
    ) -> float:
        """Compute number of units (fractional) to buy.

        Args:
            equity: Current account equity.
            prices: OHLCV data for volatility calculation.
            current_price: Current price of the asset.

        Returns:
            Number of units to buy (fractional for crypto).
        """
        if current_price <= 0:
            return 0.0
        size_usd = self.compute_position_size(equity, prices)
        return size_usd / current_price
