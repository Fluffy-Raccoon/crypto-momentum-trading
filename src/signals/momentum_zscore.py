"""Momentum Z-score signal generator with hysteresis."""

import pandas as pd

from src.signals.base import Signal


class MomentumZScore(Signal):
    """Generates signals based on trailing return Z-score.

    Entry: Z-score > entry_threshold -> +1 (long).
    Exit: Z-score < exit_threshold -> 0 (flat).
    Uses hysteresis to prevent flickering between states.
    """

    def __init__(
        self,
        lookback_days: int = 14,
        zscore_window: int = 90,
        entry_threshold: float = 1.0,
        exit_threshold: float = 0.0,
    ) -> None:
        """Initialize Z-score signal parameters.

        Args:
            lookback_days: Period for trailing return calculation.
            zscore_window: Rolling window for Z-score mean/std estimation.
            entry_threshold: Z-score above this triggers entry.
            exit_threshold: Z-score below this triggers exit.
        """
        self._lookback = lookback_days
        self._window = zscore_window
        self._entry = entry_threshold
        self._exit = exit_threshold

    def generate(self, prices: pd.DataFrame) -> pd.Series:
        """Generate momentum Z-score signals with hysteresis.

        Args:
            prices: OHLCV DataFrame with 'close' column.

        Returns:
            Series of {0, 1} signals indexed like the input.
        """
        close = prices["close"]
        trailing_return = close.pct_change(self._lookback)
        rolling_mean = trailing_return.rolling(self._window).mean()
        rolling_std = trailing_return.rolling(self._window).std()
        zscore = (trailing_return - rolling_mean) / rolling_std

        # Hysteresis: iterate forward maintaining state to prevent flickering
        signal = pd.Series(0, index=prices.index, dtype=int)
        in_position = False

        for i in range(len(zscore)):
            z = zscore.iloc[i]
            if pd.isna(z):
                signal.iloc[i] = 0
                in_position = False
                continue
            if in_position:
                if z < self._exit:
                    in_position = False
                    signal.iloc[i] = 0
                else:
                    signal.iloc[i] = 1
            else:
                if z > self._entry:
                    in_position = True
                    signal.iloc[i] = 1
                else:
                    signal.iloc[i] = 0

        return signal

    def compute_zscore(self, prices: pd.DataFrame) -> pd.Series:
        """Compute the raw Z-score series (for ranking/strength).

        Args:
            prices: OHLCV DataFrame with 'close' column.

        Returns:
            Series of Z-score values.
        """
        close = prices["close"]
        trailing_return = close.pct_change(self._lookback)
        rolling_mean = trailing_return.rolling(self._window).mean()
        rolling_std = trailing_return.rolling(self._window).std()
        return (trailing_return - rolling_mean) / rolling_std

    @property
    def name(self) -> str:
        """Signal name."""
        return f"momentum_zscore_{self._lookback}"

    @property
    def min_warmup_days(self) -> int:
        """Minimum warmup equals the Z-score rolling window."""
        return self._window
