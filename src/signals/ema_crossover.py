"""EMA crossover signal generator."""

import pandas as pd

from src.signals.base import Signal


class EMACrossover(Signal):
    """Generates signals based on fast/slow EMA crossover.

    Signal = +1 when fast EMA > slow EMA, 0 otherwise.
    """

    def __init__(self, fast_period: int = 10, slow_period: int = 20) -> None:
        """Initialize with EMA periods.

        Args:
            fast_period: Fast EMA lookback (default 10).
            slow_period: Slow EMA lookback (default 20).
        """
        if fast_period >= slow_period:
            raise ValueError(f"fast_period ({fast_period}) must be < slow_period ({slow_period})")
        self._fast = fast_period
        self._slow = slow_period

    def generate(self, prices: pd.DataFrame) -> pd.Series:
        """Generate EMA crossover signals.

        Args:
            prices: OHLCV DataFrame with 'close' column.

        Returns:
            Series of {0, 1} signals indexed like the input.
        """
        close = prices["close"]
        fast_ema = close.ewm(span=self._fast, adjust=False).mean()
        slow_ema = close.ewm(span=self._slow, adjust=False).mean()
        signal = (fast_ema > slow_ema).astype(int)
        # Zero out warmup period where EMA is unreliable
        signal.iloc[: self._slow] = 0
        return signal

    def signal_strength(self, prices: pd.DataFrame) -> pd.Series:
        """Compute signal strength as (fast_ema - slow_ema) / price.

        Used for ranking when selecting top-N positions.

        Args:
            prices: OHLCV DataFrame with 'close' column.

        Returns:
            Series of signal strength values.
        """
        close = prices["close"]
        fast_ema = close.ewm(span=self._fast, adjust=False).mean()
        slow_ema = close.ewm(span=self._slow, adjust=False).mean()
        strength = (fast_ema - slow_ema) / close
        strength.iloc[: self._slow] = 0.0
        return strength

    @property
    def name(self) -> str:
        """Signal name."""
        return f"ema_crossover_{self._fast}_{self._slow}"

    @property
    def min_warmup_days(self) -> int:
        """Minimum warmup equals the slow period."""
        return self._slow
