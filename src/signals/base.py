"""Abstract base class for all signal generators."""

from abc import ABC, abstractmethod

import pandas as pd


class Signal(ABC):
    """Base class for all signal generators.

    Subclasses must implement generate(), name, and min_warmup_days.
    """

    @abstractmethod
    def generate(self, prices: pd.DataFrame) -> pd.Series:
        """Generate trading signals from OHLCV price data for one asset.

        Args:
            prices: DataFrame with at least a 'close' column for one asset.

        Returns:
            Series of signal values: +1 = long, 0 = flat/cash, -1 = short.
            Index must match the input price index.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this signal."""
        ...

    @property
    @abstractmethod
    def min_warmup_days(self) -> int:
        """Minimum rows of data needed before signal is valid."""
        ...
