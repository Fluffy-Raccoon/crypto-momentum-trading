"""Shared data contracts and types for the crypto momentum backtester."""

from dataclasses import dataclass, field
from typing import NamedTuple

import pandas as pd

# --- Column contracts ---

REQUIRED_OHLCV_COLS = ["timestamp", "open", "high", "low", "close", "volume", "symbol"]

REQUIRED_SIGNAL_COLS = ["timestamp", "symbol", "signal"]

REQUIRED_POSITION_COLS = ["timestamp", "symbol", "weight", "risk_pct"]


# --- Structured types ---


class WalkForwardWindow(NamedTuple):
    """A single walk-forward analysis window."""

    formation_start: pd.Timestamp
    formation_end: pd.Timestamp
    trading_start: pd.Timestamp
    trading_end: pd.Timestamp


@dataclass
class BacktestResult:
    """Container for backtest output data."""

    equity_curve: pd.DataFrame  # columns: ['timestamp', 'equity']
    trade_log: pd.DataFrame  # columns: ['entry_date', 'exit_date', 'symbol', ...]
    daily_returns: pd.Series  # indexed by timestamp
    positions_over_time: pd.DataFrame  # columns: ['timestamp', 'symbol', 'weight']
    signal_name: str
    config: dict = field(default_factory=dict)


# --- Validation helpers ---


def validate_ohlcv(df: pd.DataFrame) -> None:
    """Validate that a DataFrame conforms to the OHLCV contract."""
    missing = set(REQUIRED_OHLCV_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV DataFrame missing columns: {missing}")


def validate_signals(df: pd.DataFrame) -> None:
    """Validate that a DataFrame conforms to the signal contract."""
    missing = set(REQUIRED_SIGNAL_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Signal DataFrame missing columns: {missing}")
    invalid = set(df["signal"].unique()) - {-1, 0, 1}
    if invalid:
        raise ValueError(f"Signal values must be in {{-1, 0, 1}}, got: {invalid}")
