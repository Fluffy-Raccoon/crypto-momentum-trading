"""Portfolio state management and P&L tracking."""

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open position.

    size > 0 for long positions, size < 0 for short positions.
    """

    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    size: float  # positive = long, negative = short
    cost: float = 0.0  # entry transaction cost

    @property
    def notional_value(self) -> float:
        """Absolute notional value at entry price."""
        return abs(self.size) * self.entry_price

    @property
    def side(self) -> str:
        """Position side: 'long' or 'short'."""
        return "long" if self.size > 0 else "short"


class Portfolio:
    """Tracks portfolio state: positions, equity, and P&L.

    Supports both long and short positions.
    - Long: buy units, profit when price rises.
    - Short: borrow and sell units, profit when price falls.
    """

    def __init__(self, initial_capital: float, max_positions: int = 5) -> None:
        """Initialize portfolio.

        Args:
            initial_capital: Starting cash in USD.
            max_positions: Maximum concurrent open positions.
        """
        self._initial_capital = initial_capital
        self._cash = initial_capital
        self._max_positions = max_positions
        self._positions: dict[str, Position] = {}
        self._trade_log: list[dict] = []
        self._equity_history: list[dict] = []

    @property
    def cash(self) -> float:
        """Current cash balance."""
        return self._cash

    @property
    def positions(self) -> dict[str, Position]:
        """Currently open positions."""
        return self._positions

    @property
    def num_positions(self) -> int:
        """Number of open positions."""
        return len(self._positions)

    @property
    def max_positions(self) -> int:
        """Maximum allowed concurrent positions."""
        return self._max_positions

    def open_position(
        self,
        symbol: str,
        date: pd.Timestamp,
        price: float,
        size: float,
        cost: float = 0.0,
    ) -> bool:
        """Open a new long or short position.

        Args:
            symbol: Trading pair.
            date: Entry date.
            price: Entry price.
            size: Number of units (positive=long, negative=short).
            cost: Transaction cost.

        Returns:
            True if position was opened, False if rejected.
        """
        if symbol in self._positions:
            logger.warning(f"Already have position in {symbol}, skipping")
            return False

        if self.num_positions >= self._max_positions:
            logger.warning(f"Max positions ({self._max_positions}) reached, rejecting {symbol}")
            return False

        notional = abs(size) * price
        # For longs: we pay notional + cost
        # For shorts: we receive notional but need margin (use notional as collateral)
        margin_required = notional + cost

        if margin_required > self._cash:
            logger.warning(
                f"Insufficient cash for {symbol}: "
                f"need {margin_required:.2f}, have {self._cash:.2f}"
            )
            return False

        self._cash -= margin_required
        self._positions[symbol] = Position(
            symbol=symbol,
            entry_date=date,
            entry_price=price,
            size=size,
            cost=cost,
        )
        side = "long" if size > 0 else "short"
        logger.info(
            f"Opened {side} {symbol}: {abs(size):.4f} units @ {price:.2f}, cost={cost:.2f}"
        )
        return True

    def close_position(
        self,
        symbol: str,
        date: pd.Timestamp,
        price: float,
        cost: float = 0.0,
    ) -> float | None:
        """Close an existing position (long or short).

        Args:
            symbol: Trading pair to close.
            date: Exit date.
            price: Exit price.
            cost: Transaction cost.

        Returns:
            Realized P&L, or None if no position exists.
        """
        if symbol not in self._positions:
            logger.warning(f"No position in {symbol} to close")
            return None

        pos = self._positions.pop(symbol)
        notional_entry = abs(pos.size) * pos.entry_price
        notional_exit = abs(pos.size) * price

        if pos.size > 0:
            # Long: bought at entry, sell at exit
            pnl = notional_exit - notional_entry - pos.cost - cost
            # Return notional_exit minus exit cost
            self._cash += notional_exit - cost
        else:
            # Short: sold at entry, buy back at exit
            # Profit when price goes down
            pnl = notional_entry - notional_exit - pos.cost - cost
            # Return collateral + profit (or - loss), minus exit cost
            self._cash += notional_entry + (notional_entry - notional_exit) - cost

        self._trade_log.append({
            "entry_date": pos.entry_date,
            "exit_date": date,
            "symbol": symbol,
            "side": pos.side,
            "entry_price": pos.entry_price,
            "exit_price": price,
            "size": pos.size,
            "pnl": pnl,
            "entry_cost": pos.cost,
            "exit_cost": cost,
        })
        logger.info(
            f"Closed {pos.side} {symbol}: {abs(pos.size):.4f} units @ {price:.2f}, P&L={pnl:.2f}"
        )
        return pnl

    def mark_to_market(self, date: pd.Timestamp, prices: dict[str, float]) -> float:
        """Mark all positions to current prices and record equity.

        Args:
            date: Current date.
            prices: Dict of symbol -> current price.

        Returns:
            Total portfolio equity.
        """
        positions_value = 0.0
        for pos in self._positions.values():
            current_price = prices.get(pos.symbol, pos.entry_price)
            if pos.size > 0:
                # Long: value is current market value
                positions_value += pos.size * current_price
            else:
                # Short: collateral (notional at entry) + unrealized P&L
                notional_entry = abs(pos.size) * pos.entry_price
                notional_current = abs(pos.size) * current_price
                positions_value += notional_entry + (notional_entry - notional_current)

        equity = self._cash + positions_value

        self._equity_history.append({
            "timestamp": date,
            "equity": equity,
            "cash": self._cash,
            "positions_value": positions_value,
            "num_positions": self.num_positions,
        })
        return equity

    def get_equity_curve(self) -> pd.DataFrame:
        """Get the equity history as a DataFrame."""
        if not self._equity_history:
            return pd.DataFrame(columns=["timestamp", "equity"])
        return pd.DataFrame(self._equity_history)

    def get_trade_log(self) -> pd.DataFrame:
        """Get the trade log as a DataFrame."""
        if not self._trade_log:
            return pd.DataFrame(columns=[
                "entry_date", "exit_date", "symbol", "side",
                "entry_price", "exit_price", "size", "pnl",
                "entry_cost", "exit_cost",
            ])
        return pd.DataFrame(self._trade_log)

    def get_total_exposure(self, prices: dict[str, float]) -> float:
        """Get total exposure as a fraction of equity (using absolute values).

        Args:
            prices: Dict of symbol -> current price.

        Returns:
            Total exposure as fraction (e.g., 0.8 = 80%).
        """
        positions_value = sum(
            abs(pos.size) * prices.get(pos.symbol, pos.entry_price)
            for pos in self._positions.values()
        )
        equity = self._cash + positions_value
        if equity <= 0:
            return 0.0
        return positions_value / equity

    def to_dict(self) -> dict:
        """Serialize portfolio state to a dict (for persistence)."""
        return {
            "cash": self._cash,
            "initial_capital": self._initial_capital,
            "max_positions": self._max_positions,
            "positions": {
                sym: {
                    "symbol": p.symbol,
                    "entry_date": str(p.entry_date),
                    "entry_price": p.entry_price,
                    "size": p.size,
                    "side": p.side,
                    "cost": p.cost,
                }
                for sym, p in self._positions.items()
            },
        }
