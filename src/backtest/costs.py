"""Transaction cost model."""


class TransactionCostModel:
    """Models trading costs: commission + slippage.

    cost = trade_value * (commission_pct + slippage_pct) / 100
    Applied on both entry and exit.
    """

    def __init__(self, config: dict) -> None:
        """Initialize from config.

        Args:
            config: Must contain 'costs' key with commission_pct, slippage_pct.
        """
        costs_cfg = config["costs"]
        self._commission_pct = costs_cfg["commission_pct"]
        self._slippage_pct = costs_cfg["slippage_pct"]

    def compute_cost(self, trade_value: float) -> float:
        """Compute transaction cost for a single trade (entry or exit).

        Args:
            trade_value: Absolute value of the trade in USD.

        Returns:
            Cost in USD (always non-negative).
        """
        if trade_value <= 0:
            return 0.0
        return trade_value * (self._commission_pct + self._slippage_pct) / 100.0

    @property
    def total_pct(self) -> float:
        """Total cost percentage per trade."""
        return self._commission_pct + self._slippage_pct
