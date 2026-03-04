"""Pre-trade and portfolio-level risk checks."""

import logging

logger = logging.getLogger(__name__)

# Minimum trade size in USD
MIN_TRADE_SIZE = 10.0

# Exposure warning threshold
EXPOSURE_WARNING_THRESHOLD = 1.0  # 100% of equity


class RiskManager:
    """Performs pre-trade risk checks and portfolio-level monitoring."""

    def __init__(self, config: dict) -> None:
        """Initialize from config.

        Args:
            config: Must contain 'portfolio' key with max_positions,
                    risk_per_position_pct.
        """
        port_cfg = config["portfolio"]
        self._max_positions = port_cfg["max_positions"]
        self._risk_pct_min = 1.0  # min risk per position %
        self._risk_pct_max = 2.0  # max risk per position %

    def check_new_position(
        self,
        current_positions: int,
        trade_size_usd: float,
        equity: float,
    ) -> tuple[bool, str]:
        """Check if a new position can be opened.

        Args:
            current_positions: Number of currently open positions.
            trade_size_usd: Proposed trade size in USD.
            equity: Current portfolio equity.

        Returns:
            Tuple of (allowed, reason).
        """
        if current_positions >= self._max_positions:
            return False, f"Max positions ({self._max_positions}) already reached"

        if trade_size_usd < MIN_TRADE_SIZE:
            return False, f"Trade size ${trade_size_usd:.2f} below minimum ${MIN_TRADE_SIZE}"

        if trade_size_usd > equity:
            return False, f"Trade size ${trade_size_usd:.2f} exceeds equity ${equity:.2f}"

        return True, "OK"

    def clamp_risk_pct(self, risk_pct: float) -> float:
        """Clamp risk percentage to allowed range [1%, 2%].

        Args:
            risk_pct: Desired risk percentage.

        Returns:
            Clamped risk percentage.
        """
        return max(self._risk_pct_min, min(self._risk_pct_max, risk_pct))

    def check_exposure(
        self,
        total_exposure: float,
    ) -> bool:
        """Check portfolio exposure and log warning if too high.

        Args:
            total_exposure: Total exposure as fraction of equity.

        Returns:
            True if exposure is acceptable, False if over threshold.
        """
        if total_exposure > EXPOSURE_WARNING_THRESHOLD:
            logger.warning(
                f"Portfolio exposure {total_exposure:.1%} exceeds "
                f"{EXPOSURE_WARNING_THRESHOLD:.0%} of equity"
            )
            return False
        return True
