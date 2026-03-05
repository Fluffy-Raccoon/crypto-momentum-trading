"""Signal factory for creating signal generators from config."""

from src.signals.ema_crossover import EMACrossover
from src.signals.momentum_zscore import MomentumZScore


def create_signal(signal_type: str, config: dict):
    """Create a signal generator from config.

    Args:
        signal_type: "ema" or "zscore".
        config: Full configuration dict.

    Returns:
        Signal instance.
    """
    if signal_type == "ema":
        ema_cfg = config["signals"]["ema_crossover"]
        return EMACrossover(
            fast_period=ema_cfg["fast_period"],
            slow_period=ema_cfg["slow_period"],
        )
    elif signal_type == "zscore":
        zs_cfg = config["signals"]["momentum_zscore"]
        return MomentumZScore(
            lookback_days=zs_cfg["lookback_days"],
            zscore_window=zs_cfg["zscore_window"],
            entry_threshold=zs_cfg["entry_threshold"],
            exit_threshold=zs_cfg["exit_threshold"],
        )
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")
