"""Parameter search space definitions for Optuna optimization."""

import optuna


def suggest_ema_params(trial: optuna.Trial) -> dict:
    """Suggest EMA crossover parameters.

    Args:
        trial: Optuna trial.

    Returns:
        Dict with fast_period and slow_period.
    """
    fast_period = trial.suggest_int("ema_fast_period", 5, 15)
    slow_period = trial.suggest_int("ema_slow_period", 15, 50)
    if fast_period >= slow_period:
        raise optuna.TrialPruned()
    return {"fast_period": fast_period, "slow_period": slow_period}


def suggest_zscore_params(trial: optuna.Trial) -> dict:
    """Suggest momentum Z-score parameters.

    Args:
        trial: Optuna trial.

    Returns:
        Dict with lookback_days, zscore_window, entry_threshold, exit_threshold.
    """
    lookback_days = trial.suggest_int("zscore_lookback_days", 7, 30)
    zscore_window = trial.suggest_int("zscore_window", 30, 180, step=15)
    entry_threshold = trial.suggest_float("zscore_entry_threshold", 0.5, 2.0, step=0.25)
    exit_threshold = trial.suggest_float("zscore_exit_threshold", -0.5, 0.5, step=0.25)
    if entry_threshold <= exit_threshold:
        raise optuna.TrialPruned()
    return {
        "lookback_days": lookback_days,
        "zscore_window": zscore_window,
        "entry_threshold": entry_threshold,
        "exit_threshold": exit_threshold,
    }


def suggest_portfolio_params(
    trial: optuna.Trial,
    max_pos_range: tuple[int, int] = (2, 4),
) -> dict:
    """Suggest portfolio parameters.

    Args:
        trial: Optuna trial.
        max_pos_range: Min/max for max_positions search.

    Returns:
        Dict with max_positions, risk_per_position_pct, vol_lookback_days.
    """
    max_positions = trial.suggest_int("max_positions", max_pos_range[0], max_pos_range[1])
    risk_per_position_pct = trial.suggest_float("risk_per_position_pct", 5.0, 10.0, step=1.0)
    vol_lookback_days = trial.suggest_int("vol_lookback_days", 14, 56, step=7)
    return {
        "max_positions": max_positions,
        "risk_per_position_pct": risk_per_position_pct,
        "vol_lookback_days": vol_lookback_days,
    }


def suggest_signal_params(trial: optuna.Trial, signal_type: str) -> dict:
    """Dispatch to the appropriate signal parameter suggester.

    Args:
        trial: Optuna trial.
        signal_type: "ema" or "zscore".

    Returns:
        Dict of signal parameters.
    """
    if signal_type == "ema":
        return suggest_ema_params(trial)
    elif signal_type == "zscore":
        return suggest_zscore_params(trial)
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")
