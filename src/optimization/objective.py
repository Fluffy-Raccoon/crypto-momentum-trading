"""Optimization objective function wrapping BacktestEngine."""

import copy
import logging
import math

import optuna
import pandas as pd

from src.backtest.engine import BacktestEngine
from src.optimization.search_space import suggest_portfolio_params, suggest_signal_params
from src.reporting.metrics import PerformanceMetrics, compute_metrics
from src.signals.factory import create_signal

logger = logging.getLogger(__name__)

MIN_TRADES_THRESHOLD = 20


class OptimizationObjective:
    """Callable objective for Optuna that wraps BacktestEngine.

    Fetches data once and reuses it across all trials.
    """

    def __init__(
        self,
        base_config: dict,
        signal_type: str,
        ohlcv_data: dict[str, pd.DataFrame],
        objective_metric: str = "sharpe",
        capital: float | None = None,
        max_pos_range: tuple[int, int] = (2, 4),
        optimize_portfolio: bool = True,
    ) -> None:
        """Initialize the objective.

        Args:
            base_config: Base configuration dict (will be deep-copied per trial).
            signal_type: "ema" or "zscore".
            ohlcv_data: Pre-fetched OHLCV data (read-only, shared across trials).
            objective_metric: Metric to maximize ("sharpe", "sortino", "calmar", "cagr").
            capital: Override initial capital (e.g. 2000 for EUR portfolio).
            max_pos_range: Range for max_positions search.
            optimize_portfolio: Whether to also optimize portfolio parameters.
        """
        self._base_config = copy.deepcopy(base_config)
        if capital is not None:
            self._base_config["portfolio"]["initial_capital"] = capital
        self._signal_type = signal_type
        self._ohlcv_data = ohlcv_data
        self._objective_metric = objective_metric
        self._max_pos_range = max_pos_range
        self._optimize_portfolio = optimize_portfolio
        self.results: list[dict] = []

    def __call__(self, trial: optuna.Trial) -> float:
        """Run a single trial.

        Args:
            trial: Optuna trial.

        Returns:
            Objective value (higher is better).
        """
        # Suggest parameters
        signal_params = suggest_signal_params(trial, self._signal_type)
        portfolio_params = (
            suggest_portfolio_params(trial, self._max_pos_range)
            if self._optimize_portfolio
            else {}
        )

        # Build trial config
        config = self._build_config(signal_params, portfolio_params)

        # Run backtest
        try:
            signal = create_signal(self._signal_type, config)
            engine = BacktestEngine(config)
            result = engine.run(signal, self._ohlcv_data)
        except Exception as e:
            logger.debug(f"Trial {trial.number} failed: {e}")
            return -999.0

        # Compute metrics
        metrics = compute_metrics(result.equity_curve, result.trade_log)

        # Penalize too few trades
        if metrics.total_trades < MIN_TRADES_THRESHOLD:
            return -999.0

        # Store results
        self.results.append({
            "trial_number": trial.number,
            "params": {**signal_params, **portfolio_params},
            "metrics": metrics,
            "config": config,
        })

        return self._extract_objective(metrics)

    def _build_config(self, signal_params: dict, portfolio_params: dict) -> dict:
        """Build a trial-specific config by overriding base config."""
        config = copy.deepcopy(self._base_config)

        if self._signal_type == "ema":
            config["signals"]["ema_crossover"].update(signal_params)
        elif self._signal_type == "zscore":
            config["signals"]["momentum_zscore"].update(signal_params)

        if portfolio_params:
            config["portfolio"].update(portfolio_params)

        return config

    def _extract_objective(self, metrics: PerformanceMetrics) -> float:
        """Extract scalar objective value from metrics."""
        mapping = {
            "sharpe": metrics.sharpe_ratio,
            "sortino": metrics.sortino_ratio,
            "calmar": metrics.calmar_ratio,
            "cagr": metrics.cagr,
        }
        value = mapping[self._objective_metric]
        if math.isinf(value) or math.isnan(value):
            return -999.0
        return value


def validate_best_config(
    best_config: dict,
    signal_type: str,
    ohlcv_data: dict[str, pd.DataFrame],
    holdout_fraction: float = 0.2,
) -> tuple[PerformanceMetrics, PerformanceMetrics]:
    """Run the best config on in-sample and holdout periods separately.

    Args:
        best_config: Configuration dict with optimal parameters.
        signal_type: "ema" or "zscore".
        ohlcv_data: Full OHLCV data.
        holdout_fraction: Fraction of data to hold out (default 0.2).

    Returns:
        Tuple of (in_sample_metrics, holdout_metrics).
    """
    # Find temporal split point across all symbols
    all_dates: set[pd.Timestamp] = set()
    for df in ohlcv_data.values():
        all_dates.update(df["timestamp"].tolist())
    sorted_dates = sorted(all_dates)

    split_idx = int(len(sorted_dates) * (1 - holdout_fraction))
    split_date = sorted_dates[split_idx]

    # Split data
    in_sample_data = {
        sym: df[df["timestamp"] <= split_date].copy()
        for sym, df in ohlcv_data.items()
    }
    holdout_data = {
        sym: df[df["timestamp"] > split_date].copy()
        for sym, df in ohlcv_data.items()
    }

    # Filter out empty DataFrames
    in_sample_data = {s: d for s, d in in_sample_data.items() if len(d) > 0}
    holdout_data = {s: d for s, d in holdout_data.items() if len(d) > 0}

    is_metrics = _run_and_compute(best_config, signal_type, in_sample_data)
    oos_metrics = _run_and_compute(best_config, signal_type, holdout_data)

    return is_metrics, oos_metrics


def _run_and_compute(
    config: dict,
    signal_type: str,
    ohlcv_data: dict[str, pd.DataFrame],
) -> PerformanceMetrics:
    """Run a backtest and compute metrics."""
    signal = create_signal(signal_type, config)
    engine = BacktestEngine(config)
    result = engine.run(signal, ohlcv_data)
    return compute_metrics(result.equity_curve, result.trade_log)
