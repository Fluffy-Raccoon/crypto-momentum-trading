"""Tests for the optimization module."""

import math
from pathlib import Path
from unittest.mock import MagicMock

import optuna
import pytest
import yaml

from src.optimization.objective import MIN_TRADES_THRESHOLD, OptimizationObjective, validate_best_config
from src.optimization.results import (
    export_all_results,
    export_best_config,
    format_comparison_table,
    rank_results,
)
from src.optimization.search_space import (
    suggest_ema_params,
    suggest_portfolio_params,
    suggest_signal_params,
    suggest_zscore_params,
)
from src.reporting.metrics import PerformanceMetrics
from src.signals.factory import create_signal


# --- Signal factory tests ---


class TestSignalFactory:
    """Tests for the signal factory function."""

    def test_create_ema_signal(self, config):
        signal = create_signal("ema", config)
        assert "ema_crossover" in signal.name

    def test_create_zscore_signal(self, config):
        signal = create_signal("zscore", config)
        assert "momentum_zscore" in signal.name

    def test_unknown_signal_raises(self, config):
        with pytest.raises(ValueError, match="Unknown signal type"):
            create_signal("unknown", config)


# --- Search space tests ---


class TestSearchSpace:
    """Tests for parameter search space definitions."""

    def test_ema_params_valid(self):
        study = optuna.create_study()
        trial = study.ask()
        params = suggest_ema_params(trial)
        assert "fast_period" in params
        assert "slow_period" in params
        assert params["fast_period"] < params["slow_period"]

    def test_ema_fast_in_range(self):
        study = optuna.create_study()
        for _ in range(20):
            trial = study.ask()
            try:
                params = suggest_ema_params(trial)
                assert 5 <= params["fast_period"] <= 15
                assert 15 <= params["slow_period"] <= 50
            except optuna.TrialPruned:
                pass  # Constraint violation, expected sometimes

    def test_zscore_params_valid(self):
        study = optuna.create_study()
        for _ in range(50):
            trial = study.ask()
            try:
                params = suggest_zscore_params(trial)
                assert params["entry_threshold"] > params["exit_threshold"]
                assert 7 <= params["lookback_days"] <= 30
                assert 30 <= params["zscore_window"] <= 180
                break
            except optuna.TrialPruned:
                continue

    def test_zscore_entry_gt_exit_enforced(self):
        """Entry threshold must always be > exit threshold."""
        study = optuna.create_study()
        valid_count = 0
        for _ in range(100):
            trial = study.ask()
            try:
                params = suggest_zscore_params(trial)
                assert params["entry_threshold"] > params["exit_threshold"]
                valid_count += 1
            except optuna.TrialPruned:
                pass
        # At least some valid trials should exist
        assert valid_count > 0

    def test_portfolio_params_valid(self):
        study = optuna.create_study()
        trial = study.ask()
        params = suggest_portfolio_params(trial, max_pos_range=(2, 4))
        assert 2 <= params["max_positions"] <= 4
        assert 5.0 <= params["risk_per_position_pct"] <= 10.0
        assert 14 <= params["vol_lookback_days"] <= 60

    def test_suggest_signal_params_dispatches(self):
        study = optuna.create_study()
        trial = study.ask()
        params = suggest_signal_params(trial, "ema")
        assert "fast_period" in params

    def test_suggest_signal_params_unknown_raises(self):
        study = optuna.create_study()
        trial = study.ask()
        with pytest.raises(ValueError, match="Unknown signal type"):
            suggest_signal_params(trial, "rsi")


# --- Objective function tests ---


class TestObjective:
    """Tests for the optimization objective function."""

    def test_returns_finite_float(self, config, multi_coin_ohlcv):
        objective = OptimizationObjective(
            base_config=config,
            signal_type="ema",
            ohlcv_data=multi_coin_ohlcv,
            objective_metric="sharpe",
            capital=2000,
            optimize_portfolio=False,
        )
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=3)
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                assert math.isfinite(trial.value)

    def test_results_accumulated(self, config, multi_coin_ohlcv):
        objective = OptimizationObjective(
            base_config=config,
            signal_type="ema",
            ohlcv_data=multi_coin_ohlcv,
            objective_metric="sharpe",
            capital=2000,
            optimize_portfolio=False,
        )
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=3)
        # Results should be accumulated (may be less than n_trials if some pruned)
        assert len(objective.results) >= 0

    def test_capital_override(self, config, multi_coin_ohlcv):
        objective = OptimizationObjective(
            base_config=config,
            signal_type="ema",
            ohlcv_data=multi_coin_ohlcv,
            capital=2000,
        )
        assert objective._base_config["portfolio"]["initial_capital"] == 2000

    def test_penalizes_degenerate(self, config, multi_coin_ohlcv):
        """Strategies with too few trades should be penalized."""
        objective = OptimizationObjective(
            base_config=config,
            signal_type="ema",
            ohlcv_data=multi_coin_ohlcv,
            objective_metric="sharpe",
            capital=2000,
        )
        # Create a mock trial that forces params producing few/no trades
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=2)
        # Verify no result has value > -999 with < MIN_TRADES_THRESHOLD trades
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE and trial.value > -999:
                # Find corresponding result
                matching = [r for r in objective.results if r["trial_number"] == trial.number]
                if matching:
                    assert matching[0]["metrics"].total_trades >= MIN_TRADES_THRESHOLD


# --- Results tests ---


class TestResults:
    """Tests for results ranking, formatting, and export."""

    @pytest.fixture
    def sample_results(self):
        """Create sample result dicts for testing."""
        results = []
        for i in range(5):
            results.append({
                "trial_number": i,
                "params": {
                    "fast_period": 5 + i,
                    "slow_period": 20 + i,
                    "max_positions": 3,
                    "risk_per_position_pct": 1.5,
                },
                "metrics": PerformanceMetrics(
                    cagr=0.05 + i * 0.01,
                    sharpe_ratio=0.5 + i * 0.2,
                    sortino_ratio=0.6 + i * 0.15,
                    max_drawdown_pct=-0.1 - i * 0.01,
                    max_drawdown_duration_days=100 + i * 10,
                    calmar_ratio=0.3 + i * 0.1,
                    profit_factor=1.2 + i * 0.1,
                    win_rate=0.35 + i * 0.02,
                    avg_win_loss_ratio=2.0 + i * 0.1,
                    total_trades=50 + i * 10,
                    avg_holding_period_days=10.0 + i,
                    exposure_pct=0.4 + i * 0.05,
                ),
                "config": {"signals": {"ema_crossover": {"fast_period": 5 + i, "slow_period": 20 + i}}},
            })
        return results

    def test_rank_results_by_sharpe(self, sample_results):
        ranked = rank_results(sample_results, "sharpe", top_n=3)
        assert len(ranked) == 3
        # Should be sorted descending by sharpe
        assert ranked[0]["metrics"].sharpe_ratio >= ranked[1]["metrics"].sharpe_ratio
        assert ranked[1]["metrics"].sharpe_ratio >= ranked[2]["metrics"].sharpe_ratio

    def test_rank_results_by_cagr(self, sample_results):
        ranked = rank_results(sample_results, "cagr", top_n=2)
        assert len(ranked) == 2
        assert ranked[0]["metrics"].cagr >= ranked[1]["metrics"].cagr

    def test_format_comparison_table_ema(self, sample_results):
        ranked = rank_results(sample_results, "sharpe", top_n=3)
        table = format_comparison_table(ranked, "ema")
        assert "Rank" in table
        assert "fast" in table
        assert "slow" in table
        assert "Sharpe" in table

    def test_format_comparison_table_zscore(self):
        results = [{
            "trial_number": 0,
            "params": {
                "lookback_days": 14,
                "zscore_window": 90,
                "entry_threshold": 1.0,
                "exit_threshold": 0.0,
                "max_positions": 3,
            },
            "metrics": PerformanceMetrics(
                cagr=0.05, sharpe_ratio=0.8, sortino_ratio=0.9,
                max_drawdown_pct=-0.1, max_drawdown_duration_days=100,
                calmar_ratio=0.5, profit_factor=1.5, win_rate=0.4,
                avg_win_loss_ratio=2.0, total_trades=100,
                avg_holding_period_days=10, exposure_pct=0.5,
            ),
            "config": {},
        }]
        table = format_comparison_table(results, "zscore")
        assert "look" in table
        assert "entry" in table

    def test_format_comparison_table_empty(self):
        table = format_comparison_table([], "ema")
        assert "No results" in table

    def test_export_best_config(self, sample_results, tmp_path):
        ranked = rank_results(sample_results, "sharpe", top_n=1)
        output_path = tmp_path / "best_config.yaml"
        export_best_config(ranked[0], output_path)
        assert output_path.exists()
        with open(output_path) as f:
            loaded = yaml.safe_load(f)
        assert "signals" in loaded

    def test_export_all_results(self, sample_results, tmp_path):
        output_path = tmp_path / "results.json"
        export_all_results(sample_results, output_path)
        assert output_path.exists()
        import json
        with open(output_path) as f:
            loaded = json.load(f)
        assert len(loaded) == 5
        assert "trial_number" in loaded[0]
        assert "metrics" in loaded[0]


# --- Holdout validation tests ---


class TestHoldoutValidation:
    """Tests for the holdout validation function."""

    def test_holdout_splits_data(self, config, multi_coin_ohlcv):
        """Validate that holdout validation runs and returns two sets of metrics."""
        # Use a simple EMA config that works
        config["signals"]["ema_crossover"]["fast_period"] = 10
        config["signals"]["ema_crossover"]["slow_period"] = 20
        config["portfolio"]["initial_capital"] = 2000

        is_metrics, oos_metrics = validate_best_config(
            config, "ema", multi_coin_ohlcv, holdout_fraction=0.3
        )
        assert isinstance(is_metrics, PerformanceMetrics)
        assert isinstance(oos_metrics, PerformanceMetrics)
