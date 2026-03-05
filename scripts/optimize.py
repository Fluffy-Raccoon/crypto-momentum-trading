#!/usr/bin/env python3
"""CLI entry point for parameter optimization."""

import argparse
import logging
import sys
from pathlib import Path

import optuna
import yaml

from src.backtest.engine import BacktestEngine
from src.data.fetcher import BinanceFetcher
from src.data.universe import CoinUniverse
from src.optimization.objective import OptimizationObjective, validate_best_config
from src.optimization.results import (
    export_all_results,
    export_best_config,
    print_holdout_summary,
    print_summary,
    rank_results,
)
from src.reporting.tearsheet import generate_tearsheet
from src.signals.factory import create_signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress Optuna's verbose trial logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Optimize crypto momentum strategy parameters",
    )
    parser.add_argument(
        "--signal",
        choices=["ema", "zscore", "both"],
        required=True,
        help="Signal type to optimize",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to base config YAML (default: config.yaml)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials per signal type (default: 100)",
    )
    parser.add_argument(
        "--objective",
        choices=["sharpe", "sortino", "calmar", "cagr"],
        default="sharpe",
        help="Metric to maximize (default: sharpe)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=None,
        help="Override initial capital (e.g. 2000 for EUR portfolio)",
    )
    parser.add_argument(
        "--max-positions-range",
        type=int,
        nargs=2,
        default=[2, 4],
        metavar=("MIN", "MAX"),
        help="Range for max_positions search (default: 2 4)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Show top N results (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/optimization/",
        help="Output directory (default: results/optimization/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no-portfolio-opt",
        action="store_true",
        help="Skip portfolio parameter optimization (signal params only)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Total optimization timeout in minutes",
    )
    return parser.parse_args(argv)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def fetch_data(config: dict) -> dict:
    """Fetch OHLCV data once for all trials.

    Args:
        config: Configuration dict.

    Returns:
        Dict mapping symbol to OHLCV DataFrame.
    """
    logger.info("Fetching OHLCV data (will use cache if available)...")
    fetcher = BinanceFetcher(config)
    universe = CoinUniverse(config)
    symbols = universe.get_all_candidate_symbols()

    start_date = config["data"]["start_date"]
    end_date = config["data"].get("end_date")

    ohlcv_data = fetcher.fetch_multiple(symbols, start_date, end_date)
    logger.info(f"Fetched data for {len(ohlcv_data)} symbols")

    if not ohlcv_data:
        logger.error("No data fetched. Check API connectivity and symbol list.")
        sys.exit(1)

    return ohlcv_data


def optimize_signal(
    signal_type: str,
    config: dict,
    ohlcv_data: dict,
    args: argparse.Namespace,
) -> list[dict]:
    """Run optimization for a single signal type.

    Args:
        signal_type: "ema" or "zscore".
        config: Base configuration dict.
        ohlcv_data: Pre-fetched OHLCV data.
        args: CLI arguments.

    Returns:
        Ranked list of top results.
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"OPTIMIZING: {signal_type.upper()} ({args.n_trials} trials, "
                f"objective={args.objective})")
    logger.info(f"{'=' * 60}")

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"optimize_{signal_type}",
    )

    objective = OptimizationObjective(
        base_config=config,
        signal_type=signal_type,
        ohlcv_data=ohlcv_data,
        objective_metric=args.objective,
        capital=args.capital,
        max_pos_range=tuple(args.max_positions_range),
        optimize_portfolio=not args.no_portfolio_opt,
    )

    timeout_seconds = args.timeout * 60 if args.timeout else None
    study.optimize(objective, n_trials=args.n_trials, timeout=timeout_seconds)

    if not objective.results:
        logger.warning(f"No valid results for {signal_type}. All trials may have been pruned.")
        return []

    ranked = rank_results(objective.results, args.objective, args.top_n)
    print_summary(ranked, args.objective, signal_type)

    # Export results
    output_dir = Path(args.output) / signal_type
    output_dir.mkdir(parents=True, exist_ok=True)

    export_best_config(ranked[0], output_dir / "best_config.yaml")
    export_all_results(objective.results, output_dir / "optimization_results.json")

    # Holdout validation
    logger.info("Running holdout validation on best parameters...")
    try:
        is_metrics, oos_metrics = validate_best_config(
            ranked[0]["config"], signal_type, ohlcv_data
        )
        print_holdout_summary(is_metrics, oos_metrics)
    except Exception as e:
        logger.warning(f"Holdout validation failed: {e}")

    # Generate tearsheet for best result
    logger.info("Generating tearsheet for best parameters...")
    try:
        best_signal = create_signal(signal_type, ranked[0]["config"])
        best_engine = BacktestEngine(ranked[0]["config"])
        best_result = best_engine.run(best_signal, ohlcv_data)
        generate_tearsheet(best_result, output_dir / "best_tearsheet")
    except Exception as e:
        logger.warning(f"Tearsheet generation failed: {e}")

    return ranked


def main(argv: list[str] | None = None) -> int:
    """Main entry point.

    Args:
        argv: Command-line arguments (for testing).

    Returns:
        Exit code.
    """
    args = parse_args(argv)
    config = load_config(args.config)

    # Fetch data once
    ohlcv_data = fetch_data(config)

    # Determine signal types to optimize
    signal_types = ["ema", "zscore"] if args.signal == "both" else [args.signal]
    all_ranked: dict[str, list[dict]] = {}

    for sig_type in signal_types:
        ranked = optimize_signal(sig_type, config, ohlcv_data, args)
        if ranked:
            all_ranked[sig_type] = ranked

    # Comparison if both were optimized
    if len(all_ranked) > 1:
        logger.info(f"\n{'=' * 60}")
        logger.info("STRATEGY COMPARISON — best of each signal type")
        logger.info(f"{'=' * 60}")
        for sig_type, ranked in all_ranked.items():
            m = ranked[0]["metrics"]
            logger.info(
                f"  {sig_type.upper():>8}: Sharpe={m.sharpe_ratio:.2f}  "
                f"CAGR={m.cagr:.2%}  MaxDD={m.max_drawdown_pct:.2%}  "
                f"Trades={m.total_trades}"
            )

        # Determine winner
        best_type = max(
            all_ranked,
            key=lambda t: getattr(
                all_ranked[t][0]["metrics"],
                {"sharpe": "sharpe_ratio", "sortino": "sortino_ratio",
                 "calmar": "calmar_ratio", "cagr": "cagr"}[args.objective],
            ),
        )
        logger.info(f"\n  Winner by {args.objective}: {best_type.upper()}")
        logger.info(
            f"  Config: {Path(args.output) / best_type / 'best_config.yaml'}"
        )

    logger.info("\nOptimization complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
