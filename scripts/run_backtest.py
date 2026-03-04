#!/usr/bin/env python3
"""CLI entry point for the crypto momentum backtester."""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from src.backtest.engine import BacktestEngine
from src.data.fetcher import BinanceFetcher
from src.data.universe import CoinUniverse
from src.reporting.metrics import compute_metrics
from src.reporting.tearsheet import generate_tearsheet
from src.signals.ema_crossover import EMACrossover
from src.signals.momentum_zscore import MomentumZScore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Crypto Time-Series Momentum Backtester",
    )
    parser.add_argument(
        "--signal",
        choices=["ema", "zscore", "both"],
        default="both",
        help="Signal type to backtest (default: both)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/",
        help="Output directory for results (default: results/)",
    )
    return parser.parse_args(argv)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file.

    Returns:
        Configuration dict.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


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


def run_single_backtest(signal_type: str, config: dict, ohlcv_data: dict, output_dir: Path):
    """Run a single backtest for one signal type.

    Args:
        signal_type: "ema" or "zscore".
        config: Full configuration dict.
        ohlcv_data: Pre-fetched OHLCV data.
        output_dir: Output directory.

    Returns:
        BacktestResult.
    """
    signal = create_signal(signal_type, config)
    logger.info(f"Running backtest with signal: {signal.name}")

    engine = BacktestEngine(config)
    result = engine.run(signal, ohlcv_data)

    # Generate tearsheet
    generate_tearsheet(result, output_dir)

    # Print summary
    metrics = compute_metrics(result.equity_curve, result.trade_log)
    logger.info(f"\n{'=' * 50}")
    logger.info(f"Results for {signal.name}")
    logger.info(f"{'=' * 50}")
    for k, v in metrics.to_dict().items():
        logger.info(f"  {k}: {v}")

    return result


def main(argv: list[str] | None = None) -> int:
    """Main entry point.

    Args:
        argv: Command-line arguments (for testing).

    Returns:
        Exit code.
    """
    args = parse_args(argv)
    config = load_config(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch data
    logger.info("Fetching OHLCV data...")
    fetcher = BinanceFetcher(config)
    universe = CoinUniverse(config)
    symbols = universe.get_all_candidate_symbols()

    start_date = config["data"]["start_date"]
    end_date = config["data"].get("end_date")

    ohlcv_data = fetcher.fetch_multiple(symbols, start_date, end_date)
    logger.info(f"Fetched data for {len(ohlcv_data)} symbols")

    if not ohlcv_data:
        logger.error("No data fetched. Check API connectivity and symbol list.")
        return 1

    # Run backtests
    signal_types = ["ema", "zscore"] if args.signal == "both" else [args.signal]
    results = {}

    for sig_type in signal_types:
        sig_output = output_dir / sig_type
        result = run_single_backtest(sig_type, config, ohlcv_data, sig_output)
        results[sig_type] = result

    # Comparison table for --signal both
    if len(results) > 1:
        logger.info(f"\n{'=' * 60}")
        logger.info("COMPARISON")
        logger.info(f"{'=' * 60}")
        comparison = {}
        for sig_type, result in results.items():
            metrics = compute_metrics(result.equity_curve, result.trade_log)
            comparison[sig_type] = metrics.to_dict()

        # Save comparison JSON
        comp_path = output_dir / "comparison.json"
        comp_path.write_text(json.dumps(comparison, indent=2))
        logger.info(f"Comparison saved to {comp_path}")

    logger.info("Backtest complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
