"""Results collection, ranking, and export for optimization."""

import json
import logging
from pathlib import Path

import yaml

from src.reporting.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


def rank_results(
    results: list[dict],
    objective_metric: str,
    top_n: int = 10,
) -> list[dict]:
    """Sort results by objective metric descending, return top N.

    Args:
        results: List of trial result dicts with 'metrics' key.
        objective_metric: Metric name to sort by.
        top_n: Number of top results to return.

    Returns:
        Sorted list of top N result dicts.
    """
    metric_attr = {
        "sharpe": "sharpe_ratio",
        "sortino": "sortino_ratio",
        "calmar": "calmar_ratio",
        "cagr": "cagr",
    }
    attr = metric_attr[objective_metric]
    sorted_results = sorted(
        results,
        key=lambda r: getattr(r["metrics"], attr),
        reverse=True,
    )
    return sorted_results[:top_n]


def format_comparison_table(ranked: list[dict], signal_type: str) -> str:
    """Format top-N results as a readable ASCII table.

    Args:
        ranked: Ranked list of result dicts.
        signal_type: "ema" or "zscore" (affects which param columns to show).

    Returns:
        Formatted table string.
    """
    if not ranked:
        return "No results to display."

    lines = []

    if signal_type == "ema":
        header = (
            f"{'Rank':>4} | {'fast':>4} | {'slow':>4} | {'max_pos':>7} | {'risk%':>5} | "
            f"{'Sharpe':>6} | {'CAGR':>7} | {'MaxDD':>7} | {'Trades':>6} | {'Win%':>5}"
        )
        sep = "-" * len(header)
        lines.append(header)
        lines.append(sep)

        for i, r in enumerate(ranked, 1):
            p = r["params"]
            m: PerformanceMetrics = r["metrics"]
            lines.append(
                f"{i:>4} | {p.get('fast_period', '-'):>4} | {p.get('slow_period', '-'):>4} | "
                f"{p.get('max_positions', '-'):>7} | {p.get('risk_per_position_pct', '-'):>5} | "
                f"{m.sharpe_ratio:>6.2f} | {m.cagr:>6.2%} | {m.max_drawdown_pct:>6.2%} | "
                f"{m.total_trades:>6} | {m.win_rate:>4.1%}"
            )
    elif signal_type == "zscore":
        header = (
            f"{'Rank':>4} | {'look':>4} | {'win':>3} | {'entry':>5} | {'exit':>5} | "
            f"{'max_pos':>7} | {'Sharpe':>6} | {'CAGR':>7} | {'MaxDD':>7} | {'Trades':>6}"
        )
        sep = "-" * len(header)
        lines.append(header)
        lines.append(sep)

        for i, r in enumerate(ranked, 1):
            p = r["params"]
            m: PerformanceMetrics = r["metrics"]
            lines.append(
                f"{i:>4} | {p.get('lookback_days', '-'):>4} | "
                f"{p.get('zscore_window', '-'):>3} | "
                f"{p.get('entry_threshold', '-'):>5} | "
                f"{p.get('exit_threshold', '-'):>5} | "
                f"{p.get('max_positions', '-'):>7} | "
                f"{m.sharpe_ratio:>6.2f} | {m.cagr:>6.2%} | {m.max_drawdown_pct:>6.2%} | "
                f"{m.total_trades:>6}"
            )

    return "\n".join(lines)


def export_best_config(best_result: dict, output_path: Path) -> None:
    """Save the best trial's config as a YAML file.

    Args:
        best_result: Result dict with 'config' key.
        output_path: Path to write YAML file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(best_result["config"], f, default_flow_style=False, sort_keys=False)
    logger.info(f"Best config saved to {output_path}")


def export_all_results(results: list[dict], output_path: Path) -> None:
    """Save all trial results as a JSON file.

    Args:
        results: List of trial result dicts.
        output_path: Path to write JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for r in results:
        serializable.append({
            "trial_number": r["trial_number"],
            "params": r["params"],
            "metrics": r["metrics"].to_dict(),
        })

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"All results saved to {output_path}")


def print_summary(
    ranked: list[dict],
    objective_metric: str,
    signal_type: str,
) -> None:
    """Print optimization summary to logger.

    Args:
        ranked: Ranked results list.
        objective_metric: Metric that was optimized.
        signal_type: "ema" or "zscore".
    """
    if not ranked:
        logger.info("No valid results found.")
        return

    best = ranked[0]
    m: PerformanceMetrics = best["metrics"]
    logger.info(f"\n{'=' * 60}")
    logger.info(f"OPTIMIZATION RESULTS ({signal_type.upper()}) — objective: {objective_metric}")
    logger.info(f"{'=' * 60}")
    logger.info(f"Best parameters: {best['params']}")
    logger.info(f"  Sharpe: {m.sharpe_ratio:.2f}  |  CAGR: {m.cagr:.2%}  |  "
                f"MaxDD: {m.max_drawdown_pct:.2%}  |  Trades: {m.total_trades}")
    logger.info(f"\nTop {len(ranked)} results:")
    logger.info("\n" + format_comparison_table(ranked, signal_type))


def print_holdout_summary(
    is_metrics: PerformanceMetrics,
    oos_metrics: PerformanceMetrics,
) -> None:
    """Print in-sample vs holdout comparison.

    Args:
        is_metrics: In-sample performance metrics.
        oos_metrics: Out-of-sample (holdout) performance metrics.
    """
    logger.info(f"\n{'=' * 60}")
    logger.info("OVERFITTING CHECK (in-sample vs holdout)")
    logger.info(f"{'=' * 60}")
    logger.info(
        f"  In-sample:  Sharpe={is_metrics.sharpe_ratio:.2f}  "
        f"CAGR={is_metrics.cagr:.2%}  MaxDD={is_metrics.max_drawdown_pct:.2%}"
    )
    logger.info(
        f"  Holdout:    Sharpe={oos_metrics.sharpe_ratio:.2f}  "
        f"CAGR={oos_metrics.cagr:.2%}  MaxDD={oos_metrics.max_drawdown_pct:.2%}"
    )

    if is_metrics.sharpe_ratio > 0:
        degradation = 1 - (oos_metrics.sharpe_ratio / is_metrics.sharpe_ratio)
        logger.info(f"  Sharpe degradation: {degradation:.0%}")
        if degradation > 0.5:
            logger.warning("  WARNING: >50% Sharpe degradation suggests overfitting!")
    else:
        logger.info("  In-sample Sharpe <= 0, holdout comparison not meaningful.")
