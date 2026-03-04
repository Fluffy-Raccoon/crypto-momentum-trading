"""Performance metrics computation."""

import math
from dataclasses import dataclass

import pandas as pd


@dataclass
class PerformanceMetrics:
    """Container for all computed performance metrics."""

    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    calmar_ratio: float
    profit_factor: float
    win_rate: float
    avg_win_loss_ratio: float
    total_trades: int
    avg_holding_period_days: float
    exposure_pct: float

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return {
            "CAGR": f"{self.cagr:.2%}",
            "Sharpe Ratio": f"{self.sharpe_ratio:.2f}",
            "Sortino Ratio": f"{self.sortino_ratio:.2f}",
            "Max Drawdown": f"{self.max_drawdown_pct:.2%}",
            "Max DD Duration (days)": self.max_drawdown_duration_days,
            "Calmar Ratio": f"{self.calmar_ratio:.2f}",
            "Profit Factor": f"{self.profit_factor:.2f}",
            "Win Rate": f"{self.win_rate:.2%}",
            "Avg Win/Loss": f"{self.avg_win_loss_ratio:.2f}",
            "Total Trades": self.total_trades,
            "Avg Holding Period (days)": f"{self.avg_holding_period_days:.1f}",
            "Exposure": f"{self.exposure_pct:.2%}",
        }


def compute_metrics(
    equity_curve: pd.DataFrame,
    trade_log: pd.DataFrame,
) -> PerformanceMetrics:
    """Compute all performance metrics from equity curve and trade log.

    Args:
        equity_curve: DataFrame with 'timestamp' and 'equity' columns.
        trade_log: DataFrame with trade details including 'pnl'.

    Returns:
        PerformanceMetrics dataclass.
    """
    cagr = compute_cagr(equity_curve)
    sharpe = compute_sharpe(equity_curve)
    sortino = compute_sortino(equity_curve)
    dd_pct, dd_days = compute_max_drawdown(equity_curve)
    calmar = cagr / abs(dd_pct) if dd_pct != 0 else 0.0
    pf = compute_profit_factor(trade_log)
    wr = compute_win_rate(trade_log)
    awl = compute_avg_win_loss_ratio(trade_log)
    trades = len(trade_log)
    ahp = compute_avg_holding_period(trade_log)
    exp = compute_exposure(equity_curve)

    return PerformanceMetrics(
        cagr=cagr,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown_pct=dd_pct,
        max_drawdown_duration_days=dd_days,
        calmar_ratio=calmar,
        profit_factor=pf,
        win_rate=wr,
        avg_win_loss_ratio=awl,
        total_trades=trades,
        avg_holding_period_days=ahp,
        exposure_pct=exp,
    )


def compute_cagr(equity_curve: pd.DataFrame) -> float:
    """Compute compound annual growth rate."""
    if len(equity_curve) < 2:
        return 0.0
    start_equity = equity_curve.iloc[0]["equity"]
    end_equity = equity_curve.iloc[-1]["equity"]
    if start_equity <= 0:
        return 0.0

    start_date = pd.Timestamp(equity_curve.iloc[0]["timestamp"])
    end_date = pd.Timestamp(equity_curve.iloc[-1]["timestamp"])
    years = (end_date - start_date).days / 365.25
    if years <= 0:
        return 0.0

    return (end_equity / start_equity) ** (1 / years) - 1


def compute_sharpe(equity_curve: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
    """Compute annualized Sharpe ratio using sqrt(365) for crypto."""
    daily_returns = _daily_returns(equity_curve)
    if len(daily_returns) < 2:
        return 0.0

    excess = daily_returns - risk_free_rate / 365
    std = excess.std()
    if std == 0 or math.isnan(std):
        return 0.0

    return (excess.mean() / std) * math.sqrt(365)


def compute_sortino(equity_curve: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
    """Compute annualized Sortino ratio (downside deviation only)."""
    daily_returns = _daily_returns(equity_curve)
    if len(daily_returns) < 2:
        return 0.0

    excess = daily_returns - risk_free_rate / 365
    downside = excess[excess < 0]
    if len(downside) == 0:
        return 0.0  # No downside -> undefined, return 0

    downside_std = downside.std()
    if downside_std == 0 or math.isnan(downside_std):
        return 0.0

    return (excess.mean() / downside_std) * math.sqrt(365)


def compute_max_drawdown(equity_curve: pd.DataFrame) -> tuple[float, int]:
    """Compute maximum drawdown percentage and duration in days.

    Returns:
        Tuple of (max_drawdown_pct as negative float, duration_in_days).
    """
    if len(equity_curve) < 2:
        return 0.0, 0

    equity = equity_curve["equity"].values
    peak = equity[0]
    max_dd = 0.0
    max_dd_duration = 0
    current_dd_start = 0

    for i in range(len(equity)):
        if equity[i] > peak:
            peak = equity[i]
            current_dd_start = i
        dd = (equity[i] - peak) / peak
        if dd < max_dd:
            max_dd = dd
            max_dd_duration = i - current_dd_start

    return max_dd, max_dd_duration


def compute_profit_factor(trade_log: pd.DataFrame) -> float:
    """Compute profit factor: gross profit / gross loss."""
    if len(trade_log) == 0 or "pnl" not in trade_log.columns:
        return 0.0
    wins = trade_log[trade_log["pnl"] > 0]["pnl"].sum()
    losses = abs(trade_log[trade_log["pnl"] < 0]["pnl"].sum())
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return wins / losses


def compute_win_rate(trade_log: pd.DataFrame) -> float:
    """Compute win rate: % of trades with positive P&L."""
    if len(trade_log) == 0 or "pnl" not in trade_log.columns:
        return 0.0
    return (trade_log["pnl"] > 0).mean()


def compute_avg_win_loss_ratio(trade_log: pd.DataFrame) -> float:
    """Compute average win / average loss ratio."""
    if len(trade_log) == 0 or "pnl" not in trade_log.columns:
        return 0.0
    wins = trade_log[trade_log["pnl"] > 0]["pnl"]
    losses = trade_log[trade_log["pnl"] < 0]["pnl"]
    if len(wins) == 0 or len(losses) == 0:
        return 0.0
    avg_win = wins.mean()
    avg_loss = abs(losses.mean())
    if avg_loss == 0:
        return 0.0
    return avg_win / avg_loss


def compute_avg_holding_period(trade_log: pd.DataFrame) -> float:
    """Compute average holding period in days."""
    if len(trade_log) == 0:
        return 0.0
    if "entry_date" not in trade_log.columns or "exit_date" not in trade_log.columns:
        return 0.0
    durations = (
        pd.to_datetime(trade_log["exit_date"]) - pd.to_datetime(trade_log["entry_date"])
    ).dt.days
    return durations.mean()


def compute_exposure(equity_curve: pd.DataFrame) -> float:
    """Compute exposure: % of days with at least one position open."""
    if len(equity_curve) == 0:
        return 0.0
    if "num_positions" not in equity_curve.columns:
        return 0.0
    return (equity_curve["num_positions"] > 0).mean()


def _daily_returns(equity_curve: pd.DataFrame) -> pd.Series:
    """Compute daily returns from equity curve."""
    if len(equity_curve) < 2:
        return pd.Series(dtype=float)
    return equity_curve["equity"].pct_change().dropna()
