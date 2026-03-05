"""Walk-forward backtest engine."""

import logging

import pandas as pd

from src.backtest.costs import TransactionCostModel
from src.contracts import BacktestResult, WalkForwardWindow
from src.data.universe import CoinUniverse
from src.portfolio.portfolio import Portfolio
from src.portfolio.position_sizer import VolatilityPositionSizer
from src.portfolio.risk import RiskManager
from src.signals.base import Signal
from src.signals.ema_crossover import EMACrossover
from src.signals.momentum_zscore import MomentumZScore

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Walk-forward backtest engine.

    Implements the walk-forward analysis loop:
    1. Generate rolling windows (formation + trading).
    2. Per window: determine universe, generate signals, rank, select top-N.
    3. Size positions (long and short), execute trades, mark-to-market.
    4. Stitch trading windows into final equity curve.
    """

    def __init__(self, config: dict) -> None:
        """Initialize engine from config.

        Args:
            config: Full configuration dict.
        """
        self._config = config
        self._cost_model = TransactionCostModel(config)
        self._sizer = VolatilityPositionSizer(config)
        self._risk_mgr = RiskManager(config)
        self._universe = CoinUniverse(config)

        wf = config["backtest"]["walk_forward"]
        self._formation_days = wf["formation_window_days"]
        self._roll_step_days = wf["roll_step_days"]
        self._max_positions = config["portfolio"]["max_positions"]
        self._initial_capital = config["portfolio"]["initial_capital"]

    def generate_windows(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> list[WalkForwardWindow]:
        """Generate non-overlapping walk-forward windows.

        Args:
            start_date: Earliest date in the dataset.
            end_date: Latest date in the dataset.

        Returns:
            List of WalkForwardWindow namedtuples.
        """
        windows = []
        formation_start = start_date

        while True:
            formation_end = formation_start + pd.Timedelta(days=self._formation_days - 1)
            trading_start = formation_end + pd.Timedelta(days=1)
            trading_end = trading_start + pd.Timedelta(days=self._roll_step_days - 1)

            if trading_end > end_date:
                # If we can't fit the full trading window, use remaining days
                if trading_start <= end_date:
                    trading_end = end_date
                else:
                    break

            # Critical: no lookahead bias
            assert formation_end < trading_start, (
                f"Lookahead violation: formation_end={formation_end} >= "
                f"trading_start={trading_start}"
            )

            windows.append(WalkForwardWindow(
                formation_start=formation_start,
                formation_end=formation_end,
                trading_start=trading_start,
                trading_end=trading_end,
            ))

            formation_start = formation_start + pd.Timedelta(days=self._roll_step_days)

        logger.info(f"Generated {len(windows)} walk-forward windows")
        return windows

    def run(
        self,
        signal: Signal,
        ohlcv_data: dict[str, pd.DataFrame],
    ) -> BacktestResult:
        """Run the full walk-forward backtest.

        Args:
            signal: Signal generator to use.
            ohlcv_data: Dict mapping symbol to OHLCV DataFrame.

        Returns:
            BacktestResult with equity curve, trade log, etc.
        """
        # Determine date range from data
        all_dates = set()
        for df in ohlcv_data.values():
            all_dates.update(df["timestamp"].tolist())

        if not all_dates:
            raise ValueError("No data provided")

        sorted_dates = sorted(all_dates)
        start_date = sorted_dates[0]
        end_date = sorted_dates[-1]

        windows = self.generate_windows(start_date, end_date)
        if not windows:
            raise ValueError("No valid walk-forward windows could be generated")

        portfolio = Portfolio(self._initial_capital, self._max_positions)
        all_positions_over_time: list[dict] = []

        for window in windows:
            self._run_trading_window(
                window, signal, ohlcv_data, portfolio, all_positions_over_time
            )

        # Close any remaining positions at the last available price
        last_date = windows[-1].trading_end
        self._close_all_positions(portfolio, ohlcv_data, last_date)

        # Final mark-to-market
        last_prices = self._get_prices_on_date(ohlcv_data, last_date)
        portfolio.mark_to_market(last_date, last_prices)

        equity_curve = portfolio.get_equity_curve()
        trade_log = portfolio.get_trade_log()

        # Compute daily returns
        if len(equity_curve) > 1:
            daily_returns = equity_curve["equity"].pct_change().fillna(0)
            daily_returns.index = equity_curve["timestamp"]
        else:
            daily_returns = pd.Series(dtype=float)

        positions_df = (
            pd.DataFrame(all_positions_over_time)
            if all_positions_over_time
            else pd.DataFrame(columns=["timestamp", "symbol", "weight"])
        )

        return BacktestResult(
            equity_curve=equity_curve,
            trade_log=trade_log,
            daily_returns=daily_returns,
            positions_over_time=positions_df,
            signal_name=signal.name,
            config=self._config,
        )

    def _run_trading_window(
        self,
        window: WalkForwardWindow,
        signal: Signal,
        ohlcv_data: dict[str, pd.DataFrame],
        portfolio: Portfolio,
        positions_log: list[dict],
    ) -> None:
        """Execute trades for a single trading window."""
        # Get universe as of formation_end (no lookahead)
        universe = self._universe.get_universe(ohlcv_data, window.formation_end)
        if not universe:
            return

        # Generate signals and rank for all coins in universe
        signal_scores: dict[str, tuple[int, float]] = {}

        for symbol in universe:
            if symbol not in ohlcv_data:
                continue
            df = ohlcv_data[symbol]
            # Use only data up to formation_end for signal parameter estimation
            formation_data = df[df["timestamp"] <= window.formation_end]
            if len(formation_data) < signal.min_warmup_days:
                continue

            signals = signal.generate(formation_data)
            if signals.empty:
                continue

            last_signal = signals.iloc[-1]

            # Compute signal strength for ranking
            if isinstance(signal, EMACrossover):
                strength_series = signal.signal_strength(formation_data)
                strength = strength_series.iloc[-1] if not strength_series.empty else 0.0
            elif isinstance(signal, MomentumZScore):
                zscore_series = signal.compute_zscore(formation_data)
                strength = zscore_series.iloc[-1] if not zscore_series.empty else 0.0
            else:
                strength = float(last_signal)

            if pd.isna(strength):
                strength = 0.0

            signal_scores[symbol] = (int(last_signal), float(strength))

        # Select top-N longs (signal=1, ranked by strength descending)
        long_candidates = [
            (sym, strength)
            for sym, (sig_val, strength) in signal_scores.items()
            if sig_val == 1
        ]
        long_candidates.sort(key=lambda x: x[1], reverse=True)

        # Select top-N shorts (signal=-1, ranked by strength ascending = most negative)
        short_candidates = [
            (sym, strength)
            for sym, (sig_val, strength) in signal_scores.items()
            if sig_val == -1
        ]
        short_candidates.sort(key=lambda x: x[1])  # most negative first

        # Split max_positions between longs and shorts
        half = self._max_positions // 2
        long_targets = [sym for sym, _ in long_candidates[:half or self._max_positions]]
        short_targets = [sym for sym, _ in short_candidates[:half]]

        # Build target map: symbol -> desired side (1 or -1)
        target_map: dict[str, int] = {}
        for sym in long_targets:
            target_map[sym] = 1
        for sym in short_targets:
            if sym not in target_map:  # don't conflict with long
                target_map[sym] = -1

        # Get trading days within the window
        trading_days = self._get_trading_days(ohlcv_data, window.trading_start, window.trading_end)

        for day in trading_days:
            self._step(day, target_map, signal, ohlcv_data, portfolio, positions_log, window)

    def _step(
        self,
        date: pd.Timestamp,
        target_map: dict[str, int],
        signal: Signal,
        ohlcv_data: dict[str, pd.DataFrame],
        portfolio: Portfolio,
        positions_log: list[dict],
        window: WalkForwardWindow,
    ) -> None:
        """Process a single trading day.

        Args:
            date: Current date.
            target_map: Symbol -> desired side (1=long, -1=short).
            signal: Signal generator (for re-evaluation).
            ohlcv_data: Full price data.
            portfolio: Portfolio to modify.
            positions_log: Running log of positions.
            window: Current walk-forward window.
        """
        current_prices = self._get_prices_on_date(ohlcv_data, date)
        if not current_prices:
            return

        # Re-evaluate signals using data up to yesterday (no lookahead)
        yesterday = date - pd.Timedelta(days=1)
        active_targets: dict[str, int] = {}

        for symbol, desired_side in target_map.items():
            if symbol not in ohlcv_data:
                continue
            df = ohlcv_data[symbol]
            available = df[df["timestamp"] <= yesterday]
            if len(available) < signal.min_warmup_days:
                continue
            signals = signal.generate(available)
            if not signals.empty:
                current_signal = int(signals.iloc[-1])
                # Only keep if signal still agrees with desired side
                if current_signal == desired_side:
                    active_targets[symbol] = desired_side

        # Close positions that are no longer in target or changed side
        for symbol in list(portfolio.positions.keys()):
            pos = portfolio.positions[symbol]
            pos_side = 1 if pos.size > 0 else -1
            target_side = active_targets.get(symbol)

            if target_side is None or target_side != pos_side:
                price = current_prices.get(symbol)
                if price is not None:
                    trade_value = abs(pos.size) * price
                    cost = self._cost_model.compute_cost(trade_value)
                    portfolio.close_position(symbol, date, price, cost)

        # Open new positions for targets not already held
        for symbol, side in active_targets.items():
            if symbol in portfolio.positions:
                continue
            if symbol not in current_prices:
                continue

            price = current_prices[symbol]
            if symbol not in ohlcv_data:
                continue

            df = ohlcv_data[symbol]
            available = df[df["timestamp"] <= yesterday]
            if len(available) < 2:
                continue

            # Estimate equity for sizing
            equity = portfolio.cash
            for p in portfolio.positions.values():
                cp = current_prices.get(p.symbol, p.entry_price)
                equity += abs(p.size) * cp

            size_usd = self._sizer.compute_position_size(equity, available)

            # Risk checks
            allowed, reason = self._risk_mgr.check_new_position(
                portfolio.num_positions, size_usd, portfolio.cash
            )
            if not allowed:
                logger.debug(f"Skipping {symbol}: {reason}")
                continue

            units = size_usd / price
            if side == -1:
                units = -units  # negative size for short
            cost = self._cost_model.compute_cost(size_usd)
            portfolio.open_position(symbol, date, price, units, cost)

        # Mark to market
        equity = portfolio.mark_to_market(date, current_prices)

        # Record positions
        for symbol, pos in portfolio.positions.items():
            price = current_prices.get(symbol, pos.entry_price)
            total_equity = equity if equity > 0 else 1.0
            positions_log.append({
                "timestamp": date,
                "symbol": symbol,
                "weight": (abs(pos.size) * price) / total_equity,
                "side": pos.side,
            })

    def _close_all_positions(
        self,
        portfolio: Portfolio,
        ohlcv_data: dict[str, pd.DataFrame],
        date: pd.Timestamp,
    ) -> None:
        """Close all remaining open positions."""
        prices = self._get_prices_on_date(ohlcv_data, date)
        for symbol in list(portfolio.positions.keys()):
            price = prices.get(symbol)
            if price is not None:
                trade_value = abs(portfolio.positions[symbol].size) * price
                cost = self._cost_model.compute_cost(trade_value)
                portfolio.close_position(symbol, date, price, cost)

    def _get_prices_on_date(
        self,
        ohlcv_data: dict[str, pd.DataFrame],
        date: pd.Timestamp,
    ) -> dict[str, float]:
        """Get close prices for all symbols on a given date.

        Falls back to the most recent available price before the date.
        """
        prices = {}
        for symbol, df in ohlcv_data.items():
            available = df[df["timestamp"] <= date]
            if not available.empty:
                prices[symbol] = float(available.iloc[-1]["close"])
        return prices

    def _get_trading_days(
        self,
        ohlcv_data: dict[str, pd.DataFrame],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> list[pd.Timestamp]:
        """Get sorted list of unique dates across all symbols in range."""
        dates = set()
        for df in ohlcv_data.values():
            mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
            dates.update(df.loc[mask, "timestamp"].tolist())
        return sorted(dates)
