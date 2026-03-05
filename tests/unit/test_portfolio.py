"""Tests for portfolio state management and P&L tracking."""

import pandas as pd
import pytest

from src.portfolio.portfolio import Portfolio


@pytest.fixture
def portfolio():
    """Fresh portfolio with $5000 and max 5 positions."""
    return Portfolio(initial_capital=5000.0, max_positions=5)


class TestPortfolio:
    """Tests for Portfolio."""

    def test_initial_state(self, portfolio):
        """Initial portfolio should have full cash and no positions."""
        assert portfolio.cash == 5000.0
        assert portfolio.num_positions == 0
        assert len(portfolio.positions) == 0

    def test_open_and_close_position_pnl(self, portfolio):
        """Open position, mark to market, close -> verify P&L."""
        date1 = pd.Timestamp("2023-01-01", tz="UTC")
        date2 = pd.Timestamp("2023-01-10", tz="UTC")

        # Buy 10 units @ $100 = $1000, no cost
        assert portfolio.open_position("BTC/USDT", date1, 100.0, 10.0)

        # Close @ $110 -> profit = 10 * (110 - 100) = $100
        pnl = portfolio.close_position("BTC/USDT", date2, 110.0)
        assert pnl == 100.0

        # Cash should be: 5000 - 1000 + 1100 = 5100
        assert portfolio.cash == pytest.approx(5100.0)

    def test_open_and_close_with_costs(self, portfolio):
        """P&L should account for entry and exit costs."""
        date1 = pd.Timestamp("2023-01-01", tz="UTC")
        date2 = pd.Timestamp("2023-01-10", tz="UTC")

        # Buy 10 units @ $100, entry cost $5
        portfolio.open_position("BTC/USDT", date1, 100.0, 10.0, cost=5.0)
        # Cash: 5000 - 1000 - 5 = 3995

        # Close @ $110, exit cost $5
        pnl = portfolio.close_position("BTC/USDT", date2, 110.0, cost=5.0)
        # P&L = 10*(110-100) - 5 - 5 = 90
        assert pnl == pytest.approx(90.0)

        # Cash: 3995 + 1100 - 5 = 5090
        assert portfolio.cash == pytest.approx(5090.0)

    def test_multiple_concurrent_positions(self, portfolio):
        """Track equity with multiple open positions."""
        date = pd.Timestamp("2023-01-01", tz="UTC")
        portfolio.open_position("BTC/USDT", date, 100.0, 10.0)
        portfolio.open_position("ETH/USDT", date, 50.0, 20.0)

        assert portfolio.num_positions == 2

        # Mark to market
        equity = portfolio.mark_to_market(date, {
            "BTC/USDT": 110.0,
            "ETH/USDT": 55.0,
        })

        # Cash = 5000 - 1000 - 1000 = 3000
        # BTC value = 10 * 110 = 1100
        # ETH value = 20 * 55 = 1100
        assert equity == pytest.approx(5200.0)

    def test_position_limit_rejection(self, portfolio):
        """6th position should be rejected when max is 5."""
        date = pd.Timestamp("2023-01-01", tz="UTC")
        symbols = ["A/USDT", "B/USDT", "C/USDT", "D/USDT", "E/USDT"]

        for sym in symbols:
            assert portfolio.open_position(sym, date, 10.0, 1.0)

        assert portfolio.num_positions == 5
        assert not portfolio.open_position("F/USDT", date, 10.0, 1.0)
        assert portfolio.num_positions == 5

    def test_duplicate_position_rejected(self, portfolio):
        """Cannot open same symbol twice."""
        date = pd.Timestamp("2023-01-01", tz="UTC")
        assert portfolio.open_position("BTC/USDT", date, 100.0, 10.0)
        assert not portfolio.open_position("BTC/USDT", date, 100.0, 5.0)

    def test_close_nonexistent_position(self, portfolio):
        """Closing a symbol with no position returns None."""
        date = pd.Timestamp("2023-01-01", tz="UTC")
        assert portfolio.close_position("BTC/USDT", date, 100.0) is None

    def test_insufficient_cash_rejected(self):
        """Position requiring more cash than available should be rejected."""
        portfolio = Portfolio(initial_capital=100.0, max_positions=5)
        date = pd.Timestamp("2023-01-01", tz="UTC")
        # Try to buy $200 worth
        assert not portfolio.open_position("BTC/USDT", date, 100.0, 2.0)

    def test_equity_curve(self, portfolio):
        """Equity curve should track portfolio value over time."""
        dates = pd.date_range("2023-01-01", periods=3, freq="D", tz="UTC")

        portfolio.open_position("BTC/USDT", dates[0], 100.0, 10.0)

        portfolio.mark_to_market(dates[0], {"BTC/USDT": 100.0})
        portfolio.mark_to_market(dates[1], {"BTC/USDT": 110.0})
        portfolio.mark_to_market(dates[2], {"BTC/USDT": 105.0})

        curve = portfolio.get_equity_curve()
        assert len(curve) == 3
        assert curve.iloc[0]["equity"] == pytest.approx(5000.0)
        assert curve.iloc[1]["equity"] == pytest.approx(5100.0)
        assert curve.iloc[2]["equity"] == pytest.approx(5050.0)

    def test_trade_log(self, portfolio):
        """Trade log should record all closed trades."""
        date1 = pd.Timestamp("2023-01-01", tz="UTC")
        date2 = pd.Timestamp("2023-01-10", tz="UTC")

        portfolio.open_position("BTC/USDT", date1, 100.0, 10.0)
        portfolio.close_position("BTC/USDT", date2, 110.0)

        log = portfolio.get_trade_log()
        assert len(log) == 1
        assert log.iloc[0]["symbol"] == "BTC/USDT"
        assert log.iloc[0]["entry_price"] == 100.0
        assert log.iloc[0]["exit_price"] == 110.0
        assert log.iloc[0]["pnl"] == 100.0

    def test_exposure_calculation(self, portfolio):
        """Total exposure should be positions_value / equity."""
        date = pd.Timestamp("2023-01-01", tz="UTC")
        portfolio.open_position("BTC/USDT", date, 100.0, 10.0)

        exposure = portfolio.get_total_exposure({"BTC/USDT": 100.0})
        # Cash = 4000, position = 1000, equity = 5000
        # Exposure = 1000 / 5000 = 0.2
        assert exposure == pytest.approx(0.2)

    def test_to_dict_serialization(self, portfolio):
        """to_dict should produce a serializable representation."""
        date = pd.Timestamp("2023-01-01", tz="UTC")
        portfolio.open_position("BTC/USDT", date, 100.0, 10.0)

        state = portfolio.to_dict()
        assert state["cash"] == pytest.approx(4000.0)
        assert "BTC/USDT" in state["positions"]
        assert state["positions"]["BTC/USDT"]["entry_price"] == 100.0

    def test_empty_trade_log(self, portfolio):
        """Empty trade log should return DataFrame with correct columns."""
        log = portfolio.get_trade_log()
        assert len(log) == 0
        assert "symbol" in log.columns
        assert "pnl" in log.columns

    # --- Short position tests ---

    def test_open_short_position(self, portfolio):
        """Opening a short position should deduct margin (notional + cost) from cash."""
        date = pd.Timestamp("2023-01-01", tz="UTC")
        # Short 10 units @ $100 = $1000 notional as margin
        assert portfolio.open_position("BTC/USDT", date, 100.0, -10.0)
        assert portfolio.num_positions == 1
        # Cash: 5000 - 1000 = 4000
        assert portfolio.cash == pytest.approx(4000.0)
        # Position should be short
        assert portfolio.positions["BTC/USDT"].side == "short"
        assert portfolio.positions["BTC/USDT"].size == -10.0

    def test_short_profit_price_drops(self, portfolio):
        """Short position profits when price falls."""
        date1 = pd.Timestamp("2023-01-01", tz="UTC")
        date2 = pd.Timestamp("2023-01-10", tz="UTC")

        # Short 10 units @ $100
        portfolio.open_position("BTC/USDT", date1, 100.0, -10.0)

        # Close @ $90 -> profit = 10 * (100 - 90) = $100
        pnl = portfolio.close_position("BTC/USDT", date2, 90.0)
        assert pnl == pytest.approx(100.0)

        # Cash: 5000 - 1000 (margin) + 1000 (collateral) + 100 (profit) = 5100
        assert portfolio.cash == pytest.approx(5100.0)

    def test_short_loss_price_rises(self, portfolio):
        """Short position loses when price rises."""
        date1 = pd.Timestamp("2023-01-01", tz="UTC")
        date2 = pd.Timestamp("2023-01-10", tz="UTC")

        # Short 10 units @ $100
        portfolio.open_position("BTC/USDT", date1, 100.0, -10.0)

        # Close @ $110 -> loss = 10 * (100 - 110) = -$100
        pnl = portfolio.close_position("BTC/USDT", date2, 110.0)
        assert pnl == pytest.approx(-100.0)

        # Cash: 5000 - 1000 (margin) + 1000 (collateral) - 100 (loss) = 4900
        assert portfolio.cash == pytest.approx(4900.0)

    def test_short_with_costs(self, portfolio):
        """Short P&L should account for entry and exit costs."""
        date1 = pd.Timestamp("2023-01-01", tz="UTC")
        date2 = pd.Timestamp("2023-01-10", tz="UTC")

        # Short 10 units @ $100, entry cost $5
        portfolio.open_position("BTC/USDT", date1, 100.0, -10.0, cost=5.0)
        # Cash: 5000 - 1000 - 5 = 3995

        # Close @ $90, exit cost $5
        pnl = portfolio.close_position("BTC/USDT", date2, 90.0, cost=5.0)
        # P&L = (1000 - 900) - 5 - 5 = 90
        assert pnl == pytest.approx(90.0)

    def test_short_mark_to_market_profit(self, portfolio):
        """Mark-to-market for a short should increase equity when price drops."""
        date = pd.Timestamp("2023-01-01", tz="UTC")

        # Short 10 units @ $100 = $1000 margin
        portfolio.open_position("BTC/USDT", date, 100.0, -10.0)

        # Mark to market at $90: unrealized profit
        equity = portfolio.mark_to_market(date, {"BTC/USDT": 90.0})

        # Cash = 4000
        # Short value = collateral + unrealized P&L = 1000 + (1000 - 900) = 1100
        # Equity = 4000 + 1100 = 5100
        assert equity == pytest.approx(5100.0)

    def test_short_mark_to_market_loss(self, portfolio):
        """Mark-to-market for a short should decrease equity when price rises."""
        date = pd.Timestamp("2023-01-01", tz="UTC")

        # Short 10 units @ $100
        portfolio.open_position("BTC/USDT", date, 100.0, -10.0)

        # Mark to market at $110: unrealized loss
        equity = portfolio.mark_to_market(date, {"BTC/USDT": 110.0})

        # Cash = 4000
        # Short value = 1000 + (1000 - 1100) = 900
        # Equity = 4000 + 900 = 4900
        assert equity == pytest.approx(4900.0)

    def test_mixed_long_short_positions(self, portfolio):
        """Portfolio with both long and short positions should track correctly."""
        date = pd.Timestamp("2023-01-01", tz="UTC")

        portfolio.open_position("BTC/USDT", date, 100.0, 10.0)   # long
        portfolio.open_position("ETH/USDT", date, 50.0, -20.0)   # short

        assert portfolio.num_positions == 2
        assert portfolio.positions["BTC/USDT"].side == "long"
        assert portfolio.positions["ETH/USDT"].side == "short"

        # Both up: long profits, short loses
        equity = portfolio.mark_to_market(date, {
            "BTC/USDT": 110.0,  # long +100
            "ETH/USDT": 55.0,   # short -100
        })
        # Cash: 5000 - 1000 (BTC) - 1000 (ETH margin) = 3000
        # Long value: 10 * 110 = 1100
        # Short value: 1000 + (1000 - 1100) = 900
        # Equity = 3000 + 1100 + 900 = 5000
        assert equity == pytest.approx(5000.0)

    def test_short_trade_log_records_side(self, portfolio):
        """Trade log should record 'short' side for short trades."""
        date1 = pd.Timestamp("2023-01-01", tz="UTC")
        date2 = pd.Timestamp("2023-01-10", tz="UTC")

        portfolio.open_position("BTC/USDT", date1, 100.0, -10.0)
        portfolio.close_position("BTC/USDT", date2, 90.0)

        log = portfolio.get_trade_log()
        assert len(log) == 1
        assert log.iloc[0]["side"] == "short"
        assert log.iloc[0]["size"] == -10.0
        assert log.iloc[0]["pnl"] == pytest.approx(100.0)

    def test_short_exposure_uses_absolute_value(self, portfolio):
        """Exposure calculation should use absolute notional for shorts."""
        date = pd.Timestamp("2023-01-01", tz="UTC")
        portfolio.open_position("BTC/USDT", date, 100.0, -10.0)

        exposure = portfolio.get_total_exposure({"BTC/USDT": 100.0})
        # abs(-10) * 100 = 1000; cash = 4000; equity ≈ 5000
        # exposure = 1000 / 5000 = 0.2
        assert exposure == pytest.approx(0.2)

    def test_short_insufficient_margin_rejected(self):
        """Short requiring more margin than available cash should be rejected."""
        portfolio = Portfolio(initial_capital=100.0, max_positions=5)
        date = pd.Timestamp("2023-01-01", tz="UTC")
        # Short $200 worth -> needs $200 margin
        assert not portfolio.open_position("BTC/USDT", date, 100.0, -2.0)
