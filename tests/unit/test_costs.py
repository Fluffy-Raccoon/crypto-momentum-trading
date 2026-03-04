"""Tests for transaction cost model."""

import pytest

from src.backtest.costs import TransactionCostModel


@pytest.fixture
def cost_config():
    """Standard cost config."""
    return {
        "costs": {
            "commission_pct": 0.10,
            "slippage_pct": 0.02,
        }
    }


class TestTransactionCostModel:
    """Tests for TransactionCostModel."""

    def test_known_trade_value(self, cost_config):
        """Verify cost calculation for a known trade value."""
        model = TransactionCostModel(cost_config)
        # cost = 1000 * (0.10 + 0.02) / 100 = 1.2
        cost = model.compute_cost(1000.0)
        assert cost == pytest.approx(1.2)

    def test_zero_trade_value(self, cost_config):
        """Zero trade value should produce zero cost."""
        model = TransactionCostModel(cost_config)
        assert model.compute_cost(0.0) == 0.0

    def test_negative_trade_value(self, cost_config):
        """Negative trade value should return 0."""
        model = TransactionCostModel(cost_config)
        assert model.compute_cost(-100.0) == 0.0

    def test_cost_non_negative(self, cost_config):
        """Cost should always be non-negative."""
        model = TransactionCostModel(cost_config)
        for val in [0.0, 1.0, 100.0, 1e6, -50.0]:
            assert model.compute_cost(val) >= 0

    def test_total_pct(self, cost_config):
        """total_pct should be sum of commission and slippage."""
        model = TransactionCostModel(cost_config)
        assert model.total_pct == pytest.approx(0.12)

    def test_large_trade(self, cost_config):
        """Large trade should scale linearly."""
        model = TransactionCostModel(cost_config)
        cost_small = model.compute_cost(100.0)
        cost_large = model.compute_cost(10000.0)
        assert cost_large == pytest.approx(cost_small * 100)

    def test_custom_config(self):
        """Custom commission/slippage rates."""
        config = {"costs": {"commission_pct": 0.50, "slippage_pct": 0.10}}
        model = TransactionCostModel(config)
        # cost = 1000 * 0.60 / 100 = 6.0
        assert model.compute_cost(1000.0) == pytest.approx(6.0)
