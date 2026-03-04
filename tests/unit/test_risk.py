"""Tests for risk management checks."""

import pytest

from src.portfolio.risk import MIN_TRADE_SIZE, RiskManager


@pytest.fixture
def risk_config():
    """Standard risk config."""
    return {
        "portfolio": {
            "max_positions": 5,
            "risk_per_position_pct": 1.5,
        }
    }


class TestRiskManager:
    """Tests for RiskManager."""

    def test_allow_valid_trade(self, risk_config):
        """Valid trade should be allowed."""
        rm = RiskManager(risk_config)
        allowed, reason = rm.check_new_position(
            current_positions=2, trade_size_usd=500.0, equity=5000.0
        )
        assert allowed is True
        assert reason == "OK"

    def test_reject_max_positions(self, risk_config):
        """Should reject when max positions already reached."""
        rm = RiskManager(risk_config)
        allowed, reason = rm.check_new_position(
            current_positions=5, trade_size_usd=500.0, equity=5000.0
        )
        assert allowed is False
        assert "Max positions" in reason

    def test_reject_small_trade(self, risk_config):
        """Should reject trade below minimum size."""
        rm = RiskManager(risk_config)
        allowed, reason = rm.check_new_position(
            current_positions=0, trade_size_usd=5.0, equity=5000.0
        )
        assert allowed is False
        assert "minimum" in reason.lower()

    def test_reject_exceeds_equity(self, risk_config):
        """Should reject trade exceeding equity."""
        rm = RiskManager(risk_config)
        allowed, reason = rm.check_new_position(
            current_positions=0, trade_size_usd=6000.0, equity=5000.0
        )
        assert allowed is False
        assert "exceeds equity" in reason.lower()

    def test_clamp_risk_pct_low(self, risk_config):
        """Risk below 1% should be clamped to 1%."""
        rm = RiskManager(risk_config)
        assert rm.clamp_risk_pct(0.5) == 1.0

    def test_clamp_risk_pct_high(self, risk_config):
        """Risk above 2% should be clamped to 2%."""
        rm = RiskManager(risk_config)
        assert rm.clamp_risk_pct(3.0) == 2.0

    def test_clamp_risk_pct_in_range(self, risk_config):
        """Risk within [1%, 2%] should be unchanged."""
        rm = RiskManager(risk_config)
        assert rm.clamp_risk_pct(1.5) == 1.5

    def test_exposure_acceptable(self, risk_config):
        """Exposure below 100% should be acceptable."""
        rm = RiskManager(risk_config)
        assert rm.check_exposure(0.8) is True

    def test_exposure_warning(self, risk_config):
        """Exposure above 100% should trigger warning."""
        rm = RiskManager(risk_config)
        assert rm.check_exposure(1.1) is False

    def test_exact_max_positions(self, risk_config):
        """Exactly at max positions should reject new trade."""
        rm = RiskManager(risk_config)
        allowed, _ = rm.check_new_position(
            current_positions=5, trade_size_usd=100.0, equity=5000.0
        )
        assert allowed is False

    def test_min_trade_size_boundary(self, risk_config):
        """Trade at exactly MIN_TRADE_SIZE should be allowed."""
        rm = RiskManager(risk_config)
        allowed, _ = rm.check_new_position(
            current_positions=0, trade_size_usd=MIN_TRADE_SIZE, equity=5000.0
        )
        assert allowed is True
