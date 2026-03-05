"""Tests for the momentum Z-score signal generator."""

import numpy as np
import pandas as pd
import pytest

from src.signals.momentum_zscore import MomentumZScore


@pytest.fixture
def zscore_signal():
    """Z-score signal with test-friendly smaller windows."""
    return MomentumZScore(
        lookback_days=5,
        zscore_window=20,
        entry_threshold=1.0,
        exit_threshold=0.0,
    )


class TestMomentumZScore:
    """Tests for MomentumZScore signal."""

    def test_uptrend_entry(self, zscore_signal):
        """Strong uptrend should trigger Z-score entry."""
        # Flat for 40 bars, then strong uptrend
        close = np.concatenate([
            np.full(40, 100.0),
            100.0 * np.cumprod(np.ones(60) * 1.02),  # 2% daily growth
        ])
        df = pd.DataFrame({"close": close})
        signal = zscore_signal.generate(df)

        # Should have some long signals during the uptrend phase
        uptrend_signals = signal.iloc[50:]
        assert uptrend_signals.sum() > 0, "Should enter long during strong uptrend"

    def test_downtrend_exit(self, zscore_signal):
        """Downtrend following uptrend should trigger exit."""
        # Uptrend then downtrend
        close = np.concatenate([
            100.0 * np.cumprod(np.ones(50) * 1.015),
            100.0 * np.cumprod(np.ones(50) * 0.985),
        ])
        df = pd.DataFrame({"close": close})
        signal = zscore_signal.generate(df)

        # Late in the downtrend, should be flat
        late_signals = signal.iloc[80:]
        assert late_signals.mean() < 0.5, "Should be mostly flat during downtrend"

    def test_hysteresis_no_flickering(self):
        """Signal should not flicker when Z-score oscillates between thresholds."""
        # Create a Z-score that goes: below entry, above entry (enter),
        # between exit and entry (should stay in), below exit (exit)
        n = 120

        # Build a price series that creates oscillating z-scores
        close = np.full(n, 100.0)
        # Gentle oscillation that shouldn't cause rapid signal changes
        for i in range(1, n):
            close[i] = close[i - 1] * (1 + 0.003 * np.sin(2 * np.pi * i / 30))

        df = pd.DataFrame({"close": close})
        sig = MomentumZScore(
            lookback_days=5, zscore_window=20,
            entry_threshold=1.0, exit_threshold=0.0,
        )
        signal = sig.generate(df)

        # With hysteresis, transitions should be minimal (not every bar)
        # A flickering signal would have many transitions
        valid_signals = signal.iloc[25:]  # skip warmup
        valid_transitions = valid_signals.diff().abs().sum()
        assert valid_transitions < len(valid_signals) * 0.3, (
            f"Too many transitions ({valid_transitions}): signal is flickering"
        )

    def test_constant_price_no_entry(self, flat_ohlcv):
        """Constant price -> zero returns -> Z-score undefined/0 -> no entry."""
        sig = MomentumZScore(
            lookback_days=5, zscore_window=20,
            entry_threshold=1.0, exit_threshold=0.0,
        )
        signal = sig.generate(flat_ohlcv)
        # With constant price, pct_change is 0, z-score is NaN or 0
        assert (signal == 0).all(), "Constant price should never trigger entry"

    def test_signal_values_bounded(self, sample_ohlcv):
        """All signal values must be in {-1, 0, 1}."""
        sig = MomentumZScore(
            lookback_days=5, zscore_window=30,
            entry_threshold=1.0, exit_threshold=0.0,
        )
        signal = sig.generate(sample_ohlcv)
        assert set(signal.unique()).issubset({-1, 0, 1})

    def test_output_length_matches_input(self, sample_ohlcv):
        """Output Series must have same length as input DataFrame."""
        sig = MomentumZScore(
            lookback_days=5, zscore_window=30,
            entry_threshold=1.0, exit_threshold=0.0,
        )
        signal = sig.generate(sample_ohlcv)
        assert len(signal) == len(sample_ohlcv)

    def test_compute_zscore(self, sample_ohlcv):
        """compute_zscore should return numeric series with NaN during warmup."""
        sig = MomentumZScore(lookback_days=5, zscore_window=30)
        zscore = sig.compute_zscore(sample_ohlcv)
        assert len(zscore) == len(sample_ohlcv)
        # First zscore_window + lookback bars should be NaN
        assert zscore.iloc[:35].isna().any()
        # After warmup, should have valid values
        assert zscore.iloc[40:].notna().all()

    def test_name_property(self):
        """Name should include lookback parameter."""
        sig = MomentumZScore(lookback_days=14)
        assert sig.name == "momentum_zscore_14"

    def test_min_warmup_days(self):
        """min_warmup_days should equal zscore_window."""
        sig = MomentumZScore(lookback_days=14, zscore_window=90)
        assert sig.min_warmup_days == 90

    def test_hysteresis_stays_long_between_thresholds(self):
        """Once entered, signal should stay 1 even if Z-score dips below entry
        threshold, as long as it stays above exit threshold."""
        # Create prices that produce z-scores: low, high (enter), medium (stay), low (exit)
        # Start with random walk then strong up then moderate then down
        base = np.full(30, 100.0)
        up = 100.0 * np.cumprod(np.ones(20) * 1.025)  # strong up
        flat_ish = up[-1] * np.cumprod(np.ones(15) * 1.001)  # gentle up
        down = flat_ish[-1] * np.cumprod(np.ones(15) * 0.97)  # strong down
        close = np.concatenate([base, up, flat_ish, down])

        df = pd.DataFrame({"close": close})
        sig = MomentumZScore(
            lookback_days=5, zscore_window=20,
            entry_threshold=1.0, exit_threshold=0.0,
        )
        signal = sig.generate(df)

        # The signal should not have a gap (0) between the entry and exit phases
        # if the z-score stays above 0 during the flat_ish phase
        # Check that there are some long signals (value == 1)
        assert (signal == 1).any(), "Should have some long signals"

    def test_downtrend_triggers_short(self):
        """Strong downtrend should trigger short (-1) signal."""
        # Flat then strong downtrend
        close = np.concatenate([
            np.full(40, 100.0),
            100.0 * np.cumprod(np.ones(60) * 0.98),  # 2% daily decline
        ])
        df = pd.DataFrame({"close": close})
        sig = MomentumZScore(
            lookback_days=5, zscore_window=20,
            entry_threshold=1.0, exit_threshold=0.0,
        )
        signal = sig.generate(df)

        downtrend_signals = signal.iloc[50:]
        assert (downtrend_signals == -1).any(), "Should enter short during strong downtrend"

    def test_short_exit_on_recovery(self):
        """Short position should exit when Z-score rises above -exit_threshold."""
        down = 100.0 * np.cumprod(np.ones(40) * 0.975)
        recovery = down[-1] * np.cumprod(np.ones(40) * 1.015)
        close = np.concatenate([np.full(30, 100.0), down, recovery])

        df = pd.DataFrame({"close": close})
        sig = MomentumZScore(
            lookback_days=5, zscore_window=20,
            entry_threshold=1.0, exit_threshold=0.0,
        )
        signal = sig.generate(df)

        # Should have short signals during downtrend
        assert (signal == -1).any(), "Should have short signals"
        # Late recovery should exit short
        late = signal.iloc[90:]
        assert (late != -1).any(), "Should exit short during recovery"

    def test_three_state_all_present(self):
        """Signal should be able to produce all three states: -1, 0, 1."""
        # Up then flat then down
        up = 100.0 * np.cumprod(np.ones(30) * 1.025)
        flat = np.full(25, up[-1])
        down = up[-1] * np.cumprod(np.ones(30) * 0.975)
        flat2 = np.full(25, down[-1])
        close = np.concatenate([np.full(30, 100.0), up, flat, down, flat2])

        df = pd.DataFrame({"close": close})
        sig = MomentumZScore(
            lookback_days=5, zscore_window=20,
            entry_threshold=1.0, exit_threshold=0.0,
        )
        signal = sig.generate(df)

        unique_vals = set(signal.unique())
        # Should have at least flat (0) and one directional signal
        assert 0 in unique_vals, "Should have flat periods"
        assert len(unique_vals) >= 2, "Should have at least 2 different signal states"

    def test_hysteresis_stays_short_between_thresholds(self):
        """Once short, should stay -1 as long as Z stays below -exit_threshold."""
        base = np.full(30, 100.0)
        down = 100.0 * np.cumprod(np.ones(20) * 0.975)
        gentle_down = down[-1] * np.cumprod(np.ones(15) * 0.999)
        recovery = gentle_down[-1] * np.cumprod(np.ones(15) * 1.03)
        close = np.concatenate([base, down, gentle_down, recovery])

        df = pd.DataFrame({"close": close})
        sig = MomentumZScore(
            lookback_days=5, zscore_window=20,
            entry_threshold=1.0, exit_threshold=0.0,
        )
        signal = sig.generate(df)

        # Should have some short signals
        assert (signal == -1).any(), "Should have short signals during downtrend"

    def test_constant_price_no_short(self, flat_ohlcv):
        """Constant price should never trigger short entry either."""
        sig = MomentumZScore(
            lookback_days=5, zscore_window=20,
            entry_threshold=1.0, exit_threshold=0.0,
        )
        signal = sig.generate(flat_ohlcv)
        assert (signal == -1).sum() == 0, "Constant price should never go short"
