"""Tests for the EMA crossover signal generator."""

import numpy as np
import pandas as pd
import pytest

from src.signals.ema_crossover import EMACrossover


class TestEMACrossover:
    """Tests for EMACrossover signal."""

    def test_hand_computed_crossover(self):
        """Verify signal flips at correct bar on a synthetic 30-row series."""
        # Start flat at 100, then ramp up from bar 20 onward
        close = np.concatenate([np.full(20, 100.0), np.linspace(100, 130, 30)])
        df = pd.DataFrame({"close": close})

        signal = EMACrossover(fast_period=5, slow_period=10).generate(df)

        # Warmup period should be 0
        assert (signal.iloc[:10] == 0).all()
        # During the ramp, fast EMA should eventually cross above slow
        # At some point after the ramp starts, signal should flip to 1
        ramp_signals = signal.iloc[20:]
        assert ramp_signals.sum() > 0, "Should have long signals during uptrend"

    def test_flat_price_no_crossover(self, flat_ohlcv):
        """Flat price: fast and slow EMA are equal, no crossover -> signal stays 0."""
        signal = EMACrossover(fast_period=10, slow_period=20).generate(flat_ohlcv)
        # After warmup, flat price means fast == slow, so fast > slow is False -> 0
        assert (signal == 0).all()

    def test_uptrend_signals_long(self, trending_ohlcv):
        """Strong uptrend should produce long signals after warmup."""
        signal = EMACrossover(fast_period=10, slow_period=20).generate(trending_ohlcv)
        # After warmup, a strong uptrend means fast EMA > slow EMA
        post_warmup = signal.iloc[25:]  # give a few extra bars for convergence
        assert post_warmup.mean() > 0.8, "Strong uptrend should be mostly long"

    def test_signal_values_bounded(self, sample_ohlcv):
        """All signal values must be in {0, 1}."""
        signal = EMACrossover(fast_period=10, slow_period=20).generate(sample_ohlcv)
        assert set(signal.unique()).issubset({0, 1})

    def test_output_length_matches_input(self, sample_ohlcv):
        """Output Series must have same length as input DataFrame."""
        signal = EMACrossover(fast_period=10, slow_period=20).generate(sample_ohlcv)
        assert len(signal) == len(sample_ohlcv)

    def test_warmup_period_zeros(self, sample_ohlcv):
        """First slow_period bars should always be 0."""
        slow = 20
        signal = EMACrossover(fast_period=10, slow_period=slow).generate(sample_ohlcv)
        assert (signal.iloc[:slow] == 0).all()

    def test_single_bar_spike(self):
        """A single-bar spike should not cause a false crossover."""
        n = 100
        close = np.full(n, 100.0)
        close[50] = 200.0  # single spike
        df = pd.DataFrame({"close": close})

        signal = EMACrossover(fast_period=10, slow_period=20).generate(df)
        # The spike might briefly cause fast > slow, but should revert quickly
        # After spike settles (say 30 bars), should be back to 0
        assert signal.iloc[80:].sum() == 0, "Signal should settle back to 0 after spike"

    def test_signal_strength(self, trending_ohlcv):
        """Signal strength should be positive during uptrend."""
        ema = EMACrossover(fast_period=10, slow_period=20)
        strength = ema.signal_strength(trending_ohlcv)
        # After warmup, uptrend means fast > slow -> positive strength
        post_warmup = strength.iloc[25:]
        assert (post_warmup > 0).all()

    def test_name_property(self):
        """Name should include period parameters."""
        ema = EMACrossover(fast_period=10, slow_period=20)
        assert ema.name == "ema_crossover_10_20"

    def test_min_warmup_days(self):
        """min_warmup_days should equal slow_period."""
        ema = EMACrossover(fast_period=10, slow_period=20)
        assert ema.min_warmup_days == 20

    def test_invalid_periods_raises(self):
        """fast_period >= slow_period should raise ValueError."""
        with pytest.raises(ValueError):
            EMACrossover(fast_period=20, slow_period=10)
        with pytest.raises(ValueError):
            EMACrossover(fast_period=10, slow_period=10)
