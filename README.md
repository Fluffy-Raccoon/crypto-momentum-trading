# Crypto Momentum Backtester

A modular Python backtesting system for crypto momentum trading strategies. Trades top-20 cryptocurrencies using two signal generators (EMA crossover and momentum Z-score), with walk-forward analysis, volatility-scaled position sizing, and HTML tearsheet reporting.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run backtest with both signals
python scripts/run_backtest.py --signal both --config config.yaml --output results/

# Run a single signal
python scripts/run_backtest.py --signal ema
python scripts/run_backtest.py --signal zscore
```

## Architecture

```
src/
├── contracts.py          # Shared data contracts and validation
├── data/
│   ├── fetcher.py        # Binance OHLCV fetcher with parquet caching
│   └── universe.py       # Dynamic top-N coin selection (no survivorship bias)
├── signals/
│   ├── base.py           # Abstract Signal interface
│   ├── ema_crossover.py  # EMA 10/20 crossover signal
│   └── momentum_zscore.py# Trailing return Z-score with hysteresis
├── portfolio/
│   ├── position_sizer.py # Volatility-scaled position sizing
│   ├── portfolio.py      # Stateful position tracking and mark-to-market
│   └── risk.py           # Pre-trade risk checks
├── backtest/
│   ├── engine.py         # Walk-forward backtest engine
│   └── costs.py          # Transaction cost model
└── reporting/
    ├── metrics.py        # CAGR, Sharpe, Sortino, drawdown, etc.
    ├── plots.py          # Equity curves, drawdowns, heatmaps
    └── tearsheet.py      # HTML tearsheet generation
```

## How It Works

**Walk-forward analysis** avoids overfitting by splitting history into formation (180-day) and trading (30-day) windows that roll forward through time. Signals are calibrated on the formation window and tested out-of-sample on the trading window.

Each trading day, the engine:
1. Reconstructs the coin universe using only historically available data
2. Generates signals and ranks coins by signal strength
3. Selects the top candidates (up to 5 concurrent positions)
4. Sizes positions using inverse-volatility scaling
5. Executes trades with realistic cost modeling (0.10% commission + 0.02% slippage)
6. Marks all positions to market

See [STRATEGY_SPEC.md](STRATEGY_SPEC.md) for the full strategy rationale.

## Signals

| Signal | Entry | Exit | Ranking |
|--------|-------|------|---------|
| **EMA Crossover** | Fast EMA (10) > Slow EMA (20) | Fast < Slow | (fast - slow) / price |
| **Momentum Z-Score** | Z-score > +1.0 | Z-score < 0.0 | Raw Z-score value |

Both signals output binary {0, 1} values (long-only system). Hysteresis on the Z-score signal prevents flickering when values oscillate near thresholds.

## Configuration

All parameters live in `config.yaml` — no magic numbers in code:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Initial capital | $5,000 | Starting equity |
| Max positions | 5 | Concurrent position limit |
| Risk per position | 1.5% | Equity risked per trade |
| Formation window | 180 days | Signal calibration period |
| Trading window | 30 days | Out-of-sample period |
| Universe size | Top 20 | Coins ranked by 30-day volume |
| Commission | 0.10% | Round-trip (Binance maker/taker) |
| Slippage | 0.02% | Conservative estimate |

## Outputs

Running a backtest produces:

- **HTML tearsheet** — equity curve, drawdown chart, monthly returns heatmap, rolling Sharpe, position concentration, and a full metrics table
- **Metrics JSON** — machine-readable performance summary
- **Trade log CSV** — every entry/exit with P&L
- **Comparison table** — side-by-side when running `--signal both`

## Testing

```bash
# Full suite (136 tests, 94% coverage)
pytest tests/ -v --cov=src --cov-report=term-missing

# By category
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/property/ -v

# Single module
pytest tests/unit/test_ema_crossover.py -v
```

Tests include:
- **Unit tests** — signals, position sizing, risk checks, metrics, plots, tearsheets
- **Integration tests** — full backtests on synthetic multi-coin datasets
- **Property-based tests** (hypothesis) — signal boundedness, cost non-negativity, drawdown monotonicity, position size positivity

## Requirements

- Python 3.11+
- Dependencies: ccxt, pandas, numpy, matplotlib, scipy, pyarrow, pyyaml

## Data

Historical OHLCV data is fetched from Binance via ccxt and cached locally as Parquet files in `src/data/cache/`. Subsequent runs only fetch missing dates (incremental updates). Stablecoins (USDT, USDC, BUSD, DAI, TUSD, FDUSD) are excluded from the trading universe.
