# Crypto Time-Series Momentum — Backtester Spec

> **Purpose**: Hand this file to Claude Code. It contains everything needed to build a
> fully tested, production-quality backtesting system for a crypto time-series momentum
> strategy using Binance historical data.

---

## 1. Project Overview

Build a Python backtesting system for a **crypto time-series momentum strategy** that trades the top-20 cryptocurrencies by market cap (excluding stablecoins). The system must be modular, fully tested, and produce clear performance reports.

### Core Signals (implement both, compare in results)

| Signal | Long Entry | Exit to Cash |
|---|---|---|
| **EMA Crossover** | 10-day EMA crosses above 20-day EMA (daily close) | 10-day EMA crosses below 20-day EMA |
| **Momentum Z-Score** | 14-day trailing return Z-score > +1.0 | Z-score drops below 0.0 |

"Cash" = position fully closed (simulating rotation into stablecoins).

### Key Constraints

- **Capital**: Configurable, default $5,000.
- **Max concurrent positions**: 5.
- **Position sizing**: Volatility-scaled — `target_risk / 30-day realized volatility`. Target risk = 1–2% of account per position (configurable).
- **Rebalance frequency**: Daily at 00:00 UTC.
- **Transaction costs**: 0.1% round-trip (Binance maker/taker).
- **Data source**: Binance public API (daily OHLCV klines).
- **Backtest period**: 2018-01-01 to present (covers bull, bear, COVID crash, recovery cycles).

---

## 2. Project Structure

```
crypto_momentum/
├── pyproject.toml              # project metadata, dependencies
├── README.md                   # setup & usage instructions
├── config.yaml                 # all tunable parameters (see §3)
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetcher.py          # Binance API data download + caching
│   │   ├── universe.py         # top-20 coin selection logic
│   │   └── cache/              # local parquet cache (gitignored)
│   ├── signals/
│   │   ├── __init__.py
│   │   ├── ema_crossover.py    # 10/20 EMA crossover signal
│   │   ├── momentum_zscore.py  # 14-day return Z-score signal
│   │   └── base.py             # abstract Signal interface
│   ├── portfolio/
│   │   ├── __init__.py
│   │   ├── position_sizer.py   # volatility-scaled sizing
│   │   ├── portfolio.py        # portfolio state, P&L tracking
│   │   └── risk.py             # per-position & portfolio-level risk checks
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py           # walk-forward backtest loop
│   │   └── costs.py            # transaction cost model
│   └── reporting/
│       ├── __init__.py
│       ├── metrics.py          # Sharpe, max DD, CAGR, profit factor, etc.
│       ├── plots.py            # equity curve, drawdown, monthly returns heatmap
│       └── tearsheet.py        # full HTML tearsheet output
├── tests/
│   ├── conftest.py             # shared fixtures (sample OHLCV data, etc.)
│   ├── unit/
│   │   ├── test_ema_crossover.py
│   │   ├── test_momentum_zscore.py
│   │   ├── test_position_sizer.py
│   │   ├── test_portfolio.py
│   │   ├── test_risk.py
│   │   ├── test_costs.py
│   │   └── test_metrics.py
│   ├── integration/
│   │   ├── test_engine_ema.py       # full backtest run with EMA signal
│   │   ├── test_engine_zscore.py    # full backtest run with Z-score signal
│   │   └── test_data_pipeline.py    # fetch → cache → signal end-to-end
│   └── property/
│       └── test_invariants.py       # property-based tests (see §7)
├── notebooks/                       # optional, for exploratory analysis
│   └── exploration.ipynb
└── scripts/
    └── run_backtest.py              # CLI entry point
```

---

## 3. Configuration (`config.yaml`)

All tunable parameters live here. **No magic numbers in code.**

```yaml
# -- Data --
data:
  exchange: "binance"
  base_currency: "USDT"
  timeframe: "1d"
  start_date: "2018-01-01"
  end_date: null                    # null = up to today
  top_n_coins: 20
  exclude:
    - "USDT"
    - "USDC"
    - "BUSD"
    - "DAI"
    - "TUSD"
    - "FDUSD"
  cache_dir: "src/data/cache"

# -- Signals --
signals:
  ema_crossover:
    fast_period: 10
    slow_period: 20
  momentum_zscore:
    lookback_days: 14
    entry_threshold: 1.0
    exit_threshold: 0.0
    zscore_window: 90          # rolling window for Z-score mean/std

# -- Portfolio --
portfolio:
  initial_capital: 5000.0
  max_positions: 5
  risk_per_position_pct: 1.5   # % of account risked per position
  vol_lookback_days: 30        # for realized vol calculation
  rebalance_time_utc: "00:00"

# -- Costs --
costs:
  commission_pct: 0.10         # round-trip, so 0.05% each way
  slippage_pct: 0.02           # conservative slippage estimate

# -- Backtest --
backtest:
  walk_forward:
    formation_window_days: 180
    roll_step_days: 30
  benchmark: "BTC"             # compare vs. buy-and-hold BTC
```

---

## 4. Module Specifications

### 4.1 `data/fetcher.py`

- Use `ccxt` library to fetch daily OHLCV from Binance.
- Cache to local Parquet files (one file per symbol). On subsequent runs, only fetch missing dates.
- Retry logic: 3 retries with exponential backoff on API errors.
- Rate limit: respect Binance's rate limits (max 1200 requests/min; add sleep between batch fetches).
- **Return type**: `pd.DataFrame` with columns `[timestamp, open, high, low, close, volume]`, indexed by `datetime`.

### 4.2 `data/universe.py`

- Determine the top-20 coins by market cap **as of each rebalance date** to avoid survivorship bias.
- Approach: use historical daily volume as a proxy for market cap ranking (or fetch snapshots from CoinGecko if available in cache). Document the chosen approach clearly.
- **Critical**: the universe must be determined using only data available *at the time* — no look-ahead bias.

### 4.3 `signals/base.py`

```python
from abc import ABC, abstractmethod
import pandas as pd

class Signal(ABC):
    """Base class for all signal generators."""

    @abstractmethod
    def generate(self, prices: pd.DataFrame) -> pd.Series:
        """
        Given a DataFrame of daily close prices for ONE asset,
        return a Series of signal values:
            +1 = long
             0 = flat / cash
        Index must match the input price index.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def min_warmup_days(self) -> int:
        """Minimum rows of data needed before signal is valid."""
        ...
```

### 4.4 `signals/ema_crossover.py`

- Compute fast EMA and slow EMA on daily close.
- Signal = +1 when fast > slow, 0 otherwise.
- `min_warmup_days` = `slow_period`.

### 4.5 `signals/momentum_zscore.py`

- Compute trailing `lookback_days` return.
- Z-score = `(trailing_return - rolling_mean) / rolling_std` over `zscore_window`.
- Entry: Z-score > `entry_threshold` → +1.
- Exit: Z-score < `exit_threshold` → 0.
- Must handle the *hysteresis* correctly: once long, stay long until exit threshold is breached. Do not flicker between states.
- `min_warmup_days` = `zscore_window`.

### 4.6 `portfolio/position_sizer.py`

- Volatility-scaled sizing:
  ```
  position_size_usd = (account_equity * risk_per_position_pct) / realized_vol_annual
  ```
  Where `realized_vol_annual = daily_std * sqrt(365)` computed over `vol_lookback_days`.
- Clamp so no single position exceeds `account_equity / max_positions`.
- Return number of units (fractional is fine for crypto).

### 4.7 `portfolio/risk.py`

- **Pre-trade checks**: reject new positions if `max_positions` already filled, or if position size < minimum trade size ($10).
- **Portfolio-level**: track total exposure as % of equity. Log a warning if total exposure > 100%.

### 4.8 `backtest/engine.py`

- **Walk-forward loop**:
  1. At each rebalance date, determine the current coin universe.
  2. Generate signals for all coins in the universe.
  3. Rank coins by signal strength (for Z-score: rank by Z-score value; for EMA: rank by distance between fast/slow EMA as % of price).
  4. Select top-N (up to `max_positions`) coins with signal = +1.
  5. Compute position sizes via volatility scaler.
  6. Execute trades: apply costs from `costs.py`.
  7. Mark-to-market daily. Track equity curve.
- **No look-ahead bias**: at each step, only data up to *yesterday's close* is available.
- Return a `BacktestResult` dataclass containing: equity curve, trade log, daily returns, positions over time.

### 4.9 `backtest/costs.py`

- Model: `cost = trade_value * (commission_pct + slippage_pct) / 100`
- Applied on both entry and exit.

### 4.10 `reporting/metrics.py`

Compute all of the following from the daily equity curve:

| Metric | Formula / Notes |
|---|---|
| CAGR | Compound annual growth rate |
| Sharpe Ratio | Annualized, using `sqrt(365)` for crypto (trades every day) |
| Sortino Ratio | Downside deviation only |
| Max Drawdown | Peak-to-trough, in % and duration (days) |
| Calmar Ratio | CAGR / Max Drawdown |
| Profit Factor | Gross profit / Gross loss (from trade log) |
| Win Rate | % of trades with positive P&L |
| Avg Win / Avg Loss | Ratio |
| Total Trades | Count |
| Avg Holding Period | Days |
| Exposure % | % of days with at least one position open |

Also compute all metrics for the **BTC buy-and-hold benchmark** for comparison.

### 4.11 `reporting/plots.py`

Generate the following as static PNGs (matplotlib) and optionally interactive HTML (plotly):

1. **Equity curve** — strategy vs. BTC buy-and-hold, log scale.
2. **Drawdown chart** — underwater plot.
3. **Monthly returns heatmap** — year × month grid.
4. **Rolling 90-day Sharpe** — to visualize regime performance.
5. **Position concentration** — stacked area of # positions over time.

### 4.12 `reporting/tearsheet.py`

- Combine all metrics + plots into a single HTML file for easy review.

---

## 5. Walk-Forward Analysis Details

This is the core methodology. Do not use a single train/test split.

```
|--- formation (180d) ---|--- trading (30d) ---|
                         |--- formation (180d) ---|--- trading (30d) ---|
                                                  |--- formation ...
```

- **Formation window** (180 days): used to compute signal parameters (EMA periods are fixed, but Z-score mean/std are estimated here).
- **Trading window** (30 days): generate signals and simulate trades using only parameters estimated during the formation window.
- **Roll**: advance by 30 days and repeat.
- **Aggregate**: stitch together all trading-window equity curves into the final out-of-sample result.

---

## 6. Survivorship Bias Handling

- The coin universe must be reconstructed at each rebalance date using *only data available at that time*.
- Coins that were delisted or collapsed (e.g., LUNA, FTT) **must appear in the universe** during their relevant periods and their full price history (including the crash to ~$0) must be included.
- The fetcher should attempt to get data for a broad set of coins (top-50 historically) even if some are no longer trading.

---

## 7. Testing Requirements

### 7.1 Unit Tests

Every module gets dedicated tests. Use `pytest`. Minimum expectations:

**`test_ema_crossover.py`**:
- Hand-computed EMA on a 30-row synthetic series → verify signal flips at the correct bar.
- Edge case: flat price (no crossover ever) → signal stays 0 or stays 1 depending on initial state.
- Edge case: single-bar spike → verify no false crossover.

**`test_momentum_zscore.py`**:
- Synthetic uptrend → verify Z-score entry triggers at the right bar.
- Synthetic downtrend → verify exit triggers.
- **Hysteresis test**: Z-score oscillates around thresholds → verify no rapid flickering.
- Edge case: constant price → Z-score = 0, no entry.

**`test_position_sizer.py`**:
- Known volatility → verify position size matches hand calculation.
- Verify clamp: no single position > `equity / max_positions`.
- Zero/near-zero volatility → verify graceful handling (don't divide by zero).

**`test_portfolio.py`**:
- Open position → mark to market → close → verify P&L correct.
- Multiple concurrent positions → verify equity tracking.
- Position limit: try to open 6th position when max is 5 → rejected.

**`test_risk.py`**:
- Verify exposure warning threshold.
- Verify minimum trade size check.

**`test_costs.py`**:
- Known trade value → verify cost calculation.
- Zero-value trade → cost = 0.

**`test_metrics.py`**:
- Synthetic equity curve (e.g., linear growth) → verify Sharpe, CAGR, max DD analytically.
- Flat equity curve → Sharpe = 0, max DD = 0.
- Single large drawdown → verify max DD calculation and duration.

### 7.2 Integration Tests

**`test_engine_ema.py`** / **`test_engine_zscore.py`**:
- Run a full backtest on a **small synthetic dataset** (5 coins, 365 days of generated OHLCV).
- Verify: equity curve length matches trading days, no NaNs, starting equity = initial capital, all trades appear in the trade log.
- Verify: equity on day 0 = initial capital (no off-by-one).
- Verify: final equity = initial capital + sum of all trade P&Ls - sum of all costs.

**`test_data_pipeline.py`**:
- Fetch → cache → re-fetch verifies cache hit (no API call on second run).
- Verify Parquet schema: correct columns, correct dtypes, no duplicated timestamps.

### 7.3 Property-Based Tests (`test_invariants.py`)

Use `hypothesis` library:

- **Equity conservation**: for any sequence of trades with zero costs, final equity = initial equity + sum of mark-to-market changes. No money created or destroyed.
- **Signal boundedness**: for any input price series, signal output ∈ {0, 1} only.
- **Position size positivity**: for any positive volatility and equity, position size > 0.
- **Cost non-negativity**: transaction costs ≥ 0 for any trade.
- **Monotonic drawdown**: the max drawdown of a subseries is ≤ the max drawdown of the full series.

### 7.4 Test Fixtures (`conftest.py`)

Provide shared fixtures:
- `sample_ohlcv`: a deterministic 500-row OHLCV DataFrame (use a seeded random walk).
- `trending_ohlcv`: a synthetic uptrend (for signal entry tests).
- `mean_reverting_ohlcv`: a synthetic sine wave (for signal exit tests).
- `flat_ohlcv`: constant price.
- `config`: a default config dict loaded from `config.yaml` with test overrides (smaller windows for speed).

---

## 8. Dependencies

```toml
[project]
name = "crypto-momentum"
requires-python = ">=3.11"

[project.dependencies]
ccxt = ">=4.0"
pandas = ">=2.0"
numpy = ">=1.24"
pyarrow = ">=14.0"       # parquet support
pyyaml = ">=6.0"
matplotlib = ">=3.8"
plotly = ">=5.18"
scipy = ">=1.11"

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.0",
    "hypothesis>=6.0",
    "ruff>=0.4",           # linting
]
```

---

## 9. CLI Entry Point (`scripts/run_backtest.py`)

```
python scripts/run_backtest.py --signal ema --config config.yaml --output results/
python scripts/run_backtest.py --signal zscore --config config.yaml --output results/
python scripts/run_backtest.py --signal both --config config.yaml --output results/
```

- `--signal both` runs both signals independently and produces a comparison table.
- `--output` directory gets: tearsheet HTML, equity curve PNG, trade log CSV, metrics JSON.

---

## 10. Definition of Done

The backtester is complete when:

- [ ] All unit, integration, and property-based tests pass (`pytest --cov` ≥ 85% line coverage).
- [ ] `ruff check` passes with zero errors.
- [ ] A full backtest (2018–present, both signals) completes without errors.
- [ ] Tearsheet HTML is generated with all metrics and plots.
- [ ] `README.md` documents setup, usage, and a summary of results.
- [ ] No look-ahead bias (verified by integration test: shuffling future data doesn't change past signals).
- [ ] No survivorship bias (verified: delisted coins appear in historical universe).

---

## 11. Phase 2 Outline (Live Paper Trading — Future)

> **Do not build this yet.** This section exists so the backtester architecture accommodates future extension.

### Scope
- Connect to Binance Testnet (paper trading) via `ccxt`.
- Run the signal generation + position sizing logic from the backtester on live daily data.
- Execute paper trades and log them to a local SQLite database.
- Daily reconciliation: compare paper portfolio state to expected state.
- Alerting: send a summary (positions, P&L, signals) via Telegram or email at each rebalance.

### Architecture Implications for Phase 1
- Keep the `Signal` interface generic — it should work on both historical DataFrames and a live "latest row" append.
- Keep `portfolio.py` stateful but serializable (can dump/load state to JSON or SQLite).
- Keep `costs.py` pluggable — live trading may use actual fill data instead of modeled costs.
- The `engine.py` backtest loop should be decomposable into a `step()` function that processes one day at a time, making it reusable for live daily execution.

---

## 12. Important Reminders

- **No look-ahead bias.** Every signal, every universe selection, every parameter must use only past data. When in doubt, lag by one day.
- **No overfitting.** The walk-forward design is the guard rail. Do not optimize parameters on the full dataset.
- **Test first, implement second.** Write the test for a module's expected behavior before implementing it. The tests are the specification.
- **Config, not code.** Every tunable number comes from `config.yaml`. If you're tempted to hardcode a number, put it in config instead.
