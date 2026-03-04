# Strategy Specification: Crypto Time-Series Momentum

## Thesis

Cryptocurrency markets exhibit persistent **time-series momentum** — assets that have been trending up tend to continue trending up over intermediate horizons (days to weeks). This effect is well-documented in traditional markets (Moskowitz, Ooi & Pedersen 2012) and is amplified in crypto due to:

1. **Retail-dominated order flow** — herding behavior and trend-chasing create self-reinforcing price dynamics
2. **24/7 trading with no circuit breakers** — trends develop without the overnight gaps and halts that dampen momentum in equities
3. **Narrative-driven cycles** — crypto themes (DeFi summer, NFT mania, AI tokens) produce prolonged sector rotations
4. **Low institutional arbitrage** — fewer systematic funds dampening mispricings compared to equities or FX

The strategy is **long-only** — it buys into uptrends and moves to cash when trends break down. It does not short, reflecting the asymmetric risk profile of crypto (unbounded upside on rallies, bounded downside to zero) and the high cost of borrowing for short positions on most exchanges.

## Signal Design

### Signal 1: EMA Crossover (Trend Following)

**Rationale:** Exponential moving average crossovers are the simplest way to identify regime changes between trending and mean-reverting environments. The EMA weights recent prices more heavily than a simple moving average, making it more responsive to trend shifts while still filtering noise.

**Parameters:**
- Fast EMA: 10 days
- Slow EMA: 20 days

**Logic:**
- **Entry (signal = 1):** Fast EMA crosses above slow EMA, indicating the short-term trend has turned bullish relative to the intermediate trend
- **Exit (signal = 0):** Fast EMA crosses below slow EMA

**Why 10/20?** These periods capture trends on the 2-4 week horizon — long enough to filter daily noise but short enough to exit before major reversals complete. Crypto's higher volatility means shorter lookbacks are more appropriate than the 50/200-day crossovers commonly used in equities.

**Ranking metric:** `(fast_ema - slow_ema) / price` — normalizes crossover magnitude across coins with different price levels. Coins with stronger momentum get priority for position allocation.

### Signal 2: Momentum Z-Score (Statistical Momentum)

**Rationale:** Raw momentum (trailing returns) is noisy and not comparable across assets with different volatility profiles. Z-scoring normalizes the momentum reading relative to its own recent history, answering: "Is this asset's current momentum unusually strong relative to what we've seen recently?"

**Parameters:**
- Lookback: 14 days (trailing return window)
- Z-score window: 90 days (normalization period)
- Entry threshold: +1.0 (one standard deviation above mean)
- Exit threshold: 0.0 (revert to mean)

**Logic:**
1. Compute 14-day trailing return: `close / close[14 days ago] - 1`
2. Z-score this return over a 90-day rolling window: `(return - mean) / std`
3. **Entry:** Z-score exceeds +1.0 (momentum is unusually strong)
4. **Exit:** Z-score falls below 0.0 (momentum has reverted to average)

**Hysteresis:** The gap between entry (+1.0) and exit (0.0) thresholds prevents signal flickering. Without hysteresis, a Z-score oscillating around a single threshold would generate rapid buy/sell signals, each incurring transaction costs. The implementation uses a forward-iteration state machine rather than vectorized comparison to enforce this correctly.

**Why 14/90?** The 14-day return captures the intermediate momentum factor. The 90-day normalization window provides enough history to establish a meaningful distribution while adapting to changing volatility regimes.

## Universe Construction

**Universe:** Top 20 cryptocurrencies by 30-day trailing USD volume on Binance, excluding stablecoins.

**Stablecoin exclusions:** USDT, USDC, BUSD, DAI, TUSD, FDUSD — these are designed to trade at $1 and have no meaningful momentum signal.

**No survivorship bias:** The universe is reconstructed at each rebalance point using only data available at that time. Coins that were in the top 20 historically but later delisted or fell out of the top 20 are included during the periods when they qualified. This prevents the common backtest bias of only trading coins that survived to the present day.

**Why top 20?** Sufficient liquidity to execute without excessive slippage. Smaller-cap coins may show stronger momentum effects but carry higher execution risk and delisting probability.

## Position Sizing

**Method:** Inverse-volatility scaling (risk parity within the momentum portfolio).

```
position_size_usd = (equity × risk_per_position) / annualized_volatility
```

Where:
- `risk_per_position` = 1.5% of equity (configurable 1-2%)
- `annualized_volatility` = daily_std × sqrt(365), computed over trailing 30 days

**Rationale:** Equal-risk allocation means each position contributes roughly the same amount of portfolio risk regardless of the underlying asset's volatility. A high-volatility asset like DOGE gets a smaller dollar position than a lower-volatility asset like BTC, so neither dominates portfolio returns.

**Constraints:**
- Maximum position: equity / max_positions (hard cap at 20% for 5 positions)
- Minimum trade: $10 USD (avoids dust trades that cost more in fees than they're worth)
- Maximum concurrent positions: 5

## Walk-Forward Analysis

**Why walk-forward?** Standard backtesting optimizes parameters on the same data used to evaluate performance, leading to overfitting. Walk-forward analysis simulates what would actually happen: calibrate on past data, trade on future data, roll forward, repeat.

**Windows:**
- **Formation:** 180 days — the in-sample window where signal parameters would be calibrated
- **Trading:** 30 days — the out-of-sample window where the strategy is evaluated
- **Roll step:** 30 days — advance one trading window at a time

**Timeline:**
```
|-------- Formation (180d) --------|--- Trading (30d) ---|
                        |-------- Formation (180d) --------|--- Trading (30d) ---|
```

**No-lookahead guarantee:** The engine enforces `formation_end < trading_start` via assertion. On each trading day, signals are generated using only data up to the previous close — the current day's data is never used for signal generation.

## Cost Model

| Component | Rate | Rationale |
|-----------|------|-----------|
| Commission | 0.10% round-trip | Binance standard maker/taker (0.05% each side) |
| Slippage | 0.02% | Conservative for top-20 coins with high liquidity |
| **Total** | **0.12% per round-trip** | Applied to entry and exit separately |

These are deliberately conservative. Actual costs may be lower with BNB fee discounts or VIP tiers, but it's better to overestimate costs in backtesting.

## Risk Management

**Pre-trade checks:**
1. Reject trade if maximum positions (5) already open
2. Reject trade if position size < $10 USD
3. Warn if total portfolio exposure exceeds 100% of equity

**Risk per position:** Clamped to [1%, 2%] regardless of configuration to prevent accidental over-leverage.

**Position-level:** No stop losses. Exit is driven entirely by signal (trend reversal). The rationale: fixed stop losses on volatile crypto assets generate excessive whipsaws. Trend-following signals provide a natural "stop" by going to zero when the trend breaks.

## Expected Characteristics

Based on crypto momentum literature and this system's parameters:

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| CAGR | 5-15% | After costs, excluding 2020-2021 bull run outlier |
| Sharpe | 0.5-1.2 | Annualized, using sqrt(365) |
| Max Drawdown | -30% to -60% | Crypto drawdowns are severe |
| Win Rate | 35-45% | Trend-following wins less often but wins bigger |
| Avg Win/Loss | 1.5-3.0x | Positive expectancy from fat right tail |
| Exposure | 40-70% | Significant time in cash during bear markets |

**Comparison benchmark:** BTC buy-and-hold. The strategy should outperform on a risk-adjusted basis (higher Sharpe) even if absolute returns trail BTC during strong bull markets, because it avoids the worst of bear market drawdowns.

## Limitations

1. **Daily resolution only** — misses intraday dynamics and assumes execution at daily close
2. **No funding rate modeling** — relevant only if extended to perpetual futures
3. **Single exchange** — Binance only; cross-exchange arbitrage effects ignored
4. **Fixed parameters** — EMA periods and Z-score thresholds are static across the entire backtest; adaptive parameter selection could improve results
5. **No regime detection** — the system trades the same way in bull, bear, and ranging markets; adding a macro regime filter could reduce drawdowns
6. **Transaction costs are approximate** — real slippage varies with order size and market conditions

## References

- Moskowitz, T., Ooi, Y.H., Pedersen, L.H. (2012). "Time Series Momentum." *Journal of Financial Economics*.
- Jegadeesh, N., Titman, S. (1993). "Returns to Buying Winners and Selling Losers." *Journal of Finance*.
- Liu, Y., Tsyvinski, A. (2021). "Risks and Returns of Cryptocurrency." *Review of Financial Studies*.
