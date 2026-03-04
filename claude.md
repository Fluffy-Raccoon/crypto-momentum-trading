
## Task Understanding
You're building a modular Python backtesting system for crypto momentum trading from scratch. This involves multiple files across data fetching, signal generation, portfolio management, a backtest engine, reporting with HTML tearsheets, CLI entry point, and comprehensive testing with pytest/hypothesis targeting 85% coverage.

## Recommended Workflow

**Phase 1: Plan in Plan Mode**
```bash
claude --permission-mode plan
```
Start by having Claude map out the full architecture before writing any code. This is a multi-file project with interdependencies—getting the structure right first prevents expensive rework.

```
Design a modular Python backtesting system with these components:
- data/ module: Binance OHLCV fetcher with caching, top-20 coin universe
- signals/ module: EMA crossover (10/20), momentum Z-score (14-day)  
- portfolio/ module: volatility-scaled sizing (1-2% risk, max 5 concurrent)
- engine/ module: walk-forward analysis (180-day formation, 30-day trading)
- reporting/ module: HTML tearsheet with equity curves, drawdowns, metrics
- CLI entry point with argparse
- Test suite with pytest + hypothesis targeting 85% coverage

Create a detailed plan with file structure, class interfaces, and data flow.
```

Press `Ctrl+G` to review/edit the plan in your editor before proceeding.

**Phase 2: Implement incrementally**
Switch to normal mode (`Shift+Tab`) and implement module by module, testing each:

```
Implement the plan. Start with the data module and its tests. 
Run tests after each module before moving to the next.
```

**Phase 3: Integration and verification**
```
Run the full test suite with coverage reporting. Fix any failures.
Then do a dry run of the CLI with sample data.
```

## Key Best Practices

1. **Set up the project skeleton first** — Have Claude create `pyproject.toml` / `setup.py`, directory structure, and `conftest.py` before any implementation. This prevents import path issues later.

2. **Provide verification criteria upfront** — Include this in your prompt:
   ```
   After implementing each module, run its tests and verify:
   - pytest passes with no failures
   - Coverage for that module is above 85%
   - Type hints are consistent across interfaces
   ```

3. **Use subagents for parallel test writing** — Once the interfaces are defined, Claude can use the Explore subagent to investigate test patterns while implementing. For the hypothesis property-based tests specifically, consider asking:
   ```
   Write hypothesis strategies for the signal generators - test that 
   EMA crossover signals are always -1, 0, or 1, and Z-scores are 
   bounded for reasonable inputs.
   ```

4. **Pin down data contracts early** — The biggest risk in a pipeline like this is mismatched DataFrames between modules. Have Claude define the exact column schemas (e.g., `['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']`) in a shared types module or CLAUDE.md.

5. **Use `@` references for context** — When working on a downstream module:
   ```
   Look at @src/signals/ema_crossover.py and @src/signals/momentum_zscore.py 
   to understand the signal output format, then implement the portfolio 
   position sizer that consumes these signals.
   ```

6. **Run tests incrementally, not the full suite** — Per the best practices doc, prefer running single test files for speed:
   ```
   Run pytest tests/test_signals.py -v, not the whole suite
   ```

7. **Use `/compact` strategically** — After finishing each module + tests, compact with focus:
   ```
   /compact focus on the interfaces between modules and remaining work
   ```

## Configuration Checklist

**CLAUDE.md — create before starting:**
```markdown
# Crypto Backtester Project

## Build & Test
- Python 3.11+, use `uv` or `pip` for dependencies
- Run tests: `pytest tests/ -v --cov=src --cov-report=term-missing`
- Run single module tests: `pytest tests/test_<module>.py -v`
- Type check: `mypy src/`
- Lint: `ruff check src/ tests/`

## Code Style
- Type hints on all public functions
- Docstrings on all classes and public methods (Google style)
- Use pandas DataFrames with explicit column typing
- All monetary values as Decimal or float64, never float32
- Timestamps as UTC-aware datetime64[ns]

## Architecture
- DataFrame column contract: ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']
- Signal outputs: DataFrame with ['timestamp', 'symbol', 'signal'] where signal in {-1, 0, 1}
- Position sizes: DataFrame with ['timestamp', 'symbol', 'weight', 'risk_pct']
- Walk-forward windows: namedtuple(formation_start, formation_end, trading_start, trading_end)

## Testing
- Target 85% coverage minimum
- Use hypothesis for property-based tests on signals and position sizing
- Mock Binance API calls in data fetcher tests (use pytest-vcr or responses)
- Fixture files in tests/fixtures/ for sample OHLCV data

## Important
- Never make real API calls in tests - always mock
- Position sizing must enforce max 5 concurrent positions
- Risk per position must be clamped to [1%, 2%] range
- Walk-forward windows must not overlap (no lookahead bias)
```

**Permissions — allowlist common commands:**
```json
// .claude/settings.json
{
  "permissions": {
    "allow": [
      "Bash(python *)",
      "Bash(pytest *)",
      "Bash(pip install *)",
      "Bash(uv *)",
      "Bash(mypy *)",
      "Bash(ruff *)",
      "Bash(mkdir *)"
    ],
    "deny": [
      "Bash(rm -rf *)"
    ]
  }
}
```

**Install a code intelligence plugin (recommended):**
```
/plugin install pyright-lsp@claude-plugins-official
```
This gives Claude real-time type error feedback after each edit, catching interface mismatches between modules immediately.

**Status line for context tracking:**
```
/statusline show model, context percentage with progress bar, and session cost
```

## Watch Out For

1. **Lookahead bias in walk-forward** — The most critical correctness issue. Your formation and trading windows must not overlap. Add an explicit assertion in the engine: `assert formation_end < trading_start`. Have Claude write a hypothesis test that generates random window parameters and verifies no overlap.

2. **Binance API rate limits** — Even in dev, if Claude tries to fetch real data during testing, you'll hit rate limits fast. The CLAUDE.md says "never make real API calls in tests" but reinforce this when implementing the data module. Use `responses` or `pytest-vcr` to mock HTTP.

3. **Context window filling up** — This is a large multi-file project. You'll likely hit context pressure around module 3-4. Plan for `/compact` or `/clear` between major modules. Consider using `--continue` to resume if you need to split across sessions.

4. **DataFrame column mismatches** — The #1 runtime bug in pandas pipelines. The shared column contract in CLAUDE.md mitigates this, but also ask Claude to add runtime assertions at module boundaries:
   ```python
   REQUIRED_OHLCV_COLS = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']
   assert all(col in df.columns for col in REQUIRED_OHLCV_COLS), f"Missing columns: {set(REQUIRED_OHLCV_COLS) - set(df.columns)}"
   ```

5. **Hypothesis test slowness** — Property-based tests with large DataFrames can be very slow. Constrain hypothesis strategies to small sizes:
   ```python
   @settings(max_examples=50, deadline=timedelta(seconds=5))
   ```

6. **HTML tearsheet dependencies** — Libraries like `quantstats` or `jinja2` for tearsheets add complexity. Decide upfront whether to use `quantstats` (easy but opinionated) or roll your own with `jinja2` + `plotly` (flexible). Specify this in your initial prompt.

7. **pytest-cov path gotcha** — If your project uses `src/` layout, make sure `--cov=src` in the test command matches your actual package path. Misconfigured coverage paths will show 0% even with passing tests.

8. **Git worktrees for parallel work** — If you want to test the CLI while Claude is still implementing reporting, use:
   ```bash
   git worktree add ../backtester-test feature/testing
   ```

## Sources
- Best Practices for Claude Code
- Common workflows
- How Claude Code works
- Extend Claude Code (features overview)
- Extend Claude with skills
- Create custom subagents
- Discover and install prebuilt plugins through marketplaces
- Automate workflows with hooks
- Customize your status line
- How Claude remembers your project (CLAUDE.md)
- Claude Code settings
- Configure permissions
- Manage costs effectively
- Model configuration
