# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research-grade, **paper-mode** systematic trading system for Indian index derivatives (`NSE:NIFTY50-INDEX`, `NSE:NIFTYBANK-INDEX`). No real orders are ever placed — every signal is simulated and logged. The system's core purpose is not just to trade but to **research which strategies and filters actually work**, using a counterfactual engine that also simulates rejected setups.

The strategy is **structural, not indicator-based**: it trades institutional order-flow footprints (higher highs/lows, supply/demand zones, volume imbalances, break-of-structure), and exits the moment a trade's structural thesis is invalidated. See `README.md` and `INSTITUTIONAL_LOGIC_V3.md` for the trading philosophy.

## Daily operating sequence

The system depends on a Fyers access token that **expires every morning** and a TimescaleDB running in Docker.

```bash
docker-compose up -d                              # TimescaleDB (port 5433) + pgAdmin (port 5050)
python3 authenticate_fyers.py                     # Refresh Fyers token — REQUIRED each morning
python3 src/analytics/monday_readiness_report.py  # Pre-market readiness check
./run_indian_trader.sh                            # Start live paper trading (run at 9:15 IST)
```

## Common commands

```bash
# Live paper trading (main entry point). run_indian_trader.sh just sets PYTHONPATH=. and runs it.
./run_indian_trader.sh
tail -f logs/paper_trading_$(date +%Y-%m-%d).log  # Follow today's log (auto-rotates by date)

# Backtesting — arg is the lookback window in days; each run writes a unique log to backtest_runs/
python3 src/backtesting/advanced_backtester.py 30

# EOD report (also auto-triggered by the trader at 15:35 IST). Writes reports/<date>.md + .json
python3 generate_report.py

# Analytics / research
python3 src/analytics/filter_attribution.py       # Filter quality attribution (needs weeks of data)
python3 src/analytics/trade_auditor.py            # EOD trade audit
./run_parity.sh                                   # Backtest-vs-live consistency check

# Dashboards
./run_dashboard_server.sh                          # Streamlit EOD dashboard on port 8080
```

`COMMANDS.md` is a fuller quick reference including psql queries for the key tables.

## Running Python directly

Every entry point injects the repo root onto `sys.path`, but modules use absolute `src.` imports. Run everything **from the repo root** with `PYTHONPATH` set to it — `run_*.sh` scripts do `export PYTHONPATH=$(pwd)`. Python 3 (see `.python-version`). Install deps with `pip install -r requirements.txt`.

## Testing

There is **no pytest suite, config, or CI**. "Tests" are ad-hoc verification scripts in `scratch/` (e.g. `scratch/test_recovery_counterfactuals.py`, `scratch/check_db_observability.py`, `scratch/test_live_start.py`) run manually against a live DB / API. Note: `scratch/`, `scratch_*.py`, and `trade_journal.csv` are **gitignored** — don't rely on them being present or commit new work there expecting it to persist in git.

## Architecture

### Strategy Research Framework (v4.0) — the outer loop

The live trader (`src/trading/indian_trader.py`) runs **many strategies as parallel experiments** against one shared market view. The flow per 5-minute candle:

1. **One `MarketSnapshot` per symbol per candle** (`src/core/market_snapshot.py`) — MTF data (Daily/1H/5M) computed once and shared, so experiments never recompute or diverge on inputs.
2. **`IndicatorPipeline`** (`src/core/indicator_pipeline.py`) enriches the snapshot with indicators/features.
3. **`ExperimentRegistry`** (`src/core/experiment_registry.py`) runs every registered `Experiment` against the snapshot. **One experiment crashing never affects the others.**
4. Each `Experiment` (`src/core/experiment.py`) wraps a strategy (a `BaseStrategy` subclass) + a params dict. Positions are keyed by **`(symbol, experiment_name)`**, so experiments hold independent positions simultaneously.
5. `PortfolioManager` (`src/core/portfolio.py`) tracks per-experiment analytics as a passive observer.

**To add a strategy:** subclass `BaseStrategy` (`src/core/base_strategy.py`), returning `StrategyResult` from `evaluate()`, then register one `Experiment(...)` in `indian_trader.py` (see the block of `Experiment(...)` registrations there). If adding a strategy requires editing anything else, that's hidden coupling to avoid — the framework is designed so this is the only change needed.

### EnhancedStrategyEngine (v3.2) — the frozen core strategy

`src/core/enhanced_strategy_engine.py` is the flagship structural strategy, **code-frozen**. It is wrapped, unchanged, inside `src/strategies/structural_strategy.py`. It performs MTF bias gating (Daily→1H→5M), detects 3 setup types (SWEEP / BREAKOUT / TRAP), and applies accept/reject filters (RVOL, bias alignment, move efficiency, wickiness, R:R, target-zone existence). Treat this file as locked unless explicitly asked to change strategy logic; new ideas belong in new strategies/experiments, not edits here.

The `src/core/*_engine.py` modules are the structural primitives it composes: `structure_engine` (HH/HL fractals), `zone_engine` (supply/demand), `volume_engine` (time-of-day-normalized RVOL), `liquidity_engine`, `fft_engine` (trap detection), `regime_engine`. `src/core/quant_utils.py` holds the pure structural-math helpers.

### Counterfactual / shadow trades — the research engine

Rejected candidates are **not discarded**. They are entered as counterfactual "shadow" trades running through the **exact same position-update engine** (`_update_position()`) as real trades — zero code divergence — but written to separate DB tables. This builds a dataset to answer "was this filter actually correct?" (via `filter_attribution.py`). Preserving this "same engine, different storage" guarantee is critical: don't fork the update logic for shadow trades.

### Persistence — TimescaleDB, append-only

`src/models/postgres_database.py` is the single adapter for all reads/writes. Connection defaults to `postgresql://trader:trading_pass@127.0.0.1:5433/trading_warehouse` (note port **5433**). Key tables: `signal_audit` (every candidate), `signals`, `trade_performance` + `trade_events` (real), `counterfactual_results` + `counterfactual_trade_events` (shadow), `option_snapshots`.

**Schema is append-only. Never drop tables or columns.** All migrations use `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` to protect historical research data. On restart the trader restores all open positions (real + shadow), including trailed stops and extremes, logging a `POSITION_RECOVERED` event.

### Reporting

`src/reports/eod_report_generator.py` orchestrates ~12 pluggable sections (`src/reports/sections/`) rendered to Markdown + JSON (`src/reports/renderers/`). Auto-triggered by the trader at 15:35 IST; also runnable via `generate_report.py`.

### Market Knowledge Engine (MKE)

`src/core/market_knowledge.py` (+ `market_facts.py`) is a newer layered market-context model of anchors/swings. Its entities are **frozen dataclasses** (immutable for concurrency safety) — construct new instances rather than mutating.

### Data access & multi-market scaffolding

`src/adapters/data/fyers_data_provider.py` bridges the Fyers API to pandas DataFrames. There is broader multi-market/adapter scaffolding (`src/adapters/`, `src/markets/indian/`, `src/markets/crypto/`), but the Indian index-derivatives path via `indian_trader.py` is the active, maintained one.

## Conventions & gotchas

- **`.json` files are gitignored** (`tokens/`, credentials, session stats) with a narrow allowlist — Fyers tokens live in `tokens/` and must not be committed.
- Secrets come from `.env` (`FYERS_CLIENT_ID`, `FYERS_SECRET_KEY`, `FYERS_ACCESS_TOKEN`, `DATABASE_URL`). Config is centralized in `src/config/settings.py`.
- All times are **IST (`Asia/Kolkata`)**. Trading hours 09:15–15:30; all positions force-exit at 15:25.
- The `src/core/` directory contains many overlapping/legacy modules (multiple `*_strategy_engine.py`, `*_performance_optimizer.py`, `*_database.py`). When touching core logic, confirm which module the active `indian_trader.py` import graph actually uses before editing — several are dead or superseded.
