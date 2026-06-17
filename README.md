# 🏛️ Institutional Structural Trader — Indian Markets (Paper Mode)

A systematic, research-grade trading system for Indian index derivatives (NIFTY 50 & BANK NIFTY) built on **structural market analysis**, **institutional order-flow logic**, and a **counterfactual research engine** that continuously evaluates and improves strategy filters.

> **Current Phase**: Live Paper Trading + Counterfactual Research (v3.2)
> **Symbols Traded**: `NSE:NIFTY50-INDEX`, `NSE:NIFTYBANK-INDEX`
> **Execution Mode**: Paper (no real orders placed — all signals are simulated)

---

## 📖 Philosophy: What This System Believes

This is **not** an indicator-based system. It does not use RSI, MACD, Bollinger Bands, or moving average crossovers.

It is built on a single premise:

> **Markets are driven by institutional order flow. Institutional participants leave structural footprints: Higher Highs, Higher Lows, supply/demand zones, and volume imbalances. Trade with that structure, not against it.**

Every trade is a **thesis**:
- *"If price is at this structural zone, the institutional order here must hold."*
- If price violates that level → thesis is dead → exit immediately.

This is called **Structural Invalidation**.

---

## 🧠 Strategy Engine: Phase 3.2 (Locked)

The live strategy is in the [`EnhancedStrategyEngine`](src/core/enhanced_strategy_engine.py) (v3.2, code-frozen).

### Multi-Timeframe Analysis
The engine analyses **3 timeframes simultaneously** for every market pulse:

| Timeframe | Role | Data Window |
|---|---|---|
| **Daily (1D)** | Structural Bias — defines "direction of institutional commitment" | 40 days |
| **Hourly (1H)** | Setup Location — identifies supply/demand zones and confirmation | 10 days |
| **5-Minute (5M)** | Entry Trigger — detects the precise structural breakout or reversal | 5 days |

### Structural Bias (The Gate)

The engine only proceeds if there is **directional alignment** across timeframes:

- **Daily Bias**: `BULLISH` if 2 consecutive Higher Highs + Higher Lows. `BEARISH` if 2 consecutive Lower Highs + Lower Lows. Otherwise `NEUTRAL`.
- **1H Structure**: Must be aligned or neutral relative to Daily Bias. If actively opposed, the setup is rejected with `BIAS_MISMATCH`.

### 3 Setup Types

#### A. Sweep Reversal (SWEEP)
Price sweeps a key supply or demand zone and shows a **strong rejection candle** (institutional trap).
- BUY CALL: Price sweeps demand zone + rejection candle
- BUY PUT: Price sweeps supply zone + rejection candle
- **Stop**: 1 tick beyond the sweep wick

#### B. Structural Breakout (BREAKOUT)
A **Break of Structure (BOS)** on the 5M chart — a swing high is broken (bullish) or a swing low is broken (bearish) with price "accepting" above/below.
- **Stop**: 0.3×ATR below the broken structure level
- **Additional Filters**: Move Efficiency > 0.6, Wickiness < 0.5

#### C. Failed Follow-Through Trap (TRAP)
A BOS occurs but price immediately reverses, indicating a **false breakout trap** (retail trapped in wrong direction). The system fades the failed move.

### Accept/Reject Filters

For a signal to be accepted (and become a **real trade**), ALL of the following must pass:

| Filter | Threshold | Rejection Code |
|---|---|---|
| Time of Day RVOL | ≥ 1.0× historical average | `LOW_RVOL` |
| HTF Bias Alignment | Not actively opposed | `BIAS_MISMATCH` |
| Move Efficiency | > 0.6 | `LOW_EFFICIENCY` |
| Candle Wickiness | < 0.5 | `HIGH_WICKINESS` |
| Risk/Reward Ratio | ≥ 1.5:1 | `LOW_RR` |
| Target Zone Exists | Must find next valid zone | `NO_TARGET_ZONE` |

---

## 👻 Counterfactual Research Engine

This is the most critical research innovation in the system.

### The Problem It Solves

The system rejects the majority of setups due to the filters above. But:
> **Were those rejections correct? Did the rejected trades actually fail?**

Without tracking rejected trades, you cannot answer this. This is **research blindness**.

### The Solution: Shadow Trades

Every **rejected** candidate is entered as a **counterfactual (shadow) trade** — using the exact same trailing stop, TP expansion, and exit logic as real trades — but stored in **separate database tables** (`counterfactual_results`, `counterfactual_trade_events`).

This creates a **Filter Quality Research Dataset**:

```sql
-- Was LOW_RVOL a good filter?
SELECT primary_rejection_reason,
       COUNT(*) as setups_rejected,
       AVG(final_pnl_r) as avg_outcome,
       SUM(CASE WHEN final_pnl_r > 0 THEN 1 ELSE 0 END) as winners
FROM counterfactual_results
WHERE exit_time IS NOT NULL
GROUP BY primary_rejection_reason;
```

After weeks of live data, you can quantitatively answer: *"If I remove this filter, does expectancy improve or worsen?"*

### Architecture Guarantee

Counterfactual trades use the **exact same `_update_position()` engine** as real trades. Zero code divergence. The only differences:

| | Real Trades | Counterfactual Trades |
|---|---|---|
| Storage | `trade_performance` + `trade_events` | `counterfactual_results` + `counterfactual_trade_events` |
| Keyed by | `symbol` (1 per symbol) | `candidate_id` (multiple per symbol) |
| Max active | Unlimited | 500 (safety cap) |
| Lifecycle events | ENTRY, SL_TRAIL, TP_EXPANSION, EXIT, POSITION_RECOVERED | Same |

---

## 📊 Position Lifecycle & Trade Management

### Entry
- Signal accepted → `_enter_position()` → write to `trade_performance` table with `exit_time = NULL`
- Signal rejected → same flow but writing to `counterfactual_results`

### During Trade (every 5-minute candle)
- **Trailing Stop**: If price moves favorably, SL is trailed up/down by 1 ATR
- **TP Expansion**: When price hits current TP, TP extends to the next zone; SL is trailed up to current price
- **MFE/MAE tracking**: Max Favourable Excursion and Max Adverse Excursion tracked in real-time

### Exit Conditions
| Reason | Code |
|---|---|
| Trailed stop hit | `TRAILING_SL` |
| Original stop hit (no trail) | `INITIAL_SL` |
| Market closes at 15:25 IST | `SESSION_END` |

### Capture Rate
On exit: `capture_rate = final_pnl_r / mfe_r`
This tells you how much of the maximum available profit was captured. A capture rate of 0.7 means the position captured 70% of its maximum excursion.

---

## 🗄️ Database Architecture (TimescaleDB)

All data is stored in **TimescaleDB** (PostgreSQL extension for time-series data), running in Docker on port `5433`.

```
trading_warehouse
├── signal_audit            # Every candidate evaluated (accepted + rejected) with full context
├── signals                 # Only accepted signals
├── trade_performance       # Real trade lifecycle (open + closed)
├── trade_events            # Event-sourced trail of each real trade
├── counterfactual_results  # Shadow trade outcomes (for filter research)
├── counterfactual_trade_events  # Event trail of shadow trades
└── option_snapshots        # Raw options chain data from warehouse
```

**Schema philosophy**: Never drop tables. All schema changes use `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` migrations to protect historical data.

---

## ⏱️ Live Runtime Loop

```
Every 5 Minutes (scheduled by `schedule` library):
│
├── 1. Fetch MTF Data (D1/H1/5M) for all symbols
│
├── 2. Update Active Positions
│   ├── Real Trades → _update_position() → trail SL / expand TP / exit
│   └── Shadow Trades → _update_position() → trail SL / expand TP / exit
│
└── 3. Scan for New Setups (only for symbols with no active REAL trade)
    ├── Accepted → _enter_position(is_counterfactual=False) → Real Trade
    └── Rejected → _enter_position(is_counterfactual=True)  → Shadow Trade
```

**Startup Recovery**: On restart, the trader reads all open positions (real and counterfactual) from the database and restores full state — including trailed stop levels, extremes (highest/lowest price), and bars held. A `POSITION_RECOVERED` event is logged for full auditability.

---

## 📁 Project Structure (Key Files)

```
Bot/
├── src/
│   ├── trading/
│   │   └── indian_trader.py           # 🚀 MAIN: Live paper trading engine
│   ├── core/
│   │   ├── enhanced_strategy_engine.py # 🧠 Strategy logic (v3.2, frozen)
│   │   ├── structure_engine.py         # HH/HL fractal analysis
│   │   ├── zone_engine.py              # Supply/demand zone detection
│   │   ├── volume_engine.py            # ToD-normalised RVOL
│   │   ├── liquidity_engine.py         # Liquidity pool detection
│   │   ├── fft_engine.py               # FFT trap detection
│   │   ├── regime_engine.py            # Market regime classification
│   │   └── quant_utils.py              # Structural bias, move efficiency, wickiness
│   ├── models/
│   │   └── postgres_database.py        # 🗄️ TimescaleDB adapter (all reads/writes)
│   ├── adapters/
│   │   └── data/fyers_data_provider.py # Fyers API → pandas DataFrame bridge
│   ├── backtesting/
│   │   └── advanced_backtester.py      # 📊 Transparent backtester with per-run log files
│   ├── analytics/
│   │   ├── filter_attribution.py       # Filter quality analysis
│   │   ├── monday_readiness_report.py  # Pre-market system readiness check
│   │   └── trade_auditor.py            # EOD trade audit report
│   └── warehouse/
│       └── option_warehouse.py         # Options chain data collector
├── logs/
│   └── paper_trading_YYYY-MM-DD.log   # Daily live trading log (auto-rotates)
├── backtest_runs/
│   └── backtest_run_YYYYMMDD_HHMMSS.log # Per-run backtest logs
├── INSTITUTIONAL_LOGIC_V3.md           # Strategy white paper
├── docker-compose.yml                  # TimescaleDB + pgAdmin
└── run_indian_trader.sh                # Entry point shell script
```

---

## 🚀 How to Run

### Prerequisites

1. **Start the database**:
   ```bash
   docker-compose up -d
   ```

2. **Set environment variables** (`.env` file):
   ```bash
   FYERS_CLIENT_ID=your_client_id
   FYERS_SECRET_KEY=your_secret_key
   FYERS_ACCESS_TOKEN=your_daily_access_token   # Must be refreshed each day
   DATABASE_URL=postgresql://trader:trading_pass@127.0.0.1:5433/trading_warehouse
   ```

3. **Authenticate Fyers** (must be done each morning):
   ```bash
   python3 authenticate_fyers.py
   ```

### Start Live Paper Trading
```bash
./run_indian_trader.sh
# or
python3 src/trading/indian_trader.py
```
Logs are written to `logs/paper_trading_YYYY-MM-DD.log`.

### Run a Backtest
```bash
python3 src/backtesting/advanced_backtester.py 30   # 30-day backtest
python3 src/backtesting/advanced_backtester.py 60   # 60-day backtest
```
Each run generates a unique log in `backtest_runs/`.

### Monday Morning Readiness Check
```bash
python3 src/analytics/monday_readiness_report.py
```

---

## 📈 Current Backtest Results (as of 2026-06-16)

Run over 30-day and 60-day windows with `rvol_threshold=1.0`, `min_zone_score=50.0`:

| Metric | 30 Days | 60 Days |
|---|---|---|
| Total Trades | 6 | 9 |
| Win Rate | 50.0% | 44.4% |
| Total Return | -1.05R | -2.45R |
| Expectancy | -0.18R/trade | -0.27R/trade |

> **Note**: Negative expectancy at low sample size is expected and statistically uninformative. The counterfactual engine is actively collecting the data needed to calibrate filters and improve expectancy over the next 4-8 weeks.

---

## 🔍 What's Next (Research Roadmap)

The counterfactual engine has been live for 2 days. Key questions to answer with data:

1. **Is `LOW_RVOL` hurting expectancy?** — Two of the biggest shadow winners (+13.98R, +5.05R on June 17) were rejected primarily on `LOW_RVOL`.
2. **Is `BIAS_MISMATCH` over-filtering on momentum days?** — Strong trending days produce "BIAS_MISMATCH" on valid breakouts because the 1H counter-trend hasn't caught up yet.
3. **Does `HIGH_WICKINESS` as a standalone filter add value?** — Several Bank Nifty setups rejected solely on `HIGH_WICKINESS` went on to trend cleanly.

After 3-4 weeks of live data, run:
```bash
python3 src/analytics/filter_attribution.py
```
to get a quantitative attribution report on each filter's contribution to expectancy.

---

## 🛡️ Safety Mechanisms

- **Shadow Trade Cap**: Max 500 active counterfactual positions at any time (`MAX_ACTIVE_COUNTERFACTUALS`)
- **Session Force-Exit**: All positions (real and shadow) are closed at 15:25 IST daily
- **API Error Handling**: Fyers API failures are caught gracefully — the system continues running and recovers on the next candle
- **Schema Migration Safety**: Database schema uses `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` — no data loss on restart
- **Startup Recovery**: Full position state is restored from database on every restart

---

## 🧪 Verification Scripts

```bash
# Verify DB schema, recovery, and shadow trade lifecycle
python3 scratch/test_recovery_counterfactuals.py

# Check DB observability (table counts, open positions)
python3 scratch/check_db_observability.py

# Live connectivity test (runs one market loop with real API)
python3 scratch/test_live_start.py
```
