# Testing Reference — Data Storage & Backtesting Changes

Covers everything changed across the last few sessions: Aug-17 crash fixes,
the R-multiple bug fix, the local candle cache, and the backtester rewrite
(shared experiment factory, real pipeline, persistence, parity checks).

Run from the repo root with `PYTHONPATH=$(pwd)` (or use the `run_*.sh`
wrappers, which already set it).

---

## 0. Prerequisites

```bash
docker-compose up -d                    # TimescaleDB (5433) + pgAdmin (5050)
python3 authenticate_fyers.py           # Fyers token — required, expires daily
```

Confirm DB is actually up before anything else:

```bash
docker ps | grep timescale              # expect a running container
```

If any test below fails with a connection error first, this is why.

---

## 1. Live trader — Aug 17 crash fixes

**Run:**
```bash
./run_indian_trader.sh
tail -f logs/paper_trading_$(date +%Y-%m-%d).log
```

**Expect — none of these should appear:**
```bash
grep -c "Snapshot computation failed" logs/paper_trading_$(date +%Y-%m-%d).log   # expect 0
grep -c "LevelsView' object has no attribute" logs/paper_trading_$(date +%Y-%m-%d).log  # expect 0
grep -c "FyersClient' object has no attribute 'quotes'" logs/paper_trading_$(date +%Y-%m-%d).log  # expect 0
```

**Expect — these should appear (proof the pipeline is actually running):**
```bash
grep -c "Market Pulse" logs/paper_trading_$(date +%Y-%m-%d).log        # > 0, grows through the day
grep -c "OI_Scalping_v1.0" logs/paper_trading_$(date +%Y-%m-%d).log    # should NOT be all INSUFFICIENT_OI_DATA
```
Check a few `OI_Scalping_v1.0` lines by hand — some should show real evaluation (accepted or a *different* rejection reason), not 100% `INSUFFICIENT_OI_DATA` like before (that meant `atm_chain` was never wired up; now it is).

**Registry sanity** (confirms `experiment_factory.py` wiring didn't break the constructor):
```bash
python3 -c "
from src.core.experiment_factory import build_registry
r = build_registry()
print(r.summary())
"
```
Expect ~29 experiments listed, all `[ACTIVE ]`, no import errors.

---

## 2. EOD report — VIX + trade counts

**Run:** (auto-fires at 15:35 IST, or manually)
```bash
python3 generate_report.py
```

**Expect:**
- `reports/<date>.md` executive summary shows a real `VIX: <number>`, not `VIX: N/A`.
- If `pnl_calculation_method == 'premium'` trades exist, `Real Trades` count in
  the executive summary should now roughly match what shows up in `Trade
  Review` / `Losing Trades` sections (previously these disagreed — 4 vs 0 —
  because the old R-multiple bug made `validate_trade_data()` mark every
  premium-priced trade `valid=FALSE`).

---

## 3. R-multiple bug fix — sanity query

After at least one live session with real option-leg trades:

```bash
python3 -c "
from src.models.postgres_database import PostgresDatabase
db = PostgresDatabase()
with db._get_connection() as conn, conn.cursor() as cur:
    cur.execute('''
        SELECT trade_id, pnl_calculation_method, final_pnl_r, valid
        FROM trade_performance
        WHERE exit_time IS NOT NULL
        ORDER BY entry_time DESC LIMIT 20
    ''')
    for row in cur.fetchall():
        print(row)
"
```

**Expect:** `final_pnl_r` for `pnl_calculation_method='premium'` rows stays in
a sane range — roughly **-1.0 to +5.0R** (max loss on a long option is the
premium paid = -1R by definition now). You should NOT see values like `-51R`,
`+192R`, `+650R` again — those were the bug. `valid` should be `TRUE` for
these rows unless something else is actually wrong with them.

---

## 4. Local candle cache

**First run (cold cache, hits Fyers):**
```bash
time python3 -c "
from src.adapters.data.fyers_data_provider import FyersDataProvider
from datetime import datetime, timedelta
p = FyersDataProvider()
end = datetime.now()
df = p.get_historical_data('NSE:NIFTY50-INDEX', end - timedelta(days=10), end, '5')
print(len(df), df.index.min(), df.index.max())
"
```

**Second run, same command, immediately after:**
```bash
time python3 -c "... same as above ..."
```

**Expect:** second run is noticeably faster (only fetches the tail gap from
Fyers, rest served from Postgres) — and returns the same row count/range.

**Verify the DB actually has rows:**
```bash
python3 -c "
from src.models.postgres_database import PostgresDatabase
db = PostgresDatabase()
with db._get_connection() as conn, conn.cursor() as cur:
    cur.execute(\"SELECT symbol, timeframe, COUNT(*), MIN(time), MAX(time) FROM candles GROUP BY symbol, timeframe\")
    for row in cur.fetchall(): print(row)
"
```
Expect rows for `('NSE:NIFTY50-INDEX', '5', ...)` etc., with `MAX(time)`
close to now.

---

## 5. Backtester — full pipeline rewrite

**Run:**
```bash
python3 src/backtesting/advanced_backtester.py 10
```
(Use a small `days` value like 10 for the first test — the full pipeline
per candle per symbol is much heavier than the old single-engine version.
Expect this to take noticeably longer than before; that's expected, not a bug.)

**Expect in stdout / `backtest_runs/backtest_run_<timestamp>.log`:**
- `📥 Fetching ... MTF data` then `✅ Loaded data for` both symbols.
- A per-trade log line for every simulated trade, tagged with an experiment
  name, e.g. `[2026-08-01 09:35] NSE:NIFTY50-INDEX [Geometry_v1.0_Score35] BUY CALL`.
- A `📈 PER-EXPERIMENT BREAKDOWN` table at the end listing (roughly) all ~29
  registered experiments that fired at least once, each with its own
  trades/win%/total_r/expectancy — **this table is the main thing to sanity
  check**: strategies like `OIWallReaction_v1.0`/`PCRExtremeReversal_v1.0`/
  `OI_Scalping_v1.0` may show 0 trades (no historical option-chain data to
  backtest against — expected, see script's module docstring), but
  non-options strategies (Structural, Geometry, EMA_Pullback, VWAP_*, CPR,
  ORB, RSI2, ATR_Squeeze, etc.) should show real trade counts.
- `Structural_v3.2_RVOL1.0` and `Structural_v3.3_ExitMgmt` should appear as
  **separate rows** with different (not identical) stats — if they're
  byte-identical, the exit-management params aren't actually being applied
  differently in the backtest path (flag this if you see it).
- `ℹ️ Skipped N multi-leg combo signal(s)` — expected, not an error. These are
  the 8 combo/spread experiments (VerticalSpread, Straddle, Strangle,
  CreditSpread, IronCondor, Butterfly, IronButterfly, ExpiryAwareTheta) —
  intentionally not simulated yet (see docstring).
- `💾 Persisted run bt_<timestamp> (<N> trades) to backtest_runs/backtest_trades`
  at the very end.

**If it errors instead:** check the DB is up (step 0) — `IndicatorPipeline`
opens a real Postgres connection even for the `BacktestDBStub`-wrapped calls
(the stub only intercepts `save_market_event`/`get_option_chain_snapshot`/
`get_atm_oi_series`; the pipeline itself still needs a live DB for other reads).

---

## 6. Backtest result persistence

```bash
python3 -c "
from src.models.postgres_database import PostgresDatabase
db = PostgresDatabase()
with db._get_connection() as conn, conn.cursor() as cur:
    cur.execute('SELECT run_id, created_at, days, overall_trades, overall_total_r, overall_expectancy FROM backtest_runs ORDER BY created_at DESC LIMIT 5')
    for row in cur.fetchall(): print(row)
    cur.execute('SELECT COUNT(*) FROM backtest_trades')
    print('total backtest_trades rows:', cur.fetchone())
"
```
Expect the run(s) from step 5 to show up, with `overall_trades` matching the
`Total Trades` printed at the end of that run, and `backtest_trades` row
count matching too.

---

## 7. Parity engine / readiness check

Run this **the morning after** a live session that actually took trades
(it compares yesterday's live trades against a same-day backtest replay —
running it same-day pre-market will correctly report "no data yet", not a
failure):

```bash
python3 src/analytics/monday_readiness_report.py
```

**Expect:**
- A `| Monday Readiness Check | Status |` table.
- `Replay Determinism` should be `✅ PASS` (running the same historical
  window twice must produce identical trades — if this fails, something
  non-deterministic crept into the pipeline, worth investigating).
- `Entry Match`, `Exit Match`, `PnL Match` rows show either a real percentage
  or `⚠️ NO DATA` (if no live trades fired the prior session, or the
  backtester found nothing to replay) — never silently "100%" or a fabricated
  pass.
- If percentages ARE computed but low (<50%), that's a real, useful signal
  that live and backtest are diverging somewhere — worth digging into rather
  than dismissing.

You can also target a specific past date directly:
```bash
python3 -c "
from src.analytics.parity_engine import ParityEngine
from datetime import date
p = ParityEngine(['NSE:NIFTY50-INDEX', 'NSE:NIFTYBANK-INDEX'])
print(p.run_fill_parity_test(target_date=date(2026, 8, 14)))
"
```

---

## 8. option_snapshots retention policy

```bash
python3 -c "
from src.models.postgres_database import PostgresDatabase
db = PostgresDatabase()
with db._get_connection() as conn, conn.cursor() as cur:
    cur.execute('''
        SELECT hypertable_name, config FROM timescaledb_information.jobs
        WHERE proc_name = 'policy_retention'
    ''')
    for row in cur.fetchall(): print(row)
"
```
Expect a row for `option_snapshots` (90 days) alongside the pre-existing
`signal_audit` (180 days) one.

---

## Quick fail/pass summary table

| # | Check | Pass looks like |
|---|---|---|
| 1 | Live trader runs a full session | 0 `Snapshot computation failed`, 0 `FyersClient... quotes` errors |
| 2 | EOD report | Real `VIX:` value; real/CF trade counts agree across sections |
| 3 | R-multiple sanity | `final_pnl_r` for premium trades stays in ~[-1, +5]R |
| 4 | Candle cache | 2nd fetch of same range is faster; `candles` table has rows |
| 5 | Backtester run | Per-experiment table populated; combo-skip count logged, not silent |
| 6 | Backtest persistence | `backtest_runs`/`backtest_trades` rows match the run's printed summary |
| 7 | Parity/readiness | Determinism PASS; match %s computed or honestly `NO DATA`, never fabricated |
| 8 | Retention policy | `option_snapshots` job present, 90-day interval |

If something fails, the fix is almost always one of: DB not running (step 0),
Fyers token expired (re-run `authenticate_fyers.py`), or genuinely not enough
historical data cached yet for the symbol/window you're testing.
