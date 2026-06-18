# ⚡ Quick Command Reference

## 🗄️ Database

```bash
# Start TimescaleDB + pgAdmin
docker-compose up -d

# Stop database
docker-compose down

# Check if DB is running
docker ps

# Open pgAdmin (browser)
open http://localhost:5050
# Login: admin@trading.bot / admin
# DB:    host=timescaledb, port=5432, user=trader, pass=trading_pass
```

---

## 🔐 Authentication

```bash
# Refresh Fyers access token (MUST do every morning before trading)
python3 authenticate_fyers.py
```

---

## 🚀 Live Paper Trading

```bash
# Start the live trader
./run_indian_trader.sh

# Or directly
python3 src/trading/indian_trader.py

# View today's live log (tail -f for live feed)
tail -f logs/paper_trading_$(date +%Y-%m-%d).log

# View yesterday's log
cat logs/paper_trading_2026-06-17.log
```

---

## 📊 Backtesting

```bash
# 30-day backtest
python3 src/backtesting/advanced_backtester.py 30

# 60-day backtest
python3 src/backtesting/advanced_backtester.py 60

# View latest backtest run log
ls -t backtest_runs/ | head -1 | xargs -I{} cat backtest_runs/{}
```

---

## 📈 Analytics & Research

```bash
# Filter quality attribution (run after 3-4 weeks of counterfactual data)
python3 src/analytics/filter_attribution.py

# Monday morning system readiness check
python3 src/analytics/monday_readiness_report.py

# End-of-day trade audit report
python3 src/analytics/trade_auditor.py

# EOD summary report
python3 src/analytics/monday_eod_report.py

# Data quality / drift check
python3 src/analytics/data_quality_report.py
python3 src/analytics/drift_analyzer.py

# Parity check (backtest vs live consistency)
./run_parity.sh
```

---

## 🧪 Verification & Testing

```bash
# Test DB recovery + counterfactual lifecycle (end-to-end)
python3 scratch/test_recovery_counterfactuals.py

# Check DB table counts and open positions
python3 scratch/check_db_observability.py

# Live API connectivity test (one full market scan)
python3 scratch/test_live_start.py
```

---

## 🗃️ Database Queries (psql)

```bash
# Connect to DB
psql postgresql://trader:trading_pass@127.0.0.1:5433/trading_warehouse

# Or via docker
docker exec -it trading_db psql -U trader -d trading_warehouse
```

```sql
-- Open real positions
SELECT trade_id, symbol, signal_type, entry_price, stop_loss, take_profit
FROM trade_performance WHERE exit_time IS NULL;

-- Closed real trades summary
SELECT symbol, signal_type, exit_reason, final_pnl_r, capture_rate
FROM trade_performance WHERE exit_time IS NOT NULL ORDER BY exit_time DESC;

-- Counterfactual filter attribution
SELECT primary_rejection_reason,
       COUNT(*) as total,
       ROUND(AVG(final_pnl_r)::numeric, 2) as avg_pnl_r,
       SUM(CASE WHEN final_pnl_r > 0 THEN 1 ELSE 0 END) as winners
FROM counterfactual_results
WHERE exit_time IS NOT NULL
GROUP BY primary_rejection_reason
ORDER BY avg_pnl_r DESC;

-- All candidates scanned today
SELECT symbol, setup_type, accepted, rejection_reasons, entry_price, rr_ratio
FROM signal_audit
WHERE timestamp::date = CURRENT_DATE
ORDER BY timestamp DESC;

-- Recent counterfactual exits
SELECT symbol, primary_rejection_reason, entry_price, exit_price,
       final_pnl_r, capture_rate, exit_reason
FROM counterfactual_results
WHERE exit_time IS NOT NULL
ORDER BY exit_time DESC LIMIT 20;
```

---

## 🔧 Git

```bash
# Check what's changed
git status

# Commit everything
git add -A && git commit -m "your message"

# View recent commits
git log --oneline -10
```

---

## 🏥 Monday Morning Checklist

```bash
# 1. Start DB
docker-compose up -d

# 2. Refresh token
python3 authenticate_fyers.py

# 3. Readiness check
python3 src/analytics/monday_readiness_report.py

# 4. Start trader (9:15 AM IST)
./run_indian_trader.sh
```
