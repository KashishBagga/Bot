---
description: How to verify system readiness and start Monday trading
---

### Sunday Night Validation (The Gate)
1. Ensure Postgres is running:
   // turbo
   `docker-compose up -d`
2. Run the Readiness Report:
   // turbo
   `python3 src/analytics/monday_readiness_report.py`
3. Verify all checks are ✅.

### Monday Morning Execution
1. Start the Option Warehouse (09:00 AM):
   `python3 src/warehouse/option_warehouse.py`
2. Start the Trading Engine (09:15 AM):
   `python3 main.py --mode paper`

### Monday Evening Reporting
1. Run the EOD Audit:
   `python3 src/analytics/trade_auditor.py`
2. Run Filter Attribution:
   `python3 src/analytics/filter_attribution.py`
