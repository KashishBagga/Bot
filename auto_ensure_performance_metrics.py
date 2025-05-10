#!/usr/bin/env python3
"""
Auto ensure all strategy tables have required performance columns and all rows have performance metrics.
Runs migration and backfill in sequence. Idempotent and safe to call at any time.
"""
import subprocess
import sys

# Step 1: Run migration script
print("\n=== Running migration to add missing columns ===")
migrate_result = subprocess.run([sys.executable, 'migrate_add_performance_columns.py'])
if migrate_result.returncode != 0:
    print("Migration failed! Exiting.")
    sys.exit(1)

# Step 2: Run backfill script
print("\n=== Running backfill to populate missing performance metrics ===")
backfill_result = subprocess.run([sys.executable, 'backfill_performance_metrics.py'])
if backfill_result.returncode != 0:
    print("Backfill failed! Exiting.")
    sys.exit(1)

print("\nAll tables and rows are now up-to-date with required performance metrics!") 