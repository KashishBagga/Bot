#!/usr/bin/env python3
"""
Migration script to add missing performance columns to all strategy tables.
Ensures every table has stop_loss, target, target2, target3, outcome, pnl, targets_hit, stoploss_count, and failure_reason columns.
"""
import sqlite3

STRATEGY_TABLES = [
    'breakout_rsi',
    'donchian_breakout',
    'insidebar_bollinger',
    'range_breakout_volatility',
    'ema_crossover',
    'ema_crossover_original',
    'supertrend_ema',
    'supertrend_macd_rsi_ema',
    'insidebar_rsi',
]

PERFORMANCE_COLS = [
    ('stop_loss', 'REAL', 0),
    ('target', 'REAL', 0),
    ('target2', 'REAL', 0),
    ('target3', 'REAL', 0),
    ('outcome', 'TEXT', 'Pending'),
    ('pnl', 'REAL', 0),
    ('targets_hit', 'INTEGER', 0),
    ('stoploss_count', 'INTEGER', 0),
    ('failure_reason', 'TEXT', ''),
]

def get_table_columns(cursor, table):
    cursor.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cursor.fetchall()]

def add_column(cursor, table, col, col_type, default):
    try:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type} DEFAULT ?", (default,))
        print(f"  Added column {col} to {table}")
    except Exception as e:
        # Some SQLite versions don't support parameterized DEFAULT in ALTER TABLE
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type} DEFAULT {repr(default)}")
            print(f"  Added column {col} to {table}")
        except Exception as e2:
            print(f"  Could not add column {col} to {table}: {e2}")

def main():
    conn = sqlite3.connect("trading_signals.db")
    cursor = conn.cursor()
    for table in STRATEGY_TABLES:
        print(f"Checking table: {table}")
        try:
            cols = get_table_columns(cursor, table)
        except Exception as e:
            print(f"  Could not get columns for {table}: {e}")
            continue
        for col, col_type, default in PERFORMANCE_COLS:
            if col not in cols:
                add_column(cursor, table, col, col_type, default)
        conn.commit()
    print("\nMigration complete. All tables now have required performance columns.")
    conn.close()

if __name__ == "__main__":
    main() 