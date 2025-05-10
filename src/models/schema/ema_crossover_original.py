import sqlite3
"""
Schema module for ema_crossover_original strategy.
Defines the database table structure for the strategy.
"""

# Fields for the ema_crossover_original strategy table
ema_crossover_original_fields = [
    "id",
    "signal_time",
    "index_name",
    "signal",
    "price",
    "ema_9",
    "ema_21",
    "ema_20",
    "crossover_strength",
    "momentum",
    "atr",
    "confidence",
    "trade_type",
    "stop_loss",
    "target1",
    "target2",
    "target3",
    "rsi_reason",
    "macd_reason",
    "price_reason",
    "outcome",
    "pnl",
    "targets_hit",
    "stoploss_count",
    "failure_reason"
]

def setup_ema_crossover_original_table():
    """Set up the ema_crossover_original strategy table."""
    conn = sqlite3.connect("trading_signals.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ema_crossover_original (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_time TEXT,
            index_name TEXT,
            signal TEXT,
            price REAL,
            ema_9 REAL,
            ema_21 REAL,
            ema_20 REAL,
            crossover_strength REAL,
            momentum TEXT,
            atr REAL,
            confidence TEXT,
            trade_type TEXT,
            stop_loss INTEGER,
            target1 INTEGER,
            target2 INTEGER,
            target3 INTEGER,
            rsi_reason TEXT,
            macd_reason TEXT,
            price_reason TEXT,
            outcome TEXT,
            pnl REAL,
            targets_hit INTEGER,
            stoploss_count INTEGER,
            failure_reason TEXT
        )
    """)
    conn.commit()
    conn.close() 