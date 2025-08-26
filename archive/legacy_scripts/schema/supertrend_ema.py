import sqlite3
"""
Schema module for supertrend_ema strategy.
Defines the database table structure for the strategy.
"""

# Fields for the supertrend_ema strategy table
supertrend_ema_fields = [
    "id",
    "signal_time",
    "index_name",
    "signal",
    "price",
    "ema_20",
    "atr",
    "price_to_ema_ratio",
    "confidence",
    "trade_type",
    "stop_loss",
    "target",
    "target2",
    "target3",
    "price_reason",
    "outcome",
    "pnl",
    "targets_hit",
    "stoploss_count",
    "failure_reason",
    "exit_time",
    "supertrend_value",
    "supertrend_direction",
    "supertrend_upperband",
    "supertrend_lowerband"
]

def setup_supertrend_ema_table():
    """Set up the supertrend_ema strategy table."""
    conn = sqlite3.connect("trading_signals.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS supertrend_ema (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_time TEXT,
            index_name TEXT,
            signal TEXT,
            price REAL,
            ema_20 REAL,
            atr REAL,
            price_to_ema_ratio REAL,
            confidence TEXT,
            trade_type TEXT,
            stop_loss REAL,
            target REAL,
            target2 REAL,
            target3 REAL,
            price_reason TEXT,
            outcome TEXT,
            pnl REAL,
            targets_hit INTEGER,
            stoploss_count INTEGER,
            failure_reason TEXT,
            exit_time TEXT,
            supertrend_value REAL,
            supertrend_direction INTEGER,
            supertrend_upperband REAL,
            supertrend_lowerband REAL
        )
    """)
    conn.commit()
    conn.close()

