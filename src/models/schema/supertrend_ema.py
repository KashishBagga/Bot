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
    "rsi",
    "macd",
    "macd_signal",
    "ema_20",
    "atr",
    "confidence",
    "trade_type",
    "stop_loss",
    "target",
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
    "exit_time"
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
            rsi REAL,
            macd REAL,
            macd_signal REAL,
            ema_20 REAL,
            atr REAL,
            confidence TEXT,
            trade_type TEXT,
            stop_loss INTEGER,
            target INTEGER,
            target2 INTEGER,
            target3 INTEGER,
            rsi_reason TEXT,
            macd_reason TEXT,
            price_reason TEXT,
            outcome TEXT,
            pnl REAL,
            targets_hit INTEGER,
            stoploss_count INTEGER,
            failure_reason TEXT,
            exit_time TEXT
        )
    """)
    conn.commit()
    conn.close()

