import sqlite3
"""
Schema module for breakout_rsi strategy.
Defines the database table structure for the strategy.
"""

# Fields for the breakout_rsi strategy table
breakout_rsi_fields = [
    "id",
    "signal_time",
    "index_name",
    "signal",
    "price",
    "rsi",
    "confidence",
    "trade_type",
    "breakout_strength",
    "rsi_alignment",
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
]

def setup_breakout_rsi_table():
    """Set up the breakout_rsi strategy table."""
    conn = sqlite3.connect("trading_signals.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS breakout_rsi (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_time TEXT,
            index_name TEXT,
            signal TEXT,
            price REAL,
            rsi REAL,
            confidence TEXT,
            trade_type TEXT,
            breakout_strength REAL,
            rsi_alignment TEXT,
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
            failure_reason TEXT
        )
    """)
    conn.commit()
    conn.close()

