import sqlite3
"""
Schema module for donchian_breakout strategy.
Defines the database table structure for the strategy.
"""

# Fields for the donchian_breakout strategy table
donchian_breakout_fields = [
    "id",
    "signal_time",
    "index_name",
    "signal",
    "price",
    "confidence",
    "trade_type",
    "channel_width",
    "breakout_size",
    "volume_ratio",
    "stop_loss",
    "target",
    "target2",
    "target3",
    "rsi_reason",
    "price_reason",
    "outcome",
    "pnl",
    "targets_hit",
    "stoploss_count",
    "failure_reason"
    "exit_time"
]

def setup_donchian_breakout_table():
    """Set up the donchian_breakout strategy table."""
    conn = sqlite3.connect("trading_signals.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS donchian_breakout (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_time TEXT,
            index_name TEXT,
            signal TEXT,
            price REAL,
            confidence TEXT,
            trade_type TEXT,
            channel_width REAL,
            breakout_size REAL,
            volume_ratio REAL,
            stop_loss INTEGER,
            target INTEGER,
            target2 INTEGER,
            target3 INTEGER,
            rsi_reason TEXT,
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

