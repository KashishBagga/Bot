import sqlite3
"""
Schema module for range_breakout_volatility strategy.
Defines the database table structure for the strategy.
"""

# Fields for the range_breakout_volatility strategy table
range_breakout_volatility_fields = [
    "id",
    "signal_time",
    "index_name",
    "signal",
    "price",
    "atr",
    "volatility_rank",
    "range_width",
    "breakout_size",
    "confidence",
    "price_reason",
    "trade_type",
    "rsi_reason",
    "outcome",
    "pnl",
    "targets_hit",
    "stoploss_count",
    "failure_reason",
    "exit_time"
]

def setup_range_breakout_volatility_table():
    """Set up the range_breakout_volatility strategy table."""
    conn = sqlite3.connect("trading_signals.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS range_breakout_volatility (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_time TEXT,
            index_name TEXT,
            signal TEXT,
            price REAL,
            atr REAL,
            volatility_rank REAL,
            range_width REAL,
            breakout_size REAL,
            confidence TEXT,
            price_reason TEXT,
            trade_type TEXT,
            rsi_reason TEXT,
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
